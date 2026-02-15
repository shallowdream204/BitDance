# Modified from:
#   DiT:  https://github.com/facebookresearch/DiT/blob/main/sample.py
import math
import os
import time

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm

from src.model import create_model, get_model_args

from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor


def save_image(sample, folder_dir, index):
    """Worker function to save a single image."""
    try:
        Image.fromarray(sample).save(f"{folder_dir}/{index:06d}.png")
    except Exception as e:
        print(f"Error saving image {index}: {e}")

def _read_one(path):
    with Image.open(path) as im:
        return np.asarray(im.convert('RGB'), dtype=np.uint8)

def create_npz_from_sample_folder(sample_dir, num=50_000, max_workers=16, save_compressed=False, delete_folder=True):
    first_path = os.path.join(sample_dir, f"{0:06d}.png")
    first = _read_one(first_path)
    H, W, C = first.shape
    assert C == 3, f"Expect 3 channels, got {C}"

    samples = np.empty((num, H, W, 3), dtype=np.uint8)
    samples[0] = first

    paths = [os.path.join(sample_dir, f"{i:06d}.png") for i in range(1, num)]

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_idx = {ex.submit(_read_one, p): i+1 for i, p in enumerate(paths)}
        for fut in tqdm(as_completed(future_to_idx), total=len(paths), desc="Building .npz from samples (threads)"):
            i = future_to_idx[fut]
            arr = fut.result()
            if arr.shape != (H, W, 3):
                raise ValueError(f"Image shape mismatch at index {i}: got {arr.shape}, expected {(H, W, 3)}")
            samples[i] = arr

    npz_path = f"{sample_dir}.npz"
    if save_compressed:
        np.savez_compressed(npz_path, arr_0=samples)
    else:
        np.savez(npz_path, arr_0=samples)

    print(f"Saved .npz file to {npz_path} [shape={samples.shape}, compressed={save_compressed}].")

    if delete_folder:
        os.system(f"rm -r {sample_dir}")

    return npz_path


def main(args):
    assert (
        torch.cuda.is_available()
    ), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)

    # create and load gpt model
    precision = {"none": torch.float32, "bf16": torch.bfloat16}[args.mixed_precision]
    model = create_model(args, device)

    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if "ema" in checkpoint and not args.no_ema:
        print("use ema weight")
        model_weight = checkpoint["ema"]
    elif "model" in checkpoint:  # ddp
        model_weight = checkpoint["model"]
    else:
        raise Exception("please check model weight")

    model.load_state_dict(model_weight, strict=True)
    if hasattr(model, "load_vae_weight"):
        model.load_vae_weight()
    model.eval()
    del checkpoint

    # Create folder to save samples:
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = (
        os.path.basename(args.ckpt).replace(".pth", "").replace(".pt", "")
    )
    folder_name = (
        f"{model_string_name}-{ckpt_string_name}-size-{args.image_size}-"
        f"steps-{args.sample_steps}-cfg-{args.cfg_scale}-seed-{args.seed}"
    )

    sample_folder_dir = f"{args.sample_dir}/{folder_name}"

    if os.path.isfile(sample_folder_dir + ".npz"):
        if rank == 0:
            print(f"Found {sample_folder_dir}.npz, skipping sampling.")
        dist.barrier()
        dist.destroy_process_group()
        return 1
        
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    
    n = args.per_proc_batch_size
    world_size = dist.get_world_size()
    
    if args.num_fid_samples % args.num_classes != 0:
        if rank == 0:
            print(f"Warning: num_fid_samples ({args.num_fid_samples}) is not divisible by num_classes ({args.num_classes}). Truncating/Adjusting logic may apply.")
    
    images_per_class = args.num_fid_samples // args.num_classes
    class_label_gen_world = np.arange(0, args.num_classes).repeat(images_per_class)
    class_label_gen_world = np.hstack([class_label_gen_world, np.zeros(50000)])
    
    iterations = args.num_fid_samples // (n * world_size) + 1
    
    total_sampled = 0
    start_time = time.time()
    
    iter_wrapper = tqdm(range(iterations), desc="Sampling") if rank == 0 else range(iterations)

    for i in iter_wrapper:
        idx_start = world_size * n * i + rank * n
        idx_end = idx_start + n
        
        if idx_start >= args.num_fid_samples:
            break
            
        labels_np = class_label_gen_world[idx_start : idx_end]
        c_indices = torch.from_numpy(labels_np).long().to(device)
        
        current_batch_size = len(c_indices)
        if current_batch_size == 0:
            break

        # Sample inputs:
        with torch.amp.autocast("cuda", dtype=precision):
            samples = model.sample(
                c_indices,
                sample_steps=args.sample_steps,
                cfg_scale=args.cfg_scale,
                chunk_size=args.chunk_size,
            )

        samples = (
            torch.clamp(127.5 * samples + 128.0, 0, 255)
            .permute(0, 2, 3, 1)
            .to("cpu", dtype=torch.uint8)
            .numpy()
        )

        with ProcessPoolExecutor(int(os.cpu_count()// (world_size+1))) as executor:
            futures = []
            for b_id, sample in enumerate(samples):
                img_id = world_size * n * i + rank * n + b_id
                
                if img_id >= args.num_fid_samples:
                    break
                futures.append(executor.submit(save_image, sample, sample_folder_dir, img_id))


        total_sampled += (current_batch_size * world_size)
        if rank == 0 and i % 10 == 0: 
             print(f"Step {i}/{iterations}, sampled so far (approx): {total_sampled}, cost {time.time() - start_time:.2f} s")

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    
    if rank == 0 and args.to_npz:
        print(f"Total time taken for sampling: {time.time() - start_time:.2f} seconds")
        create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("Done.")
    
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = get_model_args()
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50000)
    parser.add_argument("--cfg-scale", type=float, default=4.6)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--sample-steps", type=int, default=100)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument(
        "--mixed-precision", type=str, default="bf16", choices=["none", "bf16"]
    )
    parser.add_argument("--to-npz", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=0)
    args = parser.parse_args()
    main(args)
