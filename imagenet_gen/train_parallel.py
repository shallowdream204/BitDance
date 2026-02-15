# Modified from:
#   llamagen: https://github.com/FoundationVision/LlamaGen/
import math

import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import inspect
import os
import shutil
import time
from copy import deepcopy
from multiprocessing import Pool

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from src.dataset import build_dataset
from src.model_parallel import create_model, get_model_args
from src.utils import create_logger, requires_grad, update_ema

import re
import glob


def create_optimizer(model, weight_decay, learning_rate, betas, logger):
    def is_decay_param(name, param, no_decay_keys):
        for key in no_decay_keys:
            if key in name:
                return False
        if param.dim() < 2:
            return False
        return True

    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    no_decay_keys = model.non_decay_keys() if hasattr(model, "non_decay_keys") else []
    decay_params = [
        p for n, p in param_dict.items() if is_decay_param(n, p, no_decay_keys)
    ]
    nodecay_params = [
        p for n, p in param_dict.items() if not is_decay_param(n, p, no_decay_keys)
    ]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    logger.info(
        f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
    )
    logger.info(
        f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
    )
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if fused_available else dict()
    optimizer = torch.optim.AdamW(
        optim_groups, lr=learning_rate, betas=betas, **extra_args
    )
    logger.info(f"using fused AdamW: {fused_available}")
    return optimizer


def adjust_learning_rate(args, cur_steps, total_steps, optimizer):
    if cur_steps < args.warmup_steps and args.warmup_steps > 0:
        lr = args.lr * cur_steps / args.warmup_steps
    elif (
        args.decay_start > 0
        and cur_steps >= args.decay_start
        and args.decay_start < total_steps
    ):
        # decay from decay_start to total_steps, with learning rate cosine decay to min_lr
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (
            1.0
            + math.cos(
                math.pi
                * (cur_steps - args.decay_start)
                / max(total_steps - args.decay_start, 1e-8)
            )
        )
    else:
        lr = args.lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def init_distributed_mode(args):
    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.gpu = int(os.environ["LOCAL_RANK"])

    args.distributed = True
    device = torch.device("cuda", args.gpu)
    torch.cuda.set_device(device)

    print(f"| distributed init (rank {args.rank}, gpu {args.gpu})", flush=True)

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=args.world_size,
        rank=args.rank,
        device_id=device,
    )
    dist.barrier()
    return device


def get_orig_model(model):
    if isinstance(model, DDP):
        model = model.module
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


def _linear_decay_ratio(epoch: int, start: int, end: int) -> float:
    if start < 0 or end <= start:
        return 1.0
    if epoch < start:
        r = 1.0
    elif epoch >= end:
        r = 0.0
    else:
        r = 1.0 - (epoch - start) / float(end - start)
    return max(0.0, min(1.0, r))


def create_dataloader(dataset, sampler, epoch, args):
    sampler.set_epoch(epoch)
    dataset.set_epoch(epoch)
    # linear decay of aug_ratio
    aug_ratio = _linear_decay_ratio(
        epoch, args.aug_decay_start_epoch, args.aug_decay_end_epoch
    )
    dataset.set_aug_ratio(aug_ratio)
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader


def update_loss_dict(running_loss_dict, **kwargs):
    for k, v in kwargs.items():
        if v is not None:
            if torch.is_tensor(v):
                v = v.item()
            running_loss_dict[k] = running_loss_dict.get(k, 0.0) + v
    return running_loss_dict


def logging(
    running_loss_dict,
    running_gnorm,
    log_steps,
    steps_per_sec,
    train_steps,
    device,
    logger,
    tsb_writer,
):
    keys = sorted(running_loss_dict.keys())
    running_losses = [running_loss_dict[k] for k in keys]
    # Reduce loss history over all processes:
    all_loss = torch.tensor(
        running_losses,
        device=device,
    )
    dist.all_reduce(all_loss, op=dist.ReduceOp.SUM)

    avg_gnorm = running_gnorm / log_steps
    all_loss = [
        (keys[i], all_loss[i].item() / dist.get_world_size() / log_steps)
        for i in range(len(keys))
    ]
    loss_str = ", ".join([f"{k}: {v:.4f}" for k, v in all_loss])

    logger.info(
        f"(step={train_steps:07d}): {loss_str} ,Train Steps/Sec: {steps_per_sec:.2f}, Train Grad Norm: {avg_gnorm:.4f}"
    )
    if tsb_writer is not None:
        for k, v in all_loss:
            tsb_writer.add_scalar(f"train/{k}", v, train_steps)
        tsb_writer.add_scalar("train/steps_per_sec", steps_per_sec, train_steps)
        tsb_writer.add_scalar("train/grad_norm", avg_gnorm, train_steps)


def copy_ckp_func(src_file, dest_path):
    shutil.copy(src_file, dest_path)

def find_latest_epoch_ckpt(results_dir, pattern=r"epoch_(\d+)\.(pt|pth)$"):
    candidates = []
    for path in glob.glob(os.path.join(results_dir, "epoch_*.pt")) + glob.glob(os.path.join(results_dir, "epoch_*.pth")):
        filename = os.path.basename(path)
        m = re.match(pattern, filename)
        if m:
            epoch_num = int(m.group(1))
            candidates.append((epoch_num, path))
    if not candidates:
        return None, None
    candidates.sort(key=lambda x: x[0], reverse=True)
    latest_epoch, latest_path = candidates[0]
    return latest_path, latest_epoch


def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    device = init_distributed_mode(args)
    assert (
        args.global_batch_size % dist.get_world_size() == 0
    ), f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    results_dir = args.results_dir

    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        logger = create_logger(results_dir)
        logger.info(f"Experiment directory created at {results_dir}")
        ckp_async_thread = Pool(processes=1)
    else:
        logger = create_logger(None)
        ckp_async_thread = None

    # training args
    logger.info(f"{args}")

    # training env
    logger.info(
        f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}."
    )

    model = create_model(args, device)

    if args.trained_vae != "":
        model.load_vae_weight()
        model.freeze_vae()
        logger.info(f"loaded pretrained VAE from {args.trained_vae}")

    logger.info(model)
    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if args.ema > 0:
        ema_model = deepcopy(model).to(
            device
        )  # Create an EMA of the model for use after training
        requires_grad(ema_model, False)
        logger.info(
            f"EMA Parameters: {sum(p.numel() for p in ema_model.parameters()):,}"
        )

    # Setup optimizer
    optimizer = create_optimizer(
        model, args.weight_decay, args.lr, (args.beta1, args.beta2), logger
    )

    dataset = build_dataset(args)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed,
    )

    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    checkpoint_path, _ = find_latest_epoch_ckpt(results_dir)
    total_steps = args.epochs * int(len(dataset) / args.global_batch_size)

    # Prepare models for training:
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        start_epoch = checkpoint["epochs"]
        train_steps = int(start_epoch * int(len(dataset) / args.global_batch_size))
        model.load_state_dict(checkpoint["model"], strict=True)
        if args.ema > 0:
            ema_model.load_state_dict(
                checkpoint["ema"] if "ema" in checkpoint else checkpoint["model"]
            )
        optimizer.load_state_dict(checkpoint["optimizer"])

        del checkpoint

        logger.info(f"Resume training from checkpoint: {checkpoint_path}")
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
    else:
        train_steps = 0
        start_epoch = 0
        if args.ema > 0:
            update_ema(ema_model, model, decay=0)

    if not args.no_compile:
        logger.info("compiling the model... (may take several minutes)")
        model = torch.compile(model)  # requires PyTorch 2.0

    model = DDP(model.to(device), device_ids=[args.gpu])
    model.train()
    model.module.vae.eval()

    if args.ema > 0:
        ema_model.eval()

    ptdtype = {"none": torch.float32, "bf16": torch.bfloat16}[args.mixed_precision]

    log_steps = 0
    running_loss_dict = {}
    running_gnorm = 0
    start_time = time.time()

    logger.info(f"Training for {args.epochs} epochs ({total_steps} steps)")
    tsb_writer = SummaryWriter(log_dir=results_dir) if rank == 0 else None

    for epoch in range(start_epoch, args.epochs):
        loader = create_dataloader(dataset, sampler, epoch, args)

        logger.info(f"Beginning epoch {epoch}...")
        for images, classes in loader:
            classes = classes.to(device, non_blocking=True)
            images = images.to(device, non_blocking=True).contiguous(
                memory_format=torch.channels_last
            )

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=ptdtype):
                ar_loss = model(images, classes)

            running_loss_dict = update_loss_dict(running_loss_dict, loss=ar_loss)
            loss = ar_loss
            loss.backward()

            if args.max_grad_norm != 0.0:
                gnorm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm
                )
                running_gnorm += gnorm.item()
            cur_lr = adjust_learning_rate(args, train_steps, total_steps, optimizer)
            running_loss_dict = update_loss_dict(running_loss_dict, lr=cur_lr)
            optimizer.step()

            if args.ema > 0:
                update_ema(ema_model, get_orig_model(model), decay=args.ema)

            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                logging(
                    running_loss_dict,
                    running_gnorm,
                    log_steps,
                    steps_per_sec,
                    train_steps,
                    device,
                    logger,
                    tsb_writer,
                )
                running_loss_dict = {}
                running_gnorm = 0
                log_steps = 0
                start_time = time.time()

        # save checkpoint at the end of each epoch
        cur_epoch = epoch + 1
        if rank == 0 and cur_epoch % args.keep_freq == 0:
            checkpoint = {
                "model": get_orig_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "epochs": epoch + 1,
                "args": args,
            }
            if args.ema > 0:
                checkpoint["ema"] = ema_model.state_dict()
            local_file = os.path.join(args.tmp_results_dir, f"epoch_{cur_epoch}.pt")
            torch.save(checkpoint, local_file)
            ckp_async_thread.apply_async(
                copy_ckp_func,
                args=(local_file, results_dir),
                error_callback=lambda e: logger.error("async copy error :" + str(e)),
            )

        dist.barrier()

    if ckp_async_thread is not None:
        ckp_async_thread.close()
        ckp_async_thread.join()

    logger.info("Done!")
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = get_model_args()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--aug-decay-start-epoch", type=int, default=700)
    parser.add_argument("--aug-decay-end-epoch", type=int, default=750)
    parser.add_argument("--ema", default=-1, type=float)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--tmp-results-dir", type=str, default="/tmp/")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=20000)
    parser.add_argument("--decay-start", type=int, default=20000)
    parser.add_argument(
        "--weight-decay", type=float, default=5e-2, help="Weight decay to use"
    )
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument(
        "--max-grad-norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--mixed-precision", type=str, default="bf16", choices=["none", "bf16"]
    )

    parser.add_argument("--keep-freq", type=int, default=50)
    main(parser.parse_args())
