import os
import copy
from tqdm import tqdm
import json
import torch
import torch.distributed as dist
from torchvision.utils import save_image, make_grid

from eval.base_evaluator import BaseEvaluator, get_parser

class DPGEvaluator(BaseEvaluator):
    
    def build_dataset(self, data_path):
        self.datasets = []
        lines = json.load(open(data_path))
        for id, prompt in lines.items():
            data_dict = {
                "id": id,
                "prompt": prompt,
                "cond_prompt": f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
                "uncond_prompt": "<|im_start|>assistant\n"
            }
            self.datasets.append(data_dict)
        # split to different ranks
        total = len(self.datasets)
        per_rank = (total + self.world_size - 1) // self.world_size
        start = self.rank * per_rank
        end = min(start + per_rank, total)
        self.datasets = self.datasets[start:end]

    @torch.no_grad()
    def eval(self, data_path, save_dir, guidance_scale=1.0, num_sampling_steps=50, image_size=[512, 512]):
        os.makedirs(save_dir, exist_ok=True)
        self.build_dataset(data_path)
        max_length = (image_size[0] // self.model.vae_patch_size) * (image_size[1] // self.model.vae_patch_size)
        for in_data in tqdm(self.datasets):
            data = copy.deepcopy(in_data)
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                gen_images = self.model.gen_image(
                    cond_prompt=data["cond_prompt"],
                    uncond_prompt=data["uncond_prompt"],
                    guidance_scale=guidance_scale,
                    num_sampling_steps=num_sampling_steps,
                    num_images=4,
                    image_size=image_size,
                    max_length=max_length)
            cat_img = gen_images.clamp(-1, 1) * 0.5 + 0.5
            # save the 4 images as a 2x2 grid
            grid_img = make_grid(cat_img, nrow=2, padding=0)
            save_image(grid_img, os.path.join(save_dir, f"{data['id']}.png"))
        dist.barrier()
        dist.destroy_process_group()
            

if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="eval/dpg_bench/prompts.json",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/dpg",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--num_sampling_steps",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=[512, 512],
    )
    args = parser.parse_args()
    evaluator = DPGEvaluator(args.model_path)
    evaluator.eval(args.data_path, args.save_dir, args.guidance_scale, args.num_sampling_steps, args.image_size)
