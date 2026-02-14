import os
import logging
import argparse
import datetime
import torch
import torch.distributed as dist
from transformers import set_seed
from PIL import Image, ImageDraw, ImageFont
import textwrap

from modeling.t2i_pipeline import BitDanceT2IPipeline


class BaseEvaluator:
    def __init__(self, model_path):
        self.init_dist()
        
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)
        seed = 4396 * self.world_size + self.rank
        set_seed(seed)
        self.model = BitDanceT2IPipeline(model_path=model_path, device=self.device)

    def init_dist(self):
        self.rank = int(os.environ["RANK"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.device = torch.device("cuda", self.local_rank)
        torch.cuda.set_device(self.local_rank)
        dist.init_process_group(
            backend="nccl",
            rank=self.rank,
            world_size=self.world_size,
            device_id=self.device,
            timeout=datetime.timedelta(seconds=3600))

    def build_dataset(self):
        pass

    @torch.no_grad()
    def eval(self):
        self.build_dataset()
        pass

    def make_visualization(
        self, raw_pil, edited_tensor, prompt, out_path,
        max_prompt_width=100, pad=16, bg_color=(255, 255, 255),
    ):
        edited_img = (edited_tensor.clamp(0, 1).mul(255).byte().cpu())
        edited_pil = Image.fromarray(edited_img.permute(1, 2, 0).numpy())

        h = max(raw_pil.height, edited_pil.height)
        def resize_to_height(img):
            if img.height == h:
                return img
            new_w = int(img.width * (h / img.height))
            return img.resize((new_w, h), Image.BILINEAR)

        left = resize_to_height(raw_pil)
        right = resize_to_height(edited_pil)

        font = ImageFont.load_default()
        # change font size
        font = font.font_variant(size=32)

        wrapped_lines = textwrap.wrap(prompt, width=max_prompt_width)
        line_height = font.getbbox("Hg")[3] - font.getbbox("Hg")[1] if hasattr(font, "getbbox") else 20
        text_block_height = line_height * len(wrapped_lines) + pad * 2

        canvas_w = left.width + right.width + pad * 3
        canvas_h = h + text_block_height + pad
        canvas = Image.new("RGB", (canvas_w, canvas_h), color=bg_color)

        x_left = pad
        x_right = pad * 2 + left.width
        y_img = pad
        canvas.paste(left, (x_left, y_img))
        canvas.paste(right, (x_right, y_img))

        draw = ImageDraw.Draw(canvas)
        text_x = pad
        text_y = y_img + h + pad
        for i, line in enumerate(wrapped_lines):
            draw.text((text_x, text_y + i * line_height), line, fill=(0, 0, 0), font=font)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        canvas.save(out_path)

            
    
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/BitDance-14B-64x",
    )
    return parser