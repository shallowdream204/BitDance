import torch
from torch import nn
from einops import rearrange

from transformers import set_seed
from omegaconf import OmegaConf
from PIL import Image

import numpy as np
from torch import nn
from transformers import AutoTokenizer, Qwen3ForCausalLM, Qwen3Config

from modeling.utils import MLPconnector
from modeling.vision_encoder.autoencoder import VQModel
from modeling.vision_head.flow_head_parallel_x import DiffHead

from safetensors.torch import load_file as load_sft
import json
import os
from tqdm import tqdm

IMAGE_SIZE_LIST = [
    # --- 1024px Area ---
    [2048, 512],
    [1920, 512],
    [1536, 640],
    [1280, 768],
    [1152, 896],
    [1024, 1024],
    [896, 1152],
    [768, 1280],
    [640, 1536],
    [512, 1920],
    [512, 2048],
    # --- 512px Area ---
    [1024, 256],
    [896, 256],
    [640, 384],
    [512, 512],
    [384, 640],
    [256, 896],
    [256, 1024],
]

class BitDanceT2IPipeline:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        # LLM and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.llm_config = Qwen3Config.from_pretrained(model_path)
        self.llm_model = Qwen3ForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).eval().to(device)
        self.hidden_size = self.llm_config.hidden_size

        # Autoencoder
        with open(os.path.join(model_path, 'ae_config.json'), "r") as f:
            self.ae_config = json.load(f)
        self.ae = VQModel(**self.ae_config).eval()
        self.ae.load_state_dict(load_sft(os.path.join(model_path, 'ae.safetensors')), strict=True, assign=True)
        self.ae.to(device)
        self.vae_patch_size = 2 ** (len(self.ae_config['ddconfig']['ch_mult'])-1)

        # Vision head
        with open(os.path.join(model_path, 'vision_head_config.json'), "r") as f:
            self.vision_head_config = json.load(f)
        self.vision_head = DiffHead(**self.vision_head_config).eval()
        self.vision_head.load_state_dict(load_sft(os.path.join(model_path, 'vision_head.safetensors')), strict=True, assign=True)
        self.vision_head.to(device)
        self.parallel_num = self.vision_head_config['parallel_num']
        print(f'use {self.parallel_num}-token parallel prediction per step')
        self.ps = int(self.parallel_num ** 0.5)

        # Projector
        self.embed_vision_mlp = MLPconnector(self.ae_config['ddconfig']['z_channels'], self.hidden_size, "gelu_pytorch_tanh")
        self.embed_vision_mlp.load_state_dict(load_sft(os.path.join(model_path, 'projector.safetensors')), strict=True, assign=True)
        self.embed_vision_mlp.to(device)

        # 2D sinusoidal position embedding
        self.build_pos_embed()

    def build_pos_embed(self, max_len=4096):
        max_len = max_len // self.vae_patch_size
        pos_embed_1d = self._get_1d_sincos_pos_embed(self.hidden_size//2, max_len)
        pos_embed_1d = nn.Parameter(pos_embed_1d, requires_grad=False)
        self.pos_embed_1d = pos_embed_1d.to(self.device)

    def _get_1d_sincos_pos_embed(self, dim, max_len, pe_interpolation=1.0):
        assert dim % 2 == 0
        omega = torch.arange(dim // 2, dtype=torch.float32)
        omega /= dim / 2.0
        omega = 1.0 / 10000**omega  # (D/4,)

        pos = torch.arange(max_len, dtype=torch.float32) / pe_interpolation
        out = torch.einsum("m,d->md", pos, omega)  # (max_len, D/4)

        emb_sin = torch.sin(out)
        emb_cos = torch.cos(out)
        return torch.cat([emb_sin, emb_cos], dim=1)  # (max_len, D/2)

    def get_2d_embed(self, h, w, ps=1):
        emb_v = self.pos_embed_1d[:h]
        emb_h = self.pos_embed_1d[:w]

        grid_v = emb_v.view(h, 1, self.hidden_size//2).repeat(1, w, 1)
        grid_h = emb_h.view(1, w, self.hidden_size//2).repeat(h, 1, 1)

        pos_embed = torch.cat([grid_h, grid_v], dim=-1) # h w c

        return rearrange(pos_embed, '(h p1) (w p2) c -> (h w p1 p2) c', p1=ps, p2=ps)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        height: int = 1024,
        width: int = 1024,
        num_sampling_steps: int = 50,
        guidance_scale: float = 7.5,
        num_images: int = 1,
        seed: int = 1234,
    ):
        # Set seed for reproducibility
        if seed is not None:
            set_seed(seed)
        # Calculate max_length dynamically based on image_size and stride of 16
        max_length = (height // self.vae_patch_size) * (width // self.vae_patch_size)
        
        image_size = [height, width]
        if image_size not in IMAGE_SIZE_LIST:
            raise ValueError(f"image_size {image_size} is not supported. Please choose from {IMAGE_SIZE_LIST}")

        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            gen_images = self.gen_image(
                cond_prompt=f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
                uncond_prompt="<|im_start|>assistant\n",
                guidance_scale=guidance_scale,
                num_sampling_steps=num_sampling_steps,
                num_images=num_images,
                image_size=image_size,
                max_length=max_length,
                show_progress=True,
            )
        
        gen_images = (
            torch.clamp(127.5 * gen_images + 128.0, 0, 255)
            .permute(0, 2, 3, 1)
            .to("cpu", dtype=torch.uint8)
            .numpy()
        )
        pil_images = []
        for i in range(gen_images.shape[0]):
            img_array = gen_images[i]
            if img_array.dtype != np.uint8:
                img_array = img_array.astype(np.uint8)
            pil_images.append(Image.fromarray(img_array))

        return pil_images

    @torch.no_grad()
    def gen_image(self,
        cond_prompt,
        uncond_prompt=None,
        guidance_scale: float = 1.0,
        num_sampling_steps: int = 50,
        max_length: int = 64,
        num_images: int = 1,
        image_size = [256, 256],
        show_progress: bool = False,
    ):
        tokenizer = self.tokenizer
        device = self.device
        model = self.llm_model.model

        step_width = self.parallel_num
        num_steps = max_length // step_width

        cond_ids = torch.tensor(tokenizer.encode(cond_prompt), device=device, dtype=torch.long)
        cond_emb = model.embed_tokens(cond_ids)
        if guidance_scale > 1.0:
            uncond_ids = torch.tensor(tokenizer.encode(uncond_prompt), device=device, dtype=torch.long)
            uncond_emb = model.embed_tokens(uncond_ids)

        img_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
        res_h_token_id = tokenizer.convert_tokens_to_ids(f"<|res_{image_size[0] // self.vae_patch_size}|>")
        res_w_token_id = tokenizer.convert_tokens_to_ids(f"<|res_{image_size[1] // self.vae_patch_size}|>")
        img_start_emb = model.embed_tokens(torch.tensor([img_start_id, res_h_token_id, res_w_token_id], device=device))

        h, w = image_size[0] // self.vae_patch_size, image_size[1] // self.vae_patch_size
        # prepare diff pos embed
        pos_embed_for_diff = self.get_2d_embed(h, w, ps=self.ps if hasattr(self, 'ps') else 1).unsqueeze(0)

        # add query tokens for parallel decoding
        for i in range(1, self.parallel_num):
            query_token = torch.tensor([tokenizer.convert_tokens_to_ids(f"<|query_{i}|>")], device=self.device, dtype=torch.long)
            query_embed = self.llm_model.model.embed_tokens(query_token)
            img_start_emb = torch.cat([img_start_emb, query_embed], dim=0)

        input_embeds_cond = torch.cat(
            [cond_emb, img_start_emb], dim=0
        ).unsqueeze(0).repeat(num_images, 1, 1)
        outputs_c = model(
            inputs_embeds=input_embeds_cond[:, :-step_width, :],
            use_cache=True,
        )
        pkv_c = outputs_c.past_key_values

        # bidirectional attn
        bi_attn_mask = torch.ones(
                (input_embeds_cond.shape[0], 1, step_width, step_width+pkv_c[0][0].shape[2]),
                dtype=torch.bool,
                device=device,
            )
        outputs_c = model(
            inputs_embeds=input_embeds_cond[:, -step_width:, :],
            past_key_values=pkv_c,
            use_cache=True,
            attention_mask=bi_attn_mask,
        )
        pkv_c = outputs_c.past_key_values
        hidden_c = outputs_c.last_hidden_state[:, -step_width:] # [B, parallel_num, D]

        if guidance_scale > 1.0:
            input_embeds_uncond = torch.cat(
                [uncond_emb, img_start_emb], dim=0
            ).unsqueeze(0).repeat(num_images, 1, 1)
            outputs_u = model(
                inputs_embeds=input_embeds_uncond[:, :-step_width, :],
                use_cache=True,
            )
            pkv_u = outputs_u.past_key_values
            outputs_u = model(
                inputs_embeds=input_embeds_uncond[:, -step_width:, :],
                past_key_values=pkv_u,
                use_cache=True,
                attention_mask=bi_attn_mask,
            )
            pkv_u = outputs_u.past_key_values
            hidden_u = outputs_u.last_hidden_state[:, -step_width:] # [B, parallel_num, D]

        out_tokens = []
        if show_progress:
            pbar = tqdm(total=num_steps, desc="Decoding Steps")
        for step in range(num_steps):
            if show_progress:
                pbar.update(1)
            h_fused = torch.cat([hidden_c, hidden_u], dim=0) if guidance_scale > 1.0 else hidden_c
            h_fused = h_fused + pos_embed_for_diff[:, step*step_width:(step+1)*step_width, :]
            pred_latents = self.vision_head.sample(h_fused, num_sampling_steps=num_sampling_steps, cfg=guidance_scale)
            # important! LFQ is used here
            curr_tokens = torch.sign(pred_latents)
            curr_embeds = self.embed_vision_mlp(curr_tokens)
            out_tokens.append(curr_tokens[:num_images])
            model_input = curr_embeds # [B, N, D]
            # 2d pos embed
            model_input = model_input + pos_embed_for_diff[:, step*step_width:(step+1)*step_width, :]
 
            # bidirectional attn mask
            bi_attn_mask = torch.ones(
                (model_input.shape[0], 1, model_input.shape[1], model_input.shape[1]+pkv_c[0][0].shape[2]), 
                dtype=torch.bool,
                device=device
            )
            outputs_c = model(inputs_embeds=model_input[:num_images], past_key_values=pkv_c, use_cache=True, attention_mask=bi_attn_mask[:num_images])
            pkv_c = outputs_c.past_key_values
            hidden_c = outputs_c.last_hidden_state[:, -step_width:] # [B, parallel_num, D]

            if guidance_scale > 1.0:
                outputs_u = model(inputs_embeds=model_input[num_images:], past_key_values=pkv_u, use_cache=True, attention_mask=bi_attn_mask[num_images:])
                pkv_u = outputs_u.past_key_values
                hidden_u = outputs_u.last_hidden_state[:, -step_width:] # [B, parallel_num, D]

        full_output = torch.cat(out_tokens, dim=1)
        image = self.decode_image(full_output, [h, w], ps=self.ps if hasattr(self, 'ps') else 1) # [num_images, c, h, w]
        return image

    def decode_image(self, image_latents, image_size=None, ps=1):
        if image_size is None:
            h = w = int(image_latents.size(1) ** 0.5)
        else:
            h, w = image_size
            
        image_latents = rearrange(image_latents, 'b (h w p1 p2) c -> b c (h p1) (w p2)', h=h//ps, w=w//ps, p1=ps, p2=ps)
        output = self.ae.decode(image_latents)     # [1, c, h, w]
        
        return output