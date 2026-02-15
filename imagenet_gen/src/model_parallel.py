import argparse
from functools import partial

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .diff_head_parallel import DiffHead
from .layers_parallel import TransformerBlock, get_2d_pos, precompute_freqs_cis_2d
from .qae import VQModel
from .utils import patchify_raster, unpatchify_raster, patchify_raster_2d



def get_model_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, choices=list(BitDance_models.keys()), default="BitDance-L"
    )
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--down-size", type=int, default=16, choices=[16])
    parser.add_argument("--patch-size", type=int, default=1, choices=[1, 2, 4])
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cls-token-num", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--diff-batch-mul", type=int, default=4)
    parser.add_argument("--grad-checkpointing", action="store_true")
    parser.add_argument("--trained-vae", type=str, default="")
    parser.add_argument("--drop-rate", type=float, default=0.0)
    parser.add_argument("--perturb-schedule", type=str, default="constant")
    parser.add_argument("--perturb-rate", type=float, default=0.0)
    parser.add_argument("--perturb-rate-max", type=float, default=0.3)
    parser.add_argument("--time-schedule", type=str, default='logit_normal')
    parser.add_argument("--time-shift", type=float, default=1.)
    parser.add_argument("--parallel-num", type=int, default=4)
    parser.add_argument("--P-std", type=float, default=0.8)
    parser.add_argument("--P-mean", type=float, default=-0.8)
    parser.add_argument("--parallel-mode", type=str, default='patch', choices=['standard', 'patch'])
    return parser


def create_model(args, device):
    model = BitDance_models[args.model](
        resolution=args.image_size,
        down_size=args.down_size,
        patch_size=args.patch_size,
        latent_dim=args.latent_dim,
        diff_batch_mul=args.diff_batch_mul,
        cls_token_num=args.cls_token_num,
        num_classes=args.num_classes,
        grad_checkpointing=args.grad_checkpointing,
        trained_vae=args.trained_vae,
        drop_rate=args.drop_rate,
        perturb_schedule=args.perturb_schedule,
        perturb_rate=args.perturb_rate,
        perturb_rate_max=args.perturb_rate_max,
        time_schedule=args.time_schedule,
        time_shift=args.time_shift,
        parallel_num=args.parallel_num,
        P_std=args.P_std,
        P_mean=args.P_mean,
        parallel_mode=args.parallel_mode,
    ).to(device, memory_format=torch.channels_last)
    return model

class MLPConnector(nn.Module):
    def __init__(self, in_dim, dim, dropout_p=0.0):
        super().__init__()
        hidden_dim = int(dim * 1.5)
        self.w1 = nn.Linear(in_dim, hidden_dim * 2, bias=True)
        self.w2 = nn.Linear(hidden_dim, dim, bias=True)
        self.ffn_dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        h1, h2 = self.w1(x).chunk(2, dim=-1)
        return self.ffn_dropout(self.w2(F.silu(h1) * h2))

def flip_tensor_elements_uniform_prob(tensor: torch.Tensor, p_max: float) -> torch.Tensor:
    if not 0.0 <= p_max <= 1.0:
        raise ValueError(f"p_max must be in [0.0, 1.0] range, but got: {p_max}")
    r1 = torch.rand_like(tensor)
    r2 = torch.rand_like(tensor)
    flip_mask = r1 < p_max * r2
    multiplier = torch.where(flip_mask, -1.0, 1.0)
    multiplier = multiplier.to(tensor.dtype)
    flipped_tensor = tensor * multiplier
    return flipped_tensor

def get_block_causal_mask(num_tokens_total, num_tokens_causal, block_size):
    assert (num_tokens_total - num_tokens_causal) % block_size == 0
    attention_mask = torch.zeros(num_tokens_total, num_tokens_total)
    causal_mask = torch.triu(torch.ones(num_tokens_total, num_tokens_total), diagonal=1)
    attention_mask.masked_fill_(causal_mask.bool(), float('-inf'))

    for i in range(num_tokens_causal, num_tokens_total, block_size):
        start_idx = i
        end_idx = i + block_size
        attention_mask[start_idx:end_idx, start_idx:end_idx] = 0

    return attention_mask

class BitDance(nn.Module):

    def __init__(
        self,
        dim,
        n_layer,
        n_head,
        diff_layers,
        diff_dim,
        diff_adanln_layers,
        latent_dim,
        down_size,
        patch_size,
        resolution,
        diff_batch_mul,
        grad_checkpointing=False,
        cls_token_num=16,
        num_classes: int = 1000,
        class_dropout_prob: float = 0.1,
        trained_vae: str = "",
        drop_rate: float = 0.0,
        perturb_schedule: str = "constant",
        perturb_rate: float = 0.0,
        perturb_rate_max: float = 0.3,
        time_schedule: str = 'logit_normal',
        time_shift: float = 1.,
        parallel_num: int = 4,
        P_std: float = 1.,
        P_mean: float = 0.,
        parallel_mode: str = 'standard',
    ):
        super().__init__()

        self.n_layer = n_layer
        self.resolution = resolution
        self.down_size = down_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.cls_token_num = cls_token_num
        self.class_dropout_prob = class_dropout_prob
        self.latent_dim = latent_dim
        self.trained_vae = trained_vae
        self.perturb_schedule = perturb_schedule
        self.perturb_rate = perturb_rate
        self.perturb_rate_max = perturb_rate_max
        self.parallel_num = parallel_num
        self.parallel_mode = parallel_mode

        # define the vae and mar model
        ddconfig = {
        "double_z": False,
        "z_channels": latent_dim,
        "in_channels": 3,
        "out_ch": 3,
        "ch": 256,
        "ch_mult": [1,1,2,2,4],
        "num_res_blocks": 4
    }
        num_codebooks = 4
        # print(f"loading vae unexpected_keys: {unexpected_keys}")
        self.vae = VQModel(ddconfig, num_codebooks)
        self.grad_checkpointing = grad_checkpointing

        self.cls_embedding = nn.Embedding(num_classes + 1, dim * self.cls_token_num)
        self.query_token = nn.Parameter(torch.randn(1, self.parallel_num - 1, dim) * 0.02)
        self.proj_in = MLPConnector(latent_dim * self.patch_size * self.patch_size, dim, drop_rate)
        self.emb_norm = nn.RMSNorm(dim, eps=1e-6, elementwise_affine=True)
        self.h, self.w = resolution // (down_size * patch_size), resolution // (down_size * patch_size)
        self.total_tokens = self.h * self.w + self.cls_token_num

        self.layers = torch.nn.ModuleList()
        for layer_id in range(n_layer):
            self.layers.append(
                TransformerBlock(
                    dim,
                    n_head,
                    resid_dropout_p=drop_rate,
                )
            )

        self.norm = nn.RMSNorm(dim, eps=1e-6, elementwise_affine=True)
        self.pos_for_diff = nn.Embedding(self.h * self.w, dim)
        self.head = DiffHead(
            ch_target=latent_dim * self.patch_size * self.patch_size,
            ch_cond=dim,
            ch_latent=diff_dim,
            depth_latent=diff_layers,
            depth_adanln=diff_adanln_layers,
            grad_checkpointing=grad_checkpointing,
            time_shift=time_shift,
            time_schedule=time_schedule,
            parallel_num=parallel_num,
            P_std=P_std,
            P_mean=P_mean,
        )
        self.diff_batch_mul = diff_batch_mul

        patch_2d_pos = get_2d_pos(resolution, int(down_size * patch_size))

        freqs_cis = precompute_freqs_cis_2d(
                patch_2d_pos,
                dim // n_head,
                10000,
                cls_token_num=self.cls_token_num + self.parallel_num - 1,
            )
        
        if self.parallel_mode == 'patch':
            freqs_cis[-self.h * self.w:] = patchify_raster_2d(freqs_cis[-self.h * self.w:], int(self.parallel_num ** 0.5), self.h, self.w)

        self.register_buffer("freqs_cis", freqs_cis[:-self.parallel_num], persistent=False)

        attn_mask = get_block_causal_mask(self.h * self.w + self.cls_token_num -1, self.cls_token_num -1, self.parallel_num)
        self.register_buffer("attn_mask", attn_mask.unsqueeze(0).unsqueeze(0), persistent=False)
        self.freeze_vae()

        self.initialize_weights()

    def load_vae_weight(self):
        state = torch.load(
                    self.trained_vae, 
                    map_location="cpu",
                )
        missing_keys, unexpected_keys = self.vae.load_state_dict(state["state_dict"], strict=False)
        print(f"loading vae, missing_keys: {missing_keys}")
        del state

    def non_decay_keys(self):
        return ["proj_in", "cls_embedding", "query_token"]

    def freeze_module(self, module: nn.Module):
        for param in module.parameters():
            param.requires_grad = False

    def freeze_vae(self):
        self.freeze_module(self.vae)
        self.vae.eval()

    def initialize_weights(self):
        # Initialize nn.Linear and nn.Embedding
        self.apply(self.__init_weights)
        self.head.initialize_weights()
        # self.vae.initialize_weights()

    def __init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def drop_label(self, class_id):
        if self.class_dropout_prob > 0.0 and self.training:
            is_drop = (
                torch.rand(class_id.shape, device=class_id.device)
                < self.class_dropout_prob
            )
            class_id = torch.where(is_drop, self.num_classes, class_id)
        return class_id

    def patchify(self, x):
        bsz, c, h, w = x.shape
        p = self.patch_size
        h_, w_ = h // p, w // p

        x = x.reshape(bsz, c, h_, p, w_, p)
        x = torch.einsum('nchpwq->nhwcpq', x)
        x = x.reshape(bsz, h_ * w_, c * p ** 2)
        return x  # [n, l, d]

    def unpatchify(self, x):
        bsz = x.shape[0]
        p = self.patch_size
        c = self.latent_dim
        h_, w_ = self.h, self.w

        x = x.reshape(bsz, h_, w_, c, p, p)
        x = torch.einsum('nhwcpq->nchpwq', x)
        x = x.reshape(bsz, c, h_ * p, w_ * p)
        return x  # [n, c, h, w]

    def forward(
        self,
        images,
        class_id,
        cached=False
    ):
        if cached:
            vae_latent = images
        else:
            vae_latent, _, _, _ = self.vae.encode(images) # b c h w

        if self.parallel_mode == 'standard':
            vae_latent = self.patchify(vae_latent)
        elif self.parallel_mode == 'patch':
            vae_latent = patchify_raster(vae_latent, int(self.parallel_num ** 0.5))
        else:
            raise NotImplementedError(f"unknown parallel_mode {self.parallel_mode}")
        x = vae_latent.clone().detach()
        if self.training:
            if self.perturb_schedule =="constant":
                x = flip_tensor_elements_uniform_prob(x, self.perturb_rate)
            else:
                raise NotImplementedError(f"unknown perturb_schedule {self.perturb_schedule}")
        x = self.proj_in(x[:, :-self.parallel_num, :])
        class_id = self.drop_label(class_id)
        bsz = x.shape[0]
        c = self.cls_embedding(class_id).view(bsz, self.cls_token_num, -1)
        query_token = self.query_token.repeat(bsz, 1, 1)
        x = torch.cat([c, query_token, x], dim=1)
        x = self.emb_norm(x)

        if self.grad_checkpointing and self.training:
            for layer in self.layers:
                block = partial(layer.forward, mask=self.attn_mask, freqs_cis=self.freqs_cis)
                x = checkpoint(block, x, use_reentrant=False)
        else:
            for layer in self.layers:
                x = layer(x, self.attn_mask, self.freqs_cis)

        x = x[:, -self.h * self.w :, :]
        x = self.norm(x)
        x = x + self.pos_for_diff.weight

        target = vae_latent.clone().detach()
        x = x.view(-1, self.parallel_num, x.shape[-1])
        target = target.view(-1, self.parallel_num, target.shape[-1])

        x = x.repeat(self.diff_batch_mul, 1, 1)
        target = target.repeat(self.diff_batch_mul, 1, 1)
        loss = self.head(target, x)

        return loss

    def enable_kv_cache(self, bsz):
        for layer in self.layers:
            layer.attention.enable_kv_cache(bsz, self.total_tokens)

    @torch.compile()
    def forward_model(self, x, mask, start_pos, end_pos):
        x = self.emb_norm(x)
        for layer in self.layers:
            x = layer.forward_onestep(
                x, mask, self.freqs_cis[start_pos:end_pos,], start_pos, end_pos
            )
        x = self.norm(x)
        return x
    
    def head_sample(self, x, diff_pos, sample_steps, cfg_scale, cfg_schedule="linear"):
        x = x + self.pos_for_diff.weight[diff_pos*self.parallel_num : (diff_pos+1)*self.parallel_num, :]
        # x = x.view(-1, x.shape[-1])
        seq_len = self.h * self.w // self.parallel_num
        if cfg_scale > 1.0:
            if cfg_schedule == "constant":
                cfg_iter = cfg_scale
            elif cfg_schedule == "linear":
                start = 1.0
                cfg_iter = start + (cfg_scale - start) * diff_pos / seq_len
            else:
                raise NotImplementedError(f"unknown cfg_schedule {cfg_schedule}")
        else:
            cfg_iter = 1.0
        pred = self.head.sample(x, num_sampling_steps=sample_steps, cfg=cfg_iter)
        # Important: LFQ here, sign the prediction
        pred = torch.sign(pred)
        return pred

    @torch.no_grad()
    def sample(self, cond, sample_steps, cfg_scale=1.0, cfg_schedule="linear", chunk_size=0):
        self.eval()
        if cfg_scale > 1.0:
            cond_null = torch.ones_like(cond) * self.num_classes
            cond_combined = torch.cat([cond, cond_null])
        else:
            cond_combined = cond
        bsz = cond_combined.shape[0]
        act_bsz = bsz // 2 if cfg_scale > 1.0 else bsz
        self.enable_kv_cache(bsz)

        c = self.cls_embedding(cond_combined).view(bsz, self.cls_token_num, -1)
        last_pred = None
        all_preds = []
        for i in range(self.h * self.w // self.parallel_num):
            if i == 0:
                x = self.forward_model(torch.cat([c, self.query_token.repeat(bsz, 1, 1)], dim=1), self.attn_mask[:, :, :self.cls_token_num + self.parallel_num - 1, :self.cls_token_num + self.parallel_num - 1], 0, self.cls_token_num + self.parallel_num - 1)
            else:
                x = self.proj_in(last_pred)
                start_pos = self.parallel_num * (i-1) + self.cls_token_num + self.parallel_num - 1
                x = self.forward_model(
                    x,
                    self.attn_mask[:, :, start_pos : start_pos + self.parallel_num, : start_pos + self.parallel_num],
                    start_pos,
                    start_pos + self.parallel_num
                )
                
            last_pred = self.head_sample(
                x[:, -self.parallel_num:, :],
                i,
                sample_steps,
                cfg_scale,
                cfg_schedule,
            )
            all_preds.append(last_pred)

        x = torch.cat(all_preds, dim=-2)[:act_bsz]
        if x.dim() == 3: #b n c -> b c h w
            if self.parallel_mode == 'standard':
                x = self.unpatchify(x)
            elif self.parallel_mode == 'patch':
                x = unpatchify_raster(x, int(self.parallel_num ** 0.5), (self.h, self.w))
        # recon = self.vae.decode(x)
        if chunk_size > 0:
            recon = self.decode_in_chunks(x, chunk_size)
        else:
            recon = self.vae.decode(x)
        return recon
    
    def decode_in_chunks(self, latent_tensor, chunk_size=64):
        total_bsz = latent_tensor.shape[0]
        recon_chunks_on_cpu = []
        with torch.no_grad():
            for i in range(0, total_bsz, chunk_size):
                end_idx = min(i + chunk_size, total_bsz)
                latent_chunk = latent_tensor[i:end_idx]
                recon_chunk = self.vae.decode(latent_chunk)
                recon_chunks_on_cpu.append(recon_chunk.cpu())
        return torch.cat(recon_chunks_on_cpu, dim=0)

    def get_fsdp_wrap_module_list(self):
        return list(self.layers)

def BitDance_H(**kwargs):
    return BitDance(
        n_layer=40,
        n_head=20,
        dim=1280,
        diff_layers=12,
        diff_dim=1280,
        diff_adanln_layers=3,
        **kwargs,
    )


def BitDance_L(**kwargs):
    return BitDance(
        n_layer=32,
        n_head=16,
        dim=1024,
        diff_layers=8,
        diff_dim=1024,
        diff_adanln_layers=2,
        **kwargs,
    )


def BitDance_B(**kwargs):
    return BitDance(
        n_layer=24,
        n_head=12,
        dim=768,
        diff_layers=6,
        diff_dim=768,
        diff_adanln_layers=2,
        **kwargs,
    )


BitDance_models = {
    "BitDance-B": BitDance_B,
    "BitDance-L": BitDance_L,
    "BitDance-H": BitDance_H,
}