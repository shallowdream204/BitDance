import argparse
from functools import partial

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .diff_head import DiffHead
from .layers import TransformerBlock, get_2d_pos, precompute_freqs_cis_2d
from .qae import VQModel

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
    parser.add_argument("--P-std", type=float, default=0.8)
    parser.add_argument("--P-mean", type=float, default=-0.8)
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
        P_std=args.P_std,
        P_mean=args.P_mean,
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
        P_std: float = 1.,
        P_mean: float = 0.,
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
                    causal=True,
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
            P_std=P_std,
            P_mean=P_mean,
        )
        self.diff_batch_mul = diff_batch_mul

        patch_2d_pos = get_2d_pos(resolution, int(down_size * patch_size))

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis_2d(
                patch_2d_pos,
                dim // n_head,
                10000,
                cls_token_num=self.cls_token_num,
            )[:-1],
            persistent=False,
        )
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
        return ["proj_in", "cls_embedding"]

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

        vae_latent = self.patchify(vae_latent)
        x = vae_latent.clone().detach()
        if self.training:
            if self.perturb_schedule =="constant":
                x = flip_tensor_elements_uniform_prob(x, self.perturb_rate)
            else:
                raise NotImplementedError(f"unknown perturb_schedule {self.perturb_schedule}")
        x = self.proj_in(x[:, :-1, :])
        class_id = self.drop_label(class_id)
        bsz = x.shape[0]
        c = self.cls_embedding(class_id).view(bsz, self.cls_token_num, -1)
        x = torch.cat([c, x], dim=1)
        x = self.emb_norm(x)

        if self.grad_checkpointing and self.training:
            for layer in self.layers:
                block = partial(layer.forward, freqs_cis=self.freqs_cis)
                x = checkpoint(block, x, use_reentrant=False)
        else:
            for layer in self.layers:
                x = layer(x, self.freqs_cis)

        x = x[:, -self.h * self.w :, :]
        x = self.norm(x)
        x = x + self.pos_for_diff.weight

        target = vae_latent.clone().detach()
        x = x.view(-1, x.shape[-1])
        target = target.view(-1, target.shape[-1])

        x = x.repeat(self.diff_batch_mul, 1)
        target = target.repeat(self.diff_batch_mul, 1)
        loss = self.head(target, x)

        return loss

    def enable_kv_cache(self, bsz):
        for layer in self.layers:
            layer.attention.enable_kv_cache(bsz, self.total_tokens)

    @torch.compile()
    def forward_model(self, x, start_pos, end_pos):
        x = self.emb_norm(x)
        for layer in self.layers:
            x = layer.forward_onestep(
                x, self.freqs_cis[start_pos:end_pos,], start_pos, end_pos
            )
        x = self.norm(x)
        return x
    
    def head_sample(self, x, diff_pos, sample_steps, cfg_scale, cfg_schedule="linear"):
        x = x + self.pos_for_diff.weight[diff_pos : diff_pos + 1, :]
        x = x.view(-1, x.shape[-1])
        seq_len = self.h * self.w
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
        pred = pred.view(-1, 1, pred.shape[-1])
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
        for i in range(self.h * self.w):
            if i == 0:
                x = self.forward_model(c, 0, self.cls_token_num)
            else:
                x = self.proj_in(last_pred)
                x = self.forward_model(
                    x, i + self.cls_token_num - 1, i + self.cls_token_num
                )
            last_pred = self.head_sample(
                x[:, -1:, :],
                i,
                sample_steps,
                cfg_scale,
                cfg_schedule,
            )
            all_preds.append(last_pred)

        x = torch.cat(all_preds, dim=-2)[:act_bsz]
        if x.dim() == 3: #b n c -> b c h w
            x = self.unpatchify(x)
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
