import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sampling_parallel import euler_maruyama


def timestep_embedding(t, dim, max_period=10000, time_factor: float = 1000.0):
    half = dim // 2
    t = time_factor * t.float()
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
        / half
    )

    args = t[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding

def time_shift_sana(t: torch.Tensor, flow_shift: float = 1., sigma: float = 1.):
    return (1 / flow_shift) / ( (1 / flow_shift) + (1 / t - 1) ** sigma)

class DiffHead(nn.Module):
    """Diffusion Loss"""

    def __init__(
        self,
        ch_target,
        ch_cond,
        ch_latent,
        depth_latent,
        depth_adanln,
        grad_checkpointing=False,
        time_shift=1.,
        time_schedule='logit_normal',
        parallel_num=4,
        P_std: float = 1.,
        P_mean: float = 0.,
    ):
        super(DiffHead, self).__init__()
        self.ch_target = ch_target
        self.time_shift = time_shift
        self.time_schedule = time_schedule
        self.P_std = P_std
        self.P_mean = P_mean

        self.net = TransEncoder(
            in_channels=ch_target,
            model_channels=ch_latent,
            z_channels=ch_cond,
            num_res_blocks=depth_latent,
            num_ada_ln_blocks=depth_adanln,
            grad_checkpointing=grad_checkpointing,
            parallel_num=parallel_num,
        )

    def forward(self, x, cond):
        with torch.autocast(device_type="cuda", enabled=False):
            with torch.no_grad():
                if self.time_schedule == 'logit_normal':
                    t = (torch.randn((x.shape[0]), device=x.device) * self.P_std + self.P_mean).sigmoid()
                    if self.time_shift != 1.:
                        t = time_shift_sana(t, self.time_shift)
                elif self.time_schedule == 'uniform':
                    t = torch.rand((x.shape[0]), device=x.device)
                    if self.time_shift != 1.:
                        t = time_shift_sana(t, self.time_shift)
                else:
                    raise NotImplementedError(f"unknown time_schedule {self.time_schedule}")
                e = torch.randn_like(x)
                ti = t.view(-1, 1, 1)
                z = (1.0 - ti) * e + ti * x
                v = (x - z) / (1 - ti).clamp_min(0.05)

        x_pred = self.net(z, t, cond)
        v_pred = (x_pred - z) / (1 - ti).clamp_min(0.05)

        with torch.autocast(device_type="cuda", enabled=False):
            v_pred = v_pred.float()
            loss = torch.mean((v - v_pred) ** 2)
        return loss

    def sample(
        self,
        z,
        cfg,
        num_sampling_steps,
    ):
        return euler_maruyama(
            self.ch_target,
            self.net.forward,
            z,
            cfg,
            num_sampling_steps=num_sampling_steps,
            time_shift = self.time_shift,
        )

    def initialize_weights(self):
        self.net.initialize_weights()


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        t_freq = timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm = nn.LayerNorm(channels, eps=1e-6, elementwise_affine=True)
        hidden_dim = int(channels * 1.5)
        self.w1 = nn.Linear(channels, hidden_dim * 2, bias=True)
        self.w2 = nn.Linear(hidden_dim, channels, bias=True)

    def forward(self, x, scale, shift, gate):
        h = self.norm(x) * (1 + scale) + shift
        h1, h2 = self.w1(h).chunk(2, dim=-1)
        h = self.w2(F.silu(h1) * h2)
        return x + h * gate


class FinalLayer(nn.Module):
    def __init__(self, channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(channels, eps=1e-6, elementwise_affine=False)
        self.ada_ln_modulation = nn.Linear(channels, channels * 2, bias=True)
        self.linear = nn.Linear(channels, out_channels, bias=True)

    def forward(self, x, y):
        scale, shift = self.ada_ln_modulation(y).chunk(2, dim=-1)
        x = self.norm_final(x) * (1.0 + scale) + shift
        x = self.linear(x)
        return x

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        n_head,
    ):
        super().__init__()
        assert dim % n_head == 0
        self.dim = dim
        self.head_dim = dim // n_head
        self.scale = self.head_dim**-0.5
        self.n_head = n_head
        total_kv_dim = (self.n_head * 3) * self.head_dim

        self.wqkv = nn.Linear(dim, total_kv_dim, bias=True)
        self.wo = nn.Linear(dim, dim, bias=True)

    def forward(
        self,
        x: torch.Tensor,
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wqkv(x).chunk(3, dim=-1)

        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_head, self.head_dim)


        xq, xk, xv = map(lambda x: x.transpose(1, 2), (xq, xk, xv))
        xq = xq * self.scale
        attn = xq @ xk.transpose(-1, -2)
        attn = F.softmax(attn, dim=-1)
        output = (attn @ xv).transpose(1, 2).contiguous()

        # output = flash_attn_func(
        #         xq,
        #         xk,
        #         xv,
        #         causal=False,
        #     )

        output = output.view(bsz, seqlen, self.dim)

        output = self.wo(output)
        return output

class TransBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm1 = nn.LayerNorm(channels, eps=1e-6, elementwise_affine=True)
        self.attn = Attention(channels, n_head=channels//64)

        self.norm2 = nn.LayerNorm(channels, eps=1e-6, elementwise_affine=True)
        hidden_dim = int(channels * 1.5)
        self.w1 = nn.Linear(channels, hidden_dim * 2, bias=True)
        self.w2 = nn.Linear(hidden_dim, channels, bias=True)

    def forward(self, x, scale1, shift1, gate1, scale2, shift2, gate2):
        h = self.norm1(x) * (1 + scale1) + shift1
        h = self.attn(h)
        x = x + h * gate1
        h = self.norm2(x) * (1 + scale2) + shift2
        h1, h2 = self.w1(h).chunk(2, dim=-1)
        h = self.w2(F.silu(h1) * h2)
        return x + h * gate2

class TransEncoder(nn.Module):

    def __init__(
        self,
        in_channels,
        model_channels,
        z_channels,
        num_res_blocks,
        num_ada_ln_blocks=2,
        grad_checkpointing=False,
        parallel_num=4,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = in_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing
        self.parallel_num = parallel_num

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)

        self.input_proj = nn.Linear(in_channels, model_channels)
        self.res_blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            self.res_blocks.append(
                TransBlock(
                    model_channels,
                )
            )
        # share adaLN for consecutive blocks, to save computation and parameters
        self.ada_ln_blocks = nn.ModuleList()
        for i in range(num_ada_ln_blocks):
            self.ada_ln_blocks.append(
                nn.Linear(model_channels, model_channels * 6, bias=True)
            )
        self.ada_ln_switch_freq = max(1, num_res_blocks // num_ada_ln_blocks)
        assert (
            num_res_blocks % self.ada_ln_switch_freq
        ) == 0, "num_res_blocks must be divisible by num_ada_ln_blocks"
        self.final_layer = FinalLayer(model_channels, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        for block in self.ada_ln_blocks:
            nn.init.constant_(block.weight, 0)
            nn.init.constant_(block.bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.ada_ln_modulation.weight, 0)
        nn.init.constant_(self.final_layer.ada_ln_modulation.bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    @torch.compile()
    def forward(self, x, t, c):
        x = self.input_proj(x)
        t = self.time_embed(t).unsqueeze(1)
        c = self.cond_embed(c)

        y = F.silu(t+c)
        scale1, shift1, gate1, scale2, shift2, gate2 = self.ada_ln_blocks[0](y).chunk(6, dim=-1)

        for i, block in enumerate(self.res_blocks):
            if i > 0 and i % self.ada_ln_switch_freq == 0:
                ada_ln_block = self.ada_ln_blocks[i // self.ada_ln_switch_freq]
                scale1, shift1, gate1, scale2, shift2, gate2 = ada_ln_block(y).chunk(6, dim=-1)
            x = block(x, scale1, shift1, gate1, scale2, shift2, gate2)

        return self.final_layer(x, y)