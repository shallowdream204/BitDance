import math
from functools import lru_cache
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from flash_attn import flash_attn_func

from torch.nn import RMSNorm
from utils.fs import download
from collections import defaultdict


def swish(x):
    return x*torch.sigmoid(x)

class ResBlock(nn.Module):
    def __init__(self, 
                 in_filters,
                 out_filters,
                 use_conv_shortcut = False,
                 use_agn = False,
                 ) -> None:
        super().__init__()

        self.in_filters = in_filters
        self.out_filters = out_filters
        self.use_conv_shortcut = use_conv_shortcut
        self.use_agn = use_agn

        if not use_agn: ## agn is GroupNorm likewise skip it if has agn before
            self.norm1 = nn.GroupNorm(32, in_filters, eps=1e-6)
        self.norm2 = nn.GroupNorm(32, out_filters, eps=1e-6)

        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(3, 3), padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_filters, out_filters, kernel_size=(3, 3), padding=1, bias=False)

        if in_filters != out_filters:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_filters, out_filters, kernel_size=(3, 3), padding=1, bias=False)
            else:
                self.nin_shortcut = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), padding=0, bias=False)
    

    def forward(self, x, **kwargs):
        residual = x

        if not self.use_agn:
            x = self.norm1(x)
        x = swish(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = swish(x)
        x = self.conv2(x)
        if self.in_filters != self.out_filters:
            if self.use_conv_shortcut:
                residual = self.conv_shortcut(residual)
            else:
                residual = self.nin_shortcut(residual)

        return x + residual

def depth_to_space(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """ Depth-to-Space DCR mode (depth-column-row) core implementation.

        Args:
            x (torch.Tensor): input tensor. The channels-first (*CHW) layout is supported.
            block_size (int): block side size
    """
    # check inputs
    if x.dim() < 3:
        raise ValueError(
            f"Expecting a channels-first (*CHW) tensor of at least 3 dimensions"
        )
    c, h, w = x.shape[-3:]

    s = block_size**2
    if c % s != 0:
        raise ValueError(
            f"Expecting a channels-first (*CHW) tensor with C divisible by {s}, but got C={c} channels"
        )

    outer_dims = x.shape[:-3]

    # splitting two additional dimensions from the channel dimension
    x = x.view(-1, block_size, block_size, c // s, h, w)

    # putting the two new dimensions along H and W
    x = x.permute(0, 3, 4, 1, 5, 2)

    # merging the two new dimensions with H and W
    x = x.contiguous().view(*outer_dims, c // s, h * block_size,
                            w * block_size)

    return x

class Upsampler(nn.Module):
    def __init__(
        self,
        dim,
        dim_out = None
    ):
        super().__init__()
        dim_out = dim * 4
        self.conv1 = nn.Conv2d(dim, dim_out, (3, 3), padding=1)
        self.depth2space = depth_to_space

    def forward(self, x):
        """
        input_image: [B C H W]
        """
        out = self.conv1(x)
        out = self.depth2space(out, block_size=2)
        return out

class AdaptiveGroupNorm(nn.Module):
    def __init__(self, z_channel, in_filters, num_groups=32, eps=1e-6):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups=32, num_channels=in_filters, eps=eps, affine=False)
        # self.lin = nn.Linear(z_channels, in_filters * 2)
        self.gamma = nn.Linear(z_channel, in_filters)
        self.beta = nn.Linear(z_channel, in_filters)
        self.eps = eps
    
    def forward(self, x, quantizer):
        B, C, _, _ = x.shape
        # quantizer = F.adaptive_avg_pool2d(quantizer, (1, 1))
        ### calcuate var for scale
        scale = rearrange(quantizer, "b c h w -> b c (h w)")
        scale = scale.var(dim=-1) + self.eps #not unbias
        scale = scale.sqrt()
        scale = self.gamma(scale).view(B, C, 1, 1)

        ### calculate mean for bias
        bias = rearrange(quantizer, "b c h w -> b c (h w)")
        bias = bias.mean(dim=-1)
        bias = self.beta(bias).view(B, C, 1, 1)
       
        x = self.gn(x)
        x = scale * x + bias

        return x

def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(torch.nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"

def find_multiple(n: int, k: int):
    if n % k == 0:
        return n
    return n + k - (n % k)

class FeedForward(nn.Module):

    def __init__(self, dim, dropout_p=0.1, mlp_ratio=4.0):
        super().__init__()
        hidden_dim = mlp_ratio * dim
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = find_multiple(hidden_dim, 256)

        self.w1 = nn.Linear(dim, hidden_dim * 2, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.ffn_dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        h1, h2 = self.w1(x).chunk(2, dim=-1)
        return self.ffn_dropout(self.w2(F.silu(h1) * h2))

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    # x: (bs, seq_len, n_head, head_dim)
    # freqs_cis (seq_len, head_dim // 2, 2)
    xshaped = x.float().reshape(
        *x.shape[:-1], -1, 2
    )  # (bs, seq_len, n_head, head_dim//2, 2)
    freqs_cis = freqs_cis.view(
        1, xshaped.size(1), 1, xshaped.size(3), 2
    )  # (1, seq_len, 1, head_dim//2, 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        dim=-1,
    )
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)

@lru_cache(maxsize=16)
def get_causal_mask(seq_q, seq_k, device):
    offset = seq_k - seq_q
    i = torch.arange(seq_q, device=device).unsqueeze(1)
    j = torch.arange(seq_k, device=device).unsqueeze(0)
    causal_mask = (j > (offset + i)).bool()
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
    return causal_mask

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        n_head,
        attn_dropout_p,
        resid_dropout_p,
        causal: bool = True,
    ):
        super().__init__()
        assert dim % n_head == 0
        self.dim = dim
        self.head_dim = dim // n_head
        self.scale = self.head_dim**-0.5
        self.n_head = n_head
        total_kv_dim = (self.n_head * 3) * self.head_dim

        self.wqkv = nn.Linear(dim, total_kv_dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout = nn.Dropout(resid_dropout_p)
        self.causal = causal

        self.k_cache = None
        self.v_cache = None
        self.kv_cache_size = None

    def enable_kv_cache(self, bsz, max_seq_len):
        if self.kv_cache_size != (bsz, max_seq_len):
            device = self.wo.weight.device
            dtype = self.wo.weight.dtype
            self.k_cache = torch.zeros(
                (bsz, self.n_head, max_seq_len, self.head_dim),
                device=device,
                dtype=dtype,
            )
            self.v_cache = torch.zeros(
                (bsz, self.n_head, max_seq_len, self.head_dim),
                device=device,
                dtype=dtype,
            )
            self.kv_cache_size = (bsz, max_seq_len)

    def update_kv_cache(
        self, start_pos, end_pos, keys: torch.Tensor, values: torch.Tensor
    ):
        self.k_cache[:, :, start_pos:end_pos, :] = keys
        self.v_cache[:, :, start_pos:end_pos, :] = values
        return (
            self.k_cache[:, :, :end_pos, :],
            self.v_cache[:, :, :end_pos, :],
        )

    def naive_attention(self, xq, keys, values, is_causal):
        xq = xq * self.scale
        # q: [B, H, 1, D], k: [B, H, D, L] -> attn [B, H, 1, L]
        attn = xq @ keys.transpose(-1, -2)
        seq_q, seq_k = attn.shape[-2], attn.shape[-1]
        if is_causal and seq_q > 1:
            causal_mask = get_causal_mask(seq_q, seq_k, attn.device)
            attn.masked_fill_(causal_mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        if self.attn_dropout_p > 0 and self.training:
            attn = F.dropout(attn, p=self.attn_dropout_p, training=self.training)
        # [B, H, 1, L] @ [B, H, L, D] -> [B, H, 1, D]
        return attn @ values

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor = None,
        start_pos: Optional[int] = None,
        end_pos: Optional[int] = None,
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wqkv(x).chunk(3, dim=-1)

        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_head, self.head_dim)

        if freqs_cis is not None:
            xq = apply_rotary_emb(xq, freqs_cis)
            xk = apply_rotary_emb(xk, freqs_cis)

        is_causal = self.causal
        if self.k_cache is not None and start_pos is not None:
            xq, xk, xv = map(lambda x: x.transpose(1, 2), (xq, xk, xv))
            keys, values = self.update_kv_cache(start_pos, end_pos, xk, xv)
            output = self.naive_attention(xq, keys, values, is_causal=is_causal)
            output = output.transpose(1, 2).contiguous()
        else:
            output = flash_attn_func(
                xq,
                xk,
                xv,
                causal=is_causal,
                dropout_p=self.attn_dropout_p if self.training else 0,
            )

        output = output.view(bsz, seqlen, self.dim)

        output = self.resid_dropout(self.wo(output))
        return output

class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        n_head,
        attn_dropout_p: float = 0.0,
        resid_dropout_p: float = 0.0,
        drop_path: float = 0.0,
        causal: bool = True,
    ):
        super().__init__()
        self.attention = Attention(
            dim=dim,
            n_head=n_head,
            attn_dropout_p=attn_dropout_p,
            resid_dropout_p=resid_dropout_p,
            causal=causal,
        )
        self.feed_forward = FeedForward(
            dim=dim,
            dropout_p=resid_dropout_p,
        )
        self.attention_norm = RMSNorm(dim, eps=1e-6)
        self.ffn_norm = RMSNorm(dim, eps=1e-6)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        h = x + self.drop_path(self.attention(self.attention_norm(x), freqs_cis))
        out = h + self.drop_path(self.feed_forward(self.ffn_norm(h)))
        return out

    def forward_onestep(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        start_pos: int,
        end_pos: int,
    ):
        h = x + self.drop_path(
            self.attention(self.attention_norm(x), freqs_cis, start_pos, end_pos)
        )
        out = h + self.drop_path(self.feed_forward(self.ffn_norm(h)))
        return out

def precompute_freqs_cis_2d(pos_2d, n_elem: int, base: float = 10000.0, cls_token_num: int = 0):
    """
    - pos_2d: [S, 2]，S 为空间 token 数 = H*W
    - n_elem: 每个 head 的维度 (d_model // n_heads)
    返回张量形状：[S + cls_token_num, 2, n_elem//4, 2]（cos/sin）
    解释：
    half_dim = n_elem // 2
    freqs 长度 = half_dim // 2
    t.flatten() -> [2S]（x,y 拼在一起）
    outer -> [2S, half_dim//2]，再 reshape 成 [S(+cls), 2, half_dim//2]
    最后 stack cos/sin -> [..., 2]
    """
    # 注意：完全对齐你给的示例实现
    half_dim = n_elem // 2
    # 这里严格遵循示例实现的步长为 2 的采样
    freqs = 1.0 / (
        base ** (torch.arange(0, half_dim, 2, device=pos_2d.device, dtype=pos_2d.dtype)[: (half_dim // 2)] / half_dim)
    )

    t = pos_2d + 1.0
    if cls_token_num > 0:
        t = torch.cat(
            [torch.zeros((cls_token_num, 2), device=pos_2d.device, dtype=pos_2d.dtype), t], dim=0,
        )
    freqs = torch.outer(t.flatten(), freqs).view(*t.shape[:-1], -1)  # [S+cls, 2, half_dim//2]
    return torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)  # [S+cls, 2, half_dim//2, 2]

class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, in_channels, num_res_blocks, z_channels, ch_mult=(1, 2, 2, 4), 
                resolution, double_z=False, attn_blocks=2, n_heads=16,
                ):
        super().__init__()

        self.in_channels = in_channels
        self.z_channels = z_channels
        self.resolution = resolution

        self.num_res_blocks = num_res_blocks
        self.num_blocks = len(ch_mult)
        
        self.conv_in = nn.Conv2d(in_channels,
                                 ch,
                                 kernel_size=(3, 3),
                                 padding=1,
                                 bias=False,
        )

        ## construct the model
        self.down = nn.ModuleList()

        for i_level in range(self.num_blocks):
            block = nn.ModuleList()
            block_in = ch * ch_mult[i_level]  # [1, 2, 2, 4]
            for _ in range(self.num_res_blocks):
                block.append(ResBlock(block_in, block_in))
            
            down = nn.Module()
            down.block = block
            if i_level < self.num_blocks - 1:
                block_out = ch * ch_mult[i_level + 1]  # [1, 2, 2, 4]
                down.downsample = nn.Conv2d(block_in, block_out, kernel_size=(3, 3), stride=(2, 2), padding=1)

            self.down.append(down)

        self.n_heads = n_heads
        self.register_num_tokens = 4
        self.register_token = nn.Embedding(self.register_num_tokens, block_in)
        self.mid_attn_blocks = nn.ModuleList()
        for _ in range(attn_blocks):
            self.mid_attn_blocks.append(
                TransformerBlock(
                    block_in,
                    self.n_heads,
                    0.0,
                    0.0,
                    drop_path=0.0,
                    causal=False,
                )
            )
        
        ### end
        self.norm_out = nn.GroupNorm(32, block_out, eps=1e-6)
        self.conv_out = nn.Conv2d(block_out, z_channels, kernel_size=(1, 1))
            
    def forward(self, x):
        """
        前向流程：
        1) 下采样 + ResBlocks
        2) Mid ResBlocks
        3) (可选) Mid Transformer + 2D RoPE
        4) 归一化 + swish + 输出卷积
        5) 映射到 [-1, 1]
        """
        # --------- 下采样路径 ----------
        x = self.conv_in(x)
        for i_level in range(self.num_blocks):
            for i_block in range(self.num_res_blocks):
                x = self.down[i_level].block[i_block](x)

            if i_level < self.num_blocks - 1:
                x = self.down[i_level].downsample(x)

        # --------- Mid Transformer (带 2D RoPE) ----------
        if len(self.mid_attn_blocks) > 0:
            # 当前特征图尺寸
            B, C, H, W = x.shape
            device = x.device
            dtype = x.dtype

            # 维度检查：确保每个 head 的维度是偶数
            assert hasattr(self, "n_heads"), "Please set self.n_heads = n_heads in __init__."
            assert C % self.n_heads == 0, f"Channel dim {C} must be divisible by n_heads={self.n_heads}"
            head_dim = C // self.n_heads
            assert head_dim % 2 == 0, f"Head dim {head_dim} must be even for 2D RoPE."

            # 展平成 token，顺序为行主序（W 变化最快），与下面 meshgrid 生成一致
            x_tokens = x.flatten(2).transpose(1, 2).contiguous()  # [B, H*W, C]

            # 拼接 register tokens
            reg_tokens = self.register_token.weight.view(1, self.register_num_tokens, C).expand(B, -1, -1)
            x_tokens = torch.cat([reg_tokens, x_tokens], dim=1)  # [B, reg + H*W, C]

            # 根据 H×W 生成 2D 位置坐标（中心坐标 0.5, 1.5, ..., size-0.5）
            y_centers = torch.arange(H, device=device, dtype=dtype)
            y_centers *= 7.0 / y_centers[-1]
            y_centers += 0.5
            x_centers = torch.arange(W, device=device, dtype=dtype)
            x_centers *= 7.0 / x_centers[-1]
            x_centers += 0.5
            grid_y, grid_x = torch.meshgrid(y_centers, x_centers, indexing="ij")  # [H, W]
            pos_2d = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)  # [H*W, 2]

            # 预计算 2D RoPE 频率（包含 register tokens 的占位）
            freqs_cis = precompute_freqs_cis_2d(
                pos_2d,
                head_dim,
                base=10000.0,  # 与示例一致
                cls_token_num=self.register_num_tokens,
            ).to(device=device, dtype=dtype)

            # 逐层 Transformer
            for block in self.mid_attn_blocks:
                x_tokens = block(x_tokens, freqs_cis)

            # 去掉 register tokens
            x_tokens = x_tokens[:, self.register_num_tokens:, :]  # [B, H*W, C]

            # 还原回 NCHW
            x = x_tokens.transpose(1, 2).contiguous().view(B, C, H, W)

        # --------- 输出头 ----------
        x = self.norm_out(x)
        # x = swish(x)
        x = self.conv_out(x)
        return 2 * torch.sigmoid(x) - 1

class GANDecoder(nn.Module):
    def __init__(self, *, ch, out_ch, in_channels, num_res_blocks, z_channels, ch_mult=(1, 2, 2, 4), 
                resolution, double_z=False, attn_blocks=2, n_heads=16,) -> None:
        super().__init__()

        self.ch = ch
        self.num_blocks = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        block_in = ch*ch_mult[self.num_blocks-1]

        self.conv_in = nn.Conv2d(
            z_channels * 2, block_in, kernel_size=(3, 3), padding=1, bias=True
        )

        # 与 Encoder 一致的 Transformer 设置
        self.n_heads = n_heads
        self.register_num_tokens = 4
        self.register_token = nn.Embedding(self.register_num_tokens, block_in)
        self.mid_attn_blocks = nn.ModuleList()
        for _ in range(attn_blocks):
            self.mid_attn_blocks.append(
                TransformerBlock(
                    block_in,          # d_model
                    self.n_heads,      # n_heads
                    0.0,               # attn dropout
                    0.0,               # ff dropout
                    drop_path=0.0,
                    causal=False,
                )
            )

        self.mid_block = nn.ModuleList()
        for res_idx in range(self.num_res_blocks):
            self.mid_block.append(ResBlock(block_in, block_in))
        
        self.up = nn.ModuleList()

        self.adaptive = nn.ModuleList()

        for i_level in reversed(range(self.num_blocks)):
            block = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            self.adaptive.insert(0, AdaptiveGroupNorm(z_channels, block_in))
            for i_block in range(self.num_res_blocks):
                # if i_block == 0:
                #     block.append(ResBlock(block_in, block_out, use_agn=True))
                # else:
                block.append(ResBlock(block_in, block_out))
                block_in = block_out
            
            up = nn.Module()
            up.block = block
            if i_level > 0:
                up.upsample = Upsampler(block_in)
            self.up.insert(0, up)
        
        self.norm_out = nn.GroupNorm(32, block_in, eps=1e-6)

        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=(3, 3), padding=1)
    
    def forward(self, z):
        
        style = z.clone() #for adaptive groupnorm

        noise = torch.randn_like(z).to(z.device) #generate noise
        z = torch.cat([z, noise], dim=1) #concat noise to the style vector
        z = self.conv_in(z)

        # Mid Transformer（与 Encoder 一致的 2D RoPE）
        if len(self.mid_attn_blocks) > 0:
            B, C, H, W = z.shape
            device = z.device
            dtype = z.dtype
            # 维度校验：每个 head 的维度须为偶数
            assert hasattr(self, "n_heads"), "Please set self.n_heads = n_heads in __init__."
            assert C % self.n_heads == 0, f"Channel dim {C} must be divisible by n_heads={self.n_heads}"
            head_dim = C // self.n_heads
            assert head_dim % 2 == 0, f"Head dim {head_dim} must be even for 2D RoPE."
            # NCHW -> NLC（行主序展平，与位置编码网格一致）
            tokens = z.flatten(2).transpose(1, 2).contiguous()  # [B, H*W, C]
            # 拼接 register tokens
            reg_tokens = self.register_token.weight.view(1, self.register_num_tokens, C).expand(B, -1, -1)
            tokens = torch.cat([reg_tokens, tokens], dim=1)  # [B, reg + H*W, C]
            # 构建 2D 网格坐标（中心坐标），并做与 Encoder 相同的尺度归一（7.5）
            y_centers = torch.arange(H, device=device, dtype=dtype) + 0.5
            y_centers *= 7.5 / y_centers[-1]
            x_centers = torch.arange(W, device=device, dtype=dtype) + 0.5
            x_centers *= 7.5 / x_centers[-1]
            grid_y, grid_x = torch.meshgrid(y_centers, x_centers, indexing="ij")  # [H, W]
            pos_2d = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)  # [H*W, 2]
            # 预计算 2D RoPE 频率表（包含 register tokens 的占位）
            freqs_cis = precompute_freqs_cis_2d(
                pos_2d,
                head_dim,
                base=10000.0,
                cls_token_num=self.register_num_tokens,
            ).to(device=device, dtype=dtype)
            # 逐层 Transformer
            for block in self.mid_attn_blocks:
                tokens = block(tokens, freqs_cis)
            # 去掉 register tokens 并还原回 NCHW
            tokens = tokens[:, self.register_num_tokens:, :]  # [B, H*W, C]
            z = tokens.transpose(1, 2).contiguous().view(B, C, H, W)

        ## mid
        for res in range(self.num_res_blocks):
            z = self.mid_block[res](z)
        
        ## upsample
        for i_level in reversed(range(self.num_blocks)):
            ### pass in each resblock first adaGN
            z = self.adaptive[i_level](z, style)
            for i_block in range(self.num_res_blocks):
                z = self.up[i_level].block[i_block](z)
            
            if i_level > 0:
                z = self.up[i_level].upsample(z)
        
        z = self.norm_out(z)
        z = swish(z)
        z = self.conv_out(z)

        return z

class VQModel(nn.Module):
    def __init__(self,
                ddconfig,
                checkpoint="hdfs://harunafr/home/byte_ttdata_fr_ssd/content_understanding/hao.chen/checkpoints/we_tok/ImageNet/down16_wetok.ckpt",
                gan_decoder = True,
                ):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = GANDecoder(**ddconfig)

        # Load weights from the checkpoint
        self.load_from_ckpt(checkpoint)

    def load_from_ckpt(self, checkpoint):
        state = torch.load(download(checkpoint), mmap=True, map_location="cpu")["state_dict"]
        for k, v in state.items(): 
            if not "model_ema" in k:
                ema_k = "model_ema." + k.replace('.', '')
                try:
                    state[k] = state[ema_k]
                except:
                    pass
        log_info = self.load_state_dict(state, strict=False)
        has_missing_keys = bool(log_info.missing_keys)
        has_unexpected_keys = bool(log_info.unexpected_keys)
        del state
        if not has_missing_keys:
            print(f"Successfully loaded all weights from checkpoint: {checkpoint}")
        else:
            if has_missing_keys:
                print("Missing keys (model layers not in checkpoint):")
                for key in log_info.missing_keys:
                    print(f"  - {key}")
        if False and has_unexpected_keys:
            print("\nUnexpected keys (checkpoint layers not in model):")
            for key in log_info.unexpected_keys:
                print(f"  - {key}")

    def encode(self, x):
        h = self.encoder(x)
        codebook_value = torch.Tensor([1.0]).to(h)
        quant_h = torch.where(h > 0, codebook_value, -codebook_value)      # higher than 0 filled 

        return quant_h
    
    # def vt_forward(self, image_list):
    #     q_list = []
    #     for x in image_list:
    #         quant = self.encode(x)
    #         quant = rearrange(quant.squeeze(0), "c h w -> (h w) c")
    #         q_list.append(quant)

    #     return torch.cat(q_list, dim=0)   
    

    def vt_forward(self, image_list, max_bs=32):
        groups = defaultdict(list)  # {(H, W): [(idx, image_tensor), ...]}
        for i, img in enumerate(image_list):
            _, _, H, W = img.shape
            groups[(H, W)].append((i, img))

        output = [None] * len(image_list)

        for (H, W), items in groups.items():
            if H >= 1024 or W >= 1024:
                max_bs = math.ceil(max_bs / 8.0)
            elif H >= 768 or W >= 768:
                max_bs = math.ceil(max_bs / 4.0)
            elif H >= 512 or W >= 512:
                max_bs = math.ceil(max_bs / 2.0)
            for start in range(0, len(items), max_bs):
                chunk = items[start:start + max_bs]
                idxs = [x[0] for x in chunk]
                imgs = [x[1] for x in chunk]

                batch = torch.cat(imgs, dim=0)  # [B, 3, H, W]

                quant = self.encode(batch)                      # [B, C, h, w]

                for b in range(quant.size(0)):
                    q = rearrange(quant[b], "c h w -> (h w) c")
                    output[idxs[b]] = q

        return torch.cat(output, dim=0)


    def decode(self, quant):
        dec = self.decoder(quant)
        return dec

    def forward(self, input):
        quant = self.encode(input)
        dec = self.decode(quant)
        return dec, quant