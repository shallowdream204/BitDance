from functools import lru_cache
from typing import Optional

import torch
import torch.nn as nn

from flash_attn import flash_attn_func

from torch.nn import RMSNorm
from torch.nn import functional as F


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


def get_2d_pos(resolution, patch_size, num_scales=1):
    max_pos = resolution // patch_size
    coords_list = []

    for i in range(num_scales):
        scale = 2 ** (num_scales - i - 1)
        P = max(resolution // scale // patch_size, 1)
        edge = float(max_pos) / P
        centers = (torch.arange(P, dtype=torch.float32) + 0.5) * edge
        grid_y, grid_x = torch.meshgrid(centers, centers, indexing="ij")
        coords = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)
        coords_list.append(coords)

    return torch.cat(coords_list, dim=0)


def precompute_freqs_cis_2d(
    pos_2d, n_elem: int, base: float = 10000, cls_token_num=120
):
    # split the dimension into half, one for x and one for y
    half_dim = n_elem // 2
    freqs = 1.0 / (
        base ** (torch.arange(0, half_dim, 2)[: (half_dim // 2)].float() / half_dim)
    )
    t = pos_2d + 1.0
    if cls_token_num > 0:
        t = torch.cat(
            [torch.zeros((cls_token_num, 2), device=freqs.device), t],
            dim=0,
        )
    freqs = torch.outer(t.flatten(), freqs).view(*t.shape[:-1], -1)
    return torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)


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
