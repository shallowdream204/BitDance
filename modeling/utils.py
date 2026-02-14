import torch
import torch.nn.functional as F
from torch import nn

from torch.nn.attention.flex_attention import or_masks, and_masks

from transformers.activations import ACT2FN

class MLPconnector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_act: str):
        super().__init__()
        self.activation_fn = ACT2FN[hidden_act]
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

def create_sparse_mask(document_lens, split_lens, attn_modes, parallel_num, device):
    parallel_causal_num = 2
    parallel_block_causal_num = parallel_num

    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    def parallel_block_mask(b, h, q_idx, kv_idx):
        same_seg = segment_ids[q_idx] == segment_ids[kv_idx]
        is_par = is_parallel[q_idx] 
        
        lq = local_ids[q_idx]
        lk = local_ids[kv_idx]
        
        in_block_region = (lq >= parallel_causal_num) & (lk >= parallel_causal_num)
        
        same_block = ((lq - parallel_causal_num) // parallel_block_causal_num) == ((lk - parallel_causal_num) // parallel_block_causal_num)
        
        return same_seg & is_par & in_block_region & same_block

    def sample_mask(b, h, q_idx, kv_idx):
        return document_id[q_idx] == document_id[kv_idx]

    segment_ids_list = []
    local_ids_list = []
    is_parallel_list = []
    
    current_seg_id = 0
    for length, mode in zip(split_lens, attn_modes):
        segment_ids_list.extend([current_seg_id] * length)
        local_ids_list.extend(list(range(length)))
        is_parallel_list.extend([True if mode == 'parallel' else False] * length)
        current_seg_id += 1

    segment_ids = torch.tensor(segment_ids_list, device=device, dtype=torch.long)
    local_ids = torch.tensor(local_ids_list, device=device, dtype=torch.long)
    is_parallel = torch.tensor(is_parallel_list, device=device, dtype=torch.bool)

    document_id = torch.cat([torch.full((l,), i, device=device) for i, l in enumerate(document_lens, start=1)])

    return and_masks(or_masks(causal_mask, parallel_block_mask), sample_mask)

def top_k_top_p_filtering(
    logits,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
):
    """Filter a distribution of logits using top-k and/or top-p (nucleus) filtering."""
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value

    return logits


def sample_codebook(
    pred_logits,
    cur_item_type,
    codebook,
    do_sample: bool = True,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
):
    """
    pred_logits: (B, vocab_size)
    cur_item_type: 'text' or 'vision'
    """
    # 1. Apply temperature
    logits = pred_logits / max(temperature, 1e-5)

    # 2. Apply top-k / top-p filtering
    if top_k > 0 or top_p < 1.0:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

    # 3. Get probabilities
    probs = F.softmax(logits, dim=-1)

    # 4. Sample or take argmax
    if do_sample:
        curr_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
    else:
        curr_tokens = torch.argmax(probs, dim=-1)

    curr_embeds = codebook(curr_tokens)

    return curr_tokens, curr_embeds


def flip_tensor_elements_uniform_prob(tensor: torch.Tensor, p_max: float) -> torch.Tensor:
    if not 0.0 <= p_max <= 1.0:
        raise ValueError(f"p_max must in [0.0, 1.0]")

    r1 = torch.rand_like(tensor)
    r2 = torch.rand_like(tensor)

    flip_mask = r1 < p_max * r2

    multiplier = torch.where(flip_mask, -1.0, 1.0)
    multiplier = multiplier.to(tensor.dtype)

    flipped_tensor = tensor * multiplier
    return flipped_tensor

def gaussian_sample(raw_output):
    mu, log_var = raw_output.chunk(2, dim=-1)
    sigma = torch.exp(0.5 * log_var)
    sample = mu + torch.randn_like(mu) * sigma

    return sample

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0, pe_interpolation=1.0):
    """
    grid_size: int or tuple/list of (h, w)
    return:
    pos_embed: [grid_h*grid_w, embed_dim] or [extra_tokens+grid_h*grid_w, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, int):
        grid_h_size, grid_w_size = grid_size, grid_size
    else:
        grid_h_size, grid_w_size = grid_size

    grid_h = torch.arange(grid_h_size, dtype=torch.float32) / pe_interpolation
    grid_w = torch.arange(grid_w_size, dtype=torch.float32) / pe_interpolation
    
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='xy')
    
    grid = torch.stack([grid_w, grid_h], dim=0) # shape: (2, grid_h_size, grid_w_size)

    grid = grid.reshape([2, 1, grid_h_size, grid_w_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

    if cls_token and extra_tokens > 0:
        pos_embed = torch.cat([torch.zeros([extra_tokens, embed_dim]), pos_embed], dim=0)
    
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = torch.cat([emb_h, emb_w], dim=1)  # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb

def remove_first_user_block(x: str) -> str:
    start_marker = "<|im_start|>user\n"
    end_marker = "<|im_end|>\n"
    start_index = x.find(start_marker)
    if start_index == -1:
        return x
    end_index = x.find(end_marker, start_index + len(start_marker))
    if end_index == -1:
        return x
    result = x[:start_index] + x[end_index + len(end_marker):]
    return result