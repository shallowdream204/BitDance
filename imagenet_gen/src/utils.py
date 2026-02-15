import logging

import torch
import torch.distributed as dist
from torch.nn import functional as F


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/log.txt"),
            ],
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_ps = []
    ps = []

    for e, m in zip(ema_model.parameters(), model.parameters()):
        if m.requires_grad:
            ema_ps.append(e)
            ps.append(m)
    torch._foreach_lerp_(ema_ps, ps, 1.0 - decay)


@torch.no_grad()
def sync_frozen_params_once(ema_model, model):
    for e, m in zip(ema_model.parameters(), model.parameters()):
        if not m.requires_grad:
            e.copy_(m)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def patchify_raster(x, p):
    B, C, H, W = x.shape
    
    assert H % p == 0 and W % p == 0, f"Image dimensions ({H},{W}) must be divisible by patch size {p}"
    
    h_patches = H // p
    w_patches = W // p
    

    x = x.view(B, C, h_patches, p, w_patches, p)
    

    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()

    x = x.reshape(B, -1, C)
    
    return x

def unpatchify_raster(x, p, target_shape):
    B, N, C = x.shape
    H, W = target_shape
    
    h_patches = H // p
    w_patches = W // p
    
    x = x.view(B, h_patches, w_patches, p, p, C)
    
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    
    x = x.reshape(B, C, H, W)
    
    return x

def patchify_raster_2d(x: torch.Tensor, p: int, H: int, W: int) -> torch.Tensor:
    N, C1, C2 = x.shape
    
    assert N == H * W, f"N ({N}) must equal H*W ({H*W})"
    assert H % p == 0 and W % p == 0, f"H/W ({H}/{W}) must be divisible by patch size {p}"
    
    C_prime = C1 * C2
    x_flat = x.view(N, C_prime)
    
    x_2d = x_flat.view(H, W, C_prime)
    
    h_patches = H // p
    w_patches = W // p
    
    x_split = x_2d.view(h_patches, p, w_patches, p, C_prime)
    
    x_permuted = x_split.permute(0, 2, 1, 3, 4)
    
    x_reordered = x_permuted.reshape(N, C_prime)
    
    out = x_reordered.view(N, C1, C2)
    
    return out