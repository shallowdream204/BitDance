from PIL import Image
from functools import lru_cache

import torch
import numpy as np


def sigmoid_scalar_np(x: float) -> float:
    x = np.float64(x)
    if x >= 0:
        return float(1.0 / (1.0 + np.exp(-x)))
    else:
        ex = np.exp(x)
        return float(ex / (1.0 + ex))


def patchify(image, patch_size):
    p = patch_size
    c, h, w = image.shape
    assert h % p == 0 and w % p == 0
    image = image.reshape(c, h // p, p, w // p, p)
    image = torch.einsum("chpwq->hwpqc", image)
    image = image.reshape(-1, p**2 * c)
    return image


def get_flattened_position_ids_extrapolate(num_patches_h, num_patches_w, max_num_patches_per_side):
    coords_h = torch.arange(0, num_patches_h)
    coords_w = torch.arange(0, num_patches_w)
    pos_ids = (coords_h[:, None] * max_num_patches_per_side + coords_w).flatten()
    return pos_ids


def get_flattened_position_ids_interpolate(num_patches_h, num_patches_w, ref_h, ref_w, max_num_patches_per_side):
    boundaries_h = torch.arange(1 / ref_h, 1.0, 1 / ref_h)
    boundaries_w = torch.arange(1 / ref_w, 1.0, 1 / ref_w)
    fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / num_patches_h)
    fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / num_patches_w)
    bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries_h, right=True)
    bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries_w, right=True)
    pos_ids = (bucket_coords_h[:, None] * max_num_patches_per_side + bucket_coords_w).flatten()
    return pos_ids


@lru_cache(maxsize=32)
def get_patches_center_coordinates(
    num_patches_h: int, num_patches_w: int, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Computes the 2D coordinates of the centers of image patches, normalized to the range [-1, +1].
    The center of each patch is exactly halfway between its top-left and bottom-right corners.

    Args:
        num_patches_h (int): Number of patches along the vertical (height) axis.
        num_patches_w (int): Number of patches along the horizontal (width) axis.
        dtype (torch.dtype): The desired data type of the returned tensor.

    Returns:
        torch.Tensor: A tensor of shape (height * width, 2), where each row contains the (y, x)
            coordinates of a patch center, normalized to [-1, +1].
    """
    coords_h = torch.arange(0.5, num_patches_h, dtype=dtype)
    coords_w = torch.arange(0.5, num_patches_w, dtype=dtype)
    coords_h = coords_h / num_patches_h
    coords_w = coords_w / num_patches_w
    # (height, width, 2) -> (height * width, 2)
    coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
    coords = coords.flatten(0, 1)
    # Shift range [0, 1] to [-1, +1]
    coords = 2.0 * coords - 1.0
    return coords


def pil_img2rgb(image):
    if image.mode == "RGBA" or image.info.get("transparency", None) is not None:
        image = image.convert("RGBA")
        white = Image.new(mode="RGB", size=image.size, color=(255, 255, 255))
        white.paste(image, mask=image.split()[3])
        image = white
    else:
        image = image.convert("RGB")

    return image

def alias_special_token(tokenizer, attr, token_str):
    setattr(tokenizer, f"_{attr}_token", token_str)

    cls = type(tokenizer)
    if not hasattr(cls, attr):
        setattr(cls, attr, property(lambda self, a=attr: getattr(self, f"_{a}_token")))
    if not hasattr(cls, f"{attr}_id"):
        setattr(cls, f"{attr}_id", property(lambda self, a=attr:
            self.convert_tokens_to_ids(getattr(self, f"_{a}_token"))))

def set_special_token_aliases(
    tokenizer,
    im_start_str="<|im_start|>",
    im_end_str="<|im_end|>",
    vision_start_str="<|vision_start|>",
    vision_end_str="<|vision_end|>",
    image_pad="<|image_pad|>",
):
    alias_special_token(tokenizer, "im_start", im_start_str)
    alias_special_token(tokenizer, "im_end", im_end_str)
    alias_special_token(tokenizer, "start_of_image", vision_start_str)
    alias_special_token(tokenizer, "end_of_image", vision_end_str)
    alias_special_token(tokenizer, "image_pad", image_pad)

    return tokenizer


def add_resolution_special_tokens(tokenizer, max_resolution=4096, patch_size=16):
    assert max_resolution % patch_size == 0, \
        f"max_resolution ({max_resolution}) must be divisible by patch_size ({patch_size})"

    num_levels = max_resolution // patch_size
    res_tokens = [f"<|res_{i}|>" for i in range(1, num_levels + 1)]
    added = tokenizer.add_special_tokens({
        "additional_special_tokens": res_tokens
    })

    for i in range(1, num_levels + 1):
        tok = f"<|res_{i}|>"
        alias_special_token(tokenizer, f"res_{i}", tok)
    
    return tokenizer

def add_query_special_tokens(tokenizer, parallel_num=1):
    if parallel_num ==1:
        return tokenizer
    else:
        query_tokens = [f"<|query_{i}|>" for i in range(1, parallel_num)]
        added = tokenizer.add_special_tokens({
            "additional_special_tokens": query_tokens
        })

        for i in range(1, parallel_num):
            tok = f"<|query_{i}|>"
            alias_special_token(tokenizer, f"query_{i}", tok)
        
        return tokenizer