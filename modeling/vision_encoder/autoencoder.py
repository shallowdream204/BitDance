import torch
import torch.nn as nn
import math
from einops import rearrange
import torch.nn.functional as F
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
    
class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, in_channels, num_res_blocks, z_channels, ch_mult=(1, 2, 2, 4), 
                resolution=None, double_z=False,
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
                                 bias=False
        )

        ## construct the model
        self.down = nn.ModuleList()

        in_ch_mult = (1,)+tuple(ch_mult)
        for i_level in range(self.num_blocks):
            block = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level] #[1, 1, 2, 2, 4]
            block_out = ch*ch_mult[i_level] #[1, 2, 2, 4]
            for _ in range(self.num_res_blocks):
                block.append(ResBlock(block_in, block_out))
                block_in = block_out
            
            down = nn.Module()
            down.block = block
            if i_level < self.num_blocks - 1:
                down.downsample = nn.Conv2d(block_out, block_out, kernel_size=(3, 3), stride=(2, 2), padding=1)

            self.down.append(down)
        
        ### mid
        self.mid_block = nn.ModuleList()
        for res_idx in range(self.num_res_blocks):
            self.mid_block.append(ResBlock(block_in, block_in))
        
        ### end
        self.norm_out = nn.GroupNorm(32, block_out, eps=1e-6)
        self.conv_out = nn.Conv2d(block_out, z_channels, kernel_size=(1, 1))
            
    def forward(self, x):

        ## down
        x = self.conv_in(x)
        for i_level in range(self.num_blocks):
            for i_block in range(self.num_res_blocks):
                x = self.down[i_level].block[i_block](x)
            
            if i_level <  self.num_blocks - 1:
                x = self.down[i_level].downsample(x)
        
        ## mid 
        for res in range(self.num_res_blocks):
            x = self.mid_block[res](x)
        

        x = self.norm_out(x)
        x = swish(x)
        x = self.conv_out(x)

        return x

class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, in_channels, num_res_blocks, z_channels, ch_mult=(1, 2, 2, 4), 
                resolution=None, double_z=False,) -> None:
        super().__init__()

        self.ch = ch
        self.num_blocks = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        block_in = ch*ch_mult[self.num_blocks-1]

        self.conv_in = nn.Conv2d(
            z_channels, block_in, kernel_size=(3, 3), padding=1, bias=True
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

        z = self.conv_in(z)

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

class GANDecoder(nn.Module):
    def __init__(self, *, ch, out_ch, in_channels, num_res_blocks, z_channels, ch_mult=(1, 2, 2, 4), 
                resolution=None, double_z=False,) -> None:
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
                checkpoint=None,
                gan_decoder = False,
                ):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = GANDecoder(**ddconfig) if gan_decoder else Decoder(**ddconfig)

        # Load weights from the checkpoint
        if checkpoint is not None:
            self.load_from_ckpt(checkpoint)

    def load_from_ckpt(self, checkpoint):
        state = torch.load(download(checkpoint), mmap=True, map_location="cpu")
        log_info = self.load_state_dict(state["state_dict"], strict=False)
        has_missing_keys = bool(log_info.missing_keys)
        has_unexpected_keys = bool(log_info.unexpected_keys)
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
    

    def vt_forward(self, image_list, max_bs=32, ps=1):
        groups = defaultdict(list)  # {(H, W): [(idx, image_tensor), ...]}
        for i, img in enumerate(image_list):
            _, _, H, W = img.shape
            groups[(H, W)].append((i, img))

        output = [None] * len(image_list)

        for (H, W), items in groups.items():
            for start in range(0, len(items), max_bs):
                chunk = items[start:start + max_bs]
                idxs = [x[0] for x in chunk]
                imgs = [x[1] for x in chunk]

                batch = torch.cat(imgs, dim=0)  # [B, 3, H, W]

                quant = self.encode(batch)                      # [B, C, h, w]

                for b in range(quant.size(0)):
                    q = rearrange(quant[b], "c (h p1) (w p2) -> (h w p1 p2) c", p1=ps, p2=ps)
                    output[idxs[b]] = q

        return torch.cat(output, dim=0)
    
    def vt_forward_maxpad(
        self,
        image_list,
        max_bs=32,
        stride=32,
        min_size=256,
        max_size=2048,
        max_pixels=1024 * 1024,
        normal_buckets=(384, 512, 768, 1024),
    ):
        """
        image_list: list of [1, 3, H, W]
        return: Tensor [(sum_i Hi*Wi/stride^2), C]
        """

        def is_long_image(H, W):
            major = max(H, W)
            minor = min(H, W)
            return (
                major >= 1024 and
                minor <= 768 and
                major / minor >= 1.5
            )

        groups = defaultdict(list)
        sizes = {}

        for idx, img in enumerate(image_list):
            _, _, H, W = img.shape

            # assert H >= min_size and W >= min_size
            # assert H <= max_size and W <= max_size
            # assert H * W <= max_pixels, f"image is too large: {H}x{W}"

            if is_long_image(H, W):
                bucket = "long"
            else:
                major = max(H, W)
                for b in normal_buckets:
                    if major <= b:
                        bucket = b
                        break
                else:
                    bucket = "long"

            groups[bucket].append(idx)
            sizes[idx] = (H, W)

        output = [None] * len(image_list)


        for bucket, idxs in groups.items():
            imgs = [image_list[i] for i in idxs]

            for start in range(0, len(imgs), max_bs):
                batch_imgs = imgs[start:start + max_bs]
                batch_idxs = idxs[start:start + max_bs]

                H_max = max(img.shape[-2] for img in batch_imgs)
                W_max = max(img.shape[-1] for img in batch_imgs)

                H_pad = math.ceil(H_max / stride) * stride
                W_pad = math.ceil(W_max / stride) * stride

                padded = []
                for img in batch_imgs:
                    _, _, H, W = img.shape
                    pad_h = H_pad - H
                    pad_w = W_pad - W
                    padded.append(F.pad(img, (0, pad_w, 0, pad_h)))

                batch = torch.cat(padded, dim=0)  # [B, 3, H_pad, W_pad]

                quant = self.encode(batch)  # [B, C, h', w']

                for i, q in enumerate(quant):
                    H, W = sizes[batch_idxs[i]]
                    h_lat = math.ceil(H / stride)
                    w_lat = math.ceil(W / stride)

                    q = q[:, :h_lat, :w_lat]
                    q = rearrange(q, "c h w -> (h w) c")

                    output[batch_idxs[i]] = q

        return torch.cat(output, dim=0)


    def decode(self, quant):
        dec = self.decoder(quant)
        return dec

    def forward(self, input):
        quant = self.encode(input)
        dec = self.decode(quant)
        return dec, quant