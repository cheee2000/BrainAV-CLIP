"""
Swin Transformer 3D Implementation (Time as Channel)
Adapted from swin4d_transformer_ver7.py
"""

import itertools
from typing import Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm

from monai.networks.blocks import MLPBlock as Mlp
from monai.networks.layers import DropPath, trunc_normal_
from monai.utils import ensure_tuple_rep, look_up_option, optional_import

rearrange, _ = optional_import("einops", name="rearrange")

__all__ = [
    "SwinTransformer3D",
]

def window_partition(x, window_size):
    """
    Args:
        x: (B, D, H, W, C)
    Returns:
        windows: (B*num_windows, window_size*window_size*window_size, C)
    """
    x_shape = x.size()
    b, d, h, w, c = x_shape
    x = x.view(
        b,
        d // window_size[0],
        window_size[0],
        h // window_size[1],
        window_size[1],
        w // window_size[2],
        window_size[2],
        c,
    )
    windows = (
        x.permute(0, 1, 3, 5, 2, 4, 6, 7)
        .contiguous()
        .view(-1, window_size[0] * window_size[1] * window_size[2], c)
    )
    return windows


def window_reverse(windows, window_size, dims):
    """
    Args:
        windows: (B*num_windows, window_size, window_size, window_size, C)
        dims: (B, D, H, W)
    Returns:
        x: (B, D, H, W, C)
    """
    b, d, h, w = dims
    x = windows.view(
        b,
        d // window_size[0],
        h // window_size[1],
        w // window_size[2],
        window_size[0],
        window_size[1],
        window_size[2],
        -1,
    )
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(b, d, h, w, -1)
    return x


def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


class WindowAttention3D(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.to(attn.dtype).unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock3D(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        shift_size: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: str = "GELU",
        norm_layer: Type[LayerNorm] = nn.LayerNorm,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(hidden_size=dim, mlp_dim=mlp_hidden_dim, act=act_layer, dropout_rate=drop, dropout_mode="swin")

    def forward_part1(self, x, mask_matrix):
        b, d, h, w, c = x.shape
        window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
        x = self.norm1(x)
        pad_d0 = pad_h0 = pad_w0 = 0
        pad_d1 = (window_size[0] - d % window_size[0]) % window_size[0]
        pad_h1 = (window_size[1] - h % window_size[1]) % window_size[1]
        pad_w1 = (window_size[2] - w % window_size[2]) % window_size[2]
        
        x = F.pad(x, (0, 0, pad_w0, pad_w1, pad_h0, pad_h1, pad_d0, pad_d1))
        _, dp, hp, wp, _ = x.shape
        dims = [b, dp, hp, wp]
        
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(
                x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3)
            )
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
            
        x_windows = window_partition(shifted_x, window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, *(window_size + (c,)))
        shifted_x = window_reverse(attn_windows, window_size, dims)
        
        if any(i > 0 for i in shift_size):
            x = torch.roll(
                shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3)
            )
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_h1 > 0 or pad_w1 > 0:
            x = x[:, :d, :h, :w, :].contiguous()

        return x

    def forward_part2(self, x):
        x = self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, x, mask_matrix):
        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)
        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)
        return x


class PatchMergingV2_3D(nn.Module):
    def __init__(
        self, dim: int, norm_layer: Type[LayerNorm] = nn.LayerNorm, spatial_dims: int = 3, c_multiplier: int = 2
    ) -> None:
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, c_multiplier * dim, bias=False)
        self.norm = norm_layer(8 * dim)

    def forward(self, x):
        x_shape = x.size()
        b, d, h, w, c = x_shape
        # Merge 2x2x2 patches
        x = torch.cat(
            [x[:, i::2, j::2, k::2, :] for i, j, k in itertools.product(range(2), range(2), range(2))],
            -1,
        )
        x = self.norm(x)
        x = self.reduction(x)
        return x

MERGING_MODE = {"mergingv2": PatchMergingV2_3D}

def compute_mask(dims, window_size, shift_size, device):
    cnt = 0
    d, h, w = dims
    img_mask = torch.zeros((1, d, h, w, 1), device=device)
    for d_slice in (slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None)):
        for h_slice in (slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None)):
            for w_slice in (slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None)):
                img_mask[:, d_slice, h_slice, w_slice, :] = cnt
                cnt += 1

    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


class BasicLayer3D(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: Sequence[int],
        drop_path: list,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        norm_layer: Type[LayerNorm] = nn.LayerNorm,
        c_multiplier: int = 2,
        downsample: Optional[nn.Module] = None,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.no_shift = tuple(0 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock3D(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=self.no_shift if (i % 2 == 0) else self.shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    use_checkpoint=use_checkpoint,
                )
                for i in range(depth)
            ]
        )
        self.downsample = downsample
        if callable(self.downsample):
            self.downsample = downsample(
                dim=dim, norm_layer=norm_layer, spatial_dims=len(self.window_size), c_multiplier=c_multiplier
            )

    def forward(self, x):
        b, c, d, h, w = x.size()
        window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
        x = rearrange(x, "b c d h w -> b d h w c")
        dp = int(np.ceil(d / window_size[0])) * window_size[0]
        hp = int(np.ceil(h / window_size[1])) * window_size[1]
        wp = int(np.ceil(w / window_size[2])) * window_size[2]
        attn_mask = compute_mask([dp, hp, wp], window_size, shift_size, x.device)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = x.view(b, d, h, w, -1)
        if self.downsample is not None:
            x = self.downsample(x)
        x = rearrange(x, "b d h w c -> b c d h w")
        return x

class PositionalEmbedding3D(nn.Module):
    def __init__(self, dim: int, patch_dim: tuple) -> None:
        super().__init__()
        self.dim = dim
        self.patch_dim = patch_dim
        d, h, w = patch_dim
        self.pos_embed = nn.Parameter(torch.zeros(1, dim, d, h, w))
        trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        return x + self.pos_embed

class PatchEmbed3D(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, norm_layer=None):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
            img_size[2] // patch_size[2],
        )
        self.embed_dim = embed_dim
        self.in_chans = in_chans
        
        # 3D convolution for patch embedding
        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # x: B, C, D, H, W
        x = self.proj(x) # B, embed_dim, D_grid, H_grid, W_grid
        x = self.norm(x.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(x.shape)
        return x

class SwinTransformer3D(nn.Module):
    def __init__(
        self,
        img_size: Tuple,
        in_chans: int,
        embed_dim: int,
        window_size: Sequence[int],
        first_window_size: Sequence[int], # kept for compatibility but should be 3D
        patch_size: Sequence[int],
        depths: Sequence[int],
        num_heads: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: Type[LayerNorm] = nn.LayerNorm,
        patch_norm: bool = False,
        use_checkpoint: bool = False,
        spatial_dims: int = 4, # kept for compatibility
        c_multiplier: int = 2,
        last_layer_full_MSA: bool = False, # ignored in this simplified version
        downsample="mergingv2",
        **kwargs,
    ) -> None:
        super().__init__()
        
        # Adjust arguments for 3D
        # Expecting img_size to be (H, W, D, T) or (H, W, D)
        # BrainCLIP passes (72, 96, 96, 5) which is (H, W, D, T)
        # Note: BrainCLIP passes (72, 96, 96, 5) but input tensor is (B, C, D, H, W, T)
        # This implies standard order usually.
        # Let's check consistency.
        # SwinTransformer4D used:
        # img_size=(72, 96, 96, 5)
        # patch_size=(6, 6, 6, 1)
        # In 3D mode (Time as Channel):
        # Real 3D spatial size is (72, 96, 96).
        # Real in_chans is in_chans * T = 1 * 5 = 5.
        
        if len(img_size) == 4:
            self.spatial_img_size = img_size[:3]
            self.temporal_size = img_size[3]
        else:
            self.spatial_img_size = img_size
            self.temporal_size = 1 # Fallback
            
        # Adjust patch_size
        if len(patch_size) == 4:
            self.patch_size_3d = patch_size[:3]
        else:
            self.patch_size_3d = patch_size

        # Adjust window_size
        if len(window_size) == 4:
            self.window_size_3d = window_size[:3]
        else:
            self.window_size_3d = window_size

        if len(first_window_size) == 4:
            self.first_window_size_3d = first_window_size[:3]
        else:
            self.first_window_size_3d = first_window_size

        self.real_in_chans = in_chans * self.temporal_size
        
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        
        self.patch_embed = PatchEmbed3D(
            img_size=self.spatial_img_size,
            patch_size=self.patch_size_3d,
            in_chans=self.real_in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        patch_dim = (
            self.spatial_img_size[0] // self.patch_size_3d[0],
            self.spatial_img_size[1] // self.patch_size_3d[1],
            self.spatial_img_size[2] // self.patch_size_3d[2]
        )
        
        print("Swin3D Setup:")
        print(f"  Input channels: {self.real_in_chans} (original {in_chans} * time {self.temporal_size})")
        print(f"  Spatial Img Size: {self.spatial_img_size}")
        print(f"  Patch Size: {self.patch_size_3d}")
        print(f"  Window Size: {self.window_size_3d}")
        
        self.pos_embeds = nn.ModuleList()
        pos_embed_dim = embed_dim
        current_patch_dim = patch_dim
        for i in range(self.num_layers):
            self.pos_embeds.append(PositionalEmbedding3D(pos_embed_dim, current_patch_dim))
            pos_embed_dim = pos_embed_dim * c_multiplier
            current_patch_dim = (current_patch_dim[0]//2, current_patch_dim[1]//2, current_patch_dim[2]//2)

        self.layers = nn.ModuleList()
        down_sample_mod = look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample

        # Layer 0
        layer = BasicLayer3D(
            dim=int(embed_dim),
            depth=depths[0],
            num_heads=num_heads[0],
            window_size=self.first_window_size_3d,
            drop_path=dpr[sum(depths[:0]) : sum(depths[: 0 + 1])],
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            norm_layer=norm_layer,
            c_multiplier=c_multiplier,
            downsample=down_sample_mod if 0 < self.num_layers - 1 else None,
            use_checkpoint=use_checkpoint,
        )
        self.layers.append(layer)

        # Subsequent layers
        for i_layer in range(1, self.num_layers):
            # For last layer, downsample is None
            is_last = (i_layer == self.num_layers - 1)
            layer = BasicLayer3D(
                dim=int(embed_dim * (c_multiplier**i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=self.window_size_3d,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                c_multiplier=c_multiplier,
                downsample=down_sample_mod if not is_last else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        self.num_features = int(embed_dim * c_multiplier ** (self.num_layers - 1))
        self.norm = norm_layer(self.num_features)

    def forward(self, x):
        # Input x: (B, C, D, H, W, T)
        # We need to reshape to (B, C*T, D, H, W)
        
        if x.dim() == 6:
            B, C, D, H, W, T = x.shape
            # Permute to move T next to C -> (B, C, T, D, H, W)
            x = x.permute(0, 1, 5, 2, 3, 4).contiguous()
            # Merge C and T
            x = x.view(B, C * T, D, H, W)
            
        # x is now (B, C_new, D, H, W)
        
        x = self.patch_embed(x)
        x = self.pos_drop(x) # (B, C, D, H, W)

        for i in range(self.num_layers):
            x = self.pos_embeds[i](x)
            x = self.layers[i](x)

        return x
