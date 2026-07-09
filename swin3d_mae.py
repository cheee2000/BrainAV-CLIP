"""
Swin Transformer 3D/4D MAE Implementation

Note on Architecture:
This implementation follows the SimMIM (Simple Masked Image Modeling) design 
rather than the original MAE (Masked Autoencoder) design for ViT.
- Standard MAE (ViT): Discards masked tokens to save compute.
- SimMIM (Swin): Keeps all tokens (masked ones replaced by a learnable token) to preserve 
  the 3D window structure required by Swin Transformer. 
  
Therefore, the Encoder processes all patches (masked + unmasked).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.utils import ensure_tuple_rep
from monai.networks.layers import trunc_normal_
from swin3d_transformer import SwinTransformer3D, BasicLayer3D, PatchEmbed3D, get_window_size

class PatchExpand3D(nn.Module):
    """
    Reverse of PatchMerging. Upsamples the feature map by 2x.
    """
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        # Output dim will be dim // dim_scale
        out_dim = dim // dim_scale
        
        # We need to generate (2*2*2) * output_dim features to expand spatial dims by 2
        self.expand = nn.Linear(dim, 8 * out_dim, bias=False)
        self.norm = norm_layer(out_dim)

    def forward(self, x):
        """
        x: B, C, D, H, W
        """
        # Linear projection
        x = x.permute(0, 2, 3, 4, 1).contiguous() # (B, D, H, W, C)
        x = self.expand(x) # (B, D, H, W, 8*C_out)
        
        B, D, H, W, _ = x.shape
        
        # Rearrange to spatial dimensions:
        # 1. View as (B, D, H, W, 2, 2, 2, C_out)
        # 2. Permute to (B, D, 2, H, 2, W, 2, C_out)
        # 3. Fuse dims to (B, D*2, H*2, W*2, C_out)
        
        x = x.view(B, D, H, W, 2, 2, 2, -1)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
        x = x.view(B, D * 2, H * 2, W * 2, -1)
        
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous() # (B, C, D, H, W)
        return x

class SwinTransformer3DDecoder(nn.Module):
    """
    Lightweight Decoder for Swin MAE.
    Upsamples from the bottleneck back to the original patch grid size.
    """
    def __init__(
        self, 
        embed_dim,
        depths,
        num_heads,
        window_size,
        patch_size, # Needed for final projection
        in_chans,   # Needed for final projection
        num_patch_grid, # (D, H, W) of the patch grid
        final_upsample="expand", 
    ):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.in_chans = in_chans
        
        # Build Decoder Layers (Reverse of Encoder)
        self.layers = nn.ModuleList()
        
        for i_layer in range(self.num_layers):
            # Swin Block
            layer = BasicLayer3D(
                dim=embed_dim,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                drop_path=0.0,
                mlp_ratio=4.0,
                qkv_bias=True,
                drop=0.0,
                attn_drop=0.0,
                norm_layer=nn.LayerNorm,
                downsample=None # We handle upsampling separately
            )
            self.layers.append(layer)
            
            # Upsample Layer (PatchExpand)
            # We add PatchExpand for every layer to match the encoder's downsampling steps.
            # dim_scale=1 keeps the channel dimension constant (Isotropic Decoder), 
            # while expanding spatial resolution.
            self.layers.append(PatchExpand3D(dim=embed_dim, dim_scale=1))
                
        # Final projection to pixel space
        # Output: (B, D_grid, H_grid, W_grid, Patch_Vol * In_Chans)
        self.final_proj = nn.Linear(embed_dim, patch_size[0]*patch_size[1]*patch_size[2]*in_chans)

    def forward(self, x):
        # x: (B, C, D, H, W)
             
        for layer in self.layers:
            x = layer(x)
            
        # Final projection
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.final_proj(x)
        return x

class CorrelationLoss(nn.Module):
    def __init__(self, dim=-1, eps=1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, pred, target, mask=None):
        """
        pred: (B, L, C) or (B, N)
        target: (B, L, C) or (B, N)
        mask: (B, L)
        """
        if mask is not None:
            
            # Expand mask to match pred shape if needed
            if mask.dim() < pred.dim():
                 mask = mask.unsqueeze(-1).expand_as(pred)
            pass

        # Calculate PCC along dim
        # If input is (B, L, C), and we want spatial correlation, we flatten to (B, L*C)
        B = pred.shape[0]
        pred = pred.reshape(B, -1)
        target = target.reshape(B, -1)
        
        if mask is not None:
            mask = mask.reshape(B, -1)
            loss = 0.0
            for i in range(B):
                m = mask[i].bool()
                if m.sum() == 0: continue
                p = pred[i][m]
                t = target[i][m]
                
                vx = p - p.mean()
                vy = t - t.mean()
                cost = (vx * vy).sum() / (torch.sqrt((vx ** 2).sum()) * torch.sqrt((vy ** 2).sum()) + self.eps)
                loss += (1.0 - cost)
            return loss / B

        vx = pred - torch.mean(pred, dim=1, keepdim=True)
        vy = target - torch.mean(target, dim=1, keepdim=True)

        numerator = torch.sum(vx * vy, dim=1)
        denominator = torch.sqrt(torch.sum(vx ** 2, dim=1)) * torch.sqrt(torch.sum(vy ** 2, dim=1)) + self.eps
        return (1.0 - numerator / denominator).mean()

class SwinMAE3D(nn.Module):
    def __init__(
        self, 
        img_size=(72, 96, 96), 
        patch_size=(6, 6, 6), 
        in_chans=5, # 5 time steps as channels
        embed_dim=128,
        window_size=(4, 4, 4),
        mask_ratio=0.75,
        decoder_embed_dim=256,
        decoder_depths=(4, 4),
        decoder_num_heads=(4, 4),
        loss_type="mse_pcc", # Options: "mse", "mse_pcc"
        mask_strategy="random",  # "random" or "block"
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.in_chans = in_chans
        self.loss_type = loss_type
        self.corr_loss = CorrelationLoss()
        self.mask_strategy = mask_strategy
        
        # 1. Encoder (Pre-trained part)
        # Modified initialization: Removed hack, passing in_chans directly.
        # Assuming SwinTransformer3D correctly handles generic in_chans.
        self.encoder = SwinTransformer3D(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            window_size=window_size,
            first_window_size=(2,2,2),
            depths=(2, 2, 6),
            num_heads=(4, 8, 16),
            downsample="mergingv2",
            spatial_dims=3
        )
        
        # 2. Mask Token
        # self.mask_token is (1, 1, 1, 1, 1, embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, 1, embed_dim))
        trunc_normal_(self.mask_token, std=0.02)
        
        # 3. Decoder
        # Encoder output: (B, 512, D/4, H/4, W/4) (if depths=(2,2,6) and dims double)
        # We need to bridge 512 -> decoder_embed_dim
        
        self.decoder_embed = nn.Linear(512, decoder_embed_dim) # Project bottleneck to embedding dim
        
        self.decoder = SwinTransformer3DDecoder(
            embed_dim=decoder_embed_dim,
            depths=decoder_depths,
            num_heads=decoder_num_heads,
            window_size=window_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_patch_grid=(img_size[0]//patch_size[0], img_size[1]//patch_size[1], img_size[2]//patch_size[2])
        )
        
    def patchify(self, imgs):
        """
        imgs: (B, C, D, H, W)
        x: (B, L, patch_prod*C)
        """
        p = self.patch_size
        assert imgs.shape[2] % p[0] == 0 and imgs.shape[3] % p[1] == 0 and imgs.shape[4] % p[2] == 0
        
        x = imgs.unfold(2, p[0], p[0]) \
                .unfold(3, p[1], p[1]) \
                .unfold(4, p[2], p[2]) 
        # (B, C, D_grid, H_grid, W_grid, p, p, p)
        
        x = x.permute(0, 2, 3, 4, 1, 5, 6, 7).contiguous()
        # (B, D_grid, H_grid, W_grid, C, p, p, p)
        
        x = x.view(imgs.shape[0], -1, self.in_chans * p[0] * p[1] * p[2])
        return x

    def unpatchify(self, x):
        """
        x: (B, L, patch_prod*C)
        imgs: (B, C, D, H, W)
        """
        p = self.patch_size
        h, w, d = self.img_size[0] // p[0], self.img_size[1] // p[1], self.img_size[2] // p[2]
        
        x = x.view(x.shape[0], h, w, d, self.in_chans, p[0], p[1], p[2])
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        x = x.view(x.shape[0], self.in_chans, h * p[0], w * p[1], d * p[2])
        return x

    def forward_encoder(self, x, mask_ratio, valid_mask=None):
        # x: (B, C, D, H, W)
        
        # 1. Patch Embed
        x = self.encoder.patch_embed(x) 
        # x: (B, embed_dim, D_grid, H_grid, W_grid)
        x = x.permute(0, 2, 3, 4, 1).contiguous() # (B, D_g, H_g, W_g, Embed)

        # 2. Masking (SimMIM style)
        B, D, H, W, C = x.shape
        L = D * H * W
        
        mask = torch.zeros(B, L, device=x.device)
        # Random noise for patch selection; optionally smoothed to form block-like masks
        if getattr(self, "mask_strategy", "random") == "block":
            # Use 3D average pooling over random noise to induce local correlation,
            # which makes top-k selection produce block-ish masked regions.
            block_size = 3
            noise_3d = torch.rand(B, 1, D, H, W, device=x.device)
            noise_3d = F.avg_pool3d(
                noise_3d,
                kernel_size=block_size,
                stride=1,
                padding=block_size // 2,
            )
            noise = noise_3d.view(B, L)
        else:
            noise = torch.rand(B, L, device=x.device)
        
        if valid_mask is not None:
            # valid_mask: (1, L) or (B, L) or (1, L, 1). Should be 1.0 for valid, 0.0 for invalid.
            if valid_mask.ndim == 3: # (1, L, 1)
                valid_mask = valid_mask.view(valid_mask.shape[0], -1) # (1, L)
            
            # noise: (B, L)
            # valid_mask: (1, L) or (B, L). 1=valid, 0=invalid.

            
            # Ensure valid_mask is on same device
            valid_mask = valid_mask.to(x.device)
            
            noise = noise + (valid_mask - 1) * 100.0 
            # If valid (1): noise + 0 = noise [0..1]
            # If invalid (0): noise - 100 = [-100..-99]
            
            # Calculate number of patches to mask based on VALID count
            # We use the first element to determine count
            n_valid = valid_mask[0].sum()
            num_mask = int(n_valid * mask_ratio)
            
            # Total patches to keep = Total - num_mask
            len_keep = L - num_mask
        else:
            len_keep = int(L * (1 - mask_ratio))
            
        # Select top k patches to mask
        ids_shuffle = torch.argsort(noise, dim=1)
        # Mask indices
        mask_idx = ids_shuffle[:, len_keep:] 
        mask.scatter_(1, mask_idx, 1)
        
        mask = mask.view(B, D, H, W, 1) # (B, D, H, W, 1)
        
        # Apply mask: replace masked tokens with mask_token
        # SimMIM strategy: keep all tokens but replace masked ones
        mask_tokens = self.mask_token.expand(B, D, H, W, C)
        w = mask.type_as(x)
        x = x * (1 - w) + mask_tokens * w
        
        # 3. Add Pos Embed & Run Encoder
        x = x.permute(0, 4, 1, 2, 3).contiguous() # (B, C, D, H, W)
        
        # Swin Transformer processing
        # Note: We rely on Swin's internal mechanisms for position embedding (if absolute)
        # or relative position biases inside blocks.
        if hasattr(self.encoder, 'pos_drop'):
             x = self.encoder.pos_drop(x)

        if hasattr(self.encoder, 'pos_embeds'):

             for i in range(self.encoder.num_layers):
                x = self.encoder.pos_embeds[i](x)
                x = self.encoder.layers[i](x)
        else:
             # Standard Swin forward loop
             for layer in self.encoder.layers:
                 x = layer(x)
            
        return x, mask, ids_shuffle

    def forward_decoder(self, x):
        # x: Encoder output (B, 512, D_small, H_small, W_small)
        # Project back to embedding dim if needed
        B, C, D, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).contiguous() # (B, D, H, W, 512)
        x = self.decoder_embed(x) # (B, D, H, W, 128)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        
        x = self.decoder(x)
        return x

    def forward(self, imgs, valid_mask_patchified=None, mask_ratio=None, mask_loss_only=True):
        # imgs: (B, C, D, H, W)
        
        if mask_ratio is None:
            mask_ratio = self.mask_ratio

        # Encoder
        latent, mask, _ = self.forward_encoder(imgs, mask_ratio, valid_mask=valid_mask_patchified)
        
        # Decoder
        pred = self.forward_decoder(latent) 
        # pred: (B, D_grid, H_grid, W_grid, Patch_Vol*C)
        
        # Reshape pred to (B, L, -1) for loss calc
        pred = pred.view(pred.shape[0], -1, pred.shape[-1])
        
        # Patchify target
        target = self.patchify(imgs)
        
        # --- New Normalization & Masking Logic ---
        
        # 1. Patch-wise Normalization (×)

        # 2. Loss Calculation
        mse_loss = (pred - target) ** 2
        mse_loss = mse_loss.mean(dim=-1) # (B, L)
        
        # 3. Combine MAE Mask with Brain Valid Mask
        # mask: currently (B, D, H, W, 1) from forward_encoder -> flatten to (B, L)
        mask = mask.flatten(1, 3).squeeze(-1) # (B, L)
        
        # Determine base mask for loss
        if not mask_loss_only:
            # If we want loss on ALL patches (not just masked ones)
            combined_mask = torch.ones_like(mask)
        elif mask.sum() == 0:
            # If mask_ratio is 0 (validation/test), mask is all zeros. Compute loss on ALL.
            combined_mask = torch.ones_like(mask)
        else:
            # Standard MAE: Compute loss only on masked patches
            combined_mask = mask 
        
        if valid_mask_patchified is not None:
             # Ensure valid_mask is broadcastable or matches shape
             # valid_mask_patchified is likely (1, L, 1) -> squeeze to (1, L)
             vm = valid_mask_patchified.squeeze(-1)
             combined_mask = combined_mask * vm
        
        # 4. Final Loss Averaging
        # Sum loss over masked & valid regions, divide by count
        loss_sum = (mse_loss * combined_mask).sum()
        mask_sum = combined_mask.sum()
        
        if mask_sum > 0:
            loss = loss_sum / mask_sum
        else:
            loss = loss_sum * 0.0 # Avoid NaN if no valid masked patches

        if self.loss_type == "mse_pcc":
            # Add PCC loss on masked regions
            pcc_loss = self.corr_loss(pred, target, mask=combined_mask)
            # Combine: 1.0 * MSE + 1.0 * (1-PCC)
            # Since PCC is in [0, 1] range (loss 0..2) and MSE is also roughly 0..1 (Z-scored)
            loss = loss + pcc_loss

        return loss, pred, mask
