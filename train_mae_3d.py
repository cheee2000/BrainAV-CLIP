import os
import math
import random
from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
import wandb
import torch.distributed as dist
from scipy.stats import t as student_t

from dataset import BrainDataset, prepare_fmri_and_stim_same_movie, prepare_fmri_and_stim_test
from swin3d_mae import SwinMAE3D

WANDB_ENTITY = os.getenv("WANDB_ENTITY", "YOUR_WANDB_ENTITY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "RetrievalAV")

class WarmupCosineScheduler(LambdaLR):
    def __init__(self, optimizer, warmup_steps, total_steps):
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            # Cosine decay
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        super().__init__(optimizer, lr_lambda)

class BrainMAE(pl.LightningModule):
    def __init__(
            self,
            img_size=(72, 96, 96),
            patch_size=(6, 6, 6),
            in_chans=5,
            embed_dim=128,
            window_size=(4, 4, 4),
            mask_ratio=0.75,
            log_path=None,
            valid_mask_volume=None, # New argument: (72, 96, 96) numpy or tensor
            total_steps=None,
            decoder_embed_dim=256,
            decoder_depths=(4, 4),
            mask_loss_only=True, # New argument: If False, compute loss on all valid patches
            mask_strategy="random",  # "random" or "block"
    ):
        super().__init__()
        # We need to explicitly tell save_hyperparameters to ignore valid_mask_volume
        # because it is a large array and we don't want it in the checkpoint hparams.
        self.save_hyperparameters(ignore=['valid_mask_volume']) 
        
        self.total_steps = total_steps
        self.mask_loss_only = mask_loss_only
        
        self.model = SwinMAE3D(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            window_size=window_size,
            mask_ratio=mask_ratio,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depths=decoder_depths,
            loss_type="mse",
            mask_strategy=mask_strategy,
        )
        self.log_path = log_path
        
        self.val_pcc_list = []
        self.test_pcc_list = []
        self.val_voxel_preds = []
        self.val_voxel_targets = []
        self.test_voxel_preds = []
        self.test_voxel_targets = []
        self.test_mode = 'val'

        # Process Valid Mask
        self.valid_mask_patchified = None
        self.valid_mask_bool = None
        if valid_mask_volume is not None:
            # 1. Store boolean mask for PCC calc (voxel-wise)
            # Make sure it's on cpu first, then register_buffer moves it to device
            self.register_buffer('valid_mask_bool_buf', torch.tensor(valid_mask_volume > 0.5, dtype=torch.bool))
            
            # 2. Prepare patchified mask for Loss calc
            # Create a dummy tensor (1, 1, D, H, W) to pass through patchify
            vm_tensor = torch.tensor(valid_mask_volume, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            
            # Use model patchify logic manually
            p = patch_size
            # vm_tensor: (1, 1, 72, 96, 96)
            x = vm_tensor.unfold(2, p[0], p[0]).unfold(3, p[1], p[1]).unfold(4, p[2], p[2])
            x = x.permute(0, 2, 3, 4, 1, 5, 6, 7).contiguous()
            x = x.view(1, -1, 1 * p[0] * p[1] * p[2]) # (1, L, p*p*p)
            
            # A patch is valid if it contains brain voxels. 
            x_mean = x.mean(dim=-1, keepdim=True) # (1, L, 1)
            valid_mask_patchified = (x_mean > 0.0).float()
            
            self.register_buffer('valid_mask_buf', valid_mask_patchified)

            # Print valid patch info
            n_total = x.shape[1]
            n_valid = int(valid_mask_patchified.sum().item())
            print(f"Valid patches: {n_valid}/{n_total} ({n_valid/n_total*100:.2f}%)")
        else:
            self.valid_mask_bool_buf = None
            self.valid_mask_buf = None

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=2e-4, betas=(0.9, 0.95), weight_decay=0.05)
        
        steps = self.total_steps if self.total_steps is not None else 200 * 100
        scheduler = WarmupCosineScheduler(optimizer, warmup_steps=200, total_steps=steps)
        
        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}

    def training_step(self, batch, batch_idx):
        # batch: brain_data, stimuli_feature, index
        brain_data, _, _ = batch 
        # brain_data: (B, 1, 72, 96, 96, 5) -> We need (B, 5, 72, 96, 96)
        
        # Reshape: (B, 1, D, H, W, T) -> (B, T, D, H, W)
        x = brain_data.squeeze(1).permute(0, 4, 1, 2, 3).contiguous()
        
        loss, pred, mask = self.model(x, valid_mask_patchified=self.valid_mask_buf, mask_loss_only=self.mask_loss_only)
        
        self.log("train/mae_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)
        
        return loss

    def compute_pcc(self, pred, target, mask=None):
        """
        Compute PCC on valid voxels only (using self.valid_mask_bool_buf).
        
        pred: (B, L, C_patch) -> need to unpatchify to (B, C, D, H, W)
        target: (B, L, C_patch) -> need to unpatchify to (B, C, D, H, W)
        """
        # 1. Unpatchify
        # pred and target are (B, L, C*P*P*P)
        pred_vol = self.model.unpatchify(pred) # (B, C, D, H, W)
        target_vol = self.model.unpatchify(target) # (B, C, D, H, W)
        
        # 2. Flatten to (B, C, Voxels)
        B, C, D, H, W = pred_vol.shape
        pred_flat = pred_vol.view(B, C, -1)
        target_flat = target_vol.view(B, C, -1)
        
        # 3. Select valid voxels
        if self.valid_mask_bool_buf is not None:
             valid_indices = self.valid_mask_bool_buf.view(-1) # Flatten mask
             pred_flat = pred_flat[:, :, valid_indices]
             target_flat = target_flat[:, :, valid_indices]
        
        # 4. Compute PCC (Spatial Correlation)
        # Here we compute PCC across voxels (spatial correlation) per sample, per channel
        # Usually for reconstruction: Spatial Correlation per sample.
        # Shape: (B, C, N_valid_voxels)
        
        vx = pred_flat - torch.mean(pred_flat, dim=2, keepdim=True)
        vy = target_flat - torch.mean(target_flat, dim=2, keepdim=True)
        
        # Sum over spatial dim (dim=2)
        numerator = torch.sum(vx * vy, dim=2)
        denominator = torch.sqrt(torch.sum(vx ** 2, dim=2)) * torch.sqrt(torch.sum(vy ** 2, dim=2)) + 1e-8
        cost = numerator / denominator
        
        return cost.mean(), pred_flat, target_flat # Return mean PCC and flat tensors

    def validation_step(self, batch, batch_idx):
        brain_data, _, _ = batch
        x = brain_data.squeeze(1).permute(0, 4, 1, 2, 3).contiguous()
        
        loss, pred, mask = self.model(x, valid_mask_patchified=self.valid_mask_buf, mask_ratio=0.0)
        self.log("val/mae_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
        # Calculate PCC
        target = self.model.patchify(x)
        # Note: We compute PCC on un-normalized data usually, but here x is already normalized.
        # We should probably use the reconstruction directly.
        
        pcc, pred_flat, target_flat = self.compute_pcc(pred, target)
        self.val_pcc_list.append(pcc)
        
        # Store for voxel-wise PCC (move to CPU to save memory)
        self.val_voxel_preds.append(pred_flat.cpu())
        self.val_voxel_targets.append(target_flat.cpu())
        
        return loss

    def on_validation_epoch_end(self):
        # Aggregate Sample-wise PCC
        if len(self.val_pcc_list) > 0:
            avg_pcc = torch.stack(self.val_pcc_list).mean()
            
            # Sync across ranks
            if dist.is_initialized():
                dist.all_reduce(avg_pcc, op=dist.ReduceOp.SUM)
                avg_pcc /= dist.get_world_size()
            
            self.log("val/pcc", avg_pcc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=False)
            print(f"\nValidation PCC: {avg_pcc:.4f}")
            
        self.val_pcc_list = [] # Reset

        # Aggregate Voxel-wise PCC
        if len(self.val_voxel_preds) > 0:
            # Concatenate all batches: (Total_Samples, C, V)
            all_preds = torch.cat(self.val_voxel_preds, dim=0)
            all_targets = torch.cat(self.val_voxel_targets, dim=0)
            
            # If DDP, gather from all ranks
            # Note: For strict correctness, we should gather. 
            # However, if data is large, this might be heavy. 
            # Assuming simple DDP usage or single GPU for now based on user script.
            # If running DDP, we should gather.
            if dist.is_initialized() and dist.get_world_size() > 1:
                # Need to move to GPU for gather? Usually yes.
                # But it's large. Let's try to do it if needed.
                # For now, calculating local voxel-wise PCC might be acceptable approximation 
                # if data is shuffled, but validation is NOT shuffled.
                # So we MUST gather.
                
                # Move to device for gathering
                all_preds = all_preds.to(self.device)
                all_targets = all_targets.to(self.device)
                
                # Gather
                preds_list = [torch.zeros_like(all_preds) for _ in range(dist.get_world_size())]
                targets_list = [torch.zeros_like(all_targets) for _ in range(dist.get_world_size())]
                dist.all_gather(preds_list, all_preds)
                dist.all_gather(targets_list, all_targets)
                
                all_preds = torch.cat(preds_list, dim=0)
                all_targets = torch.cat(targets_list, dim=0)
            
            # Flatten samples and channels: (N*C, V)
            # We want correlation per voxel across all samples (and channels?)
            # Usually we treat (Batch, Channel) as the sample dimension for voxel-wise correlation
            all_preds = all_preds.view(-1, all_preds.shape[-1]).float() # Ensure float
            all_targets = all_targets.view(-1, all_targets.shape[-1]).float()
            
            # Compute PCC per voxel (dim=0 is sample dimension)
            vx = all_preds - all_preds.mean(dim=0, keepdim=True)
            vy = all_targets - all_targets.mean(dim=0, keepdim=True)
            
            numerator = (vx * vy).sum(dim=0)
            denominator = torch.sqrt((vx ** 2).sum(dim=0)) * torch.sqrt((vy ** 2).sum(dim=0)) + 1e-8
            voxel_pcc = numerator / denominator
            
            avg_voxel_pcc = voxel_pcc.mean()
            
            self.log("val/voxel_pcc", avg_voxel_pcc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=False)
            print(f"Validation Voxel-wise PCC: {avg_voxel_pcc:.4f}")

        self.val_voxel_preds = []
        self.val_voxel_targets = []

    def test_step(self, batch, batch_idx):
        brain_data, _, _ = batch
        x = brain_data.squeeze(1).permute(0, 4, 1, 2, 3).contiguous()
        
        loss, pred, mask = self.model(x, valid_mask_patchified=self.valid_mask_buf, mask_ratio=0.0)
        
        target = self.model.patchify(x)
        pcc, pred_flat, target_flat = self.compute_pcc(pred, target)
        self.test_pcc_list.append(pcc)
        
        # Store for voxel-wise PCC
        self.test_voxel_preds.append(pred_flat.cpu())
        self.test_voxel_targets.append(target_flat.cpu())

    def on_test_epoch_end(self):
        if len(self.test_pcc_list) > 0:
            avg_pcc = torch.stack(self.test_pcc_list).mean()
            
            if dist.is_initialized():
                dist.all_reduce(avg_pcc, op=dist.ReduceOp.SUM)
                avg_pcc /= dist.get_world_size()
            
            log_name = f"{self.test_mode}/pcc"
            self.log(log_name, avg_pcc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=False)
            print(f"\n{self.test_mode} PCC: {avg_pcc:.4f}")
            
        self.test_pcc_list = []
        
        # Aggregate Voxel-wise PCC for Test
        if len(self.test_voxel_preds) > 0:
            all_preds = torch.cat(self.test_voxel_preds, dim=0)
            all_targets = torch.cat(self.test_voxel_targets, dim=0)
            
            if dist.is_initialized() and dist.get_world_size() > 1:
                all_preds = all_preds.to(self.device)
                all_targets = all_targets.to(self.device)
                preds_list = [torch.zeros_like(all_preds) for _ in range(dist.get_world_size())]
                targets_list = [torch.zeros_like(all_targets) for _ in range(dist.get_world_size())]
                dist.all_gather(preds_list, all_preds)
                dist.all_gather(targets_list, all_targets)
                all_preds = torch.cat(preds_list, dim=0)
                all_targets = torch.cat(targets_list, dim=0)
            
            all_preds = all_preds.view(-1, all_preds.shape[-1]).float()
            all_targets = all_targets.view(-1, all_targets.shape[-1]).float()
            
            vx = all_preds - all_preds.mean(dim=0, keepdim=True)
            vy = all_targets - all_targets.mean(dim=0, keepdim=True)
            
            numerator = (vx * vy).sum(dim=0)
            denominator = torch.sqrt((vx ** 2).sum(dim=0)) * torch.sqrt((vy ** 2).sum(dim=0)) + 1e-8
            voxel_pcc = numerator / denominator

            # Calculate P-values
            n_samples = all_preds.shape[0]
            # Clamp PCC to avoid division by zero in t-statistic calculation
            r = torch.clamp(voxel_pcc, -1.0 + 1e-6, 1.0 - 1e-6)
            t_stat = r * torch.sqrt((n_samples - 2) / (1 - r**2))
            
            # Calculate p-value (two-tailed): p = 2 * sf(|t|, df)
            # scipy is used here because torch.distributions.StudentT.cdf is not
            # implemented in some PyTorch versions.
            t_stat_np = torch.abs(t_stat).detach().cpu().numpy()
            p_values_np = 2.0 * student_t.sf(t_stat_np, df=n_samples - 2)
            p_values = torch.from_numpy(p_values_np).to(voxel_pcc.device, dtype=voxel_pcc.dtype)
            
            print(f"{self.test_mode} Voxel PCC shape: {voxel_pcc.shape}")
            # Save voxel_pcc to npy
            np.save(os.path.join(self.log_path, f"{self.test_mode}_voxel_pcc.npy"), voxel_pcc.cpu().numpy())
            # Save voxel_p_value to npy
            np.save(os.path.join(self.log_path, f"{self.test_mode}_voxel_p_value.npy"), p_values.cpu().numpy())
            
            avg_voxel_pcc = voxel_pcc.mean()
            
            log_name = f"{self.test_mode}/voxel_pcc"
            self.log(log_name, avg_voxel_pcc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=False)
            print(f"{self.test_mode} Voxel-wise PCC: {avg_voxel_pcc:.4f}")

        self.test_voxel_preds = []
        self.test_voxel_targets = []

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_mae():
    fix_seed(42)
    torch.set_float32_matmul_precision("high")

    subject_name = 'S06'
    epoch_num = 200 # MAE needs more epochs usually
    batch_size = 64

    print('Loading data..')
    start_time = time()
    # We only need fMRI data, labels don't matter for MAE
    # Added mask_3d return
    Resp_train_volume, Stim_train, Resp_val_volume, Stim_val, Resp_test_volume, Stim_test, mask_3d = \
        prepare_fmri_and_stim_same_movie(subject_name, feat_name='clip_base_img', feat_type='Visual_Model')
    
    # We might also want to test on unseen set
    # Note: prepare_fmri_and_stim_test does not return mask, but we assume it's same subject same mask
    Resp_test_volume_unseen, Stim_test_unseen = prepare_fmri_and_stim_test(subject_name, feat_name='clip_base_img', feat_type='Visual_Model')

    print('Time for loading data: %f' % (time() - start_time) + ' seconds')

    train_set = BrainDataset(Resp_train_volume, Stim_train) # Features are dummy here
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    
    valid_set = BrainDataset(Resp_val_volume, Stim_val)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    test_set = BrainDataset(Resp_test_volume, Stim_test)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    test_unseen_set = BrainDataset(Resp_test_volume_unseen, Stim_test_unseen)
    test_unseen_loader = DataLoader(test_unseen_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Calculate total steps for scheduler
    total_steps = len(train_loader) * epoch_num
    print(f"Total training steps: {total_steps}")

    log_path = ("PATH/TO/project/RetrievalAV/log/MAE/%s_Swin3DMAE_Mask50_voxel_pcc/"
                % (subject_name))
    wandb_path = log_path
    checkpoint_path = os.path.join(log_path, "checkpoints")
    os.makedirs(checkpoint_path, exist_ok=True)

    model = BrainMAE(
        img_size=(72, 96, 96),
        patch_size=(6, 6, 6),
        in_chans=5,
        embed_dim=128,
        window_size=(4, 4, 4),
        mask_ratio=0.50, # Optimized: 0.50
        log_path=log_path,
        valid_mask_volume=mask_3d, # Pass mask to model
        total_steps=total_steps,
        decoder_embed_dim=256, # Increased capacity
        decoder_depths=(4, 4), # Deeper decoder
        # mask_loss_only=False  # Control training loss: False = Compute loss on ALL valid patches
    )

    # Monitor val/pcc instead of val/loss
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        monitor="val/pcc",
        mode="max",
        filename="mae-epoch={epoch:02d}-pcc={val/pcc:.4f}",
        save_top_k=1,
        save_last=True
    )

    wandb_logger = WandbLogger(
        save_dir=wandb_path,
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=f"{subject_name}_Swin3DMAE_Mask50_voxel_pcc"
    )

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        logger=wandb_logger,
        max_epochs=epoch_num,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=20,
        strategy=DDPStrategy(find_unused_parameters=True),
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, train_loader, valid_loader)
    
    # Run test
    model.test_mode = 'test'
    trainer.test(model, test_loader, ckpt_path="best")
    
    # Run test on unseen
    model.test_mode = 'test_unseen'
    trainer.test(model, test_unseen_loader, ckpt_path="best")
    
    wandb.finish()

def test_mae(ckpt_path):
    fix_seed(42)
    torch.set_float32_matmul_precision("high")

    subject_name = 'S02'
    batch_size = 64

    print('Loading data..')
    # We need mask_3d from this function
    Resp_train_volume, Stim_train, Resp_val_volume, Stim_val, Resp_test_volume, Stim_test, mask_3d = \
        prepare_fmri_and_stim_same_movie(subject_name, feat_name='clip_base_img', feat_type='Visual_Model')
    
    Resp_test_volume_unseen, Stim_test_unseen = prepare_fmri_and_stim_test(subject_name, feat_name='clip_base_img', feat_type='Visual_Model')

    test_set = BrainDataset(Resp_test_volume, Stim_test)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    test_unseen_set = BrainDataset(Resp_test_volume_unseen, Stim_test_unseen)
    test_unseen_loader = DataLoader(test_unseen_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Extract log path from ckpt_path
    log_path = os.path.dirname(os.path.dirname(ckpt_path))
    
    model = BrainMAE(
        img_size=(72, 96, 96),
        patch_size=(6, 6, 6),
        in_chans=5,
        embed_dim=128,
        window_size=(4, 4, 4),
        mask_ratio=0.50, 
        log_path=log_path,
        valid_mask_volume=mask_3d,
        total_steps=1000, # Dummy value
        decoder_embed_dim=256,
        decoder_depths=(4, 4),
        # mask_loss_only=False
    )

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        logger=False, 
    )

    print(f"Testing checkpoint: {ckpt_path}")

    # Run test
    model.test_mode = 'test'
    trainer.test(model, test_loader, ckpt_path=ckpt_path)
    
    # Run test on unseen
    model.test_mode = 'test_unseen'
    trainer.test(model, test_unseen_loader, ckpt_path=ckpt_path)

if __name__ == '__main__':
    train_mae()

    # ckpt = "PATH/TO/project/RetrievalAV/log/MAE/S02_Swin3DMAE_Mask50_voxel_pcc/checkpoints/last.ckpt"
    # test_mae(ckpt)
