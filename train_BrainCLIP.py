import os
import random
from time import time
import argparse
import json
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

from dataset import BrainDataset, prepare_fmri_and_stim_test, prepare_fmri_and_stim_same_movie, prepare_stim_same_movie, \
    prepare_stim_test
from visualize import get_valid_voxel_mask_3d
from swin3d_transformer import SwinTransformer3D
# from swin4d_transformer_ver7 import SwinTransformer4D

FEAT_PATH = "PATH/TO/Narrative_Movie_fMRI_Dataset/derivatives/feat/"
WANDB_ENTITY = os.getenv("WANDB_ENTITY", "YOUR_WANDB_ENTITY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "RetrievalAV")


class WarmupScheduler(LambdaLR):
    def __init__(self, optimizer, warmup_steps):
        """
        Warmup learning rate scheduler.
        Args:
            optimizer: The optimizer for which to adjust the learning rate.
            warmup_steps: Number of steps for warmup.
        """

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            return 1.0  # After warmup, keep lr constant at base_lr

        super().__init__(optimizer, lr_lambda)


class BrainCLIP(pl.LightningModule):
    def __init__(
            self,
            img_size=(72, 96, 96, 5),
            in_chans=1,
            embed_dim=96,
            window_size=(4, 4, 4, 5),
            first_window_size=(2, 2, 2, 5),
            patch_size=(6, 6, 6, 1),
            depths=(2, 2, 6),
            num_heads=(3, 6, 12),
            downsample="mergingv2",
            output_dim=1024,
            log_path=None,
            pretrained_mae_path=None,
            brain_encoder_type="3d",  # "3d" or "4d"
    ):
        super().__init__()
        self.brain_encoder_type = brain_encoder_type.lower()

        # if self.brain_encoder_type == "4d":
        #     # SwinTransformer4D: input (B, C, D, H, W, T), img_size 4-tuple (D, H, W, T)
        #     _first_ws = first_window_size if len(first_window_size) == 4 else (first_window_size[0],
        #                                                                        first_window_size[1],
        #                                                                        first_window_size[2], img_size[3])
        #     self.brain_encoder = SwinTransformer4D(
        #         img_size=img_size,
        #         in_chans=in_chans,
        #         embed_dim=embed_dim,
        #         window_size=window_size,
        #         first_window_size=_first_ws,
        #         patch_size=patch_size,
        #         depths=depths,
        #         num_heads=num_heads,
        #         downsample=downsample,
        #         spatial_dims=4,
        #     )
        # else:
        self.brain_encoder = SwinTransformer3D(
            img_size=img_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            window_size=window_size,
            first_window_size=first_window_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            downsample=downsample,
        )

        if pretrained_mae_path is not None:
            print(
                f"Loading pretrained MAE brain encoder from {pretrained_mae_path} (encoder_type={self.brain_encoder_type})...")
            checkpoint = torch.load(pretrained_mae_path, map_location='cpu')
            state_dict = checkpoint.get('state_dict', checkpoint)
            encoder_state_dict = {}
            prefix = "model.encoder."
            for k, v in state_dict.items():
                if k.startswith(prefix):
                    new_k = k[len(prefix):]
                    encoder_state_dict[new_k] = v
            if not encoder_state_dict:
                raise ValueError(
                    f"No keys with prefix '{prefix}' in checkpoint. Ensure the checkpoint is from BrainMAE (3D) or BrainMAE4D (4D).")
            msg = self.brain_encoder.load_state_dict(encoder_state_dict, strict=False)
            print(f"MAE Encoder loaded with message: {msg}")

        feature_dim = getattr(self.brain_encoder, 'num_features', embed_dim * 2 ** (len(depths) - 1))
        self.norm = nn.LayerNorm(feature_dim)
        self.head = nn.Linear(feature_dim, output_dim)
        
        self.temperature = torch.tensor(0.07)
        self.log_path = log_path
        self.test_mode = 'val'  # 'test' # or 'test_ood'


    def configure_optimizers(self):
        # Separate parameters for the encoder (pretrained) and the rest (randomly initialized)
        encoder_params = list(map(id, self.brain_encoder.parameters()))
        head_params = filter(lambda p: id(p) not in encoder_params, self.parameters())

        # Use a lower learning rate for the pretrained encoder and a higher one for the head
        optimizer = torch.optim.AdamW([
            {'params': self.brain_encoder.parameters(), 'lr': 2e-5},  # Lower LR for pretrained encoder
            {'params': head_params, 'lr': 2e-4}  # Higher LR for new layers
        ], weight_decay=1e-2)

        # Use Cosine Annealing Scheduler instead of constant LR after warmup
        # This helps in fine-tuning to settle into a better minima
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }

    def forward(self, x):
        # x: (B, C, D, H, W, T)
        x = self.brain_encoder(x)  # B, C, D, H, W, T
        x = x.flatten(2).transpose(1, 2)  # B, N, C
        x = self.norm(x.mean(dim=1))
        x = self.head(x)
        return x

    def mse_loss(self, brain_embeds, stimuli_embeds):
        """
        Mean Squared Error loss between brain_embeds and stimuli_embeds
        brain_embeds: Tensor of shape (B, D)
        stimuli_embeds: Tensor of shape (B, D)
        """
        brain_embeds = F.normalize(brain_embeds, dim=-1)
        stimuli_embeds = F.normalize(stimuli_embeds, dim=-1)
        return F.mse_loss(brain_embeds, stimuli_embeds)

    def info_nce_loss(self, brain_embeds, stimuli_embeds):
        """
        brain_embeds: Tensor of shape (B, D)
        stimuli_embeds: Tensor of shape (B, D)
        temperature: scaling factor
        """
        # Normalize
        brain_embeds = F.normalize(brain_embeds, dim=-1)
        stimuli_embeds = F.normalize(stimuli_embeds, dim=-1)

        # Similarity matrix (B, B)
        logits = brain_embeds @ stimuli_embeds.T / self.temperature


        labels = torch.arange(len(logits)).to(logits.device)

        # CrossEntropy loss from both directions
        loss_b2s = F.cross_entropy(logits, labels)
        loss_s2b = F.cross_entropy(logits.T, labels)

        return (loss_b2s + loss_s2b) / 2

    def training_step(self, batch, batch_idx):
        brain_data, stimuli_feature, _ = batch  # unpack index but ignore it
        brain_feature = self(brain_data)

        # mse_loss = self.mse_loss(brain_feature, stimuli_feature)

        # 1. Contrastive Loss (InfoNCE)

        cont_loss = self.info_nce_loss(brain_feature, stimuli_feature)

        # Total Loss
        loss = cont_loss

        log_dic = {
            'cont_loss': cont_loss,
        }

        self.log_dict(
            {("train/" + k): float(v) for k, v in log_dic.items()},
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
            sync_dist=True
        )

        self.log(
            "train/temperature",
            float(self.temperature),
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )

        self.log(
            "global_step",
            float(self.global_step),
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log(
            "lr_abs",
            float(lr),
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )

        return loss

    def on_validation_epoch_start(self) -> None:
        self.brain_feature_list = []
        self.stimuli_feature_list = []
        self.index_list = []

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        brain_data, stimuli_feature, index = batch
        brain_feature = self(brain_data)
        self.brain_feature_list.append(brain_feature)
        self.stimuli_feature_list.append(stimuli_feature)
        self.index_list.append(index)

        # Normalize
        brain_feature = F.normalize(brain_feature, dim=-1)
        stimuli_feature = F.normalize(stimuli_feature, dim=-1)
        cos_sim = (brain_feature * stimuli_feature).sum(dim=-1).mean()
        self.log(
            "val/sim",
            cos_sim,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True
        )

    def on_validation_epoch_end(self) -> None:
        final_brain, final_stim = self._gather_epoch_tensors()
        final_brain_norm = F.normalize(final_brain, dim=-1)
        final_stim_norm = F.normalize(final_stim, dim=-1)
        sim_matrix = final_brain_norm @ final_stim_norm.T
        ranks = sim_matrix.argsort(dim=-1, descending=True)
        labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)

        top_vals = []
        for k in (1, 5, 10):
            k_eff = min(k, sim_matrix.size(1))
            correct = (ranks[:, :k_eff] == labels.unsqueeze(1)).any(dim=1)
            top_vals.append(correct.float().mean().item())

        self.log_dict(
            {
                "val/top1": top_vals[0],
                "val/top5": top_vals[1],
                "val/top10": top_vals[2],
            },
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=False,
        )

    def on_test_epoch_start(self) -> None:
        self.brain_feature_list = []
        self.stimuli_feature_list = []
        self.index_list = []

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self) -> None:
        final_brain, final_stim = self._gather_epoch_tensors()
        final_brain_norm = F.normalize(final_brain, dim=-1)
        final_stim_norm = F.normalize(final_stim, dim=-1)
        sim_matrix = final_brain_norm @ final_stim_norm.T
        ranks = sim_matrix.argsort(dim=-1, descending=True)
        labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)

        top_vals = []
        for k in (1, 5, 10):
            k_eff = min(k, sim_matrix.size(1))
            correct = (ranks[:, :k_eff] == labels.unsqueeze(1)).any(dim=1)
            top_vals.append(correct.float().mean().item())

        if self.trainer.is_global_zero:
            np.save(os.path.join(self.log_path, f"{self.test_mode}_brain_features.npy"), final_brain.detach().cpu().numpy())
            if self.test_mode != "val":
                res = {
                    "top1": top_vals[0],
                    "top5": top_vals[1],
                    "top10": top_vals[2],
                    "sim": float((final_brain_norm * final_stim_norm).sum(dim=-1).mean().item()),
                }
                with open(os.path.join(self.log_path, f"{self.test_mode}_acc_results.json"), "w", encoding="utf-8") as f:
                    json.dump(res, f, indent=4)

                topk = 10
                with open(os.path.join(self.log_path, f"{self.test_mode}_top{topk}_retrieval_results.txt"), "w",
                          encoding="utf-8") as f:
                    for i in range(sim_matrix.size(0)):
                        top_indices = ranks[i, :topk]
                        top_scores = sim_matrix[i, top_indices]
                        line = f"{i}: " + ", ".join(
                            f"{int(idx)}({score:.4f})" for idx, score in zip(top_indices, top_scores)
                        )
                        f.write(line + "\n")

        if self.test_mode != "val":
            self.log_dict(
                {
                    f"{self.test_mode}/top1": top_vals[0],
                    f"{self.test_mode}/top5": top_vals[1],
                    f"{self.test_mode}/top10": top_vals[2],
                },
                prog_bar=True,
                logger=True,
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )

    def _gather_epoch_tensors(self):
        local_brain = torch.vstack(self.brain_feature_list)
        local_stim = torch.vstack(self.stimuli_feature_list)
        local_indices = torch.cat(self.index_list)

        all_brain = self.all_gather(local_brain).reshape(-1, local_brain.size(-1))
        all_stim = self.all_gather(local_stim).reshape(-1, local_stim.size(-1))
        all_indices = self.all_gather(local_indices).reshape(-1)

        max_idx = all_indices.max().item()
        dataset_len = max_idx + 1

        final_brain = torch.zeros(dataset_len, all_brain.size(1), device=self.device, dtype=all_brain.dtype)
        final_stim = torch.zeros(dataset_len, all_stim.size(1), device=self.device, dtype=all_stim.dtype)

        final_brain[all_indices] = all_brain
        final_stim[all_indices] = all_stim
        return final_brain, final_stim


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


DEFAULT_VISUAL_FEATURE_CONFIGS = [
    # ("clip_base_img", 512),
    ("clip_large_img", 768),
    # ("siglip_base_img", 768),
    ("siglip_large_img", 1024),
    # ("siglip2_base_img", 768),
    # ("siglip2_large_img", 1024),
    # ("siglip_so400m_img", 1152),
    # ("blip_caption_base_img", 768),
    ("blip_caption_large_img", 1024),
    ("blip2_opt_2.7b_img", 1408),
    # ("dinov2_base", 768),
    ("dinov2_large", 1024),
    # ("dinov3_vitl16", 1024),
    ("imagebind_img", 1024),
    ("Qwen3-VL-Embedding-8B", 4096),
    ("resnet50", 2048),
    # ("convnext_base", 1024),
]

DEFAULT_AUDIO_FEATURE_CONFIGS = [
    ("clap_audio", 512),
    ("whisper", 1280),
    ("imagebind_aud", 1024),
    # ("wav2vec2", 1024),
    # ("ast", 768),
    ("panns", 2048),
    # ("gemini_embedding2_audio", 3072),
]

VALID_ROI_GROUPS = {
    "LVC": ["V1", "V2", "V3"],
    "HVC": ["FFA", "OFA", "PPA", "MT+"],
    "AC": ["AC"],
}

CUSTOM_ROI_MASK_KEYS = {
    "ROI_V": "val_visual_top_mask",
    "ROI_A": "val_audio_top_mask",
    "ROI_S": "val_shared_top_mask",
    "ROI_vld": "val_valid_mask",
}


CUSTOM_ROI_COMPOSITE_KEYS = {
    "ROI_VA": ("val_visual_top_mask", "val_audio_top_mask"),
    "ROI_VS": ("val_visual_top_mask", "val_shared_top_mask"),
    "ROI_AS": ("val_audio_top_mask", "val_shared_top_mask"),
}

CUSTOM_ROI_MASK_DIR_TEMPLATE = (
    "./encoding/encoding_av/"
    "{subject_name}_vpa_mean_across_pairs"
)

FULL_VOLUME_SHAPE = (72, 96, 96)
FULL_VOLUME_SIZE = int(np.prod(FULL_VOLUME_SHAPE))

_CUSTOM_ROI_ALL_KEYS = frozenset(CUSTOM_ROI_MASK_KEYS) | frozenset(CUSTOM_ROI_COMPOSITE_KEYS)


def _normalize_roi_mode(roi):
    roi = str(roi).strip()
    if not roi:
        return "full"
    if roi.lower() == "full":
        return "full"

    if roi not in VALID_ROI_GROUPS and roi not in _CUSTOM_ROI_ALL_KEYS:
        raise ValueError(
            "Unsupported roi mode: {0}. Expected one of: full, LVC, HVC, AC, "
            "ROI_V, ROI_A, ROI_S, ROI_vld, ROI_VA, ROI_VS, ROI_AS.".format(roi)
        )
    return roi


def _npz_array_to_mask_3d(loaded_mask, subject_name, roi_mode, mask_key):
    loaded_mask = loaded_mask.astype(bool)
    if loaded_mask.shape == FULL_VOLUME_SHAPE:
        return loaded_mask
    if loaded_mask.size == FULL_VOLUME_SIZE:
        return loaded_mask.reshape(FULL_VOLUME_SHAPE)
    cortical_mask_3d, _, _, _ = get_valid_voxel_mask_3d(
        subject_name=subject_name,
        roi_mode="full",
    )
    cortical_voxel_count = int(np.count_nonzero(cortical_mask_3d))
    if loaded_mask.size != cortical_voxel_count:
        raise ValueError(
            f"Custom ROI mask shape mismatch for roi={roi_mode}, key={mask_key}, subject={subject_name}: "
            f"mask shape={loaded_mask.shape}, size={loaded_mask.size}, "
            f"expected full size={FULL_VOLUME_SIZE} or cortical size={cortical_voxel_count}."
        )
    mask_3d = np.zeros(FULL_VOLUME_SIZE, dtype=bool)
    mask_3d[cortical_mask_3d.reshape(-1)] = loaded_mask.reshape(-1)
    return mask_3d.reshape(FULL_VOLUME_SHAPE)


def _load_custom_roi_mask_3d(subject_name, roi_mode):
    if roi_mode in CUSTOM_ROI_COMPOSITE_KEYS:
        mask_keys = CUSTOM_ROI_COMPOSITE_KEYS[roi_mode]
    elif roi_mode in CUSTOM_ROI_MASK_KEYS:
        mask_keys = (CUSTOM_ROI_MASK_KEYS[roi_mode],)
    else:
        raise ValueError(f"Unsupported custom ROI mode: {roi_mode}")

    base_dir = CUSTOM_ROI_MASK_DIR_TEMPLATE.format(subject_name=subject_name)
    # TODO top p
    candidate_npz_path = os.path.join(base_dir, "vpa_topk_masks_top0.2.npz")

    if not os.path.exists(candidate_npz_path):
        raise FileNotFoundError(
            f"Cannot load custom ROI for roi={roi_mode}, subject={subject_name}. "
            f"Expected file: {candidate_npz_path}"
        )

    masks_3d = []
    with np.load(candidate_npz_path) as npz_data:
        for mask_key in mask_keys:
            if mask_key not in npz_data:
                raise FileNotFoundError(
                    f"Cannot find '{mask_key}' for roi={roi_mode}, subject={subject_name}. "
                    f"Expected file: {candidate_npz_path}"
                )
            arr = np.asarray(npz_data[mask_key])
            masks_3d.append(_npz_array_to_mask_3d(arr, subject_name, roi_mode, mask_key))

    mask_3d = masks_3d[0] if len(masks_3d) == 1 else np.logical_or.reduce(masks_3d)
    keys_desc = "|".join(mask_keys)
    print(
        f"[ROI] loaded custom ROI mask: mode={roi_mode}, key(s)={keys_desc}, "
        f"path={candidate_npz_path}, voxels={int(np.count_nonzero(mask_3d))}"
    )
    return mask_3d


def _apply_voxel_mask_to_resp(resp_volume, valid_mask_3d):
    mask_6d = valid_mask_3d.astype(resp_volume.dtype)[None, None, :, :, :, None]
    resp_volume *= mask_6d
    return resp_volume


def load_subject_fmri_once(subject_name, ref_feat_type="Visual_Model", ref_feat_name="clip_base_img", roi="full"):
    resp_train, _, resp_val, _, resp_test, _, _ = prepare_fmri_and_stim_same_movie(
        subject_name, feat_name=ref_feat_name, feat_type=ref_feat_type, feat_path=FEAT_PATH
    )
    resp_test_unseen, _ = prepare_fmri_and_stim_test(
        subject_name, feat_name=ref_feat_name, feat_type=ref_feat_type, feat_path=FEAT_PATH
    )
    roi_mode = _normalize_roi_mode(roi)
    if roi_mode in _CUSTOM_ROI_ALL_KEYS:
        valid_mask_3d = _load_custom_roi_mask_3d(subject_name, roi_mode)
        roi_targets = [roi_mode]
        if roi_mode in CUSTOM_ROI_COMPOSITE_KEYS:
            matched_roi_names = ["|".join(CUSTOM_ROI_COMPOSITE_KEYS[roi_mode])]
        else:
            matched_roi_names = [CUSTOM_ROI_MASK_KEYS[roi_mode]]
    else:
        valid_mask_3d, roi_mode, roi_targets, matched_roi_names = get_valid_voxel_mask_3d(
            subject_name=subject_name,
            roi_mode=roi_mode,
        )
    resp_train = _apply_voxel_mask_to_resp(resp_train, valid_mask_3d)
    resp_val = _apply_voxel_mask_to_resp(resp_val, valid_mask_3d)
    resp_test = _apply_voxel_mask_to_resp(resp_test, valid_mask_3d)
    resp_test_unseen = _apply_voxel_mask_to_resp(resp_test_unseen, valid_mask_3d)

    valid_voxel_count = int(np.count_nonzero(valid_mask_3d))
    if roi_mode == "full":
        print(f"[ROI] mode=full, valid_voxels={valid_voxel_count}")
    else:
        print(
            f"[ROI] mode={roi_mode}, roi_targets={roi_targets}, "
            f"matched={matched_roi_names}, valid_voxels={valid_voxel_count}"
        )

    return {
        "Resp_train_volume": resp_train,
        "Resp_val_volume": resp_val,
        "Resp_test_volume": resp_test,
        "Resp_test_volume_unseen": resp_test_unseen,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train BrainCLIP with configurable feature settings.")
    parser.add_argument("--subject_name", type=str, default="S01", help="Subject ID, e.g. S01/S02")
    parser.add_argument("--feat_type", type=str, default="Visual_Model",
                        help="Feature category, e.g. Visual_Model/Audio_Model/Text_Model")
    parser.add_argument("--feat_name", type=str, default="clip_base_img", help="Feature name directory under feat_type")
    parser.add_argument("--output_dim", type=int, default=512, help="Projection output dimension")
    parser.add_argument(
        "--run_av_grid_once_fmri",
        action="store_true",
        help="Run all audio-visual feature pairs in both directions with one-time fMRI loading.",
    )
    parser.add_argument(
        "--run_modality",
        type=str,
        default="VA",
        choices=["V", "A", "VA"],
        help="Run modality for AV grid and single-modal grid: V (visual only), A (audio only), or VA (both).",
    )
    parser.add_argument(
        "--run_single_modal_grid_once_fmri",
        action="store_true",
        help="Run all visual/audio configs in single-modality mode with one-time fMRI loading.",
    )
    parser.add_argument("--num_workers_train", type=int, default=4, help="Num workers for train dataloader")
    parser.add_argument("--num_workers_eval", type=int, default=1, help="Num workers for eval/test dataloaders")
    parser.add_argument(
        "--roi",
        type=str,
        default="full",
        help=(
            "ROI mode: full (all cortical voxels), LVC (V1/V2/V3), HVC (FFA/OFA/PPA/MT+), AC (AC), "
            "ROI_V/ROI_A/ROI_S/ROI_vld (single npz keys), ROI_VA/ROI_VS/ROI_AS (union of two keys, VPA npz)."
        ),
    )
    return parser.parse_args()


def train_brain_clip(feat_type, feat_name, output_dim, shared_fmri=None,
                     subject_name="S01", roi="full"):

    clip_variant_name = "BrainCLIP"

    print(
        f"Training variant: {clip_variant_name} "
    )
    fix_seed(42)
    torch.set_float32_matmul_precision("high")

    epoch_num = 20  # Reduced from 100 for fine-tuning
    # Choose brain encoder: "3d" (SwinTransformer3D) or "4d" (SwinTransformer4D)
    brain_encoder_type = "3d"
    # Path to pretrained MAE checkpoint (3D: BrainMAE ckpt; 4D: BrainMAE4D ckpt)
    pretrained_mae_path = (
        f"PATH/TO/Narrative_Movie_fMRI_Dataset/pretrained_mae_ckpt/"
        f"{args.subject_name}_Swin3DMAE_Mask50_voxel_pcc/last.ckpt"
    )

    print('Loading data..')
    start_time = time()
    if shared_fmri is None:
        shared_fmri = load_subject_fmri_once(
            subject_name=subject_name,
            ref_feat_type=feat_type,
            ref_feat_name=feat_name,
            roi=roi,
        )
    Resp_train_volume = shared_fmri["Resp_train_volume"]
    Resp_val_volume = shared_fmri["Resp_val_volume"]
    Resp_test_volume = shared_fmri["Resp_test_volume"]
    Resp_test_volume_unseen = shared_fmri["Resp_test_volume_unseen"]
    Stim_train, Stim_val, Stim_test = prepare_stim_same_movie(
        feat_name=feat_name, feat_type=feat_type, feat_path=FEAT_PATH
    )
    Stim_test_unseen = prepare_stim_test(
        feat_name=feat_name, feat_type=feat_type, feat_path=FEAT_PATH
    )

    print('Time for loading data: %f' % (time() - start_time) + ' seconds')

    train_set = BrainDataset(Resp_train_volume, Stim_train)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=args.num_workers_train)
    valid_set = BrainDataset(Resp_val_volume, Stim_val)
    valid_loader = DataLoader(valid_set, batch_size=64, shuffle=False, num_workers=args.num_workers_eval)
    test_set = BrainDataset(Resp_test_volume, Stim_test)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=args.num_workers_eval)
    test_unseen_set = BrainDataset(Resp_test_volume_unseen, Stim_test_unseen)
    test_unseen_loader = DataLoader(test_unseen_set, batch_size=64, shuffle=False, num_workers=args.num_workers_eval)

    exp_name = f"{clip_variant_name}-L_{subject_name}_{epoch_num}epoch_{feat_name}"
    if str(roi).lower() != "full":
        exp_name = f"{exp_name}_{roi}"
        if roi not in VALID_ROI_GROUPS and roi != "ROI_vld":
            # TODO top p
            exp_name = f"{exp_name}_top0.2"

    log_path = f"log/{feat_type}/{exp_name}/"
    wandb_path = log_path
    if os.path.isfile(os.path.join(log_path, "test_unseen_acc_results.json")):
        print(
            f"[BrainCLIP] Skip training: test_unseen_acc_results.json already exists under {log_path}"
        )
        return
    checkpoint_path = os.path.join(log_path, "checkpoints")
    os.makedirs(checkpoint_path, exist_ok=True)

    model = BrainCLIP(
        log_path=log_path,
        output_dim=output_dim,
        embed_dim=128,
        num_heads=(4, 8, 16),
        first_window_size=(2, 2, 2, 5),  # for 4D: last dim = time; for 3D only first 3 used
        pretrained_mae_path=pretrained_mae_path,
        brain_encoder_type=brain_encoder_type,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        monitor="val/top10",
        mode="max",
        filename="checkpoint_step={global_step:.0f}-sim={val/sim:.3f}",
        auto_insert_metric_name=False,
        # every_n_train_steps=save_checkpoint_every_n_steps,
        # every_n_epochs=10,
        save_top_k=1,
        save_last=False,
        save_weights_only=True
    )

    wandb_logger = WandbLogger(
        save_dir=wandb_path,
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=f"[{feat_type}]{exp_name}"
    )

    import time as time_module
    max_retries = 5
    for attempt in range(max_retries):
        try:
            _ = wandb_logger.experiment
            print(f"[Wandb] Successfully initialized on attempt {attempt + 1}")
            break
        except Exception as e:
            print(f"[Wandb] Initialization failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print("Retrying in 30 seconds...")
                time_module.sleep(30)
            else:
                raise e

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = 1
    strategy = "auto"
    if accelerator == "gpu" and torch.cuda.device_count() > 1:
        devices = torch.cuda.device_count()
        strategy = DDPStrategy(find_unused_parameters=False)

    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        logger=wandb_logger,
        max_epochs=epoch_num,
        num_sanity_val_steps=0,
        # limit_val_batches=10,
        check_val_every_n_epoch=2,
        strategy=strategy,
        callbacks=[checkpoint_callback],
        # gradient_clip_val=1.0
    )

    trainer.fit(model, train_loader, valid_loader)
    trainer.test(model, valid_loader, ckpt_path="best")
    model.test_mode = 'test'
    trainer.test(model, test_loader, ckpt_path="best")
    model.test_mode = 'test_unseen'
    trainer.test(model, test_unseen_loader, ckpt_path="best")
    wandb.finish()


def train_all_av_pairs_with_single_fmri(
        subject_name="S01",
        roi="full",
        run_modality="VA",
        visual_feature_configs=None,
        audio_feature_configs=None,
):
    visual_feature_configs = visual_feature_configs or DEFAULT_VISUAL_FEATURE_CONFIGS
    audio_feature_configs = audio_feature_configs or DEFAULT_AUDIO_FEATURE_CONFIGS
    print("[AV-GRID] Loading shared fMRI once...")
    shared_fmri = load_subject_fmri_once(subject_name=subject_name, roi=roi)
    per_direction_jobs = len(visual_feature_configs) * len(audio_feature_configs)
    run_visual_primary = run_modality in ("V", "VA")
    run_audio_primary = run_modality in ("A", "VA")
    total_jobs = int(run_visual_primary) * per_direction_jobs + int(run_audio_primary) * per_direction_jobs
    job_id = 0

    if run_visual_primary:
        for aux_audio_name, _ in audio_feature_configs:
            for vis_name, vis_dim in visual_feature_configs:
                job_id += 1
                print(
                    f"[AV-GRID][{job_id}/{total_jobs}] primary=Visual_Model/{vis_name}, "
                    f"aux=Audio_Model/{aux_audio_name}"
                )
                train_brain_clip(
                    feat_type="Visual_Model",
                    feat_name=vis_name,
                    output_dim=vis_dim,
                    subject_name=subject_name,
                    shared_fmri=shared_fmri,
                    roi=roi
                )

    if run_audio_primary:
        for aux_visual_name, _ in visual_feature_configs:
            for aud_name, aud_dim in audio_feature_configs:
                job_id += 1
                print(
                    f"[AV-GRID][{job_id}/{total_jobs}] primary=Audio_Model/{aud_name}, "
                    f"aux=Visual_Model/{aux_visual_name}"
                )
                train_brain_clip(
                    feat_type="Audio_Model",
                    feat_name=aud_name,
                    output_dim=aud_dim,
                    subject_name=subject_name,
                    shared_fmri=shared_fmri,
                    roi=roi,
                )


def train_all_single_modal_with_single_fmri(
        subject_name="S01",
        roi="full",
        run_modality="VA",
        visual_feature_configs=None,
        audio_feature_configs=None,
):
    visual_feature_configs = visual_feature_configs or DEFAULT_VISUAL_FEATURE_CONFIGS
    audio_feature_configs = audio_feature_configs or DEFAULT_AUDIO_FEATURE_CONFIGS
    run_visual_primary = run_modality in ("V", "VA")
    run_audio_primary = run_modality in ("A", "VA")
    print("[SINGLE-GRID] Loading shared fMRI once...")
    shared_fmri = load_subject_fmri_once(subject_name=subject_name, roi=roi)
    total_jobs = (
        int(run_visual_primary) * len(visual_feature_configs)
        + int(run_audio_primary) * len(audio_feature_configs)
    )
    job_id = 0

    if run_visual_primary:
        for vis_name, vis_dim in visual_feature_configs:
            job_id += 1
            print(f"[SINGLE-GRID][{job_id}/{total_jobs}] Visual_Model/{vis_name}")
            train_brain_clip(
                feat_type="Visual_Model",
                feat_name=vis_name,
                output_dim=vis_dim,
                subject_name=subject_name,
                shared_fmri=shared_fmri,
                roi=roi,
            )

    if run_audio_primary:
        for aud_name, aud_dim in audio_feature_configs:
            job_id += 1
            print(f"[SINGLE-GRID][{job_id}/{total_jobs}] Audio_Model/{aud_name}")
            train_brain_clip(
                feat_type="Audio_Model",
                feat_name=aud_name,
                output_dim=aud_dim,
                subject_name=subject_name,
                shared_fmri=shared_fmri,
                roi=roi,
            )


if __name__ == '__main__':
    args = parse_args()
    if args.run_single_modal_grid_once_fmri:
        train_all_single_modal_with_single_fmri(
            subject_name=args.subject_name,
            roi=args.roi,
            run_modality=args.run_modality,
        )
    elif args.run_av_grid_once_fmri:
        train_all_av_pairs_with_single_fmri(
            subject_name=args.subject_name,
            roi=args.roi,
            run_modality=args.run_modality,
        )
    else:
        train_brain_clip(
            feat_type=args.feat_type,
            feat_name=args.feat_name,
            output_dim=args.output_dim,
            subject_name=args.subject_name,
            roi=args.roi,
        )
