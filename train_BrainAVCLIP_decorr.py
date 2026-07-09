"""Separate-encoder joint training with cross-modal hard negative InfoNCE.

Two independent Swin backbones (one for visual, one for audio) are trained jointly. Each encoder can read from a different ROI (enabling the ROI_V+A protocol). During training, each branch uses the partner branch's current decision scores to up-weight shared hard negatives (cross-modal hard negative samples), which encourages complementary decision boundaries for late fusion.

Supported optimisation mode (controlled by flags):
  * Cross-modal hard negative InfoNCE (default ON)
"""

import argparse
import json
import os
import random
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
import wandb

from dataset import (
    prepare_fmri_and_stim_same_movie,
    prepare_fmri_and_stim_test,
    prepare_stim_same_movie,
    prepare_stim_test,
)
from visualize import get_valid_voxel_mask_3d
from retrieval_av_analysis.util import topk_retrieval_reference_similarity
from swin3d_transformer import SwinTransformer3D
# from swin4d_transformer_ver7 import SwinTransformer4D

from train_BrainCLIP import (
    DEFAULT_VISUAL_FEATURE_CONFIGS,
    DEFAULT_AUDIO_FEATURE_CONFIGS,
    FEAT_PATH,
    VALID_ROI_GROUPS,
    CUSTOM_ROI_MASK_KEYS,
    CUSTOM_ROI_COMPOSITE_KEYS,
    _CUSTOM_ROI_ALL_KEYS,
    _normalize_roi_mode,
    _load_custom_roi_mask_3d,
    _apply_voxel_mask_to_resp,
    FULL_VOLUME_SHAPE,
)

WANDB_ENTITY = os.getenv("WANDB_ENTITY", "YOUR_WANDB_ENTITY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "RetrievalAV")


# ---------------------------------------------------------------------------
#  Dataset: carries two (possibly different) brain inputs
# ---------------------------------------------------------------------------
class BrainDualROIDataset(Dataset):
    """Each sample: (brain_visual, brain_audio, stim_visual, stim_audio, index)."""

    def __init__(self, brain_visual, brain_audio, stim_visual, stim_audio):
        n = len(brain_visual)
        assert n == len(brain_audio) == len(stim_visual) == len(stim_audio)
        self.brain_visual = brain_visual
        self.brain_audio = brain_audio
        self.stim_visual = stim_visual
        self.stim_audio = stim_audio

    def __getitem__(self, idx):
        return (
            self.brain_visual[idx],
            self.brain_audio[idx],
            self.stim_visual[idx],
            self.stim_audio[idx],
            idx,
        )

    def __len__(self):
        return len(self.brain_visual)


# ---------------------------------------------------------------------------
#  Model
# ---------------------------------------------------------------------------
def _build_swin_encoder(brain_encoder_type, img_size, embed_dim, window_size,
                        first_window_size, patch_size, depths, num_heads,
                        downsample, in_chans=1):
    # if brain_encoder_type == "4d":
    #     _fws = (first_window_size if len(first_window_size) == 4
    #             else (*first_window_size[:3], img_size[3]))
    #     return SwinTransformer4D(
    #         img_size=img_size, in_chans=in_chans, embed_dim=embed_dim,
    #         window_size=window_size, first_window_size=_fws,
    #         patch_size=patch_size, depths=depths, num_heads=num_heads,
    #         downsample=downsample, spatial_dims=4,
    #     )
    return SwinTransformer3D(
        img_size=img_size, in_chans=in_chans, embed_dim=embed_dim,
        window_size=window_size, first_window_size=first_window_size,
        patch_size=patch_size, depths=depths, num_heads=num_heads,
        downsample=downsample,
    )


def _load_mae_weights(encoder, mae_path):
    if not mae_path:
        return
    print(f"Loading pretrained MAE encoder from: {mae_path}")
    ckpt = torch.load(mae_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    prefix = "model.encoder."
    esd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    if not esd:
        raise ValueError(f"No keys with prefix '{prefix}' in checkpoint.")
    msg = encoder.load_state_dict(esd, strict=False)
    print(f"MAE load result: {msg}")


def _load_ckpt_state_dict(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    return ckpt.get("state_dict", ckpt)


def _resolve_single_modal_ckpt(exp_dir):
    """
    Resolve latest modified .ckpt from a single-modality experiment directory.
    """
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    files = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
    if not files:
        raise FileNotFoundError(f"No .ckpt files found under: {ckpt_dir}")

    files_abs = [os.path.join(ckpt_dir, f) for f in files]
    files_abs.sort(key=os.path.getmtime)
    return files_abs[-1]


def _single_modal_roi_suffix(roi):
    roi = str(roi).strip()
    if roi.lower() == "full":
        return ""
    suffix = f"_{roi}"
    if roi in {"ROI_V", "ROI_A", "ROI_VA"}:
        suffix += "_top0.2"
    return suffix


def _build_single_modal_exp_dir(subject_name, feat_type, feat_name, roi):
    exp_name = f"BrainCLIP-L_{subject_name}_20epoch_{feat_name}{_single_modal_roi_suffix(roi)}"
    return os.path.join("log", feat_type, exp_name)


def _load_single_modal_branch_weights(model, visual_ckpt_path, audio_ckpt_path):
    """
    Load single-modality BrainCLIP checkpoint weights into AV model branches.
    Expected source keys:
      - brain_encoder.*
      - norm.*
      - head.*
    """
    vis_sd = _load_ckpt_state_dict(visual_ckpt_path)
    aud_sd = _load_ckpt_state_dict(audio_ckpt_path)

    vis_enc_sd = {k[len("brain_encoder."):]: v for k, v in vis_sd.items() if k.startswith("brain_encoder.")}
    vis_norm_sd = {k[len("norm."):]: v for k, v in vis_sd.items() if k.startswith("norm.")}
    vis_head_sd = {k[len("head."):]: v for k, v in vis_sd.items() if k.startswith("head.")}

    aud_enc_sd = {k[len("brain_encoder."):]: v for k, v in aud_sd.items() if k.startswith("brain_encoder.")}
    aud_norm_sd = {k[len("norm."):]: v for k, v in aud_sd.items() if k.startswith("norm.")}
    aud_head_sd = {k[len("head."):]: v for k, v in aud_sd.items() if k.startswith("head.")}

    if not vis_enc_sd or not aud_enc_sd:
        raise ValueError(
            "Failed to parse single-modality checkpoint keys. "
            "Expected keys starting with 'brain_encoder.'."
        )

    msg_ve = model.visual_encoder.load_state_dict(vis_enc_sd, strict=False)
    msg_vn = model.vis_norm.load_state_dict(vis_norm_sd, strict=False)
    msg_vh = model.vis_head.load_state_dict(vis_head_sd, strict=False)

    msg_ae = model.audio_encoder.load_state_dict(aud_enc_sd, strict=False)
    msg_an = model.aud_norm.load_state_dict(aud_norm_sd, strict=False)
    msg_ah = model.aud_head.load_state_dict(aud_head_sd, strict=False)

    print(f"[WarmStart] visual encoder load: {msg_ve}")
    print(f"[WarmStart] visual norm load: {msg_vn}")
    print(f"[WarmStart] visual head load: {msg_vh}")
    print(f"[WarmStart] audio encoder load: {msg_ae}")
    print(f"[WarmStart] audio norm load: {msg_an}")
    print(f"[WarmStart] audio head load: {msg_ah}")


def load_default_clip_clap_ref_features(feat_path):
    _, clip_val_ref, clip_test_ref = prepare_stim_same_movie(
        feat_name="clip_large_img",
        feat_type="Visual_Model",
        feat_path=feat_path,
    )
    clip_unseen_ref = prepare_stim_test(
        feat_name="clip_large_img",
        feat_type="Visual_Model",
        feat_path=feat_path,
    )
    clip_ref_features = {
        "val": clip_val_ref,
        "test": clip_test_ref,
        "test_unseen": clip_unseen_ref,
    }

    _, clap_val_ref, clap_test_ref = prepare_stim_same_movie(
        feat_name="clap_audio",
        feat_type="Audio_Model",
        feat_path=feat_path,
    )
    clap_unseen_ref = prepare_stim_test(
        feat_name="clap_audio",
        feat_type="Audio_Model",
        feat_path=feat_path,
    )
    clap_ref_features = {
        "val": clap_val_ref,
        "test": clap_test_ref,
        "test_unseen": clap_unseen_ref,
    }
    return clip_ref_features, clap_ref_features


class BrainAVDecorrCLIP(pl.LightningModule):
    def __init__(
        self,
        img_size=(72, 96, 96, 5),
        in_chans=1,
        embed_dim=128,
        window_size=(4, 4, 4, 5),
        first_window_size=(2, 2, 2, 5),
        patch_size=(6, 6, 6, 1),
        depths=(2, 2, 6),
        num_heads=(4, 8, 16),
        downsample="mergingv2",
        visual_output_dim=512,
        audio_output_dim=512,
        log_path=None,
        pretrained_mae_path=None,
        brain_encoder_type="3d",
        temperature=0.07,
        ref_sim_top_k=1,
        fine_tune_visual=True,
        fine_tune_audio=True,
        # --- asymmetric co-error guidance knobs ---
        use_decorr=True,
        co_error_gamma_v=0.0,
        co_error_gamma_a=2.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        enc_kwargs = dict(
            brain_encoder_type=brain_encoder_type.lower(),
            img_size=img_size, embed_dim=embed_dim, window_size=window_size,
            first_window_size=first_window_size, patch_size=patch_size,
            depths=depths, num_heads=num_heads, downsample=downsample,
            in_chans=in_chans,
        )
        self.visual_encoder = _build_swin_encoder(**enc_kwargs)
        self.audio_encoder = _build_swin_encoder(**enc_kwargs)

        _load_mae_weights(self.visual_encoder, pretrained_mae_path)
        _load_mae_weights(self.audio_encoder, pretrained_mae_path)

        feature_dim = getattr(
            self.visual_encoder, "num_features",
            embed_dim * 2 ** (len(depths) - 1),
        )
        self.vis_norm = nn.LayerNorm(feature_dim)
        self.aud_norm = nn.LayerNorm(feature_dim)
        self.vis_head = nn.Linear(feature_dim, visual_output_dim)
        self.aud_head = nn.Linear(feature_dim, audio_output_dim)

        self.temperature = float(temperature)
        self.ref_sim_top_k = int(ref_sim_top_k)
        self.log_path = log_path
        self.test_mode = "val"
        self.fine_tune_visual = bool(fine_tune_visual)
        self.fine_tune_audio = bool(fine_tune_audio)

        self.use_decorr = use_decorr
        self.co_error_gamma_v = float(co_error_gamma_v)
        self.co_error_gamma_a = float(co_error_gamma_a)

        self.alpha_grid_values = [i * 0.05 for i in range(21)]
        self.best_alpha_by_topk = {"top1": 0.5, "top5": 0.5, "top10": 0.5}
        self.clip_ref_features_by_mode = {}
        self.clap_ref_features_by_mode = {}

        if not self.fine_tune_visual:
            for p in self.visual_encoder.parameters():
                p.requires_grad = False
            for p in self.vis_norm.parameters():
                p.requires_grad = False
            for p in self.vis_head.parameters():
                p.requires_grad = False
        if not self.fine_tune_audio:
            for p in self.audio_encoder.parameters():
                p.requires_grad = False
            for p in self.aud_norm.parameters():
                p.requires_grad = False
            for p in self.aud_head.parameters():
                p.requires_grad = False

    def set_reference_features(self, clip_ref_features, clap_ref_features):
        self.clip_ref_features_by_mode = {
            k: torch.as_tensor(v, dtype=torch.float32) for k, v in clip_ref_features.items()
        }
        self.clap_ref_features_by_mode = {
            k: torch.as_tensor(v, dtype=torch.float32) for k, v in clap_ref_features.items()
        }

    def _get_reference_features_for_current_mode(self):
        mode = self.test_mode if self.test_mode in self.clip_ref_features_by_mode else "val"
        if mode not in self.clip_ref_features_by_mode or mode not in self.clap_ref_features_by_mode:
            raise RuntimeError(
                f"Reference features are not ready for mode='{mode}'. "
                "Call set_reference_features(...) before validation/test."
            )
        return self.clip_ref_features_by_mode[mode], self.clap_ref_features_by_mode[mode]

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------
    def _encode(self, encoder, norm, head, x):
        feat = encoder(x)
        feat = feat.flatten(2).transpose(1, 2)
        feat = norm(feat.mean(dim=1))
        return head(feat)

    def forward(self, brain_v, brain_a):
        z_v = self._encode(self.visual_encoder, self.vis_norm, self.vis_head, brain_v)
        z_a = self._encode(self.audio_encoder, self.aud_norm, self.aud_head, brain_a)
        return z_v, z_a

    # ------------------------------------------------------------------
    #  Losses
    # ------------------------------------------------------------------
    def info_nce_loss(self, pred, target):
        pred = F.normalize(pred, dim=-1)
        target = F.normalize(target, dim=-1)
        logits = pred @ target.T / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        return 0.5 * (F.cross_entropy(logits, labels)
                       + F.cross_entropy(logits.T, labels))

    def cross_modal_hard_negative_info_nce_loss(self, pred, target, partner_logits, gamma, eps=1e-8):
        """InfoNCE with co-error weighting from partner-branch decision scores."""
        pred = F.normalize(pred, dim=-1)
        target = F.normalize(target, dim=-1)
        logits = pred @ target.T / self.temperature
        bsz = logits.size(0)
        if partner_logits is None or bsz <= 1 or gamma <= 0.0:
            labels = torch.arange(bsz, device=logits.device)
            return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))

        with torch.no_grad():
            partner_probs = F.softmax(partner_logits.detach(), dim=-1)
            eye = torch.eye(bsz, device=logits.device, dtype=torch.bool)
            partner_probs = partner_probs.masked_fill(eye, 0.0)
            # If partner already gives a high score to a negative, this negative is risky for fusion.
            weights = 1.0 + float(gamma) * (partner_probs * bsz)
            log_w = torch.log(weights + eps).masked_fill(eye, 0.0)

        weighted_logits = logits + log_w
        labels = torch.arange(bsz, device=logits.device)
        return 0.5 * (F.cross_entropy(weighted_logits, labels)
                      + F.cross_entropy(weighted_logits.T, labels))

    # ------------------------------------------------------------------
    #  Optimiser
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        # Constant, symmetric fine-tuning LR across visual/audio branches.
        lr_encoder = 1e-6
        lr_head = 5e-6
        opts = []
        if self.fine_tune_visual:
            opt_v = torch.optim.AdamW([
                {"params": list(self.visual_encoder.parameters()), "lr": lr_encoder},
                {"params": list(self.vis_norm.parameters()) + list(self.vis_head.parameters()), "lr": lr_head},
            ], weight_decay=1e-2)
            opts.append(opt_v)

        if self.fine_tune_audio:
            opt_a = torch.optim.AdamW([
                {"params": list(self.audio_encoder.parameters()), "lr": lr_encoder},
                {"params": list(self.aud_norm.parameters()) + list(self.aud_head.parameters()), "lr": lr_head},
            ], weight_decay=1e-2)
            opts.append(opt_a)
        return opts

    # ------------------------------------------------------------------
    #  Training
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        brain_v, brain_a, stim_v, stim_a, _ = batch
        opts = self.optimizers()
        if isinstance(opts, (list, tuple)):
            opt_list = list(opts)
        elif opts is None:
            opt_list = []
        else:
            opt_list = [opts]

        idx = 0
        opt_v = opt_list[idx] if self.fine_tune_visual and idx < len(opt_list) else None
        if self.fine_tune_visual:
            idx += 1
        opt_a = opt_list[idx] if self.fine_tune_audio and idx < len(opt_list) else None

        if opt_v is not None:
            opt_v.zero_grad()
        if opt_a is not None:
            opt_a.zero_grad()

        z_v, z_a = self(brain_v, brain_a)
        zv_n = F.normalize(z_v, dim=-1)
        za_n = F.normalize(z_a, dim=-1)
        sv_n = F.normalize(stim_v, dim=-1)
        sa_n = F.normalize(stim_a, dim=-1)
        logits_v = zv_n @ sv_n.T / self.temperature
        logits_a = za_n @ sa_n.T / self.temperature
        zero = torch.zeros((), device=z_v.device, dtype=z_v.dtype)
        loss_v = zero
        loss_a = zero

        if self.fine_tune_visual:
            if self.use_decorr:
                loss_v = self.cross_modal_hard_negative_info_nce_loss(
                    z_v, stim_v, logits_a, gamma=self.co_error_gamma_v
                )
            else:
                loss_v = self.info_nce_loss(z_v, stim_v)

        if self.fine_tune_audio:
            if self.use_decorr:
                loss_a = self.cross_modal_hard_negative_info_nce_loss(
                    z_a, stim_a, logits_v, gamma=self.co_error_gamma_a
                )
            else:
                loss_a = self.info_nce_loss(z_a, stim_a)

        active_loss_count = int(self.fine_tune_visual) + int(self.fine_tune_audio)
        loss = (loss_v + loss_a) / max(active_loss_count, 1)

        if self.fine_tune_visual and self.fine_tune_audio:
            self.manual_backward(loss_v, retain_graph=True)
            self.manual_backward(loss_a)
        elif self.fine_tune_visual:
            self.manual_backward(loss_v)
        elif self.fine_tune_audio:
            self.manual_backward(loss_a)

        if opt_v is not None:
            opt_v.step()
        if opt_a is not None:
            opt_a.step()

        log_dict = {
            "train/loss_vis": loss_v.detach(),
            "train/loss_aud": loss_a.detach(),
            "train/lr_visual": torch.tensor(
                (opt_v.param_groups[0]["lr"] if opt_v is not None else 0.0),
                device=loss.device, dtype=loss.dtype
            ),
            "train/lr_audio": torch.tensor(
                (opt_a.param_groups[0]["lr"] if opt_a is not None else 0.0),
                device=loss.device, dtype=loss.dtype
            ),
        }
        if self.use_decorr:
            log_dict["train/co_error_gamma_v"] = torch.tensor(
                self.co_error_gamma_v, device=loss.device, dtype=loss.dtype
            )
            log_dict["train/co_error_gamma_a"] = torch.tensor(
                self.co_error_gamma_a, device=loss.device, dtype=loss.dtype
            )

        log_dict["train/loss_total"] = loss.detach()
        self.log_dict(log_dict, prog_bar=True, logger=True,
                      on_step=True, on_epoch=False, sync_dist=True)
        return loss

    # ------------------------------------------------------------------
    #  Eval helpers
    # ------------------------------------------------------------------
    @staticmethod
    def compute_topk(sim_matrix, ks=(1, 5, 10)):
        ranks = sim_matrix.argsort(dim=-1, descending=True)
        labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
        return [float((ranks[:, :min(k, sim_matrix.size(1))] == labels.unsqueeze(1))
                       .any(dim=1).float().mean()) for k in ks]

    def _weighted_fusion_sim(self, sim_v, sim_a, alpha):
        return alpha * sim_v + (1.0 - alpha) * sim_a

    def _search_best_alpha_by_topk(self, sim_v, sim_a):
        metric_map = {"top1": (1, 0), "top5": (5, 1), "top10": (10, 2)}
        best_alpha, best_metric = {}, {}
        for name, (_, idx) in metric_map.items():
            ba, bm = self.best_alpha_by_topk.get(name, 0.5), -1.0
            for a in self.alpha_grid_values:
                vals = self.compute_topk(self._weighted_fusion_sim(sim_v, sim_a, a))
                if vals[idx] > bm:
                    bm, ba = vals[idx], float(a)
            best_alpha[name], best_metric[name] = ba, bm
        return best_alpha, best_metric

    def _compute_retrieval_and_ref_metrics(self, sim_v, sim_a, use_search):
        if use_search:
            ba, bv = self._search_best_alpha_by_topk(sim_v, sim_a)
            self.best_alpha_by_topk = {k: float(v) for k, v in ba.items()}
        else:
            ba = self.best_alpha_by_topk
            bv = None
        top_v = self.compute_topk(sim_v)
        top_a = self.compute_topk(sim_a)
        f1 = self.compute_topk(self._weighted_fusion_sim(sim_v, sim_a, ba["top1"]), ks=(1,))[0]
        f5 = self.compute_topk(self._weighted_fusion_sim(sim_v, sim_a, ba["top5"]), ks=(5,))[0]
        f10 = self.compute_topk(self._weighted_fusion_sim(sim_v, sim_a, ba["top10"]), ks=(10,))[0]
        sim_f = self._weighted_fusion_sim(sim_v, sim_a, ba["top10"])
        k_ref = self.ref_sim_top_k
        ref_clip, ref_clap = self._get_reference_features_for_current_mode()
        ref_clip = ref_clip.to(device=sim_v.device, dtype=sim_v.dtype)
        ref_clap = ref_clap.to(device=sim_v.device, dtype=sim_v.dtype)
        return {
            "top_v": top_v, "top_a": top_a, "top_f": [f1, f5, f10],
            "best_alpha_by_topk": ba, "best_val_by_topk": bv,
            "visual_clip_sim": topk_retrieval_reference_similarity(sim_v, ref_clip, top_k=k_ref),
            "visual_clap_sim": topk_retrieval_reference_similarity(sim_v, ref_clap, top_k=k_ref),
            "audio_clip_sim": topk_retrieval_reference_similarity(sim_a, ref_clip, top_k=k_ref),
            "audio_clap_sim": topk_retrieval_reference_similarity(sim_a, ref_clap, top_k=k_ref),
            "fusion_clip_sim": topk_retrieval_reference_similarity(sim_f, ref_clip, top_k=k_ref),
            "fusion_clap_sim": topk_retrieval_reference_similarity(sim_f, ref_clap, top_k=k_ref),
        }

    # ------------------------------------------------------------------
    #  Validation / Test
    # ------------------------------------------------------------------
    def on_validation_epoch_start(self):
        self._vis_pred, self._aud_pred = [], []
        self._vis_raw, self._aud_raw = [], []
        self._idx_list = []

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        brain_v, brain_a, stim_v, stim_a, idx = batch
        z_v, z_a = self(brain_v, brain_a)
        self._vis_pred.append(z_v); self._aud_pred.append(z_a)
        self._vis_raw.append(stim_v); self._aud_raw.append(stim_a)
        self._idx_list.append(idx)

    def _gather_all(self):
        lv = torch.vstack(self._vis_pred)
        la = torch.vstack(self._aud_pred)
        lv_r = torch.vstack(self._vis_raw)
        la_r = torch.vstack(self._aud_raw)
        li = torch.cat(self._idx_list)
        av, aa = self.all_gather(lv).reshape(-1, lv.size(-1)), self.all_gather(la).reshape(-1, la.size(-1))
        avr, aar = self.all_gather(lv_r).reshape(-1, lv_r.size(-1)), self.all_gather(la_r).reshape(-1, la_r.size(-1))
        ai = self.all_gather(li).reshape(-1)
        ds = int(ai.max().item()) + 1
        fv = torch.zeros(ds, av.size(1), device=self.device, dtype=av.dtype)
        fa = torch.zeros(ds, aa.size(1), device=self.device, dtype=aa.dtype)
        fvr = torch.zeros(ds, avr.size(1), device=self.device, dtype=avr.dtype)
        far = torch.zeros(ds, aar.size(1), device=self.device, dtype=aar.dtype)
        fv[ai], fa[ai], fvr[ai], far[ai] = av, aa, avr, aar
        return fv, fa, fvr, far

    def on_validation_epoch_end(self):
        fv, fa, fvr, far = self._gather_all()
        pv, pa = F.normalize(fv, dim=-1), F.normalize(fa, dim=-1)
        rv, ra = F.normalize(fvr, dim=-1), F.normalize(far, dim=-1)
        m = self._compute_retrieval_and_ref_metrics(pv @ rv.T, pa @ ra.T, use_search=True)
        self.log_dict({
            "val/visual_top1": m["top_v"][0], "val/visual_top5": m["top_v"][1], "val/visual_top10": m["top_v"][2],
            "val/audio_top1": m["top_a"][0], "val/audio_top5": m["top_a"][1], "val/audio_top10": m["top_a"][2],
            "val/fusion_top1": m["top_f"][0], "val/fusion_top5": m["top_f"][1], "val/fusion_top10": m["top_f"][2],
            "val/visual_clip_sim": m["visual_clip_sim"], "val/visual_clap_sim": m["visual_clap_sim"],
            "val/audio_clip_sim": m["audio_clip_sim"], "val/audio_clap_sim": m["audio_clap_sim"],
            "val/fusion_clip_sim": m["fusion_clip_sim"], "val/fusion_clap_sim": m["fusion_clap_sim"],
        }, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=False)

    def on_test_epoch_start(self):
        self.on_validation_epoch_start()

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        fv, fa, fvr, far = self._gather_all()
        if self.trainer.is_global_zero:
            np.save(os.path.join(self.log_path, f"{self.test_mode}_brain_visual_pred.npy"), fv.cpu().numpy())
            np.save(os.path.join(self.log_path, f"{self.test_mode}_brain_audio_pred.npy"), fa.cpu().numpy())
        pv, pa = F.normalize(fv, dim=-1), F.normalize(fa, dim=-1)
        rv, ra = F.normalize(fvr, dim=-1), F.normalize(far, dim=-1)
        m = self._compute_retrieval_and_ref_metrics(
            pv @ rv.T,
            pa @ ra.T,
            use_search=(self.test_mode == "val"),
        )
        res = None
        if self.test_mode != "val":
            res = {
                "visual_top1": m["top_v"][0], "visual_top5": m["top_v"][1], "visual_top10": m["top_v"][2],
                "audio_top1": m["top_a"][0], "audio_top5": m["top_a"][1], "audio_top10": m["top_a"][2],
                "fusion_top1": m["top_f"][0], "fusion_top5": m["top_f"][1], "fusion_top10": m["top_f"][2],
                "visual_clip_sim": m["visual_clip_sim"], "visual_clap_sim": m["visual_clap_sim"],
                "audio_clip_sim": m["audio_clip_sim"], "audio_clap_sim": m["audio_clap_sim"],
                "fusion_clip_sim": m["fusion_clip_sim"], "fusion_clap_sim": m["fusion_clap_sim"],
                "best_alpha_top1": m["best_alpha_by_topk"]["top1"],
                "best_alpha_top5": m["best_alpha_by_topk"]["top5"],
                "best_alpha_top10": m["best_alpha_by_topk"]["top10"],
            }
        if self.trainer.is_global_zero and self.test_mode != "val":
            with open(os.path.join(self.log_path, f"{self.test_mode}_acc_results.json"), "w") as fw:
                json.dump(res, fw, indent=4)
        if self.test_mode != "val":
            test_log_dict = {f"{self.test_mode}/{k}": v for k, v in res.items()}
            self.log_dict(test_log_dict, prog_bar=True, logger=True,
                          on_step=False, on_epoch=True, sync_dist=False)


# ---------------------------------------------------------------------------
#  Data loading helpers
# ---------------------------------------------------------------------------
def _load_roi_mask(subject_name, roi):
    roi_mode = _normalize_roi_mode(roi)
    if roi_mode in _CUSTOM_ROI_ALL_KEYS:
        return _load_custom_roi_mask_3d(subject_name, roi_mode)
    mask, _, _, _ = get_valid_voxel_mask_3d(subject_name=subject_name, roi_mode=roi_mode)
    return mask


def load_dual_roi_fmri(subject_name, visual_roi="full", audio_roi="full"):
    """Load fMRI and apply two (possibly different) ROI masks."""
    resp_train, _, resp_val, _, resp_test, _, _ = prepare_fmri_and_stim_same_movie(
        subject_name, feat_name="clip_base_img", feat_type="Visual_Model", feat_path=FEAT_PATH,
    )
    resp_unseen, _ = prepare_fmri_and_stim_test(
        subject_name, feat_name="clip_base_img", feat_type="Visual_Model", feat_path=FEAT_PATH,
    )

    vis_mask = _load_roi_mask(subject_name, visual_roi)
    aud_mask = _load_roi_mask(subject_name, audio_roi)

    def _mask(resp, m):
        return _apply_voxel_mask_to_resp(resp.copy(), m)

    print(f"[DualROI] visual_roi={visual_roi} ({int(np.count_nonzero(vis_mask))} voxels), "
          f"audio_roi={audio_roi} ({int(np.count_nonzero(aud_mask))} voxels)")
    return {
        "vis_train": _mask(resp_train, vis_mask), "vis_val": _mask(resp_val, vis_mask),
        "vis_test": _mask(resp_test, vis_mask), "vis_unseen": _mask(resp_unseen, vis_mask),
        "aud_train": _mask(resp_train, aud_mask), "aud_val": _mask(resp_val, aud_mask),
        "aud_test": _mask(resp_test, aud_mask), "aud_unseen": _mask(resp_unseen, aud_mask),
    }


# ---------------------------------------------------------------------------
#  Training entry points
# ---------------------------------------------------------------------------
def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one(args, visual_feat_name, audio_feat_name, shared_fmri=None):
    print(f"[DecorrCLIP] visual={visual_feat_name}, audio={audio_feat_name}")
    t0 = time()

    if shared_fmri is None:
        shared_fmri = load_dual_roi_fmri(
            args.subject_name,
            visual_roi=args.visual_roi,
            audio_roi=args.audio_roi,
        )

    stim_v_train, stim_v_val, stim_v_test = prepare_stim_same_movie(
        feat_name=visual_feat_name, feat_type="Visual_Model", feat_path=FEAT_PATH,
    )
    stim_v_unseen = prepare_stim_test(
        feat_name=visual_feat_name, feat_type="Visual_Model", feat_path=FEAT_PATH,
    )
    stim_a_train, stim_a_val, stim_a_test = prepare_stim_same_movie(
        feat_name=audio_feat_name, feat_type="Audio_Model", feat_path=FEAT_PATH,
    )
    stim_a_unseen = prepare_stim_test(
        feat_name=audio_feat_name, feat_type="Audio_Model", feat_path=FEAT_PATH,
    )
    clip_ref_features, clap_ref_features = load_default_clip_clap_ref_features(FEAT_PATH)
    print(f"Data loaded in {time() - t0:.1f}s")

    train_set = BrainDualROIDataset(shared_fmri["vis_train"], shared_fmri["aud_train"], stim_v_train, stim_a_train)
    val_set = BrainDualROIDataset(shared_fmri["vis_val"], shared_fmri["aud_val"], stim_v_val, stim_a_val)
    test_set = BrainDualROIDataset(shared_fmri["vis_test"], shared_fmri["aud_test"], stim_v_test, stim_a_test)
    unseen_set = BrainDualROIDataset(shared_fmri["vis_unseen"], shared_fmri["aud_unseen"], stim_v_unseen, stim_a_unseen)

    train_loader = DataLoader(train_set, batch_size=args.batch_size_train, shuffle=True, num_workers=args.num_workers_train)
    val_loader = DataLoader(val_set, batch_size=args.batch_size_eval, shuffle=False, num_workers=args.num_workers_eval)
    test_loader = DataLoader(test_set, batch_size=args.batch_size_eval, shuffle=False, num_workers=args.num_workers_eval)
    unseen_loader = DataLoader(unseen_set, batch_size=args.batch_size_eval, shuffle=False, num_workers=args.num_workers_eval)

    loss_tag = "decorr" if args.use_decorr else "base"
    run_name = (
        f"DecorrCLIP_{args.subject_name}_{visual_feat_name}+{audio_feat_name}"
        f"_vROI-{args.visual_roi}_aROI-{args.audio_roi}_{loss_tag}"
        f"_gv{args.co_error_gamma_v}_ga{args.co_error_gamma_a}"
    )
    if args.fine_tune_visual is False:
        run_name = run_name + "_freezeV"
    if args.fine_tune_audio is False:
        run_name = run_name + "_freezeA"

    log_path = os.path.join("log", "AV_Decorr", run_name)
    if os.path.isfile(os.path.join(log_path, "test_unseen_acc_results.json")):
        print(
            f"[AVCLIP] Skip training: test_unseen_acc_results.json already exists under {log_path}"
        )
        return
    ckpt_path = os.path.join(log_path, "checkpoints")
    os.makedirs(ckpt_path, exist_ok=True)

    mae_path = (
        f"PATH/TO/Narrative_Movie_fMRI_Dataset/pretrained_mae_ckpt/"
        f"{args.subject_name}_Swin3DMAE_Mask50_voxel_pcc/last.ckpt"
    )

    model = BrainAVDecorrCLIP(
        log_path=log_path,
        visual_output_dim=int(stim_v_train.shape[1]),
        audio_output_dim=int(stim_a_train.shape[1]),
        embed_dim=128,
        num_heads=(4, 8, 16),
        first_window_size=(2, 2, 2, 5),
        pretrained_mae_path=mae_path,
        brain_encoder_type=args.brain_encoder_type,
        temperature=args.temperature,
        ref_sim_top_k=args.ref_sim_top_k,
        fine_tune_visual=args.fine_tune_visual,
        fine_tune_audio=args.fine_tune_audio,
        use_decorr=args.use_decorr,
        co_error_gamma_v=args.co_error_gamma_v,
        co_error_gamma_a=args.co_error_gamma_a,
    )
    model.set_reference_features(
        clip_ref_features=clip_ref_features,
        clap_ref_features=clap_ref_features,
    )

    visual_pretrained_dir = _build_single_modal_exp_dir(
        subject_name=args.subject_name,
        feat_type="Visual_Model",
        feat_name=visual_feat_name,
        roi=args.visual_roi,
    )
    audio_pretrained_dir = _build_single_modal_exp_dir(
        subject_name=args.subject_name,
        feat_type="Audio_Model",
        feat_name=audio_feat_name,
        roi=args.audio_roi,
    )
    vis_ckpt = _resolve_single_modal_ckpt(visual_pretrained_dir)
    aud_ckpt = _resolve_single_modal_ckpt(audio_pretrained_dir)
    print(f"[WarmStart] visual ckpt: {vis_ckpt}")
    print(f"[WarmStart] audio ckpt: {aud_ckpt}")
    _load_single_modal_branch_weights(model, vis_ckpt, aud_ckpt)

    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_path, monitor="val/fusion_top10", mode="max",
        filename="best-{val/fusion_top10:.3f}", auto_insert_metric_name=False,
        save_top_k=1, save_last=False, save_weights_only=True,
    )
    import time as time_module
    logger = WandbLogger(save_dir=log_path, project=WANDB_PROJECT, entity=WANDB_ENTITY, name=run_name)
    

    max_retries = 5
    for attempt in range(max_retries):
        try:

            _ = logger.experiment
            print(f"[Wandb] Successfully initialized on attempt {attempt + 1}")
            break
        except Exception as e:
            print(f"[Wandb] Initialization failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print("Retrying in 30 seconds...")
                time_module.sleep(30)
            else:
                raise e

    accel = "gpu" if torch.cuda.is_available() else "cpu"
    devices = 1
    strategy = "auto"
    if accel == "gpu" and torch.cuda.device_count() > 1:
        devices = torch.cuda.device_count()
        strategy = DDPStrategy(find_unused_parameters=False)

    trainer = Trainer(
        accelerator=accel, devices=devices, logger=logger,
        max_epochs=args.epoch_num, num_sanity_val_steps=0,
        check_val_every_n_epoch=2, strategy=strategy,
        callbacks=[ckpt_cb],
    )

    # Validate before training and keep an initial fallback checkpoint.
    init_metrics = trainer.validate(model, dataloaders=val_loader, verbose=False)
    init_fusion_top10 = float(init_metrics[0].get("val/fusion_top10", float("-inf"))) if init_metrics else float("-inf")
    init_ckpt_name = f"init-{init_fusion_top10:.3f}.ckpt"
    init_ckpt_path = os.path.join(ckpt_path, init_ckpt_name)
    trainer.save_checkpoint(init_ckpt_path, weights_only=True)
    print(f"[InitVal] val/fusion_top10={init_fusion_top10:.6f}, saved: {init_ckpt_path}")

    trainer.fit(model, train_loader, val_loader)

    best_ckpt_path = ckpt_cb.best_model_path
    best_score = ckpt_cb.best_model_score
    best_fusion_top10 = float(best_score) if best_score is not None else float("-inf")
    selected_ckpt_path = best_ckpt_path if best_fusion_top10 >= init_fusion_top10 else init_ckpt_path
    selected_tag = "best" if best_fusion_top10 >= init_fusion_top10 else "init"
    print(
        f"[CkptSelect] init={init_fusion_top10:.6f}, best={best_fusion_top10:.6f}, "
        f"use={selected_tag}: {selected_ckpt_path}"
    )

    model.test_mode = "val"
    trainer.test(model, val_loader, ckpt_path=selected_ckpt_path)
    model.test_mode = "test"
    trainer.test(model, test_loader, ckpt_path=selected_ckpt_path)
    model.test_mode = "test_unseen"
    trainer.test(model, unseen_loader, ckpt_path=selected_ckpt_path)
    wandb.finish()


def train_grid(args):
    vis_cfgs = DEFAULT_VISUAL_FEATURE_CONFIGS
    aud_cfgs = DEFAULT_AUDIO_FEATURE_CONFIGS
    print("[DecorrCLIP-GRID] Loading dual-ROI fMRI once …")
    shared_fmri = load_dual_roi_fmri(args.subject_name, args.visual_roi, args.audio_roi)
    total = len(vis_cfgs) * len(aud_cfgs)
    job = 0
    for vn, _ in vis_cfgs:
        for an, _ in aud_cfgs:
            job += 1
            print(f"[DecorrCLIP-GRID][{job}/{total}] {vn} + {an}")
            train_one(args, vn, an, shared_fmri)


def parse_args():
    p = argparse.ArgumentParser(description="Separate-encoder joint training with co-error guided InfoNCE.")
    p.add_argument("--subject_name", type=str, default="S01")
    p.add_argument("--visual_feat_name", type=str, default=None)
    p.add_argument("--audio_feat_name", type=str, default=None)
    p.add_argument("--visual_roi", type=str, default="full",
                   help="ROI for visual encoder (e.g. full, ROI_V, ROI_vld, ROI_VA)")
    p.add_argument("--audio_roi", type=str, default="full",
                   help="ROI for audio encoder (e.g. full, ROI_A, ROI_vld, ROI_VA)")
    p.add_argument("--epoch_num", type=int, default=10)
    p.add_argument("--batch_size_train", type=int, default=128)
    p.add_argument("--batch_size_eval", type=int, default=64)
    p.add_argument("--num_workers_train", type=int, default=1)
    p.add_argument("--num_workers_eval", type=int, default=1)
    p.add_argument("--brain_encoder_type", type=str, default="3d", choices=["3d", "4d"])
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--ref_sim_top_k", type=int, default=1)
    p.add_argument("--fine_tune_visual", action="store_true", default=True,
                   help="Fine-tune visual branch parameters (default: on)")
    p.add_argument("--freeze_visual", dest="fine_tune_visual", action="store_false",
                   help="Freeze visual branch: no optimizer/loss/backprop for visual branch")
    p.add_argument("--fine_tune_audio", action="store_true", default=True,
                   help="Fine-tune audio branch parameters (default: on)")
    p.add_argument("--freeze_audio", dest="fine_tune_audio", action="store_false",
                   help="Freeze audio branch: no optimizer/loss/backprop for audio branch")
    # Cross-modal hard negative InfoNCE
    p.add_argument("--use_decorr", action="store_true", default=True,
                   help="Enable cross-modal hard negative InfoNCE (default: on)")
    p.add_argument("--no_decorr", dest="use_decorr", action="store_false",
                   help="Disable cross-modal hard negative guidance and use plain InfoNCE")
    p.add_argument("--co_error_gamma_v", type=float, default=0.0,
                   help="Cross-modal hard negative strength for visual branch (typically 2.0 to protect visual semantics)")
    p.add_argument("--co_error_gamma_a", type=float, default=0.0,
                   help="Cross-modal hard negative strength for audio branch (typically 10.0 to learn visual blind spots)")
    # Grid mode
    p.add_argument("--run_grid", action="store_true",
                   help="Train all default visual-audio pairs.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    fix_seed(42)
    torch.set_float32_matmul_precision("high")
    if args.run_grid:
        train_grid(args)
    else:
        if not args.visual_feat_name or not args.audio_feat_name:
            raise ValueError("Provide --visual_feat_name and --audio_feat_name, or use --run_grid.")
        train_one(args, args.visual_feat_name, args.audio_feat_name)
