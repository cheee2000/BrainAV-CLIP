import os
import json
import cortex
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr, spearmanr

from dataset import BrainDataset, prepare_fmri_and_stim_test, prepare_fmri_and_stim_same_movie, prepare_stim_same_movie, \
    center_stim_by_train_mean
from fMRI_Narrative_movie.util import util_dataload as udl
from fMRI_Narrative_movie.util import util_pycortex as utlpy
from fMRI_Narrative_movie.util import util_visualization as uvis


ROI_GROUPS_FOR_DECODING = {
    "LVC": ["V1", "V2", "V3"],
    "HVC": ["FFA", "OFA", "PPA", "MT+"],
    "AC": ["AC"],
}


def _normalize_roi_mode_for_decoding(roi_mode):
    roi_mode = str(roi_mode).strip()
    if not roi_mode:
        return "full"
    if roi_mode.lower() == "full":
        return "full"
    roi_mode_upper = roi_mode.upper()
    if roi_mode_upper not in ROI_GROUPS_FOR_DECODING:
        raise ValueError(
            f"Unsupported roi mode: {roi_mode}. Expected one of: full, LVC, HVC, AC."
        )
    return roi_mode_upper


def get_valid_voxel_mask_3d(subject_name='S02', roi_mode='full'):
    """
    Build the valid voxel mask (72, 96, 96) for the given subject/ROI mode.
    Returns:
      valid_mask_3d, normalized_roi_mode, roi_targets, matched_roi_names
    """
    normalized_roi_mode = _normalize_roi_mode_for_decoding(roi_mode)

    with open('fMRI_Narrative_movie/util/config__drama_data.yaml', 'r') as f_yml:
        config = yaml.safe_load(f_yml)
    data_info = utlpy.get_dataInfo(config, subject_name)

    subject_id = udl.get_subjectID_from_subjectName(config, subject_name)
    load_items = udl.set_load_items(config, subjectID=subject_id, movID=0)
    cortical_mask_indices = np.asarray(udl.load_mask_data(load_items['mask_path']), dtype=np.int64)

    cortical_mask = np.zeros(663552, dtype=bool)
    cortical_mask[cortical_mask_indices] = True
    valid_mask = cortical_mask.copy()
    roi_targets = ROI_GROUPS_FOR_DECODING.get(normalized_roi_mode, [])
    matched_roi_names = []

    if normalized_roi_mode != "full":
        roi_mask_dict = cortex.utils.get_roi_masks(
            data_info['subjectNamePycortex'],
            data_info['dataSetName'],
            roi_list=None,
            gm_sampler='cortical-conservative',
            split_lr=True,
        )
        roi_union_3d = np.zeros((72, 96, 96), dtype=bool)
        for roi_name in roi_targets:
            roi_found = False
            for key, roi_mask in roi_mask_dict.items():
                if key == roi_name or key.startswith(roi_name + "_"):
                    roi_union_3d |= (np.asarray(roi_mask) > 0)
                    matched_roi_names.append(key)
                    roi_found = True
            if not roi_found:
                print(f"[ROI] Warning: no mask found for ROI '{roi_name}' (subject={subject_name}).")

        valid_mask = roi_union_3d.reshape(-1) & cortical_mask

    return (
        valid_mask.reshape(72, 96, 96),
        normalized_roi_mode,
        roi_targets,
        sorted(set(matched_roi_names)),
    )


def cal_grad(ckpt_path, subject_name='S02', feat_name='imagebind_img'):
    from train_BrainCLIP import BrainCLIP

    # Resp_test_volume, Stim_test = \
    #     prepare_fmri_and_stim_test(subject_name, feat_name=feat_name)
    print('Loading data..')
    Resp_train_volume, Stim_train, Resp_val_volume, Stim_val, Resp_test_volume, Stim_test = \
        prepare_fmri_and_stim_same_movie(subject_name, feat_name=feat_name)

    test_set = BrainDataset(Resp_test_volume, Stim_test)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=True, num_workers=1)

    print('Loading model..')
    model = BrainCLIP()
    ckpt = torch.load(ckpt_path)["state_dict"]
    model.load_state_dict(ckpt, strict=False)
    model.cuda()

    model.zero_grad()

    total_grad = []
    for batch_idx, batch in tqdm(enumerate(test_loader), desc="Calculate", total=len(test_loader)):
        brain_data, stimuli_feature = batch


        brain_data.requires_grad_(True)


        if brain_data.grad is not None:
            brain_data.grad.zero_()


        brain_feature = model(brain_data.cuda())
        loss = model.info_nce_loss(brain_feature, stimuli_feature.cuda())


        brain_data.retain_grad()
        loss.backward()


        gradients = brain_data.grad.detach().clone()
        gradients = torch.abs(gradients)
        total_grad.append(gradients.squeeze(1).cpu().numpy())


        brain_data.grad.zero_()

    total_grads = np.vstack(total_grad).mean(-1).mean(0)
    print(total_grads.shape)

    img_path = os.path.dirname(os.path.dirname(ckpt_path))
    visualize_weight(total_grads, subject_name=subject_name, img_path=img_path)


def visualize_weight(weight, subject_name='S02', img_path=None):
    with open('fMRI_Narrative_movie/util/config__drama_data.yaml', 'r') as f_yml:
        config = yaml.safe_load(f_yml)
     # S01, S02, ..., S06

    dataInfo = utlpy.get_dataInfo(config, subject_name)

    # subjectID = udl.get_subjectID_from_subjectName(config, subject_name)
    # load_items = udl.set_load_items(config, subjectID, 0)
    # t_path_mask = load_items['mask_path']
    # mask = udl.load_mask_data(t_path_mask)
    #
    # flatten_weight = weight.reshape(-1)
    # masked_weight = flatten_weight[mask]
    # masked_weight_z = (masked_weight - masked_weight.mean()) / masked_weight.std()
    #
    # roi_masks = np.zeros(663552) * np.nan
    # roi_masks[mask] = masked_weight_z
    # roi_masks = roi_masks.reshape(72, 96, 96)

    roi_data = cortex.Volume(weight, dataInfo['subjectNamePycortex'], dataInfo['dataSetName'],  # vmin=0, vmax=1,
                             cmap="inferno")
    cortex.quickflat.make_figure(roi_data, thick=1, with_curvature=True, with_colorbar=True)

    # plt.show()
    plt.savefig(os.path.join(img_path, 'weight.png'))


def get_available_roi_names(subject_name='S02', split_lr=False):
    """
    Return all ROI names available for the current subject (from pycortex overlays).
    - When roi_list=None, get_roi_masks returns all ROIs.
    - When split_lr=True, each ROI is split into xxx_L / xxx_R (left/right hemisphere).
    Returns: list of str; use len() to get the number of ROIs.
    """
    with open('fMRI_Narrative_movie/util/config__drama_data.yaml', 'r') as f_yml:
        config = yaml.safe_load(f_yml)
    dataInfo = utlpy.get_dataInfo(config, subject_name)
    roi_mask_dict = cortex.utils.get_roi_masks(
        dataInfo['subjectNamePycortex'], dataInfo['dataSetName'],
        roi_list=None, gm_sampler='cortical-conservative', split_lr=split_lr
    )
    return sorted(roi_mask_dict.keys())


def get_roi_weight_mean_dict(weight, roi_list, subject_name='S02', split_lr=False):
    """
    Compute the mean of non-NaN voxel weights within each ROI.

    Parameters
    ------
    weight : 1D array
        Same as in encoding/decoding: length equals the number of voxels inside the
        mask (weights matching the indices returned by udl.load_mask_data).
    roi_list : list[str]
        ROI names to aggregate, e.g. ['MT+', 'AC_L'].
    subject_name : str
        Subject name, default 'S02'.
    split_lr : bool
        When True, use left/right split ROI names:
        - if roi_list has a name without _L/_R (e.g. 'MT+'), take the union of its
          left/right hemispheres (MT+_L, MT+_R) and average;
        - if roi_list has a suffix (e.g. 'AC_L'), only that hemisphere is counted.

    Returns
    ------
    dict[str, float]
        key is the passed ROI name (matching roi_list); value is the mean of non-NaN
        weights within that ROI (0 if the ROI has no non-NaN voxels).
    """
    if roi_list is None or len(roi_list) == 0:
        raise ValueError("roi_list must not be empty; it specifies which ROIs to aggregate.")

    with open('fMRI_Narrative_movie/util/config__drama_data.yaml', 'r') as f_yml:
        config = yaml.safe_load(f_yml)
    dataInfo = utlpy.get_dataInfo(config, subject_name)

    subjectID = udl.get_subjectID_from_subjectName(config, subject_name)
    load_items = udl.set_load_items(config, subjectID, 0)
    t_path_mask = load_items['mask_path']
    mask = udl.load_mask_data(t_path_mask)


    weight_volume = np.zeros(663552) * np.nan
    weight_volume[mask] = weight
    weight_volume = weight_volume.reshape(72, 96, 96)



    roi_list_arg = None if split_lr else roi_list
    roi_mask_dict = cortex.utils.get_roi_masks(
        dataInfo['subjectNamePycortex'], dataInfo['dataSetName'],
        roi_list=roi_list_arg, gm_sampler='cortical-conservative', split_lr=split_lr
    )

    roi_mean_dict = {}
    for req_roi in roi_list:

        cur_mask = np.zeros_like(weight_volume)

        if split_lr:

            if req_roi in roi_mask_dict:
                cur_mask = np.asarray(roi_mask_dict[req_roi])
            else:

                for roi_name, roi_mask in roi_mask_dict.items():
                    if roi_name == req_roi or roi_name.startswith(req_roi + '_'):
                        cur_mask = np.maximum(cur_mask, np.asarray(roi_mask))
        else:

            if req_roi in roi_mask_dict:
                cur_mask = np.asarray(roi_mask_dict[req_roi])


        roi_weights = weight_volume[cur_mask > 0]
        if roi_weights.size == 0:
            roi_mean = 0.0
        else:

            roi_weights = roi_weights[~np.isnan(roi_weights)]
            roi_mean = 0.0 if roi_weights.size == 0 else float(roi_weights.mean())

        roi_mean_dict[req_roi] = roi_mean


    full_weights = weight_volume[~np.isnan(weight_volume)]
    full_mean = 0.0 if full_weights.size == 0 else float(full_weights.mean())
    roi_mean_dict['full'] = full_mean

    return roi_mean_dict


def visualize_masked_weight(weight, img_path, subject_name='S02', roi_list=None, split_lr=False, vmin=0, vmax=1, cmap="inferno"):
    """
    Visualize voxel weights. Defaults to the whole brain (all voxels in the mask).
    - roi_list=None or []: no ROI restriction; plot all voxels.
    - roi_list with ROIs (e.g. ["MT+"], ["AC"]): plot only voxels within those ROIs.
    get_available_roi_names() lists all ROIs. With split_lr=True you can filter by
    hemisphere (e.g. AC_L/AC_R).
    """
    with open('fMRI_Narrative_movie/util/config__drama_data.yaml', 'r') as f_yml:
        config = yaml.safe_load(f_yml)
     # S01, S02, ..., S06

    dataInfo = utlpy.get_dataInfo(config, subject_name)

    subjectID = udl.get_subjectID_from_subjectName(config, subject_name)
    load_items = udl.set_load_items(config, subjectID, 0)
    t_path_mask = load_items['mask_path']
    mask = udl.load_mask_data(t_path_mask)


    weight_volume = np.zeros(663552) * np.nan
    weight_volume[mask] = weight
    weight_volume = weight_volume.reshape(72, 96, 96)


    if roi_list:

        roi_list_arg = None if split_lr else roi_list
        roi_mask_dict = cortex.utils.get_roi_masks(
            dataInfo['subjectNamePycortex'], dataInfo['dataSetName'],
            roi_list=roi_list_arg, gm_sampler='cortical-conservative', split_lr=split_lr
        )
        roi_combined = np.zeros_like(weight_volume)
        for roi_name in roi_mask_dict:
            if any(roi_name == r or roi_name.startswith(r + '_') for r in roi_list):
                roi_combined = np.maximum(roi_combined, np.asarray(roi_mask_dict[roi_name]))
        weight_volume[roi_combined <= 0] = np.nan

    roi_data = cortex.Volume(weight_volume, dataInfo['subjectNamePycortex'], dataInfo['dataSetName'], vmin=vmin, vmax=vmax,
                             cmap=cmap)
    cortex.quickflat.make_figure(roi_data, thick=1, with_curvature=True, with_colorbar=True)

    # plt.show()
    plt.savefig(img_path)


def tri_color_from_percent(vis_pct, aud_pct):
    """Generate a red-blue mixed color from visual/audio percentages (origin is white)."""

    total = vis_pct + aud_pct
    scale = np.ones_like(total, dtype=float)
    need_rescale = total > 100.0
    np.divide(100.0, total, out=scale, where=need_rescale)
    vis_pct = vis_pct * scale
    aud_pct = aud_pct * scale

    vis_frac = np.clip(vis_pct / 100.0, 0.0, 1.0)
    aud_frac = np.clip(aud_pct / 100.0, 0.0, 1.0)
    shared_frac = np.clip(1.0 - (vis_frac + aud_frac), 0.0, 1.0)

    r = shared_frac + vis_frac
    g = shared_frac
    b = shared_frac + aud_frac
    return np.stack([r, g, b], axis=-1)


def get_bivariate_brain_colors(vis_ev, aud_ev, joint_ev, vmax_joint):
    """
    Compute 2D colormap RGB (X: modality preference, Y: joint EV brightness); returns uint8.
    """
    if vmax_joint is None or not np.isfinite(vmax_joint) or vmax_joint <= 0:
        vmax_joint = 1e-6

    vis_ev = np.asarray(vis_ev, dtype=float)
    aud_ev = np.asarray(aud_ev, dtype=float)
    joint_ev = np.asarray(joint_ev, dtype=float)
    if not (vis_ev.shape == aud_ev.shape == joint_ev.shape):
        raise ValueError(
            f"vis_ev/aud_ev/joint_ev shapes mismatch: {vis_ev.shape}, {aud_ev.shape}, {joint_ev.shape}"
        )


    vis_pos = np.maximum(vis_ev, 0.0)
    aud_pos = np.maximum(aud_ev, 0.0)
    total_unique = vis_pos + aud_pos

    bias = np.divide(
        vis_pos - aud_pos,
        total_unique,
        out=np.zeros_like(vis_pos),
        where=total_unique != 0,
    )
    bias = np.clip(bias, -1.0, 1.0)

    intensity = np.clip(joint_ev / vmax_joint, 0.0, 1.0)
    return _bivariate_rgb_from_bias_intensity(bias, intensity)


def _bivariate_rgb_from_bias_intensity(bias, intensity):
    """
    Map bias([-1,1]) and intensity([0,1]) to uint8 RGB.
    - bias=-1: blue
    - bias=0 : white
    - bias=1 : red
    """
    bias = np.clip(np.asarray(bias, dtype=float), -1.0, 1.0)
    intensity = np.clip(np.asarray(intensity, dtype=float), 0.0, 1.0)
    if bias.shape != intensity.shape:
        raise ValueError(f"bias/intensity shapes mismatch: {bias.shape}, {intensity.shape}")

    r = np.ones_like(bias, dtype=float)
    g = np.ones_like(bias, dtype=float)
    b = np.ones_like(bias, dtype=float)

    aud_mask = bias < 0


    r[aud_mask] = 1.0 + bias[aud_mask]
    g[aud_mask] = 1.0 + bias[aud_mask]
    b[aud_mask] = 1.0

    vis_mask = ~aud_mask


    r[vis_mask] = 1.0
    g[vis_mask] = 1.0 - bias[vis_mask]
    b[vis_mask] = 1.0 - bias[vis_mask]

    r = r * intensity
    g = g * intensity
    b = b * intensity

    r_uint8 = (np.clip(r, 0.0, 1.0) * 255).astype(np.uint8)
    g_uint8 = (np.clip(g, 0.0, 1.0) * 255).astype(np.uint8)
    b_uint8 = (np.clip(b, 0.0, 1.0) * 255).astype(np.uint8)
    return r_uint8, g_uint8, b_uint8


def draw_bivariate_legend(vmax_joint, save_path="bivariate_legend.png", min_joint_ev_threshold=0.02):
    """
    Draw the 2D colormap legend (X: modality preference, Y: joint EV transparency).
    X axis fixed hues: blue (Audio) -> magenta (Shared) -> red (Visual).
    Y axis keeps brightness constant and only changes transparency: low EV near
    transparent, high EV opaque.
    """
    vmax_joint = float(vmax_joint)
    if not np.isfinite(vmax_joint) or vmax_joint <= 0:
        vmax_joint = 1e-6
    min_joint_ev_threshold = float(min_joint_ev_threshold)
    if not np.isfinite(min_joint_ev_threshold):
        min_joint_ev_threshold = 0.0
    min_joint_ev_threshold = max(min_joint_ev_threshold, 0.0)

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    x = np.linspace(-1, 1, 300)
    y_start = min_joint_ev_threshold
    y_end = vmax_joint
    if y_end <= y_start:
        y_end = y_start + 1e-6
    y = np.linspace(y_start, y_end, 300)
    x_grid, y_grid = np.meshgrid(x, y)

    bias_grid = np.clip(x_grid, -1.0, 1.0)
    t_neg = np.clip(bias_grid + 1.0, 0.0, 1.0)
    t_pos = np.clip(bias_grid, 0.0, 1.0)
    neg_mask = bias_grid < 0

    r = np.zeros_like(bias_grid, dtype=float)
    g = np.zeros_like(bias_grid, dtype=float)
    b = np.zeros_like(bias_grid, dtype=float)
    r[neg_mask] = t_neg[neg_mask]
    b[neg_mask] = 1.0
    r[~neg_mask] = 1.0
    b[~neg_mask] = 1.0 - t_pos[~neg_mask]

    rgb = np.stack([r, g, b], axis=-1)
    rgba = np.concatenate([rgb, np.ones((*rgb.shape[:2], 1), dtype=float)], axis=-1)


    alpha_min = 0.1
    y_for_alpha = np.clip(y_grid, y_start, y_end)
    alpha_base = np.clip(
        (y_for_alpha - y_start) / max(y_end - y_start, 1e-6),
        0.0,
        1.0,
    )
    rgba[..., 3] = alpha_min + (1.0 - alpha_min) * alpha_base

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.imshow(rgba, extent=[-1, 1, y_start, y_end], origin='lower', aspect='auto')
    # ax.set_xlabel('Modality Bias\n<-- Auditory  |  Shared  |  Visual -->', fontsize=12, fontweight='bold')
    # ax.set_ylabel(r'Total Explained Variance ($EV_{joint}$)', fontsize=12, fontweight='bold')
    # ax.set_xticks([-1, 0, 1])
    ax.set_xticks([])
    tick_candidates = [0.01, 0.05, 0.10, 0.15]
    y_ticks = [t for t in tick_candidates if y_start - 1e-12 <= t <= y_end + 1e-12]
    if not y_ticks:
        y_ticks = [y_start, y_end]
    y_tick_labels = []
    for t in y_ticks:
        y_tick_labels.append(f"{t:.2f}")
        # if np.isclose(t, y_start):
        #     y_tick_labels.append(f"{t:.2f} (thr)")
        # else:
        #     y_tick_labels.append(f"{t:.2f}")
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels)
    # ax.set_xticklabels(['Audio', 'Shared', 'Visual'])
    ax.tick_params(axis='both', which='major', labelsize=36)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, transparent=True)
    plt.close(fig)
    print(f"[save] bivariate legend: {save_path}")


def visualize_bivariate_ev_brain(
    vis_ev,
    aud_ev,
    joint_ev,
    img_path,
    subject_name='S02',
    vmax_joint=None,
    min_joint_ev_threshold=0.02,
):
    """
    Map the 2D colormap onto the brain:
    - color (RGB) depends only on modality preference: blue (Audio) -> magenta (Shared) -> red (Visual)
    - joint EV only controls the alpha transparency: low EV more transparent, high EV more opaque
    """
    with open('fMRI_Narrative_movie/util/config__drama_data.yaml', 'r') as f_yml:
        config = yaml.safe_load(f_yml)
    data_info = utlpy.get_dataInfo(config, subject_name)

    subject_id = udl.get_subjectID_from_subjectName(config, subject_name)
    load_items = udl.set_load_items(config, subject_id, 0)
    mask = np.asarray(udl.load_mask_data(load_items['mask_path']))

    vis_ev = np.asarray(vis_ev, dtype=float)
    aud_ev = np.asarray(aud_ev, dtype=float)
    joint_ev = np.asarray(joint_ev, dtype=float)
    min_joint_ev_threshold = float(min_joint_ev_threshold)
    if not np.isfinite(min_joint_ev_threshold):
        min_joint_ev_threshold = 0.0
    min_joint_ev_threshold = max(min_joint_ev_threshold, 0.0)
    if not (vis_ev.shape == aud_ev.shape == joint_ev.shape):
        raise ValueError(
            f"vis_ev/aud_ev/joint_ev shapes mismatch: {vis_ev.shape}, {aud_ev.shape}, {joint_ev.shape}"
        )

    valid = np.isfinite(vis_ev) & np.isfinite(aud_ev) & np.isfinite(joint_ev)
    valid_and_above_thr = valid & (joint_ev >= min_joint_ev_threshold)

    if vmax_joint is None:
        joint_valid = joint_ev[valid_and_above_thr]
        if joint_valid.size == 0:
            joint_valid = joint_ev[valid]
        if joint_valid.size > 0:
            vmax_joint = float(np.percentile(joint_valid, 99))
        else:
            vmax_joint = 1e-6
    if not np.isfinite(vmax_joint) or vmax_joint <= 0:
        vmax_joint = 1e-6

    red = np.zeros(663552, dtype=np.uint8)
    green = np.zeros(663552, dtype=np.uint8)
    blue = np.zeros(663552, dtype=np.uint8)
    alpha = np.zeros(663552, dtype=np.uint8)

    valid_flat_inds = mask[valid_and_above_thr]


    vis_valid = vis_ev[valid_and_above_thr]
    aud_valid = aud_ev[valid_and_above_thr]
    joint_valid = joint_ev[valid_and_above_thr]
    denom = vis_valid + aud_valid
    eps = 1e-8
    bias = np.zeros_like(joint_valid, dtype=float)
    nz = np.abs(denom) > eps
    bias[nz] = (vis_valid[nz] - aud_valid[nz]) / denom[nz]
    bias = np.clip(bias, -1.0, 1.0)

    t_neg = np.clip(bias + 1.0, 0.0, 1.0)
    t_pos = np.clip(bias, 0.0, 1.0)
    neg_mask = bias < 0

    r = np.zeros_like(bias, dtype=float)
    g = np.zeros_like(bias, dtype=float)
    b = np.zeros_like(bias, dtype=float)
    r[neg_mask] = t_neg[neg_mask]
    b[neg_mask] = 1.0
    r[~neg_mask] = 1.0
    b[~neg_mask] = 1.0 - t_pos[~neg_mask]

    r_uint8 = (np.clip(r, 0.0, 1.0) * 255).astype(np.uint8)
    g_uint8 = (np.clip(g, 0.0, 1.0) * 255).astype(np.uint8)
    b_uint8 = (np.clip(b, 0.0, 1.0) * 255).astype(np.uint8)
    red[valid_flat_inds] = r_uint8
    green[valid_flat_inds] = g_uint8
    blue[valid_flat_inds] = b_uint8


    alpha_min = 0.1
    joint_for_alpha = np.clip(joint_valid, 0.0, vmax_joint)
    alpha_base = np.clip(
        (joint_for_alpha - min_joint_ev_threshold) / max(vmax_joint - min_joint_ev_threshold, 1e-6),
        0.0,
        1.0,
    )
    alpha_uint8 = (np.clip(alpha_min + (1.0 - alpha_min) * alpha_base, 0.0, 1.0) * 255).astype(np.uint8)
    alpha[valid_flat_inds] = alpha_uint8

    red = red.reshape(72, 96, 96)
    green = green.reshape(72, 96, 96)
    blue = blue.reshape(72, 96, 96)
    alpha = alpha.reshape(72, 96, 96)

    roi_data = cortex.VolumeRGB(
        red,
        green,
        blue,
        data_info['subjectNamePycortex'],
        data_info['dataSetName'],
        alpha=alpha,
    )
    cortex.quickflat.make_figure(
        roi_data,
        thick=1,
        with_curvature=True,
        with_colorbar=False,
    )
    plt.savefig(img_path, dpi=300)
    plt.close()


def visualize_discrete_modality_masks_rgb(
    shared_mask,
    vis_mask,
    aud_mask,
    img_path,
    subject_name='S02',
):
    """
    Visualize discrete modality voxel masks (three-channel overlay):
    - visual=red, shared=green, auditory=blue
    - overlaps combine via RGB (yellow/cyan/magenta/white)
    Input masks are 1D 0/1 arrays whose length matches the number of voxels in the cortical mask.
    """
    with open('fMRI_Narrative_movie/util/config__drama_data.yaml', 'r') as f_yml:
        config = yaml.safe_load(f_yml)
    data_info = utlpy.get_dataInfo(config, subject_name)

    subject_id = udl.get_subjectID_from_subjectName(config, subject_name)
    load_items = udl.set_load_items(config, subject_id, 0)
    mask = np.asarray(udl.load_mask_data(load_items['mask_path']))

    shared_mask = np.asarray(shared_mask, dtype=np.uint8).reshape(-1)
    vis_mask = np.asarray(vis_mask, dtype=np.uint8).reshape(-1)
    aud_mask = np.asarray(aud_mask, dtype=np.uint8).reshape(-1)
    if not (shared_mask.shape == vis_mask.shape == aud_mask.shape):
        raise ValueError(
            f"shared/vis/aud mask shape mismatch: {shared_mask.shape}, {vis_mask.shape}, {aud_mask.shape}"
        )

    n_masked = mask.shape[0]
    if shared_mask.shape[0] != n_masked:
        raise ValueError(
            f"mask length does not match the subject voxel count: got={shared_mask.shape[0]}, expected={n_masked}"
        )


    # R <- visual, G <- shared, B <- auditory
    shared_bool = shared_mask > 0
    vis_bool = vis_mask > 0
    aud_bool = aud_mask > 0
    any_bool = shared_bool | vis_bool | aud_bool

    red = np.zeros(663552, dtype=np.uint8)
    green = np.zeros(663552, dtype=np.uint8)
    blue = np.zeros(663552, dtype=np.uint8)
    alpha = np.zeros(663552, dtype=np.uint8)

    flat_inds = mask
    red[flat_inds[vis_bool]] = 255
    green[flat_inds[shared_bool]] = 255
    blue[flat_inds[aud_bool]] = 255

    alpha[flat_inds[any_bool]] = 255

    red = red.reshape(72, 96, 96)
    green = green.reshape(72, 96, 96)
    blue = blue.reshape(72, 96, 96)
    alpha = alpha.reshape(72, 96, 96)

    roi_data = cortex.VolumeRGB(
        red,
        green,
        blue,
        data_info['subjectNamePycortex'],
        data_info['dataSetName'],
        alpha=alpha,
    )
    cortex.quickflat.make_figure(
        roi_data,
        thick=1,
        with_curvature=True,
        with_colorbar=False,
    )
    plt.savefig(img_path, dpi=300)
    plt.close()


def draw_tricolor_circle_mixing_colorbar(save_path, size=700):
    """
    Draw a three-primary-color circle mixing diagram (additive mixing):
    - Visual Unique: Red
    - Shared: Green
    - Auditory Unique: Blue
    """
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    size = int(size)
    if size <= 0:
        size = 700

    x = np.linspace(-1.0, 1.0, size)
    y = np.linspace(-1.0, 1.0, size)
    x_grid, y_grid = np.meshgrid(x, y)

    radius = 0.52
    centers = {
        "R": (-0.36, -0.18),  # visual
        "G": (0.00, 0.36),    # shared
        "B": (0.36, -0.18),   # auditory
    }

    mask_r = (x_grid - centers["R"][0]) ** 2 + (y_grid - centers["R"][1]) ** 2 <= radius ** 2
    mask_g = (x_grid - centers["G"][0]) ** 2 + (y_grid - centers["G"][1]) ** 2 <= radius ** 2
    mask_b = (x_grid - centers["B"][0]) ** 2 + (y_grid - centers["B"][1]) ** 2 <= radius ** 2

    rgb = np.zeros((size, size, 3), dtype=float)
    rgb[..., 0] = mask_r.astype(float)
    rgb[..., 1] = mask_g.astype(float)
    rgb[..., 2] = mask_b.astype(float)

    alpha = (mask_r | mask_g | mask_b).astype(float)
    rgba = np.concatenate([rgb, alpha[..., None]], axis=-1)

    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    ax.imshow(rgba, origin='lower', extent=[-1, 1, -1, 1], interpolation='nearest')
    ax.set_axis_off()

    ax.text(-0.73, -0.7, "Visual", color='red', fontsize=10, ha='center', va='center', fontweight='bold')
    ax.text(0.00, 0.93, "Shared", color='lime', fontsize=10, ha='center', va='center', fontweight='bold')
    ax.text(0.73, -0.7, "Auditory", color='dodgerblue', fontsize=10, ha='center', va='center', fontweight='bold')
    # ax.set_title("Additive RGB Mixing (Top-mask Colorbar)", fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, transparent=True)
    plt.close(fig)
    print(f"[save] tricolor mixing colorbar: {save_path}")


def visualize_unique_ratio_rgb_brain(vis_pct, aud_pct, img_path, subject_name='S02'):
    """Map visual/audio unique fractions to an RGB brain map (red=visual, blue=audio, white=low fraction)."""
    with open('fMRI_Narrative_movie/util/config__drama_data.yaml', 'r') as f_yml:
        config = yaml.safe_load(f_yml)
    data_info = utlpy.get_dataInfo(config, subject_name)

    subject_id = udl.get_subjectID_from_subjectName(config, subject_name)
    load_items = udl.set_load_items(config, subject_id, 0)
    mask = udl.load_mask_data(load_items['mask_path'])

    valid = np.isfinite(vis_pct) & np.isfinite(aud_pct)
    valid &= (vis_pct >= 0) & (aud_pct >= 0)
    valid &= (vis_pct + aud_pct <= 100 + 1e-8)

    red = np.zeros(663552, dtype=float)
    green = np.zeros(663552, dtype=float)
    blue = np.zeros(663552, dtype=float)
    alpha = np.zeros(663552, dtype=float)

    vis_valid = vis_pct[valid]
    aud_valid = aud_pct[valid]
    rgb_valid = tri_color_from_percent(vis_valid, aud_valid)

    valid_flat_inds = np.asarray(mask)[valid]
    red[valid_flat_inds] = rgb_valid[:, 0]
    green[valid_flat_inds] = rgb_valid[:, 1]
    blue[valid_flat_inds] = rgb_valid[:, 2]
    alpha[valid_flat_inds] = 1.0

    red = red.reshape(72, 96, 96)
    green = green.reshape(72, 96, 96)
    blue = blue.reshape(72, 96, 96)
    alpha = alpha.reshape(72, 96, 96)



    red_uint8 = (np.clip(red, 0.0, 1.0) * 255).astype(np.uint8)
    green_uint8 = (np.clip(green, 0.0, 1.0) * 255).astype(np.uint8)
    blue_uint8 = (np.clip(blue, 0.0, 1.0) * 255).astype(np.uint8)
    alpha_uint8 = (np.clip(alpha, 0.0, 1.0) * 255).astype(np.uint8)


    roi_data = cortex.VolumeRGB(
        red_uint8,
        green_uint8,
        blue_uint8,
        data_info['subjectNamePycortex'],
        data_info['dataSetName'],
        alpha=alpha_uint8,
    )
    cortex.quickflat.make_figure(
        roi_data,
        thick=1,
        with_curvature=True,
        with_colorbar=False,
    )
    plt.savefig(img_path, dpi=300)
    plt.close()


def visualize_acc():
    with open('fMRI_Narrative_movie/util/config__drama_data.yaml', 'r') as f_yml:
        config = yaml.safe_load(f_yml)

    subject_name = 'S02'
    # stat_path = 'fMRI_Narrative_movie/demo_files/ridge_stats__DM01.pkl'
    resultType = 'banded__obj_speech_story'  # banded__obj_speech_story, loc_{no}:{no}=[1,2,...,11] (e.g. loc_3)

    with_rois = True
    with_sulci = True
    with_curvature = True

    dataInfo = utlpy.get_dataInfo(config, subject_name)

    ###
    ### Banded ridge

    if resultType == 'banded__obj_speech_story':
        dt = uvis.get_results_visualization_demo(config, dataInfo['subjectName'], resultType)
        dt3d_stats_show, dt3d_sig_show, mask, showType = uvis.get_bandedridge_dt3d(dataInfo, dt)

        saveFigName = 'visualization_test'
        with_colorbar = False
        colorType = 'rgb'
        max_stat = 0.02

        uvis.pycortex_visualization(dataInfo, dt3d_stats_show, dt3d_sig_show, \
                                    showType='stats', colorType=colorType, max_stat=max_stat, \
                                    mask=mask, saveFigName=saveFigName, \
                                    with_rois=with_rois, with_sulci=with_sulci, with_curvature=with_curvature,
                                    with_colorbar=with_colorbar)

    ###
    ### functional ROIs

    if 'loc_' in resultType:
        dt, locName = uvis.get_results_visualization_demo(config, dataInfo['subjectName'], resultType)
        dt3d_stats_show, dt3d_sig_show, mask, showType, colorType = uvis.get_froi_dt3d(dataInfo, dt, locName)

        saveFigName = 'visualization_test'
        with_colorbar = False
        max_stat = uvis.set_max_stat(locName, colorType)

        uvis.pycortex_visualization(dataInfo, dt3d_stats_show, dt3d_sig_show, \
                                    showType='stats', colorType=colorType, max_stat=max_stat, \
                                    mask=mask, saveFigName=saveFigName, \
                                    with_rois=with_rois, with_sulci=with_sulci, with_curvature=with_curvature,
                                    with_colorbar=with_colorbar)


def get_roi_index():
    with open('fMRI_Narrative_movie/util/config__drama_data.yaml', 'r') as f_yml:
        config = yaml.safe_load(f_yml)

    roi = "MT+"
    subject_name = 'S02'  # S01, S02, ..., S06

    dataInfo = utlpy.get_dataInfo(config, subject_name)

    subjectID = udl.get_subjectID_from_subjectName(config, subject_name)
    load_items = udl.set_load_items(config, subjectID, 0)
    t_path_mask = load_items['mask_path']
    mask = udl.load_mask_data(t_path_mask)

    roi_masks = np.zeros(663552)
    roi_masks[mask] = 1
    roi_masks = roi_masks.reshape(72, 96, 96)

    # roi_masks = cortex.utils.get_roi_masks(dataInfo['subjectNamePycortex'], dataInfo['dataSetName'],
    #                                        roi_list=[roi], gm_sampler='cortical-conservative')

    roi_data = cortex.Volume(roi_masks, dataInfo['subjectNamePycortex'], dataInfo['dataSetName'], vmin=0, vmax=1,
                             cmap="inferno")
    cortex.quickflat.make_figure(roi_data, thick=1, with_curvature=True, with_colorbar=True)

    plt.show()

    # roi_mask_3d = roi_masks[roi]
    # roi_mask_voxelIDs = np.where(roi_mask_3d.flatten() > 0)[0]
    #
    # datasize = dataInfo['datasize']
    # voxels_in_epi_space__flatten = np.zeros(np.prod(datasize))
    # voxels_in_epi_space__flatten[roi_mask_voxelIDs] = 0.8
    # voxels_in_epi_space__3d = voxels_in_epi_space__flatten.reshape([datasize[2], datasize[1], datasize[0]])
    #
    # # cortex 3d
    # subjectID = np.where(np.array(config['subjectInfo']['names']) == subject_name)[0][0]
    # dtinfo = udl.set_load_items(config, subjectID=subjectID, movID=0)
    # tvoxels = udl.load_mask_data(dtinfo['mask_path'])
    # cortex__flatten = np.zeros(np.prod(datasize))
    # cortex__flatten[tvoxels] = 0.2
    # cortex__3d = cortex__flatten.reshape([datasize[2], datasize[1], datasize[0]])
    #
    # # Show the voxel IDs in the slice image.
    # vol_value = cortex__3d + voxels_in_epi_space__3d
    # plt.imshow(vol_value[25, :, :])
    # plt.show()


if __name__ == '__main__':
    draw_bivariate_legend(
        vmax_joint=0.15,
        save_path="encoding/encoding_av/S01_vpa_mean_across_pairs/vpa_bivariate_legend.png",
        min_joint_ev_threshold=0.01,
    )



