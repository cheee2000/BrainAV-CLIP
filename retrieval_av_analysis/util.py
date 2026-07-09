import os

import numpy as np
import torch


TOP_K_LIST = (1, 5, 10)


def compute_similarity_matrix(pred_feature, target_feature, eps=1e-8):
    pred = torch.as_tensor(pred_feature, dtype=torch.float32)
    target = torch.as_tensor(target_feature, dtype=torch.float32)
    pred = pred / pred.norm(dim=1, keepdim=True).clamp_min(eps)
    target = target / target.norm(dim=1, keepdim=True).clamp_min(eps)
    return pred @ target.T


def retrieval_accuracy(similarity_matrix, top_k_list=TOP_K_LIST):
    n_candidates = int(similarity_matrix.shape[1])
    max_k = min(max(top_k_list), n_candidates)
    _, indices = similarity_matrix.topk(max_k, dim=1, largest=True)
    indices = indices.cpu().numpy()
    n_samples = len(indices)

    top_k_acc = {}
    for k in top_k_list:
        k_eff = min(k, max_k)
        correct = 0
        for i in range(n_samples):
            if i in indices[i, :k_eff]:
                correct += 1
        top_k_acc[f"top_{k}_acc"] = float(correct / n_samples)
    return top_k_acc


def zscore(matrix, eps=1e-8):
    std = matrix.std()
    return (matrix - matrix.mean()) / (std + eps)


def evaluate_modality(pred_feature, target_feature):
    sim = compute_similarity_matrix(pred_feature, target_feature)
    acc = retrieval_accuracy(sim, top_k_list=TOP_K_LIST)
    return sim, acc


def evaluate_fusion(vis_sim, aud_sim, alpha):
    fused_sim = alpha * zscore(vis_sim) + (1.0 - alpha) * zscore(aud_sim)
    return retrieval_accuracy(fused_sim, top_k_list=TOP_K_LIST)


def orthogonal_projection_fusion(vis_sim, aud_sim, alpha):
    """Remove the component of audio scores linearly explained by visual scores,
    then fuse the orthogonalised audio with the original visual scores."""
    vis_z = zscore(vis_sim)
    aud_z = zscore(aud_sim)
    dot = (aud_z * vis_z).sum(dim=1, keepdim=True)
    vis_norm_sq = (vis_z * vis_z).sum(dim=1, keepdim=True).clamp_min(1e-8)
    aud_orth = aud_z - (dot / vis_norm_sq) * vis_z
    return alpha * vis_z + (1.0 - alpha) * aud_orth


def evaluate_fusion_confidence_dynamic(
    vis_sim,
    aud_sim,
    temperature=0.1,
    weight_rule="softmax",
    alpha_prior=None,
    dynamic_strength=0.5,
    eps=1e-8,
    return_fused_sim=False,
):
    vis = torch.as_tensor(vis_sim, dtype=torch.float32)
    aud = torch.as_tensor(aud_sim, dtype=torch.float32)
    if vis.shape != aud.shape:
        raise ValueError(
            f"Shape mismatch for dynamic fusion: vis={tuple(vis.shape)} aud={tuple(aud.shape)}"
        )
    if vis.ndim != 2:
        raise ValueError(f"Expected 2D similarity matrices, got ndim={vis.ndim}.")

    n_candidates = int(vis.shape[1])
    margin_topk = min(10, n_candidates)
    if alpha_prior is not None:
        alpha_prior = float(alpha_prior)
        alpha_prior = max(0.0, min(1.0, alpha_prior))
    dynamic_strength = float(dynamic_strength)
    dynamic_strength = max(0.0, min(1.0, dynamic_strength))

    if n_candidates <= 1:
        alpha_dynamic = torch.full((vis.shape[0],), 0.5, dtype=torch.float32, device=vis.device)
        vis_margin = torch.zeros_like(alpha_dynamic)
        aud_margin = torch.zeros_like(alpha_dynamic)
    else:
        vis_topk, _ = vis.topk(k=margin_topk, dim=1, largest=True)
        aud_topk, _ = aud.topk(k=margin_topk, dim=1, largest=True)
        if margin_topk <= 1:
            vis_margin = torch.zeros(vis.shape[0], dtype=torch.float32, device=vis.device)
            aud_margin = torch.zeros(aud.shape[0], dtype=torch.float32, device=aud.device)
        else:
            vis_margin = (vis_topk[:, 0] - vis_topk[:, 1:].mean(dim=1)).clamp_min(0.0)
            aud_margin = (aud_topk[:, 0] - aud_topk[:, 1:].mean(dim=1)).clamp_min(0.0)

        if weight_rule == "ratio":
            denom = vis_margin + aud_margin
            alpha_dynamic = torch.where(
                denom > eps,
                vis_margin / denom.clamp_min(eps),
                torch.full_like(denom, 0.5),
            )
        elif weight_rule == "softmax":
            if temperature <= 0:
                raise ValueError("temperature must be > 0 when weight_rule='softmax'.")
            logits = torch.stack(
                [vis_margin / float(temperature), aud_margin / float(temperature)],
                dim=1,
            )
            alpha_dynamic = torch.softmax(logits, dim=1)[:, 0]
        else:
            raise ValueError(f"Unsupported weight_rule: {weight_rule}")

    if alpha_prior is None:
        alpha = alpha_dynamic
    else:
        alpha_prior_tensor = torch.full_like(alpha_dynamic, float(alpha_prior))
        alpha = (1.0 - dynamic_strength) * alpha_prior_tensor + dynamic_strength * alpha_dynamic

    fused_sim = alpha.unsqueeze(1) * zscore(vis) + (1.0 - alpha).unsqueeze(1) * zscore(aud)
    acc = retrieval_accuracy(fused_sim, top_k_list=TOP_K_LIST)
    alpha_np = alpha.detach().cpu().numpy()
    alpha_dynamic_np = alpha_dynamic.detach().cpu().numpy()
    vis_margin_np = vis_margin.detach().cpu().numpy()
    aud_margin_np = aud_margin.detach().cpu().numpy()
    meta = {
        "alpha_prior": None if alpha_prior is None else float(alpha_prior),
        "dynamic_strength": float(dynamic_strength),
        "alpha_dynamic_mean": float(np.mean(alpha_dynamic_np)),
        "alpha_dynamic_std": float(np.std(alpha_dynamic_np)),
        "alpha_mean": float(np.mean(alpha_np)),
        "alpha_std": float(np.std(alpha_np)),
        "alpha_min": float(np.min(alpha_np)),
        "alpha_max": float(np.max(alpha_np)),
        "margin_visual_mean": float(np.mean(vis_margin_np)),
        "margin_audio_mean": float(np.mean(aud_margin_np)),
        "margin_topk": int(margin_topk),
        "weight_rule": weight_rule,
        "temperature": float(temperature),
    }
    if return_fused_sim:
        return acc, meta, fused_sim
    return acc, meta


def evaluate_joint_by_method(
    vis_sim,
    aud_sim,
    method,
    alpha=None,
    confidence_temperature=0.1,
    confidence_weight_rule="softmax",
    confidence_dynamic_strength=0.5,
    return_fused_sim=False,
):
    if method == "global_alpha":
        if alpha is None:
            raise ValueError("alpha is required for method='global_alpha'.")
        fused_sim = alpha * zscore(vis_sim) + (1.0 - alpha) * zscore(aud_sim)
        acc = retrieval_accuracy(fused_sim, top_k_list=TOP_K_LIST)
        meta = {
            "method": "global_alpha",
            "alpha_mean": float(alpha),
            "alpha_std": 0.0,
            "alpha_min": float(alpha),
            "alpha_max": float(alpha),
        }
        if return_fused_sim:
            return acc, meta, fused_sim
        return acc, meta
    if method == "orthogonal_alpha":
        if alpha is None:
            raise ValueError("alpha is required for method='orthogonal_alpha'.")
        vis_t = torch.as_tensor(vis_sim, dtype=torch.float32)
        aud_t = torch.as_tensor(aud_sim, dtype=torch.float32)
        fused_sim = orthogonal_projection_fusion(vis_t, aud_t, float(alpha))
        acc = retrieval_accuracy(fused_sim, top_k_list=TOP_K_LIST)
        meta = {
            "method": "orthogonal_alpha",
            "alpha_mean": float(alpha),
            "alpha_std": 0.0,
            "alpha_min": float(alpha),
            "alpha_max": float(alpha),
        }
        if return_fused_sim:
            return acc, meta, fused_sim
        return acc, meta
    if method == "confidence_dynamic":
        acc, meta, fused_sim = evaluate_fusion_confidence_dynamic(
            vis_sim=vis_sim,
            aud_sim=aud_sim,
            temperature=confidence_temperature,
            weight_rule=confidence_weight_rule,
            alpha_prior=alpha,
            dynamic_strength=confidence_dynamic_strength,
            return_fused_sim=True,
        )
        meta["method"] = "confidence_dynamic"
        if return_fused_sim:
            return acc, meta, fused_sim
        return acc, meta
    raise ValueError(f"Unknown fusion method: {method}")


def topk_retrieval_reference_similarity(
    similarity_matrix,
    reference_features,
    top_k=10,
    eps=1e-8,
):
    sim = torch.as_tensor(similarity_matrix, dtype=torch.float32)
    ref = torch.as_tensor(reference_features, dtype=torch.float32)
    if sim.ndim != 2:
        raise ValueError(f"similarity_matrix should be 2D, got ndim={sim.ndim}.")
    if ref.ndim != 2:
        raise ValueError(f"reference_features should be 2D, got ndim={ref.ndim}.")
    if sim.shape[0] != sim.shape[1]:
        raise ValueError(
            f"Expected square similarity matrix for retrieval, got {tuple(sim.shape)}."
        )
    if ref.shape[0] != sim.shape[0]:
        raise ValueError(
            f"reference_features sample num mismatch: ref={ref.shape[0]} vs sim={sim.shape[0]}."
        )
    top_k = int(top_k)
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}.")

    k_eff = min(top_k, int(sim.shape[1]))
    _, topk_idx = sim.topk(k=k_eff, dim=1, largest=True)
    query_ref = ref / ref.norm(dim=1, keepdim=True).clamp_min(eps)
    retrieved_ref = ref[topk_idx]
    retrieved_ref = retrieved_ref / retrieved_ref.norm(dim=2, keepdim=True).clamp_min(eps)
    cos_sim = (retrieved_ref * query_ref.unsqueeze(1)).sum(dim=2)
    return float(cos_sim.mean().item())


def get_feat_name_by_prefix_suffix(folder_name, prefix, suffix):
    if not folder_name.startswith(prefix):
        return None
    if suffix and (not folder_name.endswith(suffix)):
        return None
    if suffix:
        return folder_name[len(prefix) : -len(suffix)]
    return folder_name[len(prefix) :]


def collect_feature_folders(log_dir, feat_type, prefix, suffix):
    feat_root = os.path.join(log_dir, feat_type)
    if not os.path.isdir(feat_root):
        raise FileNotFoundError(f"{feat_type} dir not found: {feat_root}")

    items = []
    for folder_name in sorted(os.listdir(feat_root)):
        feat_name = get_feat_name_by_prefix_suffix(folder_name, prefix, suffix)
        if feat_name is None:
            continue
        exp_path = os.path.join(feat_root, folder_name)
        if not os.path.isdir(exp_path):
            continue
        items.append(
            {
                "feat_name": feat_name,
                "folder_name": folder_name,
                "exp_path": exp_path,
            }
        )
    return items


def resolve_audio_feat_names(log_dir, aud_prefix, aud_suffix, audio_feat_names=None):
    if audio_feat_names is not None:
        cleaned = []
        for n in audio_feat_names:
            if n is None:
                continue
            n = str(n).strip()
            if n.lower() in {"none", "null"}:
                continue
            if n:
                cleaned.append(n)
        unique_names = sorted(set(cleaned))
        if unique_names:
            return unique_names

    audio_items = collect_feature_folders(
        log_dir=log_dir,
        feat_type="Audio_Model",
        prefix=aud_prefix,
        suffix=aud_suffix,
    )
    if not audio_items:
        raise RuntimeError("No Audio_Model experiments matched aud_prefix/aud_suffix.")
    return sorted({item["feat_name"] for item in audio_items})


def resolve_visual_feat_names(log_dir, vis_prefix, vis_suffix, visual_feat_names=None):
    if visual_feat_names is not None:
        cleaned = []
        for n in visual_feat_names:
            if n is None:
                continue
            n = str(n).strip()
            if n.lower() in {"none", "null"}:
                continue
            if n:
                cleaned.append(n)
        unique_names = sorted(set(cleaned))
        if unique_names:
            return unique_names

    visual_items = collect_feature_folders(
        log_dir=log_dir,
        feat_type="Visual_Model",
        prefix=vis_prefix,
        suffix=vis_suffix,
    )
    if not visual_items:
        raise RuntimeError("No Visual_Model experiments matched vis_prefix/vis_suffix.")
    return sorted({item["feat_name"] for item in visual_items})


def collect_visual_audio_pairs(
    log_dir,
    vis_prefix,
    vis_suffix,
    aud_prefix,
    aud_suffix,
    visual_feat_names=None,
    audio_feat_names=None,
):
    visual_items = collect_feature_folders(
        log_dir=log_dir,
        feat_type="Visual_Model",
        prefix=vis_prefix,
        suffix=vis_suffix,
    )
    if not visual_items:
        raise RuntimeError("No Visual_Model experiments matched vis_prefix/vis_suffix.")

    audio_root = os.path.join(log_dir, "Audio_Model")
    if not os.path.isdir(audio_root):
        raise FileNotFoundError(f"Audio model dir not found: {audio_root}")

    resolved_visual_names = resolve_visual_feat_names(
        log_dir=log_dir,
        vis_prefix=vis_prefix,
        vis_suffix=vis_suffix,
        visual_feat_names=visual_feat_names,
    )
    resolved_visual_name_set = set(resolved_visual_names)
    visual_items = [it for it in visual_items if it["feat_name"] in resolved_visual_name_set]
    if not visual_items:
        raise RuntimeError("No Visual_Model experiments remained after visual_feat_names filtering.")

    resolved_audio_names = resolve_audio_feat_names(
        log_dir=log_dir,
        aud_prefix=aud_prefix,
        aud_suffix=aud_suffix,
        audio_feat_names=audio_feat_names,
    )

    pairs = []
    for vis_item in visual_items:
        for audio_feat_name in resolved_audio_names:
            audio_folder_name = f"{aud_prefix}{audio_feat_name}{aud_suffix}"
            aud_exp_path = os.path.join(audio_root, audio_folder_name)
            pairs.append(
                {
                    "visual_feat_name": vis_item["feat_name"],
                    "visual_folder_name": vis_item["folder_name"],
                    "visual_exp_path": vis_item["exp_path"],
                    "audio_feat_name": audio_feat_name,
                    "audio_folder_name": audio_folder_name,
                    "audio_exp_path": aud_exp_path,
                    "audio_exists": os.path.isdir(aud_exp_path),
                    "pair_name": f"{vis_item['feat_name']}__{audio_feat_name}",
                }
            )
    return pairs, resolved_visual_names, resolved_audio_names


def load_brain_predictions(exp_path):
    return {
        "val": np.load(os.path.join(exp_path, "val_brain_features.npy")),
        "test": np.load(os.path.join(exp_path, "test_brain_features.npy")),
        "test_unseen": np.load(os.path.join(exp_path, "test_unseen_brain_features.npy")),
    }


def search_best_alpha(vis_sim_val, aud_sim_val, alphas, optimize_top_k=10, fusion_method="global_alpha"):
    vis_t = torch.as_tensor(vis_sim_val, dtype=torch.float32) if not isinstance(vis_sim_val, torch.Tensor) else vis_sim_val
    aud_t = torch.as_tensor(aud_sim_val, dtype=torch.float32) if not isinstance(aud_sim_val, torch.Tensor) else aud_sim_val
    use_orth = (fusion_method == "orthogonal_alpha")
    curve = {f"top_{k}": [] for k in TOP_K_LIST}

    for alpha in alphas:
        if use_orth:
            fused = orthogonal_projection_fusion(vis_t, aud_t, float(alpha))
        else:
            fused = float(alpha) * zscore(vis_t) + (1.0 - float(alpha)) * zscore(aud_t)
        acc = retrieval_accuracy(fused, top_k_list=TOP_K_LIST)
        for k in TOP_K_LIST:
            curve[f"top_{k}"].append(float(acc[f"top_{k}_acc"]))

    best_alpha_by_topk = {}
    best_val_joint_acc_by_topk = {}
    for k in TOP_K_LIST:
        key = f"top_{k}"
        best_idx = int(np.argmax(curve[key]))
        best_alpha_by_topk[key] = float(alphas[best_idx])
        best_val_joint_acc_by_topk[key] = float(curve[key][best_idx])
    if f"top_{optimize_top_k}" not in best_alpha_by_topk:
        raise ValueError(f"optimize_top_k={optimize_top_k} is not in TOP_K_LIST={TOP_K_LIST}.")
    return best_alpha_by_topk, best_val_joint_acc_by_topk, curve


def collect_guided_visual_pairs(log_dir, prefix, suffix):
    visual_items = collect_feature_folders(
        log_dir=log_dir,
        feat_type="Visual_Model",
        prefix=prefix,
        suffix=suffix,
    )
    pair_items = []
    for item in visual_items:
        raw_name = item["feat_name"]
        if "+" not in raw_name:
            continue
        visual_feat_name, aux_feat_name = raw_name.rsplit("+", 1)
        visual_feat_name = visual_feat_name.strip()
        aux_feat_name = aux_feat_name.strip()
        if not visual_feat_name or not aux_feat_name:
            continue
        pair_items.append(
            {
                "visual_feat_name": visual_feat_name,
                "aux_feat_name": aux_feat_name,
                "folder_name": item["folder_name"],
                "exp_path": item["exp_path"],
            }
        )
    return pair_items


def parse_acc_triplet(dataset_metrics, modality):
    one = float(dataset_metrics[modality]["top_1_acc"])
    five = float(dataset_metrics[modality]["top_5_acc"])
    ten = float(dataset_metrics[modality]["top_10_acc"])
    return {"top_1_acc": one, "top_5_acc": five, "top_10_acc": ten}
