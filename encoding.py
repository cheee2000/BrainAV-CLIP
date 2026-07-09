import json
import os
from time import time
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import check_cv
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import pearsonr
from himalaya.ridge import RidgeCV, GroupRidgeCV
from himalaya.ridge import ColumnTransformerNoStack
from himalaya.kernel_ridge import KernelRidgeCV
from himalaya.backend import set_backend
from himalaya.viz import plot_alphas_diagnostic

backend = set_backend("torch_cuda", on_error="warn")
from scipy.stats import zscore
from voxelwise_tutorials.delayer import Delayer

from dataset import (
    prepare_fmri_and_stim_same_movie,
    prepare_fmri_and_stim_test,
    prepare_stim_same_movie,
    prepare_stim_test,
)
from fMRI_Narrative_movie.util import util_ridge as uridge
from visualize import (
    visualize_masked_weight,
    get_roi_weight_mean_dict,
    visualize_unique_ratio_rgb_brain,
    visualize_bivariate_ev_brain,
    draw_bivariate_legend,
    visualize_discrete_modality_masks_rgb,
    draw_tricolor_circle_mixing_colorbar,
)
from fMRI_Narrative_movie.util.util_ridge import get_sig_voxel_inds


# FEAT_PATH = "PATH/TO/Narrative_Movie_fMRI_Dataset/derivatives/feat/"
FEAT_PATH = "PATH/TO/Narrative_Movie_fMRI_Dataset/derivatives/feat/"


def encoding(feat_name_list=None, feat_type='Visual_Model', subject_name='S02', base_proj_dir="PATH/TO/project/RetrievalAV/encoding/encoding_img"):
    base_feat_type = 'Visual_Model'
    base_feat_name = 'clip_base_img'          # 'Qwen3-VL-Embedding-8B'

    # feat_path = "PATH/TO/Narrative_Movie_fMRI_Dataset/derivatives/feat"
    # feat_path = "PATH/TO/Narrative_Movie_fMRI_Dataset/stimuli/feature/"
    if feat_name_list is None:
        feat_name_list = []  # 'Qwen3-VL-Embedding-8B'

    print('Loading fMRI data..')
    start_time = time()
    Resp_train, _, Resp_val, _, Resp_test, _, _ = \
        prepare_fmri_and_stim_same_movie(
            subject_name,
            feat_name=base_feat_name,
            feat_type=base_feat_type,
            return_volume=False,
            feat_path=FEAT_PATH,
        )
    Resp_test_unseen, _ = prepare_fmri_and_stim_test(subject_name, feat_name=base_feat_name,
                                                                           feat_type=base_feat_type, return_volume=False,
                                                                           feat_path=FEAT_PATH)
    print('Time for loading fMRI data: %f' % (time() - start_time) + ' seconds')

    Resp_train = Resp_train.reshape(len(Resp_train) // 5, 5, -1).mean(1)
    Resp_val = Resp_val.reshape(len(Resp_val) // 5, 5, -1).mean(1)
    Resp_test = Resp_test.reshape(len(Resp_test) // 5, 5, -1).mean(1)
    Resp_test_unseen = Resp_test_unseen.reshape(len(Resp_test_unseen) // 5, 5, -1).mean(1)
    print("(n_samples_train, n_voxels) =", Resp_train.shape)
    print("(n_samples_val, n_voxels) =", Resp_val.shape)
    print("(n_samples_test, n_voxels) =", Resp_test.shape)
    print("(n_samples_test_unseen, n_voxels) =", Resp_test_unseen.shape)

    # print("(n_samples_train, n_features) =", Stim_train.shape)
    # print("(n_samples_test, n_features) =", Stim_test.shape)
    # print("(n_samples_test_unseen, n_features) =", Stim_test_unseen.shape)

    n_samples_train = Resp_train.shape[0]

    ###
    ### Train
    onsets = []
    # TODO alphas
    alphas = np.logspace(-4, 15, 20)
    intercept = True  # Seeing False of True
    nCV = 5

    # n_samples_ = udl.get_n_samples_from_movIDs(config, movIDs_train)
    cv_shuffled_samples, cv_chunk_onset = uridge.get_subset_samples_and_onset(n_samples_train, chunkLen=100, seedno=33)

    # (train data) shuffle resp and stim (100s chunk): and divided for 5-fold cv
    if len(onsets) == 0:
        cv_onsets = [np.array_split(cv_chunk_onset, nCV)[cv_id][0] for cv_id in range(nCV)]
    else:  # 'nCV' is not used in this case.
        cv_onsets = onsets
    cv = uridge.generate_leave_one_run_out(n_samples_train, cv_onsets, random_state=0, n_runs_out=1)
    cv = check_cv(cv)
    ncv_save = len(cv_onsets)

    print('Model fitting for multiple features ...')

    for feat_name in feat_name_list:
        print(f'\n========== Encoding for feature: {feat_name} =========')

        # 1) load Stim for the current feature via dataset
        Stim_train, Stim_val, Stim_test = prepare_stim_same_movie(
            feat_name=feat_name,
            feat_type=feat_type,
            feat_path=FEAT_PATH,
        )
        Stim_test_unseen = prepare_stim_test(
            feat_name=feat_name,
            feat_type=feat_type,
            feat_path=FEAT_PATH,
        )

        # 2) check Stim/Resp alignment along time
        assert Stim_train.shape[0] == Resp_train.shape[0]
        assert Stim_val.shape[0] == Resp_val.shape[0]
        assert Stim_test.shape[0] == Resp_test.shape[0]
        assert Stim_test_unseen.shape[0] == Resp_test_unseen.shape[0]

        # 3)
        print('Model fitting ...')
        scaler = StandardScaler(with_mean=True, with_std=False)
        # delayer = Delayer(delays=[1, 2, 3, 4, 5, 6])
        kernel_ridge_cv = KernelRidgeCV(
            alphas=alphas, cv=cv,
            solver_params=dict(n_targets_batch=500, n_alphas_batch=5,
                            n_targets_batch_refit=100))

        pipeline = make_pipeline(
            scaler,
            # delayer,
            kernel_ridge_cv,
        )

        _ = pipeline.fit(Stim_train[cv_shuffled_samples, :], Resp_train[cv_shuffled_samples, :])

        print('done.')

        # ====== compute val voxel-wise PCC and p-values ======
        p_Resp_val = pipeline.predict(Stim_val)
        rs_val = []
        ps_val = []
        print('\n[val] Prediction: Calc rs and ps ...')
        for i in range(p_Resp_val.shape[1]):
            r_, p_ = pearsonr(p_Resp_val[:, i], Resp_val[:, i])
            rs_val.append(r_)
            ps_val.append(p_)
        rs_val = np.array(rs_val)
        ps_val = np.array(ps_val)
        print('[val] done.')

        # ====== compute val voxel-wise explained variance (R^2)======
        ev_val = pipeline.score(Stim_val, Resp_val)
        ev_val = backend.to_numpy(ev_val)
        print(f"[val] Explained variance mean/max: {ev_val.mean():.6f}/{ev_val.max():.6f}")

        # ====== test / test_unseen evaluation ======
        p_Resp_test = pipeline.predict(Stim_test)
        rs = []
        ps = []
        print('\n[test] Prediction: Calc rs and ps ...')
        for i in range(p_Resp_test.shape[1]):
            r_, p_ = pearsonr(p_Resp_test[:, i], Resp_test[:, i])
            rs.append(r_)
            ps.append(p_)
        rs = np.array(rs)
        ps = np.array(ps)
        print('[test] done.')
        
        p_Resp_test_unseen = pipeline.predict(Stim_test_unseen)
        rs_unseen = []
        ps_unseen = []
        print('\n[test_unseen] Prediction: Calc rs and ps ...')
        for i in range(p_Resp_test_unseen.shape[1]):
            r_, p_ = pearsonr(p_Resp_test_unseen[:, i], Resp_test_unseen[:, i])
            rs_unseen.append(r_)
            ps_unseen.append(p_)
        rs_unseen = np.array(rs_unseen)
        ps_unseen = np.array(ps_unseen)
        print('[test_unseen] done.')
        
        ev_test = pipeline.score(Stim_test, Resp_test)
        ev_test = backend.to_numpy(ev_test)
        ev_test_unseen = pipeline.score(Stim_test_unseen, Resp_test_unseen)
        ev_test_unseen = backend.to_numpy(ev_test_unseen)
        print(f"[test] Explained variance mean/max: {ev_test.mean():.6f}/{ev_test.max():.6f}")
        print(f"[test_unseen] Explained variance mean/max: {ev_test_unseen.mean():.6f}/{ev_test_unseen.max():.6f}")

        # ====== save PCC and p-values ======
        proj_path = (f"{base_proj_dir}/{subject_name}_{feat_name}/")
        os.makedirs(proj_path, exist_ok=True)

        np.save(os.path.join(proj_path, "val_voxel_pcc.npy"), rs_val)
        np.save(os.path.join(proj_path, "val_voxel_p_value.npy"), ps_val)
        np.save(os.path.join(proj_path, "val_voxel_explained_variance.npy"), ev_val)
        # np.save(os.path.join(proj_path, "test_voxel_pcc.npy"), rs)
        # np.save(os.path.join(proj_path, "test_voxel_p_value.npy"), ps)
        # np.save(os.path.join(proj_path, "test_unseen_voxel_pcc.npy"), rs_unseen)
        # np.save(os.path.join(proj_path, "test_unseen_voxel_p_value.npy"), ps_unseen)
        # np.save(os.path.join(proj_path, "test_voxel_explained_variance.npy"), ev_test)
        # np.save(os.path.join(proj_path, "test_unseen_voxel_explained_variance.npy"), ev_test_unseen)

        # sig_positive_mask = (ps < 0.05) & (rs > 0)
        # weights = np.full_like(rs, np.nan, dtype=float)
        # weights[sig_positive_mask] = rs[sig_positive_mask]
        #
        # print(f'[test] voxels with p<0.05 and pcc>0: {np.sum(sig_positive_mask)} / {rs.shape[0]}')
        # if np.sum(sig_positive_mask) > 0:
        #     print(f'[test] selected voxel PCC range: [{np.nanmin(weights):.4f}, {np.nanmax(weights):.4f}]')
        #
        # # visualize_masked_weight(
        # #     weights,
        # #     proj_path + 'upp_pcc_test.png',
        # #     subject_name=subject_name,
        # #     vmin=0,
        # #     vmax=0.5,
        # # )
        # roi_mean_dict = get_roi_weight_mean_dict(weights, roi_names, subject_name=subject_name, split_lr=True)
        # print('[test] ROI mean:', roi_mean_dict)
        # with open(os.path.join(proj_path, 'upp_pcc_test_roi_mean.json'), 'w') as f:
        #     json.dump(roi_mean_dict, f)
        #
        # sig_positive_mask_unseen = (ps_unseen < 0.05) & (rs_unseen > 0)
        # weights_unseen = np.full_like(rs_unseen, np.nan, dtype=float)
        # weights_unseen[sig_positive_mask_unseen] = rs_unseen[sig_positive_mask_unseen]
        #
        # print(f'[test_unseen] voxels with p<0.05 and pcc>0: {np.sum(sig_positive_mask_unseen)} / {rs_unseen.shape[0]}')
        # if np.sum(sig_positive_mask_unseen) > 0:
        #     print(f'[test_unseen] selected voxel PCC range: [{np.nanmin(weights_unseen):.4f}, {np.nanmax(weights_unseen):.4f}]')
        #
        # # visualize_masked_weight(
        # #     weights_unseen,
        # #     proj_path + 'upp_pcc_test_unseen.png',
        # #     subject_name=subject_name,
        # #     vmin=0,
        # #     vmax=0.5,
        # # )
        # roi_mean_dict_unseen = get_roi_weight_mean_dict(
        #     weights_unseen, roi_names, subject_name=subject_name, split_lr=True
            # print('[test_unseen] ROI mean:', roi_mean_dict_unseen)
        # with open(os.path.join(proj_path, 'upp_pcc_test_unseen_roi_mean.json'), 'w') as f:
        #     json.dump(roi_mean_dict_unseen, f)
        #
        # best_alphas = backend.to_numpy(pipeline[-1].best_alphas_)
        # plot_alphas_diagnostic(best_alphas=best_alphas, alphas=alphas)
        #
        # plt.savefig(proj_path + 'best_alpha.png')


def av_encoding(
    vis_feat_name_list=None,
    aud_feat_name_list=None,
    vis_feat_type='Visual_Model',
    aud_feat_type='Audio_Model',
    subject_name='S02',
    base_proj_dir="./encoding/encoding_av",
):
    base_feat_type = 'Visual_Model'
    base_feat_name = 'clip_base_img'

    if vis_feat_name_list is None:
        vis_feat_name_list = []
    if aud_feat_name_list is None:
        aud_feat_name_list = []

    if len(vis_feat_name_list) == 0 or len(aud_feat_name_list) == 0:
        raise ValueError("vis_feat_name_list and aud_feat_name_list must not be empty.")

    print('Loading fMRI data..')
    start_time = time()
    Resp_train, _, Resp_val, _, Resp_test, _, _ = prepare_fmri_and_stim_same_movie(
        subject_name, feat_name=base_feat_name, feat_type=base_feat_type, return_volume=False, feat_path=FEAT_PATH
    )
    Resp_test_unseen, _ = prepare_fmri_and_stim_test(
        subject_name, feat_name=base_feat_name, feat_type=base_feat_type, return_volume=False, feat_path=FEAT_PATH
    )
    print('Time for loading fMRI data: %f' % (time() - start_time) + ' seconds')

    Resp_train = Resp_train.reshape(len(Resp_train) // 5, 5, -1).mean(1)
    Resp_val = Resp_val.reshape(len(Resp_val) // 5, 5, -1).mean(1)
    Resp_test = Resp_test.reshape(len(Resp_test) // 5, 5, -1).mean(1)
    Resp_test_unseen = Resp_test_unseen.reshape(len(Resp_test_unseen) // 5, 5, -1).mean(1)
    print("(n_samples_train, n_voxels) =", Resp_train.shape)
    print("(n_samples_val, n_voxels) =", Resp_val.shape)
    print("(n_samples_test, n_voxels) =", Resp_test.shape)
    print("(n_samples_test_unseen, n_voxels) =", Resp_test_unseen.shape)

    n_samples_train = Resp_train.shape[0]

    onsets = []
    alphas = np.logspace(-4, 15, 20)
    nCV = 5

    cv_shuffled_samples, cv_chunk_onset = uridge.get_subset_samples_and_onset(
        n_samples_train, chunkLen=100, seedno=33
    )
    if len(onsets) == 0:
        cv_onsets = [np.array_split(cv_chunk_onset, nCV)[cv_id][0] for cv_id in range(nCV)]
    else:
        cv_onsets = onsets
    cv = uridge.generate_leave_one_run_out(n_samples_train, cv_onsets, random_state=0, n_runs_out=1)
    cv = check_cv(cv)

    print('Model fitting for AV cartesian-product features ...')

    for vis_feat_name in vis_feat_name_list:
        vis_Stim_train, vis_Stim_val, vis_Stim_test = prepare_stim_same_movie(
            feat_name=vis_feat_name,
            feat_type=vis_feat_type,
            feat_path=FEAT_PATH,
        )
        vis_Stim_test_unseen = prepare_stim_test(
            feat_name=vis_feat_name,
            feat_type=vis_feat_type,
            feat_path=FEAT_PATH,
        )

        assert vis_Stim_train.shape[0] == Resp_train.shape[0]
        assert vis_Stim_val.shape[0] == Resp_val.shape[0]
        assert vis_Stim_test.shape[0] == Resp_test.shape[0]
        assert vis_Stim_test_unseen.shape[0] == Resp_test_unseen.shape[0]

        for aud_feat_name in aud_feat_name_list:
            print(f'\n========== AV Encoding: {vis_feat_name} + {aud_feat_name} =========')

            aud_Stim_train, aud_Stim_val, aud_Stim_test = prepare_stim_same_movie(
                feat_name=aud_feat_name,
                feat_type=aud_feat_type,
                feat_path=FEAT_PATH,
            )
            aud_Stim_test_unseen = prepare_stim_test(
                feat_name=aud_feat_name,
                feat_type=aud_feat_type,
                feat_path=FEAT_PATH,
            )

            assert aud_Stim_train.shape[0] == Resp_train.shape[0]
            assert aud_Stim_val.shape[0] == Resp_val.shape[0]
            assert aud_Stim_test.shape[0] == Resp_test.shape[0]
            assert aud_Stim_test_unseen.shape[0] == Resp_test_unseen.shape[0]
            assert vis_Stim_train.shape[0] == aud_Stim_train.shape[0]
            assert vis_Stim_val.shape[0] == aud_Stim_val.shape[0]
            assert vis_Stim_test.shape[0] == aud_Stim_test.shape[0]
            assert vis_Stim_test_unseen.shape[0] == aud_Stim_test_unseen.shape[0]

            Stim_train = np.concatenate([vis_Stim_train, aud_Stim_train], axis=1)
            Stim_val = np.concatenate([vis_Stim_val, aud_Stim_val], axis=1)
            Stim_test = np.concatenate([vis_Stim_test, aud_Stim_test], axis=1)
            Stim_test_unseen = np.concatenate([vis_Stim_test_unseen, aud_Stim_test_unseen], axis=1)

            vis_dim = vis_Stim_train.shape[1]
            aud_dim = aud_Stim_train.shape[1]
            vis_slice = slice(0, vis_dim)
            aud_slice = slice(vis_dim, vis_dim + aud_dim)

            print('Model fitting ...')
            preprocess_pipeline = make_pipeline(StandardScaler(with_mean=True, with_std=False))
            ct = ColumnTransformerNoStack(
                [
                    ("visual", preprocess_pipeline, vis_slice),
                    ("audio", preprocess_pipeline, aud_slice),
                ]
            )

            solver_params = dict(
                alphas=alphas,
                n_iter=20,
                n_targets_batch=500,
                n_alphas_batch=5,
                n_targets_batch_refit=100,
            )
            group_ridge_cv = GroupRidgeCV(
                groups="input",
                cv=cv,
                solver="random_search",
                solver_params=solver_params,
                random_state=42,
            )
            pipeline = make_pipeline(ct, group_ridge_cv)
            _ = pipeline.fit(Stim_train[cv_shuffled_samples, :], Resp_train[cv_shuffled_samples, :])

            print('done.')

            p_Resp_val = pipeline.predict(Stim_val)
            rs_val = []
            ps_val = []
            print('\n[val] Prediction: Calc rs and ps ...')
            for i in range(p_Resp_val.shape[1]):
                r_, p_ = pearsonr(p_Resp_val[:, i], Resp_val[:, i])
                rs_val.append(r_)
                ps_val.append(p_)
            rs_val = np.array(rs_val)
            ps_val = np.array(ps_val)
            print('[val] done.')

            # ====== compute val voxel-wise explained variance (R^2)======
            ev_val = pipeline.score(Stim_val, Resp_val)
            ev_val = backend.to_numpy(ev_val)
            print(f"[val] Explained variance mean/max: {ev_val.mean():.6f}/{ev_val.max():.6f}")

            # ====== test / test_unseen evaluation disabled ======
            # p_Resp_test = pipeline.predict(Stim_test)
            # rs = []
            # ps = []
            # print('\n[test] Prediction: Calc rs and ps ...')
            # for i in range(p_Resp_test.shape[1]):
            #     r_, p_ = pearsonr(p_Resp_test[:, i], Resp_test[:, i])
            #     rs.append(r_)
            #     ps.append(p_)
            # rs = np.array(rs)
            # ps = np.array(ps)
            # print('[test] done.')
            #
            # p_Resp_test_unseen = pipeline.predict(Stim_test_unseen)
            # rs_unseen = []
            # ps_unseen = []
            # print('\n[test_unseen] Prediction: Calc rs and ps ...')
            # for i in range(p_Resp_test_unseen.shape[1]):
            #     r_, p_ = pearsonr(p_Resp_test_unseen[:, i], Resp_test_unseen[:, i])
            #     rs_unseen.append(r_)
            #     ps_unseen.append(p_)
            # rs_unseen = np.array(rs_unseen)
            # ps_unseen = np.array(ps_unseen)
            # print('[test_unseen] done.')
            #
            # ev_test = pipeline.score(Stim_test, Resp_test)
            # ev_test = backend.to_numpy(ev_test)
            # ev_test_unseen = pipeline.score(Stim_test_unseen, Resp_test_unseen)
            # ev_test_unseen = backend.to_numpy(ev_test_unseen)
            # print(f"[test] Explained variance mean/max: {ev_test.mean():.6f}/{ev_test.max():.6f}")
            # print(f"[test_unseen] Explained variance mean/max: {ev_test_unseen.mean():.6f}/{ev_test_unseen.max():.6f}")

            proj_path = f"{base_proj_dir}/{subject_name}_{vis_feat_name}+{aud_feat_name}/"
            os.makedirs(proj_path, exist_ok=True)

            np.save(os.path.join(proj_path, "val_voxel_pcc.npy"), rs_val)
            np.save(os.path.join(proj_path, "val_voxel_p_value.npy"), ps_val)
            np.save(os.path.join(proj_path, "val_voxel_explained_variance.npy"), ev_val)
            # np.save(os.path.join(proj_path, "test_voxel_pcc.npy"), rs)
            # np.save(os.path.join(proj_path, "test_voxel_p_value.npy"), ps)
            # np.save(os.path.join(proj_path, "test_unseen_voxel_pcc.npy"), rs_unseen)
            # np.save(os.path.join(proj_path, "test_unseen_voxel_p_value.npy"), ps_unseen)
            # np.save(os.path.join(proj_path, "test_voxel_explained_variance.npy"), ev_test)
            # np.save(os.path.join(proj_path, "test_unseen_voxel_explained_variance.npy"), ev_test_unseen)

            # sig_positive_mask = (ps < 0.05) & (rs > 0)
            # weights = np.full_like(rs, np.nan, dtype=float)
            # weights[sig_positive_mask] = rs[sig_positive_mask]
            #
            # print(f'[test] voxels with p<0.05 and pcc>0: {np.sum(sig_positive_mask)} / {rs.shape[0]}')
            # if np.sum(sig_positive_mask) > 0:
            #     print(f'[test] selected voxel PCC range: [{np.nanmin(weights):.4f}, {np.nanmax(weights):.4f}]')
            #
            # # visualize_masked_weight(
            # #     weights,
            # #     proj_path + 'upp_pcc_test.png',
            # #     subject_name=subject_name,
            # #     vmin=0,
            # #     vmax=0.5,
            # # )
            # roi_mean_dict = get_roi_weight_mean_dict(weights, roi_names, subject_name=subject_name, split_lr=True)
            # print('[test] ROI mean:', roi_mean_dict)
            # with open(os.path.join(proj_path, 'upp_pcc_test_roi_mean.json'), 'w') as f:
            #     json.dump(roi_mean_dict, f)
            #
            # sig_positive_mask_unseen = (ps_unseen < 0.05) & (rs_unseen > 0)
            # weights_unseen = np.full_like(rs_unseen, np.nan, dtype=float)
            # weights_unseen[sig_positive_mask_unseen] = rs_unseen[sig_positive_mask_unseen]
            #
            # print(f'[test_unseen] voxels with p<0.05 and pcc>0: {np.sum(sig_positive_mask_unseen)} / {rs_unseen.shape[0]}')
            # if np.sum(sig_positive_mask_unseen) > 0:
            #     print(f'[test_unseen] selected voxel PCC range: [{np.nanmin(weights_unseen):.4f}, {np.nanmax(weights_unseen):.4f}]')
            #
            # # visualize_masked_weight(
            # #     weights_unseen,
            # #     proj_path + 'upp_pcc_test_unseen.png',
            # #     subject_name=subject_name,
            # #     vmin=0,
            # #     vmax=0.5,
            # # )
            # roi_mean_dict_unseen = get_roi_weight_mean_dict(
            #     weights_unseen, roi_names, subject_name=subject_name, split_lr=True
                    # print('[test_unseen] ROI mean:', roi_mean_dict_unseen)
            # with open(os.path.join(proj_path, 'upp_pcc_test_unseen_roi_mean.json'), 'w') as f:
            #     json.dump(roi_mean_dict_unseen, f)
            #
            # best_alphas = backend.to_numpy(pipeline[-1].best_alphas_)
            # plot_alphas_diagnostic(best_alphas=best_alphas, alphas=alphas)
            # plt.savefig(proj_path + 'best_alpha.png')


def vpa_analysis_from_saved_encoding(
    subject_name='S02',
    av_base_dir="./encoding/encoding_av",
    vis_base_dir="./encoding/encoding_img",
    aud_base_dir="./encoding/encoding_aud",
    roi_list=None,
):
    """
    VPA decomposition from saved single-modal and joint encoding explained variance (R^2):
    - U_V = R^2_VA - R^2_A
    - U_A = R^2_VA - R^2_V
    - S   = R^2_V + R^2_A - R^2_VA
    """
    if roi_list is None:
        roi_list = globals().get('roi_names', ['V1', 'V2', 'V3', 'FFA', 'OFA', 'PPA', 'MT+', 'AC_L', 'AC_R'])

    # ====== VPA tunable parameters======
    JOINT_EV_MIN_THRESHOLD = 0.01

    split_to_file = {
        'val': "val_voxel_explained_variance.npy",
        # 'test': "test_voxel_explained_variance.npy",
        # 'test_unseen': "test_unseen_voxel_explained_variance.npy",
    }

    # use vis_feat_names / aud_feat_names from __main__
    try:
        selected_vis_feat_names = set(vis_feat_names)
        selected_aud_feat_names = set(aud_feat_names)
    except NameError as e:
        raise NameError(
            "Define global vis_feat_names and aud_feat_names before calling vpa_analysis_from_saved_encoding."
        ) from e

    subject_prefix = f"{subject_name}_"
    pair_records = []
    if not os.path.exists(av_base_dir):
        raise FileNotFoundError(f"Joint encoding directory not found: {av_base_dir}")

    for entry in sorted(os.listdir(av_base_dir)):
        entry_path = os.path.join(av_base_dir, entry)
        if not (os.path.isdir(entry_path) and entry.startswith(subject_prefix)):
            continue
        pair_part = entry[len(subject_prefix):]
        if '+' not in pair_part:
            continue
        vis_feat_name, aud_feat_name = pair_part.split('+', 1)
        if vis_feat_name not in selected_vis_feat_names or aud_feat_name not in selected_aud_feat_names:
            continue
        pair_records.append(
            {
                "pair_name": pair_part,
                "vis_feat_name": vis_feat_name,
                "aud_feat_name": aud_feat_name,
                "av_dir": entry_path,
                "vis_dir": os.path.join(vis_base_dir, f"{subject_name}_{vis_feat_name}"),
                "aud_dir": os.path.join(aud_base_dir, f"{subject_name}_{aud_feat_name}"),
            }
        )

    if len(pair_records) == 0:
        raise RuntimeError(f"No usable AV pair directories found under {av_base_dir}.")

    # ====== [Part 1] compute VPA per pair and save voxel-wise npz ======
    processed_count = 0
    sum_comp = {
        split_name: {"u_v": None, "u_a": None, "shared": None}
        for split_name in split_to_file
    }
    cnt_comp = {
        split_name: {"u_v": None, "u_a": None, "shared": None}
        for split_name in split_to_file
    }

    for rec in tqdm(pair_records, desc="VPA processing pairs", unit="pair"):
        comp = {}
        missing = False
        for split_name, ev_file in split_to_file.items():
            av_file = os.path.join(rec["av_dir"], ev_file)
            vis_file = os.path.join(rec["vis_dir"], ev_file)
            aud_file = os.path.join(rec["aud_dir"], ev_file)
            if not (os.path.exists(av_file) and os.path.exists(vis_file) and os.path.exists(aud_file)):
                print(f"[skip] missing EV file: {rec['pair_name']} ({split_name})")
                missing = True
                break

            r2_va = np.load(av_file)
            r2_v = np.load(vis_file)
            r2_a = np.load(aud_file)
            if not (r2_va.shape == r2_v.shape == r2_a.shape):
                print(f"[skip] shape mismatch: {rec['pair_name']} ({split_name})")
                missing = True
                break

            u_v = r2_va - r2_a
            u_a = r2_va - r2_v
            shared = r2_v + r2_a - r2_va
            comp[split_name] = {
                "u_v": u_v,
                "u_a": u_a,
                "shared": shared,
            }

        if missing:
            continue

        pair_dir = rec["av_dir"]
        pair_name = rec["pair_name"]
        # pair_npz_path = os.path.join(pair_dir, "vpa_components_voxelwise.npz")
        pair_npz_path = os.path.join(pair_dir, "vpa_components_voxelwise_val.npz")
        np.savez(
            pair_npz_path,
            val_visual_unique=comp["val"]["u_v"],
            val_audio_unique=comp["val"]["u_a"],
            val_shared=comp["val"]["shared"],
            # test_visual_unique=comp["test"]["u_v"],
            # test_audio_unique=comp["test"]["u_a"],
            # test_shared=comp["test"]["shared"],
            # test_unseen_visual_unique=comp["test_unseen"]["u_v"],
            # test_unseen_audio_unique=comp["test_unseen"]["u_a"],
            # test_unseen_shared=comp["test_unseen"]["shared"],
        )

        # pair_roi_stats = {}
        # for split_name in ['test', 'test_unseen']:
        #     u_v = comp[split_name]["u_v"]
        #     u_a = comp[split_name]["u_a"]
        #     shared = comp[split_name]["shared"]
        #     joint_ev = u_v + u_a + shared
        #     valid_mask = np.isfinite(joint_ev) & (joint_ev > JOINT_EV_MIN_THRESHOLD)
        #     valid_num = int(valid_mask.sum())
        #     total_num = int(valid_mask.shape[0])
        #     print(f"[info] {pair_name} {split_name} valid voxels: {valid_num}/{total_num}")
        #
        #     u_v_valid = np.where(valid_mask, u_v, np.nan)
        #     u_a_valid = np.where(valid_mask, u_a, np.nan)
        #     shared_valid = np.where(valid_mask, shared, np.nan)
        #     pair_roi_stats[split_name] = {
        #         "visual_unique_roi_mean": get_roi_weight_mean_dict(
        #             u_v_valid, roi_list, subject_name=subject_name, split_lr=True
        #         ),
        #         "audio_unique_roi_mean": get_roi_weight_mean_dict(
        #             u_a_valid, roi_list, subject_name=subject_name, split_lr=True
        #         ),
        #         "shared_roi_mean": get_roi_weight_mean_dict(
        #             shared_valid, roi_list, subject_name=subject_name, split_lr=True
        #         ),
            #
        # pair_roi_json_path = os.path.join(pair_dir, "vpa_roi_mean.json")
        # with open(pair_roi_json_path, 'w', encoding='utf-8') as f:
        #     json.dump(pair_roi_stats, f, indent=2, ensure_ascii=False)

        # ====== [Part 3] accumulate per-pair VPA for final averaged npz ======
        for split_name in split_to_file:
            for key in ['u_v', 'u_a', 'shared']:
                arr = np.asarray(comp[split_name][key], dtype=float)
                if sum_comp[split_name][key] is None:
                    sum_comp[split_name][key] = np.zeros_like(arr, dtype=float)
                    cnt_comp[split_name][key] = np.zeros_like(arr, dtype=np.int64)
                valid = np.isfinite(arr)
                sum_comp[split_name][key][valid] += arr[valid]
                cnt_comp[split_name][key][valid] += 1

        processed_count += 1
        print(f"[save] VPA pair stats finished: {pair_name}")
        print(f"  - voxelwise npz: {pair_npz_path}")
        # print(f"  - roi mean json: {pair_roi_json_path}")

    if processed_count == 0:
        raise RuntimeError("No AV pairs available for VPA; check that single-modal and joint EV files exist.")

    # ====== [Part 4] aggregate averaged vpa_components_voxelwise.npz across pairs ======
    mean_comp = {split_name: {} for split_name in split_to_file}
    for split_name in split_to_file:
        for key in ['u_v', 'u_a', 'shared']:
            sums = sum_comp[split_name][key]
            counts = cnt_comp[split_name][key]
            out = np.full(sums.shape, np.nan, dtype=float)
            np.divide(sums, counts, out=out, where=counts > 0)
            mean_comp[split_name][key] = out

    avg_dir = os.path.join(av_base_dir, f"{subject_name}_vpa_mean_across_pairs")
    os.makedirs(avg_dir, exist_ok=True)

    # save averaged voxel-wise decomposition
    # save_npz = os.path.join(avg_dir, "vpa_components_voxelwise.npz")
    save_npz = os.path.join(avg_dir, "vpa_components_voxelwise_val.npz")
    np.savez(
        save_npz,
        val_visual_unique=mean_comp["val"]["u_v"],
        val_audio_unique=mean_comp["val"]["u_a"],
        val_shared=mean_comp["val"]["shared"],
        # test_visual_unique=mean_comp["test"]["u_v"],
        # test_audio_unique=mean_comp["test"]["u_a"],
        # test_shared=mean_comp["test"]["shared"],
        # test_unseen_visual_unique=mean_comp["test_unseen"]["u_v"],
        # test_unseen_audio_unique=mean_comp["test_unseen"]["u_a"],
        # test_unseen_shared=mean_comp["test_unseen"]["shared"],
    )

    print("[save] VPA mean across pairs finished.")
    print(f"  - n_pairs: {processed_count}")
    print(f"  - output dir: {avg_dir}")
    print(f"  - voxelwise npz: {save_npz}")


def export_vpa_roi_mean_from_voxelwise_npz(
    vpa_npz_path,
    subject_name='S02',
    roi_list=None,
    output_dir=None,
    vpa_shared_ev_min_threshold=0.00,
    bivariate_joint_ev_min_threshold=0.01,
    bivariate_vmax_joint=0.15,
    top_k_voxels=1000,
    top_p=0.2,
):
    """
    Export directly from vpa_components_voxelwise*.npz (with val_* keys):
    1) vpa_roi_mean.json (when Part 4 is enabled)
    2) vpa_shared_val.png and related maps (val split)

    By default writes to the npz parent directory (output_dir=None).
    Bivariate maps leave voxels with joint EV < bivariate_joint_ev_min_threshold uncolored.
    bivariate_vmax_joint caps joint EV in the 2D colormap (default 0.15).
    Top masks: if top_p > 0 (default 0.2), take the top ceil(n_valid * top_p) voxels among
    finite values intersected with valid; otherwise use top_k_voxels. Mask values are 1/0.
    """
    if roi_list is None:
        roi_list = globals().get('roi_names', ['V1', 'V2', 'V3', 'FFA', 'OFA', 'PPA', 'MT+', 'AC_L', 'AC_R'])

    if not os.path.exists(vpa_npz_path):
        raise FileNotFoundError(f"VPA npz file not found: {vpa_npz_path}")
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(vpa_npz_path))
    os.makedirs(output_dir, exist_ok=True)
    try:
        top_k_voxels = int(top_k_voxels)
    except (TypeError, ValueError):
        top_k_voxels = 3000
    if top_k_voxels <= 0:
        top_k_voxels = 3000

    try:
        top_p_f = float(top_p)
    except (TypeError, ValueError):
        top_p_f = 0.0
    use_top_p = top_p_f > 0.0

    key_map = {
        'val': {
            'u_v': 'val_visual_unique',
            'u_a': 'val_audio_unique',
            'shared': 'val_shared',
        },
        # 'test': {
        #     'u_v': 'test_visual_unique',
        #     'u_a': 'test_audio_unique',
        #     'shared': 'test_shared',
        # },
        # 'test_unseen': {
        #     'u_v': 'test_unseen_visual_unique',
        #     'u_a': 'test_unseen_audio_unique',
        #     'shared': 'test_unseen_shared',
        # },
    }

    with np.load(vpa_npz_path) as npz_data:
        missing_keys = [
            key
            for split_name in key_map
            for key in key_map[split_name].values()
            if key not in npz_data
        ]
        if len(missing_keys) > 0:
            raise KeyError(f"npz missing required keys: {missing_keys}")

        comp = {split_name: {} for split_name in key_map}
        for split_name, split_keys in key_map.items():
            u_v = np.asarray(npz_data[split_keys['u_v']], dtype=float)
            u_a = np.asarray(npz_data[split_keys['u_a']], dtype=float)
            shared = np.asarray(npz_data[split_keys['shared']], dtype=float)
            if not (u_v.shape == u_a.shape == shared.shape):
                raise ValueError(
                    f"{split_name} component shapes mismatch: "
                    f"u_v={u_v.shape}, u_a={u_a.shape}, shared={shared.shape}"
                )
            comp[split_name]["u_v"] = u_v
            comp[split_name]["u_a"] = u_a
            comp[split_name]["shared"] = shared

    # ====== [Part 1] plot bivariate brain maps; save bivariate legend ======
    try:
        bivariate_vmax_joint = float(bivariate_vmax_joint)
    except (TypeError, ValueError):
        bivariate_vmax_joint = 0.15
    if not np.isfinite(bivariate_vmax_joint) or bivariate_vmax_joint <= 0:
        bivariate_vmax_joint = 0.15

    # bivariate_map_paths = {}
    # for split_name in tqdm(['val'], desc="VPA bivariate maps from npz", leave=False):
    #     vis_ev = np.maximum(comp[split_name]["u_v"], 0.0)
    #     aud_ev = np.maximum(comp[split_name]["u_a"], 0.0)
    #     joint_ev = vis_ev + aud_ev + comp[split_name]["shared"]
    #     out_img = os.path.join(output_dir, f"vpa_bivariate_{split_name}.png")
    #     visualize_bivariate_ev_brain(
    #         vis_ev=vis_ev,
    #         aud_ev=aud_ev,
    #         joint_ev=joint_ev,
    #         img_path=out_img,
    #         subject_name=subject_name,
    #         vmax_joint=bivariate_vmax_joint,
    #         min_joint_ev_threshold=bivariate_joint_ev_min_threshold,
    #     )
    #     bivariate_map_paths[split_name] = out_img
    # for split_name in tqdm(['test', 'test_unseen'], desc="VPA bivariate maps from npz", leave=False):
    #     vis_ev = np.maximum(comp[split_name]["u_v"], 0.0)
    #     aud_ev = np.maximum(comp[split_name]["u_a"], 0.0)
    #     joint_ev = vis_ev + aud_ev + comp[split_name]["shared"]
    #     out_img = os.path.join(output_dir, f"vpa_bivariate_{split_name}.png")
    #     visualize_bivariate_ev_brain(
    #         vis_ev=vis_ev,
    #         aud_ev=aud_ev,
    #         joint_ev=joint_ev,
    #         img_path=out_img,
    #         subject_name=subject_name,
    #         vmax_joint=bivariate_vmax_joint,
    #         min_joint_ev_threshold=bivariate_joint_ev_min_threshold,
    #     )
    #     bivariate_map_paths[split_name] = out_img

    # bivariate_legend_path = os.path.join(output_dir, "vpa_bivariate_legend.png")
    # draw_bivariate_legend(
    #     vmax_joint=bivariate_vmax_joint,
    #     save_path=bivariate_legend_path,
    #     min_joint_ev_threshold=bivariate_joint_ev_min_threshold,

    # ====== [Part 2] save valid-voxel mask (joint_ev > threshold) ======
    valid_masks = {}
    valid_counts = {}
    for split_name in ['val']:
        vis_ev = np.maximum(comp[split_name]["u_v"], 0.0)
        aud_ev = np.maximum(comp[split_name]["u_a"], 0.0)
        joint_ev = vis_ev + aud_ev + comp[split_name]["shared"]
        valid_mask = np.isfinite(joint_ev) & (joint_ev > bivariate_joint_ev_min_threshold)
        valid_masks[split_name] = valid_mask
        valid_num = int(valid_mask.sum())
        total_num = int(valid_mask.shape[0])
        valid_counts[split_name] = {
            "valid_num": valid_num,
            "total_num": total_num,
        }
        print(f"[info] {split_name} valid voxels: {valid_num}/{total_num}")
    # for split_name in ['test', 'test_unseen']:
    #     vis_ev = np.maximum(comp[split_name]["u_v"], 0.0)
    #     aud_ev = np.maximum(comp[split_name]["u_a"], 0.0)
    #     joint_ev = vis_ev + aud_ev + comp[split_name]["shared"]
    #     valid_mask = np.isfinite(joint_ev) & (joint_ev > bivariate_joint_ev_min_threshold)
    #     valid_masks[split_name] = valid_mask
    #     valid_num = int(valid_mask.sum())
    #     total_num = int(valid_mask.shape[0])
    #     valid_counts[split_name] = {
    #         "valid_num": valid_num,
    #         "total_num": total_num,
    #     print(f"[info] {split_name} valid voxels: {valid_num}/{total_num}")

    # shared_pool = []
    # for split_name in ['val']:
    #     shared = comp[split_name]["shared"]
    #     valid = valid_masks[split_name]
    #     shared_valid = shared[np.isfinite(shared) & valid]
    #     if shared_valid.size > 0:
    #         shared_pool.append(shared_valid)
    # # for split_name in ['test', 'test_unseen']:
    # #     shared = comp[split_name]["shared"]
    # #     valid = valid_masks[split_name]
    # #     shared_valid = shared[np.isfinite(shared) & valid]
    # #     if shared_valid.size > 0:
    # #         shared_pool.append(shared_valid)
    # if len(shared_pool) > 0:
    #     shared_all = np.concatenate(shared_pool) if len(shared_pool) > 1 else shared_pool[0]
    #     global_vmax = float(np.percentile(shared_all, 98))
    #     if not np.isfinite(global_vmax):
    #         global_vmax = float(np.nanmax(shared_all))
    # else:
    #     global_vmax = float(vpa_shared_ev_min_threshold + 1e-6)
    # if not np.isfinite(global_vmax):
    #     global_vmax = float(vpa_shared_ev_min_threshold + 1e-6)
    # global_vmax = max(global_vmax, float(vpa_shared_ev_min_threshold) + 1e-6)
    #
    # for split_name in tqdm(['val'], desc="VPA shared maps from npz", leave=False):
    #     shared = comp[split_name]["shared"].astype(float, copy=True)
    #     shared[~valid_masks[split_name]] = np.nan
    #     shared[shared < vpa_shared_ev_min_threshold] = np.nan
    #     out_img = os.path.join(output_dir, f"vpa_shared_{split_name}.png")
    #     visualize_masked_weight(
    #         shared,
    #         out_img,
    #         subject_name=subject_name,
    #         vmin=vpa_shared_ev_min_threshold,
    #         vmax=global_vmax,
    #         cmap='YlOrRd',
    #     )
    # # for split_name in tqdm(['test', 'test_unseen'], desc="VPA shared maps from npz", leave=False):
    # #     shared = comp[split_name]["shared"].astype(float, copy=True)
    # #     shared[~valid_masks[split_name]] = np.nan
    # #     shared[shared < vpa_shared_ev_min_threshold] = np.nan
    # #     out_img = os.path.join(output_dir, f"vpa_shared_{split_name}.png")
    # #     visualize_masked_weight(
    # #         shared,
    # #         out_img,
    # #         subject_name=subject_name,
    # #         vmin=vpa_shared_ev_min_threshold,
    # #         vmax=global_vmax,
    # #         cmap='YlOrRd',
    # #     )

    # roi_stats = {}
    # for split_name in ['val']:
    #     valid = valid_masks[split_name]
    #     u_v_valid = np.where(valid, comp[split_name]["u_v"], np.nan)
    #     u_a_valid = np.where(valid, comp[split_name]["u_a"], np.nan)
    #     shared_valid = np.where(valid, comp[split_name]["shared"], np.nan)
    #     roi_stats[split_name] = {
    #         "visual_unique_roi_mean": get_roi_weight_mean_dict(
    #             u_v_valid, roi_list, subject_name=subject_name, split_lr=True
    #         ),
    #         "audio_unique_roi_mean": get_roi_weight_mean_dict(
    #             u_a_valid, roi_list, subject_name=subject_name, split_lr=True
    #         ),
    #         "shared_roi_mean": get_roi_weight_mean_dict(
    #             shared_valid, roi_list, subject_name=subject_name, split_lr=True
    #         ),
    # # for split_name in ['test', 'test_unseen']:
    # #     valid = valid_masks[split_name]
    # #     u_v_valid = np.where(valid, comp[split_name]["u_v"], np.nan)
    # #     u_a_valid = np.where(valid, comp[split_name]["u_a"], np.nan)
    # #     shared_valid = np.where(valid, comp[split_name]["shared"], np.nan)
    # #     roi_stats[split_name] = {
    # #         "visual_unique_roi_mean": get_roi_weight_mean_dict(
    # #             u_v_valid, roi_list, subject_name=subject_name, split_lr=True
    # #         ),
    # #         "audio_unique_roi_mean": get_roi_weight_mean_dict(
    # #             u_a_valid, roi_list, subject_name=subject_name, split_lr=True
    # #         ),
    # #         "shared_roi_mean": get_roi_weight_mean_dict(
    # #             shared_valid, roi_list, subject_name=subject_name, split_lr=True
    # #         ),
    # #     }
    # roi_json_path = os.path.join(output_dir, "vpa_roi_mean.json")
    # with open(roi_json_path, 'w', encoding='utf-8') as f:
    #     json.dump(roi_stats, f, indent=2, ensure_ascii=False)

    # ====== [Part 5] save top masks and discrete RGB brain maps on valid voxels ======
    def _top_k_binary_mask(values, base_valid_mask, top_k=3000):
        values = np.asarray(values, dtype=float).reshape(-1)
        base_valid_mask = np.asarray(base_valid_mask, dtype=bool).reshape(-1)
        if values.shape != base_valid_mask.shape:
            raise ValueError(f"values/base_valid_mask shape mismatch: {values.shape}, {base_valid_mask.shape}")

        out = np.zeros(values.shape[0], dtype=np.uint8)
        valid = np.isfinite(values) & base_valid_mask
        n_valid = int(valid.sum())
        if n_valid <= 0:
            return out

        if use_top_p:
            k_req = max(1, min(n_valid, int(np.ceil(n_valid * top_p_f))))
        else:
            try:
                k_req = int(top_k)
            except (TypeError, ValueError):
                k_req = 3000
            if k_req <= 0:
                return out

        k = min(k_req, n_valid)
        valid_vals = values[valid]
        start = n_valid - k
        top_local_inds = np.argpartition(valid_vals, start)[start:]
        valid_global_inds = np.where(valid)[0]
        out[valid_global_inds[top_local_inds]] = 1
        return out

    top_masks = {}
    top_rgb_map_paths = {}
    for split_name in ['val']:
        valid = valid_masks[split_name]
        shared_top_mask = _top_k_binary_mask(comp[split_name]["shared"], valid, top_k=top_k_voxels)
        visual_top_mask = _top_k_binary_mask(comp[split_name]["u_v"], valid, top_k=top_k_voxels)
        audio_top_mask = _top_k_binary_mask(comp[split_name]["u_a"], valid, top_k=top_k_voxels)
        top_masks[split_name] = {
            "valid_mask": np.asarray(valid, dtype=np.uint8),
            "shared_top_mask": shared_top_mask,
            "visual_top_mask": visual_top_mask,
            "audio_top_mask": audio_top_mask,
        }

        # rgb_img_path = os.path.join(output_dir, f"vpa_topk_mask_rgb_{split_name}.png")
        # visualize_discrete_modality_masks_rgb(
        #     shared_mask=shared_top_mask,
        #     vis_mask=visual_top_mask,
        #     aud_mask=audio_top_mask,
        #     img_path=rgb_img_path,
        #     subject_name=subject_name,
            # top_rgb_map_paths[split_name] = rgb_img_path
    # # for split_name in ['test', 'test_unseen']:
    # #     valid = valid_masks[split_name]
    # #     shared_top_mask = _top_k_binary_mask(comp[split_name]["shared"], valid, top_k=top_k_voxels)
    # #     visual_top_mask = _top_k_binary_mask(comp[split_name]["u_v"], valid, top_k=top_k_voxels)
    # #     audio_top_mask = _top_k_binary_mask(comp[split_name]["u_a"], valid, top_k=top_k_voxels)
    # #     top_masks[split_name] = {
    # #         "valid_mask": np.asarray(valid, dtype=np.uint8),
    # #         "shared_top_mask": shared_top_mask,
    # #         "visual_top_mask": visual_top_mask,
    # #         "audio_top_mask": audio_top_mask,
    # #     }
    # #
    # #     rgb_img_path = os.path.join(output_dir, f"vpa_topk_mask_rgb_{split_name}.png")
    # #     visualize_discrete_modality_masks_rgb(
    # #         shared_mask=shared_top_mask,
    # #         vis_mask=visual_top_mask,
    # #         aud_mask=audio_top_mask,
    # #         img_path=rgb_img_path,
    # #         subject_name=subject_name,
    # #     )
    # #     top_rgb_map_paths[split_name] = rgb_img_path

    top_masks_npz_path = os.path.join(output_dir, f"vpa_topk_masks_top{top_p_f}.npz")
    np.savez(
        top_masks_npz_path,
        val_valid_mask=top_masks["val"]["valid_mask"],
        val_shared_top_mask=top_masks["val"]["shared_top_mask"],
        val_visual_top_mask=top_masks["val"]["visual_top_mask"],
        val_audio_top_mask=top_masks["val"]["audio_top_mask"],
        # test_shared_top_mask=top_masks["test"]["shared_top_mask"],
        # test_visual_top_mask=top_masks["test"]["visual_top_mask"],
        # test_audio_top_mask=top_masks["test"]["audio_top_mask"],
        # test_unseen_shared_top_mask=top_masks["test_unseen"]["shared_top_mask"],
        # test_unseen_visual_top_mask=top_masks["test_unseen"]["visual_top_mask"],
        # test_unseen_audio_top_mask=top_masks["test_unseen"]["audio_top_mask"],
    )
    # top_rgb_colorbar_path = os.path.join(output_dir, "vpa_topk_mask_rgb_colorbar.png")
    # draw_tricolor_circle_mixing_colorbar(top_rgb_colorbar_path)

    # print("[save] VPA outputs exported from npz.")
    # print(f"  - input npz: {vpa_npz_path}")
    # print(f"  - output dir: {output_dir}")
    # # print(f"  - roi mean json: {roi_json_path}")
    # print("  - shared maps: vpa_shared_val.png")
    # # print("  - shared maps: vpa_shared_test.png, vpa_shared_test_unseen.png")
    # print("  - bivariate maps: vpa_bivariate_val.png")
    # # print("  - bivariate maps: vpa_bivariate_test.png, vpa_bivariate_test_unseen.png")
    # print("  - bivariate legend: vpa_bivariate_legend.png")
    # print(f"  - bivariate joint EV threshold: {bivariate_joint_ev_min_threshold:.6f}")
    # print(
    #     "  - valid voxels: "
    #     f"val={valid_counts['val']['valid_num']}/{valid_counts['val']['total_num']}"
    # # print(
    # #     "  - valid voxels: "
    # #     f"test={valid_counts['test']['valid_num']}/{valid_counts['test']['total_num']}, "
    # #     f"test_unseen={valid_counts['test_unseen']['valid_num']}/{valid_counts['test_unseen']['total_num']}"
    # # )
    # if use_top_p:
    #     print(f"  - top p for masks (fraction of per-component valid voxels): {top_p_f}")
    # else:
    #     print(f"  - top k voxels for masks: {top_k_voxels}")
    # # print(f"  - top masks npz: {top_masks_npz_path}")
    # print("  - top-mask rgb maps: vpa_topk_mask_rgb_val.png")
    # # print("  - top-mask rgb maps: vpa_topk_mask_rgb_test.png, vpa_topk_mask_rgb_test_unseen.png")
    # print("  - top-mask rgb colorbar: vpa_topk_mask_rgb_colorbar.png")
    # return {
    #     "input_npz": vpa_npz_path,
    #     "output_dir": output_dir,
    #     "roi_json_path": roi_json_path,
    #     "shared_map_paths": {
    #         "test": os.path.join(output_dir, "vpa_shared_test.png"),
    #         "test_unseen": os.path.join(output_dir, "vpa_shared_test_unseen.png"),
    #     },
    #     "bivariate_vmax_joint": bivariate_vmax_joint,
    #     "bivariate_joint_ev_min_threshold": bivariate_joint_ev_min_threshold,
    #     "bivariate_map_paths": bivariate_map_paths,
    #     "bivariate_legend_path": bivariate_legend_path,
    #     "valid_voxel_counts": valid_counts,
    #     "top_ratio": top_ratio,
    #     "top_masks_npz_path": top_masks_npz_path,
    #     "top_rgb_map_paths": top_rgb_map_paths,
    # }


if __name__ == '__main__':
    roi_names = ['V1', 'V2', 'V3', 'FFA', 'OFA', 'PPA', 'MT+', 'AC_L', 'AC_R']
    vis_feat_names = [
        'clip_base_img',
        'clip_large_img',
        'siglip_base_img',
        'siglip_large_img',
        'siglip2_base_img',
        'siglip2_large_img',
        'siglip_so400m_img',
        'blip_caption_base_img',
        'blip_caption_large_img',
        'blip2_opt_2.7b_img',
        'dinov2_base',
        'dinov2_large',
        'dinov3_vitl16',
        'imagebind_img',
        'Qwen3-VL-Embedding-8B',
        'resnet50',
        'convnext_base',
    ]
    aud_feat_names = [
        'clap_audio',
        'whisper',
        'imagebind_aud',
        'wav2vec2',
        'ast',
        'panns',
        'gemini_embedding2_audio'
    ]
    

    for subj in ['S01', 'S02', 'S03', 'S04', 'S05', 'S06']:
        encoding(vis_feat_names, feat_type='Visual_Model', subject_name=subj, base_proj_dir="./encoding/encoding_img")
        encoding(aud_feat_names, feat_type='Audio_Model', subject_name=subj, base_proj_dir="./encoding/encoding_aud")
        av_encoding(vis_feat_names, aud_feat_names, subject_name=subj)
        vpa_analysis_from_saved_encoding(subject_name=subj, roi_list=roi_names)

        export_vpa_roi_mean_from_voxelwise_npz(f"./encoding/encoding_av/{subj}_vpa_mean_across_pairs/vpa_components_voxelwise_val.npz", subject_name=subj, top_p=0.2)

