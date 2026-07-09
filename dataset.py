import os

import yaml
from torch.utils.data import Dataset
import numpy as np

from fMRI_Narrative_movie.util import util_dataload as udl
from fMRI_Narrative_movie.util import util_ridge as uridge


def center_stim_by_train_mean(Stim_train, Stim_val, Stim_test, Stim_test_unseen=None):
    """
    Center each split's stimulus features using the batch mean of Stim_train.

    Returns:
        (Stim_train_centered, Stim_val_centered, Stim_test_centered, Stim_test_unseen_centered)
    """
    Stim_train = np.asarray(Stim_train)
    Stim_val = np.asarray(Stim_val)
    Stim_test = np.asarray(Stim_test)
    Stim_test_unseen = None if Stim_test_unseen is None else np.asarray(Stim_test_unseen)

    if Stim_train.shape[0] == 0:
        raise ValueError("Stim_train must not be empty; cannot compute batch mean.")

    stim_train_mean = Stim_train.mean(axis=0, keepdims=True)

    Stim_train_centered = Stim_train - stim_train_mean
    Stim_val_centered = Stim_val - stim_train_mean
    Stim_test_centered = Stim_test - stim_train_mean
    Stim_test_unseen_centered = None
    if Stim_test_unseen is not None:
        Stim_test_unseen_centered = Stim_test_unseen - stim_train_mean

    return Stim_train_centered, Stim_val_centered, Stim_test_centered, Stim_test_unseen_centered


def get_fmri_single_movie(subject_name='S02',):
    with open('fMRI_Narrative_movie/util/config__drama_data.yaml', 'r') as f_yml:
        config = yaml.safe_load(f_yml)

    movIDs_train = [0]  # breaking bad

    # read run
    subjectID = udl.get_subjectID_from_subjectName(config, subject_name)
    load_items = udl.set_load_items(config, subjectID, movIDs_train[0])
    n_test = load_items['list_duration'][-1] - 20

    # read mask
    t_path_mask = load_items['mask_path']
    mask = udl.load_mask_data(t_path_mask)

    Resp, _ = uridge.resp_stim_loader(config, subjectID, movIDs_train, loadStim=False)
    Resp_train = Resp[:-n_test, :]
    Resp_test = Resp[-n_test:, :]


    m_Resp_train, s_Resp_train, Resp_train = uridge.my_zscore(Resp_train)
    _, _, Resp_test = uridge.my_zscore(Resp_test, mX=m_Resp_train, sX=s_Resp_train)

    Resp_train_volume = np.zeros((Resp_train.shape[0], 663552), dtype=np.float32)
    Resp_train_volume[:, mask] = Resp_train
    Resp_train_volume = Resp_train_volume.reshape(-1, 1, 72, 96, 96, 1)

    Resp_test_volume = np.zeros((Resp_test.shape[0], 663552), dtype=np.float32)
    Resp_test_volume[:, mask] = Resp_test
    Resp_test_volume = Resp_test_volume.reshape(-1, 1, 72, 96, 96, 1)

    print('Resp train:', Resp_train_volume.shape)
    print('Resp test:', Resp_test_volume.shape)

    # Stim
    img_feat_path = "PATH/TO/Narrative_Movie_fMRI_Dataset/derivatives/feat/imagebind_img/Breaking_Bad.npz"
    Stim = np.load(img_feat_path)['imagebind_feat']
    # delete first 20s & delay
    Stim = Stim[15:-5, :]
    assert len(Stim) == len(Resp)
    Stim_train = Stim[:-n_test, :]
    Stim_test = Stim[-n_test:, :]

    print('Stim train:', Stim_train.shape)
    print('Stim test:', Stim_test.shape)

    return Resp_train_volume, Resp_test_volume, Stim_train, Stim_test


def prepare_stim_same_movie(train_movie_names=['Breaking_Bad', 'The_Big_Bang_Theory', 'The_Crown',
                                               'Heroes', 'Suits', 'Dream_Girls', 'The_Mentalist'],
                            delay=5, feat_name='imagebind_img', feat_type='Visual_feature',
                            split_ratio=(0.8, 0.1, 0.1),
                            feat_path = "PATH/TO/Narrative_Movie_fMRI_Dataset/derivatives/feat"):
    """
    Load stimulus features for the same movies and split into train/val/test by the given ratio.
    
    Args:
        split_ratio: (train_ratio, val_ratio, test_ratio), default (0.8, 0.1, 0.1)
    """
    
    Stim_train_list, Stim_val_list, Stim_test_list = [], [], []
    
    for movie_name in train_movie_names:
        stim_feat_path = "%s/%s/%s/%s.npz" % (feat_path, feat_type, feat_name, movie_name)
        data = np.load(stim_feat_path)
        if feat_name in data:
            Stim = data[feat_name].astype(np.float32)
        else:
            Stim = data['feature'].astype(np.float32)
        # Remove the first 20 seconds and then delay
        Stim = Stim[(20-delay) // 5:-delay // 5, :]
        

        movie_len = len(Stim)
        train_end = int(movie_len * split_ratio[0])
        val_end = int(movie_len * (split_ratio[0] + split_ratio[1]))
        
        Stim_train_list.append(Stim[:train_end, :])
        Stim_val_list.append(Stim[train_end:val_end, :])
        Stim_test_list.append(Stim[val_end:, :])
    
    Stim_train = np.vstack(Stim_train_list)
    Stim_val = np.vstack(Stim_val_list)
    Stim_test = np.vstack(Stim_test_list)

    print('Stim train:', Stim_train.shape, 'Stim val:', Stim_val.shape, 'Stim test:', Stim_test.shape)
    return Stim_train, Stim_val, Stim_test


def prepare_stim_test(test_movie_names=['Glee'], delay=5, feat_name='imagebind_img', feat_type='Visual_feature',
                      feat_path = "PATH/TO/Narrative_Movie_fMRI_Dataset/derivatives/feat"):

    Stim_list = []
    for movie_name in test_movie_names:
        stim_feat_path = "%s/%s/%s/%s.npz" % (feat_path, feat_type, feat_name, movie_name)
        data = np.load(stim_feat_path)
        if feat_name in data:
            Stim = data[feat_name].astype(np.float32)
        else:
            Stim = data['feature'].astype(np.float32)
        # Remove the first 20 seconds and then delay
        Stim = Stim[(20 - delay) // 5:-delay // 5, :]
        Stim_list.append(Stim)

    Stims = np.vstack(Stim_list)
    print('Stim test:', Stims.shape)
    return Stims


def prepare_fmri_and_stim_same_movie(subject_name='S02', delay=5, feat_name='imagebind_img', feat_type='Visual_feature',
                                     return_volume=True,
                                     train_movie_names=['Breaking_Bad', 'The_Big_Bang_Theory', 'The_Crown',
                                                        'Heroes', 'Suits', 'Dream_Girls', 'The_Mentalist'],
                                     feat_path="PATH/TO/Narrative_Movie_fMRI_Dataset/derivatives/feat"):
    with open('fMRI_Narrative_movie/util/config__drama_data.yaml', 'r') as f_yml:
        config = yaml.safe_load(f_yml)
    subjectID = udl.get_subjectID_from_subjectName(config, subject_name)

    # read mask
    load_items = udl.set_load_items(config, subjectID, 7)
    t_path_mask = load_items['mask_path']
    mask = udl.load_mask_data(t_path_mask)


    Resp_train_list, Resp_val_list, Resp_test_list = [], [], []
    Stim_train_list, Stim_val_list, Stim_test_list = [], [], []
    
    for movie_name in train_movie_names:
        Resp_single, Stim_single = get_fmri_and_stim(
            subject_name,
            [movie_name],
            delay,
            feat_name,
            feat_type,
            feat_path=feat_path,
        )
        


        movie_len = len(Stim_single)
        train_end = int(movie_len * 0.8)
        val_end = int(movie_len * 0.9)
        
        Resp_train_list.append(Resp_single[:train_end * 5, :])
        Resp_val_list.append(Resp_single[train_end * 5:val_end * 5, :])
        Resp_test_list.append(Resp_single[val_end * 5:, :])
        
        Stim_train_list.append(Stim_single[:train_end, :])
        Stim_val_list.append(Stim_single[train_end:val_end, :])
        Stim_test_list.append(Stim_single[val_end:, :])


    Resp_train = np.vstack(Resp_train_list)
    Resp_val = np.vstack(Resp_val_list)
    Resp_test = np.vstack(Resp_test_list)
    
    Stim_train = np.vstack(Stim_train_list)
    Stim_val = np.vstack(Stim_val_list)
    Stim_test = np.vstack(Stim_test_list)


    m_Resp_train, s_Resp_train, Resp_train = uridge.my_zscore(Resp_train)
    _, _, Resp_val = uridge.my_zscore(Resp_val, mX=m_Resp_train, sX=s_Resp_train)
    _, _, Resp_test = uridge.my_zscore(Resp_test, mX=m_Resp_train, sX=s_Resp_train)


    norm_param_path = os.path.join(os.path.dirname(t_path_mask), 'norm_param.npz')
    if not os.path.exists(norm_param_path):
        norm_param = {'m_Resp_train': m_Resp_train, 's_Resp_train': s_Resp_train}
        np.savez(norm_param_path, **norm_param)
        print('Save', norm_param_path)

    if not return_volume:
        return Resp_train, Stim_train, Resp_val, Stim_val, Resp_test, Stim_test, None


    mask_3d = np.zeros(663552, dtype=np.float32)
    mask_3d[mask] = 1.0
    mask_3d = mask_3d.reshape(72, 96, 96)


    Resp_train_volume = np.zeros((Resp_train.shape[0], 663552), dtype=np.float32)
    Resp_train_volume[:, mask] = Resp_train
    Resp_train_volume = Resp_train_volume.reshape(-1, 5, 663552)
    Resp_train_volume = Resp_train_volume.reshape(-1, 5, 72, 96, 96, 1).transpose(0, 5, 2, 3, 4, 1)

    Resp_val_volume = np.zeros((Resp_val.shape[0], 663552), dtype=np.float32)
    Resp_val_volume[:, mask] = Resp_val
    Resp_val_volume = Resp_val_volume.reshape(-1, 5, 663552)
    Resp_val_volume = Resp_val_volume.reshape(-1, 5, 72, 96, 96, 1).transpose(0, 5, 2, 3, 4, 1)

    Resp_test_volume = np.zeros((Resp_test.shape[0], 663552), dtype=np.float32)
    Resp_test_volume[:, mask] = Resp_test
    Resp_test_volume = Resp_test_volume.reshape(-1, 5, 663552)
    Resp_test_volume = Resp_test_volume.reshape(-1, 5, 72, 96, 96, 1).transpose(0, 5, 2, 3, 4, 1)

    print('Resp train:', Resp_train_volume.shape, 'Stim train:', Stim_train.shape)
    print('Resp val:', Resp_val_volume.shape, 'Stim val:', Stim_val.shape)
    print('Resp test:', Resp_test_volume.shape, 'Stim test:', Stim_test.shape)
    return Resp_train_volume, Stim_train, Resp_val_volume, Stim_val, Resp_test_volume, Stim_test, mask_3d


def prepare_fmri_and_stim_test(subject_name='S02', delay=5, feat_name='imagebind_img', feat_type='Visual_feature',
                               test_movie_names=['Glee'], return_volume=True,
                               feat_path="PATH/TO/Narrative_Movie_fMRI_Dataset/derivatives/feat"):
    with open('fMRI_Narrative_movie/util/config__drama_data.yaml', 'r') as f_yml:
        config = yaml.safe_load(f_yml)
    subjectID = udl.get_subjectID_from_subjectName(config, subject_name)

    # read mask
    load_items = udl.set_load_items(config, subjectID, 7)
    t_path_mask = load_items['mask_path']
    mask = udl.load_mask_data(t_path_mask)

    Resp_test, Stim_test = get_fmri_and_stim(
        subject_name,
        test_movie_names,
        delay,
        feat_name,
        feat_type,
        feat_path=feat_path,
    )

    norm_param_path = os.path.join(os.path.dirname(t_path_mask), 'norm_param.npz')
    norm_param = np.load(norm_param_path)
    m_Resp_train = norm_param['m_Resp_train']
    s_Resp_train = norm_param['s_Resp_train']

    _, _, Resp_test = uridge.my_zscore(Resp_test, mX=m_Resp_train, sX=s_Resp_train)

    if not return_volume:
        return Resp_test, Stim_test

    Resp_test_volume = np.zeros((Resp_test.shape[0], 663552), dtype=np.float32)
    Resp_test_volume[:, mask] = Resp_test
    Resp_test_volume = Resp_test_volume.reshape(-1, 5, 663552)
    Resp_test_volume = Resp_test_volume.reshape(-1, 5, 72, 96, 96, 1).transpose(0, 5, 2, 3, 4, 1)
    print('Resp test:', Resp_test_volume.shape, 'Stim test:', Stim_test.shape)
    return Resp_test_volume, Stim_test


def prepare_fmri_and_stim(subject_name='S02', delay=5, feat_name='imagebind_img',
                          train_movie_names=['Breaking_Bad', 'The_Big_Bang_Theory', 'The_Crown',
                                             'Heroes', 'Suits', 'Dream_Girls'],
                          val_movie_names=['Glee'], test_movie_names=['The_Mentalist']):
    with open('fMRI_Narrative_movie/util/config__drama_data.yaml', 'r') as f_yml:
        config = yaml.safe_load(f_yml)
    subjectID = udl.get_subjectID_from_subjectName(config, subject_name)

    # read mask
    load_items = udl.set_load_items(config, subjectID, 7)
    t_path_mask = load_items['mask_path']
    mask = udl.load_mask_data(t_path_mask)

    Resp_train, Stim_train = get_fmri_and_stim(subject_name, train_movie_names, delay, feat_name)
    Resp_val, Stim_val = get_fmri_and_stim(subject_name, val_movie_names, delay, feat_name)
    Resp_test, Stim_test = get_fmri_and_stim(subject_name, test_movie_names, delay, feat_name)


    m_Resp_train, s_Resp_train, Resp_train = uridge.my_zscore(Resp_train)
    _, _, Resp_val = uridge.my_zscore(Resp_val, mX=m_Resp_train, sX=s_Resp_train)
    _, _, Resp_test = uridge.my_zscore(Resp_test, mX=m_Resp_train, sX=s_Resp_train)

    norm_param_path = os.path.join(os.path.dirname(t_path_mask), 'norm_param.npz')
    if not os.path.exists(norm_param_path):
        norm_param = {'m_Resp_train': m_Resp_train, 's_Resp_train': s_Resp_train}
        np.savez(norm_param_path, **norm_param)
        print('Save', norm_param_path)

    Resp_train_volume = np.zeros((Resp_train.shape[0], 663552), dtype=np.float32)
    Resp_train_volume[:, mask] = Resp_train
    Resp_train_volume = Resp_train_volume.reshape(-1, 5, 663552)
    Resp_train_volume = Resp_train_volume.reshape(-1, 5, 72, 96, 96, 1).transpose(0, 5, 2, 3, 4, 1)

    Resp_val_volume = np.zeros((Resp_val.shape[0], 663552), dtype=np.float32)
    Resp_val_volume[:, mask] = Resp_val
    Resp_val_volume = Resp_val_volume.reshape(-1, 5, 663552)
    Resp_val_volume = Resp_val_volume.reshape(-1, 5, 72, 96, 96, 1).transpose(0, 5, 2, 3, 4, 1)

    Resp_test_volume = np.zeros((Resp_test.shape[0], 663552), dtype=np.float32)
    Resp_test_volume[:, mask] = Resp_test
    Resp_test_volume = Resp_test_volume.reshape(-1, 5, 663552)
    Resp_test_volume = Resp_test_volume.reshape(-1, 5, 72, 96, 96, 1).transpose(0, 5, 2, 3, 4, 1)

    print('Resp train:', Resp_train_volume.shape, 'Stim train:', Stim_train.shape)
    print('Resp val:', Resp_val_volume.shape, 'Stim val:', Stim_val.shape)
    print('Resp test:', Resp_test_volume.shape, 'Stim test:', Stim_test.shape)
    return Resp_train_volume, Stim_train, Resp_val_volume, Stim_val, Resp_test_volume, Stim_test


def get_fmri_and_stim(subject_name='S02', movie_names=['Breaking_Bad'], delay=5,
             feat_name='imagebind_img', feat_type='Visual_feature',
             feat_path="PATH/TO/Narrative_Movie_fMRI_Dataset/derivatives/feat"):
    movie_ids = {'Breaking_Bad': 0, 'The_Big_Bang_Theory': 1, 'The_Crown': 3, 'Heroes': 4,
                 'Suits': 5, 'Dream_Girls': 6, 'Glee': 7, 'The_Mentalist': 8}
    with open('fMRI_Narrative_movie/util/config__drama_data.yaml', 'r') as f_yml:
        config = yaml.safe_load(f_yml)
    subjectID = udl.get_subjectID_from_subjectName(config, subject_name)

    Resp_list = []
    Stim_list = []
    for movie_name in movie_names:
        movID = movie_ids[movie_name]
        # Resp
        Resp, _ = uridge.resp_stim_loader(config, subjectID, [movID], loadStim=False)
        if movie_name == 'Glee':
            Resp = Resp[:-6, :]
        # In units of 5 seconds
        del_n = len(Resp) % 5
        Resp = Resp[:-del_n, :]

        # Stim
        stim_feat_path = "%s/%s/%s/%s.npz" % (feat_path, feat_type, feat_name, movie_name)
        data = np.load(stim_feat_path)
        if feat_name in data:
            Stim = data[feat_name].astype(np.float32)
        else:
            Stim = data['feature'].astype(np.float32)
        # Remove the first 20 seconds and then delay
        Stim = Stim[(20-delay) // 5:-delay // 5, :]
        assert len(Stim) * 5 == len(Resp)

        Resp_list.append(Resp)
        Stim_list.append(Stim)

    Resps = np.vstack(Resp_list)
    Stims = np.vstack(Stim_list)
    return Resps, Stims


class BrainDataset(Dataset):
    def __init__(self, brain_data, stimuli_feature):
        self.brain_data = brain_data
        self.stimuli_feature = stimuli_feature

    def __getitem__(self, index):
        return self.brain_data[index], self.stimuli_feature[index], index

    def __len__(self):
        return len(self.brain_data)
