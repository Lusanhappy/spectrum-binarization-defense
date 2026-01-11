from typing import Dict, List
import torch, librosa
import numpy as np
from default_config import model_config as mcfg
from features import get_formants, get_pitch
from tqdm import tqdm


def to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595 * np.log10(1 + hz / 700)


def extract_formants(wave: np.ndarray) -> Dict[str, torch.Tensor]:
    frame_time = mcfg.winlen / mcfg.sr
    step_time = mcfg.hoplen / mcfg.sr
    feats_len = mcfg.nmels if mcfg.use_melspec else (mcfg.winlen // 2 + 1)
    # Extract feature
    formants: np.ndarray = get_formants(
        audio_signal=wave, sample_rate=mcfg.sr, 
        frame_time=frame_time, step_time=step_time, 
        formant_num=mcfg.formants_num)                        # [frame_num, formants_num]
    # To mel
    if mcfg.use_melspec:
        formants = to_mel(formants)
    # Extract formants and pitch position
    formants_nan_pos = np.isnan(formants)                     # [frame_num, formants_num]
    formants_q = formants / (mcfg.sr / 2) * feats_len
    # Position to map
    formants_not_nan = np.where(formants_nan_pos == False)
    formants_i = formants_q[formants_not_nan].astype(np.int)  # [elements_num,]
    formants_j = formants_not_nan[0]                          # [elements_num,]
    formants_map = np.zeros(shape=[
        feats_len, int(wave.shape[0] / mcfg.hoplen) + 1])
    formants_map[(formants_i, formants_j)] = 1.0
    
    feats = \
        torch.tensor(formants_map).unsqueeze(0).float() # [1, frame_num, formants_num]
    if mcfg.formants_diff:
        # Calculate diff
        d_formants_map = formants_map[:, 1:] - formants_map[:, :-1]
        formants_feats_d = np.zeros(shape=[
            feats_len, int(wave.shape[0] / mcfg.hoplen) + 1])
        formants_feats_d[:, 1:] = \
            d_formants_map
        feats_d = \
            torch.tensor(formants_feats_d
            ).unsqueeze(0).float() # [1, frame_num, formants_num]
        return dict(formants=feats, formants_d=feats_d) \
            if mcfg.dual_feats else dict(formants_d=feats_d)
    else:
        return dict(formants=feats)


def extract_pitch(wave: np.ndarray) -> Dict[str, torch.Tensor]:
    frame_time = mcfg.winlen / mcfg.sr
    step_time = mcfg.hoplen / mcfg.sr
    feats_len = mcfg.nmels if mcfg.use_melspec else (mcfg.winlen // 2 + 1)
    # Extract feature
    pitch: np.ndarray = np.expand_dims(get_pitch(
        audio_signal=wave, sample_rate=mcfg.sr, 
        frame_time=frame_time, step_time=step_time), axis=-1)   # [frame_num, 1]
    # To mel
    if mcfg.use_melspec:
        pitch = to_mel(pitch)
    # Extract formants and pitch position
    pitch_nan_pos = np.isnan(pitch)
    pitch_q = pitch / (mcfg.sr / 2) * feats_len
    # Position to map
    pitch_not_nan = np.where(pitch_nan_pos == False)
    pitch_i = pitch_q[pitch_not_nan].astype(np.int)
    pitch_j = pitch_not_nan[0]
    pitch_map = np.zeros(shape=[
        feats_len, int(wave.shape[0] / mcfg.hoplen) + 1])
    pitch_map[(pitch_i, pitch_j)] = 1.0
    
    feats = torch.tensor(pitch_map).unsqueeze(0).float()
    if mcfg.pitch_diff:
        # Calculate diff
        d_pitch_map = pitch_map[:, 1:] - pitch_map[:, :-1]
        pitch_feats_d = np.zeros(shape=[
            feats_len, int(wave.shape[0] / mcfg.hoplen) + 1])
        pitch_feats_d[:, 1:] = d_pitch_map
        feats_d = torch.tensor(pitch_feats_d).unsqueeze(0).float()
        return dict(pitch=feats, pitch_d=feats_d) \
            if mcfg.dual_feats else dict(pitch_d=feats_d)
    else:
        return dict(pitch=feats)


def extract_features(wave: np.ndarray) -> Dict[str, torch.Tensor]:
    feats = dict()
    if mcfg.use_formants:
        feats.update(extract_formants(wave))
    if mcfg.use_pitch:
        feats.update(extract_pitch(wave))
    return feats


def subsample_feats(feats: Dict[str, torch.Tensor], start: float
) -> Dict[str, torch.Tensor]:
    step_time = mcfg.hoplen / mcfg.sr
    frame_num = int(mcfg.duration / step_time) + 1
    start_frame_i = int(start * mcfg.sr / mcfg.hoplen)
    waves_feats = dict()
    for name, feat in feats.items():
        feat_sampled = feat[:, :, start_frame_i: start_frame_i + frame_num].clone()
        if name == 'formants_d':
            feat_sampled[:, :, 0] = 0 
        if name == 'pitch_d':
            feat_sampled[:, :, 0] = 0
        waves_feats[name] = feat_sampled
    return waves_feats


def extract_features_list(wav_paths: List[str]) -> List[Dict[str, torch.Tensor]]:
    feats_list = list()
    print('Feature extraction...')
    for path in tqdm(wav_paths):
        wave, _ = librosa.load(path, sr=mcfg.sr, mono=True)
        wave = np.pad(
            wave,
            (0, int(mcfg.duration * mcfg.sr)),
            mode='constant',
            constant_values=0)
        feats_list.append(extract_features(wave))
    return feats_list

