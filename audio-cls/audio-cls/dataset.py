from typing import Dict, List, Tuple, Union
import torch, os, librosa, random
import utils
import numpy as np
from torch.utils import data
from default_config import train_config as tcfg
from default_config import model_config as mcfg
from feature_extract import extract_features, extract_features_list, subsample_feats

DEBUG = tcfg.debug_info


class DataSplit(data.Dataset):
    def __init__(self, split: str, wavs: List[Tuple[str, int]]):
        super().__init__()
        self.split = split
        self.wavs = wavs
        self.skip_min = int(tcfg.skip_min * mcfg.sr)
        self.input_length = int(mcfg.sr * mcfg.duration) \
            if mcfg.model_name == 'cnn3_dnn2' else mcfg.nb_samp
        if DEBUG is True:
            print(f'{len(self.wavs)} {self.split} samples in total.')
        if mcfg.model_name == 'cnn3_dnn2' and tcfg.static_features \
            and (mcfg.use_formants or mcfg.use_pitch):
            self.feats: List[Dict[str, torch.Tensor]] = \
                extract_features_list([item[0] for item in wavs])
        else:
            self.feats = []
    
    def _sample_wave(self, wave: np.ndarray, label: int) -> Tuple[np.ndarray, float]:
        if label == 0:  # When clean wavs
            offset = random.random() * (wave.shape[0] / mcfg.sr)
        elif label == 1: # When adv wavs
            offset = random.random() * tcfg.random_skip_max
            if tcfg.sample_the_middle is True:
                offset += (wave.shape[0] / mcfg.sr / 2) - (mcfg.duration / 2)
            else:
                wave = wave[self.skip_min:]
        else:
            raise ValueError(f'Invalid label value: "label"={label}')
        start_position = int(offset * mcfg.sr)
        wave = wave[start_position: start_position + self.input_length]
        return wave, offset
    
    def _add_noise(self, wave: np.ndarray, label: int) -> np.ndarray:
        # Add random noise
        if self.split == 'val': # ! Randomness Introduction.
            random_noise = .0
        elif self.split == 'train':
            if tcfg.clean_noise_only is True and label == 1:
                random_noise = .0
            else:
                random_noise = np.random.uniform(
                    -tcfg.random_noise_amp, 
                    tcfg.random_noise_amp, 
                    wave.shape
                ).astype(np.float32)
        else:
            raise ValueError(f'Invalid split value: "split"={self.split}')
        return wave + random_noise
    
    def _pad_wave(self, wave: np.ndarray) -> np.ndarray:
        # Pad
        if mcfg.model_name == 'cnn3_dnn2':
            wave = np.pad(
                wave,
                (0, self.input_length - wave.shape[0]),
                mode='constant',
                constant_values=0)
        elif mcfg.model_name == 'cnn_attention':
            if wave.shape[0] >= self.input_length:
                wave = wave[:self.input_length]
            else:
                num_repeats = int(self.input_length / wave.shape[0]) + 1
                wave = np.tile(wave, (1, num_repeats))[:, :self.input_length][0]
        return wave
    
    def _get_feats(self, index: int, wave: np.ndarray, offset: float
    ) -> torch.Tensor:
        if len(self.feats) == 0 or len(self.feats[index]) == 0:
            return torch.tensor(0)
        else:
            # Get features
            if tcfg.static_features is False:
                wave_feats: Dict[str, torch.Tensor] = extract_features(wave)
            else:
                wave_feats: Dict[str, torch.Tensor] = \
                    subsample_feats(self.feats[index], offset)
            # Concat
            return torch.cat(list(wave_feats.values()), dim=0)
        
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        path, label = self.wavs[index]
        wave, _ = librosa.load(path, sr=mcfg.sr, mono=True)
        # Audio sample
        wave, offset = self._sample_wave(wave, label)
        # Add random noise
        wave = self._add_noise(wave, label)
        # Pad
        wave = self._pad_wave(wave)
        # cnn_attention
        if mcfg.model_name == 'cnn_attention':
            return torch.tensor(wave).clip(min=-1, max=1), \
                    torch.tensor(0),\
                    torch.tensor(label)
        # cnn3_dnn2: add formants and pitch features
        else:
            feats: torch.Tensor = self._get_feats(index, wave, offset)
        return torch.tensor(wave).unsqueeze(0).clip(min=-1, max=1), \
                feats, torch.tensor(label)

    def __len__(self) -> int:
        return len(self.wavs)

    def __str__(self) -> str:
        return f'DataSplit(split="{self.split}", wavs=[{len(self.wavs)},])'


class AdvDataset():
    def __init__(self):
        '''
        opt1:
        datapath
            |__adv
            |   |__train
            |   |__val
            |
            |__clean
                |__train
                |__val
        '''
        if DEBUG is True:
            print(f'\nAdvDataset parameters received: {str(tcfg)}\n')
            print(f'\nAdvDataset parameters received: {str(mcfg)}\n')
        # Load wavs
        wavs: Dict[str, List[str]] = self._load_wav_paths()
        # Train/val splits
        self.train_split: List[Tuple[str, int]] = [(wav, 1)
                       for wav in wavs['wavs_adv_train']] + [(wav, 0)
                                                     for wav in wavs['wavs_cln_train']]
        self.val_split: List[Tuple[str, int]] = [(wav, 1)
                     for wav in wavs['wavs_adv_val']] + [(wav, 0)
                                                 for wav in wavs['wavs_cln_val']]
        if tcfg.no_val is False:
            self.train_dataset = DataSplit('train', self.train_split)
            self.val_dataset = DataSplit('val', self.val_split)
        else:
            print('Use all samples and no validation set.')
            self.train_dataset = DataSplit('train', self.train_split + self.val_split)
            self.val_dataset = None

    def _load_wav_paths_from_folders(
        self, folders: List[Union[str, Tuple[str, int]]]) -> List[str]:
        wav_paths = list()
        for item in folders:
            print(item)
            if isinstance(item, str) is True:
                folder, sample_limit = item, -1
            elif isinstance(item, tuple) is True and len(item) == 2:
                folder, sample_limit = item
            else:
                raise RuntimeError(f'Unparsable data folder: "{str(item)}"')
            wavs = utils.find_all_ext(folder, 'wav')
            if sample_limit > 0:
                random.shuffle(wavs)
                if DEBUG:
                    print(
                        f'Sample {folder} from {len(wavs)} to'
                        f' {min(len(wavs), sample_limit)}')
                wavs = wavs[:sample_limit]
            wav_paths += wavs
        return wav_paths
    
    def _load_wav_paths(self) -> Dict[str, List[str]]:
        # load folders
        if tcfg.data_path is not None:
            train_adv_dirs: List[str] = [os.path.join(self.datapath, 'adv', 'train')]
            val_adv_dirs: List[str] = [os.path.join(self.datapath, 'adv', 'val')]
            train_cln_dirs: List[str] = [os.path.join(self.datapath, 'clean', 'train')]
            val_cln_dirs: List[str] = [os.path.join(self.datapath, 'clean', 'val')]
        else:
            train_adv_dirs: List[Tuple[str, int]] = [tcfg.train_adv_dirs,] if \
                isinstance(tcfg.train_adv_dirs[0], str) else tcfg.train_adv_dirs
            val_adv_dirs: List[Tuple[str, int]] = [tcfg.val_adv_dirs,] if \
                isinstance(tcfg.val_adv_dirs[0], str) else tcfg.val_adv_dirs
            train_cln_dirs: List[Tuple[str, int]] = [tcfg.train_cln_dirs,] if \
                isinstance(tcfg.train_cln_dirs[0], str) else tcfg.train_cln_dirs
            val_cln_dirs: List[Tuple[str, int]] = [tcfg.val_cln_dirs,] if \
                isinstance(tcfg.val_cln_dirs[0], str) else tcfg.val_cln_dirs
        # load wavs
        wavs_adv_train = self._load_wav_paths_from_folders(train_adv_dirs)
        wavs_cln_train = self._load_wav_paths_from_folders(train_cln_dirs)
        wavs_adv_val = self._load_wav_paths_from_folders(val_adv_dirs)
        wavs_cln_val = self._load_wav_paths_from_folders(val_cln_dirs)
        if DEBUG is True:
            print('Get wavs in adv/train:', len(wavs_adv_train))
            print('Get wavs in adv/val:', len(wavs_adv_val))
            print('Get wavs in clean/train:', len(wavs_cln_train))
            print('Get wavs in clean/val:', len(wavs_cln_val))
        # Oversample/downsample/sample-limit
        wavs_adv_train, wavs_cln_train = \
            self._resample_train_split(wavs_adv_train, wavs_cln_train)
        wavs_adv_val, wavs_cln_val = \
            self._balance_val_split(wavs_adv_val, wavs_cln_val)
        return dict(
            wavs_adv_train=wavs_adv_train, wavs_cln_train=wavs_cln_train,
            wavs_adv_val=wavs_adv_val, wavs_cln_val=wavs_cln_val)

    def _resample_train_split(
        self, wavs_adv_train: list, wavs_cln_train: list
        ) -> Tuple[List[str], List[str]]:
        # Oversample and Downsample
        num_diff = len(wavs_adv_train) - len(wavs_cln_train)
        if num_diff != 0:
            if tcfg.train_oversample and tcfg.train_downsample:
                raise ValueError(
                    'Traing config "train_downsample" conflicts with "train_oversample"')
            if tcfg.train_oversample:
                to_oversample = wavs_adv_train if num_diff < 0 else wavs_cln_train
                oversampled = random.choices(to_oversample, k=abs(num_diff))
                print(
                    f"{'adv' if num_diff < 0 else 'clean'} class"
                    f" oversampled in train split."
                    f"from {len(to_oversample)} to "
                    f"{len(to_oversample) + len(oversampled)}")
                to_oversample += oversampled
            elif tcfg.train_downsample:
                to_downsample = wavs_adv_train if num_diff > 0 else wavs_cln_train
                downsampled = random.sample(to_downsample, k=abs(num_diff))
                print(
                    f"{'adv' if num_diff > 0 else 'clean'} class"
                    f" downsampled in train split."
                    f"from {len(to_downsample)} to "
                    f"{len(to_downsample) - len(downsampled)}")
                for item in downsampled:
                    to_downsample.remove(item)
        return wavs_adv_train, wavs_cln_train
    
    def _balance_val_split(self, wavs_adv_val: list, wav_cln_val: list
    ) -> Tuple[List[str], List[str]]:
        if tcfg.val_downsample is True:
            # Oversample and Downsample
            num_diff = len(wavs_adv_val) - len(wav_cln_val)
            if num_diff != 0:
                to_downsample = wavs_adv_val if num_diff > 0 else wav_cln_val
                downsampled = random.sample(to_downsample, k=abs(num_diff))
                print(
                    f"{'adv' if num_diff > 0 else 'clean'} class"
                    f" downsampled in val split."
                    f"from {len(to_downsample)} "
                    f"to {len(to_downsample) - len(downsampled)}")
                for item in downsampled:
                    to_downsample.remove(item)
        return wavs_adv_val, wav_cln_val

    def get_loaders(self, batch: int, **kwargs
        ) -> Union[Tuple[data.DataLoader, data.DataLoader], data.DataLoader]:
        train_loader = data.DataLoader(self.train_dataset,
                                       batch_size=batch,
                                       shuffle=True,
                                       pin_memory=True,
                                       **kwargs)
        if tcfg.no_val is True:
            if DEBUG is True:
                print(f'{len(self.train_dataset) / batch:.2f} training batches in total.')
            return train_loader
        else:
            val_loader = data.DataLoader(self.val_dataset,
                                        batch_size=batch,
                                        shuffle=False,
                                        pin_memory=True,
                                        **kwargs)
            if DEBUG is True:
                print(f'{len(self.train_dataset) / batch:.2f} training batches in total.')
                print(f'{len(self.val_dataset) / batch:.2f} validation batches in total.')
            return train_loader, val_loader

    def __str__(self) -> str:
        return f' train_dataset={str(self.train_dataset)},'\
                + f' val_dataset={str(self.val_dataset)})'
