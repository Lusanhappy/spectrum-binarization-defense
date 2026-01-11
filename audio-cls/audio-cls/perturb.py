import torch, torchaudio, pathlib, warnings
from utils import load_wav, save_wav, find_all_ext
from argparse import ArgumentParser
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument('--wav-dir', type=str, required=True)
parser.add_argument('--type', type=str, choices=['downsample', 'noise'])
parser.add_argument('--target-sr', type=int)
parser.add_argument('--noise-level-int16', type=int)
parser.add_argument('--save-dir', type=str)
args = parser.parse_args()

def downsample(wave: torch.Tensor, sr: int, target_sr: int) -> torch.Tensor:
    wav_downsampled = torchaudio.functional.resample(wave, sr, target_sr, lowpass_filter_width=64)
    wav_restored = torchaudio.functional.resample(wav_downsampled, target_sr, sr, lowpass_filter_width=64)
    return wav_restored

def uniform_noise(wave: torch.Tensor, noise_level: float) -> torch.Tensor:
    noise = torch.empty(wave.shape).uniform_(-noise_level, noise_level)
    return (wave + noise).clamp(-1, 1)

def save_downsampled(path: str, target_sr: int, save_path: str):
    wave, sr = load_wav(path)
    if wave.numel() == 0:
        warnings.warn(f'Empty file: {path}')
        return None
    wave_downsampled = downsample(wave, sr, target_sr)
    save_wav(wave_downsampled, save_path, sr)

def save_noised(path: str, noise_level: float, save_path: str):
    wave, sr = load_wav(path)
    wave_noised = uniform_noise(wave, noise_level)
    save_wav(wave_noised, save_path, sr)


wavs = find_all_ext(args.wav_dir, 'wav')
save_dir = pathlib.Path(args.save_dir)
wavs = tqdm(wavs)
for wav in wavs:
    file_relative_path = pathlib.Path(wav).relative_to(args.wav_dir)
    save_path = save_dir.joinpath(file_relative_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if args.type == 'downsample':
        save_downsampled(wav, args.target_sr, save_path)
    elif args.type == 'noise':
        save_noised(wav, args.noise_level_int16 / 32767, save_path)
    else:
        raise ValueError('Invalid type')

