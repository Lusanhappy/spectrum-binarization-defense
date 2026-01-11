from torchaudio import transforms
from default_config import model_config as mcfg
from feature_extract import extract_features
import librosa, torch
import numpy as np
import matplotlib.pyplot as plt

def get_spec(path):
    wave, _ = librosa.load(path, mono=True, sr=16000)
    wave_tensor = torch.tensor(wave).unsqueeze(0).unsqueeze(0)
    melspec = transforms.MelSpectrogram(
                sample_rate=mcfg.sr,
                n_fft=mcfg.nfft,
                win_length=mcfg.winlen,
                hop_length=mcfg.hoplen,
                n_mels=mcfg.nmels,
                mel_scale='slaney',
                normalized=False
            )
    return 10 * np.log(melspec(wave_tensor)[0][0].numpy())

def get_formants_pitch(path):
    wave, _ = librosa.load(path, mono=True, sr=16000)
    feats = extract_features(wave)
    return feats['formants'][0].numpy(), feats['pitch'][0].numpy()

def plot(
    spec: np.ndarray, formants: np.ndarray, pitch: np.ndarray, 
    max_frame_num: int = 50):
    map = pitch
    # map = spec \
    #     + formants * 200 \
    #         + pitch * 200
    plt.imshow(map[:, :max_frame_num])
    plt.show()

if __name__ == '__main__':
    # path = r'f:\experiment\911\911.wav'
    path = r'E:\Desktop\AliasingAttack\samples\1020music16K\010.wav'
    # path = r'E:\Desktop\420.wav'
    mcfg.use_first_diff = True
    spec = get_spec(path)
    formants, pitch = get_formants_pitch(path)
    plot(spec, formants, pitch)

