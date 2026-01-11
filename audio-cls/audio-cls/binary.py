import torch
import librosa
from torch import nn, Tensor
from torchaudio import transforms
from torchvision import utils

class BinaryFeatures(nn.Module):
    def __init__(self, thresholds: list, winlen: int, hoplen: int, nfft: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.thresholds = thresholds
        self.winlen = winlen
        self.hoplen = hoplen
        self.nfft = nfft
        self.epsilon = epsilon
        self.spec_trans = transforms.Spectrogram(
            n_fft=nfft,
            win_length=winlen,
            hop_length=hoplen,
            normalized=True,
        )
        self.amp2db = transforms.AmplitudeToDB()
    
    def _compute_binary_spec(self, x: Tensor, threshold) -> Tensor:
        x_bin = torch.threshold(x, threshold, 0.0)
        x_bin = -torch.threshold(-x_bin, -threshold, -1.0)
        return x_bin
    
    def forward(self, x: Tensor) -> Tensor:
        # compute spectrogram that value in [0, 1]
        B: int = x.shape[0]
        spec: Tensor = self.amp2db(self.spec_trans(x))
        spec = spec - spec.reshape([B, -1]).min(dim=-1)[0].reshape([B, 1, 1, 1])
        spec = spec / (spec.reshape([B, -1]).max(dim=-1)[0].reshape([B, 1, 1, 1]) + self.epsilon)
        # compute binary spectrograms
        spec_bins = torch.cat(
            [self._compute_binary_spec(spec, th) \
                for th in self.thresholds], dim=1)
        return spec_bins

if __name__ == '__main__':
    import os
    print('Running test code...')
    wav_path1 = r'20220908\C01_003_2022729142948_dakaiQQ_adv_sample_118_5d10.wav'
    wav_path2 = r'20220908\C09_0710002_035_Ch_bofangyinyue_adv_sample_390_7d10.wav'
    wave_np1, _ = librosa.load(wav_path1, sr=16_000, mono=True, offset=0.1, duration=1)
    wave1 = torch.from_numpy(wave_np1)[None,None,:]
    wave_np2, _ = librosa.load(wav_path2, sr=16_000, mono=True, offset=0.1, duration=1)
    wave2 = torch.from_numpy(wave_np2)[None,None,:]
    waves = torch.cat([wave1, wave2], dim=0)

    thresholds = [0.6, 0.65, 0.7, 0.75]
    winlen = 640
    hoplen = 320
    nfft = winlen
    binfeats = BinaryFeatures(thresholds, winlen, hoplen, nfft)
    spec_bins = binfeats(waves)
    print(waves.shape)
    print(spec_bins.shape)

    save_dir = 'specs'
    print(f'Test spectrograms saved to {save_dir}')
    os.makedirs(save_dir, exist_ok=True)
    for i in range(waves.shape[0]):
        for j, v in enumerate(thresholds):
            utils.save_image(spec_bins[i][j], 
            os.path.join(
                save_dir, 
                f'{i}_th={v}_winl={winlen}_hopl={hoplen}_nfft={nfft}.png'))
