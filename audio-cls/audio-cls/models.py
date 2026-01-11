import warnings, torch
from typing import Union, Callable, Tuple
from torch import nn, Tensor
from torchaudio import transforms
from default_config import model_config as mcfg
from rawnet import CNN_Attention, RawNet
from binary import BinaryFeatures

warnings.filterwarnings(action='ignore', category=UserWarning)

class CNN3_DNN2(nn.Module):
    def __init__(self):
        super().__init__()
        self.melspec = transforms.MelSpectrogram(
            sample_rate=mcfg.sr,
            n_fft=mcfg.nfft,
            win_length=mcfg.winlen,
            hop_length=mcfg.hoplen,
            n_mels=mcfg.nmels,
            mel_scale='slaney',
            normalized=False
        )
        self.todb = transforms.AmplitudeToDB()
        self.binfeats_trans = BinaryFeatures(
            thresholds=mcfg.thresholds,
            winlen=mcfg.winlen,
            hoplen=mcfg.hoplen,
            nfft=mcfg.nfft
        )
        self.input_transform = self._get_input_transform()
        self.cnn1 = nn.Conv2d(in_channels=self._input_channels(),
                              out_channels=16,
                              kernel_size=(7, 7),
                              padding='same')
        self.bn1 = nn.BatchNorm2d(num_features=16, momentum=0.01)
        self.relu1 = nn.ReLU()
        self.cnn2 = nn.Conv2d(in_channels=16,
                              out_channels=16,
                              kernel_size=(7, 7),
                              padding='same')
        self.bn2 = nn.BatchNorm2d(num_features=16, momentum=0.01)
        self.relu2 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(5, 5))
        self.dropout1 = nn.Dropout(p=0.5)
        self.cnn3 = nn.Conv2d(in_channels=16,
                              out_channels=32,
                              kernel_size=(7, 7),
                              padding='same')
        self.bn3 = nn.BatchNorm2d(num_features=32, momentum=0.01)
        self.relu3 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(5, 5))
        # self.maxpool2 = nn.AdaptiveMaxPool2d(output_size=(1, 6))
        self.dropout2 = nn.Dropout(p=0.5)
        self.flatten = nn.Flatten()
        # self.fc = nn.Linear(in_features=192, out_features=100)
        # self.fc = nn.Linear(in_features=768, out_features=100)
        self.fc = nn.LazyLinear(out_features=100)
        self.relu4 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=0.5)
        self.head = nn.Linear(in_features=100, out_features=2)
    
    # def _get_input_transform(self) -> Callable[[Tuple[Tensor, Tensor]], Tensor]:
    #     if mcfg.use_melspec and mcfg.use_binary_features:
    #         raise ValueError(
    #             'Mel spectrogram and linear spectrogram'
    #             ' are not compatiable to be used together.')
    #     if mcfg.use_melspec:
    #         def input_transform(ipt, feats):
    #             x = self.todb(self.melspec(ipt)) \
    #                 if mcfg.log_mel is True else self.melspec(ipt)
    #             if mcfg.melspec_diff:
    #                 melspec_diff = torch.zeros_like(x, device=x.device)
    #                 melspec_diff[:,:,:,1:] = x[:,:,:,1:] - x[:,:,:,:-1]
    #                 x = torch.cat([x, melspec_diff], dim=1) \
    #                     if mcfg.dual_feats else melspec_diff
    #             if feats is not None and feats.dim() == 4:
    #                 x = torch.cat([x, feats], dim=1)
    #             return x
    #         return input_transform
    #     elif mcfg.use_binary_features:
    #         def input_transform(ipt, feats):
    #             return self.binfeats_trans(ipt)
    #         return input_transform
    #     else:
    #         def input_transform(ipt, feats):
    #             return feats
    #         return input_transform
    def _get_input_transform(self) -> Callable[[Tuple[Tensor, Tensor]], Tensor]:
        if mcfg.use_melspec and mcfg.use_binary_features:
            raise ValueError(
                'Mel spectrogram and linear spectrogram'
                ' are not compatiable to be used together.')
        if mcfg.use_melspec:
            def input_transform(ipt, feats):
                x = self.todb(self.melspec(ipt)) \
                    if mcfg.log_mel is True else self.melspec(ipt)
                if mcfg.melspec_diff:
                    melspec_diff = torch.zeros_like(x, device=x.device)
                    melspec_diff[:,:,:,1:] = x[:,:,:,1:] - x[:,:,:,:-1]
                    x = torch.cat([x, melspec_diff], dim=1) \
                        if mcfg.dual_feats else melspec_diff
                if feats is not None and feats.dim() == 4:
                    x = torch.cat([x, feats], dim=1)
                return x
            return input_transform
        elif mcfg.use_binary_features:
            def input_transform(ipt, feats):
                x = self.binfeats_trans(ipt)
                if feats is not None and feats.dim() == 4:
                    x = torch.cat([x, feats], dim=1)
                return x
            return input_transform
        else:
            def input_transform(ipt, feats):
                return feats
            return input_transform

    def _input_channels(self) -> int:
        num = 0
        if mcfg.use_formants:
            num += 2 if (mcfg.dual_feats and mcfg.formants_diff) else 1
        if mcfg.use_pitch:
            num += 2 if (mcfg.dual_feats and mcfg.pitch_diff) else 1
        if mcfg.use_melspec:
            num += 2 if (mcfg.dual_feats and mcfg.melspec_diff) else 1
        if mcfg.use_binary_features:
            num += len(mcfg.thresholds)
        assert num >= 1
        return num

    def forward(self, ipt, feats: Tensor = None) -> Tensor:
        # if mcfg.use_melspec:
        #     x = self.todb(self.melspec(ipt)) \
        #         if mcfg.log_mel is True else self.melspec(ipt)
        #     if mcfg.melspec_diff:
        #         melspec_diff = torch.zeros_like(x, device=x.device)
        #         melspec_diff[:,:,:,1:] = x[:,:,:,1:] - x[:,:,:,:-1]
        #         x = torch.cat([x, melspec_diff], dim=1) \
        #             if mcfg.dual_feats else melspec_diff
        #     if feats.dim() == 4:
        #         x = torch.cat([x, feats], dim=1)
        # elif mcfg.use_binary_features:
        #     x = self.binfeats_trans(ipt)
        # else:
        #     x = feats
        x = self.input_transform(ipt, feats)
        x = self.cnn1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.cnn2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)
        x = self.cnn3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.relu4(x)
        x = self.dropout3(x)
        x = self.head(x)
        return x


def load_model(model_name: str = None) -> Union[CNN3_DNN2, RawNet]:
    # print(f'\nModel parameters received: {str(mcfg)}\n')
    if model_name is None:
        model_name = mcfg.model_name
    if model_name == 'cnn3_dnn2':
        model = CNN3_DNN2()
    elif model_name == 'cnn_attention':
        model = CNN_Attention(mcfg.__dict__)
    return model
