import time
import torch
import utils
import random
import librosa
import datetime
import numpy as np
from tqdm import tqdm
from models import load_model
from utils import get_logger
from argparse import ArgumentParser
# from sklearn.metrics import confusion_matrix
from default_config import model_config as mcfg
from default_config import test_config as tscfg
from typing import Dict, List, Tuple, Union
from feature_extract import extract_features, subsample_feats

logger = get_logger(name='test')

time_str = lambda: datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

parser = ArgumentParser()
parser.add_argument('model_path', type=str, help='Model weight path.')
parser.add_argument('--adv_dirs', type=str, default=None, nargs='+', help='Adversarial wav samples directories.')
parser.add_argument('--cln_dirs', type=str, default=None, nargs='+', help='Clean wav samples directories.')
parser.add_argument('--step', type=float, default=None, help='Sliding window step size in seconds.')
parser.add_argument('--length_limit', type=float, default=None, help='Wav file length limit (greater than 0).')
parser.add_argument('--padding', type=int, default=None, help='Padding the end of the audio file.')
parser.add_argument('--model_name', type=str, default=None, help='cnn3_dnn2/cnn_attention')
parser.add_argument('--test_folder_limit', type=int, default=None, help='Test folder audio number limit.')
parser.add_argument('--dual_feats', type=int, default=None, help='Use both the original feature and its 1st difference .')
parser.add_argument('--use_melspec', type=int, default=None, help='Use mel spectrogram as feature.')
parser.add_argument('--melspec_diff', type=int, default=None, help="Use mel spectrogram's 1st difference.")
parser.add_argument('--use_formants', type=int, default=None, help='Use formants as feature.')
parser.add_argument('--formants_diff', type=int, default=None, help="Use formants' 1st difference.")
parser.add_argument('--use_pitch', type=int, default=None, help='Use pitch as feature.')
parser.add_argument('--pitch_diff', type=int, default=None, help="Use pitch's 1st difference.")
args = parser.parse_args()

if args.step is not None: 
    tscfg.step = args.step
if args.padding is not None: 
    tscfg.padding = bool(args.padding)
if args.length_limit is not None:
    tscfg.length_limit = args.length_limit
if args.test_folder_limit is not None:
    tscfg.test_folder_limit = args.test_folder_limit

if args.model_name is not None: 
    mcfg.model_name = args.model_name
for k, v in args.__dict__.items():
    if v is not None:
        if k in mcfg.__dict__.keys():
            mcfg.__dict__[k] = type(mcfg.__dict__[k])(v)
            logger.info(f'Model config set: {k}={mcfg.__dict__[k]}')

logger.info(f'allcfg received: {str(tscfg)}')
####################################################################################

model = load_model(mcfg.model_name)
if args.model_path is not None:
    state = torch.load(args.model_path)
    # del state['melspec.spectrogram.window']
    # del state['melspec.mel_scale.fb']
    model.load_state_dict(state)
    logger.info(f'Restore from: {args.model_path}')
model.cuda()
model.eval()
# del model.melspec


def predict(wav: str) -> int:
    wave, _ = librosa.load(wav, sr=mcfg.sr, mono=True)
    if tscfg.length_limit > 0:
        wave, _ = librosa.load(
            wav, sr=mcfg.sr, mono=True, duration=tscfg.length_limit)
    else:
        wave, _ = librosa.load(wav, sr=mcfg.sr, mono=True)
    if mcfg.model_name == 'cnn3_dnn2':
        wave_padded = np.pad(
            wave, 
            (0, int(mcfg.sr * mcfg.duration)), 
            mode='constant', 
            constant_values=0)
        if tscfg.padding is True:
            wave = wave_padded

        if mcfg.use_formants or mcfg.use_pitch:
            feats: Dict[str, torch.Tensor] = extract_features(wave_padded)
        else:
            feats = None

        for i in range((wave.shape[0] - int(mcfg.sr * mcfg.duration))\
                 // int(tscfg.step * mcfg.sr) + 1):
            nfrom = int(i * tscfg.step * mcfg.sr)
            nto = int((i * tscfg.step + mcfg.duration) * mcfg.sr)
            wave_i = wave[nfrom:nto]
            wavei_tensor = torch.tensor(wave_i).unsqueeze(0).unsqueeze(0).float().cuda()
            if feats is not None:
                featsi: Dict[str, torch.Tensor] = \
                    subsample_feats(feats, start=nfrom / mcfg.sr)
                featsi = torch.cat(list(featsi.values()), dim=0).unsqueeze(dim=0).float().cuda()
            else:
                featsi = torch.tensor(0)
            output = model(wavei_tensor, featsi).cpu()
            pred = torch.argmax(output.view(-1)).item()
            if pred == 1:
                return 1
        return 0
    elif mcfg.model_name == 'cnn_attention':
        def pad(x):
            x_len = x.shape[0]
            if x_len >= mcfg.nb_samp:
                return x[:mcfg.nb_samp]
            # need to pad
            num_repeats = int(mcfg.nb_samp / x_len)+1
            padded_x = np.tile(x, (1, num_repeats))[:, :mcfg.nb_samp][0]
            return padded_x	
        wave = pad(wave)
        wave_tensor = torch.tensor(wave).unsqueeze(0).float().cuda()
        output = model(wave_tensor)
        return torch.argmax(output.view(-1)).item()
    else:
        raise ValueError(f'Invalid config set "model_name"={mcfg.model_name}')

def get_predictions(
    wavs: List[str], ground_truth: List[int] = None
    ) -> Union[List[int], Tuple[List[int], float]]:
    predictions = list()
    with torch.no_grad():
        progress = tqdm(wavs)
        for wav in progress:
            predictions.append(predict(wav))
            if ground_truth is not None:
                acc = sum(np.array(predictions) == \
                    np.array(ground_truth[:len(predictions)])) / len(predictions)
                progress.set_description(f'ACC: {acc:.4f}')
    # if ground_truth is None:
    return predictions
    # else:
    #     return predictions, acc


def predict_wavs_in_folder(folder: str, is_adv: bool) -> Tuple[int, int]:
    wavs = utils.find_all_ext(folder, 'wav')
    logger.info(f'Got {len(wavs)} test {"adv" if is_adv else "clean"} samples from {folder}')
    if tscfg.test_folder_limit > 0 and len(wavs) > tscfg.test_folder_limit:
        random.seed(tscfg.random_seed)
        random.shuffle(wavs)
        wavs = wavs[:tscfg.test_folder_limit]
        logger.info(f'Randomly select {len(wavs)} test samples from {folder}')
    start_t = time.time()
    preds = np.array(get_predictions(wavs, [1 if is_adv else 0] * len(wavs)))
    lapsed = time.time() - start_t
    n_right, n_all = sum((preds == (1 if is_adv else 0))), len(preds)
    tpr, lapsed_mean = n_right/n_all, lapsed / len(wavs)
    # utils.log_to_file(
    #     tscfg.log_path, 
    #     f'TIME:{time_str()}',
    #     f'MODEL:{args.model_path}', 
    #     f'allcfg:{str(tscfg)}',
    #     f'ADV:{str(folder)},{len(wavs)}',
    #     f'TPR:{tpr:.4f}', 
    #     f'LAPSED:{lapsed:.4f},{lapsed_mean:.6f}'
    #     )
    logger.info(f'Log to path: {tscfg.log_path}')
    logger.info(f'Test time: {time_str()}')
    logger.info(f'Model path: {args.model_path}')
    logger.info(f'Config: {str(tscfg)}')
    logger.info(f'Test samples folder: {str(folder)}, {len(wavs)} samples')
    logger.info(f'True positive rate: {tpr:.4f}')
    logger.info(f'Time lapsed: {lapsed:.4f}, {lapsed_mean:.6f} per sample')
    # logger.info(f'test-acc ({folder}, {len(wavs)}): {tpr:.4f}')
    # adv_preds = np.concatenate([adv_preds, preds], axis=-1)
    return n_right, n_all


# data folders
if args.adv_dirs is None and args.cln_dirs is None:
    adv_dirs = [tscfg.test_adv_dirs] if \
        isinstance(tscfg.test_adv_dirs, str) else tscfg.test_adv_dirs
    cln_dirs = [tscfg.test_cln_dirs] if \
        isinstance(tscfg.test_cln_dirs, str) else tscfg.test_cln_dirs

    # adv_dirs: List[Tuple[str, int]] = [tscfg.test_adv_dirs,] if \
    #     isinstance(tscfg.test_adv_dirs[0], str) else tscfg.test_adv_dirs
    # cln_dirs: List[Tuple[str, int]] = [tscfg.test_cln_dirs,] if \
    #     isinstance(tscfg.test_cln_dirs[0], str) else tscfg.test_cln_dirs
else:
    adv_dirs = args.adv_dirs if args.adv_dirs is not None else []
    cln_dirs = args.cln_dirs if args.cln_dirs is not None else []
if len(cln_dirs) == 0 and len(adv_dirs) == 0:
    exit(0)

adv_rights, adv_alls = [], []
for adv_dir in adv_dirs:
    n_right, n_all = predict_wavs_in_folder(adv_dir, is_adv=True)
    adv_rights.append(n_right)
    adv_alls.append(n_all)

cln_rights, cln_alls = [], []
for cln_dir in cln_dirs:
    n_right, n_all = predict_wavs_in_folder(cln_dir, is_adv=False)
    cln_rights.append(n_right)
    cln_alls.append(n_all)

all_rights = adv_rights + cln_rights
all_alls = adv_alls + cln_alls

logger.info(
    f'Accuracy of label 1: ' + \
    f'{sum(adv_rights) / sum(adv_alls) if len(adv_alls) != 0 else "nan"}')
logger.info(
    f'Accuracy of label 0: ' + \
    f'{sum(cln_rights) / sum(cln_alls) if len(cln_alls) != 0 else "nan"}')
logger.info(f'Accuracy in total  : {sum(all_rights) / sum(all_alls)}')

for folder, n_right, n_all in zip(adv_dirs + cln_dirs, all_rights, all_alls):
    logger.info(f'test-acc ({folder}, {n_all}): {n_right / n_all:.4f}')
