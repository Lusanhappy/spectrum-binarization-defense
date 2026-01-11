import librosa
import os, glob
from tqdm import tqdm
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('wav_folder', type=str)
parser.add_argument('--exts', type=str, default=None, nargs='+')
args = parser.parse_args()

root = args.wav_folder
exts = ['wav'] if args.exts is None else args.exts

paths = []
for dir_name, _, _ in os.walk(root):
    for ext in exts:
        paths += glob.glob(os.path.join(dir_name, '*.' + ext))

print(f'{len(paths)} audios found.')

length_sum = 0
for path in tqdm(paths):
    wave, _ = librosa.load(path, sr=16000, mono=True)
    length_sum += wave.shape[0] / 16000

print(f'{length_sum:.2f} seconds ({length_sum/60:.2f} min, {length_sum/3600:.2f} h) in total.')
