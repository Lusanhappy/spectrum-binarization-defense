from os import path
from tqdm import tqdm
from argparse import ArgumentParser
import utils

argparser = ArgumentParser()
argparser.add_argument('src_folder', type=str)
argparser.add_argument('dst_folder', type=str)
argparser.add_argument('duration', type=int)
argparser.add_argument('--ext', type=str, default='wav')
argparser.add_argument('--skip', type=int, default=0)
argparser.add_argument('--reverse', action='store_true', default=False)
args = argparser.parse_args()
print(args)
src_folder, dst_folder = args.src_folder.rstrip(path.sep), args.dst_folder.rstrip(path.sep)
print('source folder:', src_folder)
print('dest folder:', dst_folder)

wavs = utils.find_all_ext(src_folder, args.ext)
print('Number of wavs found:', len(wavs))
wavs = tqdm(wavs)
for wav in wavs:
    wav_dst = utils.makedir(path.join(dst_folder, wav.split(path.basename(src_folder).strip(path.sep))[-1].strip(path.sep)))
    wave = utils.load_wav(wav, sr=16000)
    wave = wave[:, 16000 * args.skip:]
    if args.reverse is False:
        utils.save_wav(wave[:, :16000 * args.duration], wav_dst, sr=16000)
    else:
        utils.save_wav(wave[:, -16000 * args.duration:], wav_dst, sr=16000)

