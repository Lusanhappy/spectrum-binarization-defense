from os import path
from argparse import ArgumentParser
from shutil import copy, move
import utils, random

argparser = ArgumentParser()
argparser.add_argument('src_folder', type=str)
argparser.add_argument('dst_folder', type=str)
argparser.add_argument('--num', type=int, default=None)
argparser.add_argument('--move', action='store_true', default=False)
args = argparser.parse_args()
src_folder, dst_folder = args.src_folder, args.dst_folder

wavs = utils.find_all_ext(src_folder, 'wav')
random.seed(999)
random.shuffle(wavs)
if args.num is not None:
    wavs = wavs[:args.num]
for wav in wavs:
    wav_dst = utils.makedir(path.join(dst_folder, wav.split(src_folder.strip(path.sep))[-1].strip(path.sep)))
    if args.move is False:
        copy(wav, wav_dst)
    else:
        move(wav, wav_dst)

