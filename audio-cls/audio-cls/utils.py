import os, glob, itertools, json, logging
import torchaudio
from ast import literal_eval

def dict_eval(d):
    for k, v in d.items():
        try:  # number/tuple/dict/list type
            d[k] = literal_eval(v)
        except:  # str type
            pass
    return d


def find_all_ext(root, ext: str):
    paths = []
    for dir_name, _, _ in os.walk(root):
        paths += glob.glob(os.path.join(dir_name, '*.' + ext))
    return paths


def group_by_key(names: list, key=None):
    names_sorted = sorted(names, key=key)
    names_grouped = itertools.groupby(names_sorted, key=key)
    groups = [list(grouper) for _, grouper in names_grouped]
    return groups


def lists_to_one(lists: list):
    return list(itertools.chain(*lists))


def load_wav(path, sr=None):
    # torchaudio.set_audio_backend('soundfile')
    wave, sr_ori = torchaudio.load(path)
    if sr is None:
        return wave, sr_ori
    if sr_ori != sr:
        wave = torchaudio.functional.resample(wave, sr_ori, sr, lowpass_filter_width=64)
    return wave


def load_wav2(path1, path2, sr=16000):
    wave1 = load_wav(path1, sr)
    wave2 = load_wav(path2, sr)
    ori_shape = (wave1.shape[1], wave2.shape[1])
    n_samples = min(wave1.shape[1], wave2.shape[1])  # ! cut the longer one
    wave1, wave2 = wave1[:, :n_samples], wave2[:, :n_samples]
    return wave1, wave2, ori_shape


def makedir(path):
    dirname = os.path.dirname(path)
    if os.path.exists(dirname) is False:
        os.makedirs(dirname)
    return path


def join_and_make(*args: list):
    return makedir(os.path.join(*args))


def basebasename(path: str):
    return os.path.basename(os.path.dirname(path))


def concate_path(path: str, root: str = None, connector: str = '-'):
    if root is None:
        rel_path = path
    else:
        root_path = os.path.abspath(root)
        file_path = os.path.abspath(path)
        rel_path = file_path.replace(root_path, '')
    return connector.join(rel_path.strip(os.path.sep).split(os.path.sep))


def save_wav(wave, path, sr=16000):
    torchaudio.save(makedir(path),
                    wave,
                    sample_rate=sr,
                    format='wav',
                    encoding='PCM_S',
                    bits_per_sample=16)
    return path


def read_json(path: str):
    with open(path) as f:
        return json.load(f)


def write_json(path: str, data: dict, update=False, convert_to_str: bool = False):
    if convert_to_str is True:
        data_w = dict()
        for k, v in data.items():
            data_w[k] = str(v)
    else:
        data_w = data
    if update is True:
        info: dict = read_json(path)
        data_w = info.update(data_w)
    with open(makedir(path), 'w') as f:
        f.write(json.dumps(data_w, indent=1))


def log_to_file(path: str, *logs: list) -> str:
    log_line = '; '.join([str(item) for item in logs])
    with open(path, 'a') as logf:
        logf.write(log_line + '\n')
    return log_line


def get_logger(
    name: str = __name__, 
    file_level: int = logging.INFO, 
    stdout_level: int = logging.DEBUG):
    formatter = logging.Formatter(
        r"%(asctime)s - %(name)s - %(levelname)-9s - %(filename)-8s : %(lineno)s line - %(message)s",
        datefmt=r"%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(name + '.log')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(file_level)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(stdout_level)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger
