# coding=utf-8
import collections
import contextlib
import copy
import glob
import os.path
import random
import shutil

import deprecation
import sox
import webrtcvad
from spleeter.separator import Separator

from deepspeech_ctc_score import deepspeech_ctc_score
from formant_processor import *
from psychological import generate_th, compute_PSD_matrix, get_masking_threshold
from utils import wav_read, EPS

# Noted: In order to make the name of clip music more readable, we suppose that the audio name you assigned retains only meaningful words as much as possible.
__default_pick_params__ = {
    "re_analysis": False,  # Whether re-analysis the music and command when the analysis file is existing
    # basic setting
    "task_name": "experiment_template",  # Control the task name
    "command_folder": None,  # Where is the command saved. Must be specified.
    "music_folder": None,  # Where the music is saved. Must be specified.
    "max_workers": 4,  # Control the maximum number of multi-process.
    "padding_frame_num": 1000,  # Ignore the start and end of the music.
    "margin_frame_num": 50,  # Keep K frames at the pre- and post-frames of the selected music clips.
    "frame_time": 0.025,  # Frame time in second.
    "step_time": 0.01,  # Step time in second.
    "pdf_frame_time": 0.025,  # Frame time while extracting pdf-id.
    "pdf_step_time": 0.01,  # Step time while extracting pdf-id.
    "step_frame_num": 5,  # Step frame num used when analyzing music.

    # wake-up setting
    "using_wake_up": False,  # Whether using wake up
    "wake_up_folder": None,  # Where is the wake-up saved.
    "wake_up_weight": 0.5,  # Control the weight of wake-up formant distance. And the weight of command's formant distance is '1-wake_up_weight'.
    "min_pause_time": 0.75,  # Control the minimum interval between wake-up and command. If you do not want any pause between wake-up and command, set 'min_pause_time=0.0' and 'max_pause_time=pause_step_time>0'.
    "max_pause_time": 1.0,  # Control the maximum interval between wake-up and command.
    "pause_step_time": 0.05,  # Control the step of interval time between min_pause_time and max_pause_time.

    # spleeter setting
    "using_spleeter": False,  # Control the script whether to use spleeter to separate the human voice from music and use it to calculate the formant distance in `align_formant`. It's better to use spleeter if the source audio is a piece of music, or not if it's a conversation.

    # VAD setting
    "music_vad_aggressive_level": 2,  # Control the aggressive level of vad for music wave file between [0, 3]. Default 3 is better than others for music.
    "vocal_vad_aggressive_level": 2,  # Control the aggressive level of vad for vocal wave file between [0, 3]. Default 2 is better than others for vocal.

    # pick setting
    "allow_overlap": False,  # Whether allow overlapping in selected music clips.
    "pick_top_k": 10,  # How many music clips will be selected.

    # sil-phoneme setting
    "do_sil_phoneme_filter": False,  # whether do filter clips by phone type alignment. A Voice/Vowel phoneme must be inserted into a not-sil frame.
    "allowed_sil_phone_ratio": 0.1,  # The smaller the better, but less clips will be picked.

    # voice-phoneme pronunciation alignment setting
    "do_pronunciation_filter": False,  # Whether do pronunciation filter.
    "pronunciation_match_len": 4,  # The minimum matching length of one phoneme.
    "matched_phone_num_threshold": 2,  # The threshold of minimum matched phoneme number.
    "pronunciation_weight": 2.0,

    # formant flatness setting
    "do_flatness_filter": False,  # whether do filter clips by formant flatness analysis. A flat formant means that there are not enough phonemes in clip.
    "flatness_threshold": 0.98,  # Specify the flatness threshold of formant on the time dimension. If the flatness value is greater than 'flatness_threshold', then the clip will be deprecated. The smaller the better.
    "flatness_window_frame_num": 100,  # Specify the calculation window of flatness.
    "flatness_step_frame_num": 25,  # Specify the calculation step frame of flatness.
    "do_formant_filter": False,  # Whether do filter clips by formant standard deviation.
    "formant_threshold": 45.,  # Specify the flatness threshold of formant list. # The bigger the better.
    "formant_window_frame_num": 50,  # Specify the calculation window of formant standard deviation.

    # pitch setting
    "do_pitch_filter": True,  # Whether do filter clips by pitch.
    "pitch_threshold": 1.0,  # threshold of the standard deviation of pitch. The bigger the better.
    "pitch_window_frame_num": 40,  # Specify the calculation window length

    # intensity setting
    "do_intensity_filter": True,  # Whether do filter clips by intensity.
    "intensity_window_frame_num": 40,
    "intensity_scale_threshold": 65.,
    "intensity_std_threshold": 1.,

    # formant distance setting
    "do_formant_similarity_filter": False,  # Whether do filter clips by formant similarity.
    "formant_similarity_weight": [0.3, 0.3, 0.4, 0, 0],  # Which only care about the 2/3-order formant.
    "max_up_shift_fre": [100, 100, 100, 100, 100],  # Control the maximum up-shift frequency. It supports int/float number or one-dimension array with shape (MAX_FORMANT+NUM, ).
    "max_down_shift_fre": [100, 100, 100, 100, 100],  # Control the maximum down-shift frequency. It supports int/float number or one-dimension array with shape (MAX_FORMANT_NUM, ).
    "formant_similarity_threshold": 2.5,  # The smaller the better
    "formant_distance_weight": 1.0,

    # psychological and balance coefficient setting, and it's better to use the most possible value which would be set in generating adversarial samples.
    "do_psychological_filter": True,  # Whether do filter clips by psychological masking threshold.
    "psychological_threshold": 3.,  # the smaller the better
    "psychological_weight": 2.0,
    "pre_emphasis": False,  # Whether add pre-emphasis while calculating power spectrum. Default is False.
    "window_func": sqrt_hanning_window,  # The window function in calculating power spectrum. Default is kaiser window.
    "bandwidth": 300,  # the bandwidth used to calculate the energy around formants.
    "delta_db": 15.,
    "formant_weight": [1.0, 1.0, 1.0, 0.0, 0.0],
    "filter_order": 2,
    "voiceless_noise_db": 65.,
    "using_formant_shift": False,
    "auto_balance_coe": False,
    "fre_interval": (100, 4000),
    "reserved_ratio": 0.3,
    "voiceless_process_type": VoicelessProcessType.BlueNoise,
    "voiceless_fre_interval": (2000., 4000.),
    "balance_weight": 0.5,

    # Deepspeech CTC Score Setting.
    "using_ctc_score": False,  # Whether using deepspeech ctc score to filter the clips.
    "ctc_reserved_ratio": 0.3,  # How many clips would be reserved after filtering clips using ctc score.
}


def gen_transaction_phone_list(word_list: list) -> Tuple[List[list], List[list]]:
    phone_list_list = word2phone(word_list[0])
    phone_length_list_list = []
    for phone_list in phone_list_list:
        phone_length_list_list.append([len(phone_list)])
    if len(word_list) == 1:
        return phone_list_list, phone_length_list_list
    else:
        transaction_phone_list_list = []
        word_phone_length_list_list = []
        sub_transaction_phone_list_list, sub_transaction_phone_length_list_list = gen_transaction_phone_list(word_list[1:])
        for phone_list in phone_list_list:
            for sub_transaction_phone_list in sub_transaction_phone_list_list:
                transaction_phone_list_list.append(phone_list + sub_transaction_phone_list)
        for phone_length_list in phone_length_list_list:
            for sub_phone_length_list in sub_transaction_phone_length_list_list:
                word_phone_length_list_list.append(phone_length_list + sub_phone_length_list)
        return transaction_phone_list_list, word_phone_length_list_list


def check_phoneme(phone_list: List[str], transaction: str, wav_name: str) -> Tuple[List[str], int, int, Union[None, list]]:
    """
    检查音素序列
    """
    # 假设command均为较为干净的语音，故将PhoneType.Sil frame全部转化为sil
    _index_ = 0
    while _index_ < len(phone_list):
        if phone_list[_index_] in ["laughter", "noise", "oov"]:
            phone_list[_index_] = "sil"
        _index_ += 1

    # 处理音素时长仅为1帧长, 修改为其后一帧的音素
    _index_ = 0
    while _index_ < len(phone_list):
        _sub_index_ = _index_ + 1
        while _sub_index_ < len(phone_list):
            if phone_list[_index_] == phone_list[_sub_index_]:
                _sub_index_ += 1
            else:
                break
        if _sub_index_ - _index_ == 1:
            if phone_list[_index_] == "sil":
                _index_ = _sub_index_
            elif _sub_index_ < len(phone_list):
                phone_list[_index_] = phone_list[_sub_index_]
            else:
                phone_list[_index_] = phone_list[_index_ - 1]
                break
        else:
            _index_ = _sub_index_

    # 音素未以sil开始
    if phone_list[0] != "sil":
        _index_ = 1
        while _index_ < len(phone_list):
            if phone_list[_index_] != phone_list[0]:
                if phone_list[_index_] == "sil":
                    logger.warning(u"The pdf phone list does not start with 'sil', and it's likely that there's something wrong with the Kaldi-PDF label. So the script modifies the first few phonemes to SIL.")
                    phone_list[:_index_] = "sil"
                break
            _index_ += 1

    # 使用transaction修正部分音素
    _did_correct_ = False
    processed_word_phone_list = []
    word_list = transaction.strip().split()
    if transaction:
        try:
            transaction_phone_list_list, word_phone_length_list_list = gen_transaction_phone_list(word_list)
        except ValueError as _err_:
            logger.error(_err_)
        else:
            gather_phoneme_list = [phone_list[0]]
            for _phone_ in phone_list:
                if _phone_ != gather_phoneme_list[-1]:
                    gather_phoneme_list.append(_phone_)
            for _available_phone_index_ in range(len(transaction_phone_list_list)):
                transaction_phone_list = transaction_phone_list_list[_available_phone_index_]
                word_phone_length_list = word_phone_length_list_list[_available_phone_index_]

                try:
                    tmp_gather_phoneme_list = copy.deepcopy(gather_phoneme_list)
                    _i_index_, _j_index_ = 0, 0
                    while _i_index_ < len(tmp_gather_phoneme_list) and _j_index_ < len(transaction_phone_list):
                        if tmp_gather_phoneme_list[_i_index_] == transaction_phone_list[_j_index_]:
                            _i_index_ += 1
                            _j_index_ += 1
                            continue
                        # 清音浊化/相同发音
                        _replace_phoneme_ = False
                        if tmp_gather_phoneme_list[_i_index_] == "d" and transaction_phone_list[_j_index_] == 't':
                            _replace_phoneme_ = True
                        elif tmp_gather_phoneme_list[_i_index_] == "b" and transaction_phone_list[_j_index_] == "p":
                            _replace_phoneme_ = True
                        elif tmp_gather_phoneme_list[_i_index_] == 'g' and transaction_phone_list[_j_index_] == 'k':
                            _replace_phoneme_ = True
                        elif tmp_gather_phoneme_list[_i_index_] == 'z' and transaction_phone_list[_j_index_] == 's':
                            _replace_phoneme_ = True
                        elif tmp_gather_phoneme_list[_i_index_] == "ah" and transaction_phone_list[_j_index_] == "er":
                            _replace_phoneme_ = True
                        elif tmp_gather_phoneme_list[_i_index_] == 'iy' and transaction_phone_list[_j_index_] == "ey":
                            _replace_phoneme_ = True
                        elif tmp_gather_phoneme_list[_i_index_] == 'ah' and transaction_phone_list[_j_index_] == 'eh':
                            _replace_phoneme_ = True
                        elif tmp_gather_phoneme_list[_i_index_] == "ng" and transaction_phone_list[_j_index_] == "n":
                            _replace_phoneme_ = True
                        elif tmp_gather_phoneme_list[_i_index_] == 'p' and transaction_phone_list[_j_index_] == "hh":
                            _replace_phoneme_ = True

                        if _replace_phoneme_:
                            transaction_phone_list[_j_index_] = tmp_gather_phoneme_list[_i_index_]
                            _i_index_ += 1
                            _j_index_ += 1
                            continue

                        tmp_gather_phoneme_list[_i_index_] = "sil"
                        _i_index_ += 1
                    while _i_index_ < len(tmp_gather_phoneme_list):
                        tmp_gather_phoneme_list[_i_index_] = "sil"
                        _i_index_ += 1
                    if _j_index_ != len(transaction_phone_list):
                        raise ValueError("Can't align the phoneme sequence.")
                except ValueError:
                    continue
                else:
                    _i_index_ = 0
                    _index_ = 0
                    pre_phone = phone_list[0]
                    while _index_ < len(phone_list):
                        if phone_list[_index_] == pre_phone:
                            phone_list[_index_] = tmp_gather_phoneme_list[_i_index_]
                        else:
                            pre_phone = phone_list[_index_]
                            _i_index_ += 1
                            phone_list[_index_] = tmp_gather_phoneme_list[_i_index_]
                        _index_ += 1
                    pre_index = 0
                    for phone_length in word_phone_length_list:
                        processed_word_phone_list.append(transaction_phone_list[pre_index:pre_index + phone_length])
                        pre_index += phone_length

                    _did_correct_ = True
                    break

    # 删除两端的静音
    _start_index_ = 0
    while _start_index_ < len(phone_list):
        if phone_list[_start_index_] == "sil":
            _start_index_ += 1
        else:
            break
    _end_index_ = len(phone_list)
    while _end_index_ >= 0:
        if phone_list[_end_index_ - 1] == "sil":
            _end_index_ -= 1
        else:
            break
    assert _end_index_ > _start_index_
    phone_list = phone_list[_start_index_: _end_index_]

    # 单词-帧序号对齐
    word_index_list = None  # for feed-back in sample generation
    if _did_correct_:
        word_index_list = []
        _index_ = 0
        _word_index_ = 0
        while _index_ < len(phone_list) and _word_index_ < len(processed_word_phone_list):
            while _index_ < len(phone_list):
                if phone_list[_index_] != 'sil':
                    break
                _index_ += 1

            _s_ = _index_

            _phone_index_ = 0
            while _phone_index_ < len(processed_word_phone_list[_word_index_]) and _index_ < len(phone_list):
                if processed_word_phone_list[_word_index_][_phone_index_] == phone_list[_index_]:
                    _index_ += 1
                else:
                    _phone_index_ += 1

            _e_ = _index_
            word_index_list.append((word_list[_word_index_], _s_, _e_))
            _word_index_ += 1

        logger.info(u"For Wave Name: '{}'".format(wav_name))
        logger.info(u"\t\tprocessed_word_phone_list: {}".format(processed_word_phone_list))
        logger.info(u"\t\tword_index_list: {}".format(word_index_list))
        logger.debug(u"\t\tNoted: the word index is a left-close and right-open interval.")
    else:
        logger.warning(u"Can't generate the word_index_list for wave '{}'. You may need to check align-algorithm or alter the pdf by yourself.".format(wav_name))

    return phone_list, _start_index_, _end_index_, word_index_list


def check_audio(params: dict):
    """
    备份音乐、命令和唤醒词文件
    """
    logger.info(u"[Check] Check Audio/Pdf-ID/Transaction Files.")

    wake_up_folder = params['wake_up_folder']
    command_folder = params['command_folder']
    music_folder = params['music_folder']
    _using_wake_up_ = params['using_wake_up']

    sample_rate = None
    music_wav_files = glob.glob(os.path.join(music_folder, "*.wav"))
    music_wav_files = filter_irrelevant_wav(music_wav_files)
    if not len(music_wav_files):
        raise FileNotFoundError("The music folder '{}' is not found, please confirm whether its folder exists.".format(music_folder))
    for music_wav_file in music_wav_files:
        m_sample_rate, music_signal = wav_read(music_wav_file)
        if sample_rate is None:
            sample_rate = m_sample_rate
        else:
            if sample_rate != m_sample_rate:
                raise ValueError("There are multi different sample_rate in music wav files. Please make sure the audios have the same sampling rate.")
        if len(music_signal) < 4096:
            logger.warning(u"The music file '{}' is too short. So the script ignore this music.".format(music_wav_file))
            os.remove(music_wav_file)
            continue
        music_name = os.path.splitext(os.path.basename(music_wav_file))[0]
        if "-" in music_name:
            raise ValueError("Please do not use '-' in wave name '{}'.".format(music_name))

    wav_folders = [command_folder] if not _using_wake_up_ else [command_folder, wake_up_folder]
    for wav_folder in wav_folders:
        wav_files = glob.glob(os.path.join(wav_folder, "*.wav"))
        wav_files = filter_irrelevant_wav(wav_files)
        if not len(wav_files):
            raise FileNotFoundError("The destination folder '{}' is not found, please confirm whether its folder exists.".format(wav_folder))

        for wav_file in wav_files:
            c_sample_rate, _ = wav_read(wav_file)
            if c_sample_rate != sample_rate:
                raise ValueError("There are multi different sample_rate between music and command wav files. Please make sure the audios have the same sampling rate.")

            wav_raw_name = os.path.splitext(os.path.basename(wav_file))[0]
            if "-" in wav_raw_name:
                raise ValueError("Please do not use '-' in wave name '{}'.".format(wav_raw_name))
            pdf_file = os.path.join(wav_folder, "{}.csv".format(wav_raw_name))
            txt_file = os.path.join(wav_folder, "{}.txt".format(wav_raw_name))

            if not os.path.exists(pdf_file):
                raise FileNotFoundError("The pdf file for wav '{}' is not found. Make sure that it's basename is equal to the wave file.".format(pdf_file))
            if not os.path.exists(txt_file):
                logger.warning("The txt file '{}' which saves the transcription of wake-up/command is not found. Without this file, you will not able to use the full functionality of scripts.".format(txt_file))
    logger.info(u"[Check] Check Done!")


def align_phoneme(music_phone_list: Union[List[str], np.ndarray], command_phone_list: Union[List[str], np.ndarray], match_len: int = 4) -> Tuple[List[str], int]:
    assert len(music_phone_list) == len(command_phone_list), "Length Unequal."
    music_pron_list = get_pronunciation(music_phone_list)
    command_pron_list = get_pronunciation(command_phone_list)

    result_len = 0
    matched_phones = []
    phone_len = 0
    matched_len = 0
    pre_phone = None
    for _index_ in range(len(music_pron_list)):
        if command_phone_list[_index_] != pre_phone:
            if pre_phone is not None and (matched_len >= match_len or phone_len == matched_len):
                matched_phones.append(pre_phone)
                result_len += matched_len
            if command_pron_list[_index_] is not None:
                pre_phone = command_phone_list[_index_]
            else:
                pre_phone = None
            phone_len = 0
            matched_len = 0
        if command_phone_list[_index_] is None:
            continue
        phone_len += 1

        if command_pron_list[_index_] == music_pron_list[_index_]:
            matched_len += 1
    if pre_phone is not None and (matched_len >= match_len or phone_len == matched_len):
        matched_phones.append(pre_phone)
        result_len += matched_len
    return matched_phones, result_len


@exception_printer
def analysis_music(music_wav_file: str, params: dict):
    frame_time = params['frame_time']
    step_time = params['step_time']
    music_vad_aggressive_level = params['music_vad_aggressive_level']
    vocal_vad_aggressive_level = params['vocal_vad_aggressive_level']
    window_func = params['window_func']
    _pre_emphasis_ = params['pre_emphasis']
    _do_pronunciation_filter_ = params['do_pronunciation_filter']

    wav_name = os.path.splitext(os.path.basename(music_wav_file))[0]
    analysis_file = music_wav_file.replace(".wav", ANALYSIS_SUFFIX)
    if os.path.exists(analysis_file):
        with open(analysis_file, 'rb') as _file_:
            script_version = pickle.load(_file_).get("script_version")
        if script_version == SCRIPT_VERSION and not params['re_analysis']:
            logger.info(u"Using existing analysis file for music '{}'.".format(wav_name))
            return
        else:
            logger.warn(u"The analysis file for music '{}' is out of date, re-analysis it now.".format(wav_name))

    m_sample_rate, music_signal = wav_read(music_wav_file)
    # music analysis
    formant_list = get_formants(music_wav_file, frame_time=frame_time, step_time=step_time)
    sil_label_list = label_sil(music_wav_file, frame_time=frame_time, step_time=step_time, vad_level=music_vad_aggressive_level)
    formant_list[sil_label_list] = np.NaN
    analysis_data = {
        "script_version": SCRIPT_VERSION,
        'formant_list': formant_list,
        'sil_label_list': sil_label_list,
        'music_power_spectrum': file_2_power_spectrum(music_wav_file, frame_time=frame_time, step_time=step_time, pre_emphasis=_pre_emphasis_, window_func=window_func),
        "pitch_list": get_pitch(music_wav_file, frame_time=frame_time, step_time=step_time),
        "intensity_list": get_intensity(music_wav_file, frame_time=frame_time, step_time=step_time),
        "psychological_masking_list": get_masking_threshold(music_wav_file, frame_time=frame_time, step_time=step_time),
        "sample_rate": m_sample_rate,
        "music_signal": music_signal
    }
    music_pdf_file = music_wav_file.replace(".wav", '.csv')
    _phone_exist_ = False
    if os.path.exists(music_pdf_file):
        music_phone_list = get_phones(music_pdf_file, frame_time, step_time)
        analysis_data['phone_list'] = music_phone_list
        _phone_exist_ = True

    # vocal analysis
    separate_vocal_file = music_wav_file.replace(".wav", VOCAL_WAV_SUFFIX)
    _, vocal_signal = wav_read(separate_vocal_file)
    formant_list = get_formants(separate_vocal_file, frame_time=frame_time, step_time=step_time)
    sil_label_list = label_sil(separate_vocal_file, frame_time, step_time, vocal_vad_aggressive_level)
    formant_list[sil_label_list] = np.NaN
    analysis_data['vocal_formant_list'] = formant_list
    analysis_data['vocal_sil_label_list'] = sil_label_list
    analysis_data['vocal_power_spectrum'] = file_2_power_spectrum(separate_vocal_file, frame_time, step_time, _pre_emphasis_, window_func)
    analysis_data['vocal_pitch_list'] = get_pitch(separate_vocal_file, frame_time=frame_time, step_time=step_time)
    analysis_data['vocal_intensity_list'] = get_intensity(separate_vocal_file, frame_time=frame_time, step_time=step_time)
    analysis_data['vocal_signal'] = vocal_signal
    vocal_pdf_file = separate_vocal_file.replace(".wav", ".csv")
    if os.path.exists(vocal_pdf_file):
        vocal_phone_list = get_phones(vocal_pdf_file, frame_time, step_time)
        analysis_data['vocal_phone_list'] = vocal_phone_list
        _phone_exist_ = True

    if _do_pronunciation_filter_ and not _phone_exist_:
        logger.warning(u"The 'do_pronunciation_filter' option will not work for this music since the pdf-file for music '{}' is not existing.".format(wav_name))

    # save the analysis information into file
    with open(analysis_file, 'wb') as _file_:
        pickle.dump(analysis_data, _file_)
    logger.info(u"Analysis for music '{}' Done!".format(wav_name))


@exception_printer
def analysis_command(command_file, params):
    frame_time = params['frame_time']
    step_time = params['step_time']
    pdf_frame_time = params['pdf_frame_time']
    pdf_step_time = params['pdf_step_time']
    vocal_vad_aggressive_level = params['vocal_vad_aggressive_level']
    window_func = params['window_func']
    _pre_emphasis_ = params['pre_emphasis']
    _re_analysis_ = params['re_analysis']

    _target_folder_ = os.path.dirname(command_file)
    wav_name = os.path.splitext(os.path.basename(command_file))[0]
    transaction_file = os.path.join(_target_folder_, wav_name + ".txt")
    pdf_file = os.path.join(_target_folder_, wav_name + ".csv")
    analysis_file = os.path.join(_target_folder_, wav_name + ANALYSIS_SUFFIX)
    if os.path.exists(analysis_file):
        with open(analysis_file, 'rb') as _file_:
            script_version = pickle.load(_file_).get('script_version')
        if script_version == SCRIPT_VERSION and not _re_analysis_:
            logger.info(u"Using existing analysis file for wake_up/command wav '{}'.".format(wav_name))
            return
        else:
            logger.warning(u"The analysis file for wav '{}' is out of date, re-analysis it now.".format(wav_name))

    analysis_data = {}
    if os.path.exists(transaction_file):
        with open(transaction_file, 'r') as _transaction_file_:
            transaction = _transaction_file_.read().strip()
            transaction = check_transaction(transaction, delete_stop_words=False)
            analysis_data['transaction'] = transaction
    else:
        transaction = None
        logger.error(u"The transaction file of '{}' not exists.".format(wav_name))
        exit(-1)

    sample_rate, audio_signal = wav_read(command_file)
    frame_sample_num = int(sample_rate * frame_time)
    step_sample_num = int(sample_rate * step_time)

    # extract the formant frequency, phoneme, phoneme type and intensity, power spectrum information
    command_formants = get_formants(audio_signal=audio_signal, sample_rate=sample_rate, frame_time=frame_time, step_time=step_time)
    frame_num = command_formants.shape[0]
    command_phones = get_phones(pdf_file, frame_time, step_time, pdf_frame_time, pdf_step_time, frame_num)
    analysis_data['raw_command_phones'] = np.copy(command_phones)
    command_sil_labels = label_sil(command_file, frame_time, step_time, vocal_vad_aggressive_level)
    command_power_spectrum = file_2_power_spectrum(command_file, frame_time, step_time, _pre_emphasis_, window_func)
    command_frame_energy_list = get_frame_energy_list(command_file, frame_time, step_time, _pre_emphasis_, window_func)
    command_pitch = get_pitch(audio_signal=audio_signal, sample_rate=sample_rate, frame_time=frame_time, step_time=step_time)
    command_phones[command_sil_labels] = "sil"
    command_formants[command_sil_labels] = np.NaN

    # check the phoneme
    #   1. Cut out the sil-phoneme on both sides.
    #   2. Replace the phoneme that appears only one in a short time with its neighbor phoneme.
    command_phones, _start_index_, _end_index_, word_index_list = check_phoneme(command_phones, transaction=transaction, wav_name=wav_name)
    command_formants = command_formants[_start_index_:_end_index_]
    command_power_spectrum = command_power_spectrum[_start_index_: _end_index_]
    command_pitch = command_pitch[_start_index_: _end_index_]
    command_frame_energy_list = command_frame_energy_list[_start_index_: _end_index_]

    # clip the command signal
    signal_start = _start_index_ * step_sample_num
    signal_length = (_end_index_ - _start_index_ - 1) * step_sample_num + frame_sample_num
    signal_end = signal_start + signal_length
    if signal_end > len(audio_signal):
        audio_signal = np.concatenate([audio_signal[signal_start:], np.zeros((signal_end - len(audio_signal),))], axis=0)
    else:
        audio_signal = audio_signal[signal_start: signal_end]

    command_phone_types = phone2type(command_phones)
    # get voiceless phoneme energy ratio
    voice_energy_ave = np.nanmean(np.where(command_phone_types == PhoneType.Vowel, command_frame_energy_list, np.NaN))
    voiceless_energy_ratio_list = np.where(command_phone_types == PhoneType.Voiceless, command_frame_energy_list / (voice_energy_ave + EPS), 0.0)

    # set the attribute
    analysis_data['script_version'] = SCRIPT_VERSION
    analysis_data['word_index_list'] = word_index_list or []
    analysis_data['phone_list'] = command_phones
    analysis_data['phone_type_list'] = command_phone_types
    analysis_data['voiceless_energy_ratio_list'] = voiceless_energy_ratio_list
    analysis_data['formant_list'] = command_formants
    analysis_data['power_spectrum'] = command_power_spectrum
    analysis_data['pitch_list'] = command_pitch
    analysis_data["sample_rate"] = sample_rate
    analysis_data['command_signal'] = audio_signal

    # save the pkl file
    with open(analysis_file, 'wb') as _file_:
        pickle.dump(analysis_data, _file_)
    logger.info(u"Analysis for wake_up/command '{}' Done!".format(wav_name))


def analysis_wav(params: dict):
    """
    Analysis the wav files.
    """
    logger.info(u"[Analysis] Start analyzing the music and command.")
    max_workers = params['max_workers']
    _using_wake_up_ = params['using_wake_up']
    command_folder = params['command_folder']
    wake_up_folder = params['wake_up_folder']
    music_folder = params['music_folder']

    with futures.ProcessPoolExecutor(max_workers=max_workers) as _executor_:
        jobs = []
        # analysis the music audio file
        music_files = glob.glob(os.path.join(music_folder, "*.wav"))
        music_files = filter_irrelevant_wav(music_files)
        for music_wav_file in music_files:
            jobs.append(
                _executor_.submit(
                    analysis_music, music_wav_file, params
                )
            )
        # analysis the wake_up and command audio file
        target_folders = [command_folder] if not _using_wake_up_ else [command_folder, wake_up_folder]
        for _target_folder_ in target_folders:
            command_files = glob.glob(os.path.join(_target_folder_, '*.wav'))
            command_files = filter_irrelevant_wav(command_files)
            for command_file in command_files:
                jobs.append(
                    _executor_.submit(
                        analysis_command, command_file, params
                    )
                )
        wait_for_jobs(jobs, _executor_, "[Analysis] Analysis Audio Progress: ")
    logger.info(u"[Analysis] Analysis Done!")


def separate_audio(params: dict):
    """
    Separate human voice from music using spleeter.
    """
    max_workers = params['max_workers']
    music_folder = params['music_folder']

    # separate the human voice using spleeter
    logger.info("[Spleeter] Separate human voice from music using spleeter ...")
    separator = Separator("spleeter:2stems-16kHz")
    music_files = glob.glob(os.path.join(music_folder, '*.wav'))
    music_files = filter_irrelevant_wav(music_files)
    non_cache_music_files = []
    for music_file in music_files:
        music_name = os.path.splitext(os.path.basename(music_file))[0]
        vocal_file = music_file.replace(".wav", VOCAL_WAV_SUFFIX)
        m_sample_rate, m_signal = wav_read(music_file)
        m_duration = len(m_signal) * 1.0 / m_sample_rate
        if os.path.exists(vocal_file):  # be cached
            v_sample_rate, v_signal = wav_read(vocal_file)
            if v_sample_rate != 16000 or len(m_signal) != len(v_signal):
                logger.warn("Existing vocal file has broken for music '{}', re-separate it now.".format(music_name))
                os.remove(vocal_file)
                acc_file = music_file.replace(".wav", ACCOMPANIMENT_WAV_SUFFIX)
                if os.path.exists(acc_file):
                    os.remove(acc_file)
                analysis_file = music_file.replace(".wav", ANALYSIS_SUFFIX)
                if os.path.exists(analysis_file):
                    os.remove(analysis_file)
            else:
                logger.info("Using existing vocal file for music '{}'.".format(music_name))
                if params['redecode'] == False:
                    continue
                else:  # redecode the vocal file
                    pass

        # separator
        non_cache_music_files.append(music_file)
        if m_duration > 600:
            for _index_ in range(int(m_duration / 600. + 1)):
                separator.separate_to_file(music_file, music_folder, offset=600. * _index_, filename_format="{filename}" + "_offset{}".format(_index_) + ".{instrument}.{codec}", synchronous=False)
        else:
            separator.separate_to_file(music_file, music_folder, filename_format="{filename}.{instrument}.{codec}", synchronous=False)
    separator.join()

    jobs = []
    tfm = sox.Transformer()
    tfm.channels(n_channels=1)
    tfm.rate(samplerate=16000)
    tmp_file_list = []
    with futures.ProcessPoolExecutor(max_workers=max_workers) as _executor_:
        for music_file in non_cache_music_files:
            vocal_file = music_file.replace(".wav", VOCAL_WAV_SUFFIX)
            accomp_file = music_file.replace('.wav', ACCOMPANIMENT_WAV_SUFFIX)
            m_sample_rate, m_signal = wav_read(music_file)
            m_duration = len(m_signal) * 1.0 / m_sample_rate
            if m_duration > 600:
                v_signal = np.zeros((0,))
                a_signal = np.zeros((0,))
                v_sample_rate = None
                for _index_ in range(int(m_duration / 600. + 1)):
                    tmp_v_file = music_file.replace('.wav', "_offset{}".format(_index_) + VOCAL_WAV_SUFFIX)
                    tmp_a_file = music_file.replace(".wav", "_offset{}".format(_index_) + ACCOMPANIMENT_WAV_SUFFIX)
                    v_sample_rate, tmp_v_signal = wav_read(tmp_v_file)
                    a_sample_rate, tmp_a_signal = wav_read(tmp_a_file)
                    v_signal = np.concatenate((v_signal, tmp_v_signal), axis=0)
                    a_signal = np.concatenate((a_signal, tmp_a_signal), axis=0)
                    os.remove(tmp_v_file)
                    os.remove(tmp_a_file)
                wav_write(v_signal, vocal_file, v_sample_rate)
                wav_write(a_signal, accomp_file, a_sample_rate)
            for t_file in [vocal_file, accomp_file]:
                tmp_file_path = t_file.replace(".wav", TMP_WAV_SUFFIX)
                tmp_file_list.append(tmp_file_path)

                jobs.append(
                    _executor_.submit(
                        tfm.build, t_file, tmp_file_path
                    )
                )
        wait_for_jobs(jobs, _executor_, "Sox Transform Progress: ")
    for tmp_file in tmp_file_list:
        s_file = tmp_file.replace(TMP_WAV_SUFFIX, '.wav')
        shutil.copy(tmp_file, s_file)
        os.remove(tmp_file)

    logger.info("[Spleeter] Separate Done!")


def label_sil(wav_file: str, frame_time: float = 0.025, step_time: float = 0.01, vad_level: int = 0) -> np.ndarray:
    """
    Noted: It's better to set vad_level=3 while labeling music wave, vad_level=0 while labeling vocal wave.
    """
    with contextlib.closing(wave.open(wav_file, 'rb')) as _wave_file_:
        num_channels = _wave_file_.getnchannels()
        sample_width = _wave_file_.getsampwidth()
        sample_rate = _wave_file_.getframerate()
        assert sample_width == 2 and num_channels == 1 and sample_rate in (8000, 16000, 32000), "Wave '{}' Format Error.".format(wav_file)
        n_frames = _wave_file_.getnframes()
        pcm_data = _wave_file_.readframes(n_frames)

    vad_duration_ms = 10
    step_n = int(sample_rate * vad_duration_ms / 1000.0 * 2)
    frames = []
    raw_time_axis = []
    timestamp = vad_duration_ms / 1000.0 / 2.0
    offset = 0
    while offset + step_n < len(pcm_data):
        frames.append(pcm_data[offset: offset + step_n])
        raw_time_axis.append(timestamp)
        offset += step_n
        timestamp += vad_duration_ms / 1000.0

    vad = webrtcvad.Vad(vad_level)
    raw_sil_labels = []
    for frame in frames:
        raw_sil_labels.append(
            vad.is_speech(frame, sample_rate) is False
        )
    raw_sil_labels = np.array(raw_sil_labels, dtype=np.bool)
    raw_time_axis = np.array(raw_time_axis)

    wav_duration = n_frames * 1.0 / sample_rate
    frame_num = int((wav_duration - frame_time) / step_time) + 1
    sil_labels = np.zeros((frame_num,), dtype=np.bool)
    start_time_axis = np.arange(frame_num) * step_time
    end_time_axis = start_time_axis + frame_time
    for _index_ in range(frame_num):
        _start_time_ = start_time_axis[_index_]
        _end_time_ = end_time_axis[_index_]
        sil_labels[_index_] = np.any(
            raw_sil_labels[
                np.logical_and(
                    raw_time_axis >= _start_time_, raw_time_axis <= _end_time_
                )
            ]
        )

    return sil_labels


def filter_sil_phoneme(music_sil_label_list: np.ndarray, phone_type_list: np.ndarray, sil_ratio: int = 0.05) -> bool:
    frame_num = music_sil_label_list.shape[0]
    sil_num = int(frame_num * sil_ratio)
    unqualified_frame_num = np.sum(
        np.logical_and(
            music_sil_label_list, np.logical_or(
                phone_type_list == PhoneType.Voice,
                phone_type_list == PhoneType.Vowel
            )
        )
    )
    return unqualified_frame_num > sil_num


@deprecation.deprecated(details="Func `filter_formant_flatness` has been deprecated.")
def filter_formant_flatness_de(formant_list: np.ndarray, params: dict) -> bool:
    frame_num = formant_list.shape[0]
    flatness_window_frame_num = params['flatness_window_frame_num']
    flatness_step_frame_num = params['flatness_step_frame_num']
    flatness_threshold = params['flatness_threshold']

    if frame_num < flatness_window_frame_num:
        gmean = np.exp(np.nanmean(np.log(formant_list), axis=0))
        amean = np.nanmean(formant_list, axis=0)
        return gmean / amean

    n_frames = math.floor((frame_num - flatness_window_frame_num) * 1.0 / flatness_step_frame_num) + 1
    x_index = np.arange(flatness_window_frame_num).reshape((1, -1))
    x_index = np.tile(x_index, (n_frames, 1))
    y_index = np.arange(n_frames).reshape((-1, 1)) * flatness_step_frame_num
    y_index = np.tile(y_index, (1, flatness_window_frame_num))
    framing_index = x_index + y_index
    framed_formant_list = formant_list[framing_index]  # shape: (n_frames, frame_num, MAX_FORMANT_NUM)
    gmean = np.exp(np.nanmean(np.log(framed_formant_list), axis=1))
    amean = np.nanmean(framed_formant_list, axis=1)  # shape: (n_frames, MAX_FORMANT_NUM)
    flatness_list = np.min(gmean / amean, axis=0)  # shape: (MAX_FORMANT_NUM)
    print(gmean)
    print('.................')
    print(amean)
    print(flatness_list)
    flatness = np.min(flatness_list[1:5])
    #print('flatness:  ' + str(flatness))
    #print('flatnessthr  ' + str(flatness_threshold))
    return flatness > flatness_threshold

def ZeroCR(waveData,frameSize,overLap):
    wlen = len(waveData)
    step = frameSize - overLap
    frameNum = math.ceil(wlen/step)
    zcr = np.zeros((frameNum,1))
    for i in range(frameNum):
        curFrame = waveData[np.arange(i*step,min(i*step+frameSize,wlen))]                          
        curFrame = curFrame - np.mean(curFrame) # zero-justified
        zcr[i] = sum(curFrame[0:-1]*curFrame[1::]<=0)
        return zcr

def filter_formant_flatness(formant_list: np.ndarray, params: dict) -> bool:
    """
    filter clips by formant flatness (std.)
    """
    flatness_window_frame_num = params['flatness_step_frame_num']
    formant_window_frame_num = params['flatness_step_frame_num']
    formant_threshold = params['formant_threshold']
    flatness_step_frame_num = params['flatness_step_frame_num']
    flatness_threshold = params['flatness_threshold']
    frame_num = formant_list.shape[0]
    
    formant_begin_s = 1
    formant_begin_e = int(frame_num/2)
    #formant_middle_s = int(frame_num/4)
    #formant_middle_e = int(frame_num/4*3)
    formant_end_s = int(frame_num/2)
    formant_end_e = frame_num-1
    #print(formant_list)
    #A = formant_list[formant_begin_s : formant_begin_e, :4]
    #mean_A = np.nanmean(A, axis = 0)
    #std_A = np.nanstd(A, axis = 0)
    #print(A.shape)
    #print(mean_A)
    #print(std_A)
    minformant1_begin = np.nanmin(formant_list[formant_begin_s : formant_begin_e, :1])
    minformant1_end = np.nanmin(formant_list[formant_end_s : formant_end_e, :1])
    minformant1 = min(minformant1_begin, minformant1_end)
    #print('....................................................')
    #print(minformant1_begin)
    #print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5%%==')
    #print(minformant1_end)
    #formant_begin_mean = np.nanmean(formant_list[formant_begin_s : formant_begin_e, :4], axis = 0)
    #formant_begin_std = np.nanstd(formant_list[formant_begin_s : formant_begin_e, :4], axis = 0)
    #formant_middle_mean = np.nanmean(formant_list[formant_middle_s : formant_middle_e, :4], axis = 0)
    #formant_middle_std = np.nanstd(formant_list[formant_middle_s : formant_middle_e, :4], axis = 0)
    #formant_end_mean = np.nanmean(formant_list[formant_end_s : formant_end_e, :4], axis = 0)
    #formant_end_std = np.nanstd(formant_list[formant_end_s : formant_end_e, :4], axis = 0)
    #formant_meanvalue = [formant_begin_mean, formant_middle_mean, formant_end_mean]
    #formant_stdvalue = [formant_begin_std, formant_middle_std, formant_end_std]
    #formant_meanmin = np.nanmin(formant_meanvalue)
    #formant_stdmin = np.nanstd(formant_stdvalue)
   

    #print(formant_meanvalue)
    #print(np.nanmin(formant_meanvalue))
    #print(formant_stdvalue)
    #print(np.nanstd(formant_stdvalue))
    #if formant_stdmin < 100:
    #    #print('delete by formant_stdmin')
    #    return True
    if minformant1 > 900:
        return True
    #return False
    #''' 
    if frame_num < formant_window_frame_num:
        return False
    for _shift_index_ in range(frame_num):
        if _shift_index_ + formant_window_frame_num > frame_num:
            break
        window_formant_list = formant_list[_shift_index_: _shift_index_ + formant_window_frame_num, :4]
        #print(window_formant_list)        
        #A = np.nanstd(window_formant_list, axis=0)
        #print(A)
        #minformant1 = np.nanmin(formant_list[, :1])
        formant_flatness = np.nanmin(np.nanstd(window_formant_list, axis=0))
        #print(formant_flatness)
        #print('flatness:  ' + str(formant_flatness))
        #print('flatnessthr  ' + str(flatness_threshold))
        #if not np.isnan(formant_flatness) and formant_flatness < formant_threshold:
        if  formant_flatness < formant_threshold:
            #print('delete by formant')
            #print(formant_flatness)
            return True
    return False
    '''    
    print(flatness_window_frame_num)
    if frame_num < flatness_window_frame_num:
        gmean = np.exp(np.nanmean(np.log(formant_list), axis=0))
        amean = np.nanmean(formant_list, axis=0)
        return gmean / amean
    n_frames = math.floor((frame_num - flatness_window_frame_num) * 1.0 / flatness_step_frame_num) + 1
    x_index = np.arange(flatness_window_frame_num).reshape((1, -1))
    x_index = np.tile(x_index, (n_frames, 1))
    y_index = np.arange(n_frames).reshape((-1, 1)) * flatness_step_frame_num
    y_index = np.tile(y_index, (1, flatness_window_frame_num))
    framing_index = x_index + y_index
    framed_formant_list = formant_list[framing_index]  # shape: (n_frames, frame_num, MAX_FORMANT_NUM)
    gmean = np.exp(np.nanmean(np.log(framed_formant_list), axis=1))
    amean = np.nanmean(framed_formant_list, axis=1)  # shape: (n_frames, MAX_FORMANT_NUM)
    flatness_list = np.min(gmean / amean, axis=0)  # shape: (MAX_FORMANT_NUM)
    print(flatness_list)
    flatness = np.min(flatness_list[1:5])
    print(flatness)
    #print('flatness:  ' + str(flatness))
    #print('flatnessthr  ' + str(flatness_threshold))
    return flatness > flatness_threshold
    #if flatness < flatness_threshold
    #    return True
    '''


def filter_formant_similarity(command_phone_type_list: np.ndarray, clip_formant_list: np.ndarray, command_formant_list: np.ndarray, params: dict) -> Union[bool, float]:
    """
    filter formant similarity
    """
    formant_similarity_threshold = params['formant_similarity_threshold']
    formant_similarity_weight = params['formant_similarity_weight']
    voiceless_phone_num = np.sum(
        (command_phone_type_list == PhoneType.Voiceless).astype(np.bool) | (command_phone_type_list == PhoneType.Sil).astype(np.bool)
    )
    voice_phone_num = command_phone_type_list.shape[0] - voiceless_phone_num
    voice_rate = voice_phone_num / (voice_phone_num + voiceless_phone_num)
    voiceless_rate = voiceless_phone_num / (voice_phone_num + voiceless_phone_num)
    phone_similarity_weight = np.where(
        (command_phone_type_list == PhoneType.Voiceless).astype(np.bool) | (command_phone_type_list == PhoneType.Sil).astype(np.bool),
        voiceless_rate, voice_rate
    )
    '''
    formant_distance = np.mean(
        np.nansum(
            np.fabs(
                (command_formant_list - clip_formant_list) / (command_formant_list + clip_formant_list + EPS) / np.expand_dims(phone_similarity_weight, axis=1)
            ) * formant_similarity_weight,
            axis=-1
        )
    )/sum(formant_similarity_weight)
    '''
    formant_distance = np.mean(
        np.nansum(
            np.fabs(
                np.fabs(command_formant_list - clip_formant_list)/ (np.fabs(command_formant_list + clip_formant_list + EPS))
            ) * formant_similarity_weight,
            axis=-1
                  )
    )/sum(formant_similarity_weight)

    #print(sum(formant_similarity_weight))
    
    #print(formant_similarity_weight)
    print(formant_distance)
    return formant_distance > formant_similarity_threshold, formant_distance


def filter_pitch(pitch_list: np.ndarray, params: dict, filter_nan_pitch: bool = False) -> bool:
    pitch_window_frame_num = params['pitch_window_frame_num']
    pitch_threshold = params['pitch_threshold']
    frame_num = pitch_list.shape[0]
    '''
    pitch_begin_s = 1
    pitch_begin_e = int(frame_num/2)
    pitch_middle_s = int(frame_num/4)
    pitch_middle_e = int(frame_num/4*3)
    pitch_end_s = int(frame_num/2)
    pitch_end_e = frame_num-1
    
    pitch_begin_mean = np.nanmean(pitch_list[pitch_begin_s : pitch_begin_e])
    pitch_begin_std = np.nanstd(pitch_list[pitch_begin_s : pitch_begin_e])
    pitch_middle_mean = np.nanmean(pitch_list[pitch_middle_s : pitch_middle_e])
    pitch_middle_std = np.nanstd(pitch_list[pitch_middle_s : pitch_middle_e])
    pitch_end_mean = np.nanmean(pitch_list[pitch_end_s : pitch_end_e])
    pitch_end_std = np.nanstd(pitch_list[pitch_end_s : pitch_end_e])
    pitch_meanvalue = [pitch_begin_mean, pitch_middle_mean, pitch_end_mean]
    pitch_stdvalue = [pitch_begin_std, pitch_middle_std, pitch_end_std]
    pitch_meanmin = np.min(pitch_meanvalue)
    pitch_stdmin = np.min(pitch_stdvalue)
    '''
    #print(pitch_meanmin)
    #print(pitch_stdmin)
    
    #if pitch_stdmin < 0.01:
    #    print('.....................')
    #    print(pitch_stdmin)
    #    print('delete by pitch_stdmin')
    #    return True
    
    #''
    if frame_num < pitch_window_frame_num:
        pitch_window_frame_num = frame_num
    #if filter_nan_pitch and np.all(pitch_list == np.NaN):
    #    return True
    for _shift_index_ in range(frame_num):
        if _shift_index_ + pitch_window_frame_num > frame_num:
            break
        window_pitch_list = pitch_list[_shift_index_: _shift_index_ + pitch_window_frame_num]
        #if np.sum(np.isnan(window_pitch_list)) > (frame_num * 0.5):
        #    continue
        #print(window_pitch_list)
        if np.sum(~np.isnan(window_pitch_list)) > pitch_window_frame_num * 0.7:
            pitch_std = np.nanstd(window_pitch_list)
            #print('***************************************')
            #print(pitch_std)
            #if not np.isnan(pitch_std) and pitch_std < pitch_threshold:
            if pitch_std < pitch_threshold:
            #if pitch_std < 1: 
                #print('delete by pitch')
                #print(pitch_std)
                #print(window_pitch_list)
                #print(pitch_window_frame_num)
                #print(frame_num)
                return True
    #'''
    return False


def filter_intensity(intensity_list: np.ndarray, params: dict, filter_nan_intensity: bool = False) -> bool:
    """
    filter intensity
    """
    intensity_window_frame_num = params['intensity_window_frame_num']
    intensity_scale_threshold = params['intensity_scale_threshold']
    intensity_std_threshold = params['intensity_std_threshold']
    frame_num = intensity_list.shape[0]
    
    
    '''
    #print(frame_num)
    inten_begin_s = 1
    inten_begin_e = int(frame_num/2)
    inten_middle_s = int(frame_num/4)
    inten_middle_e = int(frame_num/4*3)
    inten_end_s = int(frame_num/2)
    inten_end_e = frame_num-1

    inten_begin_mean = np.nanmean(intensity_list[inten_begin_s : inten_begin_e])
    inten_begin_std = np.nanstd(intensity_list[inten_begin_s : inten_begin_e])
    inten_middle_mean = np.nanmean(intensity_list[inten_middle_s : inten_middle_e])
    inten_middle_std = np.nanstd(intensity_list[inten_middle_s : inten_middle_e])
    inten_end_mean = np.nanmean(intensity_list[inten_end_s : inten_end_e])
    inten_end_std = np.nanstd(intensity_list[inten_end_s : inten_end_e])
    inten_meanvalue = [inten_begin_mean, inten_middle_mean, inten_end_mean]
    inten_stdvalue = [inten_begin_std, inten_middle_std, inten_end_std]
    inten_meanmin = np.min(inten_meanvalue)
    inten_stdmin = np.min(inten_stdvalue)
    
    #print(inten_meanmin)
    #print(inten_stdmin)
     
    #if inten_meanmin < 60:
    #    #print('delete by inten_meanmin')
    #    return True
    #if inten_stdmin < 0.8:
    #    #print('delete by inten_stdmin')
    #    return True

    #if frame_num < intensity_window_frame_num:
    #    intensity_window_frame_num = frame_num
    '''
    #if filter_nan_intensity and np.all(intensity_list == np.NaN):
    #    return True
    
    for _shift_index_ in range(frame_num):
        if _shift_index_ + intensity_window_frame_num > frame_num:
            break
        window_intensity_list = intensity_list[_shift_index_: _shift_index_ + intensity_window_frame_num]
        #print(np.nanmean(window_intensity_list))
        if np.nanmean(window_intensity_list) < intensity_scale_threshold:
            #print(np.nanmean(window_intensity_list))
            #print('delete by intensity_scale_threshold')
            #print(intensity_scale_threshold)
            return True
        intensity_std = np.nanstd(window_intensity_list)
        #if not np.isnan(intensity_std) and intensity_std < intensity_std_threshold:
        if intensity_std < intensity_std_threshold:
            #print('delete by inten_std')
            #print(intensity_std)
            return True
    return False


def moving_average_formant_list(formant_list: np.ndarray, window_len: int = 5) -> np.ndarray:
    return np.array(
        [np.convolve(a, np.ones((window_len,), dtype=np.float) / window_len, mode="valid") for a in formant_list.T]
    ).T


@exception_printer
def get_clip_command_info(clip_signal: np.ndarray, clip_music_formant_list: np.ndarray, clip_vocal_formant_list: Optional[np.ndarray], clip_music_intensity_list: np.ndarray, clip_vocal_intensity_list: Optional[np.ndarray], clip_music_pitch_list: np.ndarray, clip_vocal_pitch_list: Optional[np.ndarray], clip_music_sil_label_list: np.ndarray, clip_music_phone_list: Optional[np.ndarray], clip_vocal_sil_label_list: Optional[np.ndarray], clip_vocal_phone_list: Optional[np.ndarray], command_analysis_file: str, params: dict) -> Union[str, dict]:
    # load params
    frame_time = params['frame_time']
    step_time = params['step_time']
    allowed_sil_phone_ratio = params['allowed_sil_phone_ratio']
    psychological_threshold = params['psychological_threshold']
    pronunciation_match_len = params['pronunciation_match_len']
    _do_psychological_filter_ = params['do_psychological_filter']
    _using_spleeter_ = params['using_spleeter']
    _do_sil_phoneme_filter_ = params['do_sil_phoneme_filter']
    _do_intensity_filter_ = params['do_intensity_filter']
    _do_formant_similarity_filter_ = params['do_formant_similarity_filter']
    _do_pitch_filter_ = params['do_pitch_filter']
    _do_formant_filter_ = params['do_formant_filter']

    # load the analysis data
    with open(command_analysis_file, 'rb') as _file_:
        analysis_data = pickle.load(_file_)
    command_formant_list = analysis_data['formant_list']
    command_phone_type_list = analysis_data['phone_type_list']
    sample_rate = analysis_data['sample_rate']
    command_phone_list = analysis_data["phone_list"]

    # initial variables
    frame_sample_num = int(frame_time * sample_rate)
    step_sample_num = int(step_time * sample_rate)


    # filter by sil-phoneme
    if _do_sil_phoneme_filter_:
        _filter_sil_ = filter_sil_phoneme(clip_music_sil_label_list, command_phone_type_list, allowed_sil_phone_ratio)
        if _using_spleeter_ and not _filter_sil_:
            _filter_sil_ = filter_sil_phoneme(clip_vocal_sil_label_list, command_phone_type_list, allowed_sil_phone_ratio)
        if _filter_sil_:
            return "filter_sil"

    # filter by intensity
    if _do_intensity_filter_:
        if filter_intensity(clip_music_intensity_list, params, True):
            return "filter_intensity"
        #if _using_spleeter_ and filter_intensity(clip_music_intensity_list, params) and filter_intensity(clip_vocal_intensity_list, params, True):
        #    return "filter_intensity"
        #if not _using_spleeter_ and filter_intensity(clip_music_intensity_list, params, True):
        #    return "filter_intensity"

    # filter by pitch
    if _do_pitch_filter_:
        if filter_pitch(clip_music_pitch_list, params, True):
            return "filter_pitch"
        #if _using_spleeter_ and filter_pitch(clip_music_pitch_list, params) and filter_pitch(clip_vocal_pitch_list, params, True):
        #    return "filter_pitch"
        #if not _using_spleeter_ and filter_pitch(clip_music_pitch_list, params, True):
        #    return "filter_pitch"
        
    # filter by formant:
    if _do_formant_filter_:
        _filter_formant_ = filter_formant_flatness(clip_music_formant_list, params)
        if _using_spleeter_ and not _filter_formant_:
            _filter_formant_ = filter_formant_flatness(clip_vocal_formant_list, params)
        if _filter_formant_:
            return "filter_formant"
    
    # calculate the formant shift
    formant_shift_list = align_formant(clip_music_formant_list, command_formant_list, command_phone_type_list, params)

    # get adversarial signal and balance coe distance
    filtered_signal, balance_coe = get_filtered_signal(clip_signal, analysis_data, formant_shift_list, params)
    balance_distance = np.sqrt(np.sum((1. - balance_coe) ** 2))

    # filter by formant similarity
    # todo: 考虑2阶导对结果的影响
    formant_distance = -1
    if _do_formant_similarity_filter_:
        _filter_formant_similarity_, formant_distance = filter_formant_similarity(command_phone_type_list, clip_music_formant_list, command_formant_list, params)
        if _using_spleeter_ and not _filter_formant_similarity_:
            _filter_formant_similarity_, vocal_formant_distance = filter_formant_similarity(command_phone_type_list, clip_vocal_formant_list, command_formant_list, params)
            formant_distance = max(formant_distance, vocal_formant_distance)
        if _filter_formant_similarity_:
            #print('delete by simi')
            #print(formant_distance)
            return "filter_formant_similarity"
    '''
    # filter by psychological masking loudness
    music_masking_list, psd_max = generate_th(clip_signal, sample_rate, frame_sample_num, step_sample_num)
    k_step = np.arange(0, 10, 0.5) + 0.5
    delta_k_final = None
    delta_loudness = None
    m_num, n_num = music_masking_list.shape
    for k in k_step:
        delta_k_filtered_signal = k * filtered_signal
        delta_k_filtered_psd = compute_PSD_matrix(delta_k_filtered_signal, frame_sample_num, step_sample_num, psd_max)[0].T
        delta_k_spectrum_over_mask = delta_k_filtered_psd - 10 * np.log10(music_masking_list + EPS)
        delta_k_over_th = np.maximum(delta_k_spectrum_over_mask, 0)
        delta_k_over_num = np.count_nonzero(delta_k_over_th)
        delta_k_over_th_rate = delta_k_over_num / (m_num * n_num)
        
        delta_loudness = np.mean(delta_k_over_th)
        print(delta_loudness)
        if _do_psychological_filter_:
            if delta_k_over_th_rate > 0.3 or delta_loudness > psychological_threshold:
                if k == 0.5:
                    return "filter_psychological"
                else:
                    break
            else:
                delta_k_final = k
    '''


    # filter by psychological masking loudness
    music_masking_list, psd_max = generate_th(clip_signal, sample_rate, frame_sample_num, step_sample_num)
    begin_s = 1
    begin_e = int(len(clip_signal)/3)
    middle_s = int(len(clip_signal)/3)
    middle_e = int(len(clip_signal)*2/3)
    end_s = int(len(clip_signal)*2/3)
    end_e = int(len(clip_signal)-1)
    #print(length_clip_signal)
    #print('length................')
    timesample_begin = clip_signal[begin_s : begin_e]/32767
    timesample_middle = clip_signal[middle_s : middle_e]/32767
    timesample_end = clip_signal[end_s : end_e]/32767

    time_begin_mean = np.nanmean(np.abs(timesample_begin))
    time_begin_std = np.nanstd(np.abs(timesample_begin)) 
    time_middle_mean = np.nanmean(np.abs(timesample_middle))
    time_middle_std = np.nanstd(np.abs(timesample_middle))
    time_end_mean = np.nanmean(np.abs(timesample_end))
    time_end_std = np.nanstd(np.abs(timesample_end))
    
    meanvalue = [time_begin_mean, time_middle_mean, time_end_mean]
    stdvalue = [time_begin_std, time_middle_std, time_end_std]
    timemeanmin = np.min(meanvalue)
    timestdmin = np.min(stdvalue)
    
    time_beginZeroCR = ZeroCR(timesample_begin,256,0)
    time_middleZeroCR = ZeroCR(timesample_middle,256,0)
    time_endZeroCR = ZeroCR(timesample_end,256,0)
    time_beginZeroCR_mean = np.mean(time_beginZeroCR)
    time_middleZeroCR_mean = np.mean(time_middleZeroCR)
    time_endZeroCR_mean = np.mean(time_endZeroCR)
    time_ZeroCR_min = min(time_beginZeroCR_mean, time_middleZeroCR_mean, time_endZeroCR_mean)

    k_step = np.arange(0, 10, 0.5) + 0.5
    delta_k_final = None
    delta_loudness = None
    m_num, n_num = music_masking_list.shape
    k = 4
    delta_k_filtered_signal = k * filtered_signal
    delta_k_filtered_psd = compute_PSD_matrix(delta_k_filtered_signal, frame_sample_num, step_sample_num, psd_max)[0].T
    filter_clip_PSD = compute_PSD_matrix(delta_k_filtered_signal, frame_sample_num, step_sample_num, psd_max)[0]
    
    clip_PSD = compute_PSD_matrix(clip_signal, frame_sample_num, step_sample_num, psd_max)[0]
    clip_sumenergy_10 = np.sum(clip_PSD[1:10], axis=0)
    clip_sumenergy_20 = np.sum(clip_PSD[1:20], axis=0)
    clip_sumenergy_50 = np.sum(clip_PSD[1:50], axis=0)
    clip_sumenergy_1_100 = np.sum(clip_PSD[1:100], axis=0)
    clip_sumenergy_101_200 = np.sum(clip_PSD[101:200], axis=0)
    clip_sumenergy = np.sum(clip_PSD[1:200], axis=0)
    delta_clip_low_high = np.mean(clip_sumenergy_1_100 - clip_sumenergy_101_200)
    sumenergymean_10_rate = np.nanmean(clip_sumenergy_10/clip_sumenergy)
    sumenergymean_20_rate = np.nanmean(clip_sumenergy_20/clip_sumenergy)
    sumenergymean_50_rate = np.nanmean(clip_sumenergy_50/clip_sumenergy)
    #print(sumenergymean_10_rate)
    #print(sumenergymean_20_rate)
    #print(sumenergymean_50_rate)

    delta_k_spectrum_over_mask = delta_k_filtered_psd - 10 * np.log10(music_masking_list + EPS)
    delta_k_over_th = np.maximum(delta_k_spectrum_over_mask, 0)
    delta_k_over_num = np.count_nonzero(delta_k_over_th)
    delta_k_over_th_rate = delta_k_over_num / (m_num * n_num) 
    delta_loudness = np.mean(delta_k_over_th)
    
    if _do_psychological_filter_:
        if time_ZeroCR_min < 0.5:
            #print(time_ZeroCR_min)
            #print('deleted by ZeroCR')
            return "filter_psychological"
        if timemeanmin < 0.05 and timestdmin < 0.1:
            #print('delete by timemeanmin')
            return "filter_psychological"
        if delta_clip_low_high < 0:
            #print('delete by delta_clip_low_high')
            return "filter_psychological"
        if sumenergymean_10_rate < 0.05 or sumenergymean_20_rate < 0.1 or sumenergymean_50_rate < 0.2:
            #print('delete by sumenergymean_10_20_50')
            return "filter_psychological"
        if delta_k_over_th_rate > 0.4 or delta_loudness > psychological_threshold:
            #print('delete by psy')
            return "filter_psychological"

    # align the phoneme
    pron_align_list = []
    pron_align_num = 0
    if clip_music_phone_list is not None:
        pron_align_list, pron_align_num = align_phoneme(clip_music_phone_list, command_phone_list, pronunciation_match_len)
    if clip_vocal_phone_list is not None:
        tmp_pron_align_list, tmp_pron_align_num = align_phoneme(clip_vocal_phone_list, command_phone_list, pronunciation_match_len)
        if len(tmp_pron_align_list) > len(pron_align_list) or (len(tmp_pron_align_list) == len(pron_align_list) and tmp_pron_align_num > pron_align_num):
            pron_align_list = tmp_pron_align_list
            pron_align_num = tmp_pron_align_num
    if clip_vocal_phone_list is None and clip_music_phone_list is None:
        pron_align_list = None

    return {
        "formant_shift_list": formant_shift_list,
        "balance_distance": balance_distance,
        "delta_loudness": delta_loudness,
        "delta_k_final": delta_k_final,
        "formant_distance": formant_distance,
        "pron_align_list": pron_align_list,
        "pron_align_num": pron_align_num
    }


@exception_printer
def get_clip_info(wake_up_info: Optional[dict], command_info: dict, params: dict) -> Union[str, dict]:
    # load param
    # todo: 添加对margin部分的筛选
    _using_wake_up_ = params['using_wake_up']
    wake_up_weight = params['wake_up_weight'] if _using_wake_up_ else 0
    command_weight = 1 - wake_up_weight
    matched_phone_num_threshold = params['matched_phone_num_threshold']
    _do_pronunciation_filter_ = params['do_pronunciation_filter']

    assert 0 <= wake_up_weight <= 1, "Value 'wake_up_weight' Error."

    # load info
    wake_up_delta_loudness = wake_up_info['delta_loudness'] if _using_wake_up_ else 0
    command_delta_loudness = command_info['delta_loudness']
    wake_up_balance_distance = wake_up_info['balance_distance'] if _using_wake_up_ else 0
    command_balance_distance = command_info['balance_distance']
    wake_up_formant_distance = wake_up_info['formant_distance'] if _using_wake_up_ else 0
    command_formant_distance = command_info['formant_distance']
    wake_up_pron_align_list = wake_up_info['pron_align_list'] if _using_wake_up_ else []
    command_pron_align_list = command_info['pron_align_list']
    wake_up_pron_align_num = wake_up_info['pron_align_num'] if _using_wake_up_ else 0
    command_pron_align_num = command_info['pron_align_num']

    # filter by pronunciation
    if _do_pronunciation_filter_ and wake_up_pron_align_list is not None and command_pron_align_list is not None:
        if len(wake_up_pron_align_list) + len(command_pron_align_list) < matched_phone_num_threshold:
            return "filter_pronunciation"

    # calculate delta loudness
    delta_loudness = wake_up_delta_loudness * wake_up_weight + command_delta_loudness * command_weight

    # calculate balance distance
    balance_distance = wake_up_weight * wake_up_balance_distance + command_weight * command_balance_distance

    # calculate formant distance
    formant_distance = wake_up_weight * wake_up_formant_distance + command_weight * command_formant_distance

    return {
        "formant_distance": formant_distance,
        "delta_loudness": delta_loudness,
        "balance_distance": balance_distance,
        "matched_pron_num": wake_up_pron_align_num + command_pron_align_num
    }


def pick_command(command_analysis_file: str, music_analysis_file: str, params: dict):
    task_name = params['task_name']
    frame_time = params['frame_time']
    step_time = params['step_time']
    padding_frame_num = params['padding_frame_num']
    margin_frame_num = params['margin_frame_num']
    step_frame_num = params['step_frame_num']
    top_k = params['pick_top_k']
    formant_distance_weight = params['formant_distance_weight']
    psychological_weight = params['psychological_weight']
    balance_weight = params['balance_weight']
    max_workers = params['max_workers']
    pronunciation_weight = params['pronunciation_weight']
    ctc_reserved_ratio = params['ctc_reserved_ratio']
    _allow_overlap_ = params['allow_overlap']
    _using_spleeter_ = params['using_spleeter']
    _using_ctc_score_ = params['using_ctc_score']

    # resolve the path
    command_name = os.path.splitext(os.path.basename(command_analysis_file))[0]
    music_name = os.path.splitext(os.path.basename(music_analysis_file))[0]
    pick_folder = os.path.join(get_pick_folder(task_name), command_name, music_name)
    logger.info(u"Start picking, command name: '{}', music name: '{}'.".format(command_name, music_name))
    os.makedirs(pick_folder, exist_ok=True)

    # load the music analysis file
    with open(music_analysis_file, 'rb') as _file_:
        music_analysis_data = pickle.load(_file_)
    sample_rate = music_analysis_data['sample_rate']
    music_signal = music_analysis_data['music_signal']
    music_formant_list = music_analysis_data['formant_list']
    music_sil_label_list = music_analysis_data['sil_label_list']
    music_phone_list = music_analysis_data.get('phone_list')
    music_pitch_list = music_analysis_data['pitch_list']  # type: np.ndarray
    music_intensity_list = music_analysis_data['intensity_list']
    vocal_intensity_list = music_analysis_data.get("vocal_intensity_list")
    vocal_formant_list = music_analysis_data.get('vocal_formant_list')  # type: Union[None, np.ndarray]
    vocal_sil_label_list = music_analysis_data.get('vocal_sil_label_list')  # type: Union[None, np.ndarray]
    vocal_pitch_list = music_analysis_data.get("vocal_pitch_list")  # type: Union[None, np.ndarray]
    vocal_phone_list = music_analysis_data.get("vocal_phone_list")
    music_frame_num = music_formant_list.shape[0]

    # load the command analysis file
    with open(command_analysis_file, 'rb') as _file_:
        command_analysis_data = pickle.load(_file_)
        command_frame_num = command_analysis_data['formant_list'].shape[0]
        command_transaction = str(command_analysis_data['transaction'])

    frame_sample_num = int(frame_time * sample_rate)
    step_sample_num = int(step_time * sample_rate)
    t_frame_num = command_frame_num + 2 * margin_frame_num
    available_num = int((music_frame_num - t_frame_num - 2. * padding_frame_num) / step_frame_num) + 1
    command_sample_num = (command_frame_num - 1) * step_sample_num + frame_sample_num

    clip_info_dict = {}  # 记录全部信息
    delete_info_dict = collections.defaultdict(set)  # 记录删除信息
    index_list = []  # 用于记录音乐片段的位置
    formant_distance_list = []  # 用于记录共振峰距离
    formant_shift_matrix = []  # 用于记录共振峰偏移
    delta_loudness_list = []  # 用于记录声学掩蔽
    balance_distance_list = []  # 用于记录能量平衡系数距离
    delta_k_final_list = []  # 用于记录delta_k_final
    pron_num_list = []  # 用于记录匹配的音素长度

    # calculate the ctc loss
    if _using_ctc_score_:
        clip_signals = []
        available_index_list = []
        clip_formant_list = []
        for _available_index_ in range(available_num):
            # load index
            clip_s_index = _available_index_ * step_frame_num + padding_frame_num
            command_s_index = clip_s_index + margin_frame_num
            command_e_index = command_s_index + command_frame_num
            command_sample_s_index = command_s_index * step_sample_num
            clip_music_formant_list = music_formant_list[command_s_index: command_e_index]
            clip_signal = music_signal[command_sample_s_index: command_sample_s_index + command_sample_num]
            clip_signals.append(clip_signal)
            available_index_list.append(_available_index_)
            clip_formant_list.append(clip_music_formant_list)
        result = deepspeech_ctc_score(clip_signals, available_index_list, command_transaction, clip_formant_list, command_analysis_file, params)
        ctc_reserved_num = int(min(math.ceil(available_num * ctc_reserved_ratio), available_num))
        ctc_score_sort_index = np.argsort(result)  # ctc score 越小越好
        sorted_available_index_list = np.array(available_index_list)[ctc_score_sort_index]
        reserved_available_index_list = sorted_available_index_list[:ctc_reserved_num]
        deleted_available_index_list = sorted_available_index_list[ctc_reserved_num:]
        for _index_ in deleted_available_index_list:
            clip_s_index = _index_ * step_frame_num + padding_frame_num
            delete_info_dict["CTC_Score_Delete"].add(clip_s_index)
        with open(os.path.join(pick_folder, "./tmp-{}.pkl".format(music_name)), 'wb') as _file_:
            pickle.dump(result, _file_)
    else:
        reserved_available_index_list = np.arange(available_num)
        ctc_reserved_num = available_num

    # calculate and cache the command related info
    with futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        jobs = []
        for _available_index_ in reserved_available_index_list:
            # load index
            command_s_index = _available_index_ * step_frame_num + padding_frame_num + margin_frame_num
            command_e_index = command_s_index + command_frame_num
            command_sample_s_index = command_s_index * step_sample_num
            clip_signal = music_signal[command_sample_s_index: command_sample_s_index + command_sample_num]

            # load clip command info
            clip_music_formant_list = music_formant_list[command_s_index: command_e_index]
            clip_vocal_formant_list = vocal_formant_list[command_s_index: command_e_index]
            clip_music_intensity_list = music_intensity_list[command_s_index: command_e_index]
            clip_vocal_intensity_list = vocal_intensity_list[command_s_index: command_e_index]
            clip_music_pitch_list = music_pitch_list[command_s_index: command_e_index]
            clip_vocal_pitch_list = vocal_pitch_list[command_s_index: command_e_index]
            clip_music_sil_label_list = music_sil_label_list[command_s_index: command_e_index]
            clip_music_phone_list = music_phone_list[command_s_index: command_e_index] if music_phone_list is not None else None
            clip_vocal_sil_label_list = vocal_sil_label_list[command_s_index: command_e_index] if _using_spleeter_ else None
            clip_vocal_phone_list = vocal_phone_list[command_s_index: command_e_index] if _using_spleeter_ and vocal_phone_list is not None else None

            jobs.append(
                executor.submit(
                    get_clip_command_info, clip_signal, clip_music_formant_list, clip_vocal_formant_list, clip_music_intensity_list, clip_vocal_intensity_list, clip_music_pitch_list, clip_vocal_pitch_list, clip_music_sil_label_list, clip_music_phone_list, clip_vocal_sil_label_list, clip_vocal_phone_list, command_analysis_file, params
                )
            )
        command_info_list = wait_for_jobs(jobs, executor, "[Pick] Analysis Command Clips: ")

    # process and cache the clip related info
    with futures.ProcessPoolExecutor(max_workers) as executor:
        jobs = []
        tmp_index_list = []
        for _index_ in range(ctc_reserved_num):
            # load the index
            _available_index_ = reserved_available_index_list[_index_]
            clip_s_index = _available_index_ * step_frame_num + padding_frame_num

            # load the cached info
            command_info = command_info_list[_index_]
            if isinstance(command_info, str):
                delete_info_dict[command_info].add(clip_s_index)
                continue

            jobs.append(
                executor.submit(
                    get_clip_info, None, command_info, params
                )
            )
            tmp_index_list.append(_index_)
        clip_info_list = wait_for_jobs(jobs, executor, "[PICK] Analysis Clips: ")
    for _index_ in range(len(clip_info_list)):
        clip_info = clip_info_list[_index_]
        command_info = command_info_list[tmp_index_list[_index_]]
        _available_index_ = reserved_available_index_list[tmp_index_list[_index_]]
        clip_s_index = _available_index_ * step_frame_num + padding_frame_num
        if isinstance(clip_info, str):
            delete_info_dict[clip_info].add(clip_s_index)
            continue
        index_list.append(clip_s_index)
        formant_distance_list.append(clip_info['formant_distance'])
        delta_loudness_list.append(clip_info['delta_loudness'])
        formant_shift_matrix.append(command_info['formant_shift_list'])
        balance_distance_list.append(clip_info['balance_distance'])
        delta_k_final_list.append(command_info['delta_k_final'])
        pron_num_list.append(clip_info['matched_pron_num'])

        # cache the information
        clip_info_dict[str(clip_s_index)] = {
            "shift_index": clip_s_index,
            "formant_distance": clip_info['formant_distance'],
            "delta_loudness": clip_info['delta_loudness'],
            "delta_k_final": command_info['delta_k_final'],
            "balance_distance": clip_info['balance_distance'],
            "command_matched_phone": command_info["pron_align_list"],
            "matched_pron_num": clip_info['matched_pron_num'],
        }

    # save the information
    delete_info_file = os.path.join(pick_folder, "delete_info.json")
    save_json_data(delete_info_dict, delete_info_file)
    if not len(index_list):
        logger.error("No music clip is reserved. Please check the filter conditions specified in params. Command: '{}', Music: '{}'.".format(command_name, music_name))
        return
    else:
        logger.info("Retains {} clips. {}. Command: '{}', Music: '{}'.".format(len(index_list), ", ".join(["'{}' {} clips".format(_k, len(_v)) for _k, _v in delete_info_dict.items()]), command_name, music_name))
    clip_info_file = os.path.join(pick_folder, "clip_info.json")
    save_json_data(clip_info_dict, clip_info_file)

    index_list = np.array(index_list)  # shape:(available_num, )
    formant_distance_list = feature_normalize(formant_distance_list)
    delta_loudness_list = feature_normalize(delta_loudness_list)
    balance_distance_list = feature_normalize(balance_distance_list)
    pron_num_list = feature_normalize(pron_num_list)
    formant_shift_matrix = np.array(formant_shift_matrix)
    delta_k_final_list = np.array(delta_k_final_list)

    # calculate distance
    distance_list = formant_distance_weight * formant_distance_list + psychological_weight * delta_loudness_list + balance_weight * balance_distance_list - pronunciation_weight * pron_num_list

    # sort the distance
    if top_k < 0:
        top_k = distance_list.shape[0]
    else:
        top_k = min(top_k, distance_list.shape[0])
    sort_index_list = np.argsort(distance_list)

    # save the sorted information
    sort_json_file = os.path.join(pick_folder, "total_sort_info.json")
    sort_information = {}
    for _sort_index_ in range(sort_index_list.shape[0]):
        sort_index = sort_index_list[_sort_index_]
        sort_information[_sort_index_] = {
            "clip_name": str(index_list[sort_index]),
            "formant_distance": formant_distance_list[sort_index],
            "delta_loudness": delta_loudness_list[sort_index],
            "balance_distance": balance_distance_list[sort_index],
            "pron_num": pron_num_list[sort_index],
            "distance": distance_list[sort_index]
        }
    save_json_data(sort_information, sort_json_file)

    # 消除重叠的片段
    if not _allow_overlap_:
        tmp_index_list = []
        tmp_sort_index_list = []

        for _sort_index_ in sort_index_list:
            shift_index = index_list[_sort_index_]
            _overlap_ = False
            for tmp_index in tmp_index_list:
                if abs(tmp_index - shift_index) < t_frame_num:
                    _overlap_ = True
                    break
            if not _overlap_:
                tmp_index_list.append(shift_index)
                tmp_sort_index_list.append(_sort_index_)
                if len(tmp_index_list) == top_k:
                    break
        index_list = np.array(tmp_index_list)
        tmp_sort_index_list = np.array(tmp_sort_index_list)
        formant_shift_matrix = formant_shift_matrix[tmp_sort_index_list]
        delta_k_final_list = delta_k_final_list[tmp_sort_index_list]
    else:
        index_list = index_list[sort_index_list[:top_k]]
        formant_shift_matrix = formant_shift_matrix[sort_index_list[:top_k]]
        delta_k_final_list = delta_k_final_list[sort_index_list[:top_k]]

    # 保存结果
    for _index_ in range(len(index_list)):
        clip_name = "{}-{}-{}".format(command_name, music_name, index_list[_index_])
        # 保存clip音频
        clip_sample_index = index_list[_index_] * step_sample_num
        clip_sample_num = (t_frame_num - 1) * step_sample_num + frame_sample_num
        music_clip_signal = music_signal[clip_sample_index: clip_sample_index + clip_sample_num]
        output_clip_path = os.path.join(pick_folder, "{}.wav".format(clip_name))
        wav_write(music_clip_signal, output_clip_path, sample_rate)

        # 保存共振峰偏移
        pick_npz_file = os.path.join(pick_folder, clip_name + PICK_NPZ_SUFFIX)
        np.savez(
            pick_npz_file,
            formant_shift_list=formant_shift_matrix[_index_],
            frame_num=t_frame_num,
            margin_frame_num=margin_frame_num,
            command_frame_num=command_frame_num,
            command_delta_k_final=delta_k_final_list[_index_]
        )


@exception_printer
def random_pick_command(command_analysis_file, music_analysis_file: str, params: dict, r_pick_number: int = None):
    logger.info("Start random pick. Command analysis file: `{}`, music analysis file: `{}`, random pick number: `{}`.".format(command_analysis_file, music_analysis_file, r_pick_number))
    # load the params
    task_name = params['task_name']
    command_name = os.path.splitext(os.path.basename(command_analysis_file))[0]
    music_name = os.path.splitext(os.path.basename(music_analysis_file))[0]
    frame_time = params['frame_time']
    step_time = params['step_time']
    padding_frame_num = params['padding_frame_num']
    margin_frame_num = params['margin_frame_num']
    step_frame_num = params['step_frame_num']
    _allow_overlap_ = params['allow_overlap']
    output_folder = path_join(get_random_pick_folder(task_name), command_name, music_name)
    os.makedirs(output_folder)
    pick_folder = path_join(get_pick_folder(task_name), command_name, music_name)
    pick_k = r_pick_number or len(glob.glob(os.path.join(pick_folder, "*.wav")))

    if pick_k == 0:
        return

    # load music analysis file
    with open(music_analysis_file, 'rb') as _file_:
        music_analysis_data = pickle.load(_file_)
    sample_rate = music_analysis_data['sample_rate']
    music_signal = music_analysis_data['music_signal']
    music_formant_list = music_analysis_data['formant_list']
    music_frame_num = music_formant_list.shape[0]

    # load command analysis file
    with open(command_analysis_file, 'rb') as _file_:
        command_analysis_data = pickle.load(_file_)
    command_formant_list = command_analysis_data['formant_list']
    command_phone_type_list = command_analysis_data['phone_type_list']
    command_frame_num = command_formant_list.shape[0]

    # random pick the music clips
    frame_sample_num = int(frame_time * sample_rate)
    step_sample_num = int(step_time * sample_rate)
    t_frame_num = command_frame_num + 2 * margin_frame_num
    available_index = np.arange(padding_frame_num, music_frame_num - padding_frame_num - t_frame_num, step_frame_num)

    if _allow_overlap_:
    #if True:
        pick_index_list = np.random.choice(available_index, size=pick_k, replace=True)
    else:
        pick_index_list = []
        hpw_cnt = 1
        while len(pick_index_list) < pick_k: #and hpw_cnt < 3000:
            hpw_cnt += 1
            #print('33333333333333333')
            _pick_index_ = int(np.random.choice(available_index, 1)[0])
            if _pick_index_ not in pick_index_list:
                _overlap_ = False
                for _index_ in pick_index_list:
                    #print('44444444444444444444')
                    if abs(_index_ - _pick_index_) <= t_frame_num:
                        _overlap_ = True
                        break
                if not _overlap_:
                    pick_index_list.append(_pick_index_)

    # save the clips
    clip_sample_num = (t_frame_num - 1) * step_sample_num + frame_sample_num

    for shift_index in pick_index_list:
        # save the clip signal
        clip_name = "{}-{}-{}".format(command_name, music_name, shift_index)
        start_sample_index = shift_index * step_sample_num
        music_clip_signal = music_signal[start_sample_index: start_sample_index + clip_sample_num]
        output_clip_path = os.path.join(output_folder, "{}.wav".format(clip_name))
        wav_write(music_clip_signal, output_clip_path, sample_rate)

        # save the formant shift file
        command_start_index = shift_index + margin_frame_num
        command_end_index = command_start_index + command_frame_num
        clip_music_command_formant_list = music_formant_list[command_start_index: command_end_index]
        command_formant_shift_list = align_formant(clip_music_command_formant_list, command_formant_list, command_phone_type_list, params)
        pick_npz_file = os.path.join(output_folder, clip_name + PICK_NPZ_SUFFIX)
        np.savez(pick_npz_file, formant_shift_list=command_formant_shift_list, frame_num=t_frame_num, margin_frame_num=margin_frame_num, command_frame_num=command_frame_num)


def pick_wakeup_command(wakeup_analysis_file: str, command_analysis_file: str, music_analysis_file: str, params: dict):
    task_name = params['task_name']
    frame_time = params['frame_time']
    step_time = params['step_time']
    padding_frame_num = params['padding_frame_num']
    margin_frame_num = params['margin_frame_num']
    step_frame_num = params['step_frame_num']
    top_k = params['pick_top_k']
    max_workers = params['max_workers']
    min_pause_time = params['min_pause_time']
    max_pause_time = params['max_pause_time']
    pause_step_time = params['pause_step_time']
    formant_distance_weight = params['formant_distance_weight']
    psychological_weight = params['psychological_weight']
    balance_weight = params['balance_weight']
    pronunciation_weight = params['pronunciation_weight']
    _allow_overlap_ = params['allow_overlap']
    _using_spleeter_ = params['using_spleeter']

    assert max_pause_time > min_pause_time >= 0 and pause_step_time >= step_time, "Value 'pause_time' Error."

    # resolve the path
    wake_up_name = os.path.splitext(os.path.basename(wakeup_analysis_file))[0]
    command_name = os.path.splitext(os.path.basename(command_analysis_file))[0]
    music_name = os.path.splitext(os.path.basename(music_analysis_file))[0]
    pick_folder = os.path.join(get_pick_folder(task_name), wake_up_name, command_name, music_name)
    logger.info("Start picking, wake-up name: '{}', command name: '{}', music name: '{}'.".format(wake_up_name, command_name, music_name))
    os.makedirs(pick_folder, exist_ok=True)

    # load the music analysis file
    with open(music_analysis_file, 'rb') as _file_:
        music_analysis_data = pickle.load(_file_)
    music_formant_list = music_analysis_data['formant_list']
    music_sil_label_list = music_analysis_data['sil_label_list']
    music_psychological_masking_list = music_analysis_data['psychological_masking_list']  # type: np.ndarray
    music_pitch_list = music_analysis_data['pitch_list']
    music_signal = music_analysis_data['music_signal']
    music_phone_list = music_analysis_data.get('phone_list')  # type: Union[None, np.ndarray]
    sample_rate = music_analysis_data['sample_rate']
    vocal_formant_list = music_analysis_data.get('vocal_formant_list')  # type: Union[None, np.ndarray]
    vocal_sil_label_list = music_analysis_data.get('vocal_sil_label_list')  # type: Union[None, np.ndarray]
    vocal_pitch_list = music_analysis_data.get('vocal_pitch_list')  # type: Union[None, np.ndarray]
    vocal_phone_list = music_analysis_data.get('vocal_phone_list')  # type: Union[None, np.ndarray]
    music_intensity_list = music_analysis_data['intensity_list']
    vocal_intensity_list = music_analysis_data.get("vocal_intensity_list")
    music_frame_num = music_formant_list.shape[0]

    # load the wake-up analysis file
    with open(wakeup_analysis_file, 'rb') as _file_:
        wake_up_analysis_data = pickle.load(_file_)
        wake_up_frame_num = wake_up_analysis_data['formant_list'].shape[0]

    # load the command analysis file
    with open(command_analysis_file, 'rb') as _file_:
        command_analysis_data = pickle.load(_file_)
        command_frame_num = command_analysis_data['formant_list'].shape[0]

    frame_sample_num = int(frame_time * sample_rate)
    step_sample_num = int(step_time * sample_rate)
    pause_time_list = np.arange(min_pause_time, max_pause_time, pause_step_time)
    pause_frame_num_list = (pause_time_list / step_time).astype(np.int)

    # 筛选 & 计算得分
    clip_info_dict = {}  # 记录全部信息
    delete_info_dict = collections.defaultdict(set)  # 记录删除的信息
    index_list = []  # 记录音乐片段开始的索引
    frame_num_list = []  # 用于记录音乐片段的帧长
    formant_distance_list = []  # 用于记录共振峰距离
    delta_loudness_list = []  # 用于记录声学掩蔽
    pron_num_list = []  # 用于记录匹配的音素长度
    balance_distance_list = []  # 用以记录平衡系数的距离
    wake_up_formant_shift_matrix = []  # 记录唤醒词的共振峰偏移
    command_formant_shift_matrix = []  # 记录命令的共振峰偏移
    wake_up_delta_k_final_list = []  # 用于记录唤醒词的delta_k_final值
    command_delta_k_final_list = []  # 用于记录命令的delta_k_final值

    # calculate and cache the wake-up related info
    shift_index_list = np.arange(padding_frame_num, music_frame_num - (wake_up_frame_num + command_frame_num + 2 * margin_frame_num + pause_frame_num_list[-1]) - padding_frame_num, step_frame_num)
    wake_up_start_index_list = shift_index_list + margin_frame_num
    wake_up_sample_num = (wake_up_frame_num - 1) * step_sample_num + frame_sample_num
    with futures.ProcessPoolExecutor(max_workers) as executor:
        jobs = []
        for wake_up_s_index in wake_up_start_index_list:
            # load index
            wake_up_e_index = wake_up_s_index + wake_up_frame_num
            wake_up_sample_start_index = wake_up_s_index * step_sample_num
            clip_signal = music_signal[wake_up_sample_start_index: wake_up_sample_start_index + wake_up_sample_num]

            # load clip wake-up info
            clip_music_formant_list = music_formant_list[wake_up_s_index: wake_up_e_index]
            clip_vocal_formant_list = vocal_formant_list[wake_up_s_index: wake_up_e_index]
            clip_music_intensity_list = music_intensity_list[wake_up_s_index: wake_up_e_index]
            clip_vocal_intensity_list = vocal_intensity_list[wake_up_s_index: wake_up_e_index]
            clip_music_pitch_list = music_pitch_list[wake_up_s_index: wake_up_e_index]
            clip_vocal_pitch_list = vocal_pitch_list[wake_up_s_index: wake_up_e_index]
            clip_music_sil_label_list = music_sil_label_list[wake_up_s_index: wake_up_e_index]
            clip_music_phone_list = music_phone_list[wake_up_s_index: wake_up_e_index] if music_phone_list is not None else None
            clip_vocal_sil_label_list = vocal_sil_label_list[wake_up_s_index: wake_up_e_index] if _using_spleeter_ else None
            clip_vocal_phone_list = vocal_phone_list[wake_up_s_index: wake_up_e_index] if _using_spleeter_ and vocal_phone_list is not None else None

            jobs.append(
                executor.submit(get_clip_command_info, clip_signal, clip_music_formant_list, clip_vocal_formant_list, clip_music_intensity_list, clip_vocal_intensity_list, clip_music_pitch_list, clip_vocal_pitch_list, clip_music_sil_label_list, clip_music_phone_list, clip_vocal_sil_label_list, clip_vocal_phone_list, wakeup_analysis_file, params)
            )
        clip_info_list = wait_for_jobs(jobs, executor, "[Pick] Analysis Wake-up Clips: ")
    wake_up_info_dict = dict()
    for wake_up_s_index, clip_info in zip(wake_up_start_index_list, clip_info_list):
        wake_up_info_dict[wake_up_s_index] = clip_info

    # calculate and cache the command related info
    command_start_index_list = set()
    for wake_up_s_index in wake_up_start_index_list:
        if isinstance(wake_up_info_dict[wake_up_s_index], str):
            continue
        for pause_frame_num in pause_frame_num_list:
            command_s_index = wake_up_s_index + pause_frame_num + wake_up_frame_num
            command_start_index_list.add(command_s_index)
    command_start_index_list = list(command_start_index_list)
    command_sample_num = (command_frame_num - 1) * step_sample_num + frame_sample_num
    with futures.ProcessPoolExecutor(max_workers) as executor:
        jobs = []
        for command_s_index in command_start_index_list:
            # load index
            command_e_index = command_s_index + command_frame_num
            command_sample_start_index = command_s_index * step_sample_num
            clip_signal = music_signal[command_sample_start_index: command_sample_start_index + command_sample_num]

            # load clip command info
            clip_music_formant_list = music_formant_list[command_s_index: command_e_index]
            clip_vocal_formant_list = vocal_formant_list[command_s_index: command_e_index]
            clip_music_intensity_list = music_intensity_list[command_s_index: command_e_index]
            clip_vocal_intensity_list = vocal_intensity_list[command_s_index: command_e_index]
            clip_music_pitch_list = music_pitch_list[command_s_index: command_e_index]
            clip_vocal_pitch_list = vocal_pitch_list[command_s_index: command_e_index]
            clip_music_sil_label_list = music_sil_label_list[command_s_index: command_e_index]
            clip_music_phone_list = music_phone_list[command_s_index: command_e_index] if music_phone_list is not None else None
            clip_vocal_sil_label_list = vocal_sil_label_list[command_s_index: command_e_index] if _using_spleeter_ else None
            clip_vocal_phone_list = vocal_phone_list[command_s_index: command_e_index] if _using_spleeter_ and vocal_phone_list is not None else None

            jobs.append(
                executor.submit(
                    get_clip_command_info, clip_signal, clip_music_formant_list, clip_vocal_formant_list, clip_music_intensity_list, clip_vocal_intensity_list, clip_music_pitch_list, clip_vocal_pitch_list, clip_music_sil_label_list, clip_music_phone_list, clip_vocal_sil_label_list, clip_vocal_phone_list, command_analysis_file, params
                )
            )
        clip_info_list = wait_for_jobs(jobs, executor, "[Pick] Analysis Command Clips: ")
    command_info_dict = dict()
    for command_s_index, clip_info in zip(command_start_index_list, clip_info_list):
        command_info_dict[command_s_index] = clip_info

    # process and cache the clip related info
    tmp_index_list = []
    tmp_frame_num_list = []
    with futures.ProcessPoolExecutor(max_workers) as executor:
        jobs = []
        for shift_index in shift_index_list:
            wake_up_s_index = shift_index + margin_frame_num
            for pause_frame_num in pause_frame_num_list:
                # load the index
                command_s_index = wake_up_s_index + pause_frame_num + wake_up_frame_num
                t_frame_num = margin_frame_num * 2 + wake_up_frame_num + pause_frame_num + command_frame_num

                # load cached info
                wake_up_info = wake_up_info_dict[wake_up_s_index]
                if isinstance(wake_up_info, str):
                    delete_info_dict[wake_up_info].add(shift_index)
                    continue
                command_info = command_info_dict[command_s_index]
                if isinstance(command_info, str):
                    delete_info_dict[command_info].add(shift_index)
                    continue

                tmp_index_list.append(shift_index)
                tmp_frame_num_list.append(t_frame_num)
                jobs.append(
                    executor.submit(get_clip_info, wake_up_info, command_info, params)
                )
        clip_info_list = wait_for_jobs(jobs, executor, "[PICK] Analysis Clips: ")
    for _index_ in range(len(clip_info_list)):
        clip_info = clip_info_list[_index_]
        shift_index = tmp_index_list[_index_]
        t_frame_num = tmp_frame_num_list[_index_]
        pause_frame_num = t_frame_num - (wake_up_frame_num + command_frame_num + 2 * margin_frame_num)
        wake_up_s_index = shift_index + margin_frame_num
        command_s_index = wake_up_s_index + wake_up_frame_num + pause_frame_num
        if isinstance(clip_info, str):
            delete_info_dict[clip_info].add(shift_index)
            continue
        index_list.append(shift_index)
        frame_num_list.append(t_frame_num)
        formant_distance_list.append(clip_info['formant_distance'])
        delta_loudness_list.append(clip_info['delta_loudness'])
        wake_up_delta_k_final_list.append(wake_up_info_dict[wake_up_s_index]['delta_k_final'])
        command_delta_k_final_list.append(command_info_dict[command_s_index]['delta_k_final'])
        wake_up_formant_shift_matrix.append(wake_up_info_dict[wake_up_s_index]['formant_shift_list'])
        command_formant_shift_matrix.append(command_info_dict[command_s_index]['formant_shift_list'])
        balance_distance_list.append(clip_info['balance_distance'])
        pron_num_list.append(clip_info['matched_pron_num'])

        # cache the information
        clip_info_dict["{}-{}".format(shift_index, pause_frame_num)] = {
            "shift_index": shift_index,
            "pause_frame_num": pause_frame_num,
            "wake_up_delta_k_final": wake_up_info_dict[wake_up_s_index]['delta_k_final'],
            "command_delta_k_final": command_info_dict[command_s_index]['delta_k_final'],
            'wake_up_formant_distance': wake_up_info_dict[wake_up_s_index]['formant_distance'],
            'command_formant_distance': command_info_dict[command_s_index]['formant_distance'],
            'formant_distance': clip_info['formant_distance'],
            'wake_up_delta_loudness': wake_up_info_dict[wake_up_s_index]['delta_loudness'],
            'command_delta_loudness': command_info_dict[command_s_index]['delta_loudness'],
            'delta_loudness': clip_info['delta_loudness'],
            "wake_up_balance_distance": wake_up_info_dict[wake_up_s_index]['balance_distance'],
            "command_balance_distance": command_info_dict[command_s_index]['balance_distance'],
            "balance_distance": clip_info['balance_distance'],
            "wake_up_matched_phone": wake_up_info_dict[wake_up_s_index]['pron_align_list'],
            "command_matched_phone": command_info_dict[command_s_index]['pron_align_list'],
            "matched_pron_num": clip_info['matched_pron_num'],
        }

    # save the information
    delete_info_file = os.path.join(pick_folder, "delete_info.json")
    save_json_data(delete_info_dict, delete_info_file)
    if not len(index_list):
        logger.error("No music clip is reserved. Please check the filter conditions specified in params. Wake-up: '{}', Command: '{}', Music: '{}'.".format(wake_up_name, command_name, music_name))
        return
    else:
        logger.info("Retains {} clips. {}. Wake-up: '{}', Command: '{}', Music: '{}'.".format(len(index_list), ", ".join(["'{}' {} clips".format(_k, len(_v)) for _k, _v in delete_info_dict.items()]), wake_up_name, command_name, music_name))
    clip_info_file = os.path.join(pick_folder, "clip_info.json")
    save_json_data(clip_info_dict, clip_info_file)

    # format the array
    index_list = np.array(index_list)
    frame_num_list = np.array(frame_num_list)
    formant_distance_list = feature_normalize(formant_distance_list)
    delta_loudness_list = feature_normalize(delta_loudness_list)
    balance_distance_list = feature_normalize(balance_distance_list)
    pron_num_list = feature_normalize(pron_num_list)
    wake_up_formant_shift_matrix = np.array(wake_up_formant_shift_matrix)
    command_formant_shift_matrix = np.array(command_formant_shift_matrix)
    wake_up_delta_k_final_list = np.array(wake_up_delta_k_final_list)
    command_delta_k_final_list = np.array(command_delta_k_final_list)

    # calculate distance
    distance_list = formant_distance_weight * formant_distance_list + psychological_weight * delta_loudness_list + balance_weight * balance_distance_list - pronunciation_weight * pron_num_list

    # 选择片段
    if top_k < 0:
        top_k = distance_list.shape[0]
    else:
        top_k = min(top_k, distance_list.shape[0])
    sort_index_list = np.argsort(distance_list)

    # save the sorted information
    sort_json_file = os.path.join(pick_folder, "total_sort_info.json")
    sort_information = {}
    for _sort_index_ in range(sort_index_list.shape[0]):
        sort_index = sort_index_list[_sort_index_]
        pause_frame_num = frame_num_list[sort_index] - wake_up_frame_num - command_frame_num - 2 * margin_frame_num
        sort_information[_sort_index_] = {
            "clip_name": "{}-{}".format(index_list[sort_index], pause_frame_num),
            "formant_distance": formant_distance_list[sort_index],
            "delta_loudness": delta_loudness_list[sort_index],
            "balance_distance": balance_distance_list[sort_index],
            "pron_num": pron_num_list[sort_index],
            "distance": distance_list[sort_index]
        }
    save_json_data(sort_information, sort_json_file)

    # 消除重叠的片段
    if not _allow_overlap_:
        tmp_index_list = []
        tmp_sort_index_list = []

        for _sort_index_ in sort_index_list:
            shift_index = index_list[_sort_index_]
            _overlap_ = False
            for _tmp_sort_index_, _tmp_index_ in zip(tmp_sort_index_list, tmp_index_list):
                if -frame_num_list[_tmp_sort_index_] < _tmp_index_ - shift_index < frame_num_list[_sort_index_]:
                    _overlap_ = True
                    break
            if not _overlap_:
                tmp_index_list.append(shift_index)
                tmp_sort_index_list.append(_sort_index_)
                if len(tmp_index_list) == top_k:
                    break
        index_list = np.array(tmp_index_list)
        tmp_sort_index_list = np.array(tmp_sort_index_list)
        wake_up_formant_shift_matrix = wake_up_formant_shift_matrix[tmp_sort_index_list]
        command_formant_shift_matrix = command_formant_shift_matrix[tmp_sort_index_list]
        frame_num_list = frame_num_list[tmp_sort_index_list]
        wake_up_delta_k_final_list = wake_up_delta_k_final_list[tmp_sort_index_list]
        command_delta_k_final_list = command_delta_k_final_list[tmp_sort_index_list]
    else:
        index_list = index_list[sort_index_list[:top_k]]
        wake_up_formant_shift_matrix = wake_up_formant_shift_matrix[sort_index_list[:top_k]]
        command_formant_shift_matrix = command_formant_shift_matrix[sort_index_list[:top_k]]
        frame_num_list = frame_num_list[sort_index_list[:top_k]]
        wake_up_delta_k_final_list = wake_up_delta_k_final_list[sort_index_list[:top_k]]
        command_delta_k_final_list = command_delta_k_final_list[sort_index_list[:top_k]]

    # 保存结果
    logger.info("Pick {} clips for wake-up: '{}', command: '{}', music: '{}'.".format(len(index_list), wake_up_name, command_name, music_name))
    for _index_ in range(len(index_list)):
        clip_name = "{}-{}-{}-{}".format(wake_up_name, command_name, music_name, index_list[_index_])
        # 保存clip音频
        clip_sample_index = index_list[_index_] * step_sample_num
        clip_sample_num = (frame_num_list[_index_] - 1) * step_sample_num + frame_sample_num
        music_clip_signal = music_signal[clip_sample_index: clip_sample_index + clip_sample_num]
        output_clip_path = os.path.join(pick_folder, "{}.wav".format(clip_name))
        wav_write(music_clip_signal, output_clip_path, sample_rate)

        # 保存共振峰偏移
        wake_up_formant_shift_list = wake_up_formant_shift_matrix[_index_]
        command_formant_shift_list = command_formant_shift_matrix[_index_]
        pick_npz_file = os.path.join(pick_folder, clip_name + PICK_NPZ_SUFFIX)
        np.savez(pick_npz_file, wake_up_formant_shift_list=wake_up_formant_shift_list, command_formant_shift_list=command_formant_shift_list, frame_num=frame_num_list[_index_], margin_frame_num=margin_frame_num, wake_up_frame_num=wake_up_frame_num, command_frame_num=command_frame_num, wake_up_delta_k_final=wake_up_delta_k_final_list[_index_], command_delta_k_final=command_delta_k_final_list[_index_])

    logger.info("Finish picking, wake-up name: '{}', command name: '{}', music name: '{}'.".format(wake_up_name, command_name, music_name))


@exception_printer
def random_pick_wakeup_command(wake_up_analysis_file: str, command_analysis_file: str, music_analysis_file: str, params: dict, r_pick_num: int = None):
    logger.info("Start random pick. Wake up analysis file: `{}`, command analysis file: `{}`, music analysis file: `{}`, random pick number: `{}`.".format(wake_up_analysis_file, command_analysis_file, music_analysis_file, r_pick_num))
    # load the params
    task_name = params['task_name']
    music_name = os.path.splitext(os.path.basename(music_analysis_file))[0]
    wake_up_name = os.path.splitext(os.path.basename(wake_up_analysis_file))[0]
    command_name = os.path.splitext(os.path.basename(command_analysis_file))[0]
    frame_time = params['frame_time']
    step_time = params['step_time']
    padding_frame_num = params['padding_frame_num']
    margin_frame_num = params['margin_frame_num']
    step_frame_num = params['step_frame_num']
    min_pause_time = params['min_pause_time']
    max_pause_time = params['max_pause_time']
    pause_step_time = params["pause_step_time"]
    _allow_overlap_ = params['allow_overlap']
    output_folder = path_join(get_random_pick_folder(task_name), wake_up_name, command_name, music_name)
    os.makedirs(output_folder, exist_ok=True)
    pick_folder = path_join(get_pick_folder(task_name), wake_up_name, command_name, music_name)
    pick_k = r_pick_num or len(glob.glob(os.path.join(pick_folder, "*.wav")))
    pause_time_list = np.arange(min_pause_time, max_pause_time, pause_step_time)

    if pick_k == 0:
        return

    # load music analysis file
    with open(music_analysis_file, 'rb') as _file_:
        music_analysis_data = pickle.load(_file_)
    sample_rate = music_analysis_data['sample_rate']
    music_signal = music_analysis_data['music_signal']
    music_formant_list = music_analysis_data['formant_list']
    music_frame_num = music_formant_list.shape[0]

    # load wake up analysis file
    with open(wake_up_analysis_file, 'rb') as _file_:
        wake_up_analysis_data = pickle.load(_file_)
    wake_up_formant_list = wake_up_analysis_data['formant_list']
    wake_up_phone_type_list = wake_up_analysis_data['phone_type_list']
    wake_up_frame_num = wake_up_formant_list.shape[0]

    # load command analysis file
    with open(command_analysis_file, 'rb') as _file_:
        command_analysis_data = pickle.load(_file_)
    command_formant_list = command_analysis_data['formant_list']
    command_phone_type_list = command_analysis_data['phone_type_list']
    command_frame_num = command_formant_list.shape[0]

    assert len(pause_time_list) > 0

    # random pick the music clips
    frame_sample_num = int(frame_time * sample_rate)
    step_sample_num = int(step_time * sample_rate)
    pause_frame_num_list = (pause_time_list / step_time).astype(np.int)
    max_pause_frame_num = pause_frame_num_list[-1]
    available_index = np.arange(padding_frame_num, music_frame_num - padding_frame_num - wake_up_frame_num - command_frame_num - 2 * margin_frame_num - max_pause_frame_num, step_frame_num)
    if _allow_overlap_:
        pick_index_list = np.random.choice(available_index, size=pick_k, replace=False)
        pause_list = np.random.choice(pause_frame_num_list, size=pick_k, replace=True)
        clip_frame_num_list = wake_up_frame_num + command_frame_num + 2 * margin_frame_num + pause_list
    else:
        pick_index_list = []
        clip_frame_num_list = []
        while len(pick_index_list) < pick_k:
            _pick_index_ = int(np.random.choice(available_index, 1)[0])
            _pause_frame_num_ = np.random.choice(pause_frame_num_list, 1)[0]
            _clip_frame_num_ = wake_up_frame_num + command_frame_num + 2 * margin_frame_num + _pause_frame_num_
            if _pick_index_ not in pick_index_list:
                _overlap_ = False
                for _index_ in range(len(pick_index_list)):
                    _pre_pick_index_ = pick_index_list[_index_]
                    _pre_frame_num_ = clip_frame_num_list[_index_]
                    if -_clip_frame_num_ < _pick_index_ - _pre_pick_index_ <= _pre_frame_num_:
                        _overlap_ = True
                        break
                if not _overlap_:
                    pick_index_list.append(_pick_index_)
                    clip_frame_num_list.append(_clip_frame_num_)

    for _index_ in range(pick_k):
        shift_index = pick_index_list[_index_]
        pause_frame_num = clip_frame_num_list[_index_] - wake_up_frame_num - command_frame_num - 2 * margin_frame_num
        clip_frame_num = clip_frame_num_list[_index_]
        clip_name = "{}-{}-{}-{}".format(wake_up_name, command_name, music_name, shift_index)

        # save the wav signal
        start_sample_index = shift_index * step_sample_num
        clip_sample_num = (clip_frame_num - 1) * step_sample_num + frame_sample_num
        music_clip_signal = music_signal[start_sample_index: start_sample_index + clip_sample_num]
        output_clip_path = os.path.join(output_folder, "{}.wav".format(clip_name))
        wav_write(music_clip_signal, output_clip_path, sample_rate)

        # save the formant shift file
        wake_up_start_index = shift_index + margin_frame_num
        wake_up_end_index = wake_up_start_index + wake_up_frame_num
        command_start_index = wake_up_end_index + pause_frame_num
        command_end_index = command_start_index + command_frame_num
        clip_music_wake_up_formant_list = music_formant_list[wake_up_start_index: wake_up_end_index]
        clip_music_command_formant_list = music_formant_list[command_start_index: command_end_index]
        wake_up_formant_shift_list = align_formant(clip_music_wake_up_formant_list, wake_up_formant_list, wake_up_phone_type_list, params)
        command_formant_shift_list = align_formant(clip_music_command_formant_list, command_formant_list, command_phone_type_list, params)
        pick_npz_file = os.path.join(output_folder, clip_name + PICK_NPZ_SUFFIX)
        np.savez(pick_npz_file, wake_up_formant_shift_list=wake_up_formant_shift_list, command_formant_shift_list=command_formant_shift_list, frame_num=clip_frame_num, margin_frame_num=margin_frame_num, wake_up_frame_num=wake_up_frame_num, command_frame_num=command_frame_num)


def pick(params: dict):
    pick_top_k = params['pick_top_k']
    wake_up_folder = params['wake_up_folder']
    command_folder = params['command_folder']
    music_folder = params['music_folder']
    _using_wake_up_ = params['using_wake_up']

    if _using_wake_up_:
        logger.info("*" * 20 + "    Start Pick Music Clips Using Wake-up and Command.    " + "*" * 20)
    else:
        logger.info("*" * 20 + "    Start Pick Music Clips Using Command.    " + "*" * 20)

    if pick_top_k == 0:
        logger.warning("Parameter 'pick_top_k = 0', skip picking music clips.")
        exit(-1)
    elif pick_top_k < 0:
        logger.warning("Parameter 'pick_top_k < 0', script will save all of the reserved clips.")

    # 获取文件
    if _using_wake_up_:
        wake_up_analysis_files = glob.glob(os.path.join(wake_up_folder, "*" + ANALYSIS_SUFFIX))
    else:
        wake_up_analysis_files = [None]
    command_analysis_files = glob.glob(os.path.join(command_folder, "*" + ANALYSIS_SUFFIX))
    music_analysis_files = glob.glob(os.path.join(music_folder, "*" + ANALYSIS_SUFFIX))

    for music_analysis_file in music_analysis_files:  # music
        for wake_up_analysis_file in wake_up_analysis_files:  # wake-up
            for command_analysis_file in command_analysis_files:  # command
                if _using_wake_up_:
                    pick_wakeup_command(wake_up_analysis_file, command_analysis_file, music_analysis_file, params)
                else:
                    pick_command(command_analysis_file, music_analysis_file, params)
    logger.info("*" * 20 + "    Pick Done. Please See the result in folder '{}'.    ".format(get_pick_folder(params['task_name'])) + "*" * 20)


def pick_music(params):
    task_folder = get_task_folder(params['task_name'])
    os.makedirs(task_folder, exist_ok=True)

    # 写入参数文件
    pick_param_pkl_file = get_pick_param_pkl_path(params['task_name'])
    with open(pick_param_pkl_file, 'wb') as _file_:
        pickle.dump(params, _file_)
    pick_param_json_file = get_pick_param_json_path(params['task_name'])
    save_json_data(params, pick_param_json_file)

    # 检索并备份音乐、命令
    check_audio(params)

    # 使用spleeter分离音乐中的命令和指令
    separate_audio(params)

    # 解析音乐/命令
    analysis_wav(params)

    # 选择合适的音乐片段
    pick(params)


def random_pick_music(task_name: str, max_workers: int = None):
    # load the param
    pick_param_pkl_path = get_pick_param_pkl_path(task_name)
    with open(pick_param_pkl_path, 'rb') as _param_handler_:
        params = pickle.load(_param_handler_)

    _using_wake_up_ = params['using_wake_up']
    max_workers = max_workers or params['max_workers']

    # load the path
    random_pick_folder = get_random_pick_folder(task_name)
    shutil.rmtree(random_pick_folder, ignore_errors=True)
    os.makedirs(random_pick_folder, exist_ok=True)

    # load the files
    if _using_wake_up_:
        wake_up_folder = get_wake_up_folder(task_name)
        wake_up_folder = wake_up_folder if os.path.exists(wake_up_folder) else params['wake_up_folder']  # todo: 删除对旧脚本的兼容
        wake_up_analysis_files = glob.glob(os.path.join(wake_up_folder, "*" + ANALYSIS_SUFFIX))
    else:
        wake_up_analysis_files = [None]
    command_folder = get_command_folder(task_name)
    command_folder = command_folder if os.path.exists(command_folder) else params['command_folder']  # todo: 删除对旧脚本的兼容
    command_analysis_files = glob.glob(os.path.join(command_folder, "*" + ANALYSIS_SUFFIX))
    music_folder = get_music_folder(task_name)
    music_folder = music_folder if os.path.exists(music_folder) else params['music_folder']  # todo: 删除对旧脚本的兼容
    music_analysis_files = glob.glob(os.path.join(music_folder, "*" + ANALYSIS_SUFFIX))

    # iterate every combinations
    with futures.ProcessPoolExecutor(max_workers=max_workers) as _executor_:
        jobs = []
        for wake_up_analysis_file in wake_up_analysis_files:
            wake_up_name = None if wake_up_analysis_file is None else os.path.splitext(os.path.basename(wake_up_analysis_file))[0]
            for command_analysis_file in command_analysis_files:
                command_name = os.path.splitext(os.path.basename(command_analysis_file))[0]
                command_pick_folder = path_join(
                    get_pick_folder(task_name), wake_up_name, command_name
                )
                clip_file_num = len(
                    glob.glob(os.path.join(command_pick_folder, "**/*.wav"), recursive=True)
                )
                random_pick_num_dict = collections.Counter(
                    random.choices(music_analysis_files, k=clip_file_num)
                )

                for music_analysis_file, _value_ in random_pick_num_dict.items():
                    if _using_wake_up_:
                        jobs.append(
                            _executor_.submit(random_pick_wakeup_command, wake_up_analysis_file, command_analysis_file, music_analysis_file, params, _value_)
                        )
                    else:
                        jobs.append(
                            _executor_.submit(random_pick_command, command_analysis_file, music_analysis_file, params, _value_)
                        )

        wait_for_jobs(jobs, _executor_, "[Random] Random Pick Music Clip Progress: ")
    logger.info("*" * 20 + "    Random Pick Done. Please See the result in folder '{}'.    ".format(random_pick_folder) + "*" * 20)
