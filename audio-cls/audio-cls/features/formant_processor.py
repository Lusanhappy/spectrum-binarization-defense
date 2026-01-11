# coding=utf-8
import parselmouth
import pickle
# from physical_environment_noise import random_noise, room_reverberation, room_beamforming
from features.utils import *
from scipy import signal
from typing import Optional

# __word2phone_dict_path__ = "./dict/word2phone.json"
# with open(__word2phone_dict_path__, 'r') as _file_:
#     __word2phone_dict__ = json.load(_file_)  # type: dict
# __phone2type_dict_path__ = "./dict/phone2type.pkl"
# with open(__phone2type_dict_path__, 'rb') as _file_:
#     __phone2type_dict__ = pickle.load(_file_)  # type: dict
# __id2phone_dict_path__ = "./dict/id2phone.json"
# with open(__id2phone_dict_path__, 'r') as _json_file_:
#     __id2phone_dict__ = json.load(_json_file_)
# __phone2pron_dict_path__ = "./dict/phone2pron.json"
# with open(__phone2pron_dict_path__, 'r') as _json_file_:
#     __phone2pron_dict__ = json.load(_json_file_)
# _noise_func_dict_ = {
#     VoicelessProcessType.BlueNoise: blue_noise,
#     VoicelessProcessType.WhiteNoise: white_noise,
#     VoicelessProcessType.VioletNoise: violet_noise,
#     VoicelessProcessType.PinkNoise: pink_noise
# }


# def get_pronunciation(phone_list: List[str]) -> List[str]:
#     return [__phone2pron_dict__.get(phone) for phone in phone_list]


def get_formants(audio_path: str = None, audio_signal: np.ndarray = None, sample_rate: int = None, frame_time: float = 0.025, step_time: float = 0.01, formant_num: int = MAX_FORMANT_NUM) -> np.ndarray:
    """
    calculate audio's formants.
    :param audio_path: the path of audio.
    :param audio_signal: the audio signal.
    :param sample_rate: the sample_rate.
    :param frame_time: frame time. Default is 0.025.
    :param step_time: step time. Default is 0.01.
    :param formant_num: formant number. Default is 5.
    :return: formants.
    """
    assert formant_num <= 5

    snd = get_parselmouth_sound_ele(audio_path, audio_signal, sample_rate)
    formant_handler = snd.to_formant_burg(step_time, window_length=frame_time)
    duration = formant_handler.duration
    frame_num = int((duration - frame_time) / step_time) + 1
    _ct_ = __get_ct__(frame_num, frame_time, step_time)
    formants = np.zeros((frame_num, formant_num), dtype=np.float)

    for frame_index in range(frame_num):
        for formant_index in range(formant_num):
            formants[frame_index, formant_index] = formant_handler.get_value_at_time(formant_number=formant_index + 1, time=_ct_[frame_index])

    return formants


def get_intensity(audio_path: str = None, audio_signal: np.ndarray = None, sample_rate: int = None, frame_time: float = 0.025, step_time: float = 0.01) -> np.ndarray:
    """
    calculate audio's intensity.
    The audio_path or (audio_signal, sample_rate) must be specified.
    :param audio_path: the path of audio.
    :param audio_signal: the audio signal.
    :param sample_rate: sample rate
    :param frame_time: frame time.
    :param step_time: step time.
    :return: intensity list.
    """
    snd = get_parselmouth_sound_ele(audio_path, audio_signal, sample_rate)
    intensity_handler = snd.to_intensity(100.0)
    duration = intensity_handler.duration
    frame_num = int((duration - frame_time) / step_time) + 1
    _ct_ = __get_ct__(frame_num, frame_time, step_time)

    intensity_list = []
    for _frame_index_ in range(frame_num):
        intensity_list.append(intensity_handler.get_value(time=_ct_[_frame_index_]))

    return np.array(intensity_list)


def get_bandwidths(audio_path: str = None, audio_signal: np.ndarray = None, sample_rate: int = None, frame_time: float = 0.025, step_time: float = 0.01, formant_num: int = MAX_FORMANT_NUM) -> np.ndarray:
    """
    calculate audio's bandwidths for formant.
    The audio_path or (audio_signal, sample_rate) must be specified.
    :param audio_path: the path of audio.
    :param audio_signal: the audio signal.
    :param sample_rate: sample rate
    :param frame_time: frame time.
    :param step_time: step time.
    :param formant_num: formant number.
    :return: bandwidths.
    """
    assert formant_num <= 5

    snd = get_parselmouth_sound_ele(audio_path, audio_signal, sample_rate)
    formant_handler = snd.to_formant_burg(step_time, window_length=frame_time)
    duration = formant_handler.duration
    frame_num = int((duration - frame_time) / step_time) + 1
    _ct_ = __get_ct__(frame_num, frame_time, step_time)
    bandwidths = np.zeros((frame_num, formant_num), dtype=np.float)

    for frame_index in range(frame_num):
        for formant_index in range(formant_num):
            bandwidths[frame_index, formant_index] = formant_handler.get_bandwidth_at_time(formant_number=formant_index + 1, time=_ct_[frame_index])

    return bandwidths


def __get_ct__(frame_num, frame_time=0.025, step_time=0.01):
    return np.arange(frame_num) * step_time + frame_time / 2.0


# def read_pdf_file(pdf_path):
#     with open(pdf_path, 'r') as _pdf_file_:
#         data_lines = _pdf_file_.readlines()
#         phones = []

#         for data_line in data_lines:
#             if data_line:
#                 info = data_line.split()

#                 if len(info) == 1:
#                     pdf_id = info[0]
#                 else:
#                     pdf_id = info[4]
#                 phones.append(__id2phone_dict__[pdf_id])
#     return phones


# def word2phone(word: str) -> List:
#     """
#     get phoneme list of a word
#     """
#     word = word.lower()
#     # 直接根据当前单词获取
#     phone_queue_list = __word2phone_dict__.get(word)
#     if phone_queue_list:
#         return phone_queue_list

#     # 两个单词拼接获取
#     phone_queue_list = []
#     for _i_ in range(1, len(word) - 1):
#         _pre_ = __word2phone_dict__.get(word[:_i_ + 1])
#         _next_ = __word2phone_dict__.get(word[_i_ + 1:])
#         if _pre_ is not None and _next_ is not None:
#             for pre_phone_queue in _pre_:
#                 for next_phone_queue in _next_:
#                     phone_queue_list.append(pre_phone_queue + next_phone_queue)

#     if len(phone_queue_list) != 0:
#         return phone_queue_list
#     else:
#         raise ValueError("Can't find available phoneme list for word '{}'. You can add it into json file {} by yourself.".format(word.lower(), __word2phone_dict_path__))


# def get_phones(pdf_path: str, frame_time: float = 0.025, step_time=0.01, pdf_frame_time=0.025, pdf_step_time=0.01, frame_num=None):
#     """
#     get phone of every audio frame.
#     :param pdf_path: the path of pdf file.
#     :param frame_time: frame time when calculate formant.
#     :param step_time: step time when calculate formant.
#     :param pdf_frame_time: frame time when calculate pdf-id.
#     :param pdf_step_time: step time when calculate pdf-id.
#     :param frame_num: number of frame. Default is None, script will calculate it from pdf file.
#     :return: phones.
#     """
#     phones = read_pdf_file(pdf_path)
#     if len(phones) == 0:
#         raise ValueError("Get empty from pdf file {}.".format(pdf_path))

#     phone_num = len(phones)
#     if not frame_num:
#         duration = (phone_num - 1) * pdf_step_time + pdf_frame_time
#         frame_num = int((duration - frame_time) // step_time + 1)

#     formant_ct = __get_ct__(frame_num, frame_time, step_time)
#     pdf_ct = __get_ct__(phone_num, pdf_frame_time, pdf_step_time)

#     t_frame_phone = []
#     for t_index in range(frame_num):
#         _idx_ = int(np.argmin(np.abs(pdf_ct - formant_ct[t_index])))
#         t_frame_phone.append(phones[_idx_])
#     return np.array(t_frame_phone).astype("<U8")


# def phone2type(phone_list: Union[np.ndarray, List]) -> np.ndarray:
#     phone_type_list = []
#     for phone in phone_list:
#         phone_type_list.append(__phone2type_dict__[phone])
#     return np.array(phone_type_list)


# def generate_reverse_filter(formant_list: np.ndarray, sample_rate: int, bandwidth_list: np.ndarray, delete_check_list: np.ndarray, min_fre: float = 0.0, max_fre: float = None, filter_order: int = 4) -> List[List]:
#     """
#     generate reverse filter group
#     """
#     # pre check the parameters
#     assert np.all(formant_list.shape == bandwidth_list.shape) and np.all(formant_list.shape == delete_check_list.shape)

#     frame_num, formant_num = formant_list.shape
#     half_bandwidth_list = bandwidth_list / 2.0
#     min_fre = np.amax((min_fre, EPS))
#     if max_fre is None:
#         max_fre = sample_rate / 2.0
#     else:
#         max_fre = np.amin((sample_rate / 2.0, max_fre))
#     _nyq_ = sample_rate / 2.0 + EPS

#     frame_filter_list = []
#     for _frame_index_ in range(frame_num):
#         filter_bin_list = []
#         fre_band = []
#         pre_bin = None
#         pre_index = None
#         pre_formant = None
#         pre_half_bandwidth = None
#         for _formant_index_ in range(formant_num):
#             formant = formant_list[_frame_index_, _formant_index_]
#             half_bandwidth = half_bandwidth_list[_frame_index_, _formant_index_]

#             if np.isnan(formant) or formant < min_fre or formant > max_fre:
#                 filter_bin_list.append(None)
#                 continue

#             if pre_bin is None or formant - pre_bin[1] >= half_bandwidth:
#                 _s_fre_ = max(min_fre, formant - half_bandwidth)
#                 _e_fre_ = min(max_fre, formant + half_bandwidth)
#                 if _s_fre_ >= _e_fre_:  # Filter Overflow
#                     break
#             else:  # Filter Overlap
#                 _s_fre_, _e_fre_ = pre_bin
#                 del filter_bin_list[pre_index]
#                 gap = formant - pre_formant
#                 _c_fre_ = pre_formant + gap * pre_half_bandwidth / (pre_half_bandwidth + half_bandwidth)
#                 filter_bin_list.insert(pre_index, (_s_fre_, _c_fre_))
#                 _s_fre_ = _c_fre_
#                 _e_fre_ = min(max_fre, formant + half_bandwidth)
#                 if _s_fre_ >= _e_fre_:  # Filter Overflow
#                     del filter_bin_list[pre_index]
#                     filter_bin_list.insert(pre_index, pre_bin)
#                     break

#             pre_bin = (_s_fre_, _e_fre_)
#             filter_bin_list.append(pre_bin)
#             pre_index = len(filter_bin_list) - 1
#             pre_formant = formant
#             pre_half_bandwidth = half_bandwidth

#         # generate filters
#         filter_list = []
#         fre_point_list = []
#         for _formant_index_ in range(len(filter_bin_list)):
#             if delete_check_list[_frame_index_, _formant_index_] and filter_bin_list[_formant_index_]:
#                 fre_point_list.append((filter_bin_list[_formant_index_][0], 0))
#                 fre_point_list.append((filter_bin_list[_formant_index_][1], 1))
#         fre_point_list.sort()
#         pre_fre = EPS
#         status = 0
#         for fre_point in fre_point_list:
#             if fre_point[1]:
#                 status -= 1
#                 if status == 0:
#                     pre_fre = fre_point[0]
#             else:
#                 if status == 0 and pre_fre < fre_point[0]:
#                     filter_list.append(
#                         signal.butter(filter_order, [pre_fre / _nyq_, fre_point[0] / _nyq_], btype="band", output="sos")
#                     )
#                 status += 1
#         assert status == 0
#         if pre_fre < max_fre:
#             filter_list.append(
#                 signal.butter(filter_order, [pre_fre / _nyq_, max_fre / _nyq_], btype="band", output="sos")
#             )
#             fre_band.append((pre_fre, max_fre))
#         frame_filter_list.append(filter_list)
#     return frame_filter_list


# def generate_filter(formant_list: np.ndarray, sample_rate: int, bandwidth_list: np.ndarray, min_fre: float = 20.0, max_fre: float = None, filter_order: int = 4, reserved_fre_ratio: float = 0.2) -> List[List]:
#     """
#     generate filter group
#     todo: 针对有重叠的共振峰进行声音增强
#     """
#     # pre check the parameters
#     assert np.all(formant_list.shape == bandwidth_list.shape)
#     assert reserved_fre_ratio < 1.0

#     frame_num, formant_num = formant_list.shape
#     half_bandwidth_list = bandwidth_list / 2.0
#     min_fre = np.amax((min_fre, EPS))
#     if max_fre is None:
#         max_fre = sample_rate / 2.0
#     else:
#         max_fre = np.amin((sample_rate / 2.0, max_fre))
#     _nyq_ = sample_rate / 2.0 + EPS

#     frame_filter_list = []
#     for _frame_index_ in range(frame_num):
#         filter_bin_list = []
#         pre_bin = None
#         pre_index = None
#         pre_formant = None
#         pre_half_bandwidth = None
#         for _formant_index_ in range(formant_num):
#             formant = formant_list[_frame_index_, _formant_index_]
#             half_bandwidth = half_bandwidth_list[_frame_index_, _formant_index_]

#             if np.isnan(formant) or formant < min_fre or formant > max_fre:
#                 filter_bin_list.append(None)
#                 continue

#             _is_overlap_ = False
#             if pre_bin is None or formant - pre_bin[1] >= half_bandwidth:
#                 _s_fre_ = max(min_fre, formant - half_bandwidth)
#                 _e_fre_ = min(max_fre, formant + half_bandwidth)
#                 if _s_fre_ >= _e_fre_:  # Filter Overflow
#                     break
#             else:  # Filter Overlap
#                 _is_overlap_ = True
#                 _s_fre_, _e_fre_, _ = pre_bin
#                 del filter_bin_list[pre_index]
#                 gap = formant - pre_formant
#                 _c_fre_ = pre_formant + gap * pre_half_bandwidth / (pre_half_bandwidth + half_bandwidth) * (1 - reserved_fre_ratio)
#                 filter_bin_list.insert(pre_index, (_s_fre_, _c_fre_, True))
#                 _s_fre_ = _c_fre_ + gap * reserved_fre_ratio
#                 _e_fre_ = min(max_fre, formant + half_bandwidth)
#                 if _s_fre_ >= _e_fre_:  # Filter Overflow
#                     del filter_bin_list[pre_index]
#                     filter_bin_list.insert(pre_index, pre_bin)
#                     break

#             pre_bin = (_s_fre_, _e_fre_, _is_overlap_)
#             filter_bin_list.append(pre_bin)
#             pre_index = len(filter_bin_list) - 1
#             pre_formant = formant
#             pre_half_bandwidth = half_bandwidth

#         # generate filters
#         filter_list = []
#         for filter_bin in filter_bin_list:
#             if filter_bin is None:
#                 filter_list.append(None)
#                 continue

#             _s_fre_, _e_fre_, _is_overlap_ = filter_bin
#             _s_fre_ /= _nyq_
#             _e_fre_ /= _nyq_
#             f_order = filter_order
#             if _is_overlap_:
#                 f_order += 4
#             filter_list.append(
#                 signal.butter(f_order, [_s_fre_, _e_fre_], btype='band', output='sos')
#             )
#         frame_filter_list.append(filter_list)
#     return frame_filter_list


# def filter_signal(music_signal: np.ndarray, sample_rate: int, filter_list: List[list], phone_type_list: np.ndarray, delta_db_list: np.ndarray, balance_coe: np.ndarray, frame_time: float, step_time: float, voiceless_process_type: VoicelessProcessType = VoicelessProcessType.BlueNoise, voiceless_noise_db: float = 65.0, voiceless_noise_fre_interval: Tuple[float, float] = (0.0, 8000.0), window_func: Callable = sqrt_hanning_window) -> np.ndarray:
#     """
#     Filter the signal using filters generated by function `generate_filter`.
#     Return the filtered signal.
#     """
#     assert len(filter_list) == phone_type_list.shape[0]
#     assert np.all(delta_db_list.shape == balance_coe.shape)

#     # initial params
#     frame_num = phone_type_list.shape[0]
#     frame_sample_num = int(sample_rate * frame_time)
#     step_sample_num = int(sample_rate * step_time)

#     # for filter normalization
#     denominator_array = np.zeros_like(music_signal, dtype=np.float)
#     window = window_func(frame_sample_num)
#     power_window = window ** 2

#     # filter out the music clip with filters
#     filtered_signal = np.zeros_like(music_signal)
#     enhance_list = np.power(10, delta_db_list / 20.0) - 1.0  # shape: (frame_num, MAX_FORMANT_NUM)
#     voiceless_noise_enhance = np.power(10, voiceless_noise_db / 20.0) - 1.0
#     noise_func = _noise_func_dict_[voiceless_process_type]
#     for _frame_index_ in range(frame_num):
#         _s_ = step_sample_num * _frame_index_
#         _e_ = _s_ + frame_sample_num
#         framed_signal = music_signal[_s_: _e_] * window
#         phone_type = phone_type_list[_frame_index_]
#         denominator_array[_s_: _e_] += power_window

#         if phone_type == PhoneType.Sil:  # process the sil phoneme
#             continue

#         if phone_type == PhoneType.Voiceless and voiceless_process_type != VoicelessProcessType.FormantFilter:  # process the voiceless phoneme
#             filtered_signal[_s_: _e_] += noise_func(frame_sample_num, sample_rate, *voiceless_noise_fre_interval) * power_window * balance_coe[_frame_index_][0] * voiceless_noise_enhance
#         else:  # process the vowel and voice phoneme
#             for _formant_index_ in range(min(MAX_FORMANT_NUM, len(filter_list[_frame_index_]))):
#                 formant_filter = filter_list[_frame_index_][_formant_index_]
#                 scale = enhance_list[_frame_index_, _formant_index_]
#                 if formant_filter is not None:
#                     filtered_signal[_s_: _e_] += signal.sosfilt(formant_filter, framed_signal) * window * balance_coe[_frame_index_][_formant_index_] * scale
#     denominator_array = np.where(denominator_array < 0.5, 1, denominator_array)
#     return filtered_signal / denominator_array


# def new_filter_signal(music_signal: np.ndarray, sample_rate: int, filter_list: List[list], phone_list: np.ndarray, phone_type_list: np.ndarray, formant_list: np.ndarray, delta_db_list: np.ndarray, balance_coe: np.ndarray, voiceless_energy_ratio_list: np.ndarray, frame_time: float, step_time: float, voiceless_process_type: VoicelessProcessType = VoicelessProcessType.BlueNoise, voiceless_noise_fre_interval: Tuple[float, float] = (20.0, 6000.0), window_func: Callable = sqrt_hanning_window) -> np.ndarray:
#     """
#     Filter the signal
#     """
#     assert len(filter_list) == phone_type_list.shape[0]
#     assert np.all(delta_db_list.shape == balance_coe.shape)

#     # initial params
#     frame_num = phone_type_list.shape[0]
#     frame_sample_num = int(sample_rate * frame_time)
#     step_sample_num = int(sample_rate * step_time)

#     # for filter normalization
#     denominator_array = np.zeros_like(music_signal, dtype=np.float)
#     window = window_func(frame_sample_num)
#     power_window = window ** 2

#     # filter out the music clip with filters
#     enhance_list = np.power(10, delta_db_list / 20.0) - 1.0  # shape: (frame_num, MAX_FORMANT_NUM)
#     if voiceless_process_type == VoicelessProcessType.FormantFilter:  # using formant filter to generate voiceless phoneme.
#         filtered_signal = np.zeros_like(music_signal)
#         for _frame_index_ in range(frame_num):
#             _s_ = step_sample_num * _frame_index_
#             _e_ = _s_ + frame_sample_num
#             framed_signal = music_signal[_s_: _e_] * window
#             phone_type = phone_type_list[_frame_index_]
#             denominator_array[_s_: _e_] += power_window
#             if phone_type == PhoneType.Sil:  # process the sil phoneme
#                 continue
#             for _formant_index_ in range(min(MAX_FORMANT_NUM, len(filter_list[_frame_index_]))):
#                 formant_filter = filter_list[_frame_index_][_formant_index_]
#                 scale = enhance_list[_frame_index_, _formant_index_]
#                 if formant_filter is not None:
#                     filtered_signal[_s_: _e_] += signal.sosfilt(formant_filter, framed_signal) * window * balance_coe[_frame_index_][_formant_index_] * scale
#         denominator_array = np.where(denominator_array < 0.5, 1, denominator_array)
#         return filtered_signal / denominator_array

#     frame_filtered_signal = np.zeros((frame_num, frame_sample_num))
#     frame_energy_list = np.zeros((frame_num,))
#     for _frame_index_ in range(frame_num):
#         _s_ = step_sample_num * _frame_index_
#         _e_ = _s_ + frame_sample_num
#         framed_signal = music_signal[_s_: _e_] * window
#         phone_type = phone_type_list[_frame_index_]
#         denominator_array[_s_: _e_] += power_window
#         if phone_type == PhoneType.Sil or phone_type == PhoneType.Voiceless:  # skip the sil and voiceless phoneme
#             continue
#         for _formant_index_ in range(min(MAX_FORMANT_NUM, len(filter_list[_frame_index_]))):
#             formant_filter = filter_list[_frame_index_][_formant_index_]
#             scale = enhance_list[_frame_index_, _formant_index_]
#             if formant_filter is not None:
#                 frame_filtered_signal[_frame_index_] += signal.sosfilt(formant_filter, framed_signal) * window * balance_coe[_frame_index_][_formant_index_] * scale
#         frame_energy_list[_frame_index_] = np.sqrt(np.sum(frame_filtered_signal[_frame_index_] ** 2))
#     voice_energy_ave = np.nanmean(np.where(phone_type_list == PhoneType.Vowel, frame_energy_list, np.NaN))

#     voiceless_energy_ratio_list = np.maximum(np.minimum(voiceless_energy_ratio_list / 10.0, 0.4), 0.1)
#     if voiceless_process_type == VoicelessProcessType.AdvancedNoise:
#         filtered_signal = np.zeros_like(music_signal)
#         for _frame_index_ in range(frame_num):
#             _s_ = step_sample_num * _frame_index_
#             _e_ = _s_ + frame_sample_num
#             phone_type = phone_type_list[_frame_index_]
#             phone = str(phone_list[_frame_index_])
#             if phone_type != PhoneType.Voiceless:
#                 filtered_signal[_s_: _e_] += frame_filtered_signal[_frame_index_]
#                 continue
#             if phone in ['hh', 'p', 'k']:
#                 noise_func = red_noise
#                 noise_min_fre = 20.
#                 noise_max_fre = 7000.
#             elif phone in ['t', 'f', 'th', 's']:
#                 noise_func = blue_noise
#                 noise_min_fre = 3000.
#                 noise_max_fre = 6000.
#             elif phone in ['ch', 'sh']:
#                 noise_func = red_noise
#                 noise_min_fre = formant_list[_frame_index_, 0]
#                 noise_max_fre = 7000.
#             else:
#                 logger.warning("Something wrong in voiceless process.")
#                 noise_func = white_noise
#                 noise_min_fre = 3000.
#                 noise_max_fre = 6000.
#             filtered_signal[_s_: _e_] += noise_func(frame_sample_num, sample_rate, noise_min_fre, noise_max_fre) * power_window * balance_coe[_frame_index_][0] * voice_energy_ave * voiceless_energy_ratio_list[_frame_index_]
#     else:
#         filtered_signal = np.zeros_like(music_signal)
#         noise_func = _noise_func_dict_[voiceless_process_type]
#         for _frame_index_ in range(frame_num):
#             _s_ = step_sample_num * _frame_index_
#             _e_ = _s_ + frame_sample_num
#             phone_type = phone_type_list[_frame_index_]
#             if phone_type != PhoneType.Voiceless:
#                 filtered_signal[_s_: _e_] += frame_filtered_signal[_frame_index_]
#                 continue
#             filtered_signal[_s_: _e_] += noise_func(frame_sample_num, sample_rate, *voiceless_noise_fre_interval) * power_window * balance_coe[_frame_index_][0] * voice_energy_ave * voiceless_energy_ratio_list[_frame_index_]
#     denominator_array = np.where(denominator_array < 0.5, 1, denominator_array)
#     return filtered_signal / denominator_array


# def get_energy_balance_coe(music_signal: np.ndarray, command_signal: np.ndarray, sample_rate: int, filter_list: List[list], phone_type_list: np.ndarray, frame_time: float, step_time: float, voiceless_process_type: VoicelessProcessType = VoicelessProcessType.BlueNoise, window_func: Callable = sqrt_hanning_window, beta: int = 8) -> np.ndarray:
#     """
#     Calculate the energy balance coefficient.    
#     """
#     assert len(music_signal) == len(command_signal), "Unequal signal length between {} and {}.".format(len(music_signal), len(command_signal))
#     # get filtered sqrt-energy
#     music_filtered_energy = get_filter_signal_energy(music_signal, sample_rate, filter_list, phone_type_list, frame_time, step_time, voiceless_process_type, window_func)
#     command_filtered_energy = get_filter_signal_energy(command_signal, sample_rate, filter_list, phone_type_list, frame_time, step_time, voiceless_process_type, window_func)
#     # calculate the coefficient
#     coefficient = np.power(
#         command_filtered_energy / (music_filtered_energy + EPS),
#         1.0 / beta
#     )
#     return coefficient


# def get_filter_signal_energy(wav_signal: np.ndarray, sample_rate: int, filter_list: List[list], phone_type_list: np.ndarray, frame_time: float, step_time: float, voiceless_process_type: VoicelessProcessType = VoicelessProcessType.BlueNoise, window_func: Callable = sqrt_hanning_window) -> np.ndarray:
#     """
#     Calculate the sqrt-energy of filtered signal to be used in energy balance.
#     """
#     assert len(filter_list) == phone_type_list.shape[0]

#     # filter out the music clip with filters
#     frame_num = phone_type_list.shape[0]
#     frame_sample_num = int(sample_rate * frame_time)
#     step_sample_num = int(sample_rate * step_time)
#     window = window_func(frame_sample_num)
#     filtered_energy = np.zeros((frame_num, MAX_FORMANT_NUM), dtype=np.float)
#     for _frame_index_ in range(frame_num):
#         phone_type = phone_type_list[_frame_index_]
#         if phone_type == PhoneType.Sil:  # process the sil phonemes
#             continue
#         _s_ = step_sample_num * _frame_index_
#         _e_ = _s_ + frame_sample_num
#         framed_signal = wav_signal[_s_: _e_] * window
#         if phone_type == PhoneType.Voiceless and voiceless_process_type != VoicelessProcessType.FormantFilter:  # process the voiceless phoneme
#             filtered_energy[_frame_index_, :] = 1.
#         else:  # process the voice and vowel phonemes
#             for _formant_index_ in range(min(MAX_FORMANT_NUM, len(filter_list[_frame_index_]))):
#                 formant_filter = filter_list[_frame_index_][_formant_index_]
#                 if formant_filter is not None:
#                     f_signal = signal.sosfilt(formant_filter, framed_signal) * window
#                     filtered_energy[_frame_index_, _formant_index_] = np.sum(f_signal ** 2)

#     return np.sqrt(filtered_energy)


def get_parselmouth_sound_ele(audio_path: str = None, audio_signal: np.ndarray = None, sample_rate: int = None) -> parselmouth.Sound:
    """
    get the parselmouth sound element
    """
    if audio_path is not None:
        snd = parselmouth.Sound(audio_path)
    else:
        if audio_signal is None or sample_rate is None:
            raise ValueError("Value Error. One of `audio_path` and (`audio_signal`, `sample_rate`) must be specified.")
        snd = parselmouth.Sound(values=audio_signal, sampling_frequency=sample_rate)

    return snd


def get_pitch(audio_path: str = None, audio_signal: np.ndarray = None, sample_rate: int = None, frame_time: float = 0.025, step_time: float = 0.01) -> np.ndarray:
    """
    calculate audio's pitch.
    :param audio_path: the path of audio
    :param audio_signal: the audio signal
    :param sample_rate: the sample rate
    :param frame_time: frame time. Default is 0.025s.
    :param step_time:  step time. Default is 0.01s.
    :return: pitch list.
    """
    snd = get_parselmouth_sound_ele(audio_path, audio_signal, sample_rate)
    pitch_handler = snd.to_pitch(time_step=step_time)  # type: parselmouth.Pitch
    duration = pitch_handler.duration
    frame_num = int((duration - frame_time) / step_time) + 1
    _ct_ = __get_ct__(frame_num, frame_time, step_time)
    pitch_list = np.zeros((frame_num,), dtype=np.float)
    for frame_index in range(frame_num):
        pitch_list[frame_index] = pitch_handler.get_value_at_time(_ct_[frame_index])

    return pitch_list


# def add_physical_noise(adversarial_signal: np.ndarray, sample_rate, output_folder: str, _adversarial_filename_: str, params: dict) -> List[str]:
#     adversarial_noise_number = params['adversarial_noise_number']
#     adversarial_noise_type = params['adversarial_noise_type']
#     adversarial_noise_max_amplitude = params['adversarial_noise_max_amplitude']
#     adversarial_reverberation_number = params['adversarial_reverberation_number']
#     adversarial_beamforming_number = params['adversarial_beamforming_number']
#     adversarial_beamforming_path = params['adversarial_beamforming_path']
#     min_fre = params['min_fre']
#     max_fre = params['max_fre']
#     if adversarial_beamforming_path is None or not os.path.exists(adversarial_beamforming_path):
#         adversarial_beamforming_number = 0

#     file_path_list = []
#     # add random noise
#     for _noise_index_ in range(adversarial_noise_number):
#         noised_adversarial_signal = random_noise(sample_rate, adversarial_signal, min_fre, max_fre, adversarial_noise_max_amplitude, adversarial_noise_type)
#         noise_file = _adversarial_filename_ + ".noise_{}.wav".format(_noise_index_)
#         output_path = os.path.join(output_folder, noise_file)
#         wav_write(noised_adversarial_signal, output_path, sample_rate)
#         file_path_list.append(output_path)

#     # add reverberation
#     for _rever_index_ in range(adversarial_reverberation_number):
#         rever_adversarial_signal = room_reverberation(sample_rate, adversarial_signal)
#         rever_file = _adversarial_filename_ + ".rev_{}.wav".format(_rever_index_)
#         output_path = os.path.join(output_folder, rever_file)
#         wav_write(rever_adversarial_signal, output_path, sample_rate)
#         file_path_list.append(output_path)

#     # add beamforming
#     noise_signal = None
#     if adversarial_beamforming_number > 0:
#         _, noise_signal = wav_read(adversarial_beamforming_path, expected_sr=sample_rate)
#     for _beam_index_ in range(adversarial_beamforming_number):
#         beam_adversarial_signal = room_beamforming(sample_rate, adversarial_signal, noise_signal)
#         beam_file = _adversarial_filename_ + ".beam_{}.wav".format(_beam_index_)
#         output_path = os.path.join(output_folder, beam_file)
#         wav_write(beam_adversarial_signal, output_path, sample_rate)
#         file_path_list.append(output_path)

#     return file_path_list


# @exception_printer
# def gen_by_command(delta_db_list: np.ndarray, bandwidth: np.ndarray, command_analysis_file: str, clip_file: str, output_folder: str, adversarial_filename: str, params: dict) -> str:
#     """
#     generate adversarial samples only using command.
#     """
#     delta_db_list = np.maximum(delta_db_list, 0)
#     bandwidth = np.maximum(bandwidth, 0)

#     # load params
#     min_fre = params['min_fre']
#     max_fre = params['max_fre']
#     reserved_fre_gap_ratio = params['reserved_fre_gap_ratio']
#     frame_time = params['frame_time']
#     step_time = params['step_time']
#     margin_frame_num = params['margin_frame_num']
#     voiceless_process_type = params['voiceless_process_type']
#     voiceless_noise_fre_interval = params['voiceless_noise_fre_interval']
#     overflow_process_type = params['overflow_process_type']
#     filter_order = params['filter_order']
#     filter_window_func = params['filter_window_func']
#     special_word_enhance_dict = params.get("special_word_enhance_list")
#     _using_formant_shift_file_ = params['using_formant_shift_file']  # type: bool
#     _save_filt_wav_ = params['save_filt_wav']  # type: bool
#     _auto_balance_energy_ = params['auto_balance_energy']  # type: bool
#     _save_generation_param_ = params['save_generation_param']  # type: bool

#     # load analysis data
#     with open(command_analysis_file, 'rb') as _analysis_file_:
#         command_analysis_data = pickle.load(_analysis_file_)
#     command_formant_list = command_analysis_data['formant_list']
#     command_phone_list = command_analysis_data['phone_list']
#     command_phone_type_list = command_analysis_data['phone_type_list']
#     command_voiceless_energy_ratio_list = command_analysis_data['voiceless_energy_ratio_list']
#     command_word_index_list = command_analysis_data['word_index_list'] or []
#     command_signal = command_analysis_data['command_signal']
#     frame_num = command_phone_type_list.shape[0]

#     # apply formant shift
#     raw_command_formant_list = np.copy(command_formant_list)
#     if _using_formant_shift_file_:
#         pick_npz_file = clip_file.replace(".wav", PICK_NPZ_SUFFIX)
#         formant_shift_list = np.load(pick_npz_file, allow_pickle=True)['formant_shift_list']
#         command_formant_list += formant_shift_list
#     command_formant_list = np.clip(command_formant_list, min_fre, max_fre)

#     # apply enhancement for special words
#     if special_word_enhance_dict is not None:
#         command_special_enhance_coe = np.ones_like(delta_db_list)
#         for command_word_index in command_word_index_list:
#             enhance_coe = special_word_enhance_dict.get(command_word_index[0])
#             if enhance_coe is not None:
#                 command_special_enhance_coe[command_word_index[1]: command_word_index[2]] = enhance_coe
#         delta_db_list *= command_special_enhance_coe

#     # generate filter group
#     m_sample_rate, m_signal = wav_read(clip_file)
#     command_filter_list = generate_filter(command_formant_list, m_sample_rate, bandwidth, min_fre, max_fre, filter_order, reserved_fre_gap_ratio)

#     # signal index
#     frame_sample_num = int(frame_time * m_sample_rate)
#     step_sample_num = int(step_time * m_sample_rate)
#     _start_ = step_sample_num * margin_frame_num
#     _end_ = _start_ + (frame_num - 1) * step_sample_num + frame_sample_num

#     # calculate the auto-balance coefficient
#     if _auto_balance_energy_:
#         command_balance_coe = get_energy_balance_coe(m_signal[_start_: _end_], command_signal, m_sample_rate, command_filter_list, command_phone_type_list, frame_time, step_time, voiceless_process_type, filter_window_func)
#     else:
#         command_balance_coe = np.ones_like(bandwidth)

#     # filter out the music clip
#     command_filtered_signal = new_filter_signal(m_signal[_start_: _end_], m_sample_rate, command_filter_list, command_phone_list, command_phone_type_list, raw_command_formant_list, delta_db_list, command_balance_coe, command_voiceless_energy_ratio_list, frame_time, step_time, voiceless_process_type, voiceless_noise_fre_interval, filter_window_func)

#     # save the adversarial sample
#     adversarial_signal = m_signal
#     adversarial_signal[_start_:_end_] += command_filtered_signal
#     adversarial_signal = truncate_signal(adversarial_signal, overflow_process_type)
#     output_path = os.path.join(output_folder, adversarial_filename + ".wav")
#     wav_write(adversarial_signal, output_path, m_sample_rate)

#     # save the filtered sample
#     if _save_filt_wav_:
#         filtered_signal = np.zeros_like(m_signal)
#         filtered_signal[_start_: _end_] = command_filtered_signal
#         filtered_signal = truncate_signal(filtered_signal, overflow_process_type)
#         filter_wav_path = os.path.join(output_folder, adversarial_filename + ".filt.wav")
#         wav_write(filtered_signal, filter_wav_path, m_sample_rate)

#     # save the generate param
#     if _save_generation_param_:
#         output_param_path = os.path.join(output_folder, adversarial_filename + ".param.npz")
#         np.savez(output_param_path, delta_db_list=delta_db_list, bandwidth=bandwidth)

#     return output_path


# @exception_printer
# def gen_by_wakeup_command(delta_db_list: np.ndarray, bandwidth: np.ndarray, wakeup_analysis_file: str, command_analysis_file: str, clip_file: str, output_folder: str, adversarial_filename: str, params: dict) -> str:
#     """
#     generate adversarial samples using wakeup and command.
#     """
#     delta_db_list = np.maximum(delta_db_list, 0)
#     bandwidth = np.maximum(bandwidth, 0)

#     # load params
#     min_fre = params['min_fre']
#     max_fre = params['max_fre']
#     reserved_fre_gap_ratio = params['reserved_fre_gap_ratio']
#     frame_time = params['frame_time']
#     step_time = params['step_time']
#     margin_frame_num = params['margin_frame_num']
#     voiceless_process_type = params['voiceless_process_type']
#     voiceless_noise_fre_interval = params['voiceless_noise_fre_interval']
#     overflow_process_type = params['overflow_process_type']
#     filter_order = params['filter_order']
#     filter_window_func = params['filter_window_func']
#     wake_up_reinforce_ratio = params['wake_up_reinforce_ratio']  # type: float
#     special_word_enhance_dict = params.get("special_word_enhance_dict")
#     _using_formant_shift_file_ = params['using_formant_shift_file']  # type: bool
#     _save_filt_wav_ = params['save_filt_wav']  # type: bool
#     _auto_balance_energy_ = params['auto_balance_energy']  # type: bool
#     _save_generation_param_ = params['save_generation_param']  # type: bool

#     assert wake_up_reinforce_ratio > 0, "Illegal parameter 'wake_up_reinforce_ratio'."

#     # load wake_up analysis data
#     with open(wakeup_analysis_file, 'rb') as _analysis_file_:
#         wake_up_analysis_data = pickle.load(_analysis_file_)
#     wake_up_formant_list = wake_up_analysis_data['formant_list']
#     wake_up_phone_list = wake_up_analysis_data['phone_list']
#     wake_up_phone_type_list = wake_up_analysis_data['phone_type_list']
#     wake_up_voiceless_energy_ratio_list = wake_up_analysis_data['voiceless_energy_ratio_list']
#     wake_up_word_index_list = wake_up_analysis_data['word_index_list'] or []
#     wake_up_signal = wake_up_analysis_data['command_signal']
#     wake_up_frame_num = wake_up_phone_type_list.shape[0]

#     # load command analysis data
#     with open(command_analysis_file, 'rb') as _analysis_file_:
#         command_analysis_data = pickle.load(_analysis_file_)
#     command_formant_list = command_analysis_data['formant_list']
#     command_phone_list = command_analysis_data['phone_list']
#     command_phone_type_list = command_analysis_data['phone_type_list']
#     command_voiceless_energy_ratio_list = command_analysis_data['voiceless_energy_ratio_list']
#     command_word_index_list = command_analysis_data['word_index_list'] or []
#     command_signal = command_analysis_data['command_signal']
#     command_frame_num = command_phone_type_list.shape[0]

#     # apply formant shift
#     raw_wake_up_formant_list = np.copy(wake_up_formant_list)
#     raw_command_formant_list = np.copy(command_formant_list)
#     if _using_formant_shift_file_:
#         pick_npz_file = clip_file.replace(".wav", PICK_NPZ_SUFFIX)
#         formant_shift_data = np.load(pick_npz_file, allow_pickle=True)
#         wake_up_formant_shift_list = formant_shift_data['wake_up_formant_shift_list']
#         command_formant_shift_list = formant_shift_data['command_formant_shift_list']
#         wake_up_formant_list += wake_up_formant_shift_list
#         command_formant_list += command_formant_shift_list
#     wake_up_formant_list = np.clip(wake_up_formant_list, min_fre, max_fre)
#     command_formant_list = np.clip(command_formant_list, min_fre, max_fre)

#     # reshape the params
#     wake_up_bandwidth_list = bandwidth[:wake_up_frame_num]
#     wake_up_delta_db_list = delta_db_list[:wake_up_frame_num]
#     command_bandwidth_list = bandwidth[wake_up_frame_num:]
#     command_delta_db_list = delta_db_list[wake_up_frame_num:]

#     # apply enhancement for special words
#     if special_word_enhance_dict is not None:
#         wake_up_special_enhance_coe = np.ones_like(wake_up_delta_db_list)
#         command_special_enhance_coe = np.ones_like(command_delta_db_list)
#         for wake_up_word_index in wake_up_word_index_list:
#             enhance_coe = special_word_enhance_dict.get(wake_up_word_index[0])
#             if enhance_coe is not None:
#                 wake_up_special_enhance_coe[wake_up_word_index[1]: wake_up_word_index[2]] = enhance_coe
#         for command_word_index in command_word_index_list:
#             enhance_coe = special_word_enhance_dict.get(command_word_index[0])
#             if enhance_coe is not None:
#                 command_special_enhance_coe[command_word_index[1]: command_word_index[2]] = enhance_coe
#         wake_up_delta_db_list *= wake_up_special_enhance_coe
#         command_delta_db_list *= command_special_enhance_coe

#     # generate filter group
#     sample_rate, m_signal = wav_read(clip_file)
#     wakeup_filter_list = generate_filter(wake_up_formant_list, sample_rate, wake_up_bandwidth_list, min_fre, max_fre, filter_order, reserved_fre_gap_ratio)
#     command_filter_list = generate_filter(command_formant_list, sample_rate, command_bandwidth_list, min_fre, max_fre, filter_order, reserved_fre_gap_ratio)

#     # signal index
#     frame_sample_num = int(frame_time * sample_rate)
#     step_sample_num = int(step_time * sample_rate)
#     _wakeup_start_ = margin_frame_num * step_sample_num
#     _wakeup_end_ = _wakeup_start_ + (wake_up_frame_num - 1) * step_sample_num + frame_sample_num
#     _command_end_ = len(m_signal) - (margin_frame_num * step_sample_num)
#     _command_start_ = _command_end_ - ((command_frame_num - 1) * step_sample_num + frame_sample_num)

#     # calculate the auto-balance coefficient
#     if _auto_balance_energy_:
#         wakeup_balance_coe = get_energy_balance_coe(m_signal[_wakeup_start_: _wakeup_end_], wake_up_signal, sample_rate, wakeup_filter_list, wake_up_phone_type_list, frame_time, step_time, voiceless_process_type, filter_window_func)
#         command_balance_coe = get_energy_balance_coe(m_signal[_command_start_: _command_end_], command_signal, sample_rate, command_filter_list, command_phone_type_list, frame_time, step_time, voiceless_process_type, filter_window_func)
#     else:
#         wakeup_balance_coe = np.ones_like(wake_up_bandwidth_list)
#         command_balance_coe = np.ones_like(command_bandwidth_list)
#     wakeup_balance_coe = wakeup_balance_coe * wake_up_reinforce_ratio

#     # filter out music clip
#     wake_up_filtered_signal = new_filter_signal(m_signal[_wakeup_start_: _wakeup_end_], sample_rate, wakeup_filter_list, wake_up_phone_list, wake_up_phone_type_list, raw_wake_up_formant_list, wake_up_delta_db_list, wakeup_balance_coe, wake_up_voiceless_energy_ratio_list, frame_time, step_time, voiceless_process_type, voiceless_noise_fre_interval, filter_window_func)
#     command_filtered_signal = new_filter_signal(m_signal[_command_start_: _command_end_], sample_rate, command_filter_list, command_phone_list, command_phone_type_list, raw_command_formant_list, command_delta_db_list, command_balance_coe, command_voiceless_energy_ratio_list, frame_time, step_time, voiceless_process_type, voiceless_noise_fre_interval, filter_window_func)

#     # save the adversarial sample
#     adversarial_signal = m_signal
#     adversarial_signal[_wakeup_start_: _wakeup_end_] += wake_up_filtered_signal
#     adversarial_signal[_command_start_: _command_end_] += command_filtered_signal
#     adversarial_signal = truncate_signal(adversarial_signal, overflow_process_type)
#     output_wav_path = os.path.join(output_folder, adversarial_filename + ".wav")
#     wav_write(adversarial_signal, output_wav_path, sample_rate)

#     # save the filtered sample
#     if _save_filt_wav_:
#         filtered_signal = np.zeros_like(m_signal)
#         filtered_signal[_wakeup_start_: _wakeup_end_] = wake_up_filtered_signal
#         filtered_signal[_command_start_: _command_end_] = command_filtered_signal
#         filtered_signal = truncate_signal(filtered_signal, overflow_process_type)
#         filter_wav_path = os.path.join(output_folder, adversarial_filename + ".filt.wav")
#         wav_write(filtered_signal, filter_wav_path, sample_rate)

#     # save the generate param
#     if _save_generation_param_:
#         output_param_path = os.path.join(output_folder, adversarial_filename + ".param.npz")
#         np.savez(output_param_path, delta_db_list=delta_db_list, bandwidth=bandwidth)

#     return output_wav_path


# def get_filtered_signal(clip_signal: np.ndarray, analysis_data, formant_shift_list: Optional[np.ndarray], params: dict) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     return function to generate adversarial samples.
#     """
#     frame_time = params['frame_time']
#     step_time = params['step_time']
#     delta_db = params['delta_db']
#     bandwidth = params['bandwidth']
#     formant_weight = params['formant_weight']
#     reserved_ratio = params['reserved_ratio']
#     filter_order = params['filter_order']
#     min_fre, max_fre = params['fre_interval']
#     formant_list = analysis_data['formant_list']
#     phone_list = analysis_data['phone_list']
#     phone_type_list = analysis_data['phone_type_list']
#     sample_rate = analysis_data['sample_rate']
#     command_signal = analysis_data['command_signal']
#     voiceless_process_type = params['voiceless_process_type']
#     voiceless_fre_interval = params['voiceless_fre_interval']
#     voiceless_energy_ratio_list = analysis_data['voiceless_energy_ratio_list']
#     window_func = params['window_func']
#     _using_formant_shift_ = params['using_formant_shift']
#     _auto_balance_coe_ = params['auto_balance_coe']
#     frame_num = formant_list.shape[0]

#     if _using_formant_shift_ and formant_shift_list is not None:
#         formant_list += formant_shift_list
#     formant_list = np.clip(formant_list, min_fre, max_fre)
#     delta_db_list = np.ones((frame_num, MAX_FORMANT_NUM), dtype=np.float) * delta_db * formant_weight
#     bandwidth_list = np.ones((frame_num, MAX_FORMANT_NUM), dtype=np.float) * bandwidth

#     # generate filter list
#     filter_list = generate_filter(formant_list, sample_rate, bandwidth_list, min_fre, max_fre, filter_order, reserved_ratio)
#     # get balance coe
#     balance_coe_list = get_energy_balance_coe(clip_signal, command_signal, sample_rate, filter_list, phone_type_list, frame_time, step_time, voiceless_process_type, window_func)
#     # filter signal
#     filtered_signal = new_filter_signal(clip_signal, sample_rate, filter_list, phone_list, phone_type_list, formant_list, delta_db_list, balance_coe_list, voiceless_energy_ratio_list, frame_time, step_time, voiceless_process_type, voiceless_fre_interval, window_func)

#     return filtered_signal, balance_coe_list


# def align_formant(clip_voice_formant_list: np.ndarray, command_voice_formant_list: np.ndarray, command_phone_type_list: np.ndarray, params: dict) -> np.ndarray:
#     max_up_shift_fre = params['max_up_shift_fre']
#     max_down_shift_fre = params['max_down_shift_fre']
#     formant_num = MAX_FORMANT_NUM

#     # reshape the shift fre
#     if isinstance(max_up_shift_fre, int) or isinstance(max_up_shift_fre, float):
#         max_up_shift_fre = [max_up_shift_fre]
#     max_up_shift_fre = np.array(max_up_shift_fre)
#     assert len(max_up_shift_fre.shape) == 1
#     if max_up_shift_fre.shape[0] == 1:
#         max_up_shift_fre = np.tile(max_up_shift_fre, (formant_num,))
#     elif max_up_shift_fre.shape[0] >= formant_num:
#         max_up_shift_fre = max_up_shift_fre[:formant_num]
#     else:
#         logger.error(u"Wrong shape of max_up_shift_fre.")
#         exit(-1)
#     assert np.all(max_up_shift_fre >= 0)
#     if isinstance(max_down_shift_fre, int) or isinstance(max_down_shift_fre, float):
#         max_down_shift_fre = [max_down_shift_fre]
#     max_down_shift_fre = np.array(max_down_shift_fre)
#     assert len(max_down_shift_fre.shape) == 1
#     if max_down_shift_fre.shape[0] == 1:
#         max_down_shift_fre = np.tile(max_down_shift_fre, (formant_num,))
#     elif max_down_shift_fre.shape[0] >= formant_num:
#         max_down_shift_fre = max_down_shift_fre[:formant_num]
#     else:
#         logger.error(u"Wrong shape of max_down_shift_fre.")
#     assert np.all(max_down_shift_fre >= 0)
#     max_down_shift_fre = -max_down_shift_fre

#     frame_num = command_voice_formant_list.shape[0]
#     TMP_MAX_DISTANCE = 999999999
#     formant_shift_list = np.zeros_like(command_voice_formant_list, dtype=np.float)
#     for _frame_index_ in range(frame_num):
#         if command_phone_type_list[_frame_index_] in [PhoneType.Sil, PhoneType.Voiceless]:
#             continue
#         if np.all(np.isnan(clip_voice_formant_list[_frame_index_])):  # 均为Nan，此处没有共振峰，无法对齐
#             continue

#         for _formant_index_ in range(formant_num):
#             if np.isnan(command_voice_formant_list[_frame_index_, _formant_index_]):  # Nan不需要嵌入共振峰
#                 continue
#             cur_formant_gap = clip_voice_formant_list[_frame_index_, _formant_index_] - command_voice_formant_list[_frame_index_, _formant_index_]
#             pre_formant_gap = np.NaN
#             if _formant_index_ > 0:
#                 pre_formant_gap = clip_voice_formant_list[_frame_index_, _formant_index_ - 1] - command_voice_formant_list[_frame_index_, _formant_index_]
#             next_formant_gap = np.NaN
#             if _formant_index_ < MAX_FORMANT_NUM - 1:
#                 next_formant_gap = clip_voice_formant_list[_frame_index_, _formant_index_ + 1] - command_voice_formant_list[_frame_index_, _formant_index_]
#             formant_gap_list = [cur_formant_gap, pre_formant_gap, next_formant_gap]
#             formant_index_list = [_formant_index_, _formant_index_ - 1, _formant_index_ + 1]
#             distance = TMP_MAX_DISTANCE
#             for _i_ in range(len(formant_index_list)):  # 通过小范围平移对齐
#                 formant_gap = formant_gap_list[_i_]
#                 if not np.isnan(formant_gap) and max_down_shift_fre[_formant_index_] <= formant_gap <= max_up_shift_fre[_formant_index_]:
#                     new_distance = abs(
#                         fre2bark(clip_voice_formant_list[_frame_index_, formant_index_list[_i_]]) - fre2bark(command_voice_formant_list[_frame_index_, _formant_index_])
#                     )
#                     if new_distance < distance:
#                         formant_shift_list[_frame_index_, _formant_index_] = formant_gap
#                         distance = new_distance
#     return formant_shift_list
