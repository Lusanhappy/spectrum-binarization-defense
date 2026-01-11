# coding=utf-8
import glob
from formant_processor import *
import shutil


def mix_analysis(music_path, command_path, output_folder):
    # extract the information
    music_name = os.path.splitext(os.path.basename(music_path))[0]
    command_name = os.path.splitext(os.path.basename(command_path))[0]
    output_folder = os.path.join(output_folder, "{}_{}".format(music_name, command_name), "mix")

    # load the wave with normalization
    m_sample_rate, m_signal = wav_read(music_path)
    c_sample_rate, c_signal = wav_read(command_path)
    if m_sample_rate != c_sample_rate:
        logger.error("Un-equal sample rate.")
        exit(-1)
    shutil.rmtree(output_folder, ignore_errors=True)
    os.makedirs(output_folder, exist_ok=True)

    # calculate the mean power
    m_mean_power = 10 * np.log10(np.mean(m_signal ** 2))
    c_mean_power = 10 * np.log10(np.mean(c_signal ** 2))
    print("Music name: {}, {:.2f}dB".format(music_name, m_mean_power))
    print("Command Name: {}, {:.2f}dB".format(command_name, c_mean_power))

    for _ratio_ in range(5, 31, 5):
        _ratio_ = _ratio_ / 10.0
        t_signal = c_signal * _ratio_
        t_mean_power = 10 * np.log10(np.mean(t_signal ** 2))
        print("-- {:.2f}dB.".format(t_mean_power))
        for _index_ in range(min(len(t_signal), len(m_signal))):
            t_signal[_index_] += m_signal[_index_]
        wav_write(t_signal / np.max(np.abs(t_signal)) * 32767, os.path.join(output_folder, "ratio_{}.wav".format(_ratio_)), c_sample_rate)


def enhance_analysis(command_path, bandwidth: Union[int, list], output_folder, enhance_value, frame_time=0.025, step_time=0.01, formant_num: int = MAX_FORMANT_NUM):
    """
    增强白噪声共振峰周围能量, 观察音频解码结果
    """
    print("Enhance Analysis: ", command_path)
    bandwidth = reshape_single_dimension(bandwidth)
    half_bandwidth = bandwidth / 2.0
    assert np.all(bandwidth >= 0)
    assert step_time <= frame_time

    # load the wav
    sample_rate, command_signal = wav_read(command_path)
    raw_signal = white_noise(len(command_signal), sample_rate, 0, 5500)
    raw_signal = raw_signal * 500

    # compute the formant list
    formants = get_formants(command_path, frame_time=frame_time, step_time=step_time, formant_num=formant_num)

    # compute the spectrum
    frame_len = int(frame_time * sample_rate)
    step_len = int(step_time * sample_rate)
    spectrum = cal_spectrum(raw_signal, frame_len, step_len)
    freq_axis = np.fft.rfftfreq(frame_len, 1.0 / sample_rate)
    frame_num = spectrum.shape[0]

    # enhance the energy around the formant
    enhance_spectrum = np.copy(spectrum)
    for _frame_index_ in range(frame_num):
        for _formant_index_ in range(formant_num):
            formant = formants[_frame_index_][_formant_index_]
            if np.isnan(formant) or bandwidth[_formant_index_] == 0:
                continue
            min_fre = formant - half_bandwidth[_formant_index_]
            max_fre = formant + half_bandwidth[_formant_index_]
            in_bound_index = np.where(
                np.logical_and(
                    freq_axis >= min_fre, freq_axis <= max_fre
                )
            )[0]
            enhance_spectrum[_frame_index_][in_bound_index] = spectrum[_frame_index_][in_bound_index] * enhance_value

    # irfft
    enhance_signal = np.zeros_like(raw_signal)
    normalize_factor = np.zeros_like(raw_signal)
    irfft_signal = np.fft.irfft(enhance_spectrum)
    for _frame_index_ in range(frame_num):
        _start_, _end_ = step_len * _frame_index_, step_len * _frame_index_ + frame_len
        normalize_factor[_start_: _end_] += 1
        enhance_signal[_start_: _end_] += irfft_signal[_frame_index_]
    normalize_factor = np.where(normalize_factor == 0, 1, normalize_factor)
    enhance_signal /= normalize_factor

    # save the signal
    command_name = os.path.splitext(os.path.basename(command_path))[0]
    bandwidth_str = "_".join([str(_band_) for _band_ in bandwidth])
    output_file_path = os.path.join(output_folder, "{}_{}_noise.wav".format(command_name, bandwidth_str))
    wav_write(enhance_signal, output_file_path, sample_rate)


def delete_analysis(command_path, bandwidth: Union[int, list], output_folder, frame_time=0.025, step_time=0.01, formant_num: int = MAX_FORMANT_NUM):
    """
    删除共振峰周围能量, 删除其他能量, 观察音频解码结果和保留能量的比例
    """
    print("Delete Analysis: ", command_path)
    bandwidth = reshape_single_dimension(bandwidth)
    half_bandwidth = bandwidth / 2.0
    assert np.all(bandwidth >= 0)
    assert step_time <= frame_time

    # load the wav
    sample_rate, command_signal = wav_read(command_path)

    # compute the energy
    mean_energy = np.mean(command_signal ** 2)

    # compute the formant list
    formants = get_formants(command_path, frame_time=frame_time, step_time=step_time, formant_num=formant_num)

    # compute the spectrum
    frame_len = int(frame_time * sample_rate)
    step_len = int(step_time * sample_rate)
    spectrum = cal_spectrum(command_signal, frame_len, step_len)
    freq_axis = np.fft.rfftfreq(frame_len, 1.0 / sample_rate)
    frame_num = spectrum.shape[0]

    # filter the energy around the formant
    reserved_spectrum = np.copy(spectrum)
    for _frame_index_ in range(frame_num):
        for _formant_index_ in range(formant_num):
            formant = formants[_frame_index_][_formant_index_]
            if np.isnan(formant) or half_bandwidth[_formant_index_] == 0:
                continue
            min_fre = formant - half_bandwidth[_formant_index_]
            max_fre = formant + half_bandwidth[_formant_index_]
            in_bound_index = np.where(
                np.logical_and(
                    freq_axis >= min_fre, freq_axis <= max_fre
                )
            )[0]
            reserved_spectrum[_frame_index_][in_bound_index] = 0

    # irfft
    reserved_signal = np.zeros_like(command_signal)
    normalize_factor = np.zeros_like(command_signal)
    irfft_signal = np.fft.irfft(reserved_spectrum)
    for _frame_index_ in range(frame_num):
        _start_, _end_ = step_len * _frame_index_, step_len * _frame_index_ + frame_len
        normalize_factor[_start_: _end_] += 1
        reserved_signal[_start_: _end_] += irfft_signal[_frame_index_]
    normalize_factor = np.where(normalize_factor == 0, 1, normalize_factor)
    reserved_signal /= normalize_factor

    # compute the energy and the reserved ratio
    reserved_mean_energy = np.mean(reserved_signal ** 2)
    reserved_ratio = np.sqrt(reserved_mean_energy / mean_energy)

    # save the signal
    command_name = os.path.splitext(os.path.basename(command_path))[0]
    bandwidth_str = "_".join([str(_band_) for _band_ in bandwidth])
    output_file_path = os.path.join(output_folder, "{}_{}_{:.2f}.wav".format(command_name, bandwidth_str, reserved_ratio))
    print("reserved ratio: ", reserved_ratio)
    wav_write(reserved_signal, output_file_path, sample_rate)


def reserve_analysis(command_path, bandwidth: Union[int, list], output_folder, frame_time=0.025, step_time=0.01, formant_num: int = MAX_FORMANT_NUM, using_noise=False):
    """
    保留共振峰周围能量, 删除其他能量, 观察音频解码结果和保留能量的比例
    """
    print("Reserve Analysis: ", command_path)
    bandwidth = reshape_single_dimension(bandwidth)
    half_bandwidth = bandwidth / 2.0
    assert np.all(bandwidth >= 0)
    assert step_time <= frame_time

    # load the wav
    sample_rate, command_signal = wav_read(command_path)
    raw_signal = command_signal
    if using_noise:
        raw_signal = white_noise(len(command_signal), sample_rate, 0, 5500)
        raw_signal = raw_signal * 2000

    # compute the energy
    mean_energy = np.mean(command_signal ** 2)

    # compute the formant list
    formants = get_formants(command_path, frame_time=frame_time, step_time=step_time, formant_num=formant_num)

    # compute spectrum
    frame_len = int(frame_time * sample_rate)
    step_len = int(step_time * sample_rate)
    spectrum = cal_spectrum(raw_signal, frame_len, step_len)
    freq_axis = np.fft.rfftfreq(frame_len, 1.0 / sample_rate)
    frame_num = spectrum.shape[0]

    # filter the energy not around the formant
    reserved_spectrum = np.zeros_like(spectrum)
    for _frame_index_ in range(frame_num):
        for _formant_index_ in range(formant_num):
            formant = formants[_frame_index_][_formant_index_]
            if np.isnan(formant) or bandwidth[_formant_index_] == 0:
                continue
            min_fre = formant - half_bandwidth[_formant_index_]
            max_fre = formant + half_bandwidth[_formant_index_]
            in_bound_index = np.where(
                np.logical_and(
                    freq_axis >= min_fre, freq_axis <= max_fre
                )
            )[0]
            reserved_spectrum[_frame_index_][in_bound_index] = spectrum[_frame_index_][in_bound_index]

    # irfft
    reserved_signal = np.zeros_like(raw_signal)
    normalize_factor = np.zeros_like(raw_signal)
    irfft_signal = np.fft.irfft(reserved_spectrum)
    for _frame_index_ in range(frame_num):
        _start_, _end_ = step_len * _frame_index_, step_len * _frame_index_ + frame_len
        normalize_factor[_start_: _end_] += 1
        reserved_signal[_start_: _end_] += irfft_signal[_frame_index_]
    normalize_factor = np.where(normalize_factor == 0, 1, normalize_factor)
    reserved_signal /= normalize_factor

    # compute the energy and the reserved ratio
    reserved_mean_energy = np.mean(reserved_signal ** 2)
    reserved_ratio = np.sqrt(reserved_mean_energy / mean_energy)

    # save the signal
    command_name = os.path.splitext(os.path.basename(command_path))[0]
    bandwidth_str = "_".join([str(_band_) for _band_ in bandwidth])
    if using_noise:
        output_file_path = os.path.join(output_folder, "{}_{}_noise.wav".format(command_name, bandwidth_str))
    else:
        output_file_path = os.path.join(output_folder, "{}_{}_{:.2f}.wav".format(command_name, bandwidth_str, reserved_ratio))
        print("reserved ratio: ", reserved_ratio)
    wav_write(reserved_signal, output_file_path, sample_rate)


def formant_analysis():
    frame_time = 0.025
    step_time = 0.01
    bandwidth = [300, 300, 300, 350, 350]
    command_folder = "./command/0727_GoogleCommand"
    # output_folder = "./output/formant_analysis/delete/6"
    # output_folder = "./output/formant_analysis/reserve/1"
    output_folder = "./output/formant_analysis/enhance/1"

    os.makedirs(output_folder, exist_ok=True)
    for _command_file_ in glob.glob(os.path.join(command_folder, "*.wav")):
        # delete_analysis(_command_file_, bandwidth, output_folder, frame_time, step_time)
        # reserve_analysis(_command_file_, bandwidth, output_folder, frame_time, step_time, using_noise=True)
        enhance_analysis(_command_file_, bandwidth, output_folder, 6, frame_time, step_time)

        print("-" * 20 + "\n")


if __name__ == '__main__':
    formant_analysis()
