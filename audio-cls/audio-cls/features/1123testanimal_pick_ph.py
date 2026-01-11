# coding=utf-8
import shutil
import copy
import absl.app
from utils import *
from picker_util import pick_music, __default_pick_params__ as music_pick_default_param
from submodule.deepspeech.training.deepspeech_training.util.config import initialize_globals
from deepspeech_ctc_score import create_flags


def main(_):
    initialize_globals()
    mute_third_party_logging()
    # Using ``` default_params[<param_name>] = <param_value> ``` to set the params.
    # And view the usage of each parameter in the '__default_pick_params__' definition.
    params = copy.deepcopy(music_pick_default_param)
    
    params['redecode'] = False
    
    # specify basic setting here
    params['task_name'] = "0211-whatwouldyourecommandV3-1124musicwithoutlyrics-ph"
    params['command_folder'] = "./command/wake-up-command-task/scale/whatwouldyourecommandV3"
    #params['music_folder'] = "./music/2021music16Kv1hour"
    params['music_folder'] = "./music/1124musicwithoutlyrics"
    params['max_workers'] = 8
    params['padding_frame_num'] = 0
    params['margin_frame_num'] = 100
    params['step_frame_num'] = 50
    # specify wake-up setting here
    params['using_wake_up'] = False
    params['wake_up_folder'] = "./command/wake-up-command-task/scale/Alexa_V3"
    params['min_pause_time'] = 0.5
    params['max_pause_time'] = 0.6
    params['pause_step_time'] = 0.1
    # specify voice-phoneme pronunciation alignment setting here
    params['do_pronunciation_filter'] = True
    params['matched_phone_num_threshold'] = 3
    # specify spleeter setting here
    params['using_spleeter'] = True
    # specify pick setting here
    params['allow_overlap'] = False
    params['pick_top_k'] = -1
    
    params['do_sil_phoneme_filter'] = False
    params['allowed_sil_phone_ratio'] = 0.4

    params['do_intensity_filter'] = True
    params['intensity_scale_threshold'] = 60
    params['intensity_std_threshold'] = 1
    params['intensity_window_frame_num'] = 30
    
    params['do_formant_filter'] = True
    params['formant_threshold'] = 35  # Specify the flatness threshold of formant list. # The bigger the better.
    params['formant_window_frame_num'] = 30  # Specify the calculation window of formant standard deviation.

    #params['do_formant_similarity_filter'] = False
    params['do_pitch_filter'] = True
    params['pitch_threshold'] = 1
    params['pitch_window_frame_num'] = 30
    
    params['do_formant_similarity_filter'] = True  # Whether do filter clips by formant similarity.
    params['formant_similarity_weight'] = [1.0, 1.0, 0, 0, 0]  # Which only care about the 2/3-order formant.
    params['max_up_shift_fre'] = [200, 200, 200, 200, 200]  # Control the maximum up-shift frequency. It supports int/float number or one-dimension array with shape (MAX_FORMANT+NUM, ).
    params['max_down_shift_fre'] = [200, 200, 200, 200, 200]  # Control the maximum down-shift frequency. It supports int/float number or one-dimension array with shape (MAX_FORMANT_NUM, ).
    params['formant_similarity_threshold'] = 0.98  # The smaller the better
   
    params['do_flatness_filter'] = False  # whether do filter clips by formant flatness analysis. A flat formant means that there are not enough phonemes in clip.
    params['flatness_threshold'] = 0.7 # Specify the flatness threshold of formant on the time dimension. If the flatness value is greater than 'flatness_threshold', then the clip      will be deprecated. The smaller the better.
    params['flatness_window_frame_num'] = 30  # Specify the calculation window of flatness.

    params['do_psychological_filter'] = True
    params['formant_weight']= [1.0, 1.0, 0.0, 0.0, 0.0]
    params['psychological_threshold'] = 10

    params['using_ctc_score'] = False  # Whether using deepspeech ctc score to filter the clips.
    params['ctc_reserved_ratio'] = 0.3  # How many clips would be reserved after filtering clips using ctc score.


    task_folder = get_task_folder(params['task_name'])
    if os.path.exists(task_folder):
        logger.warning(u"The destination folder '{}' will be removed. Please backup this folder, and then enter 'Y' to continue, or others to exit...".format(task_folder))
        content = input()
        if content.upper().strip() == 'Y':
            shutil.rmtree(task_folder, ignore_errors=True)
            logger.info(u"The destination folder '{}' was removed.".format(task_folder))
        else:
            exit(0)
    pick_music(params)


if __name__ == '__main__':
    # mute_third_party_logging()
    # # Using ``` default_params[<param_name>] = <param_value> ``` to set the params.
    # # And view the usage of each parameter in the '__default_pick_params__' definition.
    # params = copy.deepcopy(music_pick_default_param)
    # # specify basic setting here
    # params['task_name'] = "20210603-100song-0027makeitwarmer-gap1p0"
    # params['command_folder'] = "./command/wake-up-command-task/scale/0027makeitwarmer"
    # params['music_folder'] = "./music/20210426music/100song"
    # params['max_workers'] = 20
    # params['padding_frame_num'] = 1000
    # params['margin_frame_num'] = 1
    # params['step_frame_num'] = 20
    # # specify wake-up setting here
    # params['using_wake_up'] = False
    # params['wake_up_folder'] = "./command/wake-up-command-task/scale/heygoogle_ibm9"
    # params['min_pause_time'] = 1.0
    # params['max_pause_time'] = 1.1
    # params['pause_step_time'] = 0.1
    # # specify voice-phoneme pronunciation alignment setting here
    # params['matched_phone_num_threshold'] = 4
    # # specify spleeter setting here
    # params['using_spleeter'] = True
    # # specify pick setting here
    # params['allow_overlap'] = False
    # params['pick_top_k'] = -1
    #
    # task_folder = get_task_folder(params['task_name'])
    # if os.path.exists(task_folder):
    #     logger.warning(u"The destination folder '{}' will be removed. Please backup this folder, and then enter 'Y' to continue, or others to exit...".format(task_folder))
    #     content = input()
    #     if content.upper().strip() == 'Y':
    #         shutil.rmtree(task_folder, ignore_errors=True)
    #         logger.info(u"The destination folder '{}' was removed.".format(task_folder))
    #     else:
    #         exit(0)
    # pick_music(params)


    create_flags()
    absl.app.run(main)
