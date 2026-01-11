from dataclasses import dataclass

@dataclass
class DefaultModelConfig:
    # Select model
    model_name: str = 'cnn3_dnn2'       # 'cnn3_dnn2', 'cnn_attention'
    # model_name: str = 'cnn_attention'   # 'cnn3_dnn2', 'cnn_attention'

    sr: int = 16000

    # 'cnn_attention' model config
    nb_samp: int = 64600                # max number of sampling points  
    first_conv: int = 256               # no. of filter coefficients 
    in_channels: int = 1
    filts: tuple = (20, (20, 20), (20, 128), (128, 128)) # no. of filters channel in residual blocks
    blocks: tuple = (2, 4)
    nb_fc_node: int = 256
    gru_node: int = 256
    nb_gru_layer: int = 3
    nb_classes: int = 2

    # 'cnn3_dnn2' model config
    duration: int = 1
    nmels: int = 128
    winlen: int = 640
    hoplen: int = 320
    nfft: int = 640
    formants_num: int = 2
    log_mel: bool = True
    dual_feats: bool = False    # Use both feats and their 1st diff when using 1st diff
    use_melspec: bool = False   # Feat1: Extract mel spectrogram as input
    melspec_diff: bool = False  #        use 1st difference
    use_formants: bool = False  # Feat2: Extract formants as input
    formants_diff: bool = True  #        use 1st difference
    use_pitch: bool = False     # Feat3: Extract pitch as input
    pitch_diff: bool = True     #        use 1st difference
    # Binary features
    use_binary_features: bool = True           # Feat: Binary linear spectrogram
    thresholds: tuple = (0.6, 0.65, 0.7, 0.75) #       with different global thresholds
    # thresholds: tuple = (0.75,) #       with different global thresholds

@dataclass
class DefaultTrainConfig:
    debug_info: bool = True
    random_seed: int = 999

    # Dataset opt1: data_path/[adv/clean]/[train/val]
    data_path: str = None
    # Dataset opt2: if data_path=None, the following works
    # -1 for all
    train_adv_dirs: tuple = (
        # ('datasets/adv/aspire-train', 5000),               # 27433
        ('datasets/adv/datatang-train', 4000),             # 33415
        # ('/home/weica/cloud_decode/aliyun-success/datatang', 4000),
        ('datasets/adv/wav2vec2-nobandlimit-train', 500),  # 9853
        ('datasets/adv/wav2vec2-bandlimit-train', 500),    # 3706
        )
    train_cln_dirs: tuple = (
        ('datasets/clean/music-train/', 6000),            # 28008
        ('datasets/clean/speech/speech-train', 1000),
        )
    val_adv_dirs: tuple = (
        ('datasets/adv/datatang-val', 1000),               # 3670
        ('datasets/adv/wav2vec2-oppo-test', 1000),
        )
    val_cln_dirs: tuple = (
        ('datasets/clean/music-infer/music-val', 2000),                # 2832
        )

    # Dataset sampling setting
    no_val: bool = False                 # train/val all for training
    train_oversample: bool = False       # Oversample when class imbalance in train split
    train_downsample: bool = False        # Downsample when class imbalance in train split
    val_downsample: bool = True          # Downsample when class imbalance in val split
    
    lr: float = .0001
    weight_decay: int = 1e-2
    
    epoch: int = 40
    save_interval: int = 10              # every n epochs, save weight
    save_better: bool = True             # better than before, save weight
    save_valacc_bound: float = 0.75      # and val-acc > bound, save weight
    batch: int = 64

    # Keep features constant (formants and pitch)
    static_features: bool = True         # Use static features regardless rand noise to save time

    # Adversarial sample preprocess. label = 1
    sample_the_middle: bool = False      # Sample the middle of the wave
    skip_min: float = .0                 # Min skip offset (sec).
    random_skip_max: float = .1          # Max offset time (sec).
    # Noise amplitude
    random_noise_amp: float = .00        # Noise maximum amplitude, in [0, 1].
    clean_noise_only: bool = True        # Only add noise to clean class samples

    # Training setup args
    save_path: str = 'logs'              # Save directory path
    restore_from: str = None             # Initialize model with restore_from.pth

@dataclass
class DefaultTestConfig:
    log_path: str = 'test_results.txt'   # Save file path

    # 'cnn3_dnn2' model config
    length_limit: float = 4              # Audio length limit
    step: float = 1                      # Window step length.
    padding: bool = False                # Pad the end.

    # Test data
    test_adv_dirs: tuple = (
        # 'datasets/adv/aspire-test',                  # 4479
        # 'datasets/adv/datatang-test',                # 14438
        # 'datasets/adv/wav2vec2-oppo-test',           # 1060
        # 'datasets/adv/wav2vec2-bandlimit-test',      # 2624
        # 'datasets/adv/wav2vec2-nobandlimit-test',    # 8339
        'selected_AEs_for_defense_total/devilswhisper',
        'selected_AEs_for_defense_total/selected_aspire_AEs_google',
        'selected_AEs_for_defense_total/occam_aliyun',
        'test_dataset_selected_deepspeech_AEs_azure',
        'test_dataset_selected_aspire_AEs_google',
        '/home/weica/cloud_decode/aliyun-success/wav2vec2',
        '/home/weica/cloud_decode/aliyun-success/datatang',
        )
    test_cln_dirs: tuple = (
        # 'datasets/clean/music-val',                 # 2000
        # 'datasets/clean/music-test',                 # 2832
        'datasets/clean/music-infer/music-val',
        'datasets/clean/music-infer/music-test',
        # 'datasets/clean/speech-test',                # 5000
        'datasets/clean/speech/speech-test',
        # 'datasets/clean/noise1-test',                # 775
        # 'datasets/clean/noise2-test',                # 743
        )
    # Test sample number limit (for save time)
    # test_folder_limit: int = -1
    test_folder_limit: int = 5000
    random_seed: int = 999


@dataclass
class DefaultAllConfig:
    model: DefaultModelConfig = DefaultModelConfig()
    train: DefaultTrainConfig = DefaultTrainConfig()
    test: DefaultTestConfig = DefaultTestConfig()

model_config: DefaultModelConfig = DefaultModelConfig()
train_config: DefaultTrainConfig = DefaultTrainConfig()
test_config: DefaultTestConfig = DefaultTestConfig()
all_config: DefaultAllConfig = DefaultAllConfig(
    model=model_config, train=train_config, test=test_config)

