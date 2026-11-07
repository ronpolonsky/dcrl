"""Config for RTCLearner: pi0.5 BC with action-prefix conditioning (RTC-SFT)."""

from configs.model import expo_ft_pi_config


def get_config():
    config = expo_ft_pi_config.get_config()

    config.model_cls = "RTCLearner"

    # Use the same pi05 backbone as the SFT / BC setups.
    config.pi05_config_name = "expo_pi05_droid_lora_finetune_sft_cartesian_state"
    config.pi05_weight_loader_path = ""

    # Offline training: do not freeze the pi05 encoder.
    config.freeze_pi05_encoder = False

    # Training-time RTC: train with clean action prefixes and postfix loss.
    config.p1_use_prefix_conditioning = True

    # Max delay for P1 prefix conditioning: d ~ Unif{0..p1_max_delay} per example.
    # -1 falls back to `replan_steps`.
    config.p1_max_delay = -1

    return config
