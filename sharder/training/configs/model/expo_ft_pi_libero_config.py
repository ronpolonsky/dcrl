from configs.model import sac_config


def get_config():
    config = sac_config.get_config()

    config.model_cls = "EXPOLearner"

    config.num_qs = 10
    config.num_min_qs = 2
    config.critic_layer_norm = True

    config.N = 8
    config.n_edit_samples = 8

    config.adjust_target_entropy = False
    config.entropy_scale = 1.0
    config.edit_scale = 0.2
    config.actor_drop = 0.0
    config.actor_lr = 3e-4
    config.critic_lr = 3e-4

    config.latent_dim_image = 512
    config.latent_dim_state = 64
    config.include_state = True
    config.encoder_stage_sizes = (3, 4, 6, 3)
    config.encoder_num_filters = 64
    config.hidden_dims = (256, 256, 256)

    config.encode_batch_split = 1
    config.batch_split = 1

    # Use OpenPI's official Pi0.5 LIBERO model/data transforms and norm stats.
    # The surrounding EXPO learner settings remain the upstream Expo-FT values.
    # Match the Pi0.5 LoRA setup used by LIBERO SFT.  The base checkpoint is
    # unchanged; only LoRA adapters receive actor gradients during RL.
    config.pi05_config_name = "pi05_libero_lora"
    config.pi05_resize_size = 224
    config.freeze_pi05_encoder = True
    config.freeze_critic_encoder = False  # if True, encoder is frozen for Q (only extract embeddings)

    # Use the already-collected official Pi0.5 LIBERO base tree on the project
    # filesystem. This avoids a GCS download and keeps the single-GPU test off
    # the slow /work filesystem.
    config.pi05_weight_loader_path = "/projects/bgqe/rpolonsky/dcrlcheckpoints/openpi_cache/openpi-assets/checkpoints/pi05_libero/params"
    # assets_dir is the base path; norm stats are loaded from assets_dir/asset_id.
    config.pi05_assets_dir = "/projects/bgqe/rpolonsky/dcrlcheckpoints/openpi_cache/openpi-assets/checkpoints/pi05_libero/assets"
    config.pi05_asset_id = "physical-intelligence/libero"
    config.actor_success_only = True
    config.use_full_augmentation = True  # False = only crop (no rotate/color jitter)

    return config
