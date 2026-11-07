from configs.model import expo_ft_pi_config


def get_config():
    config = expo_ft_pi_config.get_config()
    config.model_cls = "RealTimeEXPOFTLearner"

    # Prefix-conditioned BC fine-tuning (train_step_p1_prefix) so the inpainted
    # delayed inference is in-distribution.
    config.p1_use_prefix_conditioning = True
    config.init_temperature = 0.01

    config.q_edit_use_main_obs = False  # backup edit/Q-selection on the delayed (main-actor) obs when delay>0
    config.train_base_actor = True   # False = freeze the pi0.5 actor; only the critic/edit actor learn
    config.filter_N = 8              # noise-seed pool scored per backup sample
    config.filter_temperature = 0.0  # 0 = argmax; >0 = categorical over zscore(Q_f)/temp
    config.filter_num_qs = 2         # Q_f ensemble size
    config.filter_n_edit = -1        # backup edit count; <0 = reuse n_edit_samples
    # Q_f also conditions on the actor's delayed obs (image latent + proprio); only matters at delay>0.
    config.filter_add_delayed_obs = True

    return config
