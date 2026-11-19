"""Full EXPO defaults for the OCBench visual policy and action interface."""

from configs.model import expo_ft_pi_libero_config


def get_config():
    config = expo_ft_pi_libero_config.get_config()
    # The launcher supplies fitted OCBench actor weights and normalization.
    config.pi05_weight_loader_path = ''
    config.pi05_assets_dir = ''
    config.pi05_asset_id = 'dcrl/ocbench'
    return config
