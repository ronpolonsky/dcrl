"""Isolated Pi0.5 LIBERO EXPO config for batch-16 encoder chunking."""

from configs.model import expo_ft_pi_libero_config


def get_config():
    config = expo_ft_pi_libero_config.get_config()
    # Keep the effective replay batch at 16, but compile encoder calls as
    # four smaller chunks to avoid the failing batch-16 ResNet backward
    # fusion on the two-way FSDP mesh.
    config.encode_batch_split = 4
    return config
