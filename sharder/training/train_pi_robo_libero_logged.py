"""Test wrapper that reports the actual loaded LIBERO transition count."""

from pathlib import Path

import train_pi_robo_libero as trainer


_original_load_libero_dataset = trainer._load_libero_dataset


def _load_libero_dataset_logged(
    path: Path, instruction: str, image_size: int, max_episodes: int
) -> list[dict]:
    dataset = _original_load_libero_dataset(
        path, instruction, image_size, max_episodes
    )
    print(
        "[DATA CHECK] loaded transitions="
        f"{len(dataset)} from={path} max_episodes={max_episodes}"
        " (0 means all episodes)",
        flush=True,
    )
    return dataset


trainer._load_libero_dataset = _load_libero_dataset_logged


if __name__ == "__main__":
    trainer.app.run(trainer.main)
