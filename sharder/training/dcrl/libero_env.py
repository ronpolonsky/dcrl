"""LIBERO-90 environment and observation adapter for the small EXPO learner."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re

import numpy as np
from openpi_client import image_tools


DEFAULT_TASK_IDS = (60, 79, 29, 28, 2)
LIBERO_SUITE_MAX_STEPS = {
    'libero_spatial': 220, 'libero_object': 280, 'libero_goal': 300,
    'libero_10': 520, 'libero_90': 400,
}
LIBERO_DUMMY_ACTION = np.asarray([0.0] * 6 + [-1.0], dtype=np.float32)
LIBERO_RENDER_RESOLUTION = 256
LIBERO_SETTLE_STEPS = 10
CAMERA_KEYS = ('base_0_rgb', 'left_wrist_0_rgb')


def tokenize_instruction(instruction: str, max_length: int = 32) -> np.ndarray:
    """Stable compact vocabulary for the small actor; used for demos and rollout."""
    words = re.findall(r"[a-z0-9]+|[^\w\s]", instruction.lower())[:max_length]
    ids = [1 + int.from_bytes(hashlib.blake2b(w.encode(), digest_size=4).digest(), 'little') % 8191
           for w in words]
    return np.asarray(ids + [0] * (max_length - len(ids)), dtype=np.int32)


def _image(raw: np.ndarray, image_size: int) -> np.ndarray:
    # OpenPI's π₀.₅ LIBERO client rotates both axes before resize_with_pad.
    upright = np.ascontiguousarray(np.asarray(raw)[::-1, ::-1])
    return image_tools.convert_to_uint8(
        image_tools.resize_with_pad(upright, image_size, image_size))


def observation(raw: dict, instruction: str, image_size: int) -> dict:
    """Map live observations or official demonstration observations to one format."""
    if 'agentview_image' in raw:
        front = raw['agentview_image']
        wrist = raw['robot0_eye_in_hand_image']
        from robosuite.utils.transform_utils import quat2axisangle
        ee = np.concatenate((raw['robot0_eef_pos'], quat2axisangle(raw['robot0_eef_quat'])))
        gripper = raw['robot0_gripper_qpos']
    else:
        front = raw['agentview_rgb']
        wrist = raw['eye_in_hand_rgb']
        ee = raw['ee_states']
        gripper = raw['gripper_states']
    tokens = tokenize_instruction(instruction)
    return {
        'image': {CAMERA_KEYS[0]: _image(front, image_size), CAMERA_KEYS[1]: _image(wrist, image_size)},
        'state': np.concatenate((np.asarray(ee).reshape(-1), np.asarray(gripper).reshape(-1))).astype(np.float32),
        'tokenized_prompt': tokens,
        'tokenized_prompt_mask': tokens != 0,
    }


class LiberoTaskEnv:
    def __init__(self, suite_name: str, task_id: int, *, seed: int, image_size: int,
                 init_state_offset: int = 0, benchmark_eval: bool = False):
        from libero.libero import benchmark
        from libero.libero.envs import OffScreenRenderEnv
        self.suite = benchmark.get_benchmark_dict()[suite_name]()
        if not 0 <= task_id < self.suite.get_num_tasks():
            raise ValueError(f'{suite_name} task ID must be in [0, {self.suite.get_num_tasks()})')
        self.task_id = task_id
        self.task = self.suite.get_task(task_id)
        self.instruction = self.task.language
        self.image_size = image_size
        self.init_states = self.suite.get_task_init_states(task_id)
        self.init_state_offset = init_state_offset
        self.benchmark_eval = benchmark_eval
        self.episode_index = 0
        self.seed = seed
        self.env = OffScreenRenderEnv(
            bddl_file_name=self.suite.get_task_bddl_file_path(task_id),
            camera_heights=LIBERO_RENDER_RESOLUTION,
            camera_widths=LIBERO_RENDER_RESOLUTION,
        )
        self.env.seed(seed)

    def reset(self) -> dict:
        if not self.benchmark_eval:
            # Keep training episode resets reproducible across checkpoint resume.
            self.env.seed(self.seed + self.episode_index)
        self.env.reset()
        index = self.init_state_offset + self.episode_index
        if self.benchmark_eval and index >= len(self.init_states):
            raise ValueError(f'Benchmark evaluation has only {len(self.init_states)} initial states')
        index %= len(self.init_states)
        self.episode_index += 1
        raw = self.env.set_init_state(self.init_states[index])
        # Match OpenPI's LIBERO client: ten idle steps with the gripper open.
        for _ in range(LIBERO_SETTLE_STEPS):
            raw, _, _, _ = self.env.step(LIBERO_DUMMY_ACTION)
        return observation(raw, self.instruction, self.image_size)

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, np.ndarray]:
        action = np.asarray(action, np.float32).reshape(-1)
        if action.shape != (7,):
            raise ValueError(f'LIBERO expects a seven-dimensional action, got {action.shape}')
        # EXPO-FT and OpenPI submit policy actions without client-side clipping.
        executed = action.astype(np.float32, copy=False)
        raw, _, done, _ = self.env.step(executed)
        success = bool(done)
        return observation(raw, self.instruction, self.image_size), float(success), success, executed

    def demo_path(self, demos_root: str | None = None) -> Path:
        if demos_root is None:
            from libero.libero import get_libero_path
            demos_root = get_libero_path('datasets')
        return Path(demos_root) / self.task.problem_folder / f'{self.task.name}_demo.hdf5'

    def close(self):
        self.env.close()
