import os

import h5py
import numpy as np
from tqdm import tqdm

def _has_trajectory(ep_dir):
    path = os.path.join(ep_dir, "traj.hdf5")
    if not os.path.exists(path):
        return False
    try:
        with h5py.File(path, "r") as f:
            return "saved_observation" in f and "action" in f
    except OSError:
        return False


def _discover_episode_dirs(base_path):
    dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    # Keep only numeric directory names (episode indices); skip e.g. action_videos, lerobot
    dirs = [d for d in dirs if d.isdigit()]
    dirs = sorted(dirs, key=lambda x: int(x))
    dirs = [os.path.join(base_path, d) for d in dirs]
    # Aborted recordings leave an empty traj.hdf5; drop them before any index/count slicing.
    return [d for d in dirs if _has_trajectory(d)]

def process_droid_dataset(
    datapath, 
    task_config, 
    episode_indices = None,
    num_data = None):
    ep_dirs = _discover_episode_dirs(datapath)
    if episode_indices is not None:
        ep_dirs = [ep_dirs[i] for i in episode_indices if 0 <= i < len(ep_dirs)]
    elif num_data is not None and num_data > 0:
        ep_dirs = ep_dirs[:num_data]
    
    print(f"Find {len(_discover_episode_dirs(datapath))} episodes; using {len(ep_dirs)}")

    data = []
    for ep in tqdm(ep_dirs):
        with h5py.File(os.path.join(ep, "traj.hdf5"), "r") as f:
            def load_recursive(group):
                result = {}
                for k, v in group.items():
                    if isinstance(v, h5py.Group):
                        result[k] = load_recursive(v)
                    else:
                        arr = np.asarray(v)
                        # Decode bytes to strings (h5py stores strings as bytes)
                        if arr.dtype.kind == 'S':
                            result[k] = arr.astype('U')
                        elif arr.dtype == object and arr.size > 0 and isinstance(arr.flat[0], bytes):
                            result[k] = np.array([s.decode('utf-8') for s in arr.flat]).reshape(arr.shape)
                        else:
                            result[k] = arr
                return result
            
            ep_obs = load_recursive(f["saved_observation"])
            action_key = task_config.action_space
            gripper_key = f"gripper_{task_config.gripper_action_space}"
            a1 = np.asarray(f["action"][action_key])
            a2 = np.asarray(f["action"][gripper_key])
            ep_actions = np.concatenate([a1, a2[:, None] if len(a2.shape) == 1 else a2], axis=-1)
            
            T = len(ep_actions)
            ep_dones = np.pad(np.array([1.0], dtype=np.float32), (T-1, 0), constant_values=0)
            ep_rewards = np.pad(np.array([1.0], dtype=np.float32), (T-1, 0), constant_values=0)
            
            def extract_t(obs, t):
                return {k: extract_t(v, t) if isinstance(v, dict) else (v[t] if isinstance(v, np.ndarray) and len(v.shape) > 0 else v)
                       for k, v in obs.items()}
            
            for t in range(T):
                data.append({
                    "observations": extract_t(ep_obs, t),
                    "actions": ep_actions[t],
                    "rewards": ep_rewards[t],
                    "masks": 1 - ep_dones[t],
                    "dones": ep_dones[t]
                })

    return data
