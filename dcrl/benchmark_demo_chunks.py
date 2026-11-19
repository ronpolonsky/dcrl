"""Sample demonstration chunks from an OCBench task for Pi0.5 SFT."""

from pathlib import Path

import h5py
import numpy as np

from dcrl.ocbench_env import demonstration_groups, load_demonstrations, observation


class OCBenchDemoChunkDataset:
    def __init__(self, path, *, instruction, image_size, action_horizon, replan_steps,
                 additional_tasks=()):
        self.path = path
        self.instruction = instruction
        self.action_horizon = action_horizon
        self.image_size = image_size
        self.files = []
        self.render_envs = {}
        self.episodes = []
        self.indices = []
        self.task_ranges = []
        try:
            for demo_path, prompt in ((path, instruction), *additional_tasks):
                self._add_task(Path(demo_path), prompt, replan_steps)
        except BaseException:
            self.close()
            raise

    def _add_task(self, path, prompt, replan_steps):
        start_count = len(self.indices)
        if path.suffix == '.hdf5':
            file = h5py.File(path, 'r')
            self.files.append(file)
            released_states = file.attrs.get('observation_source') == 'released_state'
            if released_states:
                if not file.attrs.get('complete') or file.attrs['observation_interval'] != replan_steps:
                    raise ValueError(f'Released state archive must match the SFT replanning interval: {path}')
                groups = [file[key] for key in sorted(file)]
            else:
                groups = demonstration_groups(file, path)
            # The upstream Bowling behavior-cloning baseline uses all episodes.
            bowling = str(file.attrs['env_id']).startswith('visual-bowling-')
            episodes = [group for group in groups if bowling or group.attrs['success']]
        else:
            episodes = [episode for episode in load_demonstrations(
                path, instruction=prompt, image_size=self.image_size) if episode[-1].success]
        for episode in episodes:
            length = len(episode['actions']) if isinstance(episode, h5py.Group) else len(episode)
            self.indices.extend((len(self.episodes), start) for start in range(0, length, replan_steps))
            self.episodes.append((episode, prompt))
        if len(self.indices) == start_count:
            raise ValueError(f'No successful demonstration chunks in {path}')
        self.task_ranges.append((start_count, len(self.indices)))

    def __len__(self):
        return len(self.indices)

    @property
    def episode_count(self):
        return len(self.episodes)

    def sample(self, rng, count):
        rows = []
        if len(self.task_ranges) == 1:
            indices = rng.integers(len(self), size=count)
        else:
            tasks = rng.integers(len(self.task_ranges), size=count)
            indices = [rng.integers(*self.task_ranges[task]) for task in tasks]
        for index in indices:
            episode_id, start = self.indices[index]
            episode, prompt = self.episodes[episode_id]
            stop = start + self.action_horizon
            if isinstance(episode, h5py.Group):
                actions = episode['actions'][start:stop].astype(np.float32)
                source = episode.file.attrs
                if source.get('observation_source') == 'released_state':
                    frame = start // source['observation_interval']
                    env_id = str(source['env_id'])
                    if env_id not in self.render_envs:
                        import ocbench

                        env = ocbench.make(env_id)
                        self.render_envs[env_id] = env
                        env.reset(seed=0)
                    env = self.render_envs[env_id].unwrapped
                    extra_state = {key: episode[key][frame] for key in ('dynamics_info', 'button_states')
                                   if key in episode}
                    env.set_state(episode['qpos'][frame], episode['qvel'][frame], **extra_state)
                    pixels = env.compute_observation()
                else:
                    frame = start
                    pixels = episode['observations'][frame]
                obs = observation(pixels, episode['proprioception'][frame], prompt, self.image_size)
            else:
                actions = np.stack([row.action for row in episode[start:stop]])
                obs = episode[start].obs
            actions = np.pad(actions, ((0, self.action_horizon - len(actions)), (0, 0)), mode='edge')
            rows.append({'observation/image': obs['image']['base_0_rgb'],
                         'observation/wrist_image': obs['image']['left_wrist_0_rgb'],
                         'observation/state': obs['state'], 'actions': actions,
                         'prompt': prompt})
        return rows

    def close(self):
        for env in self.render_envs.values():
            env.close()
        for file in self.files:
            file.close()
