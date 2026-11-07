"""CPU checks for SFT checkpoint metadata, restoration, and rollout dispatch."""
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import full_90_libero_evals as evaluator


class EvalSftTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.run = Path(self.directory.name)
        self.root = self.run / "checkpoints"
        for step in (25000, 50000):
            (self.root / str(step)).mkdir(parents=True)
        self.assets = self.run / "student_assets"
        stats = self.assets / "physical-intelligence/libero/norm_stats.json"
        stats.parent.mkdir(parents=True)
        stats.write_text("{}")
        self.manifest = {"learner_config": "configs.sft.my_expo_libero",
                         "action_horizon": 10, "replan_steps": 5,
                         "image_size": 224, "finetune_mode": "lora", "seed": 42}
        (self.run / "sft_config.json").write_text(json.dumps(self.manifest))

    def test_paths_and_saved_configuration(self):
        for path, hint in ((self.run, None), (self.root, None), (self.root / "25000", 25000)):
            root, actual_hint, args, assets = evaluator._sft_checkpoint_config(path, None, 5, 224)
            self.assertEqual((root, actual_hint, assets), (self.root, hint, self.assets))
            self.assertEqual(args.finetune_mode, "lora")
            self.assertTrue(args.resume)
        with self.assertRaises(ValueError):
            evaluator._sft_checkpoint_config(self.run, None, 8, 224)
        (self.run / "sft_config.json").unlink()
        with self.assertRaises(FileNotFoundError):
            evaluator._sft_checkpoint_config(self.run, None, 5, 224)

    def test_restore_latest_and_explicit_step(self):
        manager = Mock()
        manager.all_steps.return_value = (25000, 50000)
        template, restored = object(), object()
        build = Mock(return_value=(template, None, None))
        restore = Mock(return_value=restored)
        modules = {
            "configs.sft.my_expo_libero": types.SimpleNamespace(build_learner=build),
            "dcrl.agents.my_expo_learner": types.SimpleNamespace(restore_checkpoint=restore),
            "dcrl.expo.agents.alg.agent": types.SimpleNamespace(
                initialize_checkpoint_dir=Mock(return_value=(manager, True))),
        }
        with patch.dict(sys.modules, modules):
            for path, step in ((self.run, 50000), (self.root / "25000", 25000)):
                result = evaluator._build_sft_agent(path, None, "task prompt", None, None, None, 5, 224)
                self.assertEqual(result, (restored, manager))
                restore.assert_called_with(manager, template, step)
                self.assertTrue(build.call_args.kwargs["restore_only"])
            restore.side_effect = RuntimeError("restore failed")
            with self.assertRaises(RuntimeError):
                evaluator._build_sft_agent(self.run, None, "task", None, None, None, 5, 224)
            manager.close.assert_called_once()

    def test_rollout_dispatch(self):
        observation = {"image": {"base_0_rgb": np.zeros((2, 2, 3)),
                                  "left_wrist_0_rgb": np.zeros((2, 2, 3))},
                       "state": np.zeros(8)}
        class Env:
            init_states = [0]
            instruction = "current task instruction"
            def __init__(self, *args, **kwargs):
                self.env = types.SimpleNamespace(seed=lambda seed: None)
            def reset(self):
                return observation
            def step(self, action):
                return observation, 1, True, None
            def close(self):
                pass

        fake_jax = types.SimpleNamespace(random=types.SimpleNamespace(PRNGKey=lambda seed: seed),
                                         device_get=lambda value: value)
        with patch.dict(sys.modules, {"jax": fake_jax,
                                      "dcrl.libero_env": types.SimpleNamespace(LiberoTaskEnv=Env)}):
            for mode in ("sft_pi", "direct_rl_pi", "compact"):
                agent = Mock()
                agent.replace.return_value = agent
                chunk = (np.zeros((10, 7)), agent, {})
                agent.eval_actor_only.return_value = chunk
                agent.sample_actions.return_value = chunk
                result = evaluator._evaluate_task(
                    agent, mode=mode, suite="libero_90", task_id=42, episodes=1,
                    max_steps=400, replan_steps=5, image_size=224, eval_seed=7)
                self.assertEqual(result["successes"], 1)
                if mode == "sft_pi":
                    agent.sample_actions.assert_not_called()
                    self.assertEqual(agent.eval_actor_only.call_args.args[0]["prompt"], Env.instruction)
                else:
                    agent.eval_actor_only.assert_not_called()
                    self.assertFalse(agent.sample_actions.call_args.kwargs["only_base_actions"])


if __name__ == "__main__":
    unittest.main()
