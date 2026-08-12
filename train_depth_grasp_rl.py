"""Train the end-to-end depth/proprioception grasp policy with PPO.

This entry point performs real multi-step rollouts in CoppeliaSim through the
Gymnasium environment.  It does not load offline labels or compute a bandit
geometry error.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--sim-steps-per-action", type=int, default=4)
    parser.add_argument("--scene-mode", default="physics", choices=("physics", "planned", "separated", "level4"))
    parser.add_argument("--planned-layout", default="auto")
    parser.add_argument(
        "--execution-mode",
        default="settle_then_kinematic",
        choices=("kinematic", "settle_then_kinematic", "dynamic"),
        help="RL execution: freeze after physics reset (default), pure kinematic, or experimental dynamic rollout",
    )
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=Path("end_to_end_grasp_rl_output"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint-freq", type=int, default=100_000)
    parser.add_argument("--curriculum", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--tensorboard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable TensorBoard logging when the optional tensorboard package is installed",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.timesteps <= 0 or args.n_envs <= 0:
        raise SystemExit("--timesteps and --n-envs must be positive")
    if args.n_envs != 1:
        raise SystemExit(
            "CoppeliaSim environment currently supports one live simulator per training process; "
            "use --n-envs 1 or launch separate simulator instances."
        )
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
        from end_to_end_grasp_env import EndToEndGraspEnv
        from depth_grasp_rl import DepthProprioFeaturesExtractor
    except ImportError as exc:
        raise SystemExit(
            "End-to-end PPO requires Gymnasium, Stable-Baselines3, PyTorch and "
            "the CoppeliaSim ZMQ client. Install requirements-rl.txt."
        ) from exc
    if DepthProprioFeaturesExtractor is None:
        raise SystemExit("DepthProprioFeaturesExtractor is unavailable; install stable-baselines3 and torch.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["ROBOT_GRASP_HEADLESS"] = "1"

    def make_env(rank: int):
        def _factory():
            env = EndToEndGraspEnv(
                scene_mode=args.scene_mode,
                planned_layout=args.planned_layout if args.planned_layout != "auto" else None,
                max_steps=args.max_steps,
                sim_steps_per_action=args.sim_steps_per_action,
                seed=int(args.seed) + rank * 104729,
                headless=True,
                curriculum_stage="A" if args.curriculum else "E",
                execution_mode=args.execution_mode,
            )
            return Monitor(env)
        return _factory

    env = DummyVecEnv([make_env(index) for index in range(args.n_envs)])
    policy_kwargs = {
        "features_extractor_class": DepthProprioFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
        "net_arch": {"pi": [256, 128], "vf": [256, 128]},
    }
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, int(args.checkpoint_freq) // max(args.n_envs, 1)),
        save_path=str(output_dir / "checkpoints"),
        name_prefix="ppo_grasp",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    class CurriculumCallback(BaseCallback):
        def __init__(self, total_timesteps: int) -> None:
            super().__init__()
            self.total_timesteps = max(1, int(total_timesteps))
            self.current_stage = "A"

        def _on_step(self) -> bool:
            progress = float(self.num_timesteps) / self.total_timesteps
            stage = "A" if progress < 0.15 else "B" if progress < 0.35 else "C" if progress < 0.55 else "D" if progress < 0.75 else "E"
            if stage != self.current_stage:
                self.training_env.env_method("set_curriculum_stage", stage)
                self.current_stage = stage
                print(f"Curriculum stage -> {stage}")
            return True

    callbacks = [checkpoint_callback]
    if args.curriculum:
        callbacks.append(CurriculumCallback(args.timesteps))
    tensorboard_log = None
    if args.tensorboard:
        if importlib.util.find_spec("tensorboard") is not None:
            tensorboard_log = str(output_dir / "tensorboard")
        else:
            print(
                "TensorBoard 未安装，自动关闭日志；如需曲线请安装 "
                "'.venv\\Scripts\\python.exe -m pip install tensorboard'。",
                flush=True,
            )
    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=float(args.learning_rate),
        n_steps=int(args.n_steps),
        batch_size=int(args.batch_size),
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=tensorboard_log,
        seed=int(args.seed),
        device=args.device,
        verbose=1,
    )
    model.learn(total_timesteps=int(args.timesteps), callback=callbacks, progress_bar=False)
    model.save(str(output_dir / "ppo_grasp_final"))
    summary = {
        "algorithm": "stable_baselines3.PPO",
        "environment": "EndToEndGraspEnv",
        "timesteps": int(args.timesteps),
        "n_envs": int(args.n_envs),
        "max_steps": int(args.max_steps),
        "scene_mode": args.scene_mode,
        "execution_mode": args.execution_mode,
        "planned_layout": args.planned_layout,
        "curriculum": bool(args.curriculum),
        "seed": int(args.seed),
        "observation": "128x128 normalized depth + 7 joint positions + 7 joint velocities + gripper + previous action",
        "action": ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"],
        "execution_note": "settle_then_kinematic uses dynamics only to settle reset scenes; PPO rollout uses IK and connector proxy",
        "ground_truth_in_observation": False,
        "target_id_in_observation": False,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    env.close()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
