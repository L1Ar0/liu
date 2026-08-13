"""Train the depth/proprioception grasp policy with sample-efficient SAC."""

from __future__ import annotations

import argparse
from collections import deque
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


MONITOR_FIELDS = (
    "task_success",
    "final_distance_m",
    "initial_distance_m",
    "episode_steps",
    "stage_a_position_index",
    "stage_a_position_name",
    "ep_approach_reward",
    "ep_ik_penalty",
    "ep_collision_penalty",
    "ik_failure_count",
    "timeout",
)


def parse_sac_ent_coef(value: str) -> str | float:
    text = str(value).strip().lower()
    if text == "auto" or text.startswith("auto_"):
        return text
    return float(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("sac", "ppo"), default="sac")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None, help="defaults to 30 for A and 80 otherwise")
    parser.add_argument("--sim-steps-per-action", type=int, default=1)
    parser.add_argument("--scene-mode", default="separated", choices=("physics", "planned", "separated", "level4"))
    parser.add_argument("--planned-layout", default="auto")
    parser.add_argument("--stage", choices=("A", "B", "C", "D", "E"), default="A")
    parser.add_argument(
        "--stage-a-position-mode",
        choices=("canonical5", "grid3", "random"),
        default="canonical5",
    )
    parser.add_argument(
        "--execution-mode",
        default="kinematic",
        choices=("kinematic", "settle_then_kinematic", "dynamic"),
    )
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, default=Path("end_to_end_grasp_rl_output/stage_A_sac"))
    parser.add_argument("--resume", type=Path, default=None, help="resume a model of the selected algorithm")
    parser.add_argument("--bc-model", type=Path, default=None, help="copy a behavior-cloned SAC actor into a fresh SAC model")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint-freq", type=int, default=2_000)
    parser.add_argument("--eval-freq", type=int, default=2_000)
    parser.add_argument("--eval-episodes-per-position", type=int, default=2)
    parser.add_argument("--demonstrations", type=Path, default=None, help="optional teacher .npz used to seed SAC replay")
    parser.add_argument("--replay-buffer", type=Path, default=None, help="optional SAC replay buffer to resume")
    parser.add_argument("--save-replay-buffer", action=argparse.BooleanOptionalAction, default=False)
    # SAC controls. The compressed 25k buffer uses roughly 0.8 GiB for depth.
    parser.add_argument("--buffer-size", type=int, default=25_000)
    parser.add_argument("--learning-starts", type=int, default=500)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--sac-ent-coef", type=parse_sac_ent_coef, default="auto")
    # PPO remains available for controlled comparison.
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--clip-range", type=float, default=0.1)
    parser.add_argument("--ppo-ent-coef", type=float, default=0.001)
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _effective_max_steps(args: argparse.Namespace) -> int:
    return int(args.max_steps if args.max_steps is not None else (30 if args.stage == "A" else 80))


def _tensorboard_path(args: argparse.Namespace, output_dir: Path) -> str | None:
    if not args.tensorboard:
        return None
    if importlib.util.find_spec("tensorboard") is None:
        print("TensorBoard is not installed; continuing without TensorBoard logging.", flush=True)
        return None
    return str(output_dir / "tensorboard")


def _policy_kwargs(algorithm: str, extractor: Any) -> dict[str, Any]:
    network = {"pi": [256, 128], "qf": [256, 128]} if algorithm == "sac" else {"pi": [256, 128], "vf": [256, 128]}
    result: dict[str, Any] = {
        "features_extractor_class": extractor,
        "features_extractor_kwargs": {"features_dim": 256},
        "net_arch": network,
    }
    if algorithm == "ppo":
        result["log_std_init"] = -1.0
    return result


def _load_demonstrations(model: Any, path: Path) -> int:
    data = np.load(path)
    required = ("depth", "proprio", "next_depth", "next_proprio", "actions", "rewards", "dones", "timeouts")
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError(f"Demonstration file is missing fields: {', '.join(missing)}")
    count = len(data["actions"])
    for index in range(count):
        observation = {
            "depth": np.asarray(data["depth"][index : index + 1], dtype=np.float32),
            "proprio": np.asarray(data["proprio"][index : index + 1], dtype=np.float32),
        }
        next_observation = {
            "depth": np.asarray(data["next_depth"][index : index + 1], dtype=np.float32),
            "proprio": np.asarray(data["next_proprio"][index : index + 1], dtype=np.float32),
        }
        model.replay_buffer.add(
            observation,
            next_observation,
            np.asarray(data["actions"][index : index + 1], dtype=np.float32),
            np.asarray(data["rewards"][index : index + 1], dtype=np.float32),
            np.asarray(data["dones"][index : index + 1], dtype=np.float32),
            [{
                "TimeLimit.truncated": bool(data["timeouts"][index]),
                "stage_a_position_index": int(data["position_index"][index]) if "position_index" in data else -1,
            }],
        )
    return count


def _serial_evaluate(model: Any, args: argparse.Namespace, max_steps: int, evaluation_index: int) -> dict[str, Any]:
    from end_to_end_grasp_env import EndToEndGraspEnv

    positions = range(5) if args.stage == "A" and args.stage_a_position_mode == "canonical5" else (None,)
    repeats = int(args.eval_episodes_per_position)
    episodes: list[dict[str, Any]] = []
    for position in positions:
        env = EndToEndGraspEnv(
            scene_mode=args.scene_mode,
            planned_layout=args.planned_layout if args.planned_layout != "auto" else None,
            max_steps=max_steps,
            sim_steps_per_action=args.sim_steps_per_action,
            seed=int(args.seed) + 50_000_000,
            headless=True,
            curriculum_stage=args.stage,
            execution_mode=args.execution_mode,
            stage_a_position_mode=args.stage_a_position_mode,
            stage_a_position_index=position,
        )
        try:
            for repeat in range(repeats):
                # Identical seeds across evaluation checkpoints make best-model
                # comparisons reflect policy changes instead of scene changes.
                seed = int(args.seed) + 50_000_000 + repeat * 1_009 + (position or 0) * 10_007
                observation, _ = env.reset(seed=seed)
                terminated = truncated = False
                total_reward = 0.0
                last_info: dict[str, Any] = {}
                while not (terminated or truncated):
                    action, _state = model.predict(observation, deterministic=True)
                    observation, reward, terminated, truncated, last_info = env.step(action)
                    total_reward += float(reward)
                episodes.append(
                    {
                        "position_index": int(last_info.get("stage_a_position_index", -1)),
                        "position_name": str(last_info.get("stage_a_position_name", "random")),
                        "success": bool(last_info.get("task_success", False)),
                        "return": total_reward,
                        "steps": int(last_info.get("episode_steps", max_steps)),
                        "final_distance_m": float(last_info.get("final_distance_m", float("inf"))),
                    }
                )
        finally:
            env.close()
    success_rate = float(np.mean([episode["success"] for episode in episodes]))
    return {
        "timesteps": int(model.num_timesteps),
        "success_rate": success_rate,
        "mean_return": float(np.mean([episode["return"] for episode in episodes])),
        "mean_final_distance_m": float(np.mean([episode["final_distance_m"] for episode in episodes])),
        "episodes": episodes,
    }


def main() -> int:
    args = parse_args()
    if args.timesteps <= 0 or args.n_envs != 1:
        raise SystemExit("Use positive --timesteps and exactly --n-envs 1 with the shared CoppeliaSim scene")
    if min(args.batch_size, args.buffer_size, args.eval_episodes_per_position) <= 0:
        raise SystemExit("Batch, buffer and evaluation counts must be positive")
    try:
        from stable_baselines3 import PPO, SAC
        from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
        from compressed_replay_buffer import CompressedDepthDictReplayBuffer
        from depth_grasp_rl import DepthProprioFeaturesExtractor
        from end_to_end_grasp_env import EndToEndGraspEnv
    except ImportError as exc:
        raise SystemExit("Install requirements-rl.txt before training") from exc
    if DepthProprioFeaturesExtractor is None:
        raise SystemExit("DepthProprioFeaturesExtractor is unavailable")

    max_steps = _effective_max_steps(args)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["ROBOT_GRASP_HEADLESS"] = "1"
    monitor_index = 0

    def make_vec_env() -> Any:
        nonlocal monitor_index
        current_monitor = monitor_index
        monitor_index += 1

        def factory() -> Any:
            raw = EndToEndGraspEnv(
                scene_mode=args.scene_mode,
                planned_layout=args.planned_layout if args.planned_layout != "auto" else None,
                max_steps=max_steps,
                sim_steps_per_action=args.sim_steps_per_action,
                seed=int(args.seed) + current_monitor * 104_729,
                headless=True,
                curriculum_stage=args.stage,
                execution_mode=args.execution_mode,
                stage_a_position_mode=args.stage_a_position_mode,
            )
            return Monitor(raw, filename=str(output_dir / f"monitor_{current_monitor}.csv"), info_keywords=MONITOR_FIELDS)

        return DummyVecEnv([factory])

    class EpisodeDiagnosticsCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self.episodes: deque[dict[str, Any]] = deque(maxlen=100)

        def _on_step(self) -> bool:
            for done, info in zip(self.locals.get("dones", []), self.locals.get("infos", [])):
                if done:
                    self.episodes.append(info)
            if self.episodes:
                values = tuple(self.episodes)
                self.logger.record("rollout/success_rate", np.mean([float(item["task_success"]) for item in values]))
                self.logger.record("rollout/timeout_rate", np.mean([float(item["timeout"]) for item in values]))
                self.logger.record("rollout/final_distance_mean", np.mean([float(item["final_distance_m"]) for item in values]))
            return True

    env = make_vec_env()
    tensorboard_log = _tensorboard_path(args, output_dir)
    policy_kwargs = _policy_kwargs(args.algorithm, DepthProprioFeaturesExtractor)
    algorithm_class = SAC if args.algorithm == "sac" else PPO
    if args.resume is not None:
        model = algorithm_class.load(str(args.resume), env=env, device=args.device)
        print(f"Resumed {args.algorithm.upper()} checkpoint: {args.resume.resolve()}")
    elif args.algorithm == "sac":
        model = SAC(
            "MultiInputPolicy",
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=float(args.learning_rate),
            buffer_size=int(args.buffer_size),
            learning_starts=int(args.learning_starts),
            batch_size=int(args.batch_size),
            tau=float(args.tau),
            gamma=0.99,
            train_freq=(int(args.train_freq), "step"),
            gradient_steps=int(args.gradient_steps),
            ent_coef=args.sac_ent_coef,
            replay_buffer_class=CompressedDepthDictReplayBuffer,
            tensorboard_log=tensorboard_log,
            seed=int(args.seed),
            device=args.device,
            verbose=1,
        )
    else:
        model = PPO(
            "MultiInputPolicy",
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=float(args.learning_rate),
            n_steps=int(args.n_steps),
            batch_size=int(args.batch_size),
            n_epochs=int(args.n_epochs),
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=float(args.clip_range),
            ent_coef=float(args.ppo_ent_coef),
            tensorboard_log=tensorboard_log,
            seed=int(args.seed),
            device=args.device,
            verbose=1,
        )

    if args.bc_model is not None:
        if args.algorithm != "sac":
            raise SystemExit("--bc-model is supported by SAC only")
        if args.resume is not None:
            raise SystemExit("Use either --resume or --bc-model, not both")
        teacher = SAC.load(str(args.bc_model), device=args.device)
        model.actor.load_state_dict(teacher.actor.state_dict())
        print(f"Initialized SAC actor from BC model: {args.bc_model.resolve()}")

    if args.algorithm == "sac" and args.replay_buffer is not None:
        model.load_replay_buffer(args.replay_buffer)
        print(f"Loaded replay buffer: {args.replay_buffer.resolve()}")
    if args.demonstrations is not None:
        if args.algorithm != "sac":
            raise SystemExit("--demonstrations is supported by SAC only")
        count = _load_demonstrations(model, args.demonstrations)
        model.learning_starts = 0
        print(f"Seeded SAC replay with {count} teacher transitions")

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, int(args.checkpoint_freq)),
        save_path=str(output_dir / "checkpoints"),
        name_prefix=f"{args.algorithm}_grasp",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    callbacks = [checkpoint_callback, EpisodeDiagnosticsCallback()]
    evaluations_path = output_dir / "evaluations.jsonl"
    best_score = (-1.0, -float("inf"))
    trained = 0
    evaluation_index = 0
    first_learn = args.resume is None
    try:
        while trained < int(args.timesteps):
            chunk = int(args.timesteps) - trained
            if args.eval_freq > 0:
                chunk = min(chunk, int(args.eval_freq))
            model.learn(
                total_timesteps=chunk,
                callback=callbacks,
                progress_bar=False,
                reset_num_timesteps=first_learn,
            )
            first_learn = False
            trained += chunk
            candidate = output_dir / "candidates" / f"{args.algorithm}_{int(model.num_timesteps):09d}"
            candidate.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(candidate))
            if args.eval_freq <= 0:
                continue
            env.close()
            evaluation = _serial_evaluate(model, args, max_steps, evaluation_index)
            evaluation_index += 1
            with evaluations_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(evaluation, ensure_ascii=False) + "\n")
            score = (float(evaluation["success_rate"]), float(evaluation["mean_return"]))
            print(json.dumps({key: value for key, value in evaluation.items() if key != "episodes"}, ensure_ascii=False, indent=2))
            if score > best_score:
                best_score = score
                model.save(str(output_dir / "best_model"))
                (output_dir / "best_evaluation.json").write_text(
                    json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            if trained < int(args.timesteps):
                env = make_vec_env()
                model.set_env(env)
    finally:
        env.close()

    model.save(str(output_dir / f"{args.algorithm}_grasp_final"))
    if args.algorithm == "sac" and args.save_replay_buffer:
        model.save_replay_buffer(output_dir / "sac_replay_buffer.pkl")
    summary = {
        "algorithm": f"stable_baselines3.{args.algorithm.upper()}",
        "timesteps_requested": int(args.timesteps),
        "model_timesteps": int(model.num_timesteps),
        "stage": args.stage,
        "max_steps": max_steps,
        "scene_mode": args.scene_mode,
        "execution_mode": args.execution_mode,
        "best_success_rate": best_score[0] if best_score[0] >= 0.0 else None,
        "stage_a_position_mode": args.stage_a_position_mode if args.stage == "A" else None,
        "observation": "128x128 table-relative height map + CoordX/CoordY + proprioception",
        "action": ["dx", "dy"] if args.stage == "A" else ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"],
        "ground_truth_in_observation": False,
        "demonstration_teacher_used_at_runtime": False,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
