"""Execute a trained end-to-end PPO grasp policy in CoppeliaSim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="SB3 .zip checkpoint")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--scene-mode", default="physics")
    parser.add_argument("--planned-layout", default="auto")
    parser.add_argument(
        "--execution-mode",
        default="settle_then_kinematic",
        choices=("kinematic", "settle_then_kinematic", "dynamic"),
    )
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--output", type=Path, default=ROOT / "end_to_end_grasp_rl_output" / "evaluation.json")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.episodes <= 0:
        raise SystemExit("--episodes must be positive")
    try:
        from stable_baselines3 import PPO
        from end_to_end_grasp_env import EndToEndGraspEnv
    except ImportError as exc:
        raise SystemExit("Install requirements-rl.txt before running the policy.") from exc

    model = PPO.load(str(args.checkpoint), device=args.device)
    results: list[dict[str, object]] = []
    for episode in range(int(args.episodes)):
        env = EndToEndGraspEnv(
            scene_mode=args.scene_mode,
            planned_layout=args.planned_layout if args.planned_layout != "auto" else None,
            max_steps=args.max_steps,
            seed=int(args.seed) + episode * 7919,
            headless=False,
            execution_mode=args.execution_mode,
        )
        observation, info = env.reset(seed=int(args.seed) + episode * 7919)
        total_reward = 0.0
        terminated = truncated = False
        last_info: dict[str, object] = dict(info)
        try:
            while not (terminated or truncated):
                action, _state = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, last_info = env.step(action)
                total_reward += float(reward)
        finally:
            env.close()
        results.append(
            {
                "episode": episode,
                "return": total_reward,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "success": bool(last_info.get("grasp_success", False)),
                "last_info": last_info,
            }
        )
        print(json.dumps(results[-1], ensure_ascii=False, indent=2))
    payload = {"checkpoint": str(args.checkpoint.resolve()), "episodes": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved evaluation: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
