"""Collect Stage A teacher demonstrations without changing policy observations."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from end_to_end_grasp_env import EndToEndGraspEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-per-position", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--output", type=Path, default=Path("end_to_end_grasp_rl_output/stage_a_demonstrations.npz"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.episodes_per_position <= 0 or args.max_steps <= 0:
        raise SystemExit("--episodes-per-position and --max-steps must be positive")
    observations: dict[str, list[np.ndarray]] = {"depth": [], "proprio": []}
    next_observations: dict[str, list[np.ndarray]] = {"depth": [], "proprio": []}
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    dones: list[bool] = []
    timeouts: list[bool] = []
    position_indices: list[int] = []
    episode_ids: list[int] = []
    env = EndToEndGraspEnv(
        scene_mode="separated",
        max_steps=int(args.max_steps),
        sim_steps_per_action=1,
        seed=int(args.seed),
        curriculum_stage="A",
        execution_mode="kinematic",
        stage_a_position_mode="canonical5",
    )
    successes = 0
    episode_id = 0
    try:
        for position_index in range(5):
            for repeat in range(int(args.episodes_per_position)):
                seed = int(args.seed) + position_index * 100_003 + repeat * 1_009
                observation, _info = env.reset(
                    seed=seed,
                    options={"stage_a_position_index": position_index},
                )
                terminated = truncated = False
                last_info: dict[str, object] = {}
                while not (terminated or truncated):
                    action = env.stage_a_expert_action()
                    observations["depth"].append(np.asarray(observation["depth"], dtype=np.float32))
                    observations["proprio"].append(np.asarray(observation["proprio"], dtype=np.float32))
                    actions.append(action)
                    position_indices.append(position_index)
                    episode_ids.append(episode_id)
                    next_observation, reward, terminated, truncated, last_info = env.step(action)
                    next_observations["depth"].append(np.asarray(next_observation["depth"], dtype=np.float32))
                    next_observations["proprio"].append(np.asarray(next_observation["proprio"], dtype=np.float32))
                    rewards.append(float(reward))
                    dones.append(bool(terminated or truncated))
                    timeouts.append(bool(truncated and not terminated))
                    observation = next_observation
                successes += int(bool(last_info.get("task_success", False)))
                episode_id += 1
    finally:
        env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        depth=np.asarray(observations["depth"], dtype=np.float32),
        proprio=np.asarray(observations["proprio"], dtype=np.float32),
        next_depth=np.asarray(next_observations["depth"], dtype=np.float32),
        next_proprio=np.asarray(next_observations["proprio"], dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        dones=np.asarray(dones, dtype=np.bool_),
        timeouts=np.asarray(timeouts, dtype=np.bool_),
        position_index=np.asarray(position_indices, dtype=np.int8),
        episode_id=np.asarray(episode_ids, dtype=np.int32),
    )
    print(
        f"Saved {len(actions)} transitions from {episode_id} episodes to {args.output.resolve()}\n"
        f"Teacher success rate: {successes / max(episode_id, 1):.1%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
