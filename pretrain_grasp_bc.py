"""Behavior-clone an SB3 SAC actor from Stage A teacher demonstrations."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demonstrations", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--output", type=Path, default=Path("end_to_end_grasp_rl_output/stage_a_bc_sac.zip"))
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise SystemExit("--epochs and --batch-size must be positive")
    try:
        import gymnasium as gym
        import torch
        from stable_baselines3 import SAC
        from depth_grasp_rl import DepthProprioFeaturesExtractor
    except ImportError as exc:
        raise SystemExit("Install requirements-rl.txt before BC pretraining") from exc
    data = np.load(args.demonstrations)
    depth = np.asarray(data["depth"], dtype=np.float32)
    proprio = np.asarray(data["proprio"], dtype=np.float32)
    actions = np.asarray(data["actions"], dtype=np.float32)
    if not (len(depth) == len(proprio) == len(actions)) or len(actions) == 0:
        raise SystemExit("Demonstration arrays must be non-empty and have equal length")
    observation_space = gym.spaces.Dict(
        {
            "depth": gym.spaces.Box(0.0, 1.0, shape=depth.shape[1:], dtype=np.float32),
            "proprio": gym.spaces.Box(-1.0, 1.0, shape=proprio.shape[1:], dtype=np.float32),
        }
    )
    action_space = gym.spaces.Box(-1.0, 1.0, shape=actions.shape[1:], dtype=np.float32)
    policy_kwargs = {
        "features_extractor_class": DepthProprioFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
        "net_arch": {"pi": [256, 128], "qf": [256, 128]},
    }
    class DemonstrationSpaceEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self) -> None:
            self.observation_space = observation_space
            self.action_space = action_space

        def reset(self, *, seed: int | None = None, options=None):
            super().reset(seed=seed)
            return {
                "depth": np.zeros(depth.shape[1:], dtype=np.float32),
                "proprio": np.zeros(proprio.shape[1:], dtype=np.float32),
            }, {}

        def step(self, action):
            observation, info = self.reset()
            return observation, 0.0, False, True, info

    model = SAC(
        "MultiInputPolicy",
        env=DemonstrationSpaceEnv(),
        policy_kwargs=policy_kwargs,
        learning_rate=float(args.learning_rate),
        buffer_size=1,
        learning_starts=1,
        seed=int(args.seed),
        device=args.device,
    )
    # SB3 does not expose a standalone BC trainer. Optimize the deterministic
    # squashed actor action directly; critics are initialized later by SAC.
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=float(args.learning_rate))
    rng = np.random.default_rng(int(args.seed))
    device = model.device
    model.actor.set_training_mode(True)
    for epoch in range(int(args.epochs)):
        order = rng.permutation(len(actions))
        losses: list[float] = []
        for start in range(0, len(order), int(args.batch_size)):
            indices = order[start : start + int(args.batch_size)]
            observation = {
                "depth": torch.as_tensor(depth[indices], device=device),
                "proprio": torch.as_tensor(proprio[indices], device=device),
            }
            target = torch.as_tensor(actions[indices], device=device)
            prediction = model.actor(observation, deterministic=True)
            loss = torch.nn.functional.mse_loss(prediction, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(f"BC epoch {epoch + 1:03d}/{args.epochs}: loss={np.mean(losses):.6f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output))
    print(f"Saved BC-initialized SAC policy: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
