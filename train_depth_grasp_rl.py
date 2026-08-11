"""Train the optional depth-to-6D contextual-bandit PPO policy."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from depth_grasp_rl import (
    ACTION_DIM,
    DepthGraspActorCritic,
    action_reward,
    make_state_tensor,
    require_torch,
)


def _metrics(actions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    values = [action_reward(action, target) for action, target in zip(actions, targets)]
    rewards = [item[0] for item in values]
    successes = [item[1] for item in values]
    return {
        "reward": float(np.mean(rewards)),
        "success_percent": 100.0 * float(np.mean(successes)),
        "position_error": float(np.mean([item[2]["position_error"] for item in values])),
        "yaw_error_deg": float(np.degrees(np.mean([item[2]["yaw_error_rad"] for item in values]))),
        "normal_error_deg": float(np.degrees(np.mean([item[2]["normal_error_rad"] for item in values]))),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--output-dir", type=Path, default=Path("depth_grasp_rl_output"))
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.episodes <= 0:
        raise SystemExit("--episodes must be positive")
    torch = require_torch()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    data = np.load(args.dataset, allow_pickle=False)
    states = np.asarray(data["states"], dtype=np.float32)
    targets = np.asarray(data["actions"], dtype=np.float32)
    if len(states) != len(targets) or len(states) < 4:
        raise RuntimeError("Dataset must contain matching states/actions and at least 4 samples")
    permutation = np.random.default_rng(args.seed).permutation(len(states))
    split = max(1, int(0.75 * len(states)))
    train_idx, val_idx = permutation[:split], permutation[split:]
    train_states = make_state_tensor(states[train_idx], args.device)
    train_targets = targets[train_idx]
    val_states = make_state_tensor(states[val_idx], args.device)
    val_targets = targets[val_idx]

    model = DepthGraspActorCritic(ACTION_DIM).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.csv"
    metrics_file = metrics_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        metrics_file,
        fieldnames=["episode", "train_reward", "train_success_percent", "val_reward", "val_success_percent", "policy_loss", "value_loss"],
    )
    writer.writeheader()
    best_success = -1.0
    best_episode = 0
    rng = np.random.default_rng(args.seed)
    for episode in range(1, int(args.episodes) + 1):
        batch_indices = rng.integers(0, len(train_idx), size=max(1, int(args.batch_size)))
        state_batch = train_states[torch.as_tensor(batch_indices, dtype=torch.long, device=args.device)]
        target_batch = train_targets[batch_indices]
        with torch.no_grad():
            distribution = model.distribution(state_batch)
            sampled = distribution.sample()
            old_log_prob = distribution.log_prob(sampled).sum(dim=-1)
            _, _, values = model(state_batch)
        sampled_np = sampled.detach().cpu().numpy()
        rewards_np = np.asarray([action_reward(action, target)[0] for action, target in zip(sampled_np, target_batch)], dtype=np.float32)
        rewards = torch.as_tensor(rewards_np, dtype=torch.float32, device=args.device)
        advantages = rewards - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        policy_loss_value = torch.tensor(0.0, device=args.device)
        value_loss_value = torch.tensor(0.0, device=args.device)
        for _ in range(max(1, int(args.ppo_epochs))):
            distribution = model.distribution(state_batch)
            log_prob = distribution.log_prob(sampled).sum(dim=-1)
            ratio = (log_prob - old_log_prob).exp()
            clipped_ratio = ratio.clamp(1.0 - args.clip_ratio, 1.0 + args.clip_ratio)
            policy_loss = -(torch.minimum(ratio * advantages, clipped_ratio * advantages)).mean()
            _, _, current_values = model(state_batch)
            value_loss = 0.5 * (current_values - rewards).pow(2).mean()
            entropy = distribution.entropy().sum(dim=-1).mean()
            loss = policy_loss + value_loss - 0.01 * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            policy_loss_value = policy_loss.detach()
            value_loss_value = value_loss.detach()

        with torch.no_grad():
            val_mean, _, _ = model(val_states)
        train_metrics = _metrics(sampled_np, train_targets[batch_indices])
        val_metrics = _metrics(val_mean.cpu().numpy(), val_targets)
        row = {
            "episode": episode,
            "train_reward": train_metrics["reward"],
            "train_success_percent": train_metrics["success_percent"],
            "val_reward": val_metrics["reward"],
            "val_success_percent": val_metrics["success_percent"],
            "policy_loss": float(policy_loss_value.cpu()),
            "value_loss": float(value_loss_value.cpu()),
        }
        writer.writerow(row)
        if val_metrics["success_percent"] > best_success:
            best_success = val_metrics["success_percent"]
            best_episode = episode
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "action_dim": ACTION_DIM,
                    "state_size": int(states.shape[-1]),
                    "episode": episode,
                    "validation": val_metrics,
                    "seed": args.seed,
                },
                output_dir / "best_checkpoint.pt",
            )
        if episode == 1 or episode % max(1, args.episodes // 20) == 0:
            print(
                f"episode={episode} train_reward={train_metrics['reward']:.3f} "
                f"val_success={val_metrics['success_percent']:.1f}%"
            )
    metrics_file.close()
    summary = {
        "episodes": int(args.episodes),
        "train_count": int(len(train_idx)),
        "validation_count": int(len(val_idx)),
        "best_episode": int(best_episode),
        "best_validation_success_percent": float(best_success),
        "seed": int(args.seed),
        "uses_color_features": False,
        "policy": "DepthGraspActorCritic_PPO_contextual_bandit",
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
