"""Optional depth-to-6D PPO components.

The policy is deliberately downstream of geometry recognition. It proposes a
pixel, in-plane yaw and surface normal from depth only; PBVS, IK, collision
checking and connector/lift verification remain authoritative.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    import torch
    from torch import nn
    from torch.distributions import Normal
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    nn = None  # type: ignore
    Normal = None  # type: ignore


ACTION_DIM = 7
STATE_SIZE = 128


def require_torch() -> Any:
    if torch is None:
        raise RuntimeError(
            "Depth PPO requires PyTorch. Install optional dependencies with "
            "'.venv\\Scripts\\python.exe -m pip install -r requirements-rl.txt'."
        )
    return torch


def normalize_depth(depth: np.ndarray, near: float = 0.05, far: float = 1.5) -> np.ndarray:
    value = np.asarray(depth, dtype=np.float32)
    value = np.nan_to_num(value, nan=far, posinf=far, neginf=near)
    return np.clip((value - float(near)) / max(float(far - near), 1e-6), 0.0, 1.0)


def encode_action(
    pixel_uv: tuple[float, float],
    width: int,
    height: int,
    yaw_rad: float,
    normal: np.ndarray,
) -> np.ndarray:
    normal_value = np.asarray(normal, dtype=np.float32)
    normal_value /= max(float(np.linalg.norm(normal_value)), 1e-8)
    return np.asarray(
        [
            2.0 * float(pixel_uv[0]) / max(width - 1, 1) - 1.0,
            2.0 * float(pixel_uv[1]) / max(height - 1, 1) - 1.0,
            math.sin(2.0 * float(yaw_rad)),
            math.cos(2.0 * float(yaw_rad)),
            *normal_value.tolist(),
        ],
        dtype=np.float32,
    )


def decode_pixel(action: np.ndarray, width: int, height: int) -> tuple[float, float]:
    value = np.asarray(action, dtype=np.float32)
    return (
        float(np.clip((value[0] + 1.0) * 0.5 * (width - 1), 0.0, width - 1)),
        float(np.clip((value[1] + 1.0) * 0.5 * (height - 1), 0.0, height - 1)),
    )


def action_reward(action: np.ndarray, target: np.ndarray) -> tuple[float, bool, dict[str, float]]:
    """One-step contextual-bandit reward used by the PPO trainer."""

    predicted = np.asarray(action, dtype=np.float32)
    expected = np.asarray(target, dtype=np.float32)
    position_error = float(np.linalg.norm(predicted[:2] - expected[:2]))
    yaw_dot = float(np.clip(np.dot(predicted[2:4], expected[2:4]), -1.0, 1.0))
    yaw_error = math.acos(yaw_dot)
    predicted_normal = predicted[4:7] / max(float(np.linalg.norm(predicted[4:7])), 1e-8)
    expected_normal = expected[4:7] / max(float(np.linalg.norm(expected[4:7])), 1e-8)
    normal_error = math.acos(float(np.clip(np.dot(predicted_normal, expected_normal), -1.0, 1.0)))
    success = position_error <= 0.12 and yaw_error <= math.radians(15.0) and normal_error <= math.radians(18.0)
    reward = (
        1.25 * math.exp(-((position_error / 0.32) ** 2))
        + 0.65 * math.exp(-((yaw_error / math.radians(45.0)) ** 2))
        + 0.90 * math.exp(-((normal_error / math.radians(35.0)) ** 2))
        + (2.0 if success else 0.0)
        - 0.05
    )
    return reward, success, {
        "position_error": position_error,
        "yaw_error_rad": yaw_error,
        "normal_error_rad": normal_error,
    }


if torch is not None:

    class DepthGraspActorCritic(nn.Module):
        def __init__(self, action_dim: int = ACTION_DIM) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 32, 5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, 5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
            self.actor = nn.Sequential(nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, action_dim))
            self.critic = nn.Sequential(nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1))
            self.log_std = nn.Parameter(torch.full((action_dim,), -1.0))

        def forward(self, state: Any) -> tuple[Any, Any, Any]:
            features = self.encoder(state)
            mean = torch.tanh(self.actor(features))
            value = self.critic(features).squeeze(-1)
            std = self.log_std.clamp(-4.0, 1.0).exp().expand_as(mean)
            return mean, std, value

        def distribution(self, state: Any) -> Any:
            mean, std, _ = self.forward(state)
            return Normal(mean, std)

else:

    class DepthGraspActorCritic:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            require_torch()


def make_state_tensor(states: np.ndarray, device: str = "cpu") -> Any:
    backend = require_torch()
    value = np.asarray(states, dtype=np.float32)
    if value.ndim == 3:
        value = value[:, None, :, :]
    return backend.as_tensor(value, dtype=backend.float32, device=device)
