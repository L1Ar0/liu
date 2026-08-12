"""Policy definitions shared by the end-to-end grasp environment and PPO.

Actions are normalized Cartesian end-effector increments.  The policy never
receives a target id, a fitted primitive, or a ground-truth object label.
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
ACTION_NAMES = ("dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper")
# Physical limits represented by a normalized action in [-1, 1].
TRANSLATION_STEP_M = 0.005
ROTATION_STEP_RAD = math.radians(5.0)


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


def decode_cartesian_action(action: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Convert normalized policy output to ``(translation_m, rotvec_rad, gripper)``."""

    value = np.asarray(action, dtype=np.float32).reshape(-1)
    if value.size != ACTION_DIM:
        raise ValueError(f"Expected {ACTION_DIM} action values, got {value.size}")
    value = np.clip(value, -1.0, 1.0)
    return (
        value[:3].astype(np.float64) * TRANSLATION_STEP_M,
        value[3:6].astype(np.float64) * ROTATION_STEP_RAD,
        float(value[6]),
    )


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
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
            )
            # The standalone actor-critic remains useful for lightweight
            # experiments.  End-to-end SB3 training uses the Dict extractor
            # below, which also consumes robot state.
            self.actor = nn.Sequential(nn.Linear(128 * 4 * 4, 128), nn.Tanh(), nn.Linear(128, action_dim))
            self.critic = nn.Sequential(nn.Linear(128 * 4 * 4, 128), nn.Tanh(), nn.Linear(128, 1))
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


if torch is not None:
    try:
        from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    except ImportError:  # pragma: no cover - optional dependency
        BaseFeaturesExtractor = None  # type: ignore

    if BaseFeaturesExtractor is not None:

        class DepthProprioFeaturesExtractor(BaseFeaturesExtractor):
            """CNN depth encoder fused with joint, velocity and gripper state."""

            def __init__(self, observation_space: Any, features_dim: int = 256) -> None:
                super().__init__(observation_space, features_dim)
                depth_space = observation_space.spaces["depth"]
                proprio_space = observation_space.spaces["proprio"]
                channels, height, width = depth_space.shape
                self.cnn = nn.Sequential(
                    nn.Conv2d(channels, 32, 5, stride=2, padding=2),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, 5, stride=2, padding=2),
                    nn.ReLU(),
                    nn.Conv2d(64, 128, 3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((4, 4)),
                    nn.Flatten(),
                )
                with torch.no_grad():
                    sample = torch.zeros(1, channels, height, width)
                    cnn_dim = int(self.cnn(sample).shape[1])
                self.proprio = nn.Sequential(
                    nn.Linear(int(np.prod(proprio_space.shape)), 64),
                    nn.Tanh(),
                )
                self.fusion = nn.Sequential(
                    nn.Linear(cnn_dim + 64, features_dim),
                    nn.ReLU(),
                )

            def forward(self, observations: dict[str, Any]) -> Any:
                depth = observations["depth"].float()
                proprio = observations["proprio"].float().flatten(start_dim=1)
                return self.fusion(torch.cat((self.cnn(depth), self.proprio(proprio)), dim=1))

    else:
        DepthProprioFeaturesExtractor = None  # type: ignore
else:
    DepthProprioFeaturesExtractor = None  # type: ignore
