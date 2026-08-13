"""Memory-efficient replay storage for depth-image SAC training."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from stable_baselines3.common.buffers import BaseBuffer, DictReplayBuffer
    from stable_baselines3.common.type_aliases import DictReplayBufferSamples
except ImportError:  # pragma: no cover - optional RL dependency
    BaseBuffer = object  # type: ignore[misc,assignment]
    DictReplayBuffer = object  # type: ignore[misc,assignment]
    DictReplayBufferSamples = None  # type: ignore[assignment]


class CompressedDepthDictReplayBuffer(DictReplayBuffer):  # type: ignore[misc]
    """Store normalized depth as uint8 while retaining float proprioception.

    SB3's default DictReplayBuffer allocates both observations and next
    observations using each observation space dtype. A 128x128 float32 image
    therefore costs about 12 GiB for only 100k transitions. This buffer keeps
    depth at 8-bit precision in memory and restores it to [0, 1] on sampling.
    """

    depth_key = "depth"

    def __init__(
        self,
        buffer_size: int,
        observation_space: Any,
        action_space: Any,
        device: Any = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ) -> None:
        if optimize_memory_usage:
            raise ValueError("Compressed dict replay already minimizes image storage")
        # Bypass DictReplayBuffer.__init__: it would transiently allocate the
        # complete float32 image buffer before we could replace it with uint8.
        BaseBuffer.__init__(self, max(int(buffer_size) // int(n_envs), 1), observation_space, action_space, device, n_envs)
        if self.depth_key not in self.obs_shape:
            raise ValueError(f"Observation space must contain '{self.depth_key}'")
        self.observations: dict[str, np.ndarray] = {}
        self.next_observations: dict[str, np.ndarray] = {}
        for key, shape in self.obs_shape.items():
            dtype = np.uint8 if key == self.depth_key else observation_space[key].dtype
            array_shape = (self.buffer_size, self.n_envs, *shape)
            self.observations[key] = np.empty(array_shape, dtype=dtype)
            self.next_observations[key] = np.empty(array_shape, dtype=dtype)
        action_dtype = self._maybe_cast_dtype(action_space.dtype)
        self.actions = np.empty((self.buffer_size, self.n_envs, self.action_dim), dtype=action_dtype)
        self.rewards = np.empty((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.empty((self.buffer_size, self.n_envs), dtype=np.float32)
        self.handle_timeout_termination = bool(handle_timeout_termination)
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.position_ids = np.full((self.buffer_size, self.n_envs), -1, dtype=np.int8)

    @staticmethod
    def _encode_depth(value: np.ndarray) -> np.ndarray:
        return np.rint(np.clip(np.asarray(value, dtype=np.float32), 0.0, 1.0) * 255.0).astype(np.uint8)

    @staticmethod
    def _decode_depth(value: np.ndarray) -> np.ndarray:
        return np.asarray(value, dtype=np.float32) / 255.0

    def add(
        self,
        obs: dict[str, np.ndarray],
        next_obs: dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        write_position = int(self.pos)
        encoded_obs = dict(obs)
        encoded_next = dict(next_obs)
        encoded_obs[self.depth_key] = self._encode_depth(obs[self.depth_key])
        encoded_next[self.depth_key] = self._encode_depth(next_obs[self.depth_key])
        super().add(encoded_obs, encoded_next, action, reward, done, infos)
        self.position_ids[write_position] = np.asarray(
            [int(info.get("stage_a_position_index", -1)) for info in infos],
            dtype=np.int8,
        )

    def sample(self, batch_size: int, env: Any = None) -> Any:
        """Balance canonical Stage A positions when position metadata exists."""
        upper_bound = self.buffer_size if self.full else self.pos
        if upper_bound <= 0:
            raise ValueError("Cannot sample from an empty replay buffer")
        # The project currently permits one simulator environment. Keep a
        # uniform fallback for other stages and old replay files.
        labels = self.position_ids[:upper_bound, 0]
        groups = [int(value) for value in np.unique(labels) if int(value) >= 0]
        if len(groups) < 2:
            indices = np.random.randint(0, upper_bound, size=batch_size)
            return self._get_samples(indices, env=env)
        per_group = int(np.ceil(batch_size / len(groups)))
        pieces = []
        for group in groups:
            candidates = np.flatnonzero(labels == group)
            pieces.append(np.random.choice(candidates, size=per_group, replace=True))
        indices = np.concatenate(pieces)[:batch_size]
        np.random.shuffle(indices)
        return self._get_samples(indices, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env: Any = None) -> Any:
        env_indices = np.random.randint(0, high=self.n_envs, size=len(batch_inds))
        observations = {
            key: value[batch_inds, env_indices]
            for key, value in self.observations.items()
        }
        next_observations = {
            key: value[batch_inds, env_indices]
            for key, value in self.next_observations.items()
        }
        observations[self.depth_key] = self._decode_depth(observations[self.depth_key])
        next_observations[self.depth_key] = self._decode_depth(next_observations[self.depth_key])
        observations = self._normalize_obs(observations, env)
        next_observations = self._normalize_obs(next_observations, env)
        return DictReplayBufferSamples(
            observations={key: self.to_torch(value) for key, value in observations.items()},
            actions=self.to_torch(self.actions[batch_inds, env_indices]),
            next_observations={key: self.to_torch(value) for key, value in next_observations.items()},
            dones=self.to_torch(
                self.dones[batch_inds, env_indices] * (1.0 - self.timeouts[batch_inds, env_indices])
            ).reshape(-1, 1),
            rewards=self.to_torch(
                self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env)
            ),
        )
