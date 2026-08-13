from __future__ import annotations

import numpy as np
import gymnasium as gym

from compressed_replay_buffer import CompressedDepthDictReplayBuffer
from end_to_end_grasp_env import EndToEndGraspEnv
from train_depth_grasp_rl import parse_sac_ent_coef


def _spaces():
    observation = gym.spaces.Dict(
        {
            "depth": gym.spaces.Box(0.0, 1.0, shape=(1, 8, 8), dtype=np.float32),
            "proprio": gym.spaces.Box(-1.0, 1.0, shape=(14,), dtype=np.float32),
        }
    )
    action = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
    return observation, action


def test_compressed_replay_stores_uint8_and_decodes_float() -> None:
    observation_space, action_space = _spaces()
    buffer = CompressedDepthDictReplayBuffer(4, observation_space, action_space, device="cpu")
    depth = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(1, 1, 8, 8)
    obs = {"depth": depth, "proprio": np.zeros((1, 14), dtype=np.float32)}
    next_obs = {"depth": 1.0 - depth, "proprio": np.ones((1, 14), dtype=np.float32)}
    buffer.add(obs, next_obs, np.zeros((1, 2), dtype=np.float32), np.asarray([1.0]), np.asarray([0.0]), [{}])
    assert buffer.observations["depth"].dtype == np.uint8
    sample = buffer.sample(1)
    assert sample.observations["depth"].dtype.is_floating_point
    assert float(sample.observations["depth"].min()) >= 0.0
    assert float(sample.observations["depth"].max()) <= 1.0


def test_compressed_replay_records_canonical_position() -> None:
    observation_space, action_space = _spaces()
    buffer = CompressedDepthDictReplayBuffer(8, observation_space, action_space, device="cpu")
    obs = {"depth": np.zeros((1, 1, 8, 8), dtype=np.float32), "proprio": np.zeros((1, 14), dtype=np.float32)}
    for position in range(5):
        buffer.add(obs, obs, np.zeros((1, 2), dtype=np.float32), np.asarray([0.0]), np.asarray([0.0]), [{"stage_a_position_index": position}])
    assert buffer.position_ids[:5, 0].tolist() == [0, 1, 2, 3, 4]
    sample = buffer.sample(5)
    assert sample.actions.shape == (5, 2)


def test_stage_a_teacher_action_is_bounded_and_points_to_target() -> None:
    class FakeSim:
        @staticmethod
        def getObjectPosition(handle, _base):
            return [0.0, 0.0, 0.1] if handle == 1 else [0.02, -0.04, 0.02]

    env = EndToEndGraspEnv(curriculum_stage="A")
    env.sim = FakeSim()
    env.tip = 1
    env.robot_base = 3
    env.episode_target_handle = 2
    action = env.stage_a_expert_action()
    np.testing.assert_allclose(action, [0.5, -1.0], atol=1e-6)
    assert np.max(np.abs(action)) <= 1.0


def test_sac_entropy_parser_accepts_auto_and_float() -> None:
    assert parse_sac_ent_coef("auto") == "auto"
    assert parse_sac_ent_coef("auto_0.2") == "auto_0.2"
    assert parse_sac_ent_coef("0.05") == 0.05
