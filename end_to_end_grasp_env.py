"""Interactive Gymnasium environment for end-to-end visual grasping.

    The policy observes only a depth image and proprioception.  Scene objects are
used internally for simulator feedback and reward computation; no object id or
ground-truth pose is placed in the observation.  Cartesian actions are solved
    through the existing incremental simIK controller.  By default, physics is
    used only while resetting a scene; rollout dynamics remain disabled so the
    kinematic IK controller cannot fight dynamic RG2 links and objects.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - optional training dependency
    gym = None  # type: ignore
    spaces = None  # type: ignore

from depth_grasp_rl import (
    ACTION_DIM,
    STATE_SIZE,
    ROTATION_STEP_RAD,
    TRANSLATION_STEP_M,
    decode_cartesian_action,
    normalize_depth,
)


ROOT = Path(__file__).resolve().parent
OBSERVATION_POSE_FILE = ROOT / "camera_output" / "initial_observation_joints.json"


@dataclass(frozen=True)
class StageConfig:
    object_count: int | None
    action_mask: tuple[float, ...]
    distance_mode: str
    distance_tolerance_m: float
    approach_gain: float
    approach_clip: float
    smoothness_weight: float
    step_penalty: float
    collision_penalty: float
    ik_penalty: float
    joint_limit_penalty: float
    one_side_contact_reward: float
    bilateral_reward: float
    connector_reward: float
    lift_progress_gain: float
    success_reward: float
    timeout_penalty: float
    premature_close_penalty: float
    allows_connector: bool
    inactive_action_penalty: float = 0.03


CURRICULUM_STAGES = ("A", "B", "C", "D", "E")
STAGE_INDEX = {stage: index for index, stage in enumerate(CURRICULUM_STAGES)}
STAGE_CONFIGS = {
    # A: lateral alignment only. Z, rotation and gripper commands are masked.
    "A": StageConfig(1, (1, 1, 0, 0, 0, 0, 0), "lateral", 0.025, 1.00, 1.0, 0.01, 0.01, 2.0, 2.0, 4.0, 0.0, 0.0, 0.0, 0.0, 15.0, 5.0, 0.0, False),
    # B: move in XYZ to a collision-free point above the object.
    "B": StageConfig(1, (1, 1, 1, 0, 0, 0, 0), "pregrasp", 0.018, 1.25, 1.25, 0.01, 0.01, 3.0, 3.0, 5.0, 0.0, 0.0, 0.0, 0.0, 25.0, 7.5, 0.0, False),
    # C: top-down translation/yaw plus gripper; bilateral contact ends the task.
    "C": StageConfig(1, (1, 1, 1, 0, 0, 1, 1), "grasp", 0.075, 1.00, 1.0, 0.015, 0.01, 4.0, 4.0, 6.0, 2.0, 12.0, 0.0, 0.0, 40.0, 10.0, 1.0, False),
    # D: complete single-object connector grasp and lift.
    "D": StageConfig(1, (1, 1, 1, 0, 0, 1, 1), "grasp", 0.075, 0.75, 0.75, 0.02, 0.01, 5.0, 5.0, 8.0, 2.0, 8.0, 20.0, 4.0, 100.0, 15.0, 1.0, True),
    # E: the same task in multi-object, stacked, occluded and noisy scenes.
    "E": StageConfig(None, (1, 1, 1, 1, 1, 1, 1), "grasp", 0.075, 0.60, 0.60, 0.025, 0.02, 7.5, 6.0, 10.0, 2.0, 8.0, 25.0, 5.0, 120.0, 20.0, 1.5, True),
}


def curriculum_stage_config(stage: str) -> StageConfig:
    value = str(stage).upper()
    if value not in STAGE_CONFIGS:
        raise ValueError(f"Unknown curriculum stage: {stage}")
    return STAGE_CONFIGS[value]


def stage_task_success(
    stage: str,
    *,
    grasp_success: bool,
    stage_contact_success: bool,
    stage_distance_success: bool,
) -> bool:
    """Map the physical event to the curriculum objective for one episode."""
    value = str(stage).upper()
    if value in {"A", "B"}:
        return bool(stage_distance_success)
    if value == "C":
        return bool(stage_contact_success)
    return bool(grasp_success)


def pregrasp_target_point(
    center: np.ndarray,
    size_m: np.ndarray | list[float] | tuple[float, ...],
    rotation_matrix: np.ndarray | None = None,
    standoff_m: float = 0.07,
) -> np.ndarray:
    """Return a point above an object's highest world-Z extent."""
    center_value = np.asarray(center, dtype=np.float64).reshape(3)
    size_value = np.asarray(size_m, dtype=np.float64).reshape(3)
    if rotation_matrix is None:
        rotation = np.eye(3, dtype=np.float64)
    else:
        rotation = np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
    half_height = 0.5 * float(np.sum(np.abs(rotation[2, :]) * size_value))
    return center_value + np.asarray([0.0, 0.0, half_height + max(0.0, float(standoff_m))])


if gym is None:

    class EndToEndGraspEnv:  # type: ignore[no-redef]
        """Importable placeholder that reports the missing optional package."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError(
                "EndToEndGraspEnv requires Gymnasium. Install requirements-rl.txt."
            )

else:

    class EndToEndGraspEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(
            self,
            scene_mode: str | None = None,
            planned_layout: str | None = None,
            max_steps: int = 80,
            sim_steps_per_action: int = 4,
            image_size: int = STATE_SIZE,
            seed: int | None = None,
            connector_enabled: bool = True,
            headless: bool = True,
            curriculum_stage: str = "E",
            execution_mode: str | None = None,
        ) -> None:
            super().__init__()
            self.scene_mode = str(scene_mode or os.environ.get("ROBOT_GRASP_SCENE_MODE", "physics"))
            self.planned_layout = planned_layout
            self.max_steps = max(1, int(max_steps))
            self.sim_steps_per_action = max(1, int(sim_steps_per_action))
            self.image_size = int(image_size)
            self.connector_enabled = bool(connector_enabled)
            self.headless = bool(headless)
            self.curriculum_stage = str(curriculum_stage).upper()
            if self.curriculum_stage not in STAGE_CONFIGS:
                raise ValueError("Curriculum stage must be A, B, C, D, or E")
            self._pending_curriculum_stage = self.curriculum_stage
            self.execution_mode = self._normalize_execution_mode(execution_mode)
            self._seed = seed
            self._base_seed = seed
            self._episode_index = 0
            self.client: Any = None
            self.sim: Any = None
            self.sim_ik: Any = None
            self.camera: int | None = None
            self.tip: int | None = None
            self.robot_base: int | None = None
            self.joints: list[int] = []
            self.ik: Any = None
            self.camera_params: dict[str, Any] = {}
            self.table_plane = np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float64)
            self.object_handles: list[int] = []
            self.connector_handle: int | None = None
            self.attached_handle: int | None = None
            self.attached_initial_z: float | None = None
            self._finger_groups: tuple[list[int], list[int]] = ([], [])
            self._gripper_drive_joint: int | None = None
            self._table_handles: list[int] = []
            self._previous_action = np.zeros(ACTION_DIM, dtype=np.float32)
            self._previous_distance = 0.0
            self._gripper_state = 0.0
            self._step_count = 0
            self._lift_hold_count = 0
            self._previous_lift = 0.0
            self._previous_left_contact = False
            self._previous_right_contact = False
            self._depth_noise_std_m = 0.0
            self._robot_shape_flags: list[tuple[int, int, int]] = []
            self.episode_target_handle: int | None = None
            self.object_records: list[dict[str, Any]] = []
            self._records_by_handle: dict[int, dict[str, Any]] = {}
            self._stage_reference_distance = 0.0

            self.action_space = spaces.Box(-1.0, 1.0, shape=(ACTION_DIM,), dtype=np.float32)
            self.observation_space = spaces.Dict(
                {
                    "depth": spaces.Box(0.0, 1.0, shape=(1, self.image_size, self.image_size), dtype=np.float32),
                    # q(7), dq(7), gripper(1), previous action(7), stage one-hot(5)
                    "proprio": spaces.Box(-1.0, 1.0, shape=(27,), dtype=np.float32),
                }
            )

        @staticmethod
        def _normalize_execution_mode(value: str | None) -> str:
            mode = str(value or os.environ.get("ROBOT_GRASP_EXECUTION_MODE", "settle_then_kinematic")).strip().lower()
            aliases = {
                "safe": "settle_then_kinematic",
                "kinematic_only": "kinematic",
                "settle": "settle_then_kinematic",
                "physics": "dynamic",
            }
            mode = aliases.get(mode, mode)
            if mode not in {"kinematic", "settle_then_kinematic", "dynamic"}:
                raise ValueError(
                    "execution_mode must be 'kinematic', 'settle_then_kinematic', or 'dynamic'"
                )
            return mode

        def _connect(self) -> None:
            if self.sim is not None:
                return
            try:
                from coppeliasim_zmqremoteapi_client import RemoteAPIClient
            except ImportError as exc:  # pragma: no cover - runtime dependency
                raise RuntimeError(
                    "CoppeliaSim training requires coppeliasim-zmqremoteapi-client."
                ) from exc
            self.client = RemoteAPIClient()
            self.sim = self.client.require("sim")
            self.sim_ik = self.client.require("simIK")

        def _stop(self) -> None:
            if self.sim is None:
                return
            try:
                state = self.sim.getSimulationState()
                if state != self.sim.simulation_stopped:
                    self.sim.stopSimulation()
                    for _ in range(100):
                        if self.sim.getSimulationState() == self.sim.simulation_stopped:
                            break
                        time.sleep(0.01)
            except Exception:
                pass

        def _load_scene(self, seed: int | None) -> None:
            """Stop, remove generated entities, randomize, and restore observation pose."""
            self._stop()
            # A previous safe episode may have made the RG2 links static. Restore
            # their original scene flags before loading the next episode so the
            # selected execution mode is applied from a known baseline.
            self._restore_robot_shape_flags()
            from point_cloud import find_unique_object_by_alias, get_kuka_joints_from_tip, get_camera_parameters
            import scene_randomizer as randomizer

            effective_mode, effective_layout = self._curriculum_scene(seed)
            os.environ["ROBOT_GRASP_SCENE_MODE"] = effective_mode
            if seed is not None:
                os.environ["ROBOT_GRASP_RANDOM_SEED"] = str(int(seed))
                random.seed(int(seed))
                np.random.seed(int(seed))
            if effective_layout:
                os.environ["ROBOT_GRASP_PLANNED_LAYOUT"] = effective_layout
            randomizer.SCENE_MODE = effective_mode
            self.camera = find_unique_object_by_alias(self.sim, self.sim.sceneobject_visionsensor, "rgbd_camera")
            self.tip = find_unique_object_by_alias(self.sim, self.sim.sceneobject_dummy, "gripper_tip")
            self._remove_old_connectors()
            self.joints = get_kuka_joints_from_tip(self.sim, self.tip)
            if len(self.joints) != 7:
                raise RuntimeError(f"Expected seven arm joints, found {len(self.joints)}")
            self.robot_base = int(self.sim.getObjectParent(self.joints[0]))
            # Scene placement and camera visibility must always be evaluated
            # from the canonical observation pose.  Previously this reset was
            # performed after randomization, so a policy that moved the wrist
            # in the previous episode could make the next workspace appear
            # completely outside the camera frustum.
            self._restore_observation_pose()
            existing = randomizer.find_existing_test_objects(self.sim)
            reference = randomizer.load_or_create_reference(self.sim, self.robot_base, existing)
            self.table_plane = np.asarray(
                [0.0, 0.0, 1.0, float(reference.get("table_z", 0.0))],
                dtype=np.float64,
            )
            randomizer.remove_existing_test_objects(self.sim, existing)
            camera_model = randomizer.build_camera_model(self.sim, self.camera, self.robot_base)
            # Training sees the real camera geometry, including partial
            # occlusion.  The production full-corner mask is too strict for
            # this oblique wrist camera and can make reset impossible.
            if self.curriculum_stage in {"A", "B", "C", "D", "E"}:
                width = int(camera_model["width"])
                height = int(camera_model["height"])
                camera_model["safe_bounds"] = {
                    "u_min": 0.03 * (width - 1),
                    "u_max": 0.97 * (width - 1),
                    "v_min": 0.03 * (height - 1),
                    "v_max": 0.97 * (height - 1),
                }
                camera_model["visibility_policy"] = "center"
            # Curriculum stage A must be generated as a one-object scene. The
            # previous post-generation filtering was too late: the separated
            # randomizer first tried to place all five default objects and
            # could fail before the environment had a chance to remove four.
            original_min_objects = randomizer.MIN_OBJECTS
            original_max_objects = randomizer.MAX_OBJECTS
            original_planned_count = None
            if effective_mode == "separated":
                curriculum_count = curriculum_stage_config(self.curriculum_stage).object_count
                if curriculum_count is not None:
                    randomizer.MIN_OBJECTS = curriculum_count
                    randomizer.MAX_OBJECTS = curriculum_count
            if effective_mode == "planned":
                import planned_scene_randomizer as planned_randomizer

                original_planned_count = planned_randomizer.PLANNED_OBJECT_COUNT
                planned_randomizer.PLANNED_OBJECT_COUNT = {
                    "C": 3,
                    "D": 4,
                }.get(self.curriculum_stage, original_planned_count)
            try:
                records = randomizer.generate_random_scene(self.sim, self.robot_base, reference, camera_model)
            except RuntimeError as exc:
                # The first attempt should already use the canonical pose. A
                # second attempt protects against a stale CoppeliaSim kinematic
                # update or an externally moved arm without hiding unrelated
                # generation errors.
                message = str(exc)
                if "无法找到足够的非重叠位置" not in message and "闅忔満宸ヤ綔鍖" not in message:
                    raise
                self._restore_observation_pose()
                camera_model = randomizer.build_camera_model(
                    self.sim, self.camera, self.robot_base
                )
                camera_model["safe_bounds"] = {
                    "u_min": 0.03 * (int(camera_model["width"]) - 1),
                    "u_max": 0.97 * (int(camera_model["width"]) - 1),
                    "v_min": 0.03 * (int(camera_model["height"]) - 1),
                    "v_max": 0.97 * (int(camera_model["height"]) - 1),
                }
                camera_model["visibility_policy"] = "center"
                records = randomizer.generate_random_scene(
                    self.sim, self.robot_base, reference, camera_model
                )
            finally:
                randomizer.MIN_OBJECTS = original_min_objects
                randomizer.MAX_OBJECTS = original_max_objects
                if original_planned_count is not None:
                    import planned_scene_randomizer as planned_randomizer

                    planned_randomizer.PLANNED_OBJECT_COUNT = original_planned_count
            records = self._apply_curriculum_object_count(records, seed)
            self.object_handles = [int(item["handle"]) for item in records if "handle" in item]
            self.object_records = list(records)
            self._records_by_handle = {
                int(item["handle"]): item for item in records if "handle" in item
            }
            if not self.object_handles:
                raise RuntimeError("Randomizer returned no usable object handles")
            # Stages A-D intentionally use a fixed, observable-by-training-task
            # single target. E is multi-object: a hidden random id would make the
            # task partially unobservable, so shaping and contact scan all objects.
            if self.curriculum_stage == "E":
                self.episode_target_handle = None
            else:
                self.episode_target_handle = int(self.object_handles[0])
            if effective_mode.lower() in {"physics", "dynamic", "settled", "drop"} and self.execution_mode == "dynamic":
                self._set_dynamic_scene_flags()
                self._randomize_object_dynamics(seed)
            elif self.execution_mode in {"kinematic", "settle_then_kinematic"}:
                self._freeze_rollout_scene()
            self.camera_params = get_camera_parameters(self.sim, self.camera, camera_model["width"], camera_model["height"])
            self._table_handles = self._find_table_handles()
            self._finger_groups = self._find_finger_groups()
            self._gripper_drive_joint = self._find_gripper_drive_joint()
            self._set_gripper(-1.0)
            # RG2 initialization may briefly start dynamics. simIK requires a
            # stable stopped/paused state when it snapshots the chain.
            if self.sim.getSimulationState() not in {
                self.sim.simulation_stopped,
                self.sim.simulation_paused,
            }:
                try:
                    self.sim.pauseSimulation()
                except Exception:
                    pass
                for _ in range(20):
                    if self.sim.getSimulationState() in {
                        self.sim.simulation_stopped,
                        self.sim.simulation_paused,
                    }:
                        break
                    try:
                        self.client.step()
                    except Exception:
                        time.sleep(0.005)
            self.ik = self._make_ik()

            print(
                "RL execution mode: "
                f"{self.execution_mode}; arm_control=kinematic_ik; "
                f"rollout_dynamics={'enabled' if self.execution_mode == 'dynamic' else 'disabled'}; "
                "grasp_proxy=connector",
                flush=True,
            )

        def set_curriculum_stage(self, stage: str) -> None:
            value = str(stage).upper()
            if value not in STAGE_CONFIGS:
                raise ValueError("Curriculum stage must be A, B, C, D, or E")
            self._pending_curriculum_stage = value

        def _curriculum_scene(self, seed: int | None) -> tuple[str, str | None]:
            rng = np.random.default_rng(seed)
            stage = self.curriculum_stage
            self._depth_noise_std_m = 0.0
            if stage == "A":
                return "separated", None
            if stage == "B":
                return "separated", None
            if stage == "C":
                return "separated", None
            if stage == "D":
                return "separated", None
            self._depth_noise_std_m = 0.002
            return self.scene_mode, self.planned_layout

        def _apply_curriculum_object_count(self, records: list[dict[str, Any]], seed: int | None) -> list[dict[str, Any]]:
            requested_count = curriculum_stage_config(self.curriculum_stage).object_count
            if requested_count is None or len(records) <= requested_count:
                return records
            rng = np.random.default_rng(seed)
            simple = [item for item in records if str(item.get("class", "")) in {"cube", "cylinder"}]
            candidates = simple if len(simple) >= requested_count else records
            selected_indices = rng.choice(len(candidates), size=requested_count, replace=False)
            selected = [candidates[int(index)] for index in np.atleast_1d(selected_indices)]
            selected_ids = {id(item) for item in selected}
            remove = [int(item["handle"]) for item in records if id(item) not in selected_ids and "handle" in item]
            if remove:
                self.sim.removeObjects(remove)
            return selected

        def _randomize_object_dynamics(self, seed: int | None) -> None:
            rng = np.random.default_rng(seed)
            engine_float = getattr(self.sim, "shapefloatparam_init_velocity_x", None)
            for handle in self.object_handles:
                # CoppeliaSim versions expose different material APIs.  Use a
                # per-shape dynamics reset universally and optional material
                # parameters where available.
                try:
                    self.sim.resetDynamicObject(handle)
                except Exception:
                    pass
                if engine_float is not None:
                    try:
                        self.sim.setObjectFloatParam(handle, engine_float, float(rng.uniform(-0.001, 0.001)))
                    except Exception:
                        pass

        def _set_dynamic_scene_flags(self) -> None:
            """Make generated shapes respondable while preserving the scene's baseline."""
            static_param = getattr(self.sim, "shapeintparam_static", None)
            respondable_param = getattr(self.sim, "shapeintparam_respondable", None)
            if static_param is None or respondable_param is None:
                return
            for handle in self.object_handles:
                try:
                    self.sim.setObjectInt32Param(handle, static_param, 0)
                    self.sim.setObjectInt32Param(handle, respondable_param, 1)
                except Exception:
                    pass

        def _capture_robot_shape_flags(self) -> None:
            if self._robot_shape_flags or self.robot_base is None:
                return
            static_param = getattr(self.sim, "shapeintparam_static", None)
            respondable_param = getattr(self.sim, "shapeintparam_respondable", None)
            if static_param is None or respondable_param is None:
                return
            for handle in self.sim.getObjectsInTree(self.robot_base, self.sim.sceneobject_shape, 0):
                try:
                    static = int(self.sim.getObjectInt32Param(handle, static_param))
                    respondable = int(self.sim.getObjectInt32Param(handle, respondable_param))
                    self._robot_shape_flags.append((int(handle), static, respondable))
                except Exception:
                    continue

        def _restore_robot_shape_flags(self) -> None:
            if self.sim is None or not self._robot_shape_flags:
                return
            static_param = getattr(self.sim, "shapeintparam_static", None)
            respondable_param = getattr(self.sim, "shapeintparam_respondable", None)
            if static_param is None or respondable_param is None:
                return
            for handle, static, respondable in self._robot_shape_flags:
                try:
                    self.sim.setObjectInt32Param(handle, static_param, static)
                    self.sim.setObjectInt32Param(handle, respondable_param, respondable)
                except Exception:
                    pass

        def _freeze_rollout_scene(self) -> None:
            """Freeze dynamic bodies before IK-only PPO rollout.

            Physics mode is still used during reset to create natural contacts.
            Once the scene is settled, static object bodies and RG2 links keep
            the visual scene intact without allowing a paused/kinematic arm to
            fight the dynamics engine during learning.
            """
            static_param = getattr(self.sim, "shapeintparam_static", None)
            respondable_param = getattr(self.sim, "shapeintparam_respondable", None)
            if static_param is None:
                return
            for handle in self.object_handles:
                try:
                    self.sim.setObjectInt32Param(handle, static_param, 1)
                    if respondable_param is not None:
                        self.sim.setObjectInt32Param(handle, respondable_param, 1)
                    self.sim.resetDynamicObject(handle)
                    if hasattr(self.sim, "setObjectVelocity"):
                        self.sim.setObjectVelocity(handle, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
                except Exception:
                    pass
            self._capture_robot_shape_flags()
            for handle, original_static, _respondable in self._robot_shape_flags:
                if original_static != 0:
                    continue
                try:
                    self.sim.setObjectInt32Param(handle, static_param, 1)
                    if respondable_param is not None:
                        self.sim.setObjectInt32Param(handle, respondable_param, 0)
                except Exception:
                    pass

        def _restore_observation_pose(self) -> None:
            if not OBSERVATION_POSE_FILE.exists():
                return
            payload = json.loads(OBSERVATION_POSE_FILE.read_text(encoding="utf-8"))
            values = [float(v) for v in payload.get("joint_positions_rad", [])]
            if len(values) == len(self.joints):
                for joint, value in zip(self.joints, values):
                    self.sim.setJointPosition(joint, value)

        def _make_ik(self) -> Any:
            from visual_servo_runner import IncrementalIK

            return IncrementalIK(self.sim, self.sim_ik, self.robot_base, self.tip)

        def _find_table_handles(self) -> list[int]:
            result: list[int] = []
            for handle in self.sim.getObjectsInTree(self.sim.handle_scene, self.sim.sceneobject_shape, 0):
                try:
                    alias = str(self.sim.getObjectAlias(handle)).lower()
                except Exception:
                    continue
                if "table" in alias or "floor" in alias:
                    result.append(int(handle))
            return result

        def _find_finger_groups(self) -> tuple[list[int], list[int]]:
            candidates: list[tuple[str, int]] = []
            roots = self.sim.getObjectsInTree(self.sim.handle_scene, self.sim.sceneobject_shape, 0)
            for handle in roots:
                try:
                    if self.sim.getObjectType(handle) != self.sim.sceneobject_shape:
                        continue
                    alias = str(self.sim.getObjectAlias(handle)).lower()
                except Exception:
                    continue
                if any(word in alias for word in ("finger", "pad", "jaw")):
                    candidates.append((alias, int(handle)))
            left = [h for alias, h in candidates if "left" in alias or "l_" in alias or "leftfinger" in alias]
            right = [h for alias, h in candidates if "right" in alias or "r_" in alias or "rightfinger" in alias]
            if not left or not right:
                # Keep two distinct finger shapes if the model uses opaque aliases.
                handles = [h for _alias, h in candidates]
                if len(handles) >= 2:
                    left, right = [handles[0]], [handles[1]]
            return left, right

        def _find_gripper_drive_joint(self) -> int | None:
            """Find RG2's scalar open/close joint for paused kinematic training."""
            for joint in self.sim.getObjectsInTree(
                self.sim.handle_scene, self.sim.sceneobject_joint, 0
            ):
                try:
                    alias = str(self.sim.getObjectAlias(joint)).lower().replace("_", "")
                    if "openclose" in alias and int(self.sim.getJointType(joint)) != int(self.sim.joint_spherical):
                        return int(joint)
                except Exception:
                    continue
            return None

        def _remove_old_connectors(self) -> None:
            handles: list[int] = []
            for handle in self.sim.getObjectsInTree(self.sim.handle_scene, self.sim.handle_all, 0):
                try:
                    alias = str(self.sim.getObjectAlias(handle)).lower()
                except Exception:
                    continue
                if alias.startswith("rl_connector_") or alias.startswith("grasp_connector_"):
                    while True:
                        try:
                            child = int(self.sim.getObjectChild(handle, 0))
                        except Exception:
                            child = -1
                        if child < 0:
                            break
                        self.sim.setObjectParent(child, -1, True)
                    handles.append(int(handle))
            if handles:
                try:
                    self.sim.removeObjects(handles)
                except Exception:
                    for handle in handles:
                        try:
                            self.sim.removeObject(handle)
                        except Exception:
                            pass

        def _set_gripper(self, value: float) -> None:
            from visual_servo_runner import (
                GRIPPER_CLOSE_VALUE,
                GRIPPER_OPEN_VALUE,
                GRIPPER_SIGNAL_NAME,
            )

            desired = -1.0 if value < 0.0 else 1.0
            if desired == self._gripper_state and self.sim is not None:
                return
            signal = GRIPPER_OPEN_VALUE if desired < 0 else GRIPPER_CLOSE_VALUE
            try:
                self.sim.clearFloatSignal(GRIPPER_SIGNAL_NAME)
            except Exception:
                pass
            try:
                self.sim.setInt32Signal(GRIPPER_SIGNAL_NAME, int(signal))
            except Exception:
                pass
            if self.execution_mode == "dynamic":
                self._advance_sim(max(2, self.sim_steps_per_action * 2))
            elif self._gripper_drive_joint is not None:
                try:
                    cyclic, interval = self.sim.getJointInterval(self._gripper_drive_joint)
                    if not cyclic:
                        lower = float(interval[0])
                        upper = lower + float(interval[1])
                        self.sim.setJointPosition(
                            self._gripper_drive_joint,
                            upper if desired < 0.0 else lower,
                        )
                except Exception:
                    pass
            self._gripper_state = desired

        def _detach(self) -> None:
            if self.attached_handle is not None:
                try:
                    self.sim.setObjectParent(self.attached_handle, -1, True)
                    self.sim.resetDynamicObject(self.attached_handle)
                except Exception:
                    pass
            if self.connector_handle is not None:
                try:
                    self.sim.removeObject(self.connector_handle)
                except Exception:
                    pass
            self.connector_handle = None
            self.attached_handle = None
            self.attached_initial_z = None

        def _advance_sim(self, steps: int | None = None) -> None:
            if self.sim is None or self.execution_mode != "dynamic":
                return
            count = max(1, int(steps or self.sim_steps_per_action))
            try:
                self.client.setStepping(True)
                if self.sim.getSimulationState() == self.sim.simulation_stopped:
                    self.sim.startSimulation()
                for _ in range(count):
                    self.client.step()
                if self.sim.getSimulationState() != self.sim.simulation_stopped:
                    self.sim.pauseSimulation()
                for _ in range(20):
                    if self.sim.getSimulationState() in {
                        self.sim.simulation_paused,
                        self.sim.simulation_stopped,
                    }:
                        break
                    try:
                        self.client.step()
                    except Exception:
                        time.sleep(0.005)
            except Exception:
                # Kinematic scenes may reject stepping; retaining a paused pose
                # is still useful for deterministic IK-only curriculum stages.
                pass

        def _capture_observation(self) -> dict[str, np.ndarray]:
            from point_cloud import capture_rgbd

            _rgb, depth, width, height = capture_rgbd(self.sim, self.camera, announce=False)
            resized = np.asarray(
                Image.fromarray(np.asarray(depth, dtype=np.float32), mode="F").resize(
                    (self.image_size, self.image_size), Image.Resampling.BILINEAR
                ),
                dtype=np.float32,
            )
            if self._depth_noise_std_m > 0.0:
                resized = resized + self.np_random.normal(0.0, self._depth_noise_std_m, resized.shape).astype(np.float32)
            params = self.camera_params or {"near": 0.05, "far": 1.5}
            depth_state = normalize_depth(resized, params["near"], params["far"])[None, ...]
            q = np.asarray([self.sim.getJointPosition(joint) for joint in self.joints], dtype=np.float64)
            velocities = []
            for joint in self.joints:
                try:
                    velocities.append(float(self.sim.getJointVelocity(joint)))
                except Exception:
                    velocities.append(0.0)
            q_state = np.clip(q / math.pi, -1.0, 1.0)
            dq_state = np.clip(np.asarray(velocities, dtype=np.float64) / 10.0, -1.0, 1.0)
            stage_one_hot = np.zeros(len(CURRICULUM_STAGES), dtype=np.float64)
            stage_one_hot[STAGE_INDEX[self.curriculum_stage]] = 1.0
            proprio = np.concatenate(
                (q_state, dq_state, [self._gripper_state], self._previous_action, stage_one_hot)
            ).astype(np.float32)
            return {"depth": depth_state.astype(np.float32), "proprio": proprio}

        def _stage_target_point(self, handle: int) -> np.ndarray | None:
            try:
                center = np.asarray(self.sim.getObjectPosition(handle, self.robot_base), dtype=np.float64)
            except Exception:
                return None
            if curriculum_stage_config(self.curriculum_stage).distance_mode != "pregrasp":
                return center
            record = self._records_by_handle.get(int(handle), {})
            size = record.get("size_m", [0.04, 0.04, 0.04])
            rotation = record.get("rotation_matrix")
            return pregrasp_target_point(center, size, rotation)

        def _distance_to_handle(self, handle: int, tip: np.ndarray) -> float:
            target = self._stage_target_point(handle)
            if target is None:
                return float("inf")
            if curriculum_stage_config(self.curriculum_stage).distance_mode == "lateral":
                return float(np.linalg.norm(tip[:2] - target[:2]))
            return float(np.linalg.norm(tip - target))

        def _target_distance(self) -> float:
            """Return target distance, using a smooth minimum for multi-object E."""
            if self.tip is None or self.robot_base is None or not self.object_handles:
                return float("inf")
            try:
                tip = np.asarray(self.sim.getObjectPosition(self.tip, self.robot_base), dtype=np.float64)
            except Exception:
                return float("inf")
            handles = ([self.episode_target_handle]
                       if self.episode_target_handle is not None
                       else list(self.object_handles))
            distances = [self._distance_to_handle(int(handle), tip) for handle in handles if handle is not None]
            distances = [value for value in distances if math.isfinite(value)]
            if not distances:
                return float("inf")
            if self.curriculum_stage == "E" and len(distances) > 1:
                tau = 0.02
                values = np.asarray(distances, dtype=np.float64)
                minimum = float(values.min())
                return float(max(0.0, minimum - tau * np.log(np.exp(-(values - minimum) / tau).sum())))
            return float(distances[0])

        def _nearest_object(self) -> tuple[int | None, float]:
            if self.tip is None or not self.object_handles:
                return None, float("inf")
            tcp = np.asarray(self.sim.getObjectPosition(self.tip, self.robot_base), dtype=np.float64)
            if self.attached_handle is not None:
                try:
                    pos = np.asarray(self.sim.getObjectPosition(self.attached_handle, self.robot_base), dtype=np.float64)
                    return self.attached_handle, float(np.linalg.norm(pos - tcp))
                except Exception:
                    pass
            best: tuple[int | None, float] = (None, float("inf"))
            handles = (
                [self.episode_target_handle]
                if self.episode_target_handle is not None
                else list(self.object_handles)
            )
            for handle in handles:
                if handle is None:
                    continue
                try:
                    if self.sim.getObjectParent(handle) == self.tip or handle == self.attached_handle:
                        continue
                    pos = np.asarray(self.sim.getObjectPosition(handle, self.robot_base), dtype=np.float64)
                    distance = float(np.linalg.norm(pos - tcp))
                    if distance < best[1]:
                        best = (handle, distance)
                except Exception:
                    continue
            return best

        def _collision(self, first: int, second: int) -> bool:
            try:
                result = self.sim.checkCollision(int(first), int(second))
                return bool(result[0] if isinstance(result, (list, tuple)) else result)
            except Exception:
                return False

        def _joint_limit_violation(self) -> bool:
            if self.ik is None:
                return False
            for joint, (lower, upper) in zip(self.joints, self.ik.limits):
                position = float(self.sim.getJointPosition(joint))
                if position < lower + math.radians(1.0) or position > upper - math.radians(1.0):
                    return True
            return False

        def _contact_candidate(self) -> tuple[int | None, bool, bool]:
            left, right = self._finger_groups
            if not left or not right:
                return None, False, False
            best: tuple[int | None, bool, bool] = (None, False, False)
            handles = (
                [self.episode_target_handle]
                if self.episode_target_handle is not None
                else list(self.object_handles)
            )
            for handle in handles:
                if handle is None:
                    continue
                left_hit = any(self._collision(finger, handle) for finger in left)
                right_hit = any(self._collision(finger, handle) for finger in right)
                if left_hit and right_hit:
                    return handle, True, True
                if left_hit or right_hit:
                    best = (handle, left_hit, right_hit)
            return best

        def _between_fingers(self, handle: int) -> bool:
            """Check that an object center lies inside the two-jaw envelope."""
            left, right = self._finger_groups
            if not left or not right:
                return False
            try:
                left_pos = np.asarray(self.sim.getObjectPosition(left[0], self.robot_base), dtype=np.float64)
                right_pos = np.asarray(self.sim.getObjectPosition(right[0], self.robot_base), dtype=np.float64)
                object_pos = np.asarray(self.sim.getObjectPosition(handle, self.robot_base), dtype=np.float64)
            except Exception:
                return False
            axis = right_pos - left_pos
            span = float(np.linalg.norm(axis))
            if span < 1e-8:
                return False
            unit = axis / span
            along = float(np.dot(object_pos - left_pos, unit))
            perpendicular = float(np.linalg.norm(object_pos - (left_pos + along * unit)))
            return -0.004 <= along <= span + 0.004 and perpendicular <= 0.035

        def _table_collision(self) -> bool:
            robot_shapes = self.sim.getObjectsInTree(self.robot_base, self.sim.sceneobject_shape, 0)
            return any(self._collision(int(robot), int(table)) for robot in robot_shapes for table in self._table_handles)

        def _attach(self, handle: int) -> None:
            if not self.connector_enabled or self.tip is None:
                return
            target_pose = list(self.sim.getObjectPose(handle, -1))
            try:
                self.connector_handle = int(self.sim.createDummy(0.002))
                self.sim.setObjectAlias(self.connector_handle, f"rl_connector_{handle}")
                self.sim.setObjectPose(self.connector_handle, target_pose, -1)
                self.sim.setObjectParent(self.connector_handle, self.tip, True)
                self.sim.setObjectParent(handle, self.connector_handle, True)
            except Exception:
                try:
                    self.sim.setObjectParent(handle, self.tip, True)
                except Exception:
                    self.connector_handle = None
            self.attached_handle = int(handle)
            self.attached_initial_z = float(self.sim.getObjectPosition(handle, self.robot_base)[2])

        def _lift_progress(self) -> float:
            if self.attached_handle is None or self.attached_initial_z is None:
                return 0.0
            current_z = float(self.sim.getObjectPosition(self.attached_handle, self.robot_base)[2])
            return max(0.0, current_z - self.attached_initial_z)

        def _cleanup(self) -> None:
            if self.ik is not None:
                try:
                    self.ik.close()
                except Exception:
                    pass
                self.ik = None
            # Do not leave RG2 dynamic flags modified in the shared CoppeliaSim
            # scene after a safe-mode training/evaluation process exits.
            self._restore_robot_shape_flags()
            self._stop()

        def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
            # SB3 passes seed=None on ordinary episode resets. Advance the
            # episode seed explicitly so training does not replay one scene.
            if seed is None:
                if self._base_seed is None:
                    self._base_seed = int(np.random.SeedSequence().generate_state(1)[0])
                seed = int(self._base_seed) + self._episode_index * 1000003
            self._episode_index += 1
            super().reset(seed=int(seed))
            self._seed = seed
            self._previous_action.fill(0.0)
            if self._pending_curriculum_stage != self.curriculum_stage:
                self.curriculum_stage = self._pending_curriculum_stage
            self._gripper_state = 0.0
            self._step_count = 0
            self._lift_hold_count = 0
            self._previous_lift = 0.0
            self._previous_left_contact = False
            self._previous_right_contact = False
            self.connector_handle = None
            self.attached_handle = None
            self.attached_initial_z = None
            if self.ik is not None:
                try:
                    self.ik.close()
                except Exception:
                    pass
                self.ik = None
            self._connect()
            self._load_scene(seed)
            _handle, _ = self._nearest_object()
            distance = self._target_distance()
            self._previous_distance = distance
            return self._capture_observation(), {
                "object_count": len(self.object_handles),
                "seed": seed,
                "execution_mode": self.execution_mode,
                "arm_control": "kinematic_ik",
                "rollout_dynamics": self.execution_mode == "dynamic",
                "grasp_proxy": "connector" if self.connector_enabled else "none",
                "curriculum_stage": self.curriculum_stage,
                "target_handle_hidden": True,
            }

        def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
            if self.ik is None:
                raise RuntimeError("reset() must be called before step()")
            config = curriculum_stage_config(self.curriculum_stage)
            policy_action = np.asarray(action, dtype=np.float32).reshape(ACTION_DIM)
            policy_action = np.clip(policy_action, -1.0, 1.0)
            action_mask = np.asarray(config.action_mask, dtype=np.float32)
            inactive_action = policy_action * (1.0 - action_mask)
            executed_action = policy_action * action_mask
            translation, rotvec, gripper = decode_cartesian_action(executed_action)
            current = self.ik.current_pose()
            target = current.copy()
            target[:3, 3] += translation
            target[:3, :3] = current[:3, :3] @ Rotation.from_rotvec(rotvec).as_matrix()
            table_z = float(self.table_plane[3])
            collision = False
            ik_failed = False
            try:
                if float(target[2, 3]) < table_z + 0.015:
                    collision = True
                else:
                    self.ik.apply(target)
                    self._advance_sim()
            except Exception:
                ik_failed = True
            released = False
            if gripper < -0.2:
                if self.attached_handle is not None:
                    released = True
                    self._detach()
                self._set_gripper(-1.0)
            elif gripper > 0.5:
                self._set_gripper(1.0)

            candidate, left_hit, right_hit = self._contact_candidate()
            tcp_to_object = float("inf")
            if candidate is not None:
                tcp = np.asarray(self.sim.getObjectPosition(self.tip, self.robot_base), dtype=np.float64)
                center = np.asarray(self.sim.getObjectPosition(candidate, self.robot_base), dtype=np.float64)
                tcp_to_object = float(np.linalg.norm(tcp - center))
            bilateral = bool(left_hit and right_hit)
            connector_created = False
            if (
                config.allows_connector
                and self._gripper_state > 0.0
                and self.curriculum_stage in {"D", "E"}
                and bilateral
                and candidate is not None
                and self._between_fingers(candidate)
                and tcp_to_object <= 0.075
                and not self._table_collision()
            ):
                if self.attached_handle is None:
                    self._attach(candidate)
                    connector_created = self.attached_handle is not None

            _nearest, _ = self._nearest_object()
            distance = self._target_distance()
            distance_progress = 0.0 if not math.isfinite(self._previous_distance) else self._previous_distance - distance
            self._previous_distance = distance
            stage_distance_success = (
                distance <= config.distance_tolerance_m
                and self._step_count >= 2
            )
            stage_contact_success = (
                self.curriculum_stage == "C"
                and candidate is not None
                and bilateral
                and self._gripper_state > 0.0
                and self._between_fingers(candidate)
                and tcp_to_object <= config.distance_tolerance_m
                and not self._table_collision()
            )
            lift = self._lift_progress()
            if lift >= 0.015:
                self._lift_hold_count += 1
            else:
                self._lift_hold_count = 0
            grasp_success = self.attached_handle is not None and self._lift_hold_count >= 3
            dropped = released or (self.attached_handle is not None and self._gripper_state < 0.0)
            joint_limit = self._joint_limit_violation()
            smoothness = float(np.mean(np.square(executed_action - self._previous_action)))
            approach_progress = float(np.clip(distance_progress / max(TRANSLATION_STEP_M, 1e-6), -1.0, 1.0))
            reward = config.approach_gain * approach_progress
            reward = float(np.clip(reward, -config.approach_clip, config.approach_clip))
            reward -= config.smoothness_weight * smoothness
            reward -= config.step_penalty
            reward -= config.collision_penalty if collision or self._table_collision() else 0.0
            reward -= config.ik_penalty if ik_failed else 0.0
            reward -= config.joint_limit_penalty if joint_limit else 0.0
            reward -= config.inactive_action_penalty * float(np.mean(np.square(inactive_action)))
            if left_hit and not self._previous_left_contact:
                reward += config.one_side_contact_reward
            if right_hit and not self._previous_right_contact:
                reward += config.one_side_contact_reward
            if bilateral and self._gripper_state > 0.0 and not (
                self._previous_left_contact and self._previous_right_contact
            ):
                reward += config.bilateral_reward
            reward += config.connector_reward if connector_created else 0.0
            lift_progress = max(0.0, lift - self._previous_lift)
            reward += config.lift_progress_gain * min(lift_progress / 0.015, 1.0)
            self._previous_lift = lift
            task_success = stage_task_success(
                self.curriculum_stage,
                grasp_success=grasp_success,
                stage_contact_success=stage_contact_success,
                stage_distance_success=stage_distance_success,
            )
            reward += config.success_reward if task_success else 0.0
            reward -= config.premature_close_penalty if (
                self._gripper_state > 0.0 and not bilateral and self.curriculum_stage in {"A", "B"}
            ) else 0.0
            self._previous_left_contact = left_hit
            self._previous_right_contact = right_hit
            self._previous_action = np.clip(executed_action, -1.0, 1.0)
            self._step_count += 1
            terminated = bool(
                dropped
                or task_success
            )
            truncated = bool(not terminated and self._step_count >= self.max_steps)
            if truncated:
                reward -= config.timeout_penalty
            info = {
                "distance_m": distance,
                "distance_progress_m": distance_progress,
                "bilateral_contact": bilateral,
                "left_contact": left_hit,
                "right_contact": right_hit,
                "connector_created": connector_created,
                "lift_m": lift,
                "grasp_success": grasp_success,
                "dropped": dropped,
                "collision": collision or self._table_collision(),
                "ik_failed": ik_failed,
                "joint_limit": joint_limit,
                "step": self._step_count,
                "curriculum_stage": self.curriculum_stage,
                "target_handle": self.episode_target_handle,
                "stage_distance_success": stage_distance_success,
                "stage_contact_success": stage_contact_success,
                "task_success": task_success,
                "inactive_action_norm": float(np.linalg.norm(inactive_action)),
            }
            return self._capture_observation(), float(reward), terminated, truncated, info

        def close(self) -> None:
            self._cleanup()
            self.sim = None
            self.client = None
