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
            self._depth_noise_std_m = 0.0
            self._robot_shape_flags: list[tuple[int, int, int]] = []

            self.action_space = spaces.Box(-1.0, 1.0, shape=(ACTION_DIM,), dtype=np.float32)
            self.observation_space = spaces.Dict(
                {
                    "depth": spaces.Box(0.0, 1.0, shape=(1, self.image_size, self.image_size), dtype=np.float32),
                    # q(7), dq(7), gripper(1), previous action(7)
                    "proprio": spaces.Box(-1.0, 1.0, shape=(22,), dtype=np.float32),
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
                curriculum_count = {"A": 1, "B": 2}.get(self.curriculum_stage)
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
            if value not in {"A", "B", "C", "D", "E"}:
                raise ValueError("Curriculum stage must be A, B, C, D, or E")
            self.curriculum_stage = value

        def _curriculum_scene(self, seed: int | None) -> tuple[str, str | None]:
            rng = np.random.default_rng(seed)
            stage = self.curriculum_stage
            self._depth_noise_std_m = 0.0
            if stage == "A":
                return "separated", None
            if stage == "B":
                return "separated", None
            if stage == "C":
                return "planned", str(rng.choice(["table_only", "stack"]))
            if stage == "D":
                return "planned", str(rng.choice(["leaning", "bridge", "partial_support"]))
            self._depth_noise_std_m = 0.002
            return self.scene_mode, self.planned_layout

        def _apply_curriculum_object_count(self, records: list[dict[str, Any]], seed: int | None) -> list[dict[str, Any]]:
            if self.curriculum_stage != "A" or len(records) <= 1:
                return records
            rng = np.random.default_rng(seed)
            simple = [item for item in records if str(item.get("class", "")) in {"cube", "cylinder"}]
            keep = simple[int(rng.integers(len(simple)))] if simple else records[0]
            remove = [int(item["handle"]) for item in records if item is not keep and "handle" in item]
            if remove:
                self.sim.removeObjects(remove)
            return [keep]

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
            proprio = np.concatenate((q_state, dq_state, [self._gripper_state], self._previous_action)).astype(np.float32)
            return {"depth": depth_state.astype(np.float32), "proprio": proprio}

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
            for handle in list(self.object_handles):
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
            for handle in self.object_handles:
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
            self._gripper_state = 0.0
            self._step_count = 0
            self._lift_hold_count = 0
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
            _handle, distance = self._nearest_object()
            self._previous_distance = distance
            return self._capture_observation(), {
                "object_count": len(self.object_handles),
                "seed": seed,
                "execution_mode": self.execution_mode,
                "arm_control": "kinematic_ik",
                "rollout_dynamics": self.execution_mode == "dynamic",
                "grasp_proxy": "connector" if self.connector_enabled else "none",
            }

        def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
            if self.ik is None:
                raise RuntimeError("reset() must be called before step()")
            raw_action = np.asarray(action, dtype=np.float32).reshape(ACTION_DIM)
            translation, rotvec, gripper = decode_cartesian_action(raw_action)
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
                self._gripper_state > 0.0
                and bilateral
                and candidate is not None
                and self._between_fingers(candidate)
                and tcp_to_object <= 0.075
                and not self._table_collision()
            ):
                if self.attached_handle is None:
                    self._attach(candidate)
                    connector_created = self.attached_handle is not None

            _nearest, distance = self._nearest_object()
            distance_progress = 0.0 if not math.isfinite(self._previous_distance) else self._previous_distance - distance
            self._previous_distance = distance
            lift = self._lift_progress()
            if lift >= 0.015:
                self._lift_hold_count += 1
            else:
                self._lift_hold_count = 0
            grasp_success = self.attached_handle is not None and self._lift_hold_count >= 3
            dropped = released or (self.attached_handle is not None and self._gripper_state < 0.0)
            joint_limit = self._joint_limit_violation()
            smoothness = float(np.mean(np.square(raw_action - self._previous_action)))
            reward = 40.0 * distance_progress - 0.02 * smoothness
            reward -= 5.0 if collision or self._table_collision() else 0.0
            reward -= 5.0 if ik_failed else 0.0
            reward -= 10.0 if joint_limit else 0.0
            reward += 10.0 if bilateral and self._gripper_state > 0.0 else 0.0
            reward += 20.0 if connector_created else 0.0
            reward += 30.0 * min(lift / 0.015, 1.0)
            reward += 50.0 if self._lift_hold_count >= 3 else 0.0
            reward += 100.0 if grasp_success else 0.0
            reward -= 50.0 if dropped else 0.0
            self._previous_action = np.clip(raw_action, -1.0, 1.0)
            self._step_count += 1
            terminated = bool(grasp_success or dropped)
            truncated = bool(not terminated and self._step_count >= self.max_steps)
            if truncated:
                reward -= 10.0
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
            }
            return self._capture_observation(), float(reward), terminated, truncated, info

        def close(self) -> None:
            self._cleanup()
            self.sim = None
            self.client = None
