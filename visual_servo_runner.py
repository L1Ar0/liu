"""Run geometry-only eye-in-hand PBVS and optional RG2 grasp execution."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from scipy.spatial.transform import Rotation

from gripper_tcp_approach import get_ik_handle, get_joint_limits
from point_cloud import find_unique_object_by_alias, get_kuka_joints_from_tip
from visual_servo_controller import (
    PBVSController,
    build_surface_aligned_grasp_pose,
    build_top_down_grasp_pose,
    matrix_from_pose,
    pose_from_matrix,
)
from visual_servo_perception import (
    TargetObservation,
    TargetTrackState,
    capture_target_observation,
    select_servo_target,
    track_state_from_prediction,
    update_track_state,
)


ROOT = Path(__file__).resolve().parent
RECOGNITION_FILE = ROOT / "recognition_output" / "recognition_results.json"
SEGMENTATION_FILE = ROOT / "segmentation_output" / "segmentation_metadata.json"
OUTPUT_DIR = ROOT / "visual_servo_output"
SUMMARY_FILE = OUTPUT_DIR / "visual_servo_summary.json"
TRACE_FILE = OUTPUT_DIR / "visual_servo_trace.csv"

SERVO_RATE_HZ = float(os.environ.get("ROBOT_GRASP_SERVO_RATE_HZ", "10"))
SERVO_PREGRASP_STANDOFF_M = float(
    os.environ.get("ROBOT_GRASP_SERVO_PREGRASP_STANDOFF_M", "0.090")
)
SERVO_FINAL_STANDOFF_M = float(
    os.environ.get("ROBOT_GRASP_SERVO_FINAL_STANDOFF_M", "0.012")
)
SERVO_MAX_ITERATIONS = int(
    os.environ.get("ROBOT_GRASP_SERVO_MAX_ITERATIONS", "70")
)
SERVO_LOST_LIMIT = int(os.environ.get("ROBOT_GRASP_SERVO_LOST_LIMIT", "5"))
SERVO_STABLE_FRAMES = int(os.environ.get("ROBOT_GRASP_SERVO_STABLE_FRAMES", "3"))
SERVO_PREGRASP_STEPS = int(os.environ.get("ROBOT_GRASP_SERVO_PREGRASP_STEPS", "45"))
SERVO_POSITION_TOLERANCE_M = float(
    os.environ.get("ROBOT_GRASP_SERVO_POSITION_TOLERANCE_M", "0.003")
)
SERVO_FINAL_POSITION_TOLERANCE_M = float(
    os.environ.get("ROBOT_GRASP_SERVO_FINAL_POSITION_TOLERANCE_M", "0.0025")
)
SERVO_ORIENTATION_TOLERANCE_RAD = math.radians(
    float(os.environ.get("ROBOT_GRASP_SERVO_ORIENTATION_TOLERANCE_DEG", "5"))
)
SERVO_FINAL_ORIENTATION_TOLERANCE_RAD = math.radians(
    float(os.environ.get("ROBOT_GRASP_SERVO_FINAL_ORIENTATION_TOLERANCE_DEG", "3"))
)
SERVO_MIN_TCP_TABLE_CLEARANCE_M = float(
    os.environ.get("ROBOT_GRASP_SERVO_MIN_TCP_TABLE_CLEARANCE_M", "0.025")
)
SERVO_LIFT_M = float(os.environ.get("ROBOT_GRASP_SERVO_LIFT_M", "0.150"))
SERVO_GRASP_MIN_CONFIDENCE = float(
    os.environ.get("ROBOT_GRASP_SERVO_GRASP_MIN_CONFIDENCE", "0.55")
)
SERVO_GRASP_MIN_POINTS = int(
    os.environ.get("ROBOT_GRASP_SERVO_GRASP_MIN_POINTS", "80")
)
SERVO_GRASP_MIN_IMAGE_MARGIN_PX = float(
    os.environ.get("ROBOT_GRASP_SERVO_GRASP_MIN_IMAGE_MARGIN_PX", "20")
)
GRIPPER_SIGNAL_NAME = os.environ.get("ROBOT_GRASP_RG2_SIGNAL", "RG2_open")
GRIPPER_OPEN_VALUE = int(os.environ.get("ROBOT_GRASP_RG2_OPEN_VALUE", "1"))
GRIPPER_CLOSE_VALUE = int(os.environ.get("ROBOT_GRASP_RG2_CLOSE_VALUE", "0"))
GRIPPER_STEPS = int(os.environ.get("ROBOT_GRASP_RG2_STEPS", "70"))
GRIPPER_JOINT_MOVE_EPS = float(
    os.environ.get("ROBOT_GRASP_RG2_JOINT_MOVE_EPS", "0.0001")
)
GRASP_ORIENTATION_MODE = os.environ.get(
    "ROBOT_GRASP_SERVO_GRASP_ORIENTATION", "auto"
).lower()
GRASP_MAX_TILT_DEG = float(
    os.environ.get("ROBOT_GRASP_SERVO_MAX_TILT_DEG", "8.0")
)
USE_CONNECTOR = os.environ.get("ROBOT_GRASP_USE_CONNECTOR", "1").lower() not in {
    "0",
    "false",
    "no",
}
# The stock RG2 child script advances only while dynamics is running.  In this
# scene that brief dynamics window can make the iiwa links lose their held
# pose.  Connector mode therefore uses a paused, deterministic close command
# by default; set this to 0 only when validating the physical RG2 dynamics.
CONNECTOR_KINEMATIC_ONLY = os.environ.get(
    "ROBOT_GRASP_CONNECTOR_KINEMATIC_ONLY", "1"
).lower() not in {"0", "false", "no"}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_outputs(summary: dict[str, Any], trace: list[dict[str, Any]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with TRACE_FILE.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "iteration",
                "phase",
                "position_error_mm",
                "rotation_error_deg",
                "target_confidence",
                "point_count",
                "image_margin_px",
            ],
        )
        writer.writeheader()
        writer.writerows(trace)


def _matrix34_to_44(matrix: Any) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape == (4, 4):
        return value.copy()
    if value.size != 12:
        raise ValueError(f"Expected a 3x4 object matrix, received {value.shape}")
    result = np.eye(4, dtype=np.float64)
    result[:3, :4] = value.reshape(3, 4)
    return result


def _mode_value(mode: Any) -> int:
    if isinstance(mode, (list, tuple)):
        return int(mode[0])
    return int(mode)


def _table_signed_height(position: np.ndarray, table_plane: np.ndarray) -> float:
    plane = np.asarray(table_plane, dtype=np.float64)
    denominator = max(float(np.linalg.norm(plane[:3])), 1e-12)
    return float((np.dot(position, plane[:3]) + plane[3]) / denominator)


def _target_grasp_pose(
    state: TargetTrackState,
    standoff_m: float,
    current_tip_rotation: np.ndarray,
    table_normal: np.ndarray,
    orientation_mode: str,
) -> np.ndarray:
    """Select a geometry-only top-down or surface-aligned target pose."""

    mode = str(orientation_mode).lower()
    use_surface = mode == "surface"
    if mode == "auto":
        rotation = np.asarray(state.rotation_base, dtype=np.float64)
        table = np.asarray(table_normal, dtype=np.float64)
        table /= max(float(np.linalg.norm(table)), 1e-12)
        tilt = math.degrees(
            math.acos(float(np.clip(abs(np.dot(rotation[:, 2], table)), -1.0, 1.0)))
        )
        use_surface = tilt >= 8.0
    if use_surface:
        return build_surface_aligned_grasp_pose(
            state.center_base_m,
            state.rotation_base,
            state.dimensions_m,
            state.class_name,
            standoff_m,
            current_tip_rotation,
            table_normal,
            max_tilt_deg=GRASP_MAX_TILT_DEG,
        )
    return build_top_down_grasp_pose(
        state.center_base_m,
        state.rotation_base,
        state.dimensions_m,
        state.class_name,
        standoff_m,
        current_tip_rotation,
        table_normal,
    )


def _apply_initial_disturbance(
    pose: np.ndarray,
    translation_m: np.ndarray | None,
    euler_deg: np.ndarray | None,
) -> np.ndarray:
    result = np.asarray(pose, dtype=np.float64).copy()
    if translation_m is not None:
        result[:3, 3] += np.asarray(translation_m, dtype=np.float64).reshape(3)
    if euler_deg is not None:
        disturbance = Rotation.from_euler(
            "xyz", np.asarray(euler_deg, dtype=np.float64).reshape(3), degrees=True
        ).as_matrix()
        result[:3, :3] = disturbance @ result[:3, :3]
    return result


class IncrementalIK:
    """Keep a damped simIK group alive while PBVS updates a target dummy."""

    def __init__(self, sim: Any, sim_ik: Any, robot_base: int, tip: int) -> None:
        state = sim.getSimulationState()
        if state not in {sim.simulation_stopped, sim.simulation_paused}:
            raise RuntimeError(
                "Visual servo requires a stopped or paused CoppeliaSim simulation"
            )
        self.sim = sim
        self.sim_ik = sim_ik
        self.robot_base = int(robot_base)
        self.tip = int(tip)
        self.joints = get_kuka_joints_from_tip(sim, tip)
        if len(self.joints) != 7:
            raise RuntimeError(f"Expected seven iiwa joints, found {len(self.joints)}")
        self.limits = get_joint_limits(sim, self.joints)
        self.original_modes = [_mode_value(sim.getJointMode(joint)) for joint in self.joints]
        self.target = int(sim.createDummy(0.012))
        self.environment = None
        self.group = None
        self.element = None
        self.full_pose_constraint = False
        self._setup()

    def _setup(self) -> None:
        current = _matrix34_to_44(self.sim.getObjectMatrix(self.tip, self.robot_base))
        self.sim.setObjectPose(self.target, pose_from_matrix(current), self.robot_base)
        self.environment = int(self.sim_ik.createEnvironment())
        self.group = int(self.sim_ik.createGroup(self.environment))
        full_pose = getattr(self.sim_ik, "constraint_pose", None)
        if full_pose is None:
            constraints = self.sim_ik.constraint_position | self.sim_ik.constraint_alpha_beta
        else:
            constraints = full_pose
            self.full_pose_constraint = True
        self.element, mapping, _ = self.sim_ik.addElementFromScene(
            self.environment,
            self.group,
            self.robot_base,
            self.tip,
            self.target,
            constraints,
        )
        ik_joints = [get_ik_handle(mapping, joint) for joint in self.joints]
        self.sim_ik.setGroupCalculation(
            self.environment,
            self.group,
            self.sim_ik.method_damped_least_squares,
            0.08,
            120,
        )
        try:
            flags = int(self.sim_ik.getGroupFlags(self.environment, self.group))
            flags |= (
                self.sim_ik.group_avoidlimits
                | self.sim_ik.group_restoreonbadlintol
                | self.sim_ik.group_restoreonbadangtol
            )
            self.sim_ik.setGroupFlags(self.environment, self.group, flags)
        except Exception:
            pass
        self.sim_ik.setElementPrecision(
            self.environment,
            self.group,
            self.element,
            [0.0015, math.radians(2.0)],
        )
        for ik_joint, (lower, upper) in zip(ik_joints, self.limits):
            self.sim_ik.setJointInterval(
                self.environment,
                ik_joint,
                False,
                [lower, upper - lower],
            )
            self.sim_ik.setJointLimitMargin(
                self.environment,
                ik_joint,
                math.radians(8.0),
            )
            self.sim_ik.setJointMaxStepSize(
                self.environment,
                ik_joint,
                math.radians(5.0),
            )
        self.sim_ik.syncFromSim(self.environment, [self.group])

    def current_pose(self) -> np.ndarray:
        return _matrix34_to_44(self.sim.getObjectMatrix(self.tip, self.robot_base))

    def apply(self, target_pose: np.ndarray) -> np.ndarray:
        self.sim.setObjectPose(
            self.target,
            pose_from_matrix(target_pose),
            self.robot_base,
        )
        result, flags, precision = self.sim_ik.handleGroup(
            self.environment,
            self.group,
            {"syncWorlds": True},
        )
        if result != self.sim_ik.result_success:
            position_precision = float(precision[0]) if len(precision) else float("inf")
            orientation_precision = float(precision[1]) if len(precision) > 1 else float("inf")
            if position_precision > 0.006 or orientation_precision > math.radians(7.0):
                raise RuntimeError(
                    "Visual-servo IK failed: "
                    f"flags={flags}, position={position_precision * 1000:.2f} mm, "
                    f"orientation={math.degrees(orientation_precision):.2f} deg"
                )
        return self.current_pose()

    def move_linear(self, target_pose: np.ndarray, steps: int, table_plane: np.ndarray) -> np.ndarray:
        start = self.current_pose()
        final = np.asarray(target_pose, dtype=np.float64)
        last = start
        for index in range(1, max(1, int(steps)) + 1):
            ratio = index / max(1, int(steps))
            position = (1.0 - ratio) * start[:3, 3] + ratio * final[:3, 3]
            if _table_signed_height(position, table_plane) < SERVO_MIN_TCP_TABLE_CLEARANCE_M:
                raise RuntimeError("Visual-servo path violated table clearance")
            rotation = final[:3, :3]
            pose = np.eye(4, dtype=np.float64)
            pose[:3, 3] = position
            pose[:3, :3] = rotation
            last = self.apply(pose)
            time.sleep(1.0 / max(SERVO_RATE_HZ, 1.0) / max(steps, 1))
        return last

    def close(self) -> None:
        if self.environment is not None:
            try:
                self.sim_ik.eraseEnvironment(self.environment)
            except Exception:
                pass
            self.environment = None
        try:
            self.sim.removeObject(self.target)
        except Exception:
            pass
        for joint, mode in zip(self.joints, self.original_modes):
            try:
                self.sim.setJointMode(joint, mode)
            except Exception:
                pass


def _set_gripper_signal(sim: Any, value: int) -> None:
    try:
        sim.clearFloatSignal(GRIPPER_SIGNAL_NAME)
    except Exception:
        pass
    sim.setInt32Signal(GRIPPER_SIGNAL_NAME, int(value))


def _find_gripper_drive_joint(sim: Any) -> int | None:
    joints = sim.getObjectsInTree(sim.handle_scene, sim.sceneobject_joint, 0)
    for handle in joints:
        try:
            alias = "".join(
                character
                for character in str(sim.getObjectAlias(handle)).lower()
                if character.isalnum()
            )
        except Exception:
            continue
        if "openclosejoint" in alias or "openclose" in alias:
            return int(handle)
    return None


def _run_gripper_motion(
    client: RemoteAPIClient,
    sim: Any,
    value: int,
    steps: int = GRIPPER_STEPS,
    hold_ik: IncrementalIK | None = None,
    hold_pose: np.ndarray | None = None,
) -> tuple[float | None, float | None]:
    drive_joint = _find_gripper_drive_joint(sim)
    before = None
    if drive_joint is not None:
        try:
            before = float(sim.getJointPosition(drive_joint))
        except Exception:
            pass
    held_joint_positions: list[float] | None = None
    if hold_ik is not None:
        held_joint_positions = [
            float(sim.getJointPosition(joint)) for joint in hold_ik.joints
        ]

    def hold_robot_joints() -> None:
        if held_joint_positions is None or hold_ik is None:
            return
        for joint, position in zip(hold_ik.joints, held_joint_positions):
            try:
                sim.setJointPosition(joint, position)
            except Exception:
                pass

    _set_gripper_signal(sim, value)
    state = sim.getSimulationState()
    if state == sim.simulation_stopped:
        client.setStepping(True)
        sim.startSimulation()
        for _ in range(max(1, int(steps))):
            hold_robot_joints()
            client.step()
        try:
            sim.pauseSimulation()
        except Exception:
            pass
    elif state == sim.simulation_paused:
        client.setStepping(True)
        sim.startSimulation()
        for _ in range(max(1, int(steps))):
            hold_robot_joints()
            client.step()
        sim.pauseSimulation()
    after = None
    if drive_joint is not None:
        try:
            after = float(sim.getJointPosition(drive_joint))
        except Exception:
            pass
    if before is not None and after is not None:
        print(
            "RG2 openCloseJoint: "
            f"{before:.6f} -> {after:.6f} "
            f"(motion={abs(after - before):.6f})"
        )
    else:
        print("RG2 openCloseJoint position: unavailable")
    return before, after


def _prepare_physics_grasp(sim: Any, robot_base: int) -> int:
    """Restore dynamic/respondable state for generated physics objects.

    The scene remains paused during visual servoing, but a body must stay
    dynamic so the fingers can transfer force to it during the lift. This
    helper also restores the dataset collision bit on the robot hierarchy,
    which is useful when a scene was interrupted during generation.
    """

    if os.environ.get("ROBOT_GRASP_SCENE_MODE", "").lower() not in {
        "physics",
        "dynamic",
        "settled",
        "drop",
    }:
        return 0
    static_param = getattr(sim, "shapeintparam_static", None)
    respondable_param = getattr(sim, "shapeintparam_respondable", None)
    mask_param = getattr(sim, "shapeintparam_respondable_mask", None)
    changed = 0
    shapes = sim.getObjectsInTree(sim.handle_scene, sim.sceneobject_shape, 0)
    for handle in shapes:
        try:
            alias = str(sim.getObjectAlias(handle)).lower()
        except Exception:
            continue
        if not alias.startswith("rand_"):
            continue
        if static_param is not None:
            try:
                sim.setObjectInt32Param(handle, static_param, 0)
            except Exception:
                pass
        if respondable_param is not None:
            try:
                sim.setObjectInt32Param(handle, respondable_param, 1)
            except Exception:
                pass
        if mask_param is not None:
            try:
                mask = int(sim.getObjectInt32Param(handle, mask_param))
            except Exception:
                mask = 0xFFFF
            try:
                sim.setObjectInt32Param(handle, mask_param, mask | 0x0100)
            except Exception:
                pass
        changed += 1

    if mask_param is not None:
        for handle in sim.getObjectsInTree(
            int(robot_base),
            sim.sceneobject_shape,
            0,
        ):
            try:
                mask = int(sim.getObjectInt32Param(handle, mask_param))
                sim.setObjectInt32Param(handle, mask_param, mask | 0x0100)
            except Exception:
                pass
    return changed


def _resolve_target_shape(
    sim: Any,
    robot_base: int,
    target_state: TargetTrackState,
) -> tuple[int, str]:
    """Resolve the scene shape nearest to the tracked geometry estimate."""

    shapes = sim.getObjectsInTree(sim.handle_scene, sim.sceneobject_shape, 0)
    candidates: list[tuple[float, int, str]] = []
    for handle in shapes:
        try:
            alias = str(sim.getObjectAlias(handle))
            if not alias.startswith("rand_"):
                continue
            if target_state.class_name not in alias.lower():
                continue
            position = np.asarray(sim.getObjectPosition(handle, robot_base), dtype=np.float64)
            distance = float(np.linalg.norm(position - target_state.center_base_m))
            candidates.append((distance, int(handle), alias))
        except Exception:
            continue
    if not candidates:
        raise RuntimeError(
            f"Unable to resolve connector target for {target_state.class_name} "
            f"near {target_state.center_base_m.tolist()}"
        )
    candidates.sort(key=lambda value: value[0])
    distance, handle, alias = candidates[0]
    if distance > 0.080:
        raise RuntimeError(f"Nearest connector target is too far away: {distance:.3f} m")
    return handle, alias


def _attach_connector(
    sim: Any,
    robot_base: int,
    tip: int,
    target_state: TargetTrackState,
) -> dict[str, Any]:
    """Attach the selected shape to a gripper-side connector after closure.

    CoppeliaSim's dynamic contact model remains enabled during the approach;
    after the verified close command the connector provides deterministic lift
    verification, matching the permitted simulation convention while keeping
    the attachment explicit in the output record.
    """

    target_handle, target_alias = _resolve_target_shape(sim, robot_base, target_state)
    static_param = getattr(sim, "shapeintparam_static", None)
    respondable_param = getattr(sim, "shapeintparam_respondable", None)
    if static_param is not None:
        sim.setObjectInt32Param(target_handle, static_param, 1)
    if respondable_param is not None:
        sim.setObjectInt32Param(target_handle, respondable_param, 0)

    target_center_before = np.asarray(
        sim.getObjectPosition(target_handle, -1), dtype=np.float64
    )
    tip_pose = list(sim.getObjectPose(tip, -1))
    connector_handle: int | None = None
    connector_alias = f"grasp_connector_{target_alias}"
    try:
        connector_handle = int(sim.createDummy(0.002))
        try:
            sim.setObjectAlias(connector_handle, connector_alias)
        except Exception:
            pass
        try:
            sim.setObjectPose(connector_handle, tip_pose, -1)
            sim.setObjectParent(connector_handle, tip, True)
            sim.setObjectPosition(target_handle, -1, tip_pose[:3])
            sim.setObjectParent(target_handle, connector_handle, True)
        except Exception:
            sim.setObjectPosition(target_handle, -1, tip_pose[:3])
            sim.setObjectParent(target_handle, tip, True)
            connector_handle = None
    except Exception:
        sim.setObjectPosition(target_handle, -1, tip_pose[:3])
        sim.setObjectParent(target_handle, tip, True)

    target_center_after = np.asarray(
        sim.getObjectPosition(target_handle, -1), dtype=np.float64
    )

    return {
        "attached": True,
        "mode": "coppeliasim_connector",
        "target_handle": int(target_handle),
        "target_alias": target_alias,
        "connector_handle": connector_handle,
        "connector_alias": connector_alias if connector_handle is not None else None,
        "center_snapped_to_tip": True,
        "target_center_before_m": target_center_before.tolist(),
        "target_center_after_m": target_center_after.tolist(),
        "tip_center_m": [float(value) for value in tip_pose[:3]],
        "snap_distance_mm": float(
            np.linalg.norm(target_center_after - target_center_before) * 1000.0
        ),
    }


def _validate_grasp_observation(
    state: TargetTrackState,
    observation: TargetObservation | None,
) -> None:
    """Reject an obviously weak target before closing the physical gripper."""

    if state.confidence < SERVO_GRASP_MIN_CONFIDENCE:
        raise RuntimeError(
            "Refusing physical grasp: target confidence is "
            f"{state.confidence:.2f} < {SERVO_GRASP_MIN_CONFIDENCE:.2f}"
        )
    if observation is None:
        raise RuntimeError("Refusing physical grasp: no final RGB-D target observation")
    if observation.point_count < SERVO_GRASP_MIN_POINTS:
        raise RuntimeError(
            "Refusing physical grasp: final target point count is "
            f"{observation.point_count} < {SERVO_GRASP_MIN_POINTS}"
        )
    if observation.image_margin_px < SERVO_GRASP_MIN_IMAGE_MARGIN_PX:
        raise RuntimeError(
            "Refusing physical grasp: target is too close to the image boundary "
            f"({observation.image_margin_px:.1f} px)"
        )


def _lift_and_verify(
    client: RemoteAPIClient,
    sim: Any,
    ik: IncrementalIK,
    target_state: Any,
    table_plane: np.ndarray,
    lift_m: float,
    kinematic_only: bool = False,
) -> bool:
    before = target_state.center_base_m.copy()
    normal = np.asarray(table_plane[:3], dtype=np.float64) / max(
        float(np.linalg.norm(table_plane[:3])), 1e-12
    )
    start = ik.current_pose()
    lifted = start.copy()
    lifted[:3, 3] += normal * float(lift_m)
    if kinematic_only:
        for index in range(1, 31):
            ratio = index / 30.0
            pose = start.copy()
            pose[:3, 3] = (1.0 - ratio) * start[:3, 3] + ratio * lifted[:3, 3]
            ik.apply(pose)
            time.sleep(1.0 / max(SERVO_RATE_HZ, 1.0) / 30.0)
    else:
        client.setStepping(True)
        if sim.getSimulationState() in {sim.simulation_stopped, sim.simulation_paused}:
            sim.startSimulation()
        for index in range(1, 31):
            ratio = index / 30.0
            pose = start.copy()
            pose[:3, 3] = (1.0 - ratio) * start[:3, 3] + ratio * lifted[:3, 3]
            ik.apply(pose)
            client.step()
        sim.pauseSimulation()
    expected = before + normal * float(lift_m)
    lifted_state = TargetTrackState(
        object_id=target_state.object_id,
        class_name=target_state.class_name,
        center_base_m=expected,
        rotation_base=target_state.rotation_base.copy(),
        dimensions_m=target_state.dimensions_m.copy(),
        confidence=target_state.confidence,
    )
    observation = capture_target_observation(
        sim,
        find_unique_object_by_alias(sim, sim.sceneobject_visionsensor, "rgbd_camera"),
        ik.robot_base,
        lifted_state,
        table_plane,
    )
    if observation is None:
        return False
    height_gain = float(np.dot(observation.center_base_m - before, normal))
    expected_error = float(np.linalg.norm(observation.center_base_m - expected))
    return bool(
        height_gain >= 0.5 * float(lift_m)
        and expected_error <= max(0.030, 0.45 * float(lift_m))
        and observation.confidence >= 0.30
    )


def run_visual_servo(
    target_id: int | None = None,
    execute_grasp: bool = False,
    max_iterations: int = SERVO_MAX_ITERATIONS,
    grasp_orientation: str = GRASP_ORIENTATION_MODE,
    initial_translation_m: np.ndarray | None = None,
    initial_euler_deg: np.ndarray | None = None,
    open_loop_only: bool = False,
    align_only: bool = False,
) -> dict[str, Any]:
    recognition = _load_json(RECOGNITION_FILE)
    segmentation = _load_json(SEGMENTATION_FILE)
    target_prediction = select_servo_target(recognition, target_id)
    state = track_state_from_prediction(target_prediction)
    table_plane = np.asarray(
        segmentation.get("table_plane", [0.0, 0.0, 1.0, 0.0]),
        dtype=np.float64,
    )

    client = RemoteAPIClient()
    sim = client.require("sim")
    sim_ik = client.require("simIK")
    camera = find_unique_object_by_alias(sim, sim.sceneobject_visionsensor, "rgbd_camera")
    tip = find_unique_object_by_alias(sim, sim.sceneobject_dummy, "gripper_tip")
    joints = get_kuka_joints_from_tip(sim, tip)
    if len(joints) != 7:
        raise RuntimeError(f"Expected seven arm joints, found {len(joints)}")
    robot_base = int(sim.getObjectParent(joints[0]))
    if robot_base < 0:
        raise RuntimeError("Cannot determine Robot Base from iiwa joint chain")

    print(
        "\n========== Visual Servo Target ==========\n"
        f"Object {state.object_id}: {state.class_name}, "
        f"center={np.round(state.center_base_m, 5)}, "
        f"confidence={state.confidence:.2f}"
    )

    ik = IncrementalIK(sim, sim_ik, robot_base, tip)
    trace: list[dict[str, Any]] = []
    converged = False
    lost_count = 0
    total_lost_frames = 0
    stable_count = 0
    total_iterations = 0
    final_error = None
    pregrasp_pose = None
    last_observation: TargetObservation | None = None
    gripper_close_motion = None
    connector_attachment: dict[str, Any] | None = None
    try:
        current = ik.current_pose()
        pregrasp_pose = _target_grasp_pose(
            state,
            SERVO_PREGRASP_STANDOFF_M,
            current[:3, :3],
            table_plane[:3],
            grasp_orientation,
        )
        initial_pose = _apply_initial_disturbance(
            pregrasp_pose,
            initial_translation_m,
            initial_euler_deg,
        )
        print("Moving to visual-servo pregrasp pose...")
        current = ik.move_linear(initial_pose, SERVO_PREGRASP_STEPS, table_plane)
        if execute_grasp and not (USE_CONNECTOR and CONNECTOR_KINEMATIC_ONLY):
            # Active second-view motion may leave the TCP only a few millimetres
            # above the clearance gate. First retreat to the high pregrasp,
            # then open RG2 so finger motion cannot invalidate the safe path.
            print("Opening RG2 gripper at the safe pregrasp pose...")
            _run_gripper_motion(
                client,
                sim,
                GRIPPER_OPEN_VALUE,
                hold_ik=ik,
                hold_pose=current.copy(),
            )
            current = ik.current_pose()
        elif execute_grasp:
            print(
                "Connector mode: keeping CoppeliaSim paused; "
                "RG2 opening is represented by the post-closure connector"
            )

        controller = PBVSController()
        dt = 1.0 / max(SERVO_RATE_HZ, 1.0)
        if open_loop_only:
            observation = capture_target_observation(
                sim,
                camera,
                robot_base,
                state,
                table_plane,
            )
            if observation is None:
                raise RuntimeError("Open-loop residual measurement lost the target")
            last_observation = observation
            state = update_track_state(state, observation)
            desired = _target_grasp_pose(
                state,
                SERVO_PREGRASP_STANDOFF_M,
                current[:3, :3],
                table_plane[:3],
                grasp_orientation,
            )
            error = controller.pose_error(current, desired)
            final_error = error
            trace.append(
                {
                    "iteration": 1,
                    "phase": "open_loop_measurement",
                    "position_error_mm": error.position_norm_m * 1000.0,
                    "rotation_error_deg": math.degrees(error.rotation_norm_rad),
                    "target_confidence": observation.confidence,
                    "point_count": observation.point_count,
                    "image_margin_px": observation.image_margin_px,
                }
            )
            summary = {
                "target_id": state.object_id,
                "target_class": state.class_name,
                "control_mode": "open_loop",
                "initial_translation_mm": None
                if initial_translation_m is None
                else (np.asarray(initial_translation_m) * 1000.0).tolist(),
                "initial_euler_deg": None
                if initial_euler_deg is None
                else np.asarray(initial_euler_deg).tolist(),
                "iterations": 1,
                "target_lost_frames": 0,
                "converged": controller.converged(
                    error,
                    SERVO_POSITION_TOLERANCE_M,
                    SERVO_FINAL_ORIENTATION_TOLERANCE_RAD,
                ),
                "execute_grasp": False,
                "grasp_verified": None,
                "final_position_error_mm": error.position_norm_m * 1000.0,
                "final_rotation_error_deg": math.degrees(error.rotation_norm_rad),
                "uses_color_features": False,
                "controller": "open_loop_pose_command",
                "grasp_orientation": str(grasp_orientation),
            }
            _write_outputs(summary, trace)
            return summary
        phases = [
            (
                "align",
                SERVO_PREGRASP_STANDOFF_M,
                SERVO_POSITION_TOLERANCE_M,
                SERVO_ORIENTATION_TOLERANCE_RAD,
            ),
            (
                "approach",
                SERVO_FINAL_STANDOFF_M,
                SERVO_FINAL_POSITION_TOLERANCE_M,
                SERVO_FINAL_ORIENTATION_TOLERANCE_RAD,
            ),
        ]
        if align_only:
            phases = phases[:1]
        for phase_name, standoff, position_tol, orientation_tol in phases:
            print(f"\n--- Visual servo phase: {phase_name} ---")
            stable_count = 0
            for phase_iteration in range(max(1, int(max_iterations))):
                total_iterations += 1
                observation = capture_target_observation(
                    sim,
                    camera,
                    robot_base,
                    state,
                    table_plane,
                )
                if observation is None:
                    lost_count += 1
                    total_lost_frames += 1
                    print(
                        f"servo {phase_name} {phase_iteration + 1}: "
                        f"target lost ({lost_count}/{SERVO_LOST_LIMIT}); holding position"
                    )
                    if lost_count >= SERVO_LOST_LIMIT:
                        raise RuntimeError(
                            "Visual servo stopped because the target left the camera view "
                            "or local depth tracking failed"
                        )
                    if lost_count >= 2:
                        current = ik.current_pose()
                        recovery = _target_grasp_pose(
                            state,
                            SERVO_PREGRASP_STANDOFF_M,
                            current[:3, :3],
                            table_plane[:3],
                            grasp_orientation,
                        )
                        recovery_command = controller.command(current, recovery)
                        recovery_pose = controller.next_pose(current, recovery_command, dt)
                        if (
                            _table_signed_height(recovery_pose[:3, 3], table_plane)
                            >= SERVO_MIN_TCP_TABLE_CLEARANCE_M
                        ):
                            ik.apply(recovery_pose)
                            print("Target recovery: retreating toward the safe pregrasp view")
                    time.sleep(dt)
                    continue

                lost_count = 0
                last_observation = observation
                state = update_track_state(state, observation)
                current = ik.current_pose()
                desired = _target_grasp_pose(
                    state,
                    standoff,
                    current[:3, :3],
                    table_plane[:3],
                    grasp_orientation,
                )
                command = controller.command(current, desired)
                error = command.error
                final_error = error
                trace.append(
                    {
                        "iteration": total_iterations,
                        "phase": phase_name,
                        "position_error_mm": error.position_norm_m * 1000.0,
                        "rotation_error_deg": math.degrees(error.rotation_norm_rad),
                        "target_confidence": observation.confidence,
                        "point_count": observation.point_count,
                        "image_margin_px": observation.image_margin_px,
                    }
                )
                print(
                    f"servo {phase_name} {phase_iteration + 1}: "
                    f"pos={error.position_norm_m * 1000:.2f} mm, "
                    f"rot={math.degrees(error.rotation_norm_rad):.2f} deg, "
                    f"conf={observation.confidence:.2f}, "
                    f"points={observation.point_count}"
                )

                if controller.converged(error, position_tol, orientation_tol):
                    stable_count += 1
                    if stable_count >= SERVO_STABLE_FRAMES:
                        break
                    # Do not chase sub-millimetre observation changes while
                    # confirming convergence over consecutive RGB-D frames.
                    time.sleep(dt)
                    continue
                else:
                    stable_count = 0
                next_pose = controller.next_pose(current, command, dt)
                if _table_signed_height(next_pose[:3, 3], table_plane) < SERVO_MIN_TCP_TABLE_CLEARANCE_M:
                    raise RuntimeError("PBVS command would violate table clearance")
                current = ik.apply(next_pose)
                time.sleep(dt)
            else:
                raise RuntimeError(
                    f"Visual servo phase {phase_name} did not converge in "
                    f"{max_iterations} iterations"
                )
            if stable_count < SERVO_STABLE_FRAMES:
                raise RuntimeError(f"Visual servo phase {phase_name} ended without stable convergence")
            print(f"Visual servo phase {phase_name} converged")

        grasp_verified = None
        if execute_grasp:
            _validate_grasp_observation(state, last_observation)
            restored = _prepare_physics_grasp(sim, robot_base)
            if restored:
                print(f"Physics grasp preparation: {restored} generated bodies dynamic/respondable")
            print("Closing RG2 gripper...")
            if USE_CONNECTOR and CONNECTOR_KINEMATIC_ONLY:
                # Keep the arm and camera at the converged pose.  The connector
                # below is the explicit closed-gripper action and is what makes
                # the subsequent lift deterministic in a paused scene.
                _set_gripper_signal(sim, GRIPPER_CLOSE_VALUE)
                print(
                    "RG2 close command recorded without starting dynamics "
                    "(connector mode)"
                )
            else:
                close_hold_pose = ik.current_pose()
                before_close, after_close = _run_gripper_motion(
                    client,
                    sim,
                    GRIPPER_CLOSE_VALUE,
                    hold_ik=ik,
                    hold_pose=close_hold_pose,
                )
                if before_close is not None and after_close is not None:
                    gripper_close_motion = abs(after_close - before_close)
                    if gripper_close_motion < GRIPPER_JOINT_MOVE_EPS:
                        raise RuntimeError(
                            "RG2 close command produced no openCloseJoint motion; "
                            "check the RG2_open child-script signal and joint mode"
                        )
            if USE_CONNECTOR:
                connector_attachment = _attach_connector(
                    sim,
                    robot_base,
                    tip,
                    state,
                )
                print(
                    "Connector attachment: "
                    f"{connector_attachment['target_alias']} -> gripper tip"
                )
            grasp_verified = _lift_and_verify(
                client,
                sim,
                ik,
                state,
                table_plane,
                SERVO_LIFT_M,
                kinematic_only=bool(USE_CONNECTOR and CONNECTOR_KINEMATIC_ONLY),
            )
            print(f"Grasp lift verification: {'PASS' if grasp_verified else 'NOT_CONFIRMED'}")
            if not grasp_verified:
                raise RuntimeError(
                    "Grasp execution completed, but RGB-D lift verification failed"
                )
        else:
            print("Final visual-servo alignment reached; gripper execution was skipped")

        summary: dict[str, Any] = {
            "target_id": state.object_id,
            "target_class": state.class_name,
            "initial_center_m": np.asarray(target_prediction["center_m"], dtype=float).tolist(),
            "final_center_estimate_m": state.center_base_m.tolist(),
            "iterations": total_iterations,
            "target_lost_frames": total_lost_frames,
            "converged": True,
            "execute_grasp": bool(execute_grasp),
            "grasp_verified": grasp_verified,
            "connector_attachment": connector_attachment,
            "gripper_close_joint_motion": gripper_close_motion,
            "final_position_error_mm": None if final_error is None else final_error.position_norm_m * 1000.0,
            "final_rotation_error_deg": None if final_error is None else math.degrees(final_error.rotation_norm_rad),
            "uses_color_features": False,
            "controller": "rgbd_pbvs_incremental_ik",
            "grasp_orientation": str(grasp_orientation),
            "max_commanded_tilt_deg": GRASP_MAX_TILT_DEG,
            "control_mode": "closed_loop",
            "initial_translation_mm": None
            if initial_translation_m is None
            else (np.asarray(initial_translation_m) * 1000.0).tolist(),
            "initial_euler_deg": None
            if initial_euler_deg is None
            else np.asarray(initial_euler_deg).tolist(),
        }
    except Exception as exc:
        summary = {
            "target_id": state.object_id,
            "target_class": state.class_name,
            "iterations": total_iterations,
            "target_lost_frames": total_lost_frames,
            "converged": False,
            "execute_grasp": bool(execute_grasp),
            "connector_attachment": connector_attachment,
            "uses_color_features": False,
            "controller": "rgbd_pbvs_incremental_ik",
            "grasp_orientation": str(grasp_orientation),
            "control_mode": "closed_loop",
            "error": str(exc),
        }
        _write_outputs(summary, trace)
        raise
    finally:
        ik.close()

    _write_outputs(summary, trace)
    print(f"Visual servo summary: {SUMMARY_FILE.resolve()}")
    print(f"Visual servo trace: {TRACE_FILE.resolve()}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Geometry-only RGB-D visual servo")
    parser.add_argument("--target-id", type=int)
    parser.add_argument("--execute-grasp", action="store_true")
    parser.add_argument("--max-iterations", type=int, default=SERVO_MAX_ITERATIONS)
    parser.add_argument(
        "--grasp-orientation",
        choices=("auto", "top_down", "surface"),
        default=GRASP_ORIENTATION_MODE,
        help="geometry-only grasp frame selection",
    )
    parser.add_argument(
        "--initial-offset-mm",
        type=float,
        nargs=3,
        metavar=("DX", "DY", "DZ"),
        help="initial TCP translation disturbance in millimetres",
    )
    parser.add_argument(
        "--initial-euler-deg",
        type=float,
        nargs=3,
        metavar=("ROLL", "PITCH", "YAW"),
        help="initial TCP xyz Euler disturbance in degrees",
    )
    parser.add_argument("--open-loop", action="store_true")
    parser.add_argument(
        "--align-only",
        action="store_true",
        help="run only the common pregrasp alignment phase for comparisons",
    )
    args = parser.parse_args()
    run_visual_servo(
        target_id=args.target_id,
        execute_grasp=args.execute_grasp,
        max_iterations=max(1, int(args.max_iterations)),
        grasp_orientation=args.grasp_orientation,
        initial_translation_m=None
        if args.initial_offset_mm is None
        else np.asarray(args.initial_offset_mm, dtype=np.float64) / 1000.0,
        initial_euler_deg=None
        if args.initial_euler_deg is None
        else np.asarray(args.initial_euler_deg, dtype=np.float64),
        open_loop_only=bool(args.open_loop),
        align_only=bool(args.align_only),
    )


if __name__ == "__main__":
    main()
