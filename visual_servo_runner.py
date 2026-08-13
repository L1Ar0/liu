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
    vertical_half_extent,
)
from visual_servo_perception import (
    TargetObservation,
    TargetTrackState,
    capture_target_observation,
    select_servo_target,
    track_state_from_prediction,
    update_track_state,
)
from joint_torque_controller import JointTorqueController, TorqueControllerConfig


ROOT = Path(__file__).resolve().parent
RECOGNITION_FILE = ROOT / "recognition_output" / "recognition_results.json"
SEGMENTATION_FILE = ROOT / "segmentation_output" / "segmentation_metadata.json"
GROUND_TRUTH_FILE = ROOT / "random_scene_ground_truth.json"
OUTPUT_DIR = ROOT / "visual_servo_output"
SUMMARY_FILE = OUTPUT_DIR / "visual_servo_summary.json"
TRACE_FILE = OUTPUT_DIR / "visual_servo_trace.csv"

SERVO_RATE_HZ = float(os.environ.get("ROBOT_GRASP_SERVO_RATE_HZ", "15"))
SERVO_PREGRASP_STANDOFF_M = float(
    os.environ.get("ROBOT_GRASP_SERVO_PREGRASP_STANDOFF_M", "0.090")
)
# The final TCP is placed at the RG2 jaw plane, rather than above the
# object's visible top surface.  In the current scene the touch-pad plane is
# about 2.1 mm in front of ``gripper_tip`` along the approach axis.
SERVO_GRASP_PLANE_OFFSET_M = float(
    os.environ.get("ROBOT_GRASP_SERVO_GRASP_PLANE_OFFSET_M", "0.00213")
)
SERVO_MAX_ITERATIONS = int(
    os.environ.get("ROBOT_GRASP_SERVO_MAX_ITERATIONS", "150")
)
SERVO_LOST_LIMIT = int(os.environ.get("ROBOT_GRASP_SERVO_LOST_LIMIT", "5"))
SERVO_STABLE_FRAMES = int(os.environ.get("ROBOT_GRASP_SERVO_STABLE_FRAMES", "3"))
SERVO_PREGRASP_STEPS = int(os.environ.get("ROBOT_GRASP_SERVO_PREGRASP_STEPS", "60"))
SERVO_PREGRASP_DURATION_S = float(
    os.environ.get("ROBOT_GRASP_SERVO_PREGRASP_DURATION_S", "2.0")
)
SERVO_LIFT_STEPS = int(os.environ.get("ROBOT_GRASP_SERVO_LIFT_STEPS", "60"))
SERVO_LIFT_DURATION_S = float(
    os.environ.get("ROBOT_GRASP_SERVO_LIFT_DURATION_S", "2.0")
)
SERVO_PLACE_TRANSFER_STEPS = int(
    os.environ.get("ROBOT_GRASP_SERVO_PLACE_TRANSFER_STEPS", "90")
)
SERVO_PLACE_TRANSFER_DURATION_S = float(
    os.environ.get("ROBOT_GRASP_SERVO_PLACE_TRANSFER_DURATION_S", "3.0")
)
SERVO_PLACE_LOWER_STEPS = int(
    os.environ.get("ROBOT_GRASP_SERVO_PLACE_LOWER_STEPS", "60")
)
SERVO_PLACE_LOWER_DURATION_S = float(
    os.environ.get("ROBOT_GRASP_SERVO_PLACE_LOWER_DURATION_S", "2.0")
)
SERVO_PLACE_RETREAT_M = float(
    os.environ.get("ROBOT_GRASP_SERVO_PLACE_RETREAT_M", "0.080")
)
SERVO_POSITION_TOLERANCE_M = float(
    os.environ.get("ROBOT_GRASP_SERVO_POSITION_TOLERANCE_M", "0.003")
)
SERVO_FINAL_POSITION_TOLERANCE_M = float(
    os.environ.get("ROBOT_GRASP_SERVO_FINAL_POSITION_TOLERANCE_M", "0.0045")
)
SERVO_FINAL_MEDIAN_POSITION_TOLERANCE_M = float(
    os.environ.get("ROBOT_GRASP_SERVO_FINAL_MEDIAN_POSITION_TOLERANCE_M", "0.0035")
)
SERVO_FINAL_MAX_POSITION_TOLERANCE_M = float(
    os.environ.get("ROBOT_GRASP_SERVO_FINAL_MAX_POSITION_TOLERANCE_M", "0.005")
)
SERVO_FINAL_CENTER_STD_M = float(
    os.environ.get("ROBOT_GRASP_SERVO_FINAL_CENTER_STD_M", "0.0008")
)
SERVO_FINAL_WINDOW_SIZE = int(
    os.environ.get("ROBOT_GRASP_SERVO_FINAL_WINDOW_SIZE", "5")
)
SERVO_FINAL_RELAXED_MEDIAN_TOLERANCE_M = float(
    os.environ.get("ROBOT_GRASP_SERVO_FINAL_RELAXED_MEDIAN_TOLERANCE_M", "0.005")
)
SERVO_FINAL_RELAXED_MAX_TOLERANCE_M = float(
    os.environ.get("ROBOT_GRASP_SERVO_FINAL_RELAXED_MAX_TOLERANCE_M", "0.006")
)
SERVO_FINAL_RELAXED_CENTER_STD_M = float(
    os.environ.get("ROBOT_GRASP_SERVO_FINAL_RELAXED_CENTER_STD_M", "0.0012")
)
SERVO_ORIENTATION_TOLERANCE_RAD = math.radians(
    float(os.environ.get("ROBOT_GRASP_SERVO_ORIENTATION_TOLERANCE_DEG", "5"))
)
SERVO_FINAL_ORIENTATION_TOLERANCE_RAD = math.radians(
    float(os.environ.get("ROBOT_GRASP_SERVO_FINAL_ORIENTATION_TOLERANCE_DEG", "3"))
)
SERVO_MIN_TCP_TABLE_CLEARANCE_M = float(
    os.environ.get("ROBOT_GRASP_SERVO_MIN_TCP_TABLE_CLEARANCE_M", "0.015")
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
SERVO_CONE_MIN_CONFIDENCE = float(
    os.environ.get("ROBOT_GRASP_SERVO_CONE_MIN_CONFIDENCE", "0.40")
)
SERVO_CONE_MIN_POINTS = int(
    os.environ.get("ROBOT_GRASP_SERVO_CONE_MIN_POINTS", "300")
)
SERVO_RG2_LEFT_PAD_HALF_SPAN_M = float(
    os.environ.get("ROBOT_GRASP_RG2_LEFT_PAD_HALF_SPAN_M", "0.02586")
)
SERVO_RG2_RIGHT_PAD_HALF_SPAN_M = float(
    os.environ.get("ROBOT_GRASP_RG2_RIGHT_PAD_HALF_SPAN_M", "0.02583")
)
SERVO_JAW_CENTERLINE_TOLERANCE_M = float(
    os.environ.get("ROBOT_GRASP_SERVO_JAW_CENTERLINE_TOLERANCE_M", "0.0015")
)
SERVO_JAW_CLEARANCE_DIFFERENCE_TOLERANCE_M = float(
    os.environ.get(
        "ROBOT_GRASP_SERVO_JAW_CLEARANCE_DIFFERENCE_TOLERANCE_M",
        "0.001",
    )
)
SERVO_JAW_MAX_CENTERING_CORRECTION_M = float(
    os.environ.get("ROBOT_GRASP_SERVO_JAW_MAX_CENTERING_CORRECTION_M", "0.008")
)
SERVO_TRACK_CENTER_ALPHA = float(
    os.environ.get("ROBOT_GRASP_SERVO_TRACK_CENTER_ALPHA", "0.20")
)
SERVO_TRACK_MAX_UPDATE_M = float(
    os.environ.get("ROBOT_GRASP_SERVO_TRACK_MAX_UPDATE_M", "0.0015")
)
SERVO_ALIGN_MAX_LINEAR_SPEED_M_S = float(
    os.environ.get("ROBOT_GRASP_SERVO_ALIGN_MAX_LINEAR_SPEED_M_S", "0.025")
)
SERVO_APPROACH_MAX_LINEAR_SPEED_M_S = float(
    os.environ.get("ROBOT_GRASP_SERVO_APPROACH_MAX_LINEAR_SPEED_M_S", "0.030")
)
SERVO_APPROACH_SLOW_LINEAR_SPEED_M_S = float(
    os.environ.get("ROBOT_GRASP_SERVO_APPROACH_SLOW_LINEAR_SPEED_M_S", "0.008")
)
SERVO_APPROACH_SLOW_ZONE_M = float(
    os.environ.get("ROBOT_GRASP_SERVO_APPROACH_SLOW_ZONE_M", "0.020")
)
SERVO_MAX_ANGULAR_SPEED_DEG_S = float(
    os.environ.get("ROBOT_GRASP_SERVO_MAX_ANGULAR_SPEED_DEG_S", "18.0")
)
SERVO_ALIGN_MAX_TRANSLATION_STEP_M = float(
    os.environ.get("ROBOT_GRASP_SERVO_ALIGN_MAX_TRANSLATION_STEP_M", "0.0018")
)
SERVO_APPROACH_MAX_TRANSLATION_STEP_M = float(
    os.environ.get("ROBOT_GRASP_SERVO_APPROACH_MAX_TRANSLATION_STEP_M", "0.0020")
)
SERVO_APPROACH_SLOW_TRANSLATION_STEP_M = float(
    os.environ.get("ROBOT_GRASP_SERVO_APPROACH_SLOW_TRANSLATION_STEP_M", "0.0007")
)
SERVO_APPROACH_FINAL_LINEAR_SPEED_M_S = float(
    os.environ.get("ROBOT_GRASP_SERVO_APPROACH_FINAL_LINEAR_SPEED_M_S", "0.006")
)
SERVO_APPROACH_FINAL_TRANSLATION_STEP_M = float(
    os.environ.get("ROBOT_GRASP_SERVO_APPROACH_FINAL_TRANSLATION_STEP_M", "0.0004")
)
SERVO_APPROACH_FILTER_ZONE_M = float(
    os.environ.get("ROBOT_GRASP_SERVO_APPROACH_FILTER_ZONE_M", "0.020")
)
SERVO_APPROACH_FREEZE_ZONE_M = float(
    os.environ.get("ROBOT_GRASP_SERVO_APPROACH_FREEZE_ZONE_M", "0.008")
)
SERVO_APPROACH_FILTER_ALPHA = float(
    os.environ.get("ROBOT_GRASP_SERVO_APPROACH_FILTER_ALPHA", "0.08")
)
SERVO_APPROACH_MAX_UPDATE_M = float(
    os.environ.get("ROBOT_GRASP_SERVO_APPROACH_MAX_UPDATE_M", "0.0005")
)
SERVO_MAX_ROTATION_STEP_DEG = float(
    os.environ.get("ROBOT_GRASP_SERVO_MAX_ROTATION_STEP_DEG", "1.2")
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
DYNAMICS_CONTROL = os.environ.get("ROBOT_GRASP_DYNAMICS_CONTROL", "0").lower() not in {
    "0",
    "false",
    "no",
}
DYNAMICS_MODE = os.environ.get("ROBOT_GRASP_DYNAMICS_MODE", "torque").strip().lower()
if DYNAMICS_MODE not in {"torque", "position"}:
    DYNAMICS_MODE = "torque"

DROP_BOX_ALIAS_PREFIX = "grasp_drop_box_"
DROP_BOX_INNER_X_M = float(os.environ.get("ROBOT_GRASP_DROP_BOX_INNER_X_M", "0.140"))
DROP_BOX_INNER_Y_M = float(os.environ.get("ROBOT_GRASP_DROP_BOX_INNER_Y_M", "0.140"))
DROP_BOX_WALL_HEIGHT_M = float(
    os.environ.get("ROBOT_GRASP_DROP_BOX_WALL_HEIGHT_M", "0.080")
)
DROP_BOX_WALL_THICKNESS_M = float(
    os.environ.get("ROBOT_GRASP_DROP_BOX_WALL_THICKNESS_M", "0.008")
)
DROP_BOX_FLOOR_THICKNESS_M = float(
    os.environ.get("ROBOT_GRASP_DROP_BOX_FLOOR_THICKNESS_M", "0.008")
)
DROP_BOX_WORKSPACE_GAP_M = float(
    os.environ.get("ROBOT_GRASP_DROP_BOX_WORKSPACE_GAP_M", "0.040")
)
DROP_BOX_OBJECT_CLEARANCE_M = float(
    os.environ.get("ROBOT_GRASP_DROP_BOX_OBJECT_CLEARANCE_M", "0.003")
)


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


def _drop_box_geometry(table_plane: np.ndarray) -> dict[str, Any]:
    """Compute an external, reachable drop-box layout in Robot Base coordinates."""

    reference: dict[str, Any] = {}
    if GROUND_TRUTH_FILE.exists():
        try:
            reference = _load_json(GROUND_TRUTH_FILE).get("reference", {})
        except Exception:
            reference = {}
    center_x = float(
        os.environ.get(
            "ROBOT_GRASP_DROP_BOX_X_M",
            reference.get("workspace_center_x", 0.525),
        )
    )
    workspace_y = float(reference.get("workspace_center_y", -0.030))
    workspace_half_y = float(reference.get("workspace_half_y", 0.090))
    outer_y = DROP_BOX_INNER_Y_M + 2.0 * DROP_BOX_WALL_THICKNESS_M
    center_y = float(
        os.environ.get(
            "ROBOT_GRASP_DROP_BOX_Y_M",
            workspace_y
            - workspace_half_y
            - DROP_BOX_WORKSPACE_GAP_M
            - 0.5 * outer_y,
        )
    )
    plane = np.asarray(table_plane, dtype=np.float64).reshape(4)
    normal = plane[:3] / max(float(np.linalg.norm(plane[:3])), 1e-12)
    if abs(float(normal[2])) < 0.90:
        raise RuntimeError("Drop box requires a table plane with a usable Z component")
    table_z = float(
        -(plane[0] * center_x + plane[1] * center_y + plane[3])
        / max(abs(float(plane[2])), 1e-12)
    )
    return {
        "center_xy_m": [center_x, center_y],
        "table_z_m": table_z,
        "floor_top_z_m": table_z + DROP_BOX_FLOOR_THICKNESS_M,
        "inner_size_m": [DROP_BOX_INNER_X_M, DROP_BOX_INNER_Y_M],
        "outer_size_m": [
            DROP_BOX_INNER_X_M + 2.0 * DROP_BOX_WALL_THICKNESS_M,
            outer_y,
        ],
        "wall_height_m": DROP_BOX_WALL_HEIGHT_M,
    }


def _remove_drop_box_shapes(sim: Any) -> int:
    removed = 0
    for handle in sim.getObjectsInTree(sim.handle_scene, sim.sceneobject_shape, 0):
        try:
            alias = str(sim.getObjectAlias(handle))
        except Exception:
            continue
        if not alias.startswith(DROP_BOX_ALIAS_PREFIX):
            continue
        try:
            sim.removeObject(handle)
            removed += 1
        except Exception:
            pass
    return removed


def _create_drop_box(sim: Any, table_plane: np.ndarray) -> dict[str, Any]:
    """Create a static five-piece box outside the recognition workspace."""

    _remove_drop_box_shapes(sim)
    geometry = _drop_box_geometry(table_plane)
    center_x, center_y = geometry["center_xy_m"]
    outer_x, outer_y = geometry["outer_size_m"]
    thickness = DROP_BOX_WALL_THICKNESS_M
    floor_thickness = DROP_BOX_FLOOR_THICKNESS_M
    wall_height = DROP_BOX_WALL_HEIGHT_M
    floor_top = geometry["floor_top_z_m"]
    primitive_type = getattr(sim, "primitiveshape_cuboid")
    created: list[int] = []

    def add_piece(alias: str, size: list[float], center: list[float]) -> None:
        handle = int(sim.createPrimitiveShape(primitive_type, size, 0))
        created.append(handle)
        sim.setObjectAlias(handle, alias)
        sim.setObjectPose(handle, [*center, 0.0, 0.0, 0.0, 1.0], -1)
        static_param = getattr(sim, "shapeintparam_static", None)
        respondable_param = getattr(sim, "shapeintparam_respondable", None)
        if static_param is not None:
            try:
                sim.setObjectInt32Param(handle, static_param, 1)
            except Exception:
                pass
        if respondable_param is not None:
            try:
                sim.setObjectInt32Param(handle, respondable_param, 1)
            except Exception:
                pass
        try:
            sim.setShapeColor(
                handle,
                "",
                sim.colorcomponent_ambient_diffuse,
                [0.12, 0.35, 0.75],
            )
        except Exception:
            pass

    try:
        add_piece(
            f"{DROP_BOX_ALIAS_PREFIX}floor",
            [outer_x, outer_y, floor_thickness],
            [center_x, center_y, geometry["table_z_m"] + floor_thickness / 2.0],
        )
        add_piece(
            f"{DROP_BOX_ALIAS_PREFIX}left",
            [thickness, outer_y, wall_height],
            [center_x - DROP_BOX_INNER_X_M / 2.0 - thickness / 2.0, center_y, floor_top + wall_height / 2.0],
        )
        add_piece(
            f"{DROP_BOX_ALIAS_PREFIX}right",
            [thickness, outer_y, wall_height],
            [center_x + DROP_BOX_INNER_X_M / 2.0 + thickness / 2.0, center_y, floor_top + wall_height / 2.0],
        )
        add_piece(
            f"{DROP_BOX_ALIAS_PREFIX}front",
            [DROP_BOX_INNER_X_M, thickness, wall_height],
            [center_x, center_y - DROP_BOX_INNER_Y_M / 2.0 - thickness / 2.0, floor_top + wall_height / 2.0],
        )
        add_piece(
            f"{DROP_BOX_ALIAS_PREFIX}back",
            [DROP_BOX_INNER_X_M, thickness, wall_height],
            [center_x, center_y + DROP_BOX_INNER_Y_M / 2.0 + thickness / 2.0, floor_top + wall_height / 2.0],
        )
    except Exception:
        for handle in created:
            try:
                sim.removeObject(handle)
            except Exception:
                pass
        raise
    geometry["handles"] = created
    return geometry


def _target_grasp_pose(
    state: TargetTrackState,
    standoff_m: float,
    current_tip_rotation: np.ndarray,
    table_normal: np.ndarray,
    orientation_mode: str,
    centered_grasp: bool = False,
) -> np.ndarray:
    """Select a geometry-only target pose.

    ``centered_grasp`` is used only for the final approach.  Pregrasp and
    recovery poses must remain outside the object's bounding volume, while a
    closing RG2 needs its touch-pad plane to pass through the selected grasp
    section centre.  The returned TCP is therefore offset by a small,
    calibrated jaw-plane distance instead of by the object's half extent.
    """

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
        pose = build_surface_aligned_grasp_pose(
            state.center_base_m,
            state.rotation_base,
            state.dimensions_m,
            state.class_name,
            standoff_m,
            current_tip_rotation,
            table_normal,
            max_tilt_deg=GRASP_MAX_TILT_DEG,
        )
    else:
        pose = build_top_down_grasp_pose(
            state.center_base_m,
            state.rotation_base,
            state.dimensions_m,
            state.class_name,
            standoff_m,
            current_tip_rotation,
            table_normal,
        )
    if centered_grasp:
        approach_axis = -np.asarray(pose[:3, 2], dtype=np.float64)
        grasp_center = np.asarray(state.center_base_m, dtype=np.float64)
        # Use a wider, lower cone section for the parallel jaws.  This is a
        # catalogue-geometric correction, not a ground-truth pose lookup.
        if state.class_name == "cone" and not use_surface:
            grasp_center = _cone_cross_section_center(state, table_normal)
        pose[:3, 3] = (
            grasp_center + approach_axis * SERVO_GRASP_PLANE_OFFSET_M
        )
    return pose


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

    def __init__(
        self,
        sim: Any,
        sim_ik: Any,
        robot_base: int,
        tip: int,
        client: RemoteAPIClient | None = None,
        dynamics_control: bool = False,
        dynamics_mode: str = DYNAMICS_MODE,
    ) -> None:
        state = sim.getSimulationState()
        if state not in {sim.simulation_stopped, sim.simulation_paused}:
            raise RuntimeError(
                "Visual servo requires a stopped or paused CoppeliaSim simulation"
            )
        self.sim = sim
        self.sim_ik = sim_ik
        self.client = client
        self.dynamics_control = bool(dynamics_control)
        self.dynamics_mode = str(dynamics_mode).strip().lower()
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
        self.ik_joints: list[int] = []
        self.torque_controller: JointTorqueController | None = None
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
        self.ik_joints = [int(joint) for joint in ik_joints]
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

    def _ensure_torque_controller(self) -> JointTorqueController:
        if not self.dynamics_control:
            raise RuntimeError("Torque controller requested for a kinematic IK instance")
        if self.client is None:
            raise RuntimeError("Dynamic visual servo requires a RemoteAPIClient")
        if self.torque_controller is None:
            self.torque_controller = JointTorqueController(
                self.client,
                self.sim,
                self.joints,
                TorqueControllerConfig.from_environment(),
                control_mode=self.dynamics_mode,
            )
            self.torque_controller.start()
        return self.torque_controller

    def apply(self, target_pose: np.ndarray) -> np.ndarray:
        self.sim.setObjectPose(
            self.target,
            pose_from_matrix(target_pose),
            self.robot_base,
        )
        if self.dynamics_control:
            # The IK world is only a reference model.  Refresh it from the
            # measured dynamic arm before every solve so q_des is generated
            # from actual q, never from the previous kinematic solution.
            self.sim_ik.syncFromSim(self.environment, [self.group])
        result, flags, precision = self.sim_ik.handleGroup(
            self.environment,
            self.group,
            {"syncWorlds": not self.dynamics_control},
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
        if self.dynamics_control:
            controller = self._ensure_torque_controller()
            desired_q = [
                float(self.sim_ik.getJointPosition(self.environment, joint))
                for joint in self.ik_joints
            ]
            controller.step(desired_q)
            return self.current_pose()
        return self.current_pose()

    def move_linear(
        self,
        target_pose: np.ndarray,
        steps: int,
        table_plane: np.ndarray,
        duration_s: float | None = None,
    ) -> np.ndarray:
        start = self.current_pose()
        final = np.asarray(target_pose, dtype=np.float64)
        step_count = max(1, int(steps))
        duration = (
            float(duration_s)
            if duration_s is not None
            else step_count / max(SERVO_RATE_HZ, 1.0)
        )
        start_rotation = Rotation.from_matrix(start[:3, :3])
        rotation_delta = (
            start_rotation.inv() * Rotation.from_matrix(final[:3, :3])
        ).as_rotvec()
        last = start
        for index in range(1, step_count + 1):
            ratio = index / step_count
            # Cubic smoothstep gives zero velocity at both ends.  The same
            # scalar is used for translation and quaternion-equivalent SLERP
            # so the TCP does not snap into its final orientation.
            smooth_ratio = ratio * ratio * (3.0 - 2.0 * ratio)
            position = (
                (1.0 - smooth_ratio) * start[:3, 3]
                + smooth_ratio * final[:3, 3]
            )
            if _table_signed_height(position, table_plane) < SERVO_MIN_TCP_TABLE_CLEARANCE_M:
                raise RuntimeError("Visual-servo path violated table clearance")
            rotation = (
                start_rotation * Rotation.from_rotvec(smooth_ratio * rotation_delta)
            ).as_matrix()
            pose = np.eye(4, dtype=np.float64)
            pose[:3, 3] = position
            pose[:3, :3] = rotation
            last = self.apply(pose)
            time.sleep(duration / step_count)
        return last

    def close(self) -> None:
        if self.torque_controller is not None:
            try:
                self.torque_controller.close()
            except Exception:
                pass
            self.torque_controller = None
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


def _cone_cross_section_center(
    state: TargetTrackState,
    table_normal: np.ndarray,
    fraction_from_base: float = 0.35,
) -> np.ndarray:
    """Return a stable side-grasp point on the cone's useful cross-section.

    A cone is not uniformly wide along its axis.  Grasping at the model centre
    therefore places the RG2 pads on a smaller section than expected and makes
    a one-sided contact likely.  The catalogue cone is parameterised with its
    local Z axis from base to tip, so a point around 35% of the height gives a
    repeatable, approximately 31 mm diameter section for the 45/5/60 mm CAD.
    For a strongly leaning cone, the generic surface-aligned pose remains the
    safer choice and this correction is intentionally skipped.
    """

    center = np.asarray(state.center_base_m, dtype=np.float64)
    rotation = np.asarray(state.rotation_base, dtype=np.float64)
    normal = np.asarray(table_normal, dtype=np.float64)
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    axis = rotation[:, 2]
    if float(np.dot(axis, normal)) < 0.0:
        axis = -axis
    if abs(float(np.dot(axis, normal))) < math.cos(math.radians(25.0)):
        return center
    height = float(state.dimensions_m[2])
    fraction = float(np.clip(fraction_from_base, 0.20, 0.50))
    base_center = center - axis * (0.5 * height)
    return base_center + axis * (fraction * height)


def _approach_window_metrics(
    position_errors_m: list[float],
    center_history_m: list[np.ndarray],
    window_size: int = SERVO_FINAL_WINDOW_SIZE,
) -> tuple[float, float, float] | None:
    """Compute robust final-approach error metrics from recent RGB-D frames."""

    size = max(1, int(window_size))
    if len(position_errors_m) < size or len(center_history_m) < size:
        return None
    errors = np.asarray(position_errors_m[-size:], dtype=np.float64)
    centers = np.asarray(center_history_m[-size:], dtype=np.float64)
    median_error = float(np.median(errors))
    maximum_error = float(np.max(errors))
    center_std = float(np.max(np.std(centers, axis=0)))
    return median_error, maximum_error, center_std


def _approach_window_converged(
    position_errors_m: list[float],
    center_history_m: list[np.ndarray],
    relaxed: bool = False,
) -> bool:
    metrics = _approach_window_metrics(position_errors_m, center_history_m)
    if metrics is None:
        return False
    median_error, maximum_error, center_std = metrics
    if relaxed:
        return (
            median_error <= SERVO_FINAL_RELAXED_MEDIAN_TOLERANCE_M
            and maximum_error <= SERVO_FINAL_RELAXED_MAX_TOLERANCE_M
            and center_std <= SERVO_FINAL_RELAXED_CENTER_STD_M
        )
    return (
        median_error <= SERVO_FINAL_MEDIAN_POSITION_TOLERANCE_M
        and maximum_error <= SERVO_FINAL_MAX_POSITION_TOLERANCE_M
        and center_std <= SERVO_FINAL_CENTER_STD_M
    )


def _grasp_section_half_width(
    state: TargetTrackState,
    jaw_axis: np.ndarray,
    table_normal: np.ndarray,
) -> float:
    """Return the catalogue half-width seen by the two RG2 pads."""

    axis = np.asarray(jaw_axis, dtype=np.float64)
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    if state.class_name == "cone":
        base_diameter = float(state.dimensions_m[0])
        top_diameter = float(
            os.environ.get("ROBOT_GRASP_CONE_TOP_DIAMETER_M", "0.005")
        )
        fraction = 0.35
        return 0.5 * (
            base_diameter + fraction * (top_diameter - base_diameter)
        )
    if state.class_name == "sphere":
        return 0.5 * float(state.dimensions_m[0])
    if state.class_name == "cylinder":
        # The cylinder is rotationally symmetric around its fitted axis.  A
        # noisy PCA frame must not project the 60 mm height into the jaw
        # direction and falsely report an over-wide 52 mm section.
        return 0.5 * float(state.dimensions_m[0])
    rotation = np.asarray(state.rotation_base, dtype=np.float64)
    return 0.5 * float(
        np.sum(np.abs(rotation.T @ axis) * state.dimensions_m)
    )


def _jaw_centering_metrics(
    current_tip_pose: np.ndarray,
    desired_tip_pose: np.ndarray,
    state: TargetTrackState,
    table_normal: np.ndarray,
) -> dict[str, float]:
    """Estimate bilateral pad clearances without reading object ground truth."""

    current = np.asarray(current_tip_pose, dtype=np.float64)
    desired = np.asarray(desired_tip_pose, dtype=np.float64)
    approach_axis = -desired[:3, 2]
    approach_axis /= max(float(np.linalg.norm(approach_axis)), 1e-12)
    jaw_axis = desired[:3, 0]
    jaw_axis /= max(float(np.linalg.norm(jaw_axis)), 1e-12)
    grasp_center = desired[:3, 3] - approach_axis * SERVO_GRASP_PLANE_OFFSET_M
    jaw_midpoint = current[:3, 3] - approach_axis * SERVO_GRASP_PLANE_OFFSET_M
    center_delta = grasp_center - jaw_midpoint
    center_offset = float(np.dot(center_delta, jaw_axis))
    lateral_delta = center_delta - float(np.dot(center_delta, approach_axis)) * approach_axis
    half_width = _grasp_section_half_width(state, jaw_axis, table_normal)
    left_clearance = SERVO_RG2_LEFT_PAD_HALF_SPAN_M - (half_width + center_offset)
    right_clearance = SERVO_RG2_RIGHT_PAD_HALF_SPAN_M - (half_width - center_offset)
    return {
        "centerline_error_m": float(np.linalg.norm(lateral_delta)),
        "position_error_m": float(np.linalg.norm(desired[:3, 3] - current[:3, 3])),
        "left_clearance_m": float(left_clearance),
        "right_clearance_m": float(right_clearance),
        "clearance_difference_m": float(abs(left_clearance - right_clearance)),
        "section_width_m": float(2.0 * half_width),
    }


def _center_gripper_before_close(
    ik: IncrementalIK,
    state: TargetTrackState,
    table_plane: np.ndarray,
    grasp_orientation: str,
) -> dict[str, float]:
    """Place the fitted target section on the calibrated RG2 jaw midpoint."""

    current = ik.current_pose()
    desired = _target_grasp_pose(
        state,
        SERVO_GRASP_PLANE_OFFSET_M,
        current[:3, :3],
        table_plane[:3],
        grasp_orientation,
        centered_grasp=True,
    )
    before = _jaw_centering_metrics(current, desired, state, table_plane[:3])
    correction = float(before["position_error_m"])
    if correction > SERVO_JAW_MAX_CENTERING_CORRECTION_M:
        raise RuntimeError(
            "Refusing grasp: final jaw-centering correction is too large "
            f"({correction * 1000.0:.2f} mm)"
        )
    if correction > 0.0002:
        steps = max(8, int(math.ceil(correction / 0.0004)))
        duration = max(0.6, correction / max(SERVO_APPROACH_FINAL_LINEAR_SPEED_M_S, 1e-6))
        ik.move_linear(
            desired,
            steps,
            table_plane,
            duration_s=duration,
        )
    after = _jaw_centering_metrics(
        ik.current_pose(),
        desired,
        state,
        table_plane[:3],
    )
    if after["centerline_error_m"] > SERVO_JAW_CENTERLINE_TOLERANCE_M:
        raise RuntimeError(
            "Refusing grasp: object section is not centered between RG2 pads "
            f"({after['centerline_error_m'] * 1000.0:.2f} mm)"
        )
    if (
        after["clearance_difference_m"]
        > SERVO_JAW_CLEARANCE_DIFFERENCE_TOLERANCE_M
    ):
        raise RuntimeError(
            "Refusing grasp: predicted left/right jaw clearances differ by "
            f"{after['clearance_difference_m'] * 1000.0:.2f} mm"
        )
    if after["left_clearance_m"] < -0.001 or after["right_clearance_m"] < -0.001:
        raise RuntimeError(
            "Refusing grasp: target cross-section exceeds the calibrated RG2 opening"
        )
    return {
        "correction_mm": correction * 1000.0,
        "centerline_error_mm": after["centerline_error_m"] * 1000.0,
        "left_clearance_mm": after["left_clearance_m"] * 1000.0,
        "right_clearance_mm": after["right_clearance_m"] * 1000.0,
        "clearance_difference_mm": after["clearance_difference_m"] * 1000.0,
        "section_width_mm": after["section_width_m"] * 1000.0,
    }


def _set_gripper_signal(sim: Any, value: int) -> None:
    try:
        sim.clearFloatSignal(GRIPPER_SIGNAL_NAME)
    except Exception:
        pass
    value = int(value)
    # Current RG2 models read the namespaced property signal.  Keep the old
    # signal as a compatibility fallback for older scenes.
    try:
        sim.setIntProperty(sim.handle_scene, f"signal.{GRIPPER_SIGNAL_NAME}", value)
    except Exception:
        pass
    try:
        sim.setInt32Signal(GRIPPER_SIGNAL_NAME, value)
    except Exception:
        pass


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
        if hold_ik.dynamics_control and hold_ik.torque_controller is not None:
            # Never overwrite a dynamic arm with setJointPosition.  Refresh
            # the seven torque commands before each synchronous physics step.
            hold_ik.torque_controller.apply_torque()
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


def _run_dynamic_rg2_motion(
    client: RemoteAPIClient,
    sim: Any,
    value: int,
    steps: int = GRIPPER_STEPS,
    arm_controller: JointTorqueController | None = None,
    require_motion: bool = True,
) -> tuple[float | None, float | None]:
    """Run the stock RG2 child script in true dynamics and verify its motor."""
    drive_joint = _find_gripper_drive_joint(sim)
    before = None
    if drive_joint is not None:
        before = float(sim.getJointPosition(drive_joint))
    _set_gripper_signal(sim, value)
    client.setStepping(True)
    state = sim.getSimulationState()
    if state in {sim.simulation_stopped, sim.simulation_paused}:
        sim.startSimulation()
    # RG2's driver joint is a dynamic velocity-controlled joint.  The child
    # script normally writes this command, but external stepping can encounter
    # a paused/disabled child script.  Set the same low-level command here so
    # the gripper remains deterministic while retaining its dynamic contacts.
    drive_mode = getattr(sim, "jointmode_dynamic", 5)
    velocity_mode = getattr(sim, "jointdynctrl_velocity", 4)
    if drive_joint is not None:
        try:
            sim.setJointMode(drive_joint, drive_mode)
            sim.setObjectInt32Param(drive_joint, sim.jointintparam_dynctrlmode, velocity_mode)
            sim.setJointTargetForce(drive_joint, 20.0)
            sim.setJointTargetVelocity(drive_joint, 0.05 if int(value) != 0 else -0.05)
        except Exception:
            pass
    for _ in range(max(1, int(steps))):
        if arm_controller is not None:
            if arm_controller.control_mode == "position":
                arm_controller.apply_position_target()
            else:
                arm_controller.apply_torque()
        client.step()
    after = None
    if drive_joint is not None:
        after = float(sim.getJointPosition(drive_joint))
    if before is not None and after is not None:
        print(
            "RG2 dynamic openCloseJoint: "
            f"{before:.6f} -> {after:.6f} "
            f"(motion={abs(after - before):.6f})"
        )
        # Opening from an already-open hard stop legitimately produces zero
        # motion.  Closing must move because it is the grasp action.
        if require_motion and abs(after - before) < GRIPPER_JOINT_MOVE_EPS:
            raise RuntimeError(
                "RG2 dynamic command produced no openCloseJoint motion; "
                "check RG2_open signal, dynamic mode, and child script"
            )
    if drive_joint is not None:
        try:
            sim.setJointTargetVelocity(drive_joint, 0.0)
        except Exception:
            pass
    return before, after


def _prepare_physics_grasp(
    sim: Any,
    robot_base: int,
    target_state: TargetTrackState | None = None,
    force_dynamic: bool = False,
) -> int:
    """Restore dynamic/respondable state for generated physics objects.

    The scene remains paused during visual servoing, but a body must stay
    dynamic so the fingers can transfer force to it during the lift. This
    helper also restores the dataset collision bit on the robot hierarchy,
    which is useful when a scene was interrupted during generation.
    """

    scene_mode = os.environ.get("ROBOT_GRASP_SCENE_MODE", "").lower()
    if not force_dynamic and scene_mode not in {
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
    target_handle = None
    if target_state is not None:
        try:
            target_handle, _ = _resolve_target_shape(sim, robot_base, target_state)
        except Exception:
            target_handle = None
    dynamic_all = scene_mode in {"physics", "dynamic", "settled", "drop"}
    dynamic_all = dynamic_all or os.environ.get(
        "ROBOT_GRASP_DYNAMIC_ALL_WORKPIECES", "0"
    ).lower() not in {"0", "false", "no"}
    for handle in shapes:
        try:
            alias = str(sim.getObjectAlias(handle)).lower()
        except Exception:
            continue
        if not alias.startswith("rand_"):
            continue
        if not dynamic_all and target_handle is not None and int(handle) != int(target_handle):
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


def _assert_dynamic_target(sim: Any, robot_base: int, target_state: TargetTrackState) -> dict[str, Any]:
    """Fail loudly instead of silently running a pseudo-physical grasp."""
    handle, alias = _resolve_target_shape(sim, robot_base, target_state)
    static_param = getattr(sim, "shapeintparam_static", None)
    respondable_param = getattr(sim, "shapeintparam_respondable", None)
    static = None if static_param is None else int(sim.getObjectInt32Param(handle, static_param))
    respondable = None if respondable_param is None else int(sim.getObjectInt32Param(handle, respondable_param))
    if static == 1 or respondable == 0:
        raise RuntimeError(
            f"Physical grasp target {alias} is not dynamic/respondable "
            f"(static={static}, respondable={respondable})"
        )
    dynamic = False
    try:
        dynamic = bool(sim.isDynamicallyEnabled(handle))
    except Exception:
        pass
    return {"target_handle": int(handle), "target_alias": alias, "static": static, "respondable": respondable, "dynamically_enabled": dynamic}


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
    target_pose = list(sim.getObjectPose(target_handle, -1))
    tip_position = np.asarray(tip_pose[:3], dtype=np.float64)
    connector_handle: int | None = None
    connector_alias = f"grasp_connector_{target_alias}"
    try:
        connector_handle = int(sim.createDummy(0.002))
        try:
            sim.setObjectAlias(connector_handle, connector_alias)
        except Exception:
            pass
        try:
            # Place the connector at the object's current world pose first.
            # Parenting it to the tip with keepInPlace=True preserves the
            # measured object-to-tip offset during the subsequent lift.
            sim.setObjectPose(connector_handle, target_pose, -1)
            sim.setObjectParent(connector_handle, tip, True)
            sim.setObjectParent(target_handle, connector_handle, True)
        except Exception:
            # Fallback still preserves the current world pose.  It is less
            # explicit than the connector dummy but never snaps the object.
            sim.setObjectParent(target_handle, tip, True)
            connector_handle = None
    except Exception:
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
        "center_snapped_to_tip": False,
        "relative_attachment_preserved": True,
        "target_center_before_m": target_center_before.tolist(),
        "target_center_after_m": target_center_after.tolist(),
        "tip_center_m": [float(value) for value in tip_pose[:3]],
        "snap_distance_mm": float(
            np.linalg.norm(target_center_after - target_center_before) * 1000.0
        ),
        "attachment_offset_m": (
            target_center_before - tip_position
        ).tolist(),
    }


def _validate_grasp_observation(
    state: TargetTrackState,
    observation: TargetObservation | None,
) -> None:
    """Reject an obviously weak target before closing the physical gripper."""

    required_confidence = (
        SERVO_CONE_MIN_CONFIDENCE
        if state.class_name == "cone"
        else SERVO_GRASP_MIN_CONFIDENCE
    )
    if state.confidence < required_confidence:
        raise RuntimeError(
            "Refusing physical grasp: target confidence is "
            f"{state.confidence:.2f} < {required_confidence:.2f}"
        )
    if observation is None:
        raise RuntimeError("Refusing physical grasp: no final RGB-D target observation")
    minimum_points = (
        SERVO_CONE_MIN_POINTS
        if state.class_name == "cone"
        else SERVO_GRASP_MIN_POINTS
    )
    if observation.point_count < minimum_points:
        raise RuntimeError(
            "Refusing physical grasp: final target point count is "
            f"{observation.point_count} < {minimum_points}"
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
    lift_steps = max(1, int(SERVO_LIFT_STEPS))
    lift_duration = max(0.0, float(SERVO_LIFT_DURATION_S))
    if kinematic_only:
        for index in range(1, lift_steps + 1):
            ratio = index / lift_steps
            smooth_ratio = ratio * ratio * (3.0 - 2.0 * ratio)
            pose = start.copy()
            pose[:3, 3] = (
                (1.0 - smooth_ratio) * start[:3, 3]
                + smooth_ratio * lifted[:3, 3]
            )
            ik.apply(pose)
            time.sleep(lift_duration / lift_steps)
    else:
        client.setStepping(True)
        if sim.getSimulationState() in {sim.simulation_stopped, sim.simulation_paused}:
            sim.startSimulation()
        for index in range(1, lift_steps + 1):
            ratio = index / lift_steps
            smooth_ratio = ratio * ratio * (3.0 - 2.0 * ratio)
            pose = start.copy()
            pose[:3, 3] = (
                (1.0 - smooth_ratio) * start[:3, 3]
                + smooth_ratio * lifted[:3, 3]
            )
            ik.apply(pose)
            client.step()
            if lift_duration > 0.0:
                time.sleep(lift_duration / lift_steps)
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


def _place_attached_target_in_box(
    sim: Any,
    ik: IncrementalIK,
    target_state: TargetTrackState,
    table_plane: np.ndarray,
    connector_attachment: dict[str, Any],
) -> dict[str, Any]:
    """Transfer an attached workpiece to the external box and release it."""

    target_handle = int(connector_attachment["target_handle"])
    connector_handle = connector_attachment.get("connector_handle")
    box = _create_drop_box(sim, table_plane)
    normal = np.asarray(table_plane[:3], dtype=np.float64)
    normal /= max(float(np.linalg.norm(normal)), 1e-12)

    target_pose = matrix_from_pose(sim.getObjectPose(target_handle, ik.robot_base))
    target_center = target_pose[:3, 3].copy()
    box_x, box_y = box["center_xy_m"]

    transfer_pose = ik.current_pose()
    transfer_pose[:3, 3] += np.asarray(
        [box_x - target_center[0], box_y - target_center[1], 0.0],
        dtype=np.float64,
    )
    print(
        "Transferring grasped object to external drop box at "
        f"({box_x:.3f}, {box_y:.3f}) m..."
    )
    ik.move_linear(
        transfer_pose,
        SERVO_PLACE_TRANSFER_STEPS,
        table_plane,
        duration_s=SERVO_PLACE_TRANSFER_DURATION_S,
    )

    target_pose = matrix_from_pose(sim.getObjectPose(target_handle, ik.robot_base))
    half_extent = vertical_half_extent(
        target_pose[:3, :3],
        target_state.dimensions_m,
        normal,
    )
    desired_center = np.asarray(
        [
            box_x,
            box_y,
            float(box["floor_top_z_m"])
            + half_extent
            + DROP_BOX_OBJECT_CLEARANCE_M,
        ],
        dtype=np.float64,
    )
    lower_pose = ik.current_pose()
    lower_pose[:3, 3] += desired_center - target_pose[:3, 3]
    ik.move_linear(
        lower_pose,
        SERVO_PLACE_LOWER_STEPS,
        table_plane,
        duration_s=SERVO_PLACE_LOWER_DURATION_S,
    )

    sim.setObjectParent(target_handle, -1, True)
    static_param = getattr(sim, "shapeintparam_static", None)
    respondable_param = getattr(sim, "shapeintparam_respondable", None)
    if static_param is not None:
        try:
            sim.setObjectInt32Param(target_handle, static_param, 1)
        except Exception:
            pass
    if respondable_param is not None:
        try:
            sim.setObjectInt32Param(target_handle, respondable_param, 1)
        except Exception:
            pass
    if connector_handle is not None:
        try:
            sim.removeObject(int(connector_handle))
        except Exception:
            pass
    _set_gripper_signal(sim, GRIPPER_OPEN_VALUE)

    placed_center = np.asarray(
        sim.getObjectPosition(target_handle, ik.robot_base), dtype=np.float64
    )
    placement_error = float(np.linalg.norm(placed_center - desired_center))
    if placement_error > 0.010:
        raise RuntimeError(
            "Drop-box placement error exceeded 10 mm: "
            f"{placement_error * 1000.0:.2f} mm"
        )

    retreat_pose = ik.current_pose()
    retreat_pose[:3, 3] += normal * SERVO_PLACE_RETREAT_M
    ik.move_linear(
        retreat_pose,
        max(1, SERVO_PLACE_LOWER_STEPS // 2),
        table_plane,
        duration_s=max(0.5, SERVO_PLACE_LOWER_DURATION_S / 2.0),
    )
    connector_attachment["attached"] = False
    connector_attachment["released_to_box"] = True
    connector_attachment["connector_handle"] = None
    print(
        "Drop-box placement: PASS "
        f"(error={placement_error * 1000.0:.2f} mm)"
    )
    return {
        "completed": True,
        "target_handle": target_handle,
        "target_alias": connector_attachment.get("target_alias"),
        "box_center_m": [box_x, box_y, float(box["floor_top_z_m"])],
        "box_handles": [int(value) for value in box["handles"]],
        "desired_object_center_m": desired_center.tolist(),
        "placed_object_center_m": placed_center.tolist(),
        "placement_error_mm": placement_error * 1000.0,
        "released": True,
        "retreated": True,
    }


def run_visual_servo(
    target_id: int | None = None,
    execute_grasp: bool = False,
    max_iterations: int = SERVO_MAX_ITERATIONS,
    grasp_orientation: str = GRASP_ORIENTATION_MODE,
    target_policy: str = "upper-random",
    selection_seed: int | None = None,
    place_in_box: bool = False,
    initial_translation_m: np.ndarray | None = None,
    initial_euler_deg: np.ndarray | None = None,
    open_loop_only: bool = False,
    align_only: bool = False,
    dynamics_control: bool = DYNAMICS_CONTROL,
    dynamics_mode: str = DYNAMICS_MODE,
) -> dict[str, Any]:
    if place_in_box and not execute_grasp:
        raise RuntimeError("Drop-box placement requires grasp execution")
    if place_in_box and (not USE_CONNECTOR or dynamics_control):
        raise RuntimeError("Drop-box placement is only available in connector mode")
    recognition = _load_json(RECOGNITION_FILE)
    segmentation = _load_json(SEGMENTATION_FILE)
    target_prediction = select_servo_target(
        recognition,
        target_id,
        policy=target_policy,
        random_seed=selection_seed,
    )
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

    if dynamics_control and open_loop_only:
        raise RuntimeError("Dynamic control requires the closed-loop stepping path")
    ik = IncrementalIK(
        sim,
        sim_ik,
        robot_base,
        tip,
        client=client,
        dynamics_control=bool(dynamics_control),
        dynamics_mode=dynamics_mode,
    )
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
    jaw_centering: dict[str, float] | None = None
    connector_attachment: dict[str, Any] | None = None
    placement_result: dict[str, Any] | None = None
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
        current = ik.move_linear(
            initial_pose,
            SERVO_PREGRASP_STEPS,
            table_plane,
            duration_s=SERVO_PREGRASP_DURATION_S,
        )
        if execute_grasp and (dynamics_control or not (USE_CONNECTOR and CONNECTOR_KINEMATIC_ONLY)):
            # Active second-view motion may leave the TCP only a few millimetres
            # above the clearance gate. First retreat to the high pregrasp,
            # then open RG2 so finger motion cannot invalidate the safe path.
            print("Opening RG2 gripper at the safe pregrasp pose...")
            if dynamics_control:
                _run_dynamic_rg2_motion(
                    client, sim, GRIPPER_OPEN_VALUE,
                    arm_controller=ik.torque_controller,
                    require_motion=False,
                )
            else:
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
            state = update_track_state(
                state,
                observation,
                center_alpha=SERVO_TRACK_CENTER_ALPHA,
                max_center_update_m=SERVO_TRACK_MAX_UPDATE_M,
            )
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
                False,
            ),
            (
                "approach",
                SERVO_GRASP_PLANE_OFFSET_M,
                SERVO_FINAL_POSITION_TOLERANCE_M,
                SERVO_FINAL_ORIENTATION_TOLERANCE_RAD,
                True,
            ),
        ]
        if align_only:
            phases = phases[:1]
        for phase_name, standoff, position_tol, orientation_tol, centered_grasp in phases:
            print(f"\n--- Visual servo phase: {phase_name} ---")
            phase_controller = PBVSController(
                position_gain=1.2 if phase_name == "approach" else 1.4,
                rotation_gain=0.9 if phase_name == "approach" else 1.1,
                max_linear_speed_m_s=(
                    SERVO_APPROACH_MAX_LINEAR_SPEED_M_S
                    if phase_name == "approach"
                    else SERVO_ALIGN_MAX_LINEAR_SPEED_M_S
                ),
                max_angular_speed_rad_s=math.radians(
                    SERVO_MAX_ANGULAR_SPEED_DEG_S
                ),
                max_translation_step_m=(
                    SERVO_APPROACH_MAX_TRANSLATION_STEP_M
                    if phase_name == "approach"
                    else SERVO_ALIGN_MAX_TRANSLATION_STEP_M
                ),
                max_rotation_step_rad=math.radians(SERVO_MAX_ROTATION_STEP_DEG),
            )
            stable_count = 0
            approach_error_history: list[float] = []
            approach_observation_history: list[np.ndarray] = []
            approach_center_history: list[np.ndarray] = []
            approach_center_anchor: np.ndarray | None = None
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
                current = ik.current_pose()
                if phase_name == "approach":
                    # The last few RGB-D frames are affected by depth
                    # quantisation and by the approaching jaws.  Track their
                    # median, then freeze the centre in the final 8 mm so PBVS
                    # does not oscillate around a moving measurement.
                    approach_observation_history.append(
                        np.asarray(observation.center_base_m, dtype=np.float64).copy()
                    )
                    history_limit = max(10, 2 * SERVO_FINAL_WINDOW_SIZE)
                    if len(approach_observation_history) > history_limit:
                        del approach_observation_history[:-history_limit]
                    median_center = np.median(
                        np.asarray(approach_observation_history, dtype=np.float64)[-SERVO_FINAL_WINDOW_SIZE:],
                        axis=0,
                    )
                    filtered_observation = TargetObservation(
                        center_base_m=median_center,
                        rotation_base=observation.rotation_base,
                        point_count=observation.point_count,
                        confidence=observation.confidence,
                        image_pixel=observation.image_pixel,
                        image_margin_px=observation.image_margin_px,
                        extent_m=observation.extent_m,
                    )
                    probe_desired = _target_grasp_pose(
                        state,
                        standoff,
                        current[:3, :3],
                        table_plane[:3],
                        grasp_orientation,
                        centered_grasp=True,
                    )
                    probe_error = phase_controller.pose_error(current, probe_desired)
                    if approach_center_anchor is None:
                        if probe_error.position_norm_m <= SERVO_APPROACH_FREEZE_ZONE_M:
                            state = update_track_state(
                                state,
                                filtered_observation,
                                center_alpha=1.0,
                                max_center_update_m=SERVO_APPROACH_MAX_UPDATE_M,
                            )
                            approach_center_anchor = state.center_base_m.copy()
                        else:
                            state = update_track_state(
                                state,
                                filtered_observation,
                                center_alpha=(
                                    SERVO_APPROACH_FILTER_ALPHA
                                    if probe_error.position_norm_m <= SERVO_APPROACH_FILTER_ZONE_M
                                    else SERVO_TRACK_CENTER_ALPHA
                                ),
                                max_center_update_m=(
                                    SERVO_APPROACH_MAX_UPDATE_M
                                    if probe_error.position_norm_m <= SERVO_APPROACH_FILTER_ZONE_M
                                    else SERVO_TRACK_MAX_UPDATE_M
                                ),
                            )
                    else:
                        state = TargetTrackState(
                            object_id=state.object_id,
                            class_name=state.class_name,
                            center_base_m=approach_center_anchor.copy(),
                            rotation_base=observation.rotation_base,
                            dimensions_m=state.dimensions_m.copy(),
                            confidence=float(observation.confidence),
                            anchor_center_base_m=(
                                None
                                if state.anchor_center_base_m is None
                                else state.anchor_center_base_m.copy()
                            ),
                            anchor_rotation_base=(
                                None
                                if state.anchor_rotation_base is None
                                else state.anchor_rotation_base.copy()
                            ),
                        )
                else:
                    state = update_track_state(
                        state,
                        observation,
                        center_alpha=SERVO_TRACK_CENTER_ALPHA,
                        max_center_update_m=SERVO_TRACK_MAX_UPDATE_M,
                    )
                desired = _target_grasp_pose(
                    state,
                    standoff,
                    current[:3, :3],
                    table_plane[:3],
                    grasp_orientation,
                    centered_grasp=centered_grasp,
                )
                if phase_name == "approach":
                    approach_error = phase_controller.pose_error(current, desired)
                    slow_approach = (
                        approach_error.position_norm_m <= SERVO_APPROACH_SLOW_ZONE_M
                    )
                    phase_controller.max_linear_speed_m_s = (
                        SERVO_APPROACH_SLOW_LINEAR_SPEED_M_S
                        if slow_approach
                        else SERVO_APPROACH_MAX_LINEAR_SPEED_M_S
                    )
                    phase_controller.max_translation_step_m = (
                        SERVO_APPROACH_FINAL_TRANSLATION_STEP_M
                        if approach_error.position_norm_m <= SERVO_APPROACH_FREEZE_ZONE_M
                        else (
                            SERVO_APPROACH_SLOW_TRANSLATION_STEP_M
                            if slow_approach
                            else SERVO_APPROACH_MAX_TRANSLATION_STEP_M
                        )
                    )
                    if approach_error.position_norm_m <= SERVO_APPROACH_FREEZE_ZONE_M:
                        phase_controller.max_linear_speed_m_s = SERVO_APPROACH_FINAL_LINEAR_SPEED_M_S
                command = phase_controller.command(current, desired)
                error = command.error
                final_error = error
                if phase_name == "approach":
                    approach_error_history.append(float(error.position_norm_m))
                    approach_center_history.append(state.center_base_m.copy())
                    if len(approach_error_history) > max(10, 2 * SERVO_FINAL_WINDOW_SIZE):
                        del approach_error_history[:-max(10, 2 * SERVO_FINAL_WINDOW_SIZE)]
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

                if phase_name == "approach":
                    window_converged = _approach_window_converged(
                        approach_error_history,
                        approach_center_history,
                    )
                    converged_now = (
                        window_converged
                        and error.rotation_norm_rad <= orientation_tol
                    )
                else:
                    converged_now = phase_controller.converged(
                        error, position_tol, orientation_tol
                    )
                if converged_now:
                    stable_count += 1
                    if stable_count >= SERVO_STABLE_FRAMES:
                        break
                    # Do not chase sub-millimetre observation changes while
                    # confirming convergence over consecutive RGB-D frames.
                    time.sleep(dt)
                    continue
                else:
                    stable_count = 0
                next_pose = phase_controller.next_pose(current, command, dt)
                if _table_signed_height(next_pose[:3, 3], table_plane) < SERVO_MIN_TCP_TABLE_CLEARANCE_M:
                    raise RuntimeError("PBVS command would violate table clearance")
                current = ik.apply(next_pose)
                time.sleep(dt)
            else:
                if phase_name == "approach" and _approach_window_converged(
                    approach_error_history,
                    approach_center_history,
                    relaxed=True,
                ):
                    metrics = _approach_window_metrics(
                        approach_error_history,
                        approach_center_history,
                    )
                    print(
                        "Visual servo approach accepted with depth-noise tolerance: "
                        f"median={metrics[0] * 1000.0:.2f} mm, "
                        f"max={metrics[1] * 1000.0:.2f} mm, "
                        f"center_std={metrics[2] * 1000.0:.2f} mm"
                    )
                    stable_count = SERVO_STABLE_FRAMES
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
            jaw_centering = _center_gripper_before_close(
                ik,
                state,
                table_plane,
                grasp_orientation,
            )
            print(
                "RG2 bilateral centering: "
                f"section={jaw_centering['section_width_mm']:.1f} mm, "
                f"left_gap={jaw_centering['left_clearance_mm']:.1f} mm, "
                f"right_gap={jaw_centering['right_clearance_mm']:.1f} mm"
            )
            _validate_grasp_observation(state, last_observation)
            restored = _prepare_physics_grasp(
                sim,
                robot_base,
                target_state=state,
                force_dynamic=bool(dynamics_control),
            )
            if restored:
                print(f"Physics grasp preparation: {restored} generated bodies dynamic/respondable")
            if dynamics_control:
                target_dynamics = _assert_dynamic_target(sim, robot_base, state)
                print(
                    "Physical target validated: "
                    f"{target_dynamics['target_alias']} "
                    f"dynamic={target_dynamics['dynamically_enabled']}"
                )
            print("Closing RG2 gripper...")
            if USE_CONNECTOR and CONNECTOR_KINEMATIC_ONLY and not dynamics_control:
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
                if dynamics_control:
                    before_close, after_close = _run_dynamic_rg2_motion(
                        client,
                        sim,
                        GRIPPER_CLOSE_VALUE,
                        arm_controller=ik.torque_controller,
                    )
                else:
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
            if USE_CONNECTOR and not dynamics_control:
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
                kinematic_only=bool(USE_CONNECTOR and CONNECTOR_KINEMATIC_ONLY and not dynamics_control),
            )
            print(f"Grasp lift verification: {'PASS' if grasp_verified else 'NOT_CONFIRMED'}")
            if not grasp_verified:
                raise RuntimeError(
                    "Grasp execution completed, but RGB-D lift verification failed"
                )
            if place_in_box:
                if connector_attachment is None:
                    raise RuntimeError("Cannot place object without a connector attachment")
                placement_result = _place_attached_target_in_box(
                    sim,
                    ik,
                    state,
                    table_plane,
                    connector_attachment,
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
            "placement": placement_result,
            "gripper_close_joint_motion": gripper_close_motion,
            "jaw_centering": jaw_centering,
            "final_position_error_mm": None if final_error is None else final_error.position_norm_m * 1000.0,
            "final_rotation_error_deg": None if final_error is None else math.degrees(final_error.rotation_norm_rad),
            "uses_color_features": False,
            "controller": (
                "rgbd_pbvs_incremental_ik_dynamic_torque"
                if dynamics_control and dynamics_mode == "torque"
                else "rgbd_pbvs_incremental_ik_dynamic_position"
                if dynamics_control
                else "rgbd_pbvs_incremental_ik"
            ),
            "grasp_orientation": str(grasp_orientation),
            "target_policy": str(target_policy),
            "selection_seed": selection_seed,
            "max_commanded_tilt_deg": GRASP_MAX_TILT_DEG,
            "control_mode": "closed_loop",
            "dynamics_control": bool(dynamics_control),
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
            "placement": placement_result,
            "jaw_centering": jaw_centering,
            "uses_color_features": False,
            "controller": (
                "rgbd_pbvs_incremental_ik_dynamic_torque"
                if dynamics_control and dynamics_mode == "torque"
                else "rgbd_pbvs_incremental_ik_dynamic_position"
                if dynamics_control
                else "rgbd_pbvs_incremental_ik"
            ),
            "grasp_orientation": str(grasp_orientation),
            "target_policy": str(target_policy),
            "selection_seed": selection_seed,
            "control_mode": "closed_loop",
            "dynamics_control": bool(dynamics_control),
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
    parser.add_argument(
        "--dynamics",
        action="store_true",
        default=DYNAMICS_CONTROL,
        help=(
            "use dynamic iiwa joints with synchronous joint-space torque control; "
            "connector attachment is disabled automatically"
        ),
    )
    parser.add_argument(
        "--dynamics-mode",
        choices=("position", "torque"),
        default=DYNAMICS_MODE,
        help="dynamic joint actuator mode; position is the safer commissioning milestone",
    )
    parser.add_argument("--max-iterations", type=int, default=SERVO_MAX_ITERATIONS)
    parser.add_argument(
        "--target-policy",
        choices=("auto", "upper-random"),
        default=os.environ.get("ROBOT_GRASP_SERVO_TARGET_POLICY", "upper-random"),
        help="automatic target selection policy when --target-id is omitted",
    )
    parser.add_argument("--selection-seed", type=int)
    parser.add_argument(
        "--place-in-box",
        action="store_true",
        help="after lift verification, transfer and release the object into an external box",
    )
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
        target_policy=args.target_policy,
        selection_seed=args.selection_seed,
        place_in_box=bool(args.place_in_box),
        initial_translation_m=None
        if args.initial_offset_mm is None
        else np.asarray(args.initial_offset_mm, dtype=np.float64) / 1000.0,
        initial_euler_deg=None
        if args.initial_euler_deg is None
        else np.asarray(args.initial_euler_deg, dtype=np.float64),
        open_loop_only=bool(args.open_loop),
        align_only=bool(args.align_only),
        dynamics_control=bool(args.dynamics),
        dynamics_mode=str(args.dynamics_mode),
    )


if __name__ == "__main__":
    main()
