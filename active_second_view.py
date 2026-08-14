"""Acquire and fuse a second eye-in-hand RGB-D view.

This is an active-perception stage, not a visual-servo controller.  The first
view supplies a coarse target center; the wrist camera is moved to a safe
oblique observation pose with position-constrained IK, then both clouds are
expressed in the Robot Base frame and re-segmented by geometry only.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
from remote_session import RemoteAPIClient
from scipy.spatial.transform import Rotation, Slerp

from geometric_segmentation import refine_geometric_clusters
from gripper_tcp_approach import get_ik_handle, get_joint_limits
from point_cloud import (
    capture_rgbd,
    create_open3d_cloud,
    depth_to_camera_point_cloud,
    find_unique_object_by_alias,
    get_camera_parameters,
    get_kuka_joints_from_tip,
    transform_points,
)
from segment_multiple_objects import (
    MIN_CLUSTER_POINTS,
    analyze_clusters,
    clear_old_clusters,
    extract_objects_above_table,
    remove_outliers,
    save_clusters,
    run_dbscan,
    voxel_downsample,
)


ROOT = Path(__file__).resolve().parent
SEGMENTATION_DIR = ROOT / "segmentation_output"
RECOGNITION_JSON = ROOT / "recognition_output" / "recognition_results.json"
SCENE_GT_FILE = ROOT / "random_scene_ground_truth.json"
SEGMENTATION_METADATA_FILE = SEGMENTATION_DIR / "segmentation_metadata.json"
VIEW_01_FILE = SEGMENTATION_DIR / "view_01_object_cloud.ply"
VIEW_02_FILE = SEGMENTATION_DIR / "view_02_object_cloud.ply"
FUSED_FILE = SEGMENTATION_DIR / "fused_object_cloud.ply"
SECOND_VIEW_AZIMUTH_DEG = float(os.environ.get("ROBOT_GRASP_SECOND_VIEW_AZIMUTH_DEG", "32"))
SECOND_VIEW_HEIGHT_M = float(os.environ.get("ROBOT_GRASP_SECOND_VIEW_HEIGHT_M", "0.27"))
SECOND_VIEW_RADIUS_M = float(os.environ.get("ROBOT_GRASP_SECOND_VIEW_RADIUS_M", "0.11"))
SECOND_VIEW_STEPS = max(10, int(os.environ.get("ROBOT_GRASP_SECOND_VIEW_STEPS", "70")))
SECOND_VIEW_STEP_DELAY_S = float(os.environ.get("ROBOT_GRASP_SECOND_VIEW_STEP_DELAY_S", "0.015"))
SECOND_VIEW_MIN_TABLE_CLEARANCE_M = float(
    os.environ.get("ROBOT_GRASP_SECOND_VIEW_MIN_TABLE_CLEARANCE_M", "0.18")
)
FUSION_VOXEL_SIZE_M = float(os.environ.get("ROBOT_GRASP_FUSION_VOXEL_SIZE_M", "0.0025"))
SHOW_VISUALIZATION = os.environ.get("ROBOT_GRASP_HEADLESS") != "1"


def _matrix34_to_44(matrix: Any) -> np.ndarray:
    raw = np.asarray(matrix, dtype=np.float64).reshape(3, 4)
    result = np.eye(4, dtype=np.float64)
    result[:3, :4] = raw
    return result


def _pose_from_matrix(matrix: np.ndarray) -> list[float]:
    quaternion = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return [
        *matrix[:3, 3].tolist(),
        *quaternion.astype(float).tolist(),
    ]


def _matrix_from_pose(pose: list[float]) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_quat(np.asarray(pose[3:7], dtype=np.float64)).as_matrix()
    matrix[:3, 3] = np.asarray(pose[:3], dtype=np.float64)
    return matrix


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def _second_camera_pose(
    current_matrix: np.ndarray,
    target_center: np.ndarray,
    table_normal: np.ndarray,
) -> np.ndarray:
    """Choose an oblique view with a real baseline relative to the first view."""

    table_normal = _normalize(table_normal)
    current_position = current_matrix[:3, 3]
    relative = current_position - target_center
    height = float(np.dot(relative, table_normal))
    horizontal = relative - height * table_normal

    if float(np.linalg.norm(horizontal)) < 0.045:
        reference = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        reference -= float(np.dot(reference, table_normal)) * table_normal
        horizontal = _normalize(reference) * SECOND_VIEW_RADIUS_M
    else:
        horizontal = _normalize(horizontal) * SECOND_VIEW_RADIUS_M

    azimuth = math.radians(SECOND_VIEW_AZIMUTH_DEG)
    rotation_about_table = Rotation.from_rotvec(table_normal * azimuth).as_matrix()
    horizontal = rotation_about_table @ horizontal
    desired_position = target_center + horizontal + table_normal * max(
        SECOND_VIEW_HEIGHT_M,
        0.20,
    )

    desired_forward = _normalize(target_center - desired_position)
    current_forward = _normalize(current_matrix[:3, 2])
    alignment, _ = Rotation.align_vectors(
        np.asarray([desired_forward]),
        np.asarray([current_forward]),
    )
    desired_rotation = alignment.as_matrix() @ current_matrix[:3, :3]

    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = desired_rotation
    result[:3, 3] = desired_position
    return result


def _interpolate_pose(first: np.ndarray, second: np.ndarray, ratio: float) -> list[float]:
    first_rotation = Rotation.from_matrix(first[:3, :3])
    second_rotation = Rotation.from_matrix(second[:3, :3])
    slerp = Slerp([0.0, 1.0], Rotation.concatenate([first_rotation, second_rotation]))
    rotation = slerp([float(ratio)])[0].as_matrix()
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = (1.0 - ratio) * first[:3, 3] + ratio * second[:3, 3]
    return _pose_from_matrix(matrix)


def _table_signed_height(position: np.ndarray, table_plane: np.ndarray) -> float:
    normal = _normalize(table_plane[:3])
    return float(np.dot(position, normal) + table_plane[3] / max(float(np.linalg.norm(table_plane[:3])), 1e-12))


def _move_camera_to_second_view(
    sim: Any,
    sim_ik: Any,
    camera: int,
    gripper_tip: int,
    robot_base: int,
    target_matrix: np.ndarray,
    table_plane: np.ndarray,
) -> None:
    if sim.getSimulationState() not in {sim.simulation_stopped, sim.simulation_paused}:
        raise RuntimeError("Second-view capture requires a stopped or paused simulation")

    joints = get_kuka_joints_from_tip(sim, gripper_tip)
    if len(joints) != 7:
        raise RuntimeError(f"Expected 7 arm joints, found {len(joints)}")
    limits = get_joint_limits(sim, joints)
    original_positions = [float(sim.getJointPosition(joint)) for joint in joints]
    original_modes = []
    for joint in joints:
        raw_mode = sim.getJointMode(joint)
        original_modes.append(
            int(raw_mode[0] if isinstance(raw_mode, (tuple, list)) else raw_mode)
        )
    target_dummy = int(sim.createDummy(0.015))
    ik_environment = None

    try:
        current_matrix = _matrix34_to_44(sim.getObjectMatrix(camera, robot_base))
        sim.setObjectPose(target_dummy, _pose_from_matrix(current_matrix), robot_base)
        ik_environment = int(sim_ik.createEnvironment())
        ik_group = int(sim_ik.createGroup(ik_environment))
        ik_element, sim_to_ik_map, _ = sim_ik.addElementFromScene(
            ik_environment,
            ik_group,
            robot_base,
            camera,
            target_dummy,
            sim_ik.constraint_position | sim_ik.constraint_alpha_beta,
        )
        ik_joints = [get_ik_handle(sim_to_ik_map, joint) for joint in joints]
        sim_ik.setGroupCalculation(
            ik_environment,
            ik_group,
            sim_ik.method_damped_least_squares,
            0.08,
            160,
        )
        try:
            flags = sim_ik.getGroupFlags(ik_environment, ik_group)
            flags |= (
                sim_ik.group_avoidlimits
                | sim_ik.group_restoreonbadlintol
                | sim_ik.group_restoreonbadangtol
            )
            sim_ik.setGroupFlags(ik_environment, ik_group, flags)
        except Exception:
            pass
        sim_ik.setElementPrecision(
            ik_environment,
            ik_group,
            ik_element,
            [0.0015, math.radians(2.0)],
        )
        for ik_joint, (lower, upper) in zip(ik_joints, limits):
            sim_ik.setJointInterval(
                ik_environment,
                ik_joint,
                False,
                [lower, upper - lower],
            )
            sim_ik.setJointLimitMargin(
                ik_environment,
                ik_joint,
                math.radians(8.0),
            )
            sim_ik.setJointMaxStepSize(
                ik_environment,
                ik_joint,
                math.radians(8.0),
            )
        sim_ik.syncFromSim(ik_environment, [ik_group])

        completed_step = 0
        for step in range(1, SECOND_VIEW_STEPS + 1):
            ratio = step / SECOND_VIEW_STEPS
            pose = _interpolate_pose(current_matrix, target_matrix, ratio)
            candidate_position = np.asarray(pose[:3], dtype=np.float64)
            if _table_signed_height(candidate_position, table_plane) < SECOND_VIEW_MIN_TABLE_CLEARANCE_M:
                raise RuntimeError("Second-view path would move the camera too close to the table")
            sim.setObjectPose(target_dummy, pose, robot_base)
            result, flags, precision = sim_ik.handleGroup(
                ik_environment,
                ik_group,
                {"syncWorlds": True},
            )
            if result != sim_ik.result_success:
                position_precision = float(precision[0]) if len(precision) else float("inf")
                orientation_precision = float(precision[1]) if len(precision) > 1 else float("inf")
                if position_precision <= 0.005 and orientation_precision <= math.radians(5.0):
                    print(
                        "Second-view IK reached an acceptable near solution: "
                        f"position={position_precision * 1000:.2f} mm, "
                        f"axis={math.degrees(orientation_precision):.2f} deg"
                    )
                    completed_step = step
                    break
                raise RuntimeError(
                    f"Second-view IK failed at step {step}/{SECOND_VIEW_STEPS}, "
                    f"flags={flags}, precision={precision}"
                )
            completed_step = step
            time.sleep(SECOND_VIEW_STEP_DELAY_S)
        print(f"Second-view camera motion completed at step {completed_step}/{SECOND_VIEW_STEPS}")
    except Exception:
        for joint, position in zip(joints, original_positions):
            sim.setJointPosition(joint, position)
        raise
    finally:
        for joint, mode in zip(joints, original_modes):
            try:
                sim.setJointMode(joint, mode)
            except Exception:
                pass
        # The view motion is kinematic, but dynamic proxy links can retain a
        # stale solver velocity from an interrupted previous run.  Clear only
        # the robot descendant bodies while the scene is paused; workpieces
        # and their settled contacts are left untouched.
        set_velocity = getattr(sim, "setObjectVelocity", None)
        if set_velocity is not None:
            try:
                robot_shapes = sim.getObjectsInTree(robot_base, sim.sceneobject_shape, 0)
                for handle in robot_shapes:
                    try:
                        set_velocity(int(handle), [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
                    except Exception:
                        pass
            except Exception:
                pass
        if ik_environment is not None:
            sim_ik.eraseEnvironment(ik_environment)
        try:
            sim.removeObject(target_dummy)
        except Exception:
            pass


def _capture_base_object_cloud(sim: Any, camera: int, robot_base: int) -> tuple[o3d.geometry.PointCloud, dict, np.ndarray]:
    rgb, depth, width, height = capture_rgbd(sim, camera)
    parameters = get_camera_parameters(sim, camera, width, height)
    points_camera, colors, _ = depth_to_camera_point_cloud(depth, rgb, parameters)
    matrix = _matrix34_to_44(sim.getObjectMatrix(camera, robot_base))
    points_base = transform_points(points_camera, matrix[:3, :4])
    raw_cloud = create_open3d_cloud(points_base, colors)
    cloud = voxel_downsample(raw_cloud, voxel_size=FUSION_VOXEL_SIZE_M)
    cloud = remove_outliers(cloud)
    return cloud, parameters, matrix


def _merge_clouds(first: o3d.geometry.PointCloud, second: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    points = np.vstack([np.asarray(first.points), np.asarray(second.points)])
    colors_first = np.asarray(first.colors)
    colors_second = np.asarray(second.colors)
    result = o3d.geometry.PointCloud()
    result.points = o3d.utility.Vector3dVector(points)
    if len(colors_first) == len(first.points) and len(colors_second) == len(second.points):
        result.colors = o3d.utility.Vector3dVector(np.vstack([colors_first, colors_second]))
    return result.voxel_down_sample(FUSION_VOXEL_SIZE_M)


def _load_context() -> tuple[dict, dict, dict]:
    scene = json.loads(SCENE_GT_FILE.read_text(encoding="utf-8"))
    segmentation = json.loads(SEGMENTATION_METADATA_FILE.read_text(encoding="utf-8"))
    recognition = json.loads(RECOGNITION_JSON.read_text(encoding="utf-8"))
    if scene.get("scene_id") != segmentation.get("scene_id"):
        raise RuntimeError("Scene and segmentation metadata do not match")
    return scene, segmentation, recognition


def _select_target(recognition: dict) -> np.ndarray:
    objects = list(recognition.get("objects", []))
    if not objects:
        raise RuntimeError("Coarse recognition contains no target candidates")

    def risk_score(item: dict) -> float:
        features = dict(item.get("features", {}))
        warnings = set(str(value) for value in item.get("quality_warnings", []))
        extent = np.asarray(features.get("obb_extent_m", [0.0, 0.0, 0.0]), dtype=float)
        oversized = max(0.0, float(np.max(extent)) / 0.0875 - 1.0)
        ambiguity = (
            1.0 - float(item.get("confidence", 0.0))
            + (0.55 if "small_topk_margin" in warnings else 0.0)
            + (0.45 if "ambiguous_primitive_fit" in warnings else 0.0)
            + (0.80 if oversized > 0.0 else 0.0)
        )
        # A large, low-confidence cluster is a likely contact merge.  A
        # topmost object remains a useful target only when its geometry is
        # already unambiguous.
        point_mass = min(float(item.get("point_count", 0)) / 900.0, 1.0)
        planning = float(item.get("grasp_planning", {}).get("priority_score", 0.0))
        center = np.asarray(item.get("center_m", [0.0, 0.0, 0.0]), dtype=float)
        elevation = max(0.0, (float(center[2]) - 0.040) / 0.040)
        return (
            2.0 * oversized
            + 1.35 * ambiguity
            + 0.35 * point_mass
            + 0.55 * min(elevation, 1.5)
            + 0.10 * planning
        )

    selected = max(objects, key=risk_score)
    print(
        f"Second-view target: Object {selected.get('id')} "
        f"class={selected.get('class')} center={np.round(selected['center_m'], 4)} "
        f"risk={risk_score(selected):.2f}"
    )
    return np.asarray(selected["center_m"], dtype=np.float64)


def main() -> None:
    scene, segmentation, recognition = _load_context()
    if not VIEW_01_FILE.exists():
        raise RuntimeError(f"Missing first-view cloud: {VIEW_01_FILE}")

    table_plane = np.asarray(segmentation["table_plane"], dtype=np.float64)
    target_center = _select_target(recognition)

    client = RemoteAPIClient()
    sim = client.require("sim")
    sim_ik = client.require("simIK")
    camera = find_unique_object_by_alias(sim, sim.sceneobject_visionsensor, "rgbd_camera")
    gripper_tip = find_unique_object_by_alias(sim, sim.sceneobject_dummy, "gripper_tip")
    joints = get_kuka_joints_from_tip(sim, gripper_tip)
    robot_base = int(sim.getObjectParent(joints[0]))
    current_matrix = _matrix34_to_44(sim.getObjectMatrix(camera, robot_base))
    target_matrix = _second_camera_pose(
        current_matrix,
        target_center,
        table_plane[:3],
    )
    _move_camera_to_second_view(
        sim,
        sim_ik,
        camera,
        gripper_tip,
        robot_base,
        target_matrix,
        table_plane,
    )

    second_cloud, parameters, second_matrix = _capture_base_object_cloud(sim, camera, robot_base)
    second_cloud, _, _ = extract_objects_above_table(second_cloud, table_plane)
    if len(second_cloud.points) < 100:
        raise RuntimeError("Second view contains too few object points")
    o3d.io.write_point_cloud(str(VIEW_02_FILE), second_cloud)

    first_cloud = o3d.io.read_point_cloud(str(VIEW_01_FILE))
    fused = _merge_clouds(first_cloud, second_cloud)
    fused = remove_outliers(fused)
    if len(fused.points) < 100:
        raise RuntimeError("Fused object cloud contains too few points")
    o3d.io.write_point_cloud(str(FUSED_FILE), fused)

    clear_old_clusters()
    _, initial_clusters = run_dbscan(
        fused,
        expected_count=None,
        enable_layer_split=str(scene.get("scene_mode", "")).lower()
        in {"planned", "level4"},
        min_cluster_points=MIN_CLUSTER_POINTS,
    )
    clusters, diagnostics = refine_geometric_clusters(
        initial_clusters,
        min_cluster_points=MIN_CLUSTER_POINTS,
    )
    results = analyze_clusters(clusters)

    save_clusters(
        clusters,
        results,
        scene,
        table_plane,
        segmentation_method="multiview_dbscan_normal_curvature_primitive",
        view_metadata={
            "view_count": 2,
            "view_01": segmentation.get("views", {}),
            "view_02": {
                "camera_matrix_base": second_matrix.tolist(),
                "camera_intrinsics": {
                    "fx": float(parameters["fx"]),
                    "fy": float(parameters["fy"]),
                    "cx": float(parameters["cx"]),
                    "cy": float(parameters["cy"]),
                    "K": np.asarray(parameters["K"], dtype=float).tolist(),
                },
                "object_cloud_file": VIEW_02_FILE.name,
            },
            "fused_object_cloud_file": FUSED_FILE.name,
            "target_center_initial_m": target_center.tolist(),
        },
        geometric_diagnostics=diagnostics,
    )

    coarse_copy = RECOGNITION_JSON.with_name("coarse_recognition_results.json")
    shutil.copy2(RECOGNITION_JSON, coarse_copy)
    print(f"Two-view fused cloud saved: {FUSED_FILE.resolve()}")
    print(f"Coarse recognition saved: {coarse_copy.resolve()}")

    if SHOW_VISUALIZATION:
        o3d.visualization.draw_geometries(
            [fused],
            window_name="Geometry-only Multi-view Object Cloud",
            width=1200,
            height=800,
        )


if __name__ == "__main__":
    main()
