"""Depth-only local primitive tracking for eye-in-hand visual servoing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import open3d as o3d

from point_cloud import (
    capture_rgbd,
    depth_to_camera_point_cloud,
    get_camera_parameters,
    transform_points,
)


MIN_TRACK_POINTS = 35
TRACK_VOXEL_SIZE_M = 0.002
TRACK_DBSCAN_EPS_M = 0.008
TRACK_DBSCAN_MIN_POINTS = 5
TRACK_PADDING_M = 0.015
TRACK_VERTICAL_MARGIN_M = 0.007
TRACK_ANCHOR_VERTICAL_DRIFT_M = 0.004


def matrix34_to_44(matrix: list[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float64)
    if array.shape == (3, 4):
        result = np.eye(4, dtype=np.float64)
        result[:3, :4] = array
        return result
    if array.size == 12:
        result = np.eye(4, dtype=np.float64)
        result[:3, :4] = array.reshape(3, 4)
        return result
    if array.shape == (4, 4):
        return array.copy()
    raise ValueError(f"Expected a 3x4 or 4x4 matrix, received {array.shape}")


def dimensions_from_prediction(item: dict[str, Any]) -> np.ndarray:
    geometry = dict(item.get("geometry", {}))
    class_name = str(item.get("class", "")).lower()
    if class_name == "sphere":
        diameter = float(geometry.get("diameter_m", 0.045))
        return np.asarray([diameter, diameter, diameter], dtype=np.float64)
    if class_name == "spheroid":
        return np.asarray(
            [
                float(geometry.get("axis_x_m", 0.050)),
                float(geometry.get("axis_y_m", 0.040)),
                float(geometry.get("axis_z_m", 0.040)),
            ],
            dtype=np.float64,
        )
    if class_name == "cylinder":
        diameter = float(geometry.get("diameter_m", 0.040))
        return np.asarray(
            [diameter, diameter, float(geometry.get("height_m", 0.060))],
            dtype=np.float64,
        )
    if class_name == "cone":
        diameter = float(geometry.get("base_diameter_m", 0.045))
        return np.asarray(
            [diameter, diameter, float(geometry.get("height_m", 0.060))],
            dtype=np.float64,
        )
    return np.asarray(
        [
            float(geometry.get("length_m", 0.040)),
            float(geometry.get("width_m", 0.040)),
            float(geometry.get("height_m", 0.040)),
        ],
        dtype=np.float64,
    )


@dataclass
class TargetTrackState:
    object_id: int
    class_name: str
    center_base_m: np.ndarray
    rotation_base: np.ndarray
    dimensions_m: np.ndarray
    confidence: float
    anchor_center_base_m: np.ndarray | None = None
    anchor_rotation_base: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.anchor_center_base_m is None:
            self.anchor_center_base_m = self.center_base_m.copy()
        if self.anchor_rotation_base is None:
            self.anchor_rotation_base = self.rotation_base.copy()


@dataclass(frozen=True)
class TargetObservation:
    center_base_m: np.ndarray
    rotation_base: np.ndarray
    point_count: int
    confidence: float
    image_pixel: tuple[float, float] | None
    image_margin_px: float
    extent_m: np.ndarray


def select_servo_target(
    recognition: dict[str, Any],
    requested_id: int | None = None,
) -> dict[str, Any]:
    objects = [
        item
        for item in recognition.get("objects", [])
        if bool(item.get("pose_valid", True))
    ]
    if requested_id is not None:
        for item in objects:
            if int(item.get("id", -1)) == int(requested_id):
                return item
        raise RuntimeError(f"Prediction Object {requested_id} does not exist")
    if not objects:
        raise RuntimeError("Recognition output contains no pose-valid object")

    def score(item: dict[str, Any]) -> float:
        planning = dict(item.get("grasp_planning", {}))
        warnings = list(item.get("quality_warnings", []))
        return (
            1.5 * float(bool(planning.get("topmost", False)))
            - 2.0 * float(bool(planning.get("grasp_blocked", False)))
            + float(planning.get("priority_score", 0.0))
            + float(item.get("confidence", 0.0))
            - 0.12 * len(warnings)
        )

    return max(objects, key=score)


def track_state_from_prediction(item: dict[str, Any]) -> TargetTrackState:
    rotation = np.asarray(item.get("rotation_matrix", np.eye(3)), dtype=np.float64)
    if rotation.shape != (3, 3):
        rotation = np.eye(3, dtype=np.float64)
    return TargetTrackState(
        object_id=int(item.get("id", -1)),
        class_name=str(item.get("class", "unknown")).lower(),
        center_base_m=np.asarray(item["center_m"], dtype=np.float64),
        rotation_base=rotation,
        dimensions_m=dimensions_from_prediction(item),
        confidence=float(item.get("confidence", 0.0)),
    )


def _project_base_point(
    point_base: np.ndarray,
    camera_matrix_base: np.ndarray,
    parameters: dict[str, Any],
    width: int,
    height: int,
) -> tuple[tuple[float, float] | None, float]:
    transform = np.asarray(camera_matrix_base, dtype=np.float64)
    point_camera = (
        np.asarray(point_base, dtype=np.float64) - transform[:3, 3]
    ) @ transform[:3, :3]
    z = float(point_camera[2])
    if z <= float(parameters["near"]) or z >= float(parameters["far"]):
        return None, -1.0
    u = (1.0 - point_camera[0] / (z * math.tan(parameters["fov_x"] / 2.0))) * 0.5 * (width - 1)
    v = (1.0 - point_camera[1] / (z * math.tan(parameters["fov_y"] / 2.0))) * 0.5 * (height - 1)
    margin = min(float(u), float(v), float(width - 1 - u), float(height - 1 - v))
    return (float(u), float(v)), float(margin)


def _filter_above_table(
    points: np.ndarray,
    table_plane: np.ndarray,
    clearance_m: float = 0.002,
) -> np.ndarray:
    plane = np.asarray(table_plane, dtype=np.float64)
    normal_norm = max(float(np.linalg.norm(plane[:3])), 1e-12)
    signed = (points @ plane[:3] + plane[3]) / normal_norm
    return points[signed > float(clearance_m)]


def _downsample(points: np.ndarray) -> np.ndarray:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud = cloud.voxel_down_sample(TRACK_VOXEL_SIZE_M)
    return np.asarray(cloud.points, dtype=np.float64)


def _choose_local_component(points: np.ndarray, seed_center: np.ndarray) -> np.ndarray:
    if len(points) < MIN_TRACK_POINTS:
        return points
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    labels = np.asarray(
        cloud.cluster_dbscan(
            eps=TRACK_DBSCAN_EPS_M,
            min_points=TRACK_DBSCAN_MIN_POINTS,
            print_progress=False,
        ),
        dtype=np.int32,
    )
    valid_labels = [value for value in np.unique(labels) if value >= 0]
    if not valid_labels:
        return points
    components = [points[labels == value] for value in valid_labels]
    usable = [component for component in components if len(component) >= MIN_TRACK_POINTS]
    if not usable:
        return max(components, key=len)
    return min(
        usable,
        key=lambda component: float(
            np.linalg.norm(np.median(component, axis=0) - seed_center)
        ),
    )


def _align_axes(axes: np.ndarray, previous: np.ndarray) -> np.ndarray:
    remaining = list(range(3))
    aligned = np.zeros((3, 3), dtype=np.float64)
    for target_axis in range(3):
        choice = max(
            remaining,
            key=lambda candidate: abs(
                float(np.dot(axes[:, candidate], previous[:, target_axis]))
            ),
        )
        vector = axes[:, choice]
        if float(np.dot(vector, previous[:, target_axis])) < 0.0:
            vector = -vector
        aligned[:, target_axis] = vector
        remaining.remove(choice)
    if float(np.linalg.det(aligned)) < 0.0:
        aligned[:, 2] *= -1.0
    return aligned


def _catalog_box_shift(
    points: np.ndarray,
    anchor_center: np.ndarray,
    anchor_rotation: np.ndarray,
    dimensions_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate box translation from visible model faces, not free PCA."""

    coordinates = (points - anchor_center) @ anchor_rotation
    half = 0.5 * np.asarray(dimensions_m, dtype=np.float64)
    shifts: list[float] = []
    reliable = np.zeros(3, dtype=bool)
    for axis in range(3):
        face_margin = max(0.0025, 0.10 * float(dimensions_m[axis]))
        plus = coordinates[:, axis][coordinates[:, axis] >= half[axis] - face_margin]
        minus = coordinates[:, axis][coordinates[:, axis] <= -half[axis] + face_margin]
        candidates: list[float] = []
        if len(plus) >= 4:
            candidates.append(float(np.median(plus) - half[axis]))
        if len(minus) >= 4:
            candidates.append(float(np.median(minus) + half[axis]))
        if candidates:
            shifts.append(float(np.median(candidates)))
            reliable[axis] = True
        else:
            lower = float(np.percentile(coordinates[:, axis], 5.0))
            upper = float(np.percentile(coordinates[:, axis], 95.0))
            shifts.append(0.5 * (lower + upper))
    return np.clip(np.asarray(shifts), -0.010, 0.010), reliable


def estimate_target_from_points(
    points_base: np.ndarray,
    state: TargetTrackState,
    table_normal: np.ndarray | None = None,
) -> TargetObservation | None:
    points = np.asarray(points_base, dtype=np.float64)
    if len(points) < MIN_TRACK_POINTS:
        return None

    normal = np.asarray(
        [0.0, 0.0, 1.0] if table_normal is None else table_normal,
        dtype=np.float64,
    )
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    vertical_half_extent = 0.5 * float(
        np.sum(np.abs(state.rotation_base.T @ normal) * state.dimensions_m)
    )
    anchor_center = (
        state.center_base_m
        if state.anchor_center_base_m is None
        else state.anchor_center_base_m
    )
    center_height = float(np.dot(anchor_center, normal))
    point_heights = points @ normal
    vertical_inside = (
        (point_heights >= center_height - vertical_half_extent - TRACK_VERTICAL_MARGIN_M)
        & (point_heights <= center_height + vertical_half_extent + TRACK_VERTICAL_MARGIN_M)
    )
    points = points[vertical_inside]
    if len(points) < MIN_TRACK_POINTS:
        return None

    anchor_center = (
        state.center_base_m
        if state.anchor_center_base_m is None
        else state.anchor_center_base_m
    )
    anchor_rotation = (
        state.rotation_base
        if state.anchor_rotation_base is None
        else state.anchor_rotation_base
    )
    reference_center = state.center_base_m
    reference_rotation = state.rotation_base
    local = (points - reference_center) @ reference_rotation
    half = 0.5 * state.dimensions_m + TRACK_PADDING_M
    inside = np.all(np.abs(local) <= half, axis=1)
    points = points[inside]
    if len(points) < MIN_TRACK_POINTS:
        radius = 0.75 * float(np.max(state.dimensions_m)) + TRACK_PADDING_M
        points = np.asarray(points_base)[
            np.linalg.norm(np.asarray(points_base) - state.center_base_m, axis=1)
            <= radius
        ]
    if len(points) < MIN_TRACK_POINTS:
        return None

    points = _downsample(points)
    points = _choose_local_component(points, state.center_base_m)
    if len(points) < MIN_TRACK_POINTS:
        return None

    mean = np.mean(points, axis=0)
    covariance = np.cov(points - mean, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axes = eigenvectors[:, np.argsort(eigenvalues)[::-1]]
    axes = _align_axes(axes, state.rotation_base)

    coordinates = (points - anchor_center) @ anchor_rotation
    lower = np.percentile(coordinates, 2.0, axis=0)
    upper = np.percentile(coordinates, 98.0, axis=0)
    extent = np.maximum(upper - lower, 1e-6)
    shift_local = 0.5 * (lower + upper)
    coverage = np.clip(extent / np.maximum(state.dimensions_m, 1e-6), 0.0, 1.5)

    if state.class_name in {"cube", "cuboid"}:
        catalog_shift, reliable_faces = _catalog_box_shift(
            points,
            anchor_center,
            anchor_rotation,
            state.dimensions_m,
        )
        shift_local = np.where(reliable_faces, catalog_shift, shift_local)
        reference_center = anchor_center
        reference_rotation = anchor_rotation

    # A single visible face must not drag the estimated object centre onto the
    # surface. Update only axes with enough observed geometric span.
    reliable = coverage >= 0.42
    shift_local = np.where(reliable, shift_local, 0.0)
    shift_local = np.clip(shift_local, -0.012, 0.012)
    measured_center = reference_center + shift_local @ reference_rotation.T

    # The wrist camera normally sees the target's upper envelope. Recover the
    # model centre from that envelope and the catalog vertical half-extent.
    # This suppresses upward drift when gripper geometry approaches the ROI.
    observed_top = float(np.percentile(points @ normal, 98.0))
    model_center_height = observed_top - vertical_half_extent
    anchor_height = float(np.dot(anchor_center, normal))
    model_center_height = anchor_height + float(
        np.clip(
            model_center_height - anchor_height,
            -TRACK_ANCHOR_VERTICAL_DRIFT_M,
            TRACK_ANCHOR_VERTICAL_DRIFT_M,
        )
    )
    height_correction = float(
        np.clip(
            model_center_height - float(np.dot(measured_center, normal)),
            -0.006,
            0.006,
        )
    )
    if state.class_name not in {"cube", "cuboid"}:
        measured_center = measured_center + normal * height_correction

    if state.class_name in {"sphere", "cylinder", "cone", "cube", "cuboid"}:
        measured_rotation = reference_rotation.copy()
    else:
        measured_rotation = axes

    dimension_error = float(
        np.mean(
            np.abs(np.minimum(extent, state.dimensions_m) - state.dimensions_m)
            / np.maximum(state.dimensions_m, 1e-6)
        )
    )
    point_score = min(1.0, len(points) / 220.0)
    coverage_score = float(np.mean(np.clip(coverage, 0.0, 1.0)))
    confidence = float(
        np.clip(
            0.45 * point_score
            + 0.40 * coverage_score
            + 0.15 * math.exp(-2.5 * dimension_error),
            0.0,
            1.0,
        )
    )
    return TargetObservation(
        center_base_m=measured_center,
        rotation_base=measured_rotation,
        point_count=int(len(points)),
        confidence=confidence,
        image_pixel=None,
        image_margin_px=-1.0,
        extent_m=extent,
    )


def capture_target_observation(
    sim: Any,
    camera: int,
    robot_base: int,
    state: TargetTrackState,
    table_plane: np.ndarray,
) -> TargetObservation | None:
    rgb, depth, width, height = capture_rgbd(sim, camera)
    parameters = get_camera_parameters(sim, camera, width, height)
    points_camera, _, _ = depth_to_camera_point_cloud(depth, rgb, parameters)
    camera_matrix = matrix34_to_44(sim.getObjectMatrix(camera, robot_base))
    points_base = transform_points(points_camera, camera_matrix[:3, :4])
    points_base = _filter_above_table(points_base, table_plane)
    pixel, margin = _project_base_point(
        state.center_base_m,
        camera_matrix,
        parameters,
        width,
        height,
    )
    if pixel is None or margin < 8.0:
        return None
    observation = estimate_target_from_points(points_base, state, table_plane[:3])
    if observation is None:
        return None
    return TargetObservation(
        center_base_m=observation.center_base_m,
        rotation_base=observation.rotation_base,
        point_count=observation.point_count,
        confidence=observation.confidence,
        image_pixel=pixel,
        image_margin_px=margin,
        extent_m=observation.extent_m,
    )


def update_track_state(
    state: TargetTrackState,
    observation: TargetObservation,
    center_alpha: float = 0.35,
    max_center_update_m: float = 0.003,
) -> TargetTrackState:
    alpha = float(np.clip(center_alpha, 0.0, 1.0))
    innovation = observation.center_base_m - state.center_base_m
    innovation_norm = float(np.linalg.norm(innovation))
    if innovation_norm > float(max_center_update_m) and innovation_norm > 1e-12:
        innovation *= float(max_center_update_m) / innovation_norm
    return TargetTrackState(
        object_id=state.object_id,
        class_name=state.class_name,
        center_base_m=state.center_base_m + alpha * innovation,
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
