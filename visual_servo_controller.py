"""Geometry-only position-based visual servo control primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


SYMMETRY_STEPS = {
    "cube": 4,
    "cuboid": 2,
    "spheroid": 2,
}


def normalize(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm < 1e-12:
        raise ValueError("Cannot normalize a zero-length vector")
    return value / norm


def matrix_from_pose(pose: list[float] | tuple[float, ...]) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_quat(
        np.asarray(pose[3:7], dtype=np.float64)
    ).as_matrix()
    matrix[:3, 3] = np.asarray(pose[:3], dtype=np.float64)
    return matrix


def pose_from_matrix(matrix: np.ndarray) -> list[float]:
    quaternion = Rotation.from_matrix(
        np.asarray(matrix, dtype=np.float64)[:3, :3]
    ).as_quat()
    return [
        *np.asarray(matrix, dtype=np.float64)[:3, 3].tolist(),
        *quaternion.tolist(),
    ]


def _top_down_frame(
    x_hint: np.ndarray,
    table_normal: np.ndarray,
) -> np.ndarray:
    normal = normalize(table_normal)
    z_axis = -normal
    x_axis = np.asarray(x_hint, dtype=np.float64)
    x_axis = x_axis - float(np.dot(x_axis, z_axis)) * z_axis
    if float(np.linalg.norm(x_axis)) < 1e-8:
        x_axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        x_axis -= float(np.dot(x_axis, z_axis)) * z_axis
    x_axis = normalize(x_axis)
    y_axis = normalize(np.cross(z_axis, x_axis))
    x_axis = normalize(np.cross(y_axis, z_axis))
    return np.column_stack([x_axis, y_axis, z_axis])


def _rotation_distance(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(first) @ np.asarray(second).T
    return float(Rotation.from_matrix(relative).magnitude())


def _choose_grasp_rotation(
    class_name: str,
    object_rotation: np.ndarray,
    current_tip_rotation: np.ndarray,
    table_normal: np.ndarray,
) -> np.ndarray:
    class_name = str(class_name).lower()
    current_x = np.asarray(current_tip_rotation, dtype=np.float64)[:, 0]

    if class_name in {"sphere", "cylinder", "cone"}:
        return _top_down_frame(current_x, table_normal)

    base_x = np.asarray(object_rotation, dtype=np.float64)[:, 0]
    steps = SYMMETRY_STEPS.get(class_name, 1)
    candidates: list[np.ndarray] = []
    for index in range(steps):
        angle = 2.0 * math.pi * index / steps
        turn = Rotation.from_rotvec(normalize(table_normal) * angle).as_matrix()
        candidates.append(_top_down_frame(turn @ base_x, table_normal))

    return min(
        candidates,
        key=lambda candidate: _rotation_distance(candidate, current_tip_rotation),
    )


def vertical_half_extent(
    object_rotation: np.ndarray,
    dimensions_m: np.ndarray,
    table_normal: np.ndarray,
) -> float:
    rotation = np.asarray(object_rotation, dtype=np.float64)
    dimensions = np.asarray(dimensions_m, dtype=np.float64)
    normal = normalize(table_normal)
    return 0.5 * float(np.sum(np.abs(rotation.T @ normal) * dimensions))


def build_top_down_grasp_pose(
    object_center_m: np.ndarray,
    object_rotation: np.ndarray,
    dimensions_m: np.ndarray,
    class_name: str,
    standoff_m: float,
    current_tip_rotation: np.ndarray,
    table_normal: np.ndarray | None = None,
) -> np.ndarray:
    """Build a collision-conscious top-down TCP pose above a primitive."""

    normal = normalize(
        np.asarray(
            [0.0, 0.0, 1.0] if table_normal is None else table_normal,
            dtype=np.float64,
        )
    )
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = _choose_grasp_rotation(
        class_name,
        object_rotation,
        current_tip_rotation,
        normal,
    )
    result[:3, 3] = (
        np.asarray(object_center_m, dtype=np.float64)
        + normal
        * (
            vertical_half_extent(object_rotation, dimensions_m, normal)
            + float(standoff_m)
        )
    )
    return result


def build_surface_aligned_grasp_pose(
    object_center_m: np.ndarray,
    object_rotation: np.ndarray,
    dimensions_m: np.ndarray,
    class_name: str,
    standoff_m: float,
    current_tip_rotation: np.ndarray,
    table_normal: np.ndarray,
    approach_axis: np.ndarray | None = None,
    max_tilt_deg: float | None = None,
) -> np.ndarray:
    """Build a geometry-only grasp pose aligned to an observable surface.

    The approach direction is derived from the fitted primitive frame rather
    than RGB appearance.  ``max_tilt_deg`` keeps the requested direction
    inside the robot's experimentally reachable cone while preserving the
    measured tilt for the pose-evaluation record.
    """

    table = normalize(np.asarray(table_normal, dtype=np.float64))
    rotation = np.asarray(object_rotation, dtype=np.float64).reshape(3, 3)
    if approach_axis is None:
        # The fitted third OBB axis is a stable surface normal for the tilted
        # benchmark primitives.  Use the table normal for an upright object.
        candidates = [rotation[:, index] for index in range(3)]
        approach = min(candidates, key=lambda axis: abs(float(np.dot(axis, table))))
        if abs(float(np.dot(approach, table))) < math.sin(math.radians(8.0)):
            approach = table
    else:
        approach = np.asarray(approach_axis, dtype=np.float64)
    approach = normalize(approach)
    if float(np.dot(approach, table)) < 0.0:
        approach = -approach

    if max_tilt_deg is not None:
        max_tilt = math.radians(max(0.0, float(max_tilt_deg)))
        cosine = float(np.clip(np.dot(approach, table), -1.0, 1.0))
        tilt = math.acos(cosine)
        if tilt > max_tilt and tilt > 1e-8:
            # Spherical interpolation toward the table normal avoids a sudden
            # orientation jump at the reachability boundary.
            blend = max_tilt / tilt
            approach = normalize((1.0 - blend) * table + blend * approach)

    z_axis = -approach
    x_hint = rotation[:, int(np.argmax(np.asarray(dimensions_m, dtype=np.float64)))]
    x_axis = x_hint - float(np.dot(x_hint, z_axis)) * z_axis
    if float(np.linalg.norm(x_axis)) < 1e-8:
        x_axis = np.asarray(current_tip_rotation, dtype=np.float64)[:, 0]
        x_axis -= float(np.dot(x_axis, z_axis)) * z_axis
    x_axis = normalize(x_axis)
    y_axis = normalize(np.cross(z_axis, x_axis))
    x_axis = normalize(np.cross(y_axis, z_axis))

    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = np.column_stack([x_axis, y_axis, z_axis])
    half_extent = 0.5 * float(
        np.sum(np.abs(rotation.T @ approach) * np.asarray(dimensions_m, dtype=np.float64))
    )
    result[:3, 3] = np.asarray(object_center_m, dtype=np.float64) + approach * (
        half_extent + float(standoff_m)
    )
    return result


@dataclass(frozen=True)
class PoseError:
    translation_base_m: np.ndarray
    rotation_base_rad: np.ndarray

    @property
    def position_norm_m(self) -> float:
        return float(np.linalg.norm(self.translation_base_m))

    @property
    def rotation_norm_rad(self) -> float:
        return float(np.linalg.norm(self.rotation_base_rad))


@dataclass(frozen=True)
class PBVSCommand:
    linear_velocity_base_m_s: np.ndarray
    angular_velocity_base_rad_s: np.ndarray
    error: PoseError


class PBVSController:
    """Proportional PBVS controller with velocity and increment limits."""

    def __init__(
        self,
        position_gain: float = 1.4,
        rotation_gain: float = 1.1,
        max_linear_speed_m_s: float = 0.035,
        max_angular_speed_rad_s: float = math.radians(24.0),
        max_translation_step_m: float = 0.003,
        max_rotation_step_rad: float = math.radians(2.0),
    ) -> None:
        self.position_gain = float(position_gain)
        self.rotation_gain = float(rotation_gain)
        self.max_linear_speed_m_s = float(max_linear_speed_m_s)
        self.max_angular_speed_rad_s = float(max_angular_speed_rad_s)
        self.max_translation_step_m = float(max_translation_step_m)
        self.max_rotation_step_rad = float(max_rotation_step_rad)

    @staticmethod
    def pose_error(current: np.ndarray, desired: np.ndarray) -> PoseError:
        current = np.asarray(current, dtype=np.float64)
        desired = np.asarray(desired, dtype=np.float64)
        translation = desired[:3, 3] - current[:3, 3]
        rotation_error = desired[:3, :3] @ current[:3, :3].T
        rotation_vector = Rotation.from_matrix(rotation_error).as_rotvec()
        return PoseError(translation, rotation_vector)

    @staticmethod
    def _limit(vector: np.ndarray, maximum: float) -> np.ndarray:
        value = np.asarray(vector, dtype=np.float64)
        norm = float(np.linalg.norm(value))
        if norm <= maximum or norm < 1e-12:
            return value
        return value * (maximum / norm)

    def command(self, current: np.ndarray, desired: np.ndarray) -> PBVSCommand:
        error = self.pose_error(current, desired)
        linear = self._limit(
            self.position_gain * error.translation_base_m,
            self.max_linear_speed_m_s,
        )
        angular = self._limit(
            self.rotation_gain * error.rotation_base_rad,
            self.max_angular_speed_rad_s,
        )
        return PBVSCommand(linear, angular, error)

    def next_pose(
        self,
        current: np.ndarray,
        command: PBVSCommand,
        dt_s: float,
    ) -> np.ndarray:
        translation_step = self._limit(
            command.linear_velocity_base_m_s * float(dt_s),
            self.max_translation_step_m,
        )
        rotation_step = self._limit(
            command.angular_velocity_base_rad_s * float(dt_s),
            self.max_rotation_step_rad,
        )
        result = np.asarray(current, dtype=np.float64).copy()
        result[:3, 3] += translation_step
        result[:3, :3] = (
            Rotation.from_rotvec(rotation_step).as_matrix()
            @ result[:3, :3]
        )
        return result

    @staticmethod
    def converged(
        error: PoseError,
        position_tolerance_m: float,
        rotation_tolerance_rad: float,
    ) -> bool:
        return (
            error.position_norm_m <= float(position_tolerance_m)
            and error.rotation_norm_rad <= float(rotation_tolerance_rad)
        )
