import math

import numpy as np
from scipy.spatial.transform import Rotation

from visual_servo_controller import (
    PBVSController,
    build_surface_aligned_grasp_pose,
    build_top_down_grasp_pose,
)
from visual_servo_perception import (
    TargetTrackState,
    estimate_target_from_points,
    select_servo_target,
)
from ensure_simulation_stopped import is_grasp_connector_alias


def test_top_down_grasp_pose_has_expected_height_and_axis():
    center = np.asarray([0.5, 0.0, 0.02])
    pose = build_top_down_grasp_pose(
        center,
        np.eye(3),
        np.asarray([0.04, 0.04, 0.04]),
        "cube",
        0.08,
        np.eye(3),
        np.asarray([0.0, 0.0, 1.0]),
    )
    assert np.allclose(pose[:3, 2], [0.0, 0.0, -1.0])
    assert math.isclose(pose[2, 3], 0.12, abs_tol=1e-9)


def test_surface_aligned_pose_is_reachable_and_right_handed():
    pose = build_surface_aligned_grasp_pose(
        np.asarray([0.5, 0.0, 0.05]),
        Rotation.from_euler("y", 35, degrees=True).as_matrix(),
        np.asarray([0.07, 0.035, 0.03]),
        "cuboid",
        0.01,
        np.eye(3),
        np.asarray([0.0, 0.0, 1.0]),
        max_tilt_deg=8.0,
    )
    assert math.isclose(np.linalg.det(pose[:3, :3]), 1.0, abs_tol=1e-6)
    approach = -pose[:3, 2]
    tilt = math.degrees(math.acos(float(np.clip(np.dot(approach, [0, 0, 1]), -1.0, 1.0))))
    assert tilt <= 8.001


def test_pbvs_increment_respects_translation_and_rotation_limits():
    controller = PBVSController()
    current = np.eye(4)
    desired = np.eye(4)
    desired[:3, 3] = [0.1, 0.0, 0.0]
    desired[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    command = controller.command(current, desired)
    next_pose = controller.next_pose(current, command, 0.5)
    assert np.linalg.norm(next_pose[:3, 3]) <= 0.0030001
    angle = Rotation.from_matrix(next_pose[:3, :3]).magnitude()
    assert angle <= math.radians(2.0001)


def test_servo_target_selection_prefers_unblocked_topmost_object():
    recognition = {
        "objects": [
            {
                "id": 0,
                "confidence": 0.95,
                "pose_valid": True,
                "grasp_planning": {
                    "topmost": False,
                    "grasp_blocked": True,
                    "priority_score": 0.35,
                },
            },
            {
                "id": 1,
                "confidence": 0.70,
                "pose_valid": True,
                "grasp_planning": {
                    "topmost": True,
                    "grasp_blocked": False,
                    "priority_score": 1.0,
                },
            },
        ]
    }
    assert select_servo_target(recognition)["id"] == 1


def test_local_geometry_tracker_recovers_box_center_without_color():
    center = np.asarray([0.50, -0.02, 0.04])
    dimensions = np.asarray([0.04, 0.04, 0.04])
    values = np.linspace(-0.02, 0.02, 8)
    points = []
    for first in values:
        for second in values:
            points.extend(
                [
                    center + [first, second, -0.02],
                    center + [first, second, 0.02],
                    center + [first, -0.02, second],
                    center + [first, 0.02, second],
                    center + [-0.02, first, second],
                    center + [0.02, first, second],
                ]
            )
    state = TargetTrackState(
        object_id=0,
        class_name="cube",
        center_base_m=center + np.asarray([0.004, -0.003, 0.002]),
        rotation_base=np.eye(3),
        dimensions_m=dimensions,
        confidence=0.8,
    )
    observation = estimate_target_from_points(np.asarray(points), state)
    assert observation is not None
    assert np.linalg.norm(observation.center_base_m - center) < 0.003
    assert observation.point_count >= 35


def test_stale_connector_alias_recognizes_underscore_and_hyphen_names():
    assert is_grasp_connector_alias("grasp_connector_rand_cube_01")
    assert is_grasp_connector_alias("grasp-connector-rand-cube-01")
    assert not is_grasp_connector_alias("gripper_tip")
