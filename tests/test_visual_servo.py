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
from ensure_simulation_stopped import (
    is_grasp_connector_alias,
    is_grasp_drop_box_alias,
)
from visual_servo_runner import (
    SERVO_GRASP_PLANE_OFFSET_M,
    SERVO_MIN_TCP_TABLE_CLEARANCE_M,
    _approach_window_converged,
    _cone_cross_section_center,
    _grasp_section_half_width,
    _attach_connector,
    _target_grasp_pose,
)


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


def test_final_grasp_pose_centers_tcp_on_rg2_jaw_plane():
    center = np.asarray([0.5, 0.0, 0.0225])
    state = TargetTrackState(
        object_id=0,
        class_name="sphere",
        center_base_m=center,
        rotation_base=np.eye(3),
        dimensions_m=np.asarray([0.045, 0.045, 0.045]),
        confidence=0.9,
    )
    pregrasp = _target_grasp_pose(
        state,
        0.09,
        np.eye(3),
        np.asarray([0.0, 0.0, 1.0]),
        "top_down",
    )
    final = _target_grasp_pose(
        state,
        SERVO_GRASP_PLANE_OFFSET_M,
        np.eye(3),
        np.asarray([0.0, 0.0, 1.0]),
        "top_down",
        centered_grasp=True,
    )
    assert math.isclose(pregrasp[2, 3], 0.135, abs_tol=1e-9)
    np.testing.assert_allclose(
        final[:3, 3], center + [0.0, 0.0, SERVO_GRASP_PLANE_OFFSET_M]
    )
    assert math.isclose(
        np.linalg.norm(final[:3, 3] - center),
        SERVO_GRASP_PLANE_OFFSET_M,
        abs_tol=1e-9,
    )
    assert final[2, 3] >= SERVO_MIN_TCP_TABLE_CLEARANCE_M


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


def test_cone_grasp_uses_wider_lower_cross_section():
    state = TargetTrackState(
        object_id=1,
        class_name="cone",
        center_base_m=np.asarray([0.5, 0.0, 0.05]),
        rotation_base=np.eye(3),
        dimensions_m=np.asarray([0.045, 0.045, 0.060]),
        confidence=0.8,
    )
    section_center = _cone_cross_section_center(
        state, np.asarray([0.0, 0.0, 1.0])
    )
    assert math.isclose(section_center[2], 0.041, abs_tol=1e-9)
    assert section_center[2] < state.center_base_m[2]


def test_cylinder_jaw_width_ignores_noisy_pca_height_projection():
    state = TargetTrackState(
        object_id=2,
        class_name="cylinder",
        center_base_m=np.asarray([0.5, 0.0, 0.03]),
        rotation_base=Rotation.from_euler("y", 25, degrees=True).as_matrix(),
        dimensions_m=np.asarray([0.040, 0.040, 0.060]),
        confidence=0.8,
    )
    assert math.isclose(
        _grasp_section_half_width(state, np.asarray([1.0, 0.0, 0.0]), np.asarray([0.0, 0.0, 1.0])),
        0.020,
        abs_tol=1e-12,
    )


def test_cuboid_grasp_axis_uses_short_side_for_rg2():
    pose = build_top_down_grasp_pose(
        np.asarray([0.5, 0.0, 0.03]),
        np.eye(3),
        np.asarray([0.070, 0.035, 0.030]),
        "cuboid",
        0.01,
        np.eye(3),
        np.asarray([0.0, 0.0, 1.0]),
    )
    assert abs(float(np.dot(pose[:3, 0], [0.0, 1.0, 0.0]))) > 0.99


def test_approach_window_rejects_noisy_or_drifting_target():
    errors = [0.0030, 0.0032, 0.0031, 0.0033, 0.0032]
    stable_centers = [np.asarray([0.5, 0.0, 0.04]) for _ in errors]
    assert _approach_window_converged(errors, stable_centers)
    drifting_centers = [
        np.asarray([0.5 + index * 0.001, 0.0, 0.04])
        for index in range(len(errors))
    ]
    assert not _approach_window_converged(errors, drifting_centers)


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


def test_upper_random_target_excludes_blocked_base_and_sphere():
    recognition = {
        "objects": [
            {
                "id": 0,
                "class": "cuboid",
                "center_m": [0.5, 0.0, 0.015],
                "confidence": 0.95,
                "pose_valid": True,
                "grasp_planning": {
                    "topmost": False,
                    "grasp_blocked": True,
                },
            },
            {
                "id": 1,
                "class": "sphere",
                "center_m": [0.5, 0.0, 0.062],
                "confidence": 0.99,
                "pose_valid": True,
                "grasp_planning": {
                    "topmost": True,
                    "grasp_blocked": False,
                },
            },
            {
                "id": 2,
                "class": "cylinder",
                "center_m": [0.5, 0.0, 0.060],
                "confidence": 0.40,
                "pose_valid": True,
                "grasp_planning": {
                    "topmost": True,
                    "grasp_blocked": False,
                },
            },
        ]
    }
    selected = select_servo_target(
        recognition,
        policy="upper-random",
        random_seed=20260811,
    )
    assert selected["id"] == 2


def test_drop_box_alias_is_cleaned_separately_from_connector_alias():
    assert is_grasp_drop_box_alias("grasp_drop_box_floor")
    assert not is_grasp_drop_box_alias("rand_cube_01")


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


def test_connector_attachment_preserves_object_world_position():
    class FakeSim:
        handle_scene = -1
        sceneobject_shape = 1
        shapeintparam_static = 2
        shapeintparam_respondable = 3

        def __init__(self):
            self.positions = {
                10: np.asarray([0.50, -0.02, 0.0225], dtype=np.float64),
                20: np.asarray([0.50, -0.02, 0.0600], dtype=np.float64),
            }
            self.aliases = {10: "rand_sphere_01"}
            self.parents = []
            self.next_dummy = 30

        def getObjectsInTree(self, *_args):
            return [10]

        def getObjectAlias(self, handle):
            return self.aliases[int(handle)]

        def getObjectPosition(self, handle, _relative):
            return self.positions[int(handle)].tolist()

        def getObjectPose(self, handle, _relative):
            return [*self.positions[int(handle)].tolist(), 0.0, 0.0, 0.0, 1.0]

        def setObjectInt32Param(self, *_args):
            return None

        def createDummy(self, _size):
            handle = self.next_dummy
            self.next_dummy += 1
            self.positions[handle] = np.zeros(3, dtype=np.float64)
            return handle

        def setObjectAlias(self, handle, alias):
            self.aliases[int(handle)] = str(alias)

        def setObjectPose(self, handle, pose, _relative):
            self.positions[int(handle)] = np.asarray(pose[:3], dtype=np.float64)

        def setObjectParent(self, child, parent, keep_in_place):
            assert keep_in_place is True
            self.parents.append((int(child), int(parent)))

    sim = FakeSim()
    state = TargetTrackState(
        object_id=0,
        class_name="sphere",
        center_base_m=np.asarray([0.50, -0.02, 0.0225]),
        rotation_base=np.eye(3),
        dimensions_m=np.asarray([0.045, 0.045, 0.045]),
        confidence=0.8,
    )
    before = sim.positions[10].copy()
    result = _attach_connector(sim, robot_base=1, tip=20, target_state=state)
    np.testing.assert_allclose(sim.positions[10], before, atol=1e-12)
    assert result["center_snapped_to_tip"] is False
    assert result["relative_attachment_preserved"] is True
    assert result["snap_distance_mm"] < 1e-9
    assert (10, result["connector_handle"]) in sim.parents
