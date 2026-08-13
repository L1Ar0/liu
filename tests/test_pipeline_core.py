from __future__ import annotations

import unittest

import numpy as np
import open3d as o3d

from active_second_view import _second_camera_pose
from geometric_segmentation import split_geometric_cluster
from primitive_fitting import fit_primitive_candidates
from shape_catalog import CAD_DIMENSIONS_M, get_cad_dimensions
from grasp_planning import annotate_grasp_planning
from depth_grasp_rl import action_reward, encode_action, normalize_depth
from end_to_end_grasp_env import EndToEndGraspEnv, STAGE_CONFIGS, pregrasp_target_point, stage_task_success
from planned_scene_randomizer import _obb_penetrates, build_planned_layout


class PrimitiveFittingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(11)

    def test_sphere_candidate(self) -> None:
        u = self.rng.uniform(-1.0, 1.0, 1000)
        phi = self.rng.uniform(0.0, 2.0 * np.pi, 1000)
        radius = 0.0225
        points = np.column_stack(
            [
                radius * np.sqrt(1.0 - u * u) * np.cos(phi),
                radius * np.sqrt(1.0 - u * u) * np.sin(phi),
                radius * u,
            ]
        )
        result = fit_primitive_candidates(points)
        self.assertEqual(result["class"], "sphere")

    def test_cone_candidate(self) -> None:
        theta = self.rng.uniform(0.0, 2.0 * np.pi, 1200)
        z = self.rng.uniform(-0.03, 0.03, 1200)
        radius = 0.0225 * (0.5 - z / 0.06)
        points = np.column_stack(
            [radius * np.cos(theta), radius * np.sin(theta), z]
        )
        result = fit_primitive_candidates(points)
        self.assertEqual(result["class"], "cone")

    def test_spheroid_candidate(self) -> None:
        u = self.rng.uniform(-1.0, 1.0, 1000)
        phi = self.rng.uniform(0.0, 2.0 * np.pi, 1000)
        points = np.column_stack(
            [
                0.025 * np.sqrt(1.0 - u * u) * np.cos(phi),
                0.020 * np.sqrt(1.0 - u * u) * np.sin(phi),
                0.020 * u,
            ]
        )
        result = fit_primitive_candidates(points)
        self.assertEqual(result["class"], "spheroid")

    def test_cylinder_with_caps_candidate(self) -> None:
        theta = self.rng.uniform(0.0, 2.0 * np.pi, 900)
        z = self.rng.uniform(-0.03, 0.03, 900)
        side = np.column_stack([0.02 * np.cos(theta), 0.02 * np.sin(theta), z])
        theta_cap = self.rng.uniform(0.0, 2.0 * np.pi, 500)
        radius_cap = 0.02 * np.sqrt(self.rng.uniform(0.0, 1.0, 500))
        caps = np.vstack(
            [
                np.column_stack(
                    [radius_cap * np.cos(theta_cap), radius_cap * np.sin(theta_cap), np.full(500, -0.03)]
                ),
                np.column_stack(
                    [radius_cap * np.cos(theta_cap), radius_cap * np.sin(theta_cap), np.full(500, 0.03)]
                ),
            ]
        )
        result = fit_primitive_candidates(np.vstack([side, caps]))
        self.assertEqual(result["class"], "cylinder")

    def test_cube_candidate(self) -> None:
        faces = []
        for axis in range(3):
            for sign in (-1.0, 1.0):
                values = self.rng.uniform(-0.02, 0.02, (180, 2))
                points = np.zeros((180, 3), dtype=np.float64)
                other_axes = [index for index in range(3) if index != axis]
                points[:, axis] = sign * 0.02
                points[:, other_axes[0]] = values[:, 0]
                points[:, other_axes[1]] = values[:, 1]
                faces.append(points)
        result = fit_primitive_candidates(np.vstack(faces))
        self.assertEqual(result["class"], "cube")

    def test_occluded_cube_top_patch_uses_partial_planar_catalog_match(self) -> None:
        grid = np.linspace(-0.019, 0.019, 14)
        top = np.asarray(
            [(x, y, 0.04) for x in grid for y in grid],
            dtype=np.float64,
        )
        # Simulate points from a tilted contacting part crossing above the
        # visible top face.  The whole cluster is not planar, but the dominant
        # rectangular patch still identifies the finite-catalog cube.
        overhang = np.column_stack(
            [
                self.rng.uniform(0.010, 0.030, 90),
                self.rng.uniform(-0.018, 0.018, 90),
                self.rng.uniform(0.043, 0.064, 90),
            ]
        )
        result = fit_primitive_candidates(np.vstack([top, overhang]), top_k=6)
        self.assertEqual(result["class"], "cube")
        self.assertTrue(result["features"]["partial_planar_observation"])
        candidate_scores = {
            str(item["class"]): float(item["score"])
            for item in result["candidates"]
        }
        self.assertEqual(candidate_scores["sphere"], 0.0)


class ExchangeContractTests(unittest.TestCase):
    def test_stage_a_uses_true_two_dimensional_policy_space(self) -> None:
        env = EndToEndGraspEnv(curriculum_stage="A")
        self.assertEqual(env.action_space.shape, (2,))
        self.assertEqual(env.observation_space["proprio"].shape, (14,))

    def test_later_stage_retains_full_policy_space(self) -> None:
        env = EndToEndGraspEnv(curriculum_stage="B")
        self.assertEqual(env.action_space.shape, (7,))
        self.assertEqual(env.observation_space["proprio"].shape, (27,))

    def test_table_relative_height_is_bounded_and_table_zeroed(self) -> None:
        class FakeSim:
            @staticmethod
            def getObjectMatrix(_camera, _robot_base):
                return [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]

        env = EndToEndGraspEnv(curriculum_stage="A", image_size=4)
        env.sim = FakeSim()
        env.camera = 1
        env.robot_base = 2
        env.table_plane = np.asarray([0.0, 0.0, 1.0, 0.05], dtype=np.float64)
        env.camera_params = {
            "fov_x": np.deg2rad(60.0),
            "fov_y": np.deg2rad(45.0),
            "near": 0.01,
            "far": 1.0,
        }
        height = env._table_relative_height(np.asarray([[0.05, 0.10, 0.20]], dtype=np.float32))
        np.testing.assert_allclose(height, [[0.0, 0.5, 1.0]], atol=1e-6)

    def test_curriculum_success_is_stage_specific(self) -> None:
        self.assertTrue(stage_task_success("A", grasp_success=False, stage_contact_success=False, stage_distance_success=True))
        self.assertFalse(stage_task_success("D", grasp_success=False, stage_contact_success=False, stage_distance_success=True))
        self.assertTrue(stage_task_success("C", grasp_success=False, stage_contact_success=True, stage_distance_success=False))
        self.assertFalse(stage_task_success("E", grasp_success=False, stage_contact_success=True, stage_distance_success=True))

    def test_pregrasp_point_is_above_object_extent(self) -> None:
        point = pregrasp_target_point(np.asarray([0.0, 0.0, 0.02]), [0.04, 0.04, 0.04], standoff_m=0.07)
        self.assertAlmostEqual(float(point[2]), 0.11, places=6)
        self.assertEqual(STAGE_CONFIGS["D"].action_mask, (1, 1, 1, 0, 0, 1, 1))

    def test_cad_dimensions_exist(self) -> None:
        for cad_id, dimensions in CAD_DIMENSIONS_M.items():
            self.assertEqual(get_cad_dimensions(cad_id), dimensions)

    def test_depth_policy_reward_accepts_matching_geometry_label(self) -> None:
        action = encode_action((320.0, 240.0), 640, 480, 0.0, np.asarray([0.0, 0.0, 1.0]))
        reward, success, metrics = action_reward(action, action)
        self.assertTrue(success)
        self.assertGreater(reward, 3.0)
        self.assertAlmostEqual(metrics["position_error"], 0.0, places=6)

    def test_depth_normalization_is_bounded(self) -> None:
        value = normalize_depth(np.asarray([[0.0, 0.5, 2.0]], dtype=np.float32))
        self.assertGreaterEqual(float(value.min()), 0.0)
        self.assertLessEqual(float(value.max()), 1.0)

    def test_planar_box_patch_does_not_become_sphere(self) -> None:
        grid = np.linspace(-0.02, 0.02, 10)
        values = np.asarray([(x, y) for x in grid for y in grid], dtype=np.float64)
        points = np.column_stack(
            [values[:, 0], values[:, 1], np.full(len(values), 0.04)]
        )
        result = fit_primitive_candidates(points)
        self.assertIn(result["class"], {"cube", "cuboid"})
        self.assertTrue(result["features"]["planar_observation"])

    def test_planned_layouts_have_valid_support_graphs(self) -> None:
        reference = {
            "table_z": 0.0,
            "workspace_center_x": 0.0,
            "workspace_center_y": 0.0,
            "workspace_half_x": 0.11,
            "workspace_half_y": 0.09,
        }
        for layout in (
            "table_only",
            "stack",
            "side_contact",
            "partial_support",
            "leaning",
            "bridge",
            "mixed",
        ):
            plans, selected = build_planned_layout(reference, None, seed=20260810, layout=layout)
            self.assertEqual(selected, layout)
            self.assertGreaterEqual(len(plans), 5)
            plan_ids = {str(item["plan_id"]) for item in plans}
            for item in plans:
                self.assertTrue(np.all(np.isfinite(item["center"])))
                rotation = np.asarray(item["rotation"], dtype=np.float64)
                np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
                for support in item.get("support_refs", []):
                    self.assertTrue(support == "table" or support in plan_ids)
            for first_index, first in enumerate(plans):
                for second in plans[first_index + 1 :]:
                    self.assertFalse(_obb_penetrates(first, second))
            if layout == "stack":
                self.assertTrue(any("stack_base" in item.get("support_refs", []) for item in plans))
            if layout in {"partial_support", "bridge"}:
                self.assertTrue(any(len(item.get("support_refs", [])) > 1 for item in plans))
            if layout == "leaning":
                self.assertTrue(any("tilt_support" in item.get("contact_refs", []) for item in plans))

    def test_grasp_planning_marks_upper_object_accessible(self) -> None:
        objects = [
            {
                "id": 0,
                "class": "cuboid",
                "center_m": [0.0, 0.0, 0.02],
                "rotation_matrix": np.eye(3).tolist(),
                "geometry": {"length_m": 0.07, "width_m": 0.035, "height_m": 0.03},
            },
            {
                "id": 1,
                "class": "cube",
                "center_m": [0.0, 0.0, 0.05],
                "rotation_matrix": np.eye(3).tolist(),
                "geometry": {"length_m": 0.04, "width_m": 0.04, "height_m": 0.04},
            },
        ]
        annotated = annotate_grasp_planning(objects)
        self.assertTrue(annotated[1]["grasp_planning"]["topmost"])
        self.assertTrue(annotated[0]["grasp_planning"]["grasp_blocked"])
        self.assertGreater(annotated[1]["grasp_planning"]["candidate_count"], 0)
        self.assertIn("candidates", annotated[1]["grasp_planning"])

    def test_touching_cube_and_cylinder_split_without_color(self) -> None:
        rng = np.random.default_rng(18)
        cube_faces = []
        for axis in range(3):
            for sign in (-1.0, 1.0):
                values = rng.uniform(-0.02, 0.02, (220, 2))
                points = np.zeros((220, 3), dtype=np.float64)
                other_axes = [index for index in range(3) if index != axis]
                points[:, axis] = sign * 0.02
                points[:, other_axes[0]] = values[:, 0]
                points[:, other_axes[1]] = values[:, 1]
                points[:, 0] -= 0.03
                cube_faces.append(points)

        theta = rng.uniform(0.0, 2.0 * np.pi, 1320)
        height = rng.uniform(-0.03, 0.03, 1320)
        cylinder = np.column_stack(
            [
                0.02 * np.cos(theta) + 0.03,
                0.02 * np.sin(theta),
                height,
            ]
        )
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(
            np.vstack([*cube_faces, cylinder])
        )
        pieces, diagnostic = split_geometric_cluster(cloud, min_cluster_points=24)
        self.assertTrue(diagnostic["split"])
        self.assertEqual(len(pieces), 2)

    def test_second_view_has_oblique_baseline_and_looks_at_target(self) -> None:
        current = np.eye(4, dtype=np.float64)
        current[:3, 2] = [0.0, 0.0, -1.0]
        current[:3, 1] = [0.0, 1.0, 0.0]
        current[:3, 0] = [1.0, 0.0, 0.0]
        current[:3, 3] = [0.0, 0.0, 0.35]
        target = np.asarray([0.0, 0.0, 0.02], dtype=np.float64)
        pose = _second_camera_pose(current, target, np.asarray([0.0, 0.0, 1.0]))
        self.assertGreater(np.linalg.norm(pose[:2, 3] - target[:2]), 0.07)
        expected_forward = target - pose[:3, 3]
        expected_forward /= np.linalg.norm(expected_forward)
        self.assertGreater(float(np.dot(pose[:3, 2], expected_forward)), 0.999)


if __name__ == "__main__":
    unittest.main()
