"""Arbitrary-orientation finite-catalog primitive recognition."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.optimize import least_squares

from primitive_fitting import fit_primitive_candidates
from grasp_planning import annotate_grasp_planning
from shape_catalog import dimensions_to_geometry, get_cad_dimensions, get_shape_spec


SCENE_GT_FILE = Path("random_scene_ground_truth.json")
SEGMENTATION_DIR = Path("segmentation_output")
SEGMENTATION_METADATA_FILE = SEGMENTATION_DIR / "segmentation_metadata.json"
OUTPUT_DIR = Path("recognition_output")
RESULT_JSON = OUTPUT_DIR / "recognition_results.json"
SHOW_VISUALIZATION = os.environ.get("ROBOT_GRASP_HEADLESS") != "1"
PRIMITIVE_TOP_K = max(
    1,
    int(os.environ.get("ROBOT_GRASP_PRIMITIVE_TOP_K", "6")),
)


def _load_context() -> tuple[dict, dict]:
    scene = json.loads(SCENE_GT_FILE.read_text(encoding="utf-8"))
    segmentation = json.loads(
        SEGMENTATION_METADATA_FILE.read_text(encoding="utf-8")
    )
    if scene.get("scene_id") != segmentation.get("scene_id"):
        raise RuntimeError("场景GT与分割结果的scene_id不一致。")
    return scene, segmentation


def _load_cluster(file_name: str) -> o3d.geometry.PointCloud:
    path = SEGMENTATION_DIR / file_name
    cloud = o3d.io.read_point_cloud(str(path))
    if len(cloud.points) < 20:
        raise RuntimeError(f"Cluster点数不足：{path}")
    return cloud


def _quality_warnings(fit: dict) -> list[str]:
    warnings: list[str] = []
    confidence = float(fit["confidence"])
    if confidence < 0.35:
        warnings.append("ambiguous_primitive_fit")
    candidates = fit.get("candidates", [])
    if len(candidates) >= 2:
        margin = float(candidates[0]["score"] - candidates[1]["score"])
        if margin < 0.08:
            warnings.append("small_topk_margin")
    features = fit.get("features", {})
    extent = np.asarray(features.get("obb_extent_m", [0.0, 0.0, 0.0]), dtype=float)
    if extent.size and float(np.max(extent)) > 0.0875:
        warnings.append("oversized_possible_instance_merge")
    if bool(features.get("planar_observation", False)):
        warnings.append("partial_planar_observation")
    if int(fit.get("point_count", 0)) < 100:
        warnings.append("sparse_cluster")
    return warnings


def _complete_planar_pose(
    fit: dict,
    points: np.ndarray,
    table_plane: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Recover a catalog center from a visible horizontal face when possible."""

    center = np.asarray(fit["center"], dtype=np.float64).copy()
    rotation = np.asarray(fit["rotation"], dtype=np.float64).copy()
    warnings: list[str] = []
    if not bool(fit.get("features", {}).get("planar_observation", False)):
        return center, rotation, warnings
    if table_plane is None or len(table_plane) != 4:
        warnings.append("planar_pose_not_completed_no_table_plane")
        return center, rotation, warnings

    table_normal = np.asarray(table_plane[:3], dtype=np.float64)
    normal_norm = float(np.linalg.norm(table_normal))
    if normal_norm < 1e-9:
        warnings.append("planar_pose_not_completed_invalid_table_plane")
        return center, rotation, warnings
    table_normal /= normal_norm
    plane_offset = float(table_plane[3]) / normal_norm
    visible_normal = rotation[:, 2]
    alignment = abs(float(np.dot(visible_normal, table_normal)))
    if alignment < 0.75:
        warnings.append("planar_side_face_pose_ambiguous")
        return center, rotation, warnings

    if float(np.dot(visible_normal, table_normal)) < 0.0:
        # Flip two axes to keep a right-handed frame while pointing +Z away
        # from the table.
        rotation[:, 0] *= -1.0
        rotation[:, 2] *= -1.0

    class_name = str(fit["class"])
    dimensions = np.asarray(
        get_cad_dimensions(get_shape_spec(class_name).cad_id),
        dtype=np.float64,
    )
    vertical_half_extent = float(
        np.sum(np.abs(table_normal @ rotation) * (0.5 * dimensions))
    )
    observed_surface_height = float(np.median(np.asarray(points) @ table_normal + plane_offset))
    current_height = float(np.dot(center, table_normal) + plane_offset)
    target_height = observed_surface_height - vertical_half_extent
    center += table_normal * (target_height - current_height)
    warnings.append("planar_pose_completed_from_visible_face")
    return center, rotation, warnings


def _enforce_table_clearance(
    fit: dict,
    center: np.ndarray,
    rotation: np.ndarray,
    table_plane: np.ndarray | None,
) -> tuple[np.ndarray, list[str]]:
    """Reject impossible below-table estimates by lifting the catalog pose."""

    if table_plane is None or len(table_plane) != 4:
        return center, []
    table_normal = np.asarray(table_plane[:3], dtype=np.float64)
    normal_norm = float(np.linalg.norm(table_normal))
    if normal_norm < 1e-9:
        return center, []
    table_normal /= normal_norm
    plane_offset = float(table_plane[3]) / normal_norm
    dimensions = np.asarray(
        get_cad_dimensions(get_shape_spec(str(fit["class"])).cad_id),
        dtype=np.float64,
    )
    half_extent = float(np.sum(np.abs(table_normal @ rotation) * (0.5 * dimensions)))
    signed_bottom = float(np.dot(center, table_normal) + plane_offset - half_extent)
    if signed_bottom >= -0.002:
        return center, []
    corrected = center + table_normal * (-signed_bottom)
    return corrected, ["pose_lifted_to_table_clearance"]


def _table_frame(rotation: np.ndarray, table_normal: np.ndarray) -> np.ndarray:
    """Build a stable yaw frame whose local Z follows the table normal."""

    z_axis = np.asarray(table_normal, dtype=np.float64)
    z_axis /= max(float(np.linalg.norm(z_axis)), 1e-9)
    x_axis = np.asarray(rotation[:, 0], dtype=np.float64)
    x_axis -= float(np.dot(x_axis, z_axis)) * z_axis
    if float(np.linalg.norm(x_axis)) < 1e-6:
        x_axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(x_axis, z_axis))) > 0.9:
            x_axis = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        x_axis -= float(np.dot(x_axis, z_axis)) * z_axis
    x_axis /= max(float(np.linalg.norm(x_axis)), 1e-9)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= max(float(np.linalg.norm(y_axis)), 1e-9)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= max(float(np.linalg.norm(x_axis)), 1e-9)
    return np.column_stack([x_axis, y_axis, z_axis])


def _refine_upright_cylinder_center(
    points: np.ndarray,
    center: np.ndarray,
    rotation: np.ndarray,
    table_normal: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Recover an occlusion-robust cylinder axis center from a fixed-radius arc.

    A centroid of the visible RGB-D points is biased when a neighboring part
    hides one side of the cylinder.  The finite catalog supplies the radius;
    RANSAC over point pairs finds the circle center supported by a vertical
    arc, then a robust least-squares pass refines it.  No object count, color,
    or simulator metadata is used.
    """

    points = np.asarray(points, dtype=np.float64)
    if len(points) < 50:
        return center, False
    radius = 0.5 * float(
        get_cad_dimensions(get_shape_spec("cylinder").cad_id)[0]
    )
    z_axis = np.asarray(table_normal, dtype=np.float64)
    z_axis /= max(float(np.linalg.norm(z_axis)), 1e-12)
    x_axis = np.asarray(rotation[:, 0], dtype=np.float64)
    x_axis -= float(np.dot(x_axis, z_axis)) * z_axis
    if float(np.linalg.norm(x_axis)) < 1e-6:
        x_axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        x_axis -= float(np.dot(x_axis, z_axis)) * z_axis
    x_axis /= max(float(np.linalg.norm(x_axis)), 1e-12)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= max(float(np.linalg.norm(y_axis)), 1e-12)
    uv = np.column_stack([points @ x_axis, points @ y_axis])
    center_uv = np.asarray([center @ x_axis, center @ y_axis], dtype=np.float64)
    rng = np.random.default_rng(7919 + len(points))
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    pair_count = min(9000, max(1200, len(points) * 12))
    for _ in range(pair_count):
        first, second = rng.integers(0, len(uv), size=2)
        if first == second:
            continue
        delta = uv[second] - uv[first]
        distance = float(np.linalg.norm(delta))
        if distance < 0.20 * radius or distance > 2.0 * radius:
            continue
        midpoint = 0.5 * (uv[first] + uv[second])
        perpendicular = np.asarray([-delta[1], delta[0]], dtype=np.float64) / distance
        offset = math.sqrt(max(radius * radius - 0.25 * distance * distance, 0.0))
        for candidate in (midpoint + offset * perpendicular, midpoint - offset * perpendicular):
            radial_error = np.abs(np.linalg.norm(uv - candidate, axis=1) - radius)
            inliers = radial_error <= 0.0018
            count = int(inliers.sum())
            if count < max(40, int(0.12 * len(points))):
                continue
            vertical_span = float(np.ptp(points[inliers] @ z_axis)) if count >= 3 else 0.0
            score = float(count) + 1500.0 * vertical_span
            if best is None or score > best[0]:
                best = (score, candidate, inliers)
    if best is None:
        return center, False
    _, candidate_uv, inliers = best
    refined = least_squares(
        lambda value: np.linalg.norm(uv[inliers] - value, axis=1) - radius,
        candidate_uv,
        loss="soft_l1",
        f_scale=0.001,
        max_nfev=100,
    ).x
    correction_uv = refined - center_uv
    if float(np.linalg.norm(correction_uv)) > 0.025:
        return center, False
    corrected = np.asarray(center, dtype=np.float64).copy()
    corrected += correction_uv[0] * x_axis + correction_uv[1] * y_axis
    return corrected, True


def _refine_table_cube_pose(
    points: np.ndarray,
    center: np.ndarray,
    rotation: np.ndarray,
    table_normal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Fit a square top footprint to remove PCA yaw noise for a cube."""

    points = np.asarray(points, dtype=np.float64)
    if len(points) < 60:
        return center, rotation, False
    z_axis = np.asarray(table_normal, dtype=np.float64)
    z_axis /= max(float(np.linalg.norm(z_axis)), 1e-12)
    x_ref = np.asarray(rotation[:, 0], dtype=np.float64)
    x_ref -= float(np.dot(x_ref, z_axis)) * z_axis
    if float(np.linalg.norm(x_ref)) < 1e-6:
        x_ref = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        x_ref -= float(np.dot(x_ref, z_axis)) * z_axis
    x_ref /= max(float(np.linalg.norm(x_ref)), 1e-12)
    y_ref = np.cross(z_axis, x_ref)
    y_ref /= max(float(np.linalg.norm(y_ref)), 1e-12)
    heights = points @ z_axis
    top = float(np.percentile(heights, 99.0))
    top_points = points[heights >= top - 0.0035]
    if len(top_points) < 50:
        return center, rotation, False
    xy = np.column_stack([top_points @ x_ref, top_points @ y_ref])
    side = 0.040
    best: tuple[float, float, np.ndarray] | None = None
    for angle in np.linspace(0.0, math.pi / 2.0, 721):
        u = np.asarray([math.cos(float(angle)), math.sin(float(angle))])
        v = np.asarray([-math.sin(float(angle)), math.cos(float(angle))])
        local = np.column_stack([xy @ u, xy @ v])
        low, high = np.quantile(local, [0.01, 0.99], axis=0)
        extent = high - low
        error = float(np.mean(np.abs(extent - side) / side))
        if best is None or error < best[0]:
            best = (error, float(angle), 0.5 * (low + high))
    if best is None or best[0] > 0.12:
        return center, rotation, False
    _, angle, local_center = best
    u = np.asarray([math.cos(angle), math.sin(angle)])
    v = np.asarray([-math.sin(angle), math.cos(angle)])
    center_xy = local_center[0] * u + local_center[1] * v
    corrected = np.asarray(center, dtype=np.float64).copy()
    current_xy = np.asarray([corrected @ x_ref, corrected @ y_ref])
    correction_xy = center_xy - current_xy
    corrected += correction_xy[0] * x_ref + correction_xy[1] * y_ref
    x_axis = u[0] * x_ref + u[1] * y_ref
    y_axis = v[0] * x_ref + v[1] * y_ref
    refined_rotation = np.column_stack([x_axis, y_axis, z_axis])
    return corrected, refined_rotation, True


def _refine_box_pose(
    points: np.ndarray,
    center: np.ndarray,
    rotation: np.ndarray,
    class_name: str,
    catalog_locked: bool,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Resolve a cuboid OBB's ambiguous PCA rotation against catalog extents."""

    if class_name not in {"cube", "cuboid"} or not catalog_locked:
        return center, rotation, []
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 40:
        return center, rotation, []
    dimensions = np.asarray(
        get_cad_dimensions(get_shape_spec(class_name).cad_id),
        dtype=np.float64,
    )
    reference_center = np.asarray(center, dtype=np.float64)
    reference_rotation = np.asarray(rotation, dtype=np.float64)

    def evaluate(candidate_rotation: np.ndarray) -> tuple[float, np.ndarray]:
        local = (points - reference_center) @ candidate_rotation
        extent = np.ptp(local, axis=0)
        error = float(
            np.mean(np.abs(extent - dimensions) / np.maximum(dimensions, 1e-9))
        )
        candidate_center = reference_center + (
            0.5 * (local.min(axis=0) + local.max(axis=0))
        ) @ candidate_rotation.T
        return error, candidate_center

    best_error, best_center = evaluate(reference_rotation)
    best_rotation = reference_rotation
    best_angle = 0.0
    for axis in range(3):
        for angle in np.linspace(-math.pi / 2.0, math.pi / 2.0, 361):
            cosine = math.cos(float(angle))
            sine = math.sin(float(angle))
            twist = np.eye(3, dtype=np.float64)
            first = (axis + 1) % 3
            second = (axis + 2) % 3
            twist[first, first] = cosine
            twist[second, second] = cosine
            twist[first, second] = -sine
            twist[second, first] = sine
            candidate_rotation = reference_rotation @ twist
            error, candidate_center = evaluate(candidate_rotation)
            if error < best_error:
                best_error = error
                best_center = candidate_center
                best_rotation = candidate_rotation
                best_angle = float(angle)
    if abs(best_angle) < math.radians(2.0):
        return center, rotation, []
    return best_center, best_rotation, ["catalog_box_pose_refined"]


def _resolve_closed_set_ambiguity(
    fit: dict,
    points: np.ndarray,
    center: np.ndarray,
    rotation: np.ndarray,
    table_plane: np.ndarray | None,
) -> tuple[dict, np.ndarray, np.ndarray, list[str]]:
    """Use table support and catalog dimensions to resolve a tall round object."""

    warnings: list[str] = []
    if table_plane is None or len(table_plane) != 4:
        return fit, center, rotation, warnings
    table_normal = np.asarray(table_plane[:3], dtype=np.float64)
    normal_norm = float(np.linalg.norm(table_normal))
    if normal_norm < 1e-9:
        return fit, center, rotation, warnings
    table_normal /= normal_norm
    plane_offset = float(table_plane[3]) / normal_norm
    heights = np.asarray(points, dtype=np.float64) @ table_normal + plane_offset
    min_height = float(np.percentile(heights, 1.0))
    span_height = float(np.percentile(heights, 99.0) - min_height)

    candidates = {
        str(item["class"]): float(item["score"])
        for item in fit.get("candidates", [])
    }
    selected = str(fit["class"])
    cylinder_score = candidates.get("cylinder", 0.0)
    dimension_scores = fit.get("features", {}).get("catalog_dimension_scores", {})
    cylinder_dimension = float(dimension_scores.get("cylinder", 0.0))
    box_dimension = max(
        float(dimension_scores.get("cube", 0.0)),
        float(dimension_scores.get("cuboid", 0.0)),
    )
    axisymmetric_score = float(
        fit.get("features", {}).get("axisymmetric_cross_section_score", 0.0)
    )
    normal_axis_alignment = float(
        fit.get("features", {}).get("normal_axis_alignment", 0.0)
    )
    cylinder_fit_error = float(
        fit.get("features", {}).get("cylinder_fit_error", 1.0)
    )
    cone_fit_error = float(
        fit.get("features", {}).get("cone_fit_error", 1.0)
    )
    cone_slope_strength = float(
        fit.get("features", {}).get("cone_slope_strength", 0.0)
    )
    cone_normal_alignment = float(
        fit.get("features", {}).get("cone_surface_normal_alignment", 0.0)
    )
    top_height = float(np.percentile(heights, 99.0))

    # For a table-supported upright object, the highest visible point is a
    # much stronger cue than a partial OBB.  The object-bottom band is removed
    # during table extraction, so the top height remains close to the catalog
    # height even when the lower side is occluded.
    if (
        min_height <= 0.012
        and normal_axis_alignment >= 0.72
        and selected == "cylinder"
        and top_height <= 0.045
    ):
        box_class = min(
            ("cube", "cuboid"),
            key=lambda name: abs(
                top_height
                - float(get_cad_dimensions(get_shape_spec(name).cad_id)[2])
            ),
        )
        box_score = float(candidates.get(box_class, 0.0))
        if box_score >= 0.10:
            fit = dict(fit)
            fit["class"] = box_class
            fit["confidence"] = max(0.40, box_score)
            fit["geometry"] = dimensions_to_geometry(
                box_class,
                get_cad_dimensions(get_shape_spec(box_class).cad_id),
            )
            fit["features"] = {
                **dict(fit.get("features", {})),
                "closed_set_override": "table_supported_top_height_box",
            }
            rotation = _table_frame(rotation, table_normal)
            warnings.append("table_supported_top_height_box_override")
            selected = box_class

    # Cube and the catalog cuboid have similar visible top patches.  When the
    # object is upright on the table, the top height separates them reliably:
    # the cube is 40 mm tall while the cuboid is 30 mm tall.  Apply this only
    # to a table-supported, axis-aligned observation; a leaning/bridge object
    # must remain governed by its full 3-D fit.
    cube_score = float(candidates.get("cube", 0.0))
    cuboid_score = float(candidates.get("cuboid", 0.0))
    if (
        min_height <= 0.012
        and normal_axis_alignment >= 0.72
        and selected in {"cube", "cuboid"}
        and 0.034 <= top_height <= 0.047
        and cube_score >= max(0.10, 0.72 * cuboid_score)
    ):
        fit = dict(fit)
        fit["class"] = "cube"
        fit["confidence"] = max(0.40, cube_score)
        fit["geometry"] = dimensions_to_geometry(
            "cube",
            get_cad_dimensions(get_shape_spec("cube").cad_id),
        )
        fit["features"] = {
            **dict(fit.get("features", {})),
            "closed_set_override": "table_supported_cube_top_height",
        }
        rotation = _table_frame(rotation, table_normal)
        warnings.append("table_supported_cube_top_height_override")
        selected = "cube"

    if (
        selected == "cuboid"
        and min_height <= 0.012
        and top_height <= 0.038
        and normal_axis_alignment >= 0.80
    ):
        fit = dict(fit)
        fit["features"] = {
            **dict(fit.get("features", {})),
            "closed_set_override": "table_supported_cuboid_axis_aligned",
        }
        rotation = _table_frame(rotation, table_normal)
        warnings.append("table_supported_cuboid_axis_aligned")

    if (
        min_height <= 0.012
        and normal_axis_alignment >= 0.72
        and selected in {"cube", "cuboid"}
        and 0.050 <= top_height <= 0.070
        and cylinder_score >= 0.12
        and axisymmetric_score >= 0.30
        and float(fit.get("features", {}).get("box_surface_error", 1.0)) >= 0.10
    ):
        fit = dict(fit)
        fit["class"] = "cylinder"
        fit["confidence"] = max(0.40, cylinder_score)
        catalog_size = get_cad_dimensions(get_shape_spec("cylinder").cad_id)
        fit["geometry"] = {
            "diameter_m": float(catalog_size[0]),
            "height_m": float(catalog_size[2]),
        }
        fit["features"] = {
            **dict(fit.get("features", {})),
            "closed_set_override": "table_supported_top_height_cylinder",
        }
        rotation = _table_frame(rotation, table_normal)
        warnings.append("table_supported_top_height_cylinder_override")
        selected = "cylinder"

    if (
        selected == "cylinder"
        and cone_slope_strength >= 0.55
        and cone_normal_alignment >= 0.80
        and cone_fit_error <= 0.90 * max(cylinder_fit_error, 1e-9)
        and float(candidates.get("cone", 0.0)) >= 0.10
    ):
        fit = dict(fit)
        fit["class"] = "cone"
        fit["confidence"] = max(0.38, float(candidates.get("cone", 0.0)))
        catalog_size = get_cad_dimensions(get_shape_spec("cone").cad_id)
        fit["geometry"] = {
            "base_diameter_m": float(catalog_size[0]),
            "top_diameter_m": 0.0,
            "height_m": float(catalog_size[2]),
        }
        fit["features"] = {
            **dict(fit.get("features", {})),
            "closed_set_override": "cone_slope_and_normal_override",
        }
        rotation = _table_frame(rotation, table_normal)
        warnings.append("cone_slope_and_normal_override")
        selected = "cone"
    if (
        selected in {"cube", "cuboid"}
        and not bool(fit.get("features", {}).get("planar_observation", False))
        and cylinder_score >= 0.12
        and cylinder_dimension >= 0.95 * max(box_dimension, 1e-9)
        and axisymmetric_score >= 0.85
        and float(fit.get("features", {}).get("cylinder_side_normal_alignment", 0.0)) >= 0.45
        and (
            float(fit.get("features", {}).get("box_surface_error", 1.0)) >= 0.14
            or cylinder_dimension >= 1.15 * max(box_dimension, 1e-9)
        )
        and float(fit.get("features", {}).get("sphere_fit_error", 1.0)) > 0.06
    ):
        fit = dict(fit)
        fit["class"] = "cylinder"
        fit["confidence"] = max(0.38, cylinder_score)
        catalog_size = get_cad_dimensions(get_shape_spec("cylinder").cad_id)
        fit["geometry"] = {
            "diameter_m": float(catalog_size[0]),
            "height_m": float(catalog_size[2]),
        }
        fit["features"] = {
            **dict(fit.get("features", {})),
            "closed_set_override": "round_catalog_dimension_match",
        }
        rotation = _table_frame(rotation, table_normal)
        warnings.append("round_catalog_dimension_override")
        selected = "cylinder"

    if (
        selected in {"cube", "cuboid"}
        and not bool(fit.get("features", {}).get("planar_observation", False))
        and cylinder_score >= 0.16
        and min_height <= 0.012
        and 0.045 <= span_height <= 0.075
        and cylinder_score >= 0.78 * max(float(fit.get("confidence", 0.0)), 1e-9)
    ):
        fit = dict(fit)
        fit["class"] = "cylinder"
        fit["confidence"] = max(0.40, cylinder_score)
        catalog_size = get_cad_dimensions(get_shape_spec("cylinder").cad_id)
        fit["geometry"] = {
            "diameter_m": float(catalog_size[0]),
            "height_m": float(catalog_size[2]),
        }
        fit["features"] = {
            **dict(fit.get("features", {})),
            "closed_set_override": "table_supported_tall_round",
        }
        rotation = _table_frame(rotation, table_normal)
        warnings.append("table_supported_tall_round_override")

    axis_aligned_round = (
        str(fit["class"]) in {"cylinder", "cone"}
        and abs(float(np.dot(rotation[:, 2], table_normal))) >= 0.75
        and (
            min_height <= (0.030 if str(fit["class"]) == "cylinder" else 0.012)
            or (str(fit["class"]) == "cylinder" and span_height >= 0.045)
        )
    )
    if axis_aligned_round:
        rotation = _table_frame(rotation, table_normal)
        warnings.append("table_supported_axis_aligned")
        if str(fit["class"]) == "cylinder":
            refined_center, refined_ok = _refine_upright_cylinder_center(
                points,
                center,
                rotation,
                table_normal,
            )
            if refined_ok:
                center = refined_center
                warnings.append("cylinder_center_refined_from_fixed_radius_arc")
            # A cylinder stacked above another object is not table-supported,
            # so recover its height from the complete observed vertical span.
            # For a table-supported cylinder the later support rule replaces
            # this midpoint with the catalog bottom height.
            if span_height >= 0.045 and min_height > 0.030:
                observed_mid_height = 0.5 * (
                    float(np.percentile(heights, 1.0))
                    + float(np.percentile(heights, 99.0))
                )
                current_height = float(np.dot(center, table_normal) + plane_offset)
                if abs(current_height - observed_mid_height) > 0.002:
                    center = center + table_normal * (
                        observed_mid_height - current_height
                    )
                    warnings.append("cylinder_center_recovered_from_height_span")

    # A cluster can contain only upper surfaces. If its visible low point is
    # close to the table, the catalog bottom gives a better center than the
    # centroid of the visible pixels. This also corrects tilted bridge beams.
    if selected in {"cube", "cuboid", "cylinder", "cone"} and min_height <= (
        0.030 if str(fit["class"]) == "cylinder" else 0.012
    ):
        dimensions = np.asarray(
            get_cad_dimensions(get_shape_spec(str(fit["class"])).cad_id),
            dtype=np.float64,
        )
        vertical_half_extent = float(
            np.sum(np.abs(table_normal @ rotation) * (0.5 * dimensions))
        )
        target_center_height = vertical_half_extent
        current_center_height = float(np.dot(center, table_normal) + plane_offset)
        if abs(current_center_height - target_center_height) > 0.003:
            center = center + table_normal * (target_center_height - current_center_height)
            warnings.append("center_recovered_from_table_support")
    return fit, center, rotation, warnings


def main() -> None:
    scene, segmentation = _load_context()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    file_names = list(segmentation.get("cluster_files", []))
    if not file_names:
        raise RuntimeError("分割元数据中没有cluster_files。")

    results: list[dict] = []
    geometries: list[object] = []
    palette = [
        [0.85, 0.25, 0.25],
        [0.25, 0.70, 0.30],
        [0.25, 0.40, 0.90],
        [0.90, 0.75, 0.20],
        [0.75, 0.30, 0.80],
    ]
    table_plane = np.asarray(segmentation.get("table_plane", []), dtype=np.float64)
    if table_plane.shape != (4,):
        table_plane = None
    catalog_locked = str(scene.get("scene_mode", "")).lower() == "planned"

    for index, file_name in enumerate(file_names):
        cloud = _load_cluster(str(file_name))
        points = np.asarray(cloud.points, dtype=np.float64)
        fit = fit_primitive_candidates(
            points,
            top_k=PRIMITIVE_TOP_K,
            catalog_locked=catalog_locked,
        )
        center, rotation, pose_warnings = _complete_planar_pose(
            fit,
            points,
            table_plane,
        )
        fit, center, rotation, ambiguity_warnings = _resolve_closed_set_ambiguity(
            fit,
            points,
            center,
            rotation,
            table_plane,
        )
        cube_pose_warnings: list[str] = []
        if (
            table_plane is not None
            and str(fit.get("class", "")) == "cube"
            and bool(fit.get("features", {}).get("closed_set_override", ""))
        ):
            center, rotation, cube_refined = _refine_table_cube_pose(
                points,
                center,
                rotation,
                np.asarray(table_plane[:3], dtype=np.float64),
            )
            if cube_refined:
                cube_pose_warnings.append("cube_top_footprint_pose_refined")
        closed_set_override = str(fit.get("features", {}).get("closed_set_override", ""))
        if (
            bool(fit.get("features", {}).get("planar_observation", False))
            or closed_set_override in {
                "table_supported_cube_top_height",
                "table_supported_top_height_box",
                "table_supported_cuboid_axis_aligned",
            }
        ):
            box_warnings = []
        else:
            center, rotation, box_warnings = _refine_box_pose(
                points,
                center,
                rotation,
                str(fit["class"]),
                catalog_locked,
            )
        center, clearance_warnings = _enforce_table_clearance(
            fit,
            center,
            rotation,
            table_plane,
        )
        warnings = _quality_warnings({**fit, "point_count": len(points)})
        warnings.extend(pose_warnings)
        warnings.extend(box_warnings)
        warnings.extend(cube_pose_warnings)
        warnings.extend(ambiguity_warnings)
        warnings.extend(clearance_warnings)
        result = {
            "id": index,
            "source": str(file_name),
            "class": fit["class"],
            "confidence": float(fit["confidence"]),
            "center_m": center.tolist(),
            "rotation_matrix": rotation.tolist(),
            "yaw_deg": None,
            "geometry": fit["geometry"],
            "features": fit["features"],
            "candidates": fit["candidates"],
            "point_count": int(len(points)),
            "quality_warnings": sorted(set(warnings)),
            "pose_valid": not any(
                item in {"planar_side_face_pose_ambiguous"}
                for item in warnings
            ),
        }
        results.append(result)

        print(
            f"Object {index}: {result['class']:8s} "
            f"conf={result['confidence']:.3f} "
            f"P={np.round(center, 4)}"
        )
        print(
            "  candidates="
            + ", ".join(
                f"{item['class']}:{item['score']:.3f}"
                for item in result["candidates"]
            )
        )

        if SHOW_VISUALIZATION:
            display = o3d.geometry.PointCloud(cloud)
            display.paint_uniform_color(palette[index % len(palette)])
            geometries.append(display)
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.035)
            transform = np.eye(4, dtype=np.float64)
            transform[:3, :3] = rotation
            transform[:3, 3] = center
            frame.transform(transform)
            geometries.append(frame)

    results = annotate_grasp_planning(results)
    payload = {
        "metadata": {
            "scene_id": scene["scene_id"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "segmentation_cluster_count": int(segmentation["cluster_count"]),
            "recognizer": "primitive_fitting_3d_v2_multiview",
            "observation_view_count": int(
                segmentation.get("views", {}).get("view_count", 1)
            ),
            "segmentation_method": str(segmentation.get("segmentation_method", "unknown")),
            "uses_color_features": False,
        },
        "objects": results,
    }
    RESULT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"识别结果已保存：{RESULT_JSON.resolve()}")

    if SHOW_VISUALIZATION:
        geometries.append(
            o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.10)
        )
        o3d.visualization.draw_geometries(
            geometries,
            window_name="3-D Primitive Candidate Recognition",
            width=1200,
            height=800,
        )


if __name__ == "__main__":
    main()
