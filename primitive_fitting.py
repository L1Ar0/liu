"""Closed-set 3-D primitive classification and pose initialization.

The recognizer combines surface fits, normal/curvature cues and the finite
catalog dimensions. It also detects near-planar partial observations so a
single visible box face cannot degenerate into an enormous fitted sphere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull, QhullError, cKDTree

from shape_catalog import dimensions_to_geometry, get_cad_dimensions, get_shape_spec


EPS = 1e-9
PLANAR_EIGENVALUE_RATIO_MAX = 0.012
PLANAR_THICKNESS_RATIO_MAX = 0.060
PLANAR_SHAPE_MARGIN = 0.10


@dataclass
class AxisFit:
    axis: np.ndarray
    center: np.ndarray
    radius: float
    height: float
    residual: float
    slope: float = 0.0
    base_radius: float = 0.0
    top_radius: float = 0.0


def _normalize(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm < EPS:
        raise ValueError("Cannot normalize a zero vector")
    return value / norm


def _right_handed(rotation: np.ndarray) -> np.ndarray:
    result = np.asarray(rotation, dtype=np.float64).copy()
    if np.linalg.det(result) < 0.0:
        result[:, -1] *= -1.0
    return result


def pca_box(points: np.ndarray) -> dict:
    points = np.asarray(points, dtype=np.float64)
    center = points.mean(axis=0)
    covariance = np.cov((points - center).T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    axes = _right_handed(eigenvectors[:, order])

    near_planar = bool(
        max(float(eigenvalues[0]), 0.0) / max(float(eigenvalues[-1]), EPS)
        <= PLANAR_EIGENVALUE_RATIO_MAX
    )
    try:
        if near_planar:
            raise RuntimeError("skip robust OBB for a planar observation")
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        oriented_box = cloud.get_minimal_oriented_bounding_box(robust=True)
        box_center = np.asarray(oriented_box.center, dtype=np.float64)
        axes = _right_handed(np.asarray(oriented_box.R, dtype=np.float64))
        extent = np.asarray(oriented_box.extent, dtype=np.float64)
    except Exception:
        projected = (points - center) @ axes
        minimum = projected.min(axis=0)
        maximum = projected.max(axis=0)
        extent = maximum - minimum
        local_center = 0.5 * (minimum + maximum)
        box_center = center + axes @ local_center

    extent_order = np.argsort(extent)[::-1]
    extent = extent[extent_order]
    axes = _right_handed(axes[:, extent_order])

    local = (points - box_center) @ axes
    half = np.maximum(0.5 * extent, EPS)
    nearest_face = np.min(np.abs(np.abs(local) - half), axis=1)
    surface_error = float(np.sqrt(np.mean(nearest_face**2)) / max(float(half.min()), EPS))

    return {
        "center": box_center,
        "rotation": axes,
        "extent": extent,
        "surface_error": surface_error,
        "eigenvalues": eigenvalues[order],
    }


def fit_sphere(points: np.ndarray) -> dict:
    points = np.asarray(points, dtype=np.float64)
    matrix = np.column_stack([2.0 * points, np.ones(len(points))])
    target = np.sum(points * points, axis=1)
    solution, *_ = np.linalg.lstsq(matrix, target, rcond=None)
    center = solution[:3]
    radius_sq = float(solution[3] + np.dot(center, center))
    radius = math.sqrt(max(radius_sq, EPS))
    distances = np.linalg.norm(points - center, axis=1)
    residual = float(np.sqrt(np.mean((distances - radius) ** 2)) / radius)
    radial_vectors = points - center
    radial_norms = np.linalg.norm(radial_vectors, axis=1)
    valid = radial_norms > EPS
    radial_directions = radial_vectors[valid] / radial_norms[valid, None]
    coverage = float(np.linalg.det(np.cov(radial_directions.T))) if len(radial_directions) >= 4 else 0.0
    return {
        "center": center,
        "radius": radius,
        "residual": residual,
        "coverage": max(coverage, 0.0),
    }


def _point_to_polygon_edges(points: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Return the shortest 2-D distance from each point to a polygon edge."""

    starts = vertices
    ends = np.roll(vertices, -1, axis=0)
    segments = ends - starts
    segment_norm_sq = np.sum(segments * segments, axis=1)
    result = np.full(len(points), np.inf, dtype=np.float64)
    for start, segment, norm_sq in zip(starts, segments, segment_norm_sq):
        if norm_sq < EPS:
            continue
        relative = points - start
        ratio = np.clip((relative @ segment) / norm_sq, 0.0, 1.0)
        projection = start + ratio[:, None] * segment
        result = np.minimum(result, np.linalg.norm(points - projection, axis=1))
    return result


def _circle_boundary_error(points: np.ndarray) -> float:
    if len(points) < 5:
        return 1.0
    matrix = np.column_stack([2.0 * points, np.ones(len(points))])
    target = np.sum(points * points, axis=1)
    solution, *_ = np.linalg.lstsq(matrix, target, rcond=None)
    center = solution[:2]
    radius_sq = float(solution[2] + np.dot(center, center))
    radius = math.sqrt(max(radius_sq, EPS))
    radial = np.linalg.norm(points - center, axis=1)
    return float(np.sqrt(np.mean((radial - radius) ** 2)) / radius)


def _rectangle_boundary_error(points: np.ndarray, vertices: np.ndarray) -> tuple[float, float]:
    """Fit a minimum-error oriented rectangle to boundary samples."""

    best_error = float("inf")
    best_area = float("inf")
    for start, end in zip(vertices, np.roll(vertices, -1, axis=0)):
        edge = end - start
        edge_norm = float(np.linalg.norm(edge))
        if edge_norm < EPS:
            continue
        axis_u = edge / edge_norm
        axis_v = np.asarray([-axis_u[1], axis_u[0]], dtype=np.float64)
        projected = np.column_stack([points @ axis_u, points @ axis_v])
        low = projected.min(axis=0)
        high = projected.max(axis=0)
        extent = np.maximum(high - low, EPS)
        edge_distance = np.min(
            np.column_stack(
                [
                    np.abs(projected[:, 0] - low[0]),
                    np.abs(projected[:, 0] - high[0]),
                    np.abs(projected[:, 1] - low[1]),
                    np.abs(projected[:, 1] - high[1]),
                ]
            ),
            axis=1,
        )
        error = float(np.sqrt(np.mean(edge_distance * edge_distance)) / min(extent))
        if error < best_error:
            best_error = error
            best_area = float(np.prod(extent))
    return best_error, best_area


def _planar_descriptors(points: np.ndarray, box: dict) -> dict[str, object]:
    """Describe a thin observed surface in its local plane."""

    rotation = np.asarray(box["rotation"], dtype=np.float64)
    center = np.asarray(box["center"], dtype=np.float64)
    extent = np.maximum(np.asarray(box["extent"], dtype=np.float64), EPS)
    eigenvalues = np.maximum(
        np.asarray(box["eigenvalues"], dtype=np.float64),
        0.0,
    )
    eigenvalue_ratio = float(eigenvalues[-1] / max(eigenvalues[0], EPS))
    thickness_ratio = float(extent[-1] / max(extent[0], EPS))
    is_planar = bool(
        eigenvalue_ratio <= PLANAR_EIGENVALUE_RATIO_MAX
        and thickness_ratio <= PLANAR_THICKNESS_RATIO_MAX
    )

    uv = (np.asarray(points, dtype=np.float64) - center) @ rotation[:, :2]
    low = np.percentile(uv, 1.0, axis=0)
    high = np.percentile(uv, 99.0, axis=0)
    plane_extent = np.maximum(high - low, EPS)
    plane_extent = np.sort(plane_extent)[::-1]

    circularity = 0.0
    rectangularity = 0.0
    circle_boundary_error = 1.0
    rectangle_boundary_error = 1.0
    boundary_point_count = 0
    if len(uv) >= 4:
        try:
            hull = ConvexHull(uv)
            vertices = uv[np.asarray(hull.vertices, dtype=int)]
            shifted = np.roll(vertices, -1, axis=0)
            perimeter = float(np.sum(np.linalg.norm(shifted - vertices, axis=1)))
            area = float(hull.volume)
            circularity = float(
                np.clip(4.0 * math.pi * area / max(perimeter * perimeter, EPS), 0.0, 1.0)
            )
            boundary_distance = _point_to_polygon_edges(uv, vertices)
            tolerance = max(0.0015, 0.055 * float(np.min(plane_extent)))
            boundary = uv[boundary_distance <= tolerance]
            boundary_point_count = int(len(boundary))
            if len(boundary) >= 5:
                circle_boundary_error = _circle_boundary_error(boundary)
                rectangle_boundary_error, rectangle_area = _rectangle_boundary_error(
                    boundary,
                    vertices,
                )
                rectangularity = float(
                    np.clip(area / max(rectangle_area, EPS), 0.0, 1.0)
                )
            else:
                rectangularity = float(
                    np.clip(area / max(float(np.prod(plane_extent)), EPS), 0.0, 1.0)
                )
        except QhullError:
            pass

    return {
        "is_planar": is_planar,
        "normal": rotation[:, 2].copy(),
        "extent": plane_extent,
        "eigenvalue_ratio": eigenvalue_ratio,
        "thickness_ratio": thickness_ratio,
        "circularity": circularity,
        "rectangularity": rectangularity,
        "circle_boundary_error": circle_boundary_error,
        "rectangle_boundary_error": rectangle_boundary_error,
        "boundary_point_count": boundary_point_count,
    }


def _catalog_dimension_score(
    class_name: str,
    observed_extent: np.ndarray,
    planar: dict[str, object],
    catalog_locked: bool,
) -> float:
    expected = np.asarray(
        get_cad_dimensions(get_shape_spec(class_name).cad_id),
        dtype=np.float64,
    )
    observed = np.sort(np.maximum(np.asarray(observed_extent, dtype=np.float64), EPS))[::-1]
    if bool(planar["is_planar"]):
        visible = np.asarray(planar["extent"], dtype=np.float64)
        normal = np.asarray(planar["normal"], dtype=np.float64)
        if abs(float(normal[2])) >= 0.75:
            target = np.sort(expected[:2])[::-1]
            relative_error = np.abs(visible - target) / np.maximum(target, EPS)
        else:
            pair_errors: list[float] = []
            for first in range(3):
                for second in range(first + 1, 3):
                    target = np.sort(expected[[first, second]])[::-1]
                    pair_errors.append(
                        float(np.mean(np.abs(visible - target) / np.maximum(target, EPS)))
                    )
            relative_error = np.asarray([min(pair_errors), min(pair_errors)])
    else:
        target = np.sort(expected)[::-1]
        relative_error = np.abs(observed - target) / np.maximum(target, EPS)
    scale = 0.28 if catalog_locked else 0.55
    return _exp_score(float(np.mean(relative_error)), scale)


def _axis_candidates(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    covariance = np.cov((points - center).T)
    _, eigenvectors = np.linalg.eigh(covariance)
    return center, _right_handed(eigenvectors)


def fit_cylinder(points: np.ndarray) -> AxisFit:
    points = np.asarray(points, dtype=np.float64)
    origin, axes = _axis_candidates(points)
    best: AxisFit | None = None
    best_cost = float("inf")

    for axis_index in range(3):
        axis = _normalize(axes[:, axis_index])
        axial = (points - origin) @ axis
        low, high = np.percentile(axial, [1.0, 99.0])
        height = max(float(high - low), EPS)
        axis_center = origin + 0.5 * (low + high) * axis
        relative = points - axis_center
        axial_centered = relative @ axis
        radial_vectors = relative - axial_centered[:, None] * axis
        radii = np.linalg.norm(radial_vectors, axis=1)
        # A depth cluster contains both the cylindrical side and points from
        # the two caps. Estimate the side radius from the upper quantile and
        # measure each point against its nearest valid surface (side or cap).
        radius = max(float(np.percentile(radii, 90.0)), EPS)
        side_error = np.abs(radii - radius)
        cap_error = np.minimum(np.abs(axial - low), np.abs(axial - high))
        cap_valid = radii <= radius * 1.08
        surface_error = np.where(cap_valid, np.minimum(side_error, cap_error), side_error)
        cutoff = float(np.quantile(surface_error, 0.90))
        trimmed = surface_error[surface_error <= cutoff]
        residual = float(
            np.sqrt(np.mean(trimmed * trimmed) if len(trimmed) else np.mean(surface_error * surface_error))
            / radius
        )
        slope = float(
            np.linalg.lstsq(
                np.column_stack([axial_centered, np.ones_like(axial_centered)]),
                radii,
                rcond=None,
            )[0][0]
        )
        slope_strength = abs(slope) * height / radius
        aspect_ratio = height / max(2.0 * radius, EPS)
        # The catalog cylinders are elongated along their symmetry axis.
        # This prevents a visible circular cap from being mistaken for the
        # cylinder axis, while the slope term rejects cone-like profiles.
        cost = (
            residual
            + 0.20 * abs(math.log(max(aspect_ratio, EPS) / 1.50))
            + 0.10 * slope_strength
        )
        candidate = AxisFit(axis, axis_center, radius, height, residual, slope=slope)
        if best is None or cost < best_cost:
            best = candidate
            best_cost = cost

    assert best is not None
    return best


def fit_cone(points: np.ndarray) -> AxisFit:
    points = np.asarray(points, dtype=np.float64)
    origin, axes = _axis_candidates(points)
    best: AxisFit | None = None
    best_cost = float("inf")

    for axis_index in range(3):
        raw_axis = _normalize(axes[:, axis_index])
        for sign in (-1.0, 1.0):
            axis = raw_axis * sign
            axial = (points - origin) @ axis
            low, high = np.percentile(axial, [1.0, 99.0])
            height = max(float(high - low), EPS)
            axial_mid = 0.5 * (low + high)
            axis_center = origin + axial_mid * axis
            relative = points - axis_center
            t = relative @ axis
            radial_vectors = relative - t[:, None] * axis
            radii = np.linalg.norm(radial_vectors, axis=1)

            design = np.column_stack([t, np.ones_like(t)])
            slope, intercept = np.linalg.lstsq(design, radii, rcond=None)[0]
            predicted = slope * t + intercept
            scale = max(float(np.mean(radii)), EPS)
            residual = float(np.sqrt(np.mean((radii - predicted) ** 2)) / scale)
            slope_strength = abs(float(slope)) * height / scale
            # Reject the near-zero slope preferred by cylindrical surfaces.
            cost = residual + 0.18 * max(0.0, 0.35 - slope_strength)

            bottom_t = -0.5 * height
            top_t = 0.5 * height
            radius_a = max(float(slope * bottom_t + intercept), 0.0)
            radius_b = max(float(slope * top_t + intercept), 0.0)
            candidate = AxisFit(
                axis=axis,
                center=axis_center,
                radius=max(radius_a, radius_b),
                height=height,
                residual=residual,
                slope=float(slope),
                base_radius=max(radius_a, radius_b),
                top_radius=min(radius_a, radius_b),
            )
            if cost < best_cost:
                best = candidate
                best_cost = cost

    assert best is not None
    # Point +Z from the wider end to the narrower end for a stable cone frame.
    if best.slope > 0.0:
        best.axis *= -1.0
        best.slope *= -1.0
    return best


def axis_rotation(axis: np.ndarray, reference_rotation: np.ndarray) -> np.ndarray:
    z_axis = _normalize(axis)
    reference = reference_rotation[:, 0]
    x_axis = reference - np.dot(reference, z_axis) * z_axis
    if np.linalg.norm(x_axis) < 1e-6:
        reference = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(reference, z_axis))) > 0.9:
            reference = np.array([0.0, 1.0, 0.0])
        x_axis = reference - np.dot(reference, z_axis) * z_axis
    x_axis = _normalize(x_axis)
    y_axis = _normalize(np.cross(z_axis, x_axis))
    x_axis = _normalize(np.cross(y_axis, z_axis))
    return _right_handed(np.column_stack([x_axis, y_axis, z_axis]))


def _exp_score(error: float, scale: float) -> float:
    return float(np.clip(math.exp(-max(error, 0.0) / max(scale, EPS)), 0.0, 1.0))


def _surface_descriptors(
    points: np.ndarray,
    box: dict,
    sphere: dict,
    cylinder: AxisFit,
    cone: AxisFit,
) -> dict[str, float]:
    """Estimate normal/curvature cues used as weak class evidence.

    RGB-D clusters are often partial surfaces, so these descriptors are
    deliberately soft features. Every value has a finite fallback when a
    cluster is too sparse or Open3D cannot estimate normals.
    """

    points = np.asarray(points, dtype=np.float64)
    normals = np.empty((0, 3), dtype=np.float64)
    if len(points) >= 20:
        try:
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(points)
            extent = np.ptp(points, axis=0)
            radius = max(float(np.max(extent)) * 0.18, 0.002)
            cloud.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=radius,
                    max_nn=min(40, len(points)),
                )
            )
            normals = np.asarray(cloud.normals, dtype=np.float64)
            norms = np.linalg.norm(normals, axis=1)
            normals = normals[norms > EPS] / norms[norms > EPS, None]
        except Exception:
            normals = np.empty((0, 3), dtype=np.float64)

    normal_axis_alignment = 0.0
    sphere_radial_alignment = 0.0
    cylinder_side_alignment = 0.0
    cone_surface_alignment = 0.0
    if len(normals) and len(normals) == len(points):
        box_axes = np.asarray(box["rotation"], dtype=np.float64)
        normal_axis_alignment = float(
            np.mean(np.max(np.abs(normals @ box_axes), axis=1))
        )

        radial = points - np.asarray(sphere["center"], dtype=np.float64)
        radial_norm = np.linalg.norm(radial, axis=1)
        valid = radial_norm > EPS
        if np.any(valid):
            radial_unit = radial[valid] / radial_norm[valid, None]
            sphere_radial_alignment = float(
                np.mean(np.abs(np.sum(normals[valid] * radial_unit, axis=1)))
            )

        cylinder_axis = _normalize(cylinder.axis)
        cylinder_side_alignment = float(
            np.mean(1.0 - np.abs(normals @ cylinder_axis))
        )

        cone_axis = _normalize(cone.axis)
        relative = points - np.asarray(cone.center, dtype=np.float64)
        axial = relative @ cone_axis
        radial_vec = relative - axial[:, None] * cone_axis
        radial_norm = np.linalg.norm(radial_vec, axis=1)
        valid = radial_norm > EPS
        if np.any(valid):
            radial_unit = radial_vec[valid] / radial_norm[valid, None]
            expected = radial_unit - float(cone.slope) * cone_axis
            expected /= np.maximum(np.linalg.norm(expected, axis=1, keepdims=True), EPS)
            cone_surface_alignment = float(
                np.mean(np.abs(np.sum(normals[valid] * expected, axis=1)))
            )

    curvature_values = np.empty(0, dtype=np.float64)
    if len(points) >= 24:
        try:
            tree = cKDTree(points)
            sample_count = min(len(points), 512)
            sample_indices = np.linspace(0, len(points) - 1, sample_count, dtype=int)
            k = min(24, len(points))
            _, neighbor_indices = tree.query(points[sample_indices], k=k)
            values: list[float] = []
            for neighbors in np.atleast_2d(neighbor_indices):
                local = points[np.asarray(neighbors, dtype=int)]
                local -= local.mean(axis=0)
                covariance = (local.T @ local) / max(len(local) - 1, 1)
                eigenvalues = np.linalg.eigvalsh(covariance)
                total = float(np.sum(np.maximum(eigenvalues, 0.0)))
                values.append(float(max(eigenvalues[0], 0.0) / max(total, EPS)))
            curvature_values = np.asarray(values, dtype=np.float64)
        except Exception:
            curvature_values = np.empty(0, dtype=np.float64)

    return {
        "normal_axis_alignment": float(np.clip(normal_axis_alignment, 0.0, 1.0)),
        "sphere_radial_normal_alignment": float(
            np.clip(sphere_radial_alignment, 0.0, 1.0)
        ),
        "cylinder_side_normal_alignment": float(
            np.clip(cylinder_side_alignment, 0.0, 1.0)
        ),
        "cone_surface_normal_alignment": float(
            np.clip(cone_surface_alignment, 0.0, 1.0)
        ),
        "curvature_mean": float(np.mean(curvature_values)) if len(curvature_values) else 0.0,
        "curvature_std": float(np.std(curvature_values)) if len(curvature_values) else 0.0,
    }


def fit_primitive_candidates(
    points: np.ndarray,
    top_k: int = 3,
    catalog_locked: bool = True,
) -> dict:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 20:
        raise ValueError("At least 20 XYZ points are required for primitive fitting")

    box = pca_box(points)
    sphere = fit_sphere(points)
    cylinder = fit_cylinder(points)
    cone = fit_cone(points)
    surface = _surface_descriptors(points, box, sphere, cylinder, cone)
    planar = _planar_descriptors(points, box)

    extent = np.maximum(np.asarray(box["extent"], dtype=np.float64), EPS)
    cube_ratio = float(extent.max() / extent.min())
    local_ellipsoid = (points - box["center"]) @ box["rotation"]
    ellipsoid_radius = np.sqrt(
        np.sum((local_ellipsoid / np.maximum(0.5 * extent, EPS)) ** 2, axis=1)
    )
    ellipsoid_error = float(np.sqrt(np.mean((ellipsoid_radius - 1.0) ** 2)))
    box_score = _exp_score(float(box["surface_error"]), 0.12)
    box_score *= 0.65 + 0.35 * surface["normal_axis_alignment"]
    cube_shape_score = _exp_score(abs(math.log(cube_ratio)), 0.25)
    cube_score = box_score * cube_shape_score
    cuboid_score = box_score * (0.20 + 0.80 * (1.0 - cube_shape_score))
    sphere_score = _exp_score(float(sphere["residual"]), 0.055)
    sphere_score *= 0.75 + 0.25 * surface["sphere_radial_normal_alignment"]
    # A spheroid can have only a modest axis ratio; do not require the very
    # elongated OBB that a partially visible sphere often produces.
    spheroid_anisotropy = float(np.clip((cube_ratio - 1.05) / 0.30, 0.0, 1.0))
    ellipsoid_advantage = 1.0 / (
        1.0 + math.exp((float(ellipsoid_error) - float(sphere["residual"])) / 0.025)
    )
    spheroid_score = (
        _exp_score(ellipsoid_error, 0.12)
        * spheroid_anisotropy
        * ellipsoid_advantage
    )
    cross_section_similarity = _exp_score(
        abs(math.log(float(extent[1] / extent[2]))),
        0.20,
    )
    cylinder_slope_strength = (
        abs(cylinder.slope) * cylinder.height / max(cylinder.radius, EPS)
    )
    cylinder_score = _exp_score(cylinder.residual, 0.18)
    cylinder_score *= _exp_score(cylinder_slope_strength, 0.45)
    cylinder_score *= 0.80 + 0.20 * surface["cylinder_side_normal_alignment"]
    cylinder_score *= 0.65 + 0.35 * cross_section_similarity
    cone_slope_strength = abs(cone.slope) * cone.height / max(cone.radius, EPS)
    cone_score = _exp_score(cone.residual, 0.11) * float(
        np.clip((cone_slope_strength - 0.15) / 0.85, 0.0, 1.0)
    )
    cone_score *= 0.80 + 0.20 * surface["cone_surface_normal_alignment"]
    cone_score *= 0.65 + 0.35 * cross_section_similarity

    dimension_scores = {
        class_name: _catalog_dimension_score(
            class_name,
            extent,
            planar,
            catalog_locked,
        )
        for class_name in ("cube", "cuboid", "cylinder", "sphere", "spheroid", "cone")
    }
    dimension_weight = 0.32 if catalog_locked else 0.12
    shape_weight = 1.0 - dimension_weight
    cube_score = shape_weight * cube_score + dimension_weight * dimension_scores["cube"]
    cuboid_score = shape_weight * cuboid_score + dimension_weight * dimension_scores["cuboid"]
    cylinder_score = shape_weight * cylinder_score + dimension_weight * dimension_scores["cylinder"]
    sphere_score = shape_weight * sphere_score + dimension_weight * dimension_scores["sphere"]
    spheroid_score = shape_weight * spheroid_score + dimension_weight * dimension_scores["spheroid"]
    cone_score = shape_weight * cone_score + dimension_weight * dimension_scores["cone"]

    if bool(planar["is_planar"]):
        # A sphere fitted to a small plane has a huge radius and an artificially
        # tiny residual. Remove that ill-conditioned solution entirely.
        sphere_score = 0.0
        spheroid_score = 0.0
        circularity = float(planar["circularity"])
        rectangularity = float(planar["rectangularity"])
        circle_error = float(planar["circle_boundary_error"])
        rectangle_error = float(planar["rectangle_boundary_error"])
        plane_ratio = float(
            np.asarray(planar["extent"], dtype=np.float64)[0]
            / max(np.asarray(planar["extent"], dtype=np.float64)[1], EPS)
        )
        circle_fit_score = _exp_score(circle_error, 0.085)
        rectangle_fit_score = _exp_score(rectangle_error, 0.055)
        round_evidence = (
            0.60 * circle_fit_score
            + 0.40 * circularity
        )
        box_evidence = (
            0.60 * rectangle_fit_score
            + 0.40 * rectangularity
        )
        if (
            plane_ratio <= 1.25
            and round_evidence >= box_evidence + PLANAR_SHAPE_MARGIN
        ):
            cylinder_score = max(
                cylinder_score,
                round_evidence * dimension_scores["cylinder"],
            )
            cone_score *= 0.35
            cube_score *= 0.60
            cuboid_score *= 0.60
        elif box_evidence >= round_evidence + PLANAR_SHAPE_MARGIN:
            cube_score = max(
                cube_score,
                box_evidence * dimension_scores["cube"],
            )
            cuboid_score = max(
                cuboid_score,
                box_evidence * dimension_scores["cuboid"],
            )
            cylinder_score *= 0.35
            cone_score *= 0.20
        else:
            # A top-only square and circular cap can be genuinely ambiguous at
            # one view.  Preserve both candidates so the second view can add
            # side-surface evidence instead of forcing a brittle hard label.
            cube_score = max(cube_score, 0.75 * dimension_scores["cube"])
            cuboid_score = max(cuboid_score, 0.65 * dimension_scores["cuboid"])
            cylinder_score = max(
                cylinder_score,
                0.70 * dimension_scores["cylinder"],
            )
            cone_score *= 0.40

    raw_scores = {
        "cube": cube_score,
        "cuboid": cuboid_score,
        "cylinder": cylinder_score,
        "sphere": sphere_score,
        "spheroid": spheroid_score,
        "cone": cone_score,
    }
    total = max(sum(raw_scores.values()), EPS)
    normalized_scores = {name: float(score / total) for name, score in raw_scores.items()}
    candidates = sorted(
        (
            {
                "class": name,
                "score": score,
                "cad_id": get_shape_spec(name).cad_id,
            }
            for name, score in normalized_scores.items()
        ),
        key=lambda item: item["score"],
        reverse=True,
    )[: max(1, int(top_k))]

    selected = str(candidates[0]["class"])
    catalog_extent = np.asarray(
        get_cad_dimensions(get_shape_spec(selected).cad_id),
        dtype=np.float64,
    )
    output_extent = catalog_extent if catalog_locked else extent
    if selected in {"cube", "cuboid"}:
        center = box["center"]
        rotation = box["rotation"]
        geometry = dimensions_to_geometry(selected, output_extent)
    elif selected == "sphere":
        center = sphere["center"]
        rotation = np.eye(3, dtype=np.float64)
        diameter = float(catalog_extent[0]) if catalog_locked else 2.0 * float(sphere["radius"])
        geometry = {"diameter_m": diameter}
    elif selected == "spheroid":
        center = box["center"]
        rotation = box["rotation"]
        geometry = {
            "axis_x_m": float(output_extent[0]),
            "axis_y_m": float(output_extent[1]),
            "axis_z_m": float(output_extent[2]),
        }
    elif selected == "cylinder":
        if bool(planar["is_planar"]):
            center = box["center"]
            rotation = axis_rotation(np.asarray(planar["normal"]), box["rotation"])
        else:
            center = cylinder.center
            rotation = axis_rotation(cylinder.axis, box["rotation"])
        geometry = {
            "diameter_m": float(catalog_extent[0]) if catalog_locked else 2.0 * cylinder.radius,
            "height_m": float(catalog_extent[2]) if catalog_locked else cylinder.height,
        }
    else:
        center = cone.center
        rotation = axis_rotation(cone.axis, box["rotation"])
        geometry = {
            "base_diameter_m": float(catalog_extent[0]) if catalog_locked else 2.0 * cone.base_radius,
            "top_diameter_m": 2.0 * cone.top_radius,
            "height_m": float(catalog_extent[2]) if catalog_locked else cone.height,
        }

    return {
        "class": selected,
        "confidence": float(candidates[0]["score"]),
        "center": np.asarray(center, dtype=np.float64),
        "rotation": np.asarray(rotation, dtype=np.float64),
        "geometry": geometry,
        "candidates": candidates,
        "features": {
            "obb_extent_m": extent.tolist(),
            "box_surface_error": float(box["surface_error"]),
            "sphere_fit_error": float(sphere["residual"]),
            "spheroid_fit_error": float(ellipsoid_error),
            "cylinder_fit_error": float(cylinder.residual),
            "cylinder_slope_strength": float(cylinder_slope_strength),
            "cone_fit_error": float(cone.residual),
            "cone_slope_strength": float(cone_slope_strength),
            "axisymmetric_cross_section_score": float(cross_section_similarity),
            "catalog_dimension_scores": dimension_scores,
            "catalog_locked": bool(catalog_locked),
            "planar_observation": bool(planar["is_planar"]),
            "planar_normal": np.asarray(planar["normal"], dtype=float).tolist(),
            "planar_extent_m": np.asarray(planar["extent"], dtype=float).tolist(),
            "planar_eigenvalue_ratio": float(planar["eigenvalue_ratio"]),
            "planar_thickness_ratio": float(planar["thickness_ratio"]),
            "planar_circularity": float(planar["circularity"]),
            "planar_rectangularity": float(planar["rectangularity"]),
            "planar_circle_boundary_error": float(planar["circle_boundary_error"]),
            "planar_rectangle_boundary_error": float(planar["rectangle_boundary_error"]),
            "planar_boundary_point_count": int(planar["boundary_point_count"]),
            **surface,
            "pca_eigenvalues": np.asarray(box["eigenvalues"], dtype=float).tolist(),
        },
    }
