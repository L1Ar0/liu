"""Geometry-only refinement for RGB-D instance segmentation.

The simulator may still provide RGB values for visualization, but this module
never uses color.  A spatial DBSCAN pass is followed by normal/curvature
over-segmentation and catalog-constrained agglomeration.  The latter is what
allows a contact region to be split without relying on simulator object IDs or
unique colors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from primitive_fitting import fit_primitive_candidates
from shape_catalog import CAD_DIMENSIONS_M


NORMAL_K = 24
NORMAL_GRAPH_RADIUS_M = 0.010
NORMAL_ANGLE_MAX_DEG = 58.0
NORMAL_PLANE_GAP_M = 0.0045
PATCH_MIN_POINTS = 12
PATCH_MERGE_GAP_M = 0.012
MIN_CLUSTER_POINTS = 24
# The largest catalog primitive is 70 mm long.  A single tilted object can
# enlarge an axis-aligned extent slightly, but a contact merge produces a
# substantially larger span.  This threshold is only a trigger for geometric
# model selection; it never uses the scene object count or simulator IDs.
CATALOG_MAX_AABB_SPAN_M = 1.25 * max(
    max(float(value) for value in dimensions)
    for dimensions in CAD_DIMENSIONS_M.values()
)
MODEL_SPLIT_MIN_GAIN = 0.075


@dataclass
class Patch:
    indices: np.ndarray
    points: np.ndarray
    normal: np.ndarray
    curvature: float


def _cloud_from_indices(
    cloud: o3d.geometry.PointCloud,
    indices: np.ndarray,
) -> o3d.geometry.PointCloud:
    result = cloud.select_by_index(np.asarray(indices, dtype=np.int64).tolist())
    return result


def _bbox_gap(first: np.ndarray, second: np.ndarray) -> float:
    first_min = np.min(first, axis=0)
    first_max = np.max(first, axis=0)
    second_min = np.min(second, axis=0)
    second_max = np.max(second, axis=0)
    gap = np.maximum(
        0.0,
        np.maximum(first_min - second_max, second_min - first_max),
    )
    return float(np.linalg.norm(gap))


def _estimate_normals_and_curvature(
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate unoriented local normals and a scale-free curvature cue."""

    points = np.asarray(points, dtype=np.float64)
    if len(points) < 6:
        normals = np.zeros_like(points)
        normals[:, 2] = 1.0
        return normals, np.ones(len(points), dtype=np.float64)

    tree = cKDTree(points)
    k = min(NORMAL_K, len(points))
    _, neighbours = tree.query(points, k=k)
    normals = np.zeros_like(points)
    curvature = np.zeros(len(points), dtype=np.float64)
    for index, local_indices in enumerate(np.asarray(neighbours)):
        local = points[np.asarray(local_indices, dtype=np.int64)]
        local_center = local.mean(axis=0)
        covariance = np.cov((local - local_center).T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(np.asarray(eigenvalues, dtype=np.float64), 0.0)
        normals[index] = eigenvectors[:, int(np.argmin(eigenvalues))]
        curvature[index] = float(eigenvalues[0] / max(eigenvalues.sum(), 1e-12))

    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.maximum(norms, 1e-12)
    return normals, curvature


def _normal_graph_components(
    points: np.ndarray,
    normals: np.ndarray,
    curvature: np.ndarray,
) -> list[np.ndarray]:
    """Build geometry-only smooth-surface components with union-find."""

    count = len(points)
    parent = np.arange(count, dtype=np.int64)
    rank = np.zeros(count, dtype=np.int8)

    def find(index: int) -> int:
        root = index
        while parent[root] != root:
            root = int(parent[root])
        while parent[index] != index:
            next_index = int(parent[index])
            parent[index] = root
            index = next_index
        return root

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            return
        if rank[first_root] < rank[second_root]:
            first_root, second_root = second_root, first_root
        parent[second_root] = first_root
        if rank[first_root] == rank[second_root]:
            rank[first_root] += 1

    pairs = cKDTree(points).query_pairs(
        r=NORMAL_GRAPH_RADIUS_M,
        output_type="ndarray",
    )
    cosine_threshold = math.cos(math.radians(NORMAL_ANGLE_MAX_DEG))
    for first, second in pairs:
        first = int(first)
        second = int(second)
        delta = points[second] - points[first]
        distance = float(np.linalg.norm(delta))
        if distance < 1e-9:
            union(first, second)
            continue

        normal_similarity = abs(float(np.dot(normals[first], normals[second])))
        if normal_similarity < cosine_threshold:
            # A high-curvature patch may have a larger normal change while
            # remaining one smooth surface.  Keep this exception local and
            # strict; it does not use color or object count.
            if max(float(curvature[first]), float(curvature[second])) < 0.055:
                continue

        plane_gap = max(
            abs(float(np.dot(delta, normals[first]))),
            abs(float(np.dot(delta, normals[second]))),
        )
        if plane_gap <= NORMAL_PLANE_GAP_M:
            union(first, second)

    components: dict[int, list[int]] = {}
    for index in range(count):
        components.setdefault(find(index), []).append(index)
    return [
        np.asarray(indices, dtype=np.int64)
        for indices in components.values()
        if len(indices) >= PATCH_MIN_POINTS
    ]


def _make_patches(points: np.ndarray) -> list[Patch]:
    normals, curvature = _estimate_normals_and_curvature(points)
    components = _normal_graph_components(points, normals, curvature)
    patches: list[Patch] = []
    for indices in components:
        patch_normals = normals[indices]
        reference = patch_normals[0]
        signed = patch_normals * np.sign(patch_normals @ reference)[:, None]
        patch_normal = signed.mean(axis=0)
        patch_normal /= max(float(np.linalg.norm(patch_normal)), 1e-12)
        patches.append(
            Patch(
                indices=indices,
                points=points[indices],
                normal=patch_normal,
                curvature=float(np.median(curvature[indices])),
            )
        )
    return patches


def _primitive_coherence(points: np.ndarray) -> tuple[float, dict]:
    """Score whether points are plausibly one finite-catalog primitive."""

    if len(points) < MIN_CLUSTER_POINTS:
        return 0.0, {}
    try:
        fit = fit_primitive_candidates(points, top_k=6, catalog_locked=True)
    except (RuntimeError, ValueError, np.linalg.LinAlgError):
        return 0.0, {}

    selected = str(fit.get("class", ""))
    features = dict(fit.get("features", {}))
    dimension_scores = dict(features.get("catalog_dimension_scores", {}))
    dimension_score = float(dimension_scores.get(selected, 0.0))
    residual_key = {
        "cube": "box_surface_error",
        "cuboid": "box_surface_error",
        "cylinder": "cylinder_fit_error",
        "cone": "cone_fit_error",
        "sphere": "sphere_fit_error",
        "spheroid": "spheroid_fit_error",
    }.get(selected)
    residual = float(features.get(residual_key, 1.0)) if residual_key else 1.0
    residual_scale = {
        "cube": 0.12,
        "cuboid": 0.12,
        "cylinder": 0.22,
        "cone": 0.16,
        "sphere": 0.07,
        "spheroid": 0.14,
    }.get(selected, 0.15)
    residual_score = math.exp(-residual / max(residual_scale, 1e-9))
    confidence = float(fit.get("confidence", 0.0))
    quality = (
        0.45 * dimension_score
        + 0.35 * residual_score
        + 0.20 * confidence
    )
    return float(quality), fit


def _direction_candidates(points: np.ndarray) -> list[np.ndarray]:
    """Return deterministic spatial directions for a geometry-only split."""

    centered = np.asarray(points, dtype=np.float64) - np.mean(points, axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    directions: list[np.ndarray] = []
    for direction in [*vh, *np.eye(3, dtype=np.float64)]:
        direction = np.asarray(direction, dtype=np.float64)
        direction /= max(float(np.linalg.norm(direction)), 1e-12)
        if not any(abs(float(np.dot(direction, other))) > 0.995 for other in directions):
            directions.append(direction)
    # Contact layouts are usually separated in the table plane.  A small set
    # of azimuths catches a leaning object whose PCA axis is dominated by its
    # long vertical extent.
    for angle in np.linspace(0.0, math.pi, 12, endpoint=False):
        direction = np.asarray([math.cos(float(angle)), math.sin(float(angle)), 0.0])
        if not any(abs(float(np.dot(direction, other))) > 0.995 for other in directions):
            directions.append(direction)
    return directions


def _aabb_size_penalty(points: np.ndarray) -> float:
    extent = np.ptp(np.asarray(points, dtype=np.float64), axis=0)
    return max(0.0, float(np.max(extent)) / CATALOG_MAX_AABB_SPAN_M - 1.0)


def _model_split_candidate(
    points: np.ndarray,
    min_cluster_points: int,
) -> tuple[np.ndarray, np.ndarray, dict] | None:
    """Find a two-model partition using primitive coherence, not color.

    This is a fallback for a DBSCAN component that spans more than any single
    finite-catalog primitive.  Candidate planes are perpendicular to PCA,
    world, and table-plane directions.  The best threshold is the one that
    makes both sides more catalog-coherent and reduces the oversized extent.
    """

    whole_quality, whole_fit = _primitive_coherence(points)
    whole_penalty = _aabb_size_penalty(points)
    best: tuple[float, np.ndarray, np.ndarray, dict] | None = None
    quantiles = np.linspace(0.25, 0.75, 11)
    for direction in _direction_candidates(points):
        projected = np.asarray(points, dtype=np.float64) @ direction
        for quantile in quantiles:
            threshold = float(np.quantile(projected, float(quantile)))
            first_mask = projected <= threshold
            second_mask = ~first_mask
            if (
                int(first_mask.sum()) < min_cluster_points
                or int(second_mask.sum()) < min_cluster_points
            ):
                continue
            first = points[first_mask]
            second = points[second_mask]
            first_quality, first_fit = _primitive_coherence(first)
            second_quality, second_fit = _primitive_coherence(second)
            first_class = str(first_fit.get("class", ""))
            second_class = str(second_fit.get("class", ""))
            if not first_class or not second_class:
                continue
            if min(first_quality, second_quality) < 0.22:
                continue
            split_quality = (
                first_quality * len(first) + second_quality * len(second)
            ) / max(len(points), 1)
            split_penalty = (
                _aabb_size_penalty(first) * len(first)
                + _aabb_size_penalty(second) * len(second)
            ) / max(len(points), 1)
            class_bonus = 0.10 if first_class != second_class else 0.0
            score = split_quality + class_bonus - 0.30 * split_penalty
            baseline = whole_quality - 0.30 * whole_penalty
            gain = score - baseline
            if gain < MODEL_SPLIT_MIN_GAIN:
                continue
            diagnostic = {
                "split": True,
                "reason": "catalog_multimodel_projection",
                "direction": direction.tolist(),
                "threshold_m": threshold,
                "candidate_classes": [first_class, second_class],
                "candidate_quality": [float(first_quality), float(second_quality)],
                "quality_gain": float(gain),
                "whole_fit_class": str(whole_fit.get("class", "")),
                "whole_quality": float(whole_quality),
                "whole_extent_m": np.ptp(points, axis=0).tolist(),
            }
            if best is None or score > best[0]:
                best = (score, first_mask, second_mask, diagnostic)
    if best is None:
        return None
    _, first_mask, second_mask, diagnostic = best
    first = points[first_mask]
    second = points[second_mask]

    def reassign_top_overhang(
        lower: np.ndarray,
        other: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        if len(lower) < 100 or float(np.percentile(lower[:, 2], 1.0)) > 0.012:
            return lower, other, 0
        best_height: float | None = None
        best_count = 0
        for height in (0.030, 0.040, 0.045, 0.060):
            count = int(np.sum(np.abs(lower[:, 2] - height) <= 0.0015))
            above_count = int(np.sum(lower[:, 2] > height + 0.003))
            if (
                count >= max(45, int(0.25 * len(lower)))
                and 15 <= above_count <= int(0.35 * len(lower))
                and count > best_count
            ):
                best_height = height
                best_count = count
        if best_height is None:
            return lower, other, 0
        transfer = lower[:, 2] > best_height + 0.003
        moved = int(transfer.sum())
        return lower[~transfer], np.vstack([other, lower[transfer]]), moved

    first, second, moved_first = reassign_top_overhang(first, second)
    second, first, moved_second = reassign_top_overhang(second, first)
    moved = moved_first + moved_second
    if moved:
        diagnostic = {
            **diagnostic,
            "table_top_overhang_points_reassigned": int(moved),
        }
    return first, second, diagnostic


def _cloud_pair_from_points(
    cloud: o3d.geometry.PointCloud,
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
    """Map split point coordinates back to the original cloud indices."""

    source = np.asarray(cloud.points, dtype=np.float64)
    # Points originate from the same voxelized cloud.  A nearest-neighbour
    # lookup avoids relying on floating-point tuple equality.
    tree = cKDTree(source)
    first_indices = tree.query(first, k=1)[1]
    second_indices = tree.query(second, k=1)[1]
    return _cloud_from_indices(cloud, first_indices), _cloud_from_indices(cloud, second_indices)


def _merge_patch_pair(first: Patch, second: Patch) -> Patch:
    indices = np.unique(np.concatenate([first.indices, second.indices]))
    points = np.vstack([first.points, second.points])
    normal = first.normal + second.normal
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    return Patch(
        indices=indices,
        points=points,
        normal=normal,
        curvature=float(np.median([first.curvature, second.curvature])),
    )


def _agglomerate_patches(
    patches: list[Patch],
    merge_quality_min: float = 0.48,
) -> list[Patch]:
    """Merge adjacent surface patches only when a catalog primitive improves."""

    patches = list(patches)
    while len(patches) > 1:
        candidates: list[tuple[float, int, int, Patch]] = []
        for first_index in range(len(patches)):
            for second_index in range(first_index + 1, len(patches)):
                first = patches[first_index]
                second = patches[second_index]
                if _bbox_gap(first.points, second.points) > PATCH_MERGE_GAP_M:
                    continue

                combined = _merge_patch_pair(first, second)
                quality, fit = _primitive_coherence(combined.points)
                if quality < merge_quality_min:
                    continue

                # Prefer a pair whose surface normals are compatible with a
                # single object, but allow orthogonal box faces to merge.
                normal_dot = abs(float(np.dot(first.normal, second.normal)))
                fit_class = str(fit.get("class", ""))
                if normal_dot < 0.25 and fit_class not in {"cube", "cuboid"}:
                    continue
                candidates.append((quality, first_index, second_index, combined))

        if not candidates:
            break
        _, first_index, second_index, combined = max(candidates, key=lambda item: item[0])
        patches[first_index] = combined
        patches.pop(second_index)
    return patches


def split_geometric_cluster(
    cloud: o3d.geometry.PointCloud,
    min_cluster_points: int = MIN_CLUSTER_POINTS,
) -> tuple[list[o3d.geometry.PointCloud], dict]:
    """Split one spatially merged cluster without RGB or simulator metadata."""

    points = np.asarray(cloud.points, dtype=np.float64)
    if len(points) < max(2 * min_cluster_points, 60):
        return [cloud], {"split": False, "reason": "sparse"}

    # A contact merge can remain one connected normal graph because points on
    # the two objects touch at a narrow bridge.  Try catalog-constrained
    # multi-model fitting before accepting that graph as one instance.  The
    # trigger is deliberately conservative so an ordinary tilted primitive is
    # not cut just because its AABB is mildly anisotropic.
    oversized = _aabb_size_penalty(points) > 0.0
    whole_quality, _ = _primitive_coherence(points)
    if oversized or (whole_quality < 0.25 and len(points) >= 320):
        candidate = _model_split_candidate(points, min_cluster_points)
        if candidate is not None:
            first_points, second_points, diagnostic = candidate
            first_cloud, second_cloud = _cloud_pair_from_points(
                cloud,
                first_points,
                second_points,
            )
            diagnostic = {
                **diagnostic,
                "source_point_count": int(len(points)),
                "output_count": 2,
            }
            return [first_cloud, second_cloud], diagnostic

    patches = _make_patches(points)
    if len(patches) <= 1:
        return [cloud], {
            "split": False,
            "reason": "single_smooth_component",
            "oversized": bool(oversized),
            "whole_quality": float(whole_quality),
            "extent_m": np.ptp(points, axis=0).tolist(),
        }

    merged = _agglomerate_patches(patches)
    result: list[o3d.geometry.PointCloud] = []
    for patch in merged:
        if len(patch.indices) < min_cluster_points:
            continue
        result.append(_cloud_from_indices(cloud, patch.indices))

    if len(result) <= 1:
        return [cloud], {
            "split": False,
            "reason": "catalog_agglomeration_single_component",
            "patch_count": len(patches),
            "oversized": bool(oversized),
            "whole_quality": float(whole_quality),
            "extent_m": np.ptp(points, axis=0).tolist(),
        }
    return result, {
        "split": True,
        "reason": "normal_curvature_catalog",
        "patch_count": len(patches),
        "output_count": len(result),
    }


def refine_geometric_clusters(
    clusters: list[o3d.geometry.PointCloud],
    min_cluster_points: int = MIN_CLUSTER_POINTS,
) -> tuple[list[o3d.geometry.PointCloud], list[dict]]:
    """Refine all DBSCAN components and report geometry-only diagnostics."""

    refined: list[o3d.geometry.PointCloud] = []
    diagnostics: list[dict] = []
    for index, cluster in enumerate(clusters):
        pieces, diagnostic = split_geometric_cluster(cluster, min_cluster_points)
        diagnostic = {"source_cluster": index, **diagnostic}
        diagnostics.append(diagnostic)
        refined.extend(pieces)

    refined.sort(key=lambda item: tuple(np.asarray(item.get_center())[:2]))
    return refined, diagnostics


__all__ = [
    "MIN_CLUSTER_POINTS",
    "refine_geometric_clusters",
    "split_geometric_cluster",
]
