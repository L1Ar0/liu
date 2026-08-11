"""Geometry-only grasp candidates and conservative collision annotations."""

from __future__ import annotations

from typing import Any

import numpy as np


GRIPPER_MAX_OPENING_M = 0.085
GRIPPER_CLEARANCE_M = 0.006
PREGRASP_STANDOFF_M = 0.090
FINAL_STANDOFF_M = 0.012


def _dimensions(item: dict[str, Any]) -> np.ndarray:
    geometry = item.get("geometry", {})
    class_name = str(item.get("class", ""))
    if class_name == "cylinder":
        diameter = float(geometry.get("diameter_m", 0.0))
        return np.asarray([diameter, diameter, float(geometry.get("height_m", 0.0))])
    if class_name == "sphere":
        diameter = float(geometry.get("diameter_m", 0.0))
        return np.asarray([diameter, diameter, diameter])
    if class_name == "spheroid":
        return np.asarray(
            [
                float(geometry.get("axis_x_m", 0.0)),
                float(geometry.get("axis_y_m", 0.0)),
                float(geometry.get("axis_z_m", 0.0)),
            ]
        )
    if class_name == "cone":
        return np.asarray(
            [
                float(geometry.get("base_diameter_m", 0.0)),
                float(geometry.get("base_diameter_m", 0.0)),
                float(geometry.get("height_m", 0.0)),
            ]
        )
    return np.asarray(
        [
            float(geometry.get("length_m", 0.0)),
            float(geometry.get("width_m", 0.0)),
            float(geometry.get("height_m", 0.0)),
        ]
    )


def _bounds(item: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float, float]:
    center = np.asarray(item.get("center_m", item.get("center", [0.0, 0.0, 0.0])), dtype=np.float64)
    rotation = np.asarray(item.get("rotation_matrix", np.eye(3)), dtype=np.float64).reshape(3, 3)
    half = 0.5 * _dimensions(item)
    vertical_half = float(np.sum(np.abs(rotation[2, :]) * half))
    horizontal_half = np.sum(np.abs(rotation[:2, :]) * half[None, :], axis=1)
    lower = center[:2] - horizontal_half
    upper = center[:2] + horizontal_half
    return lower, upper, float(center[2] - vertical_half), float(center[2] + vertical_half)


def _point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    direction = end - start
    denominator = float(np.dot(direction, direction))
    if denominator < 1e-12:
        return float(np.linalg.norm(point - start))
    ratio = float(np.clip(np.dot(point - start, direction) / denominator, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + ratio * direction)))


def _candidate_collisions(
    target_index: int,
    start: np.ndarray,
    end: np.ndarray,
    objects: list[dict[str, Any]],
) -> list[int]:
    collisions: list[int] = []
    for index, other in enumerate(objects):
        if index == target_index:
            continue
        center = np.asarray(other.get("center_m", [0.0, 0.0, 0.0]), dtype=np.float64)
        radius = 0.5 * float(np.linalg.norm(_dimensions(other))) + GRIPPER_CLEARANCE_M
        if _point_segment_distance(center, start, end) <= radius:
            collisions.append(int(other.get("id", index)))
    return collisions


def _grasp_candidates(
    index: int,
    item: dict[str, Any],
    objects: list[dict[str, Any]],
    topmost: bool,
) -> list[dict[str, Any]]:
    center = np.asarray(item.get("center_m", [0.0, 0.0, 0.0]), dtype=np.float64)
    rotation = np.asarray(item.get("rotation_matrix", np.eye(3)), dtype=np.float64).reshape(3, 3)
    dimensions = _dimensions(item)
    class_name = str(item.get("class", "unknown"))
    candidates: list[dict[str, Any]] = []

    axes = [
        ("top", np.asarray([0.0, 0.0, 1.0]), int(np.argmin(dimensions[:2]))),
        ("side_x", rotation[:, 0], 1),
        ("side_y", rotation[:, 1], 0),
    ]
    if class_name == "sphere":
        axes = axes[:1]
    for name, approach, closing_axis in axes:
        approach = np.asarray(approach, dtype=np.float64)
        approach /= max(float(np.linalg.norm(approach)), 1e-12)
        if float(np.dot(approach, np.asarray([0.0, 0.0, 1.0]))) < -0.25:
            approach = -approach
        half_extent = 0.5 * float(np.sum(np.abs(rotation.T @ approach) * dimensions))
        final = center + approach * (half_extent + FINAL_STANDOFF_M)
        pregrasp = center + approach * (half_extent + PREGRASP_STANDOFF_M)
        required_opening = float(dimensions[min(max(closing_axis, 0), 2)] + GRIPPER_CLEARANCE_M)
        collisions = _candidate_collisions(index, pregrasp, final, objects)
        reachable_width = required_opening <= GRIPPER_MAX_OPENING_M
        collision_free = not collisions
        score = (
            (1.0 if name == "top" and topmost else 0.55)
            + 0.25 * float(collision_free)
            + 0.20 * float(reachable_width)
            - 0.08 * len(collisions)
        )
        candidates.append(
            {
                "type": name,
                "approach_vector_base": approach.tolist(),
                "pregrasp_center_m": pregrasp.tolist(),
                "grasp_center_m": final.tolist(),
                "required_opening_m": required_opening,
                "within_gripper_opening": reachable_width,
                "collision_free": collision_free,
                "collision_object_ids": collisions,
                "score": float(score),
            }
        )
    candidates.sort(key=lambda candidate: float(candidate["score"]), reverse=True)
    return candidates


def annotate_grasp_planning(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add topmost/blocked hints without changing class or pose estimates."""
    annotated = [dict(item) for item in objects]
    bounds = [_bounds(item) for item in annotated]
    for index, item in enumerate(annotated):
        lower, upper, bottom_z, top_z = bounds[index]
        center_z = 0.5 * (bottom_z + top_z)
        objects_above: list[int] = []
        nearby: list[int] = []
        for other_index, other in enumerate(annotated):
            if index == other_index:
                continue
            other_lower, other_upper, other_bottom, other_top = bounds[other_index]
            overlap_x = min(float(upper[0]), float(other_upper[0])) - max(float(lower[0]), float(other_lower[0]))
            overlap_y = min(float(upper[1]), float(other_upper[1])) - max(float(lower[1]), float(other_lower[1]))
            if overlap_x >= -0.003 and overlap_y >= -0.003:
                nearby.append(int(other.get("id", other_index)))
                other_center_z = 0.5 * (other_bottom + other_top)
                if (
                    other_center_z > center_z + 0.008
                    and other_top > top_z + 0.004
                ):
                    objects_above.append(int(other.get("id", other_index)))
        topmost = not objects_above
        candidates = _grasp_candidates(index, item, annotated, topmost)
        feasible = [
            candidate
            for candidate in candidates
            if candidate["collision_free"] and candidate["within_gripper_opening"]
        ]
        item["grasp_planning"] = {
            "topmost": topmost,
            "grasp_blocked": bool(objects_above),
            "objects_above": sorted(set(objects_above)),
            "nearby_objects": sorted(set(nearby)),
            "preferred_approach": "top" if topmost else "defer",
            "priority_score": 1.0 if topmost else 0.35,
            "estimated_visible_height_m": max(0.0, top_z - bottom_z),
            "candidate_count": len(candidates),
            "feasible_candidate_count": len(feasible),
            "recommended_candidate": feasible[0] if feasible else None,
            "candidates": candidates,
        }
    return annotated
