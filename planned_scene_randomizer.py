"""Constraint-based static scenes with diverse contact layouts.

The planner deliberately does not run a dynamics engine. It samples a small
scene graph, solves poses for the requested contacts, validates visibility and
non-penetration, then freezes the resulting primitive shapes. This keeps the
RGB-D benchmark reproducible while still producing physically meaningful
contact configurations.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.spatial.transform import Rotation

from shape_catalog import (
    dimensions_to_geometry,
    footprint_radius,
    get_cad_dimensions,
    get_shape_spec,
)


CONTACT_FILE = Path("random_scene_contacts.json")
CONTACT_GAP_M = float(os.environ.get("ROBOT_GRASP_PLANNED_CONTACT_GAP_M", "0.0003"))
WORKSPACE_MARGIN_M = float(os.environ.get("ROBOT_GRASP_PLANNED_WORKSPACE_MARGIN_M", "0.003"))
FREE_CLEARANCE_M = float(os.environ.get("ROBOT_GRASP_PLANNED_FREE_CLEARANCE_M", "0.006"))
MAX_LAYOUT_ATTEMPTS = int(os.environ.get("ROBOT_GRASP_PLANNED_LAYOUT_ATTEMPTS", "120"))
PLANNED_OBJECT_COUNT = int(os.environ.get("ROBOT_GRASP_PLANNED_OBJECTS", "5"))

LAYOUT_TYPES = (
    "table_only",
    "stack",
    "side_contact",
    "partial_support",
    "leaning",
    "bridge",
    "mixed",
)


def _round_class() -> str:
    value = os.environ.get(
        "ROBOT_GRASP_PLANNED_ROUND_SHAPE",
        os.environ.get("ROBOT_GRASP_PHYSICS_ROUND_SHAPE", "sphere"),
    ).strip().lower()
    if value not in {"sphere", "spheroid"}:
        raise ValueError("ROBOT_GRASP_PLANNED_ROUND_SHAPE must be sphere or spheroid")
    return value


def _rz(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _ry(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _pose_rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return _rz(yaw) @ _ry(pitch) @ np.asarray(
        [[1.0, 0.0, 0.0], [0.0, math.cos(roll), -math.sin(roll)], [0.0, math.sin(roll), math.cos(roll)]],
        dtype=np.float64,
    )


def _quaternion(rotation: np.ndarray) -> list[float]:
    return Rotation.from_matrix(np.asarray(rotation, dtype=np.float64)).as_quat().tolist()


def _corners(center: np.ndarray, size_xyz: Iterable[float], rotation: np.ndarray) -> np.ndarray:
    size = np.asarray(tuple(float(v) for v in size_xyz), dtype=np.float64)
    local = np.asarray(
        [[sx * size[0] / 2.0, sy * size[1] / 2.0, sz * size[2] / 2.0]
         for sx in (-1.0, 1.0)
         for sy in (-1.0, 1.0)
         for sz in (-1.0, 1.0)],
        dtype=np.float64,
    )
    return local @ np.asarray(rotation, dtype=np.float64).T + np.asarray(center, dtype=np.float64)


def _xy_bounds(center: np.ndarray, size_xyz: Iterable[float], rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = _corners(center, size_xyz, rotation)
    return points[:, :2].min(axis=0), points[:, :2].max(axis=0)


def _pose_on_table(
    xy: Iterable[float],
    size_xyz: Iterable[float],
    rotation: np.ndarray,
    table_z: float,
) -> np.ndarray:
    local = _corners(np.zeros(3, dtype=np.float64), size_xyz, rotation)
    center = np.asarray([float(xy[0]), float(xy[1]), 0.0], dtype=np.float64)
    center[2] = float(table_z) - float(local[:, 2].min())
    return center


def _visibility(
    center: np.ndarray,
    size_xyz: Iterable[float],
    rotation: np.ndarray,
    camera_model: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if camera_model is None:
        return {"image_bbox_px": None, "camera_depth_range_m": None}
    points = _corners(center, size_xyz, rotation)
    transform = np.asarray(camera_model["base_camera"], dtype=np.float64)
    camera_rotation = transform[:, :3]
    camera_translation = transform[:, 3]
    points_camera = (points - camera_translation) @ camera_rotation
    z = points_camera[:, 2]
    if np.any(z <= float(camera_model["near"])) or np.any(z >= float(camera_model["far"])):
        return None
    tan_x = math.tan(float(camera_model["fov_x"]) / 2.0)
    tan_y = math.tan(float(camera_model["fov_y"]) / 2.0)
    pixels = np.column_stack(
        [
            (1.0 - points_camera[:, 0] / (z * tan_x)) * 0.5 * (camera_model["width"] - 1),
            (1.0 - points_camera[:, 1] / (z * tan_y)) * 0.5 * (camera_model["height"] - 1),
        ]
    )
    bounds = camera_model["safe_bounds"]
    policy = str(camera_model.get("visibility_policy", "full")).lower()
    if policy == "center":
        centre = np.asarray([float(center[0]), float(center[1]), float(center[2])], dtype=np.float64)
        centre_camera = (centre - camera_translation) @ camera_rotation
        centre_z = float(centre_camera[2])
        if centre_z <= float(camera_model["near"]) or centre_z >= float(camera_model["far"]):
            return None
        centre_pixel = np.asarray(
            [
                (1.0 - centre_camera[0] / (centre_z * tan_x)) * 0.5 * (camera_model["width"] - 1),
                (1.0 - centre_camera[1] / (centre_z * tan_y)) * 0.5 * (camera_model["height"] - 1),
            ],
            dtype=np.float64,
        )
        if (
            float(centre_pixel[0]) < bounds["u_min"]
            or float(centre_pixel[0]) > bounds["u_max"]
            or float(centre_pixel[1]) < bounds["v_min"]
            or float(centre_pixel[1]) > bounds["v_max"]
            or float(pixels[:, 0].max()) < 0.0
            or float(pixels[:, 0].min()) > camera_model["width"] - 1
            or float(pixels[:, 1].max()) < 0.0
            or float(pixels[:, 1].min()) > camera_model["height"] - 1
        ):
            return None
    elif (
        float(pixels[:, 0].min()) < bounds["u_min"]
        or float(pixels[:, 0].max()) > bounds["u_max"]
        or float(pixels[:, 1].min()) < bounds["v_min"]
        or float(pixels[:, 1].max()) > bounds["v_max"]
    ):
        return None
    return {
        "image_bbox_px": [float(pixels[:, 0].min()), float(pixels[:, 1].min()), float(pixels[:, 0].max()), float(pixels[:, 1].max())],
        "camera_depth_range_m": [float(z.min()), float(z.max())],
    }


def _workspace_ok(
    center: np.ndarray,
    size_xyz: Iterable[float],
    rotation: np.ndarray,
    reference: dict[str, Any],
) -> bool:
    points = _corners(center, size_xyz, rotation)
    cx = float(reference["workspace_center_x"])
    cy = float(reference["workspace_center_y"])
    hx = float(reference.get("workspace_half_x", 0.11)) - WORKSPACE_MARGIN_M
    hy = float(reference.get("workspace_half_y", 0.09)) - WORKSPACE_MARGIN_M
    return (
        float(points[:, 0].min()) >= cx - hx
        and float(points[:, 0].max()) <= cx + hx
        and float(points[:, 1].min()) >= cy - hy
        and float(points[:, 1].max()) <= cy + hy
    )


def _aabb_overlap(
    bounds_a: tuple[np.ndarray, np.ndarray],
    bounds_b: tuple[np.ndarray, np.ndarray],
    clearance: float,
) -> bool:
    min_a, max_a = bounds_a
    min_b, max_b = bounds_b
    return bool(
        min_a[0] < max_b[0] + clearance
        and max_a[0] > min_b[0] - clearance
        and min_a[1] < max_b[1] + clearance
        and max_a[1] > min_b[1] - clearance
    )


def _obb_penetrates(
    first: dict[str, Any],
    second: dict[str, Any],
    tolerance_m: float = 0.0004,
) -> bool:
    """Return True only when two conservative primitive OBBs interpenetrate.

    The separating-axis test treats contact and very small numerical overlap as
    non-penetrating. Cylinders, cones and spheroids use their bounding boxes,
    which is conservative: a rejected layout may still have been valid, but an
    accepted layout cannot contain the obvious box-through-box intersections
    that are visually invalid for this benchmark.
    """

    center_a = np.asarray(first["center"], dtype=np.float64)
    center_b = np.asarray(second["center"], dtype=np.float64)
    rotation_a = np.asarray(first["rotation"], dtype=np.float64)
    rotation_b = np.asarray(second["rotation"], dtype=np.float64)
    half_a = 0.5 * np.asarray(first["size_xyz"], dtype=np.float64)
    half_b = 0.5 * np.asarray(second["size_xyz"], dtype=np.float64)

    relative_rotation = rotation_a.T @ rotation_b
    absolute_rotation = np.abs(relative_rotation) + 1e-10
    translation = rotation_a.T @ (center_b - center_a)
    tolerance = max(float(tolerance_m), 0.0)

    for axis in range(3):
        radius_a = half_a[axis]
        radius_b = float(np.dot(half_b, absolute_rotation[axis, :]))
        if abs(float(translation[axis])) >= radius_a + radius_b - tolerance:
            return False

    for axis in range(3):
        radius_a = float(np.dot(half_a, absolute_rotation[:, axis]))
        radius_b = half_b[axis]
        projected = abs(float(np.dot(translation, relative_rotation[:, axis])))
        if projected >= radius_a + radius_b - tolerance:
            return False

    for axis_a in range(3):
        next_a = (axis_a + 1) % 3
        last_a = (axis_a + 2) % 3
        for axis_b in range(3):
            next_b = (axis_b + 1) % 3
            last_b = (axis_b + 2) % 3
            radius_a = (
                half_a[next_a] * absolute_rotation[last_a, axis_b]
                + half_a[last_a] * absolute_rotation[next_a, axis_b]
            )
            radius_b = (
                half_b[next_b] * absolute_rotation[axis_a, last_b]
                + half_b[last_b] * absolute_rotation[axis_a, next_b]
            )
            projected = abs(
                translation[last_a] * relative_rotation[next_a, axis_b]
                - translation[next_a] * relative_rotation[last_a, axis_b]
            )
            if projected >= radius_a + radius_b - tolerance:
                return False

    return True


def _plan(
    plan_id: str,
    class_name: str,
    center: np.ndarray,
    rotation: np.ndarray,
    role: str,
    support_refs: list[str] | None = None,
    contact_refs: list[str] | None = None,
    contact_types: list[str] | None = None,
    contact_group: int | None = None,
    stack_level: int = 0,
    contacts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    spec = get_shape_spec(class_name)
    size = np.asarray(get_cad_dimensions(spec.cad_id), dtype=np.float64)
    return {
        "plan_id": plan_id,
        "class": class_name,
        "shape_id": spec.cad_id,
        "size_xyz": tuple(float(v) for v in size),
        "center": np.asarray(center, dtype=np.float64),
        "rotation": np.asarray(rotation, dtype=np.float64),
        "scenario_role": role,
        "support_refs": list(support_refs or []),
        "contact_refs": list(contact_refs or []),
        "contact_types": list(contact_types or []),
        "contact_group": contact_group,
        "stack_level": int(stack_level),
        "contacts": list(contacts or []),
    }


def _table_plan(
    plan_id: str,
    class_name: str,
    xy: Iterable[float],
    yaw: float,
    table_z: float,
    role: str = "table_only",
    contact_group: int | None = None,
) -> dict[str, Any]:
    rotation = _rz(float(yaw))
    spec = get_shape_spec(class_name)
    center = _pose_on_table(xy, get_cad_dimensions(spec.cad_id), rotation, table_z)
    return _plan(
        plan_id,
        class_name,
        center,
        rotation,
        role,
        support_refs=["table"],
        contact_types=["table_contact"],
        contact_group=contact_group,
        contacts=[{"other": "table", "kind": "support", "normal": [0.0, 0.0, 1.0]}],
    )


def _horizontal_extent(size_xyz: Iterable[float], rotation: np.ndarray, direction: np.ndarray) -> float:
    half = np.asarray(tuple(float(v) for v in size_xyz), dtype=np.float64) / 2.0
    return float(np.sum(np.abs(np.asarray(direction, dtype=np.float64) @ rotation) * half))


def _pair_side_contact(
    first_class: str,
    second_class: str,
    anchor_xy: np.ndarray,
    yaw: float,
    table_z: float,
    group: int,
) -> list[dict[str, Any]]:
    direction = np.asarray([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
    first_size = get_cad_dimensions(get_shape_spec(first_class).cad_id)
    second_size = get_cad_dimensions(get_shape_spec(second_class).cad_id)
    first = _table_plan("contact_a", first_class, anchor_xy, yaw, table_z, "side_contact", group)
    first_extent = _horizontal_extent(first_size, first["rotation"], np.asarray([*direction, 0.0]))
    second_extent = _horizontal_extent(second_size, _rz(yaw), np.asarray([*direction, 0.0]))
    second_xy = anchor_xy + direction * (first_extent + second_extent + CONTACT_GAP_M)
    second = _table_plan("contact_b", second_class, second_xy, yaw, table_z, "side_contact", group)
    contact_point = first["center"] + np.asarray([*direction, 0.0]) * first_extent
    second["support_refs"] = ["table"]
    second["contact_refs"] = ["contact_a"]
    second["contact_types"] = ["table_contact", "side_contact"]
    second["contacts"].append({"other": "contact_a", "kind": "side_contact", "point": contact_point.tolist()})
    return [first, second]


def _pair_stack(
    base_class: str,
    top_class: str,
    anchor_xy: np.ndarray,
    yaw: float,
    table_z: float,
    group: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    base = _table_plan("stack_base", base_class, anchor_xy, yaw, table_z, "stack", group)
    top_size = get_cad_dimensions(get_shape_spec(top_class).cad_id)
    top_yaw = float(yaw + rng.uniform(-0.25, 0.25))
    top_rotation = _rz(top_yaw)
    top_xy = anchor_xy + np.asarray(
        [
            rng.uniform(-0.18, 0.18) * float(base["size_xyz"][0]),
            rng.uniform(-0.18, 0.18) * float(base["size_xyz"][1]),
        ],
        dtype=np.float64,
    )
    top_center = _pose_on_table(top_xy, top_size, top_rotation, table_z + float(base["size_xyz"][2]) + CONTACT_GAP_M)
    top = _plan(
        "stack_top",
        top_class,
        top_center,
        top_rotation,
        "stack",
        support_refs=["stack_base"],
        contact_types=["top_face_contact"],
        contact_group=group,
        stack_level=1,
        contacts=[{"other": "stack_base", "kind": "top_face_contact", "point": [float(top_center[0]), float(top_center[1]), float(table_z + base["size_xyz"][2])]}],
    )
    # The upper object must not have its centre of mass completely outside the
    # supporting footprint. A small overhang is allowed for contact scenes.
    return [base, top]


def _tilted_pair(
    mode: str,
    support_class: str,
    beam_class: str,
    anchor_xy: np.ndarray,
    yaw: float,
    table_z: float,
    group: int,
) -> list[dict[str, Any]]:
    support = _table_plan("tilt_support", support_class, anchor_xy, yaw, table_z, mode, group)
    beam_size = get_cad_dimensions(get_shape_spec(beam_class).cad_id)
    support_size = get_cad_dimensions(get_shape_spec(support_class).cad_id)
    hx, hy, hz = np.asarray(beam_size, dtype=np.float64) / 2.0
    support_height = float(support_size[2])
    direction = np.asarray([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float64)
    support_extent = _horizontal_extent(
        support_size,
        support["rotation"],
        direction,
    )
    if mode == "leaning":
        # Leaning is a table edge plus a side tangent to a taller support;
        # the contact height is deliberately below the support top.
        pitch = math.radians(28.0)
        beam_rotation = _rz(yaw) @ _ry(-pitch)
        lower_local = np.asarray([-hx, 0.0, -hz], dtype=np.float64)
        beam_center = np.asarray([anchor_xy[0], anchor_xy[1], 0.0], dtype=np.float64) - beam_rotation @ lower_local
        beam_center[2] += table_z
        upper_local = np.asarray([hx, 0.0, 0.0], dtype=np.float64)
        upper_contact = beam_center + beam_rotation @ upper_local
        support_center = support["center"].copy()
        support_center[:2] = upper_contact[:2] + direction[:2] * (
            support_extent + CONTACT_GAP_M
        )
        support["center"] = support_center
        beam = _plan(
            "tilt_beam",
            beam_class,
            beam_center,
            beam_rotation,
            mode,
            support_refs=["table"],
            contact_refs=["tilt_support"],
            contact_types=["table_edge_contact", "side_surface_contact"],
            contact_group=group,
            contacts=[
                {"other": "table", "kind": "table_edge_contact", "point": (beam_center + beam_rotation @ lower_local).tolist()},
                {"other": "tilt_support", "kind": "side_surface_contact", "point": upper_contact.tolist()},
            ],
        )
        beam["roll_deg"] = 0.0
        beam["pitch_deg"] = float(math.degrees(pitch))
        return [support, beam]
    contact_x = hx if mode == "bridge" else 0.70 * hx
    delta_x = hx + contact_x
    sine = (support_height + CONTACT_GAP_M) / max(delta_x, 1e-9)
    if sine >= 0.94:
        contact_x = hx
        delta_x = 2.0 * hx
        sine = (support_height + CONTACT_GAP_M) / max(delta_x, 1e-9)
    pitch = math.asin(float(np.clip(sine, 0.20, 0.92)))
    beam_rotation = _rz(yaw) @ _ry(-pitch)
    lower_local = np.asarray([-hx, 0.0, -hz], dtype=np.float64)
    beam_center = np.asarray([anchor_xy[0], anchor_xy[1], 0.0], dtype=np.float64) - beam_rotation @ lower_local
    beam_center[2] += table_z
    upper_local = np.asarray([contact_x, 0.0, -hz], dtype=np.float64)
    upper_contact = beam_center + beam_rotation @ upper_local
    support_center = support["center"].copy()
    # Put the near edge of the support at the beam contact point. Centering the
    # support on that point makes half of the support occupy the beam volume.
    support_center[:2] = upper_contact[:2] + direction[:2] * (
        support_extent + CONTACT_GAP_M
    )
    support["center"] = support_center
    beam = _plan(
        "tilt_beam",
        beam_class,
        beam_center,
        beam_rotation,
        mode,
        support_refs=["table", "tilt_support"],
        contact_refs=["tilt_support"],
        contact_types=["table_edge_contact", "support_surface_contact"],
        contact_group=group,
        contacts=[
            {"other": "table", "kind": "table_edge_contact", "point": (beam_center + beam_rotation @ lower_local).tolist()},
            {"other": "tilt_support", "kind": "support_surface_contact", "point": upper_contact.tolist()},
        ],
    )
    beam["roll_deg"] = 0.0
    beam["pitch_deg"] = float(math.degrees(pitch))
    beam["support_span_ratio"] = float((contact_x + hx) / max(2.0 * hx, 1e-9))
    return [support, beam]


def _recipe(layout: str, round_class: str) -> tuple[list[str], str]:
    if layout == "table_only":
        return ["cube", "cuboid", "cylinder", round_class, "cone"], layout
    if layout == "stack":
        return ["cuboid", "cylinder", "cube", round_class, "cone"], layout
    if layout == "side_contact":
        return ["cube", "cylinder", "cuboid", round_class, "cone"], layout
    if layout == "partial_support":
        return ["cuboid", "cuboid", "cube", "cylinder", round_class, "cone"], layout
    if layout == "leaning":
        return ["cylinder", "cuboid", "cube", round_class, "cone"], layout
    if layout == "bridge":
        return ["cube", "cuboid", "cylinder", round_class, "cone"], layout
    if layout == "mixed":
        return ["cuboid", "cone", "cube", "cylinder", round_class], layout
    raise ValueError(f"Unsupported planned layout: {layout}")


def _select_layout(rng: np.random.Generator, requested: str) -> str:
    requested = requested.strip().lower()
    if requested != "auto":
        if requested not in LAYOUT_TYPES:
            raise ValueError(f"Unknown planned layout: {requested}")
        return requested
    return str(rng.choice(LAYOUT_TYPES, p=[0.12, 0.18, 0.15, 0.16, 0.15, 0.14, 0.10]))


def _append_free_objects(
    plans: list[dict[str, Any]],
    class_names: list[str],
    reference: dict[str, Any],
    camera_model: dict[str, Any] | None,
    table_z: float,
    rng: np.random.Generator,
) -> None:
    existing_bounds = [_xy_bounds(item["center"], item["size_xyz"], item["rotation"]) for item in plans]
    for index, class_name in enumerate(class_names):
        placed = None
        for _ in range(900):
            yaw = float(rng.uniform(-math.pi, math.pi))
            rotation = _rz(yaw)
            size = get_cad_dimensions(get_shape_spec(class_name).cad_id)
            cx = float(reference["workspace_center_x"])
            cy = float(reference["workspace_center_y"])
            hx = float(reference.get("workspace_half_x", 0.11)) - WORKSPACE_MARGIN_M
            hy = float(reference.get("workspace_half_y", 0.09)) - WORKSPACE_MARGIN_M
            half_xy = np.sum(np.abs(rotation[:2, :]) * (np.asarray(size) / 2.0), axis=1)
            if hx <= half_xy[0] or hy <= half_xy[1]:
                continue
            xy = np.asarray([rng.uniform(cx - hx + half_xy[0], cx + hx - half_xy[0]), rng.uniform(cy - hy + half_xy[1], cy + hy - half_xy[1])])
            center = _pose_on_table(xy, size, rotation, table_z)
            bounds = _xy_bounds(center, size, rotation)
            if any(_aabb_overlap(bounds, other, FREE_CLEARANCE_M) for other in existing_bounds):
                continue
            if _visibility(center, size, rotation, camera_model) is None or not _workspace_ok(center, size, rotation, reference):
                continue
            placed = _plan(f"single_{index:02d}", class_name, center, rotation, "table_only", support_refs=["table"], contact_types=["table_contact"], contacts=[{"other": "table", "kind": "support"}])
            existing_bounds.append(bounds)
            plans.append(placed)
            break
        if placed is None:
            raise RuntimeError(f"Unable to place planned free object {class_name}")


def _validate_layout(plans: list[dict[str, Any]], reference: dict[str, Any], camera_model: dict[str, Any] | None) -> bool:
    if not plans:
        return False
    for item in plans:
        if not np.all(np.isfinite(item["center"])) or not np.all(np.isfinite(item["rotation"])):
            return False
        if not _workspace_ok(item["center"], item["size_xyz"], item["rotation"], reference):
            return False
        if _visibility(item["center"], item["size_xyz"], item["rotation"], camera_model) is None:
            return False
        corners = _corners(item["center"], item["size_xyz"], item["rotation"])
        if float(corners[:, 2].min()) < float(reference["table_z"]) - 0.0005:
            return False

    # Contact relations may touch, but they are never allowed to occupy the
    # same volume. Previously intentional pairs skipped validation entirely,
    # which allowed a tilted cuboid to pass through its cube support.
    for i, first in enumerate(plans):
        for second in plans[i + 1 :]:
            intentional = (
                first["plan_id"] in second.get("support_refs", [])
                or second["plan_id"] in first.get("support_refs", [])
                or first["plan_id"] in second.get("contact_refs", [])
                or second["plan_id"] in first.get("contact_refs", [])
            )
            tolerance = 0.0008 if intentional else 0.0002
            if _obb_penetrates(first, second, tolerance_m=tolerance):
                return False
    return True


def build_planned_layout(
    reference: dict[str, Any],
    camera_model: dict[str, Any] | None = None,
    seed: int | None = None,
    layout: str | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Return a deterministic valid layout without requiring CoppeliaSim."""
    rng = np.random.default_rng(seed if seed is not None else None)
    selected = _select_layout(rng, layout or os.environ.get("ROBOT_GRASP_PLANNED_LAYOUT", "auto"))
    round_class = _round_class()
    classes, _ = _recipe(selected, round_class)
    target_count = max(1, PLANNED_OBJECT_COUNT)
    classes = classes[:target_count]
    while len(classes) < target_count:
        classes.append(str(rng.choice(["cube", "cuboid", "cylinder", round_class, "cone"])))
    for _attempt in range(max(1, MAX_LAYOUT_ATTEMPTS)):
        table_z = float(reference["table_z"])
        center = np.asarray([float(reference["workspace_center_x"]), float(reference["workspace_center_y"])])
        anchor = center + rng.uniform(-0.025, 0.025, size=2)
        plans: list[dict[str, Any]] = []
        try:
            if selected == "table_only":
                free_classes = classes
            elif selected == "stack":
                pair = _pair_stack(classes[0], classes[1], anchor, float(rng.uniform(-math.pi, math.pi)), table_z, 1, rng)
                plans.extend(pair)
                free_classes = classes[2:]
            elif selected == "side_contact":
                pair = _pair_side_contact(classes[0], classes[1], anchor, float(rng.uniform(-math.pi, math.pi)), table_z, 1)
                plans.extend(pair)
                free_classes = classes[2:]
            elif selected in {"partial_support", "bridge"}:
                pair = _tilted_pair(selected, classes[0], classes[1], anchor, float(rng.uniform(-math.pi, math.pi)), table_z, 1)
                plans.extend(pair)
                free_classes = classes[2:]
            elif selected == "leaning":
                pair = _tilted_pair("leaning", classes[0], classes[1], anchor, float(rng.uniform(-math.pi, math.pi)), table_z, 1)
                plans.extend(pair)
                free_classes = classes[2:]
            elif selected == "mixed":
                stack = _pair_stack(classes[0], classes[1], anchor - np.asarray([0.045, 0.0]), float(rng.uniform(-0.25, 0.25)), table_z, 1, rng)
                side = _pair_side_contact(classes[2], classes[3], anchor + np.asarray([0.060, 0.0]), float(rng.uniform(-math.pi, math.pi)), table_z, 2)
                plans.extend(stack)
                plans.extend(side)
                free_classes = classes[4:]
            else:
                raise AssertionError(selected)
            _append_free_objects(plans, list(free_classes), reference, camera_model, table_z, rng)
            if len(plans) != len(classes) or not _validate_layout(plans, reference, camera_model):
                raise RuntimeError("planned layout validation failed")
            for item in plans:
                item["visibility"] = _visibility(item["center"], item["size_xyz"], item["rotation"], camera_model)
            for item in plans:
                item["quaternion"] = _quaternion(item["rotation"])
                item["pose_base"] = [*item["center"].tolist(), *item["quaternion"]]
            return plans, selected
        except RuntimeError:
            continue
    raise RuntimeError(f"Unable to build a valid planned layout: {selected}")


def _resolve_contact_alias(contact: dict[str, Any], aliases: dict[str, str]) -> dict[str, Any]:
    item = dict(contact)
    other = str(item.get("other", ""))
    item["object_a"] = aliases.get(str(item.get("object_a", "")), str(item.get("object_a", "")))
    item["object_b"] = "table" if other == "table" else aliases.get(other, other)
    item.pop("other", None)
    return item


def generate_planned_scene(
    sim: Any,
    robot_base: int,
    reference: dict[str, Any],
    camera_model: dict[str, Any],
) -> list[dict[str, Any]]:
    """Plan and instantiate a static scene in CoppeliaSim."""
    from scene_randomizer import COLOR_PALETTE, create_object, set_shape_color

    seed_value = os.environ.get("ROBOT_GRASP_RANDOM_SEED")
    seed = int(seed_value) if seed_value else None
    plans, selected = build_planned_layout(reference, camera_model, seed=seed)
    aliases: dict[str, str] = {}
    handles_by_plan: dict[str, int] = {}
    counters: dict[str, int] = {}
    handles: list[int] = []
    records: list[dict[str, Any]] = []
    try:
        for index, item in enumerate(plans):
            class_name = str(item["class"])
            counters[class_name] = counters.get(class_name, 0) + 1
            alias = f"rand_{class_name}_{counters[class_name]:02d}"
            handle = create_object(sim, class_name, tuple(item["size_xyz"]))
            handles.append(handle)
            handles_by_plan[str(item["plan_id"])] = int(handle)
            sim.setObjectAlias(handle, alias)
            sim.setObjectPose(handle, item["pose_base"], robot_base)
            set_shape_color(sim, handle, COLOR_PALETTE[index % len(COLOR_PALETTE)])
            aliases[str(item["plan_id"])] = alias
        supported_by: dict[str, list[str]] = {str(item["plan_id"]): [] for item in plans}
        for item in plans:
            for support in item.get("support_refs", []):
                if support != "table" and support in supported_by:
                    supported_by[support].append(str(item["plan_id"]))
        for item in plans:
            plan_id = str(item["plan_id"])
            alias = aliases[plan_id]
            supports = ["table" if value == "table" else aliases[value] for value in item.get("support_refs", [])]
            blockers = [aliases[value] for value in supported_by.get(plan_id, [])]
            visibility = item.get("visibility") or {"image_bbox_px": None, "camera_depth_range_m": None}
            record = {
                "handle": handles_by_plan[plan_id],
                "alias": alias,
                "class": item["class"],
                "shape_id": item["shape_id"],
                "primitive_name": get_shape_spec(item["class"]).primitive_name,
                "size_m": [float(v) for v in item["size_xyz"]],
                "geometry": dimensions_to_geometry(item["class"], item["size_xyz"]),
                "symmetry": get_shape_spec(item["class"]).symmetry,
                "grasp_family": get_shape_spec(item["class"]).grasp_family,
                "position": [float(v) for v in item["center"]],
                "pose_base": [float(v) for v in item["pose_base"]],
                "quaternion": [float(v) for v in item["quaternion"]],
                "rotation_matrix": np.asarray(item["rotation"], dtype=np.float64).tolist(),
                "yaw_deg": float(math.degrees(math.atan2(item["rotation"][1, 0], item["rotation"][0, 0]))),
                "roll_deg": float(item.get("roll_deg", 0.0)),
                "pitch_deg": float(item.get("pitch_deg", 0.0)),
                "footprint_radius": float(footprint_radius(item["size_xyz"])),
                "image_bbox_px": visibility.get("image_bbox_px"),
                "camera_depth_range_m": visibility.get("camera_depth_range_m"),
                "scenario_role": item["scenario_role"],
                "planned_layout": selected,
                "support_refs": supports,
                "contact_refs": [
                    "table" if value == "table" else aliases[value]
                    for value in item.get("contact_refs", [])
                ],
                "supported_object_refs": blockers,
                "contact_types": list(item.get("contact_types", [])),
                "contact_group": item.get("contact_group"),
                "stack_level": int(item.get("stack_level", 0)),
                "topmost": not blockers,
                "grasp_blocked": bool(blockers),
                "grasp_priority": 1.0 if not blockers else 0.35,
            }
            records.append(record)
        contacts: list[dict[str, Any]] = []
        for item in plans:
            alias = aliases[str(item["plan_id"])]
            for contact in item.get("contacts", []):
                resolved = _resolve_contact_alias({**contact, "object_a": str(item["plan_id"])}, aliases)
                resolved["object_a"] = alias
                resolved["layout"] = selected
                contacts.append(resolved)
        CONTACT_FILE.write_text(json.dumps({"scene_mode": "planned", "layout_type": selected, "contacts": contacts}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Planned layout: {selected} | objects={len(records)}")
        for item in records:
            print(f"{item['alias']:20s} | {item['class']:8s} | role={item['scenario_role']:15s} | P={np.round(item['position'], 4)}")
        return records
    except Exception:
        if handles:
            try:
                sim.removeObjects(handles)
            except Exception:
                pass
        raise
