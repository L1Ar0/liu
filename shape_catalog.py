"""Shared finite primitive definitions for scene generation and recognition.

The project uses metres internally. The catalog keeps primitive identities,
symmetries and nominal dimensions in one place for both simulation and RGB-D
pose estimation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ShapeSpec:
    """Closed-set primitive metadata shared by every pipeline stage."""

    class_name: str
    primitive_name: str
    cad_id: str
    symmetry: str
    grasp_family: str


SHAPE_CATALOG: dict[str, ShapeSpec] = {
    "cube": ShapeSpec(
        "cube", "cuboid", "cube_40", "cube", "parallel_jaw_side",
    ),
    "cuboid": ShapeSpec(
        "cuboid", "cuboid", "cuboid_70x35x30", "cuboid", "parallel_jaw_side",
    ),
    "cylinder": ShapeSpec(
        "cylinder", "cylinder", "cylinder_d40_h60", "cylinder_axis", "parallel_jaw_side",
    ),
    # CoppeliaSim represents spheres and ellipsoids through primitiveshape_spheroid.
    "sphere": ShapeSpec(
        "sphere", "spheroid", "sphere_d45", "sphere", "parallel_jaw_equator",
    ),
    "spheroid": ShapeSpec(
        "spheroid", "spheroid", "spheroid_50x40x40", "spheroid_axis", "parallel_jaw_side",
    ),
    "cone": ShapeSpec(
        "cone", "cone", "cone_d45_h60", "cone_axis", "parallel_jaw_side",
    ),
}


CAD_DIMENSIONS_M: dict[str, tuple[float, float, float]] = {
    "cube_40": (0.040, 0.040, 0.040),
    "cuboid_70x35x30": (0.070, 0.035, 0.030),
    "cylinder_d40_h60": (0.040, 0.040, 0.060),
    "sphere_d45": (0.045, 0.045, 0.045),
    "spheroid_50x40x40": (0.050, 0.040, 0.040),
    "cone_d45_h60": (0.045, 0.045, 0.060),
}


CLASS_ALIASES = {
    "box": "cuboid",
    "rectangular_prism": "cuboid",
    "rect_prism": "cuboid",
    "spheroid": "spheroid",
    "sphere": "sphere",
}


def canonical_class(name: str) -> str:
    value = str(name).strip().lower()
    value = CLASS_ALIASES.get(value, value)
    if value not in SHAPE_CATALOG:
        raise ValueError(f"Unsupported primitive class: {name}")
    return value


def get_shape_spec(name: str) -> ShapeSpec:
    return SHAPE_CATALOG[canonical_class(name)]


def get_cad_dimensions(cad_id: str) -> tuple[float, float, float]:
    try:
        return CAD_DIMENSIONS_M[str(cad_id)]
    except KeyError as exc:
        raise ValueError(f"Unknown CAD id: {cad_id}") from exc


def supported_classes(include_spheroid: bool = False) -> tuple[str, ...]:
    classes = ["cube", "cuboid", "cylinder", "sphere", "cone"]
    if include_spheroid:
        classes.append("spheroid")
    return tuple(classes)


def sample_dimensions(
    class_name: str,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Return primitive dimensions in metres.

    All primitives use CoppeliaSim's conventional ``(x, y, z)`` bounding
    dimensions. For a cone, x/y are the base diameters and z is the height.
    """

    rng = rng or np.random.default_rng()
    cls = canonical_class(class_name)

    if cls == "cube":
        side = float(rng.uniform(0.030, 0.060))
        return side, side, side
    if cls == "cuboid":
        length = float(rng.uniform(0.055, 0.090))
        width = float(rng.uniform(0.028, 0.048))
        height = float(rng.uniform(0.025, 0.055))
        if length / max(width, 1e-9) < 1.40:
            length = min(0.090, width * 1.45)
        return length, width, height
    if cls == "cylinder":
        diameter = float(rng.uniform(0.030, 0.058))
        height = float(rng.uniform(0.035, 0.075))
        return diameter, diameter, height
    if cls == "sphere":
        diameter = float(rng.uniform(0.035, 0.060))
        return diameter, diameter, diameter
    if cls == "spheroid":
        major = float(rng.uniform(0.045, 0.065))
        minor = float(rng.uniform(0.032, 0.050))
        height = float(rng.uniform(0.032, 0.050))
        return major, minor, height
    if cls == "cone":
        base = float(rng.uniform(0.035, 0.060))
        height = float(rng.uniform(0.045, 0.080))
        return base, base, height
    raise AssertionError(cls)


def footprint_radius(size_xyz: Iterable[float], class_name: str | None = None) -> float:
    """Conservative XY radius used only for initial placement."""

    sx, sy, _ = (float(v) for v in size_xyz)
    return 0.5 * float(np.hypot(sx, sy))


def dimensions_to_geometry(
    class_name: str,
    size_xyz: Iterable[float],
) -> dict[str, float]:
    values = tuple(float(v) for v in size_xyz)
    cls = canonical_class(class_name)
    if cls in {"cylinder"}:
        return {"diameter_m": values[0], "height_m": values[2]}
    if cls == "sphere":
        return {"diameter_m": values[0]}
    if cls == "spheroid":
        return {
            "axis_x_m": values[0],
            "axis_y_m": values[1],
            "axis_z_m": values[2],
        }
    if cls == "cone":
        return {
            "base_diameter_m": values[0],
            "top_diameter_m": 0.0,
            "height_m": values[2],
        }
    return {
        "length_m": values[0],
        "width_m": values[1],
        "height_m": values[2],
    }


def class_from_alias(alias: str) -> str | None:
    name = str(alias).lower()
    for cls in ("cuboid", "cylinder", "sphere", "spheroid", "cone", "cube"):
        if cls in name:
            return cls
    return None
