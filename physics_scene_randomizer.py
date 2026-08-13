"""Physics-based random scene generation for the five supported primitives.

The static scene generator remains available for regression. Set
``ROBOT_GRASP_SCENE_MODE=physics`` to use this module through
``scene_randomizer.py``. The simulation is intentionally left paused after
settling so later RGB-D/segmentation/evaluation stages see the settled poses
without CoppeliaSim resetting them on stop.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from shape_catalog import (
    ShapeSpec,
    dimensions_to_geometry,
    footprint_radius,
    get_cad_dimensions,
    get_shape_spec,
)


CONTACT_FILE = Path("random_scene_contacts.json")
DROP_COLLISION_BIT = 0x0100
MIN_OBJECTS = int(os.environ.get("ROBOT_GRASP_PHYSICS_MIN_OBJECTS", "5"))
MAX_OBJECTS = int(os.environ.get("ROBOT_GRASP_PHYSICS_MAX_OBJECTS", "5"))
DROP_HEIGHT_M = float(os.environ.get("ROBOT_GRASP_PHYSICS_DROP_HEIGHT_M", "0.12"))
SETTLE_TIMEOUT_S = float(os.environ.get("ROBOT_GRASP_PHYSICS_TIMEOUT_S", "30.0"))
LINEAR_SPEED_THRESHOLD = float(
    os.environ.get("ROBOT_GRASP_PHYSICS_LINEAR_THRESHOLD", "0.004")
)
ANGULAR_SPEED_THRESHOLD = float(
    os.environ.get("ROBOT_GRASP_PHYSICS_ANGULAR_THRESHOLD", "0.1047198")
)
STABLE_STEPS_REQUIRED = int(
    os.environ.get("ROBOT_GRASP_PHYSICS_STABLE_STEPS", "12")
)
INITIAL_HEIGHT_JITTER_M = float(
    os.environ.get("ROBOT_GRASP_PHYSICS_INITIAL_HEIGHT_JITTER_M", "0.045")
)
INITIAL_POSE_TRIES = int(
    os.environ.get("ROBOT_GRASP_PHYSICS_INITIAL_POSE_TRIES", "200")
)
INITIAL_XY_CLEARANCE_RATIO = float(
    os.environ.get("ROBOT_GRASP_PHYSICS_XY_CLEARANCE_RATIO", "0.90")
)
INITIAL_XY_MARGIN_M = float(
    os.environ.get("ROBOT_GRASP_PHYSICS_XY_MARGIN_M", "0.004")
)
CREATE_CONTAINMENT_WALLS = os.environ.get(
    "ROBOT_GRASP_CREATE_CONTAINMENT_WALLS", "1"
).lower() not in {"0", "false", "no"}


def _random_quaternion(rng: np.random.Generator) -> list[float]:
    value = rng.normal(size=4).astype(np.float64)
    value /= max(float(np.linalg.norm(value)), 1e-12)
    return value.tolist()


def _set_shape_flag(sim: Any, handle: int, parameter_name: str, value: int) -> None:
    parameter = getattr(sim, parameter_name, None)
    if parameter is None:
        return
    try:
        sim.setObjectInt32Param(handle, parameter, int(value))
    except Exception:
        # Properties APIs differ slightly between CoppeliaSim releases.
        pass


def _get_shape_flag(sim: Any, handle: int, parameter_name: str, default: int) -> int:
    parameter = getattr(sim, parameter_name, None)
    if parameter is None:
        return default
    try:
        return int(sim.getObjectInt32Param(handle, parameter))
    except Exception:
        return default


def _suspend_robot_collisions(sim: Any, robot_base: int) -> list[tuple[int, int]]:
    """Prevent falling dataset objects from striking the robot/gripper."""

    saved: list[tuple[int, int]] = []
    shapes = sim.getObjectsInTree(robot_base, sim.sceneobject_shape, 0)
    for handle in shapes:
        original = _get_shape_flag(
            sim,
            int(handle),
            "shapeintparam_respondable_mask",
            0xFFFF,
        )
        saved.append((int(handle), original))
        _set_shape_flag(
            sim,
            int(handle),
            "shapeintparam_respondable_mask",
            original & ~DROP_COLLISION_BIT,
        )
    return saved


def _restore_robot_collisions(sim: Any, saved: list[tuple[int, int]]) -> None:
    for handle, value in saved:
        _set_shape_flag(sim, handle, "shapeintparam_respondable_mask", value)


def _capture_robot_configuration(sim: Any, robot_base: int) -> dict[str, Any]:
    joints: list[tuple[int, float]] = []
    joint_modes: list[tuple[int, int]] = []
    joint_dynctrl_modes: list[tuple[int, int | None]] = []
    mode_param = getattr(sim, "jointintparam_mode", None)
    dynctrl_param = getattr(sim, "jointintparam_dynctrlmode", None)
    for handle in sim.getObjectsInTree(robot_base, sim.sceneobject_joint, 0):
        handle = int(handle)
        try:
            joints.append((handle, float(sim.getJointPosition(handle))))
        except Exception:
            continue
        try:
            raw_mode = sim.getJointMode(handle)
            mode = int(raw_mode[0] if isinstance(raw_mode, (tuple, list)) else raw_mode)
            if mode_param is not None:
                try:
                    mode = int(sim.getObjectInt32Param(handle, mode_param))
                except Exception:
                    pass
            joint_modes.append((handle, mode))
        except Exception:
            pass
        dynctrl: int | None = None
        if dynctrl_param is not None:
            try:
                dynctrl = int(sim.getObjectInt32Param(handle, dynctrl_param))
            except Exception:
                pass
        joint_dynctrl_modes.append((handle, dynctrl))
    return {
        "base_pose_world": [float(value) for value in sim.getObjectPose(robot_base, -1)],
        "joints": joints,
        "joint_modes": joint_modes,
        "joint_dynctrl_modes": joint_dynctrl_modes,
    }


def _set_robot_kinematic_for_settle(sim: Any, configuration: dict[str, Any]) -> None:
    """Hold the arm while generated workpieces settle.

    Scene generation is not a control experiment.  The arm must therefore be
    temporarily kinematic; otherwise gravity acts on the dynamic iiwa chain
    before the visual-servo controller has been created and the model can fall
    apart.  Only the seven joint states/modes are changed here.  Descendant
    poses are intentionally never overwritten.
    """
    kinematic_mode = int(getattr(sim, "jointmode_kinematic", 0))
    dynctrl_param = getattr(sim, "jointintparam_dynctrlmode", None)
    position_mode = int(getattr(sim, "jointdynctrl_position", 8))
    for handle, position in configuration.get("joints", []):
        try:
            sim.setJointMode(int(handle), kinematic_mode)
            sim.setJointPosition(int(handle), float(position))
            if dynctrl_param is not None:
                try:
                    sim.setObjectInt32Param(int(handle), dynctrl_param, position_mode)
                except Exception:
                    pass
        except Exception:
            continue


def _restore_joint_configuration(sim: Any, configuration: dict[str, Any]) -> None:
    """Restore q and joint actuator modes, without restoring link poses."""
    for handle, position in configuration.get("joints", []):
        try:
            sim.setJointPosition(int(handle), float(position))
        except Exception:
            pass
    for handle, mode in configuration.get("joint_modes", []):
        try:
            sim.setJointMode(int(handle), int(mode))
        except Exception:
            pass
    dynctrl_param = getattr(sim, "jointintparam_dynctrlmode", None)
    if dynctrl_param is not None:
        for handle, dynctrl in configuration.get("joint_dynctrl_modes", []):
            if dynctrl is None:
                continue
            try:
                sim.setObjectInt32Param(int(handle), dynctrl_param, int(dynctrl))
            except Exception:
                pass


def _restore_robot_configuration(
    sim: Any,
    robot_base: int,
    configuration: dict[str, Any],
) -> None:
    # The base and joint coordinates are enough.  Restoring every descendant
    # pose fights the joint/force-sensor constraints and can break a dynamic
    # chain such as link8_resp -> connection -> RG2.
    try:
        sim.setObjectPose(robot_base, configuration["base_pose_world"], -1)
    except Exception:
        pass
    _restore_joint_configuration(sim, configuration)


def _set_engine_float(sim: Any, handle: int, parameter_name: str, value: float) -> None:
    parameter = getattr(sim, parameter_name, None)
    if parameter is None:
        return
    try:
        sim.setEngineFloatParam(parameter, handle, float(value))
    except Exception:
        pass


def _set_engine_bool(sim: Any, handle: int, parameter_name: str, value: bool) -> None:
    parameter = getattr(sim, parameter_name, None)
    if parameter is None:
        return
    try:
        sim.setEngineBoolParam(parameter, handle, bool(value))
    except Exception:
        pass


def _set_dynamic(sim: Any, handle: int, dynamic: bool) -> None:
    _set_shape_flag(sim, handle, "shapeintparam_static", 0 if dynamic else 1)
    _set_shape_flag(sim, handle, "shapeintparam_respondable", 1)
    _set_shape_flag(
        sim,
        handle,
        "shapeintparam_respondable_mask",
        DROP_COLLISION_BIT,
    )
    _set_engine_float(
        sim,
        handle,
        "bullet_body_friction",
        float(os.environ.get("ROBOT_GRASP_PHYSICS_FRICTION", "1.2")),
    )
    _set_engine_float(
        sim,
        handle,
        "bullet_body_restitution",
        float(os.environ.get("ROBOT_GRASP_PHYSICS_RESTITUTION", "0.0")),
    )
    _set_engine_bool(
        sim,
        handle,
        "bullet_body_sticky",
        os.environ.get("ROBOT_GRASP_PHYSICS_STICKY", "1").lower()
        not in {"0", "false", "no"},
    )

    if dynamic:
        _set_engine_float(
            sim,
            handle,
            "bullet_body_lineardamping",
            float(os.environ.get("ROBOT_GRASP_PHYSICS_LINEAR_DAMPING", "0.08")),
        )
        _set_engine_float(
            sim,
            handle,
            "bullet_body_angulardamping",
            float(os.environ.get("ROBOT_GRASP_PHYSICS_ANGULAR_DAMPING", "0.15")),
        )
        density = float(os.environ.get("ROBOT_GRASP_PHYSICS_DENSITY", "800"))
        try:
            sim.computeMassAndInertia(handle, density)
        except Exception:
            pass


def _create_primitive(sim: Any, spec: ShapeSpec, size_xyz: tuple[float, float, float]) -> int:
    primitive_constant = {
        "cuboid": "primitiveshape_cuboid",
        "cylinder": "primitiveshape_cylinder",
        "spheroid": "primitiveshape_spheroid",
        "cone": "primitiveshape_cone",
    }[spec.primitive_name]
    primitive_type = getattr(sim, primitive_constant)
    handle = int(sim.createPrimitiveShape(primitive_type, list(size_xyz), 0))
    _set_dynamic(sim, handle, dynamic=True)
    return handle


def _make_containment_walls(
    sim: Any,
    robot_base: int,
    reference: dict[str, Any],
) -> list[int]:
    if not CREATE_CONTAINMENT_WALLS:
        return []

    half_x = float(reference.get("workspace_half_x", 0.11))
    half_y = float(reference.get("workspace_half_y", 0.09))
    center_x = float(reference["workspace_center_x"])
    center_y = float(reference["workspace_center_y"])
    table_z = float(reference["table_z"])
    wall_height = 0.08
    thickness = 0.006
    wall_specs = [
        ("physics_wall_x_min", (thickness, 2.0 * half_y, wall_height), center_x - half_x),
        ("physics_wall_x_max", (thickness, 2.0 * half_y, wall_height), center_x + half_x),
        ("physics_wall_y_min", (2.0 * half_x, thickness, wall_height), center_y - half_y),
        ("physics_wall_y_max", (2.0 * half_x, thickness, wall_height), center_y + half_y),
    ]
    created: list[int] = []
    for alias, size, coordinate in wall_specs:
        handle = int(
            sim.createPrimitiveShape(
                getattr(sim, "primitiveshape_cuboid"),
                list(size),
                0,
            )
        )
        _set_dynamic(sim, handle, dynamic=False)
        sim.setObjectAlias(handle, alias)
        if alias.startswith("physics_wall_x"):
            pose = [coordinate, center_y, table_z + wall_height / 2.0, 0.0, 0.0, 0.0, 1.0]
        else:
            pose = [center_x, coordinate, table_z + wall_height / 2.0, 0.0, 0.0, 0.0, 1.0]
        sim.setObjectPose(handle, pose, robot_base)
        # Hide the wall from the vision sensor when the parameter exists.
        visibility_param = getattr(sim, "objintparam_visibility_layer", None)
        if visibility_param is not None:
            try:
                sim.setObjectInt32Param(handle, visibility_param, 0)
            except Exception:
                pass
        created.append(handle)
    return created


def _remove_old_physics_objects(sim: Any) -> None:
    handles: list[int] = []
    shapes = sim.getObjectsInTree(sim.handle_scene, sim.sceneobject_shape, 0)
    for handle in shapes:
        try:
            alias = str(sim.getObjectAlias(handle))
        except Exception:
            continue
        if alias.lower().startswith(("rand_", "physics_wall_")):
            handles.append(int(handle))
    if handles:
        sim.removeObjects(handles)


def _collect_contacts(sim: Any, handles: set[int], aliases: dict[int, str]) -> list[dict[str, Any]]:
    contacts: dict[tuple[str, str], dict[str, Any]] = {}
    get_contact_info = getattr(sim, "getContactInfo", None)
    if get_contact_info is None:
        return []
    dynamic_pass = getattr(sim, "handle_all", -1)
    for index in range(256):
        try:
            coll, point, force, normal = get_contact_info(dynamic_pass, dynamic_pass, index)
        except Exception:
            break
        if not coll or len(coll) < 2:
            continue
        first, second = int(coll[0]), int(coll[1])
        if first not in handles and second not in handles:
            continue
        for handle in (first, second):
            if handle not in aliases:
                try:
                    aliases[handle] = str(sim.getObjectAlias(handle))
                except Exception:
                    aliases[handle] = str(handle)
        names = tuple(sorted((aliases[first], aliases[second])))
        contacts[names] = {
            "object_a": names[0],
            "object_b": names[1],
            "point_m": [float(v) for v in point],
            "normal": [float(v) for v in normal],
            "force": [float(v) for v in force],
        }
    return list(contacts.values())


def _speed_stats(sim: Any, handles: list[int]) -> tuple[float, float]:
    max_linear = 0.0
    max_angular = 0.0
    for handle in handles:
        linear, angular = sim.getObjectVelocity(handle)
        max_linear = max(max_linear, float(np.linalg.norm(linear)))
        max_angular = max(max_angular, float(np.linalg.norm(angular)))
    return max_linear, max_angular


def _zero_object_velocities(sim: Any, handles: list[int]) -> None:
    """Stop residual motion without changing a body's dynamic state."""

    set_velocity = getattr(sim, "setObjectVelocity", None)
    if set_velocity is None:
        return
    for handle in handles:
        try:
            set_velocity(int(handle), [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        except Exception:
            # A paused Bullet body may reject an explicit velocity reset. The
            # settle thresholds already guarantee that the residual motion is
            # small, so this is only a best-effort cleanup.
            pass


def _sample_initial_position(
    rng: np.random.Generator,
    center_x: float,
    center_y: float,
    half_x: float,
    half_y: float,
    table_z: float,
    size_xyz: tuple[float, float, float],
    existing: list[tuple[np.ndarray, float]],
) -> tuple[float, float, float]:
    """Find a random drop pose with limited horizontal overlap."""

    radius_xy = footprint_radius(size_xyz)
    body_radius = 0.5 * float(np.linalg.norm(size_xyz))
    min_x = center_x - half_x + radius_xy
    max_x = center_x + half_x - radius_xy
    min_y = center_y - half_y + radius_xy
    max_y = center_y + half_y - radius_xy
    if min_x > max_x or min_y > max_y:
        raise RuntimeError(
            "物理工作区小于工件占地范围，请增大工作区或缩小CAD尺寸。"
        )

    for _ in range(max(INITIAL_POSE_TRIES, 1)):
        x = float(rng.uniform(min_x, max_x))
        y = float(rng.uniform(min_y, max_y))
        z = table_z + DROP_HEIGHT_M + 0.5 * max(size_xyz) + float(
            rng.uniform(0.0, max(INITIAL_HEIGHT_JITTER_M, 0.0))
        )
        candidate = np.array([x, y, z], dtype=np.float64)
        if all(
            float(np.linalg.norm(candidate[:2] - previous[:2]))
            > INITIAL_XY_CLEARANCE_RATIO * (body_radius + previous_radius)
            + INITIAL_XY_MARGIN_M
            for previous, previous_radius in existing
        ):
            return x, y, z

    # A compact workspace can make a separated horizontal sample unlikely.
    # Stack the fallback body above the current highest body; users can also
    # lower ROBOT_GRASP_PHYSICS_XY_CLEARANCE_RATIO for denser contact scenes.
    x = float(rng.uniform(min_x, max_x))
    y = float(rng.uniform(min_y, max_y))
    z = table_z + DROP_HEIGHT_M + 0.5 * max(size_xyz)
    for previous, previous_radius in existing:
        z = max(z, float(previous[2]) + body_radius + previous_radius + 0.008)
    return x, y, z


def _settle_simulation(
    sim: Any,
    handles: list[int],
    aliases: dict[int, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient

    step_client = RemoteAPIClient()
    step_client.setStepping(True)
    sim.startSimulation()
    started = time.monotonic()
    stable_steps = 0
    window_size = max(STABLE_STEPS_REQUIRED * 2, STABLE_STEPS_REQUIRED)
    quiet_window: deque[bool] = deque(maxlen=window_size)
    max_linear = float("inf")
    max_angular = float("inf")
    while time.monotonic() - started < SETTLE_TIMEOUT_S:
        step_client.step()
        max_linear, max_angular = _speed_stats(sim, handles)
        quiet_window.append(
            max_linear <= LINEAR_SPEED_THRESHOLD
            and max_angular <= ANGULAR_SPEED_THRESHOLD
        )
        stable_steps = int(sum(quiet_window))
        if len(quiet_window) == window_size and stable_steps >= STABLE_STEPS_REQUIRED:
            break

    timed_out = not (
        len(quiet_window) == window_size
        and stable_steps >= STABLE_STEPS_REQUIRED
    )
    final_contacts = _collect_contacts(sim, set(handles), aliases)
    sim.pauseSimulation()
    # Keep the settled bodies dynamic. Pausing the simulation is sufficient
    # for deterministic RGB-D capture; changing shapeintparam_static to 1
    # would make a later RG2 grasp physically unable to lift the object.
    _zero_object_velocities(sim, handles)
    try:
        step_client.setStepping(False)
    except Exception:
        pass
    settle_info = {
        "settled_by_timeout": timed_out,
        "quiet_steps_in_window": stable_steps,
        "quiet_window_size": window_size,
        "final_max_linear_speed_m_s": max_linear,
        "final_max_angular_speed_rad_s": max_angular,
    }
    if timed_out:
        print(
            "警告：物理场景达到等待上限，已冻结当前姿态。"
            f" max_linear={max_linear:.5f} m/s,"
            f" max_angular={max_angular:.5f} rad/s"
        )
    return final_contacts, settle_info


def generate_physics_scene(
    sim: Any,
    robot_base: int,
    reference: dict[str, Any],
    camera_model: dict[str, Any],
) -> list[dict[str, Any]]:
    """Generate, settle and pause a dynamic scene, returning final GT records."""

    from scene_randomizer import COLOR_PALETTE

    _remove_old_physics_objects(sim)
    seed = int(os.environ.get("ROBOT_GRASP_RANDOM_SEED", "0") or 0)
    rng = np.random.default_rng(seed if seed else None)
    minimum = max(5, min(MIN_OBJECTS, MAX_OBJECTS))
    maximum = max(minimum, MAX_OBJECTS)
    object_count = int(rng.integers(minimum, maximum + 1))
    round_class = os.environ.get("ROBOT_GRASP_PHYSICS_ROUND_SHAPE", "sphere").lower()
    if round_class not in {"sphere", "spheroid"}:
        raise ValueError("ROBOT_GRASP_PHYSICS_ROUND_SHAPE must be sphere or spheroid")
    classes = ["cube", "cuboid", "cylinder", round_class, "cone"]
    while len(classes) < object_count:
        classes.append(str(rng.choice(classes)))
    rng.shuffle(classes)

    table_z = float(reference["table_z"])
    center_x = float(reference["workspace_center_x"])
    center_y = float(reference["workspace_center_y"])
    half_x = float(reference.get("workspace_half_x", 0.11))
    half_y = float(reference.get("workspace_half_y", 0.09))

    handles: list[int] = []
    aliases: dict[int, str] = {}
    records: list[dict[str, Any]] = []
    initial_positions: list[tuple[np.ndarray, float]] = []
    wall_handles: list[int] = []
    robot_collision_state: list[tuple[int, int]] = []
    robot_configuration = _capture_robot_configuration(sim, robot_base)
    counters = {name: 0 for name in classes}
    try:
        # Workpiece settling must not be allowed to pull the uncommanded iiwa
        # through gravity.  The real dynamic controller is installed later by
        # visual_servo_runner, after perception has finished.
        _set_robot_kinematic_for_settle(sim, robot_configuration)
        robot_collision_state = _suspend_robot_collisions(sim, robot_base)
        wall_handles = _make_containment_walls(sim, robot_base, reference)
        for index, class_name in enumerate(classes):
            spec = get_shape_spec(class_name)
            # Physics scenes use the same finite primitive catalogue as the
            # geometry recognizer so class and size hypotheses remain bounded.
            size_xyz = get_cad_dimensions(spec.cad_id)
            radius = footprint_radius(size_xyz)
            x, y, z = _sample_initial_position(
                rng,
                center_x,
                center_y,
                half_x,
                half_y,
                table_z,
                size_xyz,
                initial_positions,
            )
            quaternion = _random_quaternion(rng)
            handle = _create_primitive(sim, spec, size_xyz)
            handles.append(handle)
            counters[class_name] += 1
            alias = f"rand_{class_name}_{counters[class_name]:02d}"
            aliases[handle] = alias
            sim.setObjectAlias(handle, alias)
            sim.setObjectPose(handle, [x, y, z, *quaternion], robot_base)
            initial_positions.append(
                (
                    np.array([x, y, z], dtype=np.float64),
                    0.5 * float(np.linalg.norm(size_xyz)),
                )
            )
            set_color = getattr(sim, "setShapeColor", None)
            if set_color is not None:
                try:
                    set_color(handle, "", sim.colorcomponent_ambient_diffuse, COLOR_PALETTE[index % len(COLOR_PALETTE)])
                except Exception:
                    pass
            records.append(
                {
                    "handle": handle,
                    "alias": alias,
                    "class": class_name,
                    "shape_id": spec.cad_id,
                    "primitive_name": spec.primitive_name,
                    "size_m": [float(v) for v in size_xyz],
                    "geometry": dimensions_to_geometry(class_name, size_xyz),
                    "symmetry": spec.symmetry,
                    "grasp_family": spec.grasp_family,
                    "initial_pose": [float(v) for v in [x, y, z, *quaternion]],
                    "footprint_radius": float(radius),
                    "scenario_role": "physics_settled",
                    "stack_level": 0,
                }
            )

        contacts, settle_info = _settle_simulation(sim, handles, aliases)
        _restore_robot_configuration(sim, robot_base, robot_configuration)
        _restore_robot_collisions(sim, robot_collision_state)
        robot_collision_state = []
        for item in records:
            pose = sim.getObjectPose(item["handle"], robot_base)
            item["pose_base"] = [float(v) for v in pose]
            item["position"] = [float(v) for v in pose[:3]]
            item["quaternion"] = [float(v) for v in pose[3:]]
            item["dynamic_settled"] = True
            item["frozen_after_settle"] = False
            item["settled_by_timeout"] = bool(settle_info["settled_by_timeout"])

        CONTACT_FILE.write_text(
            json.dumps(
                {
                    "scene_mode": "physics",
                    "settle_timeout_s": SETTLE_TIMEOUT_S,
                    "linear_speed_threshold_m_s": LINEAR_SPEED_THRESHOLD,
                    "angular_speed_threshold_rad_s": ANGULAR_SPEED_THRESHOLD,
                    "stable_steps_required": STABLE_STEPS_REQUIRED,
                    "settle_result": settle_info,
                    "contacts": contacts,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return records
    except Exception:
        try:
            _restore_robot_configuration(sim, robot_base, robot_configuration)
        except Exception:
            pass
        if robot_collision_state:
            _restore_robot_collisions(sim, robot_collision_state)
        cleanup_handles = [*handles, *wall_handles]
        if cleanup_handles:
            try:
                sim.removeObjects(cleanup_handles)
            except Exception:
                pass
        raise
