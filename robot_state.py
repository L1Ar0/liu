"""Shared robot reset and dynamic-handoff state for the CoppeliaSim scene.

The iiwa/RG2 model is a hierarchy of visual meshes, dynamic proxy shapes,
force sensors and gripper links.  Changing a joint mode alone is not a safe
reset: a previous interrupted dynamics run can leave descendant bodies with
non-zero velocity, while the next scene can freeze a different subset of
shapes.  This module stores the original shape flags once per scene and uses
that snapshot for both the perception reset and the dynamic controller.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable



MANIFEST_FILE = Path("robot_dynamic_state.json")


def _param(sim: Any, name: str) -> int | None:
    value = getattr(sim, name, None)
    return None if value is None else int(value)


def _get_int(sim: Any, handle: int, name: str, default: int) -> int:
    parameter = _param(sim, name)
    if parameter is None:
        return int(default)
    try:
        return int(sim.getObjectInt32Param(int(handle), parameter))
    except Exception:
        return int(default)


def _set_int(sim: Any, handle: int, name: str, value: int) -> None:
    parameter = _param(sim, name)
    if parameter is None:
        return
    try:
        sim.setObjectInt32Param(int(handle), parameter, int(value))
    except Exception:
        pass


def _path(sim: Any, handle: int) -> str:
    try:
        return str(sim.getObjectAlias(int(handle), 2))
    except Exception:
        return str(sim.getObjectAlias(int(handle)))


def _handle_by_path(sim: Any, path: str) -> int | None:
    try:
        value = sim.getObject(str(path), {"noError": True})
        if value is not None and int(value) >= 0:
            return int(value)
    except Exception:
        pass
    return None


def _iter_shapes(sim: Any, robot_base: int) -> Iterable[int]:
    return (int(value) for value in sim.getObjectsInTree(int(robot_base), sim.sceneobject_shape, 0))


def canonicalize_dynamic_shapes(sim: Any, robot_base: int) -> list[int]:
    """Repair stale flags left by an interrupted run.

    The tuned scene uses respondable collision proxies for the arm and RG2,
    while the root model and visual meshes are static.  ``stopSimulation``
    does not restore shape flags changed through the remote API, so a failed
    run can leave every proxy static.  Respondable is the reliable semantic
    marker here; the root ``/iiwa`` shape is the only exception.
    """

    changed: list[int] = []
    root = int(robot_base)
    for handle in _iter_shapes(sim, root):
        if handle == root:
            _set_int(sim, handle, "shapeintparam_static", 1)
            continue
        respondable = _get_int(sim, handle, "shapeintparam_respondable", 0)
        if respondable == 1 and _get_int(sim, handle, "shapeintparam_static", 1) != 0:
            _set_int(sim, handle, "shapeintparam_static", 0)
            changed.append(handle)
    return changed


def capture(sim: Any, robot_base: int) -> dict[str, Any]:
    """Capture the exact robot shape state before scene-generation freezing."""

    # First repair flags from a previous interrupted dynamic run.  This is
    # intentionally done only when a new baseline is captured, never during
    # perception where the robot is meant to remain frozen.
    canonicalize_dynamic_shapes(sim, robot_base)
    shapes: list[dict[str, Any]] = []
    for handle in _iter_shapes(sim, robot_base):
        try:
            record = {
                "path": _path(sim, handle),
                "alias": str(sim.getObjectAlias(handle)),
                "static": _get_int(sim, handle, "shapeintparam_static", 1),
                "respondable": _get_int(sim, handle, "shapeintparam_respondable", 0),
                "respondable_mask": _get_int(sim, handle, "shapeintparam_respondable_mask", 0xFFFF),
            }
            shapes.append(record)
        except Exception:
            continue
    try:
        base_pose = [float(value) for value in sim.getObjectPose(int(robot_base), -1)]
    except Exception:
        base_pose = None
    return {
        "version": 1,
        "robot_base_path": _path(sim, int(robot_base)),
        "robot_base_pose_world": base_pose,
        "shapes": shapes,
        "created_at": time.time(),
    }


def save(manifest: dict[str, Any], path: Path = MANIFEST_FILE) -> None:
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def load(path: Path = MANIFEST_FILE) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) and isinstance(value.get("shapes"), list) else None
    except Exception:
        return None


def load_compatible(sim: Any, robot_base: int, path: Path = MANIFEST_FILE) -> dict[str, Any] | None:
    """Load a manifest only when it describes the currently open scene tree."""

    manifest = load(path)
    if manifest is None:
        return None
    try:
        if str(manifest.get("robot_base_path")) != _path(sim, int(robot_base)):
            return None
        saved_paths = {
            str(item.get("path"))
            for item in manifest.get("shapes", [])
            if isinstance(item, dict)
        }
        current_paths = {_path(sim, handle) for handle in _iter_shapes(sim, robot_base)}
        if not saved_paths or saved_paths != current_paths:
            return None
    except Exception:
        return None
    return manifest


def resolve_manifest_handles(sim: Any, manifest: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    resolved: list[tuple[int, dict[str, Any]]] = []
    for record in manifest.get("shapes", []):
        if not isinstance(record, dict):
            continue
        handle = _handle_by_path(sim, str(record.get("path", "")))
        if handle is None:
            # Alias fallback is deliberately restricted to one match.  It is
            # useful for scenes whose object paths gained an instance suffix,
            # but never silently chooses between duplicate finger links.
            alias = str(record.get("alias", ""))
            matches = []
            for candidate in sim.getObjectsInTree(sim.handle_scene, sim.sceneobject_shape, 0):
                try:
                    if str(sim.getObjectAlias(candidate)) == alias:
                        matches.append(int(candidate))
                except Exception:
                    pass
            if len(matches) == 1:
                handle = matches[0]
        if handle is not None:
            resolved.append((int(handle), record))
    return resolved


def apply_shape_state(
    sim: Any,
    manifest: dict[str, Any],
    *,
    freeze: bool = False,
    reset_dynamic: bool = False,
) -> list[int]:
    """Apply exact saved flags, or freeze all saved shapes when requested."""

    changed_dynamic: list[int] = []
    for handle, record in resolve_manifest_handles(sim, manifest):
        if freeze:
            _set_int(sim, handle, "shapeintparam_static", 1)
        else:
            static = int(record.get("static", 1))
            respondable = int(record.get("respondable", 0))
            mask = int(record.get("respondable_mask", 0xFFFF))
            _set_int(sim, handle, "shapeintparam_static", static)
            _set_int(sim, handle, "shapeintparam_respondable", respondable)
            _set_int(sim, handle, "shapeintparam_respondable_mask", mask)
            if static == 0:
                changed_dynamic.append(handle)
                if reset_dynamic:
                    reset = getattr(sim, "resetDynamicObject", None)
                    if reset is not None:
                        try:
                            reset(int(handle))
                        except Exception:
                            pass
    return changed_dynamic


def zero_robot_velocities(sim: Any, robot_base: int) -> None:
    setter = getattr(sim, "setObjectVelocity", None)
    if setter is None:
        return
    for handle in _iter_shapes(sim, robot_base):
        try:
            setter(handle, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        except Exception:
            pass


def set_arm_kinematic(sim: Any, joints: Iterable[int], positions: Iterable[float]) -> None:
    mode = int(getattr(sim, "jointmode_kinematic", 0))
    for joint, position in zip(joints, positions):
        try:
            sim.setJointMode(int(joint), mode)
            sim.setJointPosition(int(joint), float(position))
        except Exception:
            pass


def reset_for_observation(
    sim: Any,
    robot_base: int,
    joints: list[int],
    pose: Iterable[float],
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the full robot to a quiet, kinematic observation state."""

    state = manifest or capture(sim, robot_base)
    if manifest is None:
        save(state)
    # Freeze before changing q.  This prevents a dynamic descendant from
    # fighting the joint pose assignment after an interrupted run.
    apply_shape_state(sim, state, freeze=True)
    try:
        sim.setObjectPose(int(robot_base), list(state.get("robot_base_pose_world") or pose), -1)
    except Exception:
        pass
    set_arm_kinematic(sim, joints, pose)
    zero_robot_velocities(sim, robot_base)
    return state


def restore_original_shape_state(sim: Any, manifest: dict[str, Any] | None) -> None:
    if manifest is None:
        return
    apply_shape_state(sim, manifest, freeze=False, reset_dynamic=False)
