"""Commissioning/diagnostic report for the CoppeliaSim iiwa + RG2 model.

This script deliberately does not move the robot or modify any object.  It
reports whether the scene is actually prepared for dynamic control, then
advances a few synchronous physics steps and reports dynamic enablement again.
Use it before a physical grasp; a successful Python import or a force API call
alone does not prove that the scene is dynamically valid.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from joint_torque_controller import JointTorqueController, TorqueControllerConfig


ROOT = Path(__file__).resolve().parent


def _normalize_alias(value: str) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def _find_alias(sim: Any, alias: str) -> int | None:
    wanted = _normalize_alias(alias)
    for handle in sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 0):
        try:
            current = _normalize_alias(sim.getObjectAlias(handle))
        except Exception:
            continue
        if current == wanted:
            return int(handle)
    return None


def _attachment_report(sim: Any) -> dict[str, Any]:
    """Report the physical flange -> force sensor -> RG2 chain.

    A dummy named ``iiwa_tip`` may remain as a TCP/reference frame, but it
    must not be the physical parent of RG2.  The exact force-sensor type
    constants differ between remote API releases, so the report is
    deliberately tolerant and includes both the reported type and aliases.
    """
    connection = _find_alias(sim, "connection")
    rg2 = _find_alias(sim, "RG2")
    tip = _find_alias(sim, "iiwa_tip")
    link8 = _find_alias(sim, "link8_resp")

    def parent_record(handle: int | None) -> dict[str, Any] | None:
        if handle is None:
            return None
        try:
            parent = int(sim.getObjectParent(handle))
            parent_alias = str(sim.getObjectAlias(parent)) if parent >= 0 else None
        except Exception:
            parent, parent_alias = None, None
        try:
            object_type = int(sim.getObjectType(handle))
        except Exception:
            object_type = None
        return {
            "handle": int(handle),
            "alias": str(sim.getObjectAlias(handle)),
            "object_type": object_type,
            "parent_handle": parent,
            "parent_alias": parent_alias,
            "static": _int_param(sim, int(handle), "shapeintparam_static", None),
            "dynamically_enabled": bool(sim.isDynamicallyEnabled(handle))
            if hasattr(sim, "isDynamicallyEnabled")
            else None,
        }

    connection_parent = None
    rg2_parent = None
    if connection is not None:
        try:
            connection_parent = int(sim.getObjectParent(connection))
        except Exception:
            pass
    if rg2 is not None:
        try:
            rg2_parent = int(sim.getObjectParent(rg2))
        except Exception:
            pass
    expected_parent = connection_parent == link8 if connection_parent is not None else False
    rg2_attached_to_sensor = rg2_parent == connection if rg2_parent is not None else False
    return {
        "connection": parent_record(connection),
        "rg2": parent_record(rg2),
        "iiwa_tip": parent_record(tip),
        "link8_resp": parent_record(link8),
        "connection_parent_is_link8_resp": expected_parent,
        "rg2_parent_is_connection": rg2_attached_to_sensor,
        "iiwa_tip_is_rg2_parent": tip is not None and rg2_parent == tip,
    }


def _run_dynamic_hold_test(
    client: RemoteAPIClient,
    sim: Any,
    joints: list[int],
    steps: int,
    control_mode: str,
) -> dict[str, Any]:
    """Hold the measured seven-joint pose under the actual dynamics layer."""
    if len(joints) != 7:
        raise RuntimeError(f"dynamic hold test requires 7 joints, found {len(joints)}")
    controller = JointTorqueController(
        client,
        sim,
        joints,
        TorqueControllerConfig.from_environment(),
        control_mode=control_mode,
    )
    q0 = controller.read_positions()
    max_error = 0.0
    max_velocity = 0.0
    samples = 0
    try:
        controller.start()
        controller.qd = q0.copy()
        for _ in range(max(1, int(steps))):
            result = controller.step(q0, steps=1)
            q = np.asarray(result["q"], dtype=np.float64)
            dq = np.asarray(result["dq"], dtype=np.float64)
            max_error = max(max_error, float(np.max(np.abs(q - q0))))
            max_velocity = max(max_velocity, float(np.max(np.abs(dq))))
            samples += 1
        return {
            "control_mode": str(control_mode),
            "steps": samples,
            "initial_q_rad": q0.tolist(),
            "max_position_error_rad": max_error,
            "max_position_error_deg": math.degrees(max_error),
            "max_velocity_rad_s": max_velocity,
            "final_diagnostics": controller.diagnostics(),
        }
    finally:
        controller.close()


def _int_param(sim: Any, handle: int, name: str, default: int | None = None) -> int | None:
    parameter = getattr(sim, name, None)
    if parameter is None:
        return default
    try:
        return int(sim.getObjectInt32Param(handle, parameter))
    except Exception:
        return default


def _shape_record(sim: Any, handle: int) -> dict[str, Any]:
    alias = str(sim.getObjectAlias(handle))
    try:
        mass = float(sim.getShapeMass(handle))
    except Exception:
        mass = None
    try:
        dynamic = bool(sim.isDynamicallyEnabled(handle))
    except Exception:
        dynamic = None
    return {
        "handle": int(handle),
        "alias": alias,
        "static": _int_param(sim, handle, "shapeintparam_static"),
        "respondable": _int_param(sim, handle, "shapeintparam_respondable"),
        "mass_kg": mass,
        "dynamically_enabled": dynamic,
    }


def collect_report(sim: Any, dynamic_after_step: bool = False) -> dict[str, Any]:
    iiwa = int(sim.getObject("/iiwa", {"noError": True}))
    if iiwa < 0:
        raise RuntimeError("/iiwa was not found in the open CoppeliaSim scene")
    joints = [
        int(handle)
        for handle in sim.getObjectsInTree(iiwa, sim.sceneobject_joint, 0)
        if str(sim.getObjectAlias(handle)).lower() in {f"joint{i}" if i > 1 else "joint" for i in range(1, 8)}
    ]
    joints.sort(key=lambda h: int("".join(c for c in str(sim.getObjectAlias(h)) if c.isdigit()) or "1"))
    joint_records = []
    for handle in joints:
        try:
            q = float(sim.getJointPosition(handle))
        except Exception:
            q = None
        try:
            dq = float(sim.getJointVelocity(handle))
        except Exception:
            dq = None
        joint_records.append(
            {
                "handle": handle,
                "alias": str(sim.getObjectAlias(handle)),
                "joint_mode": _int_param(sim, handle, "jointintparam_mode"),
                "reported_joint_mode": sim.getJointMode(handle),
                "dynctrl_mode": _int_param(sim, handle, "jointintparam_dynctrlmode"),
                "q_rad": q,
                "dq_rad_s": dq,
            }
        )
    shapes = [
        _shape_record(sim, int(handle))
        for handle in sim.getObjectsInTree(iiwa, sim.sceneobject_shape, 0)
    ]
    rg2 = []
    for handle in sim.getObjectsInTree(sim.handle_scene, sim.sceneobject_joint, 0):
        alias = str(sim.getObjectAlias(handle)).lower()
        if "openclose" in alias:
            rg2.append(
                {
                    "handle": int(handle),
                    "alias": str(sim.getObjectAlias(handle)),
                    "joint_mode": sim.getJointMode(handle),
                    "dynctrl_mode": _int_param(sim, handle, "jointintparam_dynctrlmode"),
                }
            )
    return {
        "simulation_state": int(sim.getSimulationState()),
        "simulation_time_s": float(sim.getSimulationTime()),
        "iiwa_handle": iiwa,
        "iiwa_joint_count": len(joints),
        "iiwa_joints": joint_records,
        "iiwa_shapes": shapes,
        "rg2_drive_joints": rg2,
        "attachment": _attachment_report(sim),
        "dynamic_after_step": bool(dynamic_after_step),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3, help="synchronous steps used for the post-start check")
    parser.add_argument("--start", action="store_true", help="temporarily start and step the simulation")
    parser.add_argument(
        "--hold-steps",
        type=int,
        default=0,
        help="run a seven-joint dynamic pose-hold test for this many physics steps",
    )
    parser.add_argument(
        "--control-mode",
        choices=("position", "torque"),
        default="position",
        help="actuator mode used by --hold-steps",
    )
    parser.add_argument("--json", type=Path, help="optional report output path")
    args = parser.parse_args()

    client = RemoteAPIClient()
    sim = client.require("sim")
    if sim.getSimulationState() != sim.simulation_stopped:
        raise RuntimeError("Stop CoppeliaSim before running the validator")

    before = collect_report(sim, False)
    hold = None
    if args.hold_steps > 0:
        joint_handles = [int(item["handle"]) for item in before["iiwa_joints"]]
        hold = _run_dynamic_hold_test(
            client,
            sim,
            joint_handles,
            int(args.hold_steps),
            str(args.control_mode),
        )
    after = None
    if args.start:
        client.setStepping(True)
        sim.startSimulation()
        for _ in range(max(1, int(args.steps))):
            client.step()
        after = collect_report(sim, True)
        sim.stopSimulation()
        deadline = time.monotonic() + 5.0
        while sim.getSimulationState() != sim.simulation_stopped and time.monotonic() < deadline:
            time.sleep(0.02)

    report = {"before_start": before, "dynamic_hold": hold, "after_start": after}
    if args.json:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    joints = before["iiwa_joints"]
    if len(joints) != 7:
        print(f"WARNING: expected 7 iiwa joints, found {len(joints)}")
    if any(item.get("static") == 0 and item.get("respondable") == 1 for item in before["iiwa_shapes"]):
        print("iiwa dynamic link shapes: present")
    else:
        print("WARNING: no dynamic/respondable iiwa link shapes found")
    if after is not None and not any(item.get("dynamically_enabled") for item in after["iiwa_shapes"]):
        print("WARNING: no iiwa shape was dynamically enabled after physics steps")
    attachment = before.get("attachment", {})
    if not attachment.get("connection_parent_is_link8_resp", False):
        print("WARNING: connection is not parented directly under link8_resp")
    if not attachment.get("rg2_parent_is_connection", False):
        print("WARNING: RG2 is not parented directly under connection")
    if attachment.get("iiwa_tip_is_rg2_parent", False):
        print("WARNING: iiwa_tip is the physical RG2 parent; use it only as a TCP/reference dummy")
    if after is not None:
        post_attachment = after.get("attachment", {})
        if post_attachment.get("connection", {}).get("dynamically_enabled") is not True:
            print("WARNING: connection force-sensor body was not dynamically enabled")
        if post_attachment.get("rg2", {}).get("dynamically_enabled") is not True:
            print("WARNING: RG2 root was not dynamically enabled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
