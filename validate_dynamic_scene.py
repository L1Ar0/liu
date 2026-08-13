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

from coppeliasim_zmqremoteapi_client import RemoteAPIClient


ROOT = Path(__file__).resolve().parent


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
        "dynamic_after_step": bool(dynamic_after_step),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3, help="synchronous steps used for the post-start check")
    parser.add_argument("--start", action="store_true", help="temporarily start and step the simulation")
    parser.add_argument("--json", type=Path, help="optional report output path")
    args = parser.parse_args()

    client = RemoteAPIClient()
    sim = client.require("sim")
    if sim.getSimulationState() != sim.simulation_stopped:
        raise RuntimeError("Stop CoppeliaSim before running the validator")

    before = collect_report(sim, False)
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

    report = {"before_start": before, "after_start": after}
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
