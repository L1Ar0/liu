"""Persist and restore the canonical first RGB-D observation pose."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from point_cloud import find_unique_object_by_alias, get_kuka_joints_from_tip


POSE_FILE = Path("camera_output") / "initial_observation_joints.json"


def _connect():
    client = RemoteAPIClient()
    sim = client.require("sim")
    gripper_tip = find_unique_object_by_alias(
        sim,
        sim.sceneobject_dummy,
        "gripper_tip",
    )
    joints = get_kuka_joints_from_tip(sim, gripper_tip)
    if len(joints) != 7:
        raise RuntimeError(f"Expected 7 arm joints, found {len(joints)}")
    return sim, joints


def capture_current(sim, joints) -> None:
    POSE_FILE.parent.mkdir(parents=True, exist_ok=True)
    values = [float(sim.getJointPosition(joint)) for joint in joints]
    POSE_FILE.write_text(
        json.dumps(
            {
                "joint_positions_rad": values,
                "joint_positions_deg": [math.degrees(value) for value in values],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Initial observation pose saved: {POSE_FILE.resolve()}")


def restore(sim, joints) -> None:
    payload = json.loads(POSE_FILE.read_text(encoding="utf-8"))
    values = [float(value) for value in payload["joint_positions_rad"]]
    if len(values) != len(joints):
        raise RuntimeError("Saved observation pose has the wrong joint count")
    for joint, value in zip(joints, values):
        sim.setJointPosition(joint, value)
    print(
        "Initial observation pose restored: "
        + ", ".join(f"{math.degrees(value):.2f} deg" for value in values)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-current", action="store_true")
    args = parser.parse_args()
    sim, joints = _connect()
    if args.capture_current or not POSE_FILE.exists():
        capture_current(sim, joints)
    else:
        restore(sim, joints)


if __name__ == "__main__":
    main()
