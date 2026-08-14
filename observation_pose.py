"""Persist and restore the canonical first RGB-D observation pose."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from remote_session import RemoteAPIClient

from point_cloud import find_unique_object_by_alias, get_kuka_joints_from_tip
from robot_state import (
    MANIFEST_FILE,
    capture as capture_robot_state,
    load_compatible as load_robot_state,
    reset_for_observation,
    save as save_robot_state,
)


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
    robot_base = int(sim.getObjectParent(joints[0]))
    manifest = load_robot_state(sim, robot_base)
    # A manifest belongs to the current scene tree, not to a generated
    # workpiece scene.  If it is missing (first run or a manually edited
    # scene), capture the tuned baseline once.  The helper freezes every
    # descendant shape before assigning q, which prevents a stale dynamic body
    # from pulling the arm apart during reset.
    if manifest is None:
        manifest = capture_robot_state(sim, robot_base)
        save_robot_state(manifest)
    reset_for_observation(sim, robot_base, joints, values, manifest)
    print(f"Robot dynamic state normalized from {MANIFEST_FILE.resolve()}")
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
