"""Normalize CoppeliaSim to STOPPED before generating a new scene."""

from __future__ import annotations

import time
import json
from pathlib import Path

from remote_session import RemoteAPIClient

from point_cloud import find_unique_object_by_alias, get_kuka_joints_from_tip
from robot_state import load_compatible, reset_for_observation, capture as capture_robot_state, save as save_robot_state


POSE_FILE = Path("camera_output") / "initial_observation_joints.json"


def normalize_robot_observation_pose(sim: object) -> bool:
    """Recover the complete robot tree after an interrupted dynamic run."""

    if not POSE_FILE.exists():
        return False
    try:
        tip = find_unique_object_by_alias(sim, sim.sceneobject_dummy, "gripper_tip")
        joints = get_kuka_joints_from_tip(sim, tip)
        if len(joints) != 7:
            return False
        robot_base = int(sim.getObjectParent(joints[0]))
        payload = json.loads(POSE_FILE.read_text(encoding="utf-8"))
        positions = [float(value) for value in payload["joint_positions_rad"]]
        if len(positions) != 7:
            return False
        manifest = load_compatible(sim, robot_base)
        if manifest is None:
            manifest = capture_robot_state(sim, robot_base)
            save_robot_state(manifest)
        reset_for_observation(sim, robot_base, joints, positions, manifest)
        return True
    except Exception as exc:
        print(f"Robot observation-pose normalization skipped: {exc}")
        return False


def is_grasp_connector_alias(alias: str) -> bool:
    normalized = str(alias).strip().lower().replace("-", "_")
    return normalized.startswith("grasp_connector_rand_")


def is_grasp_drop_box_alias(alias: str) -> bool:
    return str(alias).strip().lower().startswith("grasp_drop_box_")


def remove_stale_grasp_connectors(sim: object) -> int:
    """Detach carried shapes and remove connector dummies from earlier runs."""

    removed = 0
    dummies = sim.getObjectsInTree(sim.handle_scene, sim.sceneobject_dummy, 0)
    for handle in dummies:
        try:
            alias = str(sim.getObjectAlias(handle))
        except Exception:
            continue
        if not is_grasp_connector_alias(alias):
            continue

        # Preserve any carried workpiece until scene_randomizer removes the
        # old rand_* shapes. Removing a parent dummy must not delete its child.
        while True:
            try:
                child = int(sim.getObjectChild(handle, 0))
            except Exception:
                child = -1
            if child < 0:
                break
            sim.setObjectParent(child, -1, True)
        sim.removeObject(handle)
        removed += 1
    return removed


def remove_stale_drop_boxes(sim: object) -> int:
    """Remove the previous run's external placement box before regeneration."""

    removed = 0
    shapes = sim.getObjectsInTree(sim.handle_scene, sim.sceneobject_shape, 0)
    for handle in shapes:
        try:
            alias = str(sim.getObjectAlias(handle))
        except Exception:
            continue
        if not is_grasp_drop_box_alias(alias):
            continue
        try:
            sim.removeObject(handle)
            removed += 1
        except Exception:
            pass
    return removed


def main() -> None:
    client = RemoteAPIClient()
    try:
        sim = client.require("sim")
        # This process is not a stepping controller.  Explicitly release any
        # residual level before stopping a scene left paused by a previous
        # subprocess; otherwise the add-on can still be waiting for a step.
        try:
            client.setStepping(False)
        except Exception:
            pass
        state = sim.getSimulationState()
        if state != sim.simulation_stopped:
            print(f"CoppeliaSim state={state}; stopping before scene regeneration")
            sim.stopSimulation()
            deadline = time.monotonic() + 5.0
            while sim.getSimulationState() != sim.simulation_stopped:
                if time.monotonic() >= deadline:
                    raise RuntimeError("CoppeliaSim did not reach STOPPED state")
                time.sleep(0.05)
        print("CoppeliaSim state: STOPPED")
        if normalize_robot_observation_pose(sim):
            print("Robot state normalized to the canonical observation pose")
        removed = remove_stale_grasp_connectors(sim)
        if removed:
            print(f"Removed stale grasp connectors: {removed}")
        removed_boxes = remove_stale_drop_boxes(sim)
        if removed_boxes:
            print(f"Removed stale drop-box pieces: {removed_boxes}")
    finally:
        # Send the explicit ZMQ end request before Python destroys the socket.
        client.close()


if __name__ == "__main__":
    main()
