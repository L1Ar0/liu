"""Normalize CoppeliaSim to STOPPED before generating a new scene."""

from __future__ import annotations

import time

from coppeliasim_zmqremoteapi_client import RemoteAPIClient


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
    sim = client.require("sim")
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
    removed = remove_stale_grasp_connectors(sim)
    if removed:
        print(f"Removed stale grasp connectors: {removed}")
    removed_boxes = remove_stale_drop_boxes(sim)
    if removed_boxes:
        print(f"Removed stale drop-box pieces: {removed_boxes}")


if __name__ == "__main__":
    main()
