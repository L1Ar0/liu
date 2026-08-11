"""Normalize CoppeliaSim to STOPPED before generating a new scene."""

from __future__ import annotations

import time

from coppeliasim_zmqremoteapi_client import RemoteAPIClient


def main() -> None:
    client = RemoteAPIClient()
    sim = client.require("sim")
    state = sim.getSimulationState()
    if state == sim.simulation_stopped:
        print("CoppeliaSim state: STOPPED")
        return

    print(f"CoppeliaSim state={state}; stopping before scene regeneration")
    sim.stopSimulation()
    deadline = time.monotonic() + 5.0
    while sim.getSimulationState() != sim.simulation_stopped:
        if time.monotonic() >= deadline:
            raise RuntimeError("CoppeliaSim did not reach STOPPED state")
        time.sleep(0.05)
    print("CoppeliaSim state: STOPPED")


if __name__ == "__main__":
    main()
