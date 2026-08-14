"""Collect depth-only states and geometry labels from CoppeliaSim scenes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from remote_session import RemoteAPIClient

from depth_grasp_rl import STATE_SIZE, encode_action, normalize_depth
from point_cloud import (
    capture_rgbd,
    find_unique_object_by_alias,
    get_camera_parameters,
    get_kuka_joints_from_tip,
)


ROOT = Path(__file__).resolve().parent


def _resize_depth(depth: np.ndarray) -> np.ndarray:
    image = Image.fromarray(np.asarray(depth, dtype=np.float32), mode="F")
    resized = image.resize((STATE_SIZE, STATE_SIZE), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def _project_base_to_pixel(
    point_base: np.ndarray,
    camera_matrix: np.ndarray,
    parameters: dict[str, Any],
    width: int,
    height: int,
) -> tuple[float, float, float]:
    matrix = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 4)
    point_camera = (np.asarray(point_base, dtype=np.float64) - matrix[:, 3]) @ matrix[:, :3]
    z = float(point_camera[2])
    if z <= float(parameters["near"]) or z >= float(parameters["far"]):
        raise RuntimeError(f"Target depth {z:.3f} m is outside camera range")
    u = (
        1.0
        - point_camera[0] / (z * np.tan(float(parameters["fov_x"]) / 2.0))
    ) * 0.5 * (width - 1)
    v = (
        1.0
        - point_camera[1] / (z * np.tan(float(parameters["fov_y"]) / 2.0))
    ) * 0.5 * (height - 1)
    return float(u), float(v), z


def _choose_target(ground_truth: dict[str, Any], role: str) -> dict[str, Any]:
    objects = list(ground_truth.get("objects", []))
    if role:
        candidates = [item for item in objects if str(item.get("scenario_role", "")) == role]
        if candidates:
            return candidates[0]
    topmost = [item for item in objects if bool(item.get("topmost", False))]
    if topmost:
        return max(topmost, key=lambda item: float(item.get("grasp_priority", 0.0)))
    return objects[0]


def _capture_scene_sample(ground_truth: dict[str, Any], role: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    client = RemoteAPIClient()
    sim = client.require("sim")
    camera = find_unique_object_by_alias(sim, sim.sceneobject_visionsensor, "rgbd_camera")
    tip = find_unique_object_by_alias(sim, sim.sceneobject_dummy, "gripper_tip")
    joints = get_kuka_joints_from_tip(sim, tip)
    if not joints:
        raise RuntimeError("No robot joints found while collecting depth sample")
    robot_base = int(sim.getObjectParent(int(joints[0])))
    rgb, depth, width, height = capture_rgbd(sim, camera)
    parameters = get_camera_parameters(sim, camera, width, height)
    camera_matrix = np.asarray(sim.getObjectMatrix(camera, robot_base), dtype=np.float64).reshape(3, 4)
    target = _choose_target(ground_truth, role)
    center = np.asarray(target["position"], dtype=np.float64)
    u, v, z = _project_base_to_pixel(center, camera_matrix, parameters, width, height)
    rotation_base = np.asarray(target.get("rotation_matrix", np.eye(3)), dtype=np.float64).reshape(3, 3)
    rotation_camera = camera_matrix[:3, :3].T @ rotation_base
    yaw = float(np.arctan2(rotation_base[1, 0], rotation_base[0, 0]))
    normal = rotation_camera[:, 2]
    state = normalize_depth(_resize_depth(depth), parameters["near"], parameters["far"])
    action = encode_action((u, v), width, height, yaw, normal)
    label = {
        "alias": target.get("alias"),
        "class": target.get("class"),
        "pixel_uv": [u, v],
        "depth_m": z,
        "position_base_m": target.get("position"),
        "rotation_base": rotation_base.tolist(),
        "yaw_rad": yaw,
        "normal_camera": normal.tolist(),
        "scene_id": ground_truth.get("scene_id"),
    }
    return state, action, label


def _run_pipeline(layout: str, seed: int) -> None:
    environment = os.environ.copy()
    environment.update(
        {
            "ROBOT_GRASP_SCENE_MODE": "planned",
            "ROBOT_GRASP_PLANNED_LAYOUT": layout,
            "ROBOT_GRASP_RANDOM_SEED": str(seed),
            "ROBOT_GRASP_HEADLESS": "1",
        }
    )
    command = [
        sys.executable,
        str(ROOT / "pipeline_runner.py"),
        "--mode",
        "planned",
        "--planned-layout",
        layout,
        "--seed",
        str(seed),
        "--views",
        "1",
        "--headless",
    ]
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=80)
    parser.add_argument("--mode", choices=("planned",), default="planned")
    parser.add_argument("--layout", default="leaning")
    parser.add_argument("--target-role", default="")
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--output", type=Path, default=ROOT / "depth_grasp_dataset.npz")
    parser.add_argument("--skip-simulation", action="store_true", help="only validate an existing NPZ")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.samples <= 0:
        raise SystemExit("--samples must be positive")
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    labels: list[dict[str, Any]] = []
    if not args.skip_simulation:
        for index in range(args.samples):
            seed = int(args.seed) + index * 7919
            print(f"\nCollecting depth grasp sample {index + 1}/{args.samples}, seed={seed}")
            _run_pipeline(args.layout, seed)
            ground_truth = json.loads(
                (ROOT / "random_scene_ground_truth.json").read_text(encoding="utf-8")
            )
            state, action, label = _capture_scene_sample(ground_truth, args.target_role)
            states.append(state)
            actions.append(action)
            labels.append(label)
    else:
        data = np.load(args.output, allow_pickle=False)
        states = [np.asarray(item) for item in data["states"]]
        actions = [np.asarray(item) for item in data["actions"]]
        metadata = json.loads(
            args.output.with_suffix(".json").read_text(encoding="utf-8")
        )
        labels = list(metadata.get("labels", []))

    states_array = np.asarray(states, dtype=np.float32)
    actions_array = np.asarray(actions, dtype=np.float32)
    np.savez_compressed(args.output, states=states_array, actions=actions_array)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "state_shape": list(states_array.shape),
                "action_shape": list(actions_array.shape),
                "layout": args.layout,
                "target_role": args.target_role,
                "labels": labels,
                "uses_color_features": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved dataset: {args.output.resolve()}")
    print(f"Saved labels: {args.output.with_suffix('.json').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
