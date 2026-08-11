"""Run a trained depth-only policy on the current CoppeliaSim RGB-D frame."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from depth_grasp_rl import (
    ACTION_DIM,
    STATE_SIZE,
    DepthGraspActorCritic,
    decode_pixel,
    make_state_tensor,
    normalize_depth,
    require_torch,
)
from point_cloud import capture_rgbd, find_unique_object_by_alias, get_camera_parameters, get_kuka_joints_from_tip


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "depth_grasp_rl_output" / "current_proposal.json")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch = require_torch()
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model = DepthGraspActorCritic(int(checkpoint.get("action_dim", ACTION_DIM))).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    client = RemoteAPIClient()
    sim = client.require("sim")
    camera = find_unique_object_by_alias(sim, sim.sceneobject_visionsensor, "rgbd_camera")
    tip = find_unique_object_by_alias(sim, sim.sceneobject_dummy, "gripper_tip")
    joints = get_kuka_joints_from_tip(sim, tip)
    robot_base = int(sim.getObjectParent(joints[0]))
    _rgb, depth, width, height = capture_rgbd(sim, camera)
    parameters = get_camera_parameters(sim, camera, width, height)
    resized = np.asarray(
        Image.fromarray(np.asarray(depth, dtype=np.float32), mode="F").resize(
            (STATE_SIZE, STATE_SIZE), Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    )
    state = normalize_depth(resized, parameters["near"], parameters["far"])
    with torch.no_grad():
        action, _std, value = model(make_state_tensor(state[None], args.device))
    action_np = action[0].cpu().numpy()
    u, v = decode_pixel(action_np, width, height)
    pixel_u = int(np.clip(round(u), 0, width - 1))
    pixel_v = int(np.clip(round(v), 0, height - 1))
    z = float(depth[pixel_v, pixel_u])
    x = z * math.tan(float(parameters["fov_x"]) / 2.0) * (1.0 - 2.0 * u / max(width - 1, 1))
    y = z * math.tan(float(parameters["fov_y"]) / 2.0) * (1.0 - 2.0 * v / max(height - 1, 1))
    point_camera = np.asarray([x, y, z], dtype=np.float64)
    camera_matrix = np.asarray(sim.getObjectMatrix(camera, robot_base), dtype=np.float64).reshape(3, 4)
    point_base = camera_matrix[:, :3] @ point_camera + camera_matrix[:, 3]
    normal_camera = np.asarray(action_np[4:7], dtype=np.float64)
    normal_camera /= max(float(np.linalg.norm(normal_camera)), 1e-12)
    normal_base = camera_matrix[:, :3] @ normal_camera
    yaw = 0.5 * math.atan2(float(action_np[2]), float(action_np[3]))
    payload = {
        "pixel_uv": [u, v],
        "point_camera_m": point_camera.tolist(),
        "point_base_m": point_base.tolist(),
        "yaw_rad": yaw,
        "yaw_deg": math.degrees(yaw),
        "normal_camera": normal_camera.tolist(),
        "normal_base": normal_base.tolist(),
        "policy_value": float(value[0].cpu()),
        "raw_action": action_np.tolist(),
        "uses_color_features": False,
        "proposal_role": "coarse_input_for_geometry_validation_and_pbvs",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Saved: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
