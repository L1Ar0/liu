"""End-to-end geometry-only RGB-D recognition runner."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run_stage(
    script: str,
    environment: dict[str, str],
    arguments: list[str] | None = None,
) -> None:
    command = [sys.executable, str(ROOT / script), *(arguments or [])]
    print("\n+", " ".join(command), flush=True)
    subprocess.run(command, cwd=str(ROOT), env=environment, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("physics", "planned", "separated", "level4"),
        default="physics",
    )
    parser.add_argument(
        "--planned-layout",
        choices=(
            "auto",
            "table_only",
            "stack",
            "side_contact",
            "partial_support",
            "leaning",
            "bridge",
            "mixed",
        ),
        default=os.environ.get("ROBOT_GRASP_PLANNED_LAYOUT", "auto"),
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--views",
        type=int,
        choices=(1, 2),
        default=None,
        help="RGB-D observation views; planned/physics default to 2, other modes to 1",
    )
    parser.add_argument(
        "--visual-servo",
        action="store_true",
        help="run geometry-only eye-in-hand PBVS after recognition evaluation",
    )
    parser.add_argument(
        "--servo-execute-grasp",
        action="store_true",
        help="close RG2 and run lift verification after PBVS convergence",
    )
    parser.add_argument(
        "--servo-dynamics",
        action="store_true",
        help=(
            "use true synchronous iiwa dynamic control and dynamic RG2 opening/closing; "
            "disables connector attachment"
        ),
    )
    parser.add_argument(
        "--servo-dynamics-mode",
        choices=("position", "torque"),
        default=os.environ.get("ROBOT_GRASP_DYNAMICS_MODE", "position"),
        help="dynamic actuator mode; position is the commissioning mode, torque is the impedance-like mode",
    )
    parser.add_argument(
        "--servo-target-id",
        type=int,
        help="prediction id to servo; otherwise the safest topmost object is selected",
    )
    parser.add_argument(
        "--servo-target-policy",
        choices=("auto", "upper-random"),
        default=os.environ.get("ROBOT_GRASP_SERVO_TARGET_POLICY", "upper-random"),
        help=(
            "automatic target policy; upper-random excludes blocked stack bases "
            "and avoids sphere when another safe candidate exists"
        ),
    )
    parser.add_argument(
        "--servo-grasp-orientation",
        choices=("auto", "top_down", "surface"),
        default=os.environ.get("ROBOT_GRASP_SERVO_GRASP_ORIENTATION", "auto"),
        help="geometry-only grasp orientation used by PBVS",
    )
    parser.add_argument(
        "--no-connector",
        action="store_true",
        help="disable the explicit post-closure CoppeliaSim connector",
    )
    parser.add_argument(
        "--servo-lift-height-mm",
        type=float,
        default=1000.0
        * float(os.environ.get("ROBOT_GRASP_SERVO_LIFT_M", "0.150")),
        help="vertical lift distance after attachment; default 150 mm",
    )
    parser.add_argument(
        "--servo-place-in-box",
        action="store_true",
        help="after a verified lift, move the object into an external drop box and release it",
    )
    parser.add_argument(
        "--depth-grasp-checkpoint",
        type=str,
        help="optionally run a trained depth-only PPO proposal checkpoint",
    )

    # Retained as hidden compatibility options for older command lines.  The
    # geometry-only pipeline no longer uses Ground Truth count for segmentation
    # acceptance and no longer retries merely because contact objects merged.
    parser.add_argument("--allow-legacy-gt-count", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--allow-merged-clusters", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--physics-attempts",
        type=int,
        default=int(os.environ.get("ROBOT_GRASP_PHYSICS_ATTEMPTS", "5")),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--planned-attempts",
        type=int,
        default=int(os.environ.get("ROBOT_GRASP_PLANNED_ATTEMPTS", "8")),
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.servo_execute_grasp and not args.visual_servo:
        raise SystemExit("--servo-execute-grasp requires --visual-servo")
    if args.servo_place_in_box and not args.servo_execute_grasp:
        raise SystemExit("--servo-place-in-box requires --servo-execute-grasp")
    if args.servo_place_in_box and args.no_connector:
        raise SystemExit("--servo-place-in-box requires connector mode")
    if args.servo_dynamics and args.no_connector is False:
        # Dynamic mode never attaches a dummy.  Keep this explicit in the
        # environment so a stale shell variable cannot re-enable connectors.
        args.no_connector = True
    if args.servo_dynamics and args.servo_place_in_box:
        raise SystemExit("--servo-dynamics cannot be combined with --servo-place-in-box")
    if args.servo_lift_height_mm <= 0.0:
        raise SystemExit("--servo-lift-height-mm must be positive")
    environment = os.environ.copy()
    environment["ROBOT_GRASP_PREDICTION_JSON"] = str(
        ROOT / "recognition_output" / "recognition_results.json"
    )
    environment["ROBOT_GRASP_SCENE_MODE"] = args.mode
    environment["ROBOT_GRASP_PLANNED_LAYOUT"] = args.planned_layout

    view_count = args.views if args.views is not None else (
        2 if args.mode in {"planned", "physics"} else 1
    )
    environment["ROBOT_GRASP_VIEW_COUNT"] = str(view_count)
    if args.seed is None:
        environment.pop("ROBOT_GRASP_RANDOM_SEED", None)
    else:
        environment["ROBOT_GRASP_RANDOM_SEED"] = str(args.seed)
    if args.headless:
        environment["ROBOT_GRASP_HEADLESS"] = "1"
    else:
        environment.pop("ROBOT_GRASP_HEADLESS", None)
    environment.pop("ROBOT_GRASP_ALLOW_GT_COUNT", None)
    environment["ROBOT_GRASP_USE_CONNECTOR"] = "0" if args.no_connector else "1"
    environment["ROBOT_GRASP_DYNAMICS_CONTROL"] = "1" if args.servo_dynamics else "0"
    environment["ROBOT_GRASP_DYNAMICS_MODE"] = str(args.servo_dynamics_mode)
    environment["ROBOT_GRASP_SERVO_GRASP_ORIENTATION"] = args.servo_grasp_orientation
    environment["ROBOT_GRASP_SERVO_TARGET_POLICY"] = args.servo_target_policy
    environment["ROBOT_GRASP_SERVO_LIFT_M"] = str(
        float(args.servo_lift_height_mm) / 1000.0
    )

    print(
        f"\nGeometry-only pipeline: mode={args.mode}, views={view_count}, "
        f"layout={args.planned_layout}, visual_servo={args.visual_servo}",
        flush=True,
    )
    # Reset a leftover PAUSED/RUNNING physics scene before creating a new one.
    run_stage("ensure_simulation_stopped.py", environment)

    # Every new scene must start from the canonical global observation pose.
    # Without this, a previous visual-servo run can leave the eye-in-hand
    # camera near the last target and a one-view physical run will see only a
    # small part of the workspace.
    run_stage("observation_pose.py", environment)
    run_stage("scene_randomizer.py", environment)

    # First view: global scene segmentation and coarse object hypotheses.
    run_stage("segment_multiple_objects.py", environment)
    run_stage("recognize_objects.py", environment)

    # Second view: active camera motion, Robot Base registration, fusion and
    # complete re-segmentation.  This is active perception, not visual servo.
    if view_count == 2:
        run_stage("active_second_view.py", environment)
        run_stage("recognize_objects.py", environment)

    # Evaluate the basic task before PBVS moves the robot or, when explicitly
    # requested, changes the physical target pose during grasp execution.
    run_stage("evaluate_ground_truth.py", environment)

    if args.depth_grasp_checkpoint:
        run_stage(
            "run_depth_grasp_policy.py",
            environment,
            ["--checkpoint", args.depth_grasp_checkpoint],
        )

    if args.visual_servo:
        servo_arguments: list[str] = []
        if args.servo_target_id is not None:
            servo_arguments.extend(["--target-id", str(args.servo_target_id)])
        if args.servo_execute_grasp:
            servo_arguments.append("--execute-grasp")
        if args.servo_dynamics:
            servo_arguments.append("--dynamics")
            servo_arguments.extend(["--dynamics-mode", str(args.servo_dynamics_mode)])
        if args.servo_place_in_box:
            servo_arguments.append("--place-in-box")
        servo_arguments.extend(
            [
                "--grasp-orientation",
                args.servo_grasp_orientation,
                "--target-policy",
                args.servo_target_policy,
            ]
        )
        if args.seed is not None:
            servo_arguments.extend(["--selection-seed", str(args.seed)])
        run_stage("visual_servo_runner.py", environment, servo_arguments)


if __name__ == "__main__":
    main()
