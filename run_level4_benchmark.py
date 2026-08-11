from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EVALUATION_DIR = ROOT / "evaluation_output"
STAGES = (
    "scene_randomizer.py",
    "segment_multiple_objects.py",
    "recognize_objects.py",
    "evaluate_ground_truth.py",
)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_stage(script: str, environment: dict[str, str]) -> None:
    print(f"\n>>> {script}")
    completed = subprocess.run(
        [sys.executable, str(ROOT / script)],
        cwd=ROOT,
        env=environment,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{script} failed with exit code {completed.returncode}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reproducible random RGB-D Level 4 benchmark scenes."
    )
    parser.add_argument(
        "--mode",
        choices=("level4", "separated"),
        default="level4",
        help="Scene generator mode (default: level4).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of random scenes when --seed is omitted.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        nargs="+",
        help="Explicit seeds. One scene is run for each supplied seed.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Keep Open3D windows enabled during each run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.runs <= 0:
        raise SystemExit("--runs must be positive")

    seeds = list(args.seed) if args.seed else [None] * args.runs
    if args.seed is None:
        seeds = [None] * args.runs

    environment = os.environ.copy()
    environment["ROBOT_GRASP_SCENE_MODE"] = args.mode
    if args.visualize:
        environment.pop("ROBOT_GRASP_HEADLESS", None)
    else:
        environment["ROBOT_GRASP_HEADLESS"] = "1"

    rows: list[dict] = []
    for run_index, seed in enumerate(seeds, start=1):
        if seed is None:
            environment.pop("ROBOT_GRASP_RANDOM_SEED", None)
        else:
            environment["ROBOT_GRASP_RANDOM_SEED"] = str(seed)

        print(
            f"\n========== Benchmark {run_index}/{len(seeds)} "
            f"mode={args.mode} seed={seed if seed is not None else 'random'} =========="
        )
        for stage in STAGES:
            run_stage(stage, environment)

        ground_truth = load_json(ROOT / "random_scene_ground_truth.json")
        summary = load_json(EVALUATION_DIR / "evaluation_summary.json")
        if summary.get("scene_id") != ground_truth.get("scene_id"):
            raise RuntimeError("evaluation scene_id does not match generated scene")

        row = {
            "run": run_index,
            "seed": ground_truth.get("random_seed"),
            "scene_id": summary.get("scene_id"),
            "gt_count": summary.get("gt_count"),
            "prediction_count": summary.get("prediction_count"),
            "matched_count": summary.get("matched_count"),
            "classification_accuracy_percent": summary.get(
                "classification_accuracy_percent"
            ),
            "mean_position_error_mm": summary.get("mean_position_error_mm"),
            "max_position_error_mm": summary.get("max_position_error_mm"),
            "stacked_recall_percent": summary.get("stacked_recall_percent"),
            "stacked_mean_position_error_mm": summary.get(
                "stacked_mean_position_error_mm"
            ),
            "pass": bool(
                summary.get("level4_pass")
                if args.mode == "level4"
                else (
                    summary.get("classification_accuracy_percent", 0.0) >= 99.9
                    and summary.get("mean_position_error_mm", float("inf")) <= 10.0
                    and not summary.get("unmatched_gt")
                    and not summary.get("unmatched_predictions")
                )
            ),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False, indent=2))

    passed = sum(1 for row in rows if row["pass"])
    aggregate = {
        "scene_mode": args.mode,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_count": len(rows),
        "passed_count": passed,
        "pass_rate_percent": 100.0 * passed / max(len(rows), 1),
        "mean_position_error_mm": sum(
            float(row["mean_position_error_mm"]) for row in rows
        )
        / max(len(rows), 1),
        "max_position_error_mm": max(
            float(row["max_position_error_mm"]) for row in rows
        ),
        "rows": rows,
    }
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    output_path = EVALUATION_DIR / f"{args.mode}_benchmark_summary.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, ensure_ascii=False, indent=2)

    print("\n========== Benchmark Summary ==========")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    print(f"Saved: {output_path}")
    return 0 if passed == len(rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
