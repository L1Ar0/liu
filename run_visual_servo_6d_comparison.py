"""Open-loop versus closed-loop PBVS comparison on identical disturbances."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SERVO_SUMMARY = ROOT / "visual_servo_output" / "visual_servo_summary.json"
SERVO_TRACE = ROOT / "visual_servo_output" / "visual_servo_trace.csv"
OUTPUT_DIR = ROOT / "visual_servo_6d_output"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run(command: list[str], env: dict[str, str], log_path: Path) -> None:
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_path.write_text(result.stdout, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(command)}\n"
            f"See {log_path}"
        )


def _copy_outputs(destination: Path, prefix: str) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    summary_path = destination / f"{prefix}_summary.json"
    trace_path = destination / f"{prefix}_trace.csv"
    shutil.copy2(SERVO_SUMMARY, summary_path)
    if SERVO_TRACE.exists():
        shutil.copy2(SERVO_TRACE, trace_path)
    return _load(summary_path)


def _run_case(
    layout: str,
    seed: int,
    output: Path,
    disturbance_mm: tuple[float, float, float],
    disturbance_deg: tuple[float, float, float],
) -> dict[str, Any]:
    case_dir = output / f"{layout}_seed_{seed}"
    case_dir.mkdir(parents=True, exist_ok=True)
    base_env = os.environ.copy()
    base_env.update(
        {
            "ROBOT_GRASP_HEADLESS": "1",
            "ROBOT_GRASP_USE_CONNECTOR": "0",
            "ROBOT_GRASP_VIEW_COUNT": "2",
        }
    )
    pipeline = [
        sys.executable,
        str(ROOT / "pipeline_runner.py"),
        "--mode",
        "planned",
        "--planned-layout",
        layout,
        "--seed",
        str(seed),
        "--views",
        "2",
        "--headless",
    ]
    servo_common = [
        sys.executable,
        str(ROOT / "visual_servo_runner.py"),
        "--align-only",
        "--initial-offset-mm",
        *(str(value) for value in disturbance_mm),
        "--initial-euler-deg",
        *(str(value) for value in disturbance_deg),
        "--grasp-orientation",
        "auto",
    ]

    _run(pipeline, {**base_env, "ROBOT_GRASP_RANDOM_SEED": str(seed)}, case_dir / "scene_open.log")
    open_summary_command = [*servo_common, "--open-loop"]
    _run(open_summary_command, base_env, case_dir / "open_loop.log")
    open_summary = _copy_outputs(case_dir, "open_loop")

    # Regenerate the same scene so both controllers see identical geometry and
    # only differ in whether visual feedback is applied after the disturbance.
    _run(pipeline, {**base_env, "ROBOT_GRASP_RANDOM_SEED": str(seed)}, case_dir / "scene_closed.log")
    _run(servo_common, base_env, case_dir / "closed_loop.log")
    closed_summary = _copy_outputs(case_dir, "closed_loop")

    row = {
        "layout": layout,
        "seed": int(seed),
        "open_loop_position_error_mm": open_summary.get("final_position_error_mm"),
        "open_loop_rotation_error_deg": open_summary.get("final_rotation_error_deg"),
        "open_loop_pass": bool(open_summary.get("converged", False)),
        "closed_loop_position_error_mm": closed_summary.get("final_position_error_mm"),
        "closed_loop_rotation_error_deg": closed_summary.get("final_rotation_error_deg"),
        "closed_loop_pass": bool(
            closed_summary.get("converged", False)
            and float(closed_summary.get("final_position_error_mm", float("inf"))) <= 3.0
            and float(closed_summary.get("final_rotation_error_deg", float("inf"))) <= 3.0
        ),
        "disturbance_mm": list(disturbance_mm),
        "disturbance_deg": list(disturbance_deg),
        "uses_color_features": False,
    }
    (case_dir / "comparison_summary.json").write_text(
        json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layouts", nargs="+", default=["leaning"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[13579, 24680, 97531])
    parser.add_argument("--disturbance-mm", nargs=3, type=float, default=[12.0, -8.0, 5.0])
    parser.add_argument("--disturbance-deg", nargs=3, type=float, default=[6.0, 6.0, 12.0])
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--keep-going", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for layout in args.layouts:
        for seed in args.seeds:
            try:
                row = _run_case(
                    layout,
                    int(seed),
                    output,
                    tuple(float(value) for value in args.disturbance_mm),
                    tuple(float(value) for value in args.disturbance_deg),
                )
                print(json.dumps(row, ensure_ascii=False, indent=2))
            except Exception as exc:
                row = {"layout": layout, "seed": int(seed), "error": str(exc), "closed_loop_pass": False}
                print(f"Comparison failed for {layout} seed={seed}: {exc}")
                if not args.keep_going:
                    rows.append(row)
                    break
            rows.append(row)
        else:
            continue
        break

    csv_path = output / "per_case_metrics.csv"
    keys = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    closed_passes = sum(bool(row.get("closed_loop_pass")) for row in rows)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "case_count": len(rows),
        "closed_loop_passed_count": closed_passes,
        "closed_loop_pass_rate_percent": 100.0 * closed_passes / max(len(rows), 1),
        "open_loop_passed_count": sum(bool(row.get("open_loop_pass")) for row in rows),
        "uses_color_features": False,
        "rows": rows,
    }
    (output / "comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if rows and closed_passes == len(rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
