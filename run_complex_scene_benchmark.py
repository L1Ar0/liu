"""Reproducible multi-layout benchmark for the geometry-only pipeline.

Each case runs the real CoppeliaSim pipeline in a fresh planned scene. Ground
truth is read only after inference for matching and metrics; it never enters
segmentation, classification, tracking, or pose estimation.
"""

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
EVALUATION_FILE = ROOT / "evaluation_output" / "evaluation_summary.json"
GROUND_TRUTH_FILE = ROOT / "random_scene_ground_truth.json"
CONTACT_FILE = ROOT / "random_scene_contacts.json"
OUTPUT_DIR = ROOT / "complex_benchmark_output"

DEFAULT_LAYOUTS = ("stack", "bridge", "leaning", "mixed")
DEFAULT_SEEDS = (13579, 24680, 97531)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Missing benchmark artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def _contact_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"contact_count": 0, "contact_types": []}
    data = _load_json(path)
    contacts = data.get("contacts", data if isinstance(data, list) else [])
    if not isinstance(contacts, list):
        contacts = []
    types = sorted(
        {
            str(item.get("relation", item.get("type", item.get("kind", "unknown"))))
            for item in contacts
            if isinstance(item, dict)
        }
    )
    return {"contact_count": len(contacts), "contact_types": types}


def _run_case(layout: str, seed: int, case_dir: Path) -> dict[str, Any]:
    environment = os.environ.copy()
    environment["ROBOT_GRASP_SCENE_MODE"] = "planned"
    environment["ROBOT_GRASP_PLANNED_LAYOUT"] = layout
    environment["ROBOT_GRASP_RANDOM_SEED"] = str(seed)
    environment["ROBOT_GRASP_HEADLESS"] = "1"
    environment["ROBOT_GRASP_VIEW_COUNT"] = "2"

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
        "2",
        "--headless",
    ]
    case_dir.mkdir(parents=True, exist_ok=True)
    log_path = case_dir / "pipeline.log"
    print(f"\n========== {layout} seed={seed} ==========")
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_path.write_text(completed.stdout, encoding="utf-8", errors="replace")

    row: dict[str, Any] = {
        "layout": layout,
        "seed": int(seed),
        "return_code": int(completed.returncode),
        "pass": False,
    }
    if completed.returncode == 0 and EVALUATION_FILE.exists():
        evaluation = _load_json(EVALUATION_FILE)
        ground_truth = _load_json(GROUND_TRUTH_FILE)
        contact = _contact_summary(CONTACT_FILE)
        row.update(
            {
                "scene_id": evaluation.get("scene_id"),
                "gt_count": evaluation.get("gt_count"),
                "prediction_count": evaluation.get("prediction_count"),
                "matched_count": evaluation.get("matched_count"),
                "classification_accuracy_percent": evaluation.get(
                    "classification_accuracy_percent", 0.0
                ),
                "mean_position_error_mm": evaluation.get("mean_position_error_mm"),
                "median_position_error_mm": evaluation.get("median_position_error_mm"),
                "max_position_error_mm": evaluation.get("max_position_error_mm"),
                "mean_rotation_error_deg": evaluation.get("mean_rotation_error_deg"),
                "max_rotation_error_deg": evaluation.get("max_rotation_error_deg"),
                "stacked_recall_percent": evaluation.get("stacked_recall_percent"),
                "unmatched_gt_count": len(evaluation.get("unmatched_gt", [])),
                "unmatched_prediction_count": len(
                    evaluation.get("unmatched_predictions", [])
                ),
                "ground_truth_object_count": ground_truth.get("object_count"),
                **contact,
            }
        )
        row["pass"] = bool(
            row["gt_count"] == row["prediction_count"] == row["matched_count"]
            and float(row["classification_accuracy_percent"]) >= 99.9
            and float(row["mean_position_error_mm"]) <= 10.0
            and float(row["max_position_error_mm"]) <= 10.0
            and row["unmatched_gt_count"] == 0
            and row["unmatched_prediction_count"] == 0
        )
        for artifact in (
            EVALUATION_FILE,
            GROUND_TRUTH_FILE,
            CONTACT_FILE,
            ROOT / "segmentation_output" / "segmentation_metadata.json",
            ROOT / "recognition_output" / "recognition_results.json",
        ):
            if artifact.exists():
                shutil.copy2(artifact, case_dir / artifact.name)
    else:
        row["error"] = f"pipeline exited with code {completed.returncode}"

    (case_dir / "case_summary.json").write_text(
        json.dumps(row, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(row, ensure_ascii=False, indent=2))
    return row


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_plot(rows: list[dict[str, Any]], path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        return False
    labels = [f"{row['layout']}\n{row['seed'] % 1000:03d}" for row in rows]
    position = [float(row.get("max_position_error_mm", float("nan"))) for row in rows]
    accuracy = [float(row.get("classification_accuracy_percent", 0.0)) for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(labels, position, color="#4f6fad")
    axes[0].axhline(10.0, color="#c43c39", linestyle="--", label="10 mm gate")
    axes[0].set_ylabel("Worst position error (mm)")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].legend()
    axes[1].bar(labels, accuracy, color="#2d9d67")
    axes[1].axhline(99.9, color="#c43c39", linestyle="--", label="99.9% gate")
    axes[1].set_ylabel("Classification accuracy (%)")
    axes[1].set_ylim(0, 105)
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].legend()
    figure.suptitle("Geometry-only multi-layout RGB-D benchmark")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layouts", nargs="+", choices=DEFAULT_LAYOUTS, default=list(DEFAULT_LAYOUTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--keep-going", action="store_true", help="continue after a failed case")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    failures = 0
    for layout in args.layouts:
        for seed in args.seeds:
            case_dir = output_dir / f"{_safe_name(layout)}_seed_{int(seed)}"
            try:
                row = _run_case(layout, int(seed), case_dir)
            except Exception as exc:
                failures += 1
                row = {"layout": layout, "seed": int(seed), "pass": False, "error": str(exc)}
                print(f"Case failed: {layout} seed={seed}: {exc}")
            rows.append(row)
            if not row.get("pass") and not args.keep_going:
                failures += 1
                break
        if failures and not args.keep_going:
            break

    csv_path = output_dir / "per_case_metrics.csv"
    _write_csv(rows, csv_path)
    passed = sum(bool(row.get("pass")) for row in rows)
    numeric_positions = [
        float(row["mean_position_error_mm"])
        for row in rows
        if row.get("mean_position_error_mm") is not None
    ]
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "layouts": list(args.layouts),
        "seeds": [int(seed) for seed in args.seeds],
        "case_count": len(rows),
        "passed_count": passed,
        "pass_rate_percent": 100.0 * passed / max(len(rows), 1),
        "mean_case_position_error_mm": sum(numeric_positions) / max(len(numeric_positions), 1),
        "all_cases_pass": bool(rows) and passed == len(rows),
        "uses_color_features": False,
        "rows": rows,
    }
    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plotted = _write_plot(rows, output_dir / "benchmark_metrics.png")
    summary["plot_created"] = plotted
    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n========== Complex benchmark summary ==========")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {csv_path}")
    return 0 if summary["all_cases_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
