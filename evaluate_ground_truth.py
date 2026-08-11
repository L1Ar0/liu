from __future__ import annotations

import csv
import itertools
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d

try:
    from scipy.optimize import linear_sum_assignment
except ImportError as exc:
    raise RuntimeError(
        "缺少 scipy。\n"
        "请在当前虚拟环境中运行：\n"
        "python -m pip install scipy"
    ) from exc

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from point_cloud import (
    find_unique_object_by_alias,
    get_kuka_joints_from_tip,
    get_full_path,
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ============================================================
# 1. 文件与阈值
# ============================================================

RECOGNITION_JSON = Path(
    os.environ.get(
        "ROBOT_GRASP_PREDICTION_JSON",
        str(Path("recognition_output") / "recognition_results.json"),
    )
)
SCENE_GT_FILE = Path("random_scene_ground_truth.json")

OUTPUT_DIR = Path("evaluation_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_FILE = OUTPUT_DIR / "evaluation_results.csv"
SUMMARY_JSON = OUTPUT_DIR / "evaluation_summary.json"

# 预测与GT中心距离超过该阈值时，不认为是同一物体。
MAX_MATCH_DISTANCE_M = 0.04

# 随机化清单与当前CoppeliaSim场景之间允许的数值误差。
SCENE_STATE_TOLERANCE_M = 0.002

# 一个比较直观的定位通过阈值：
POSITION_PASS_THRESHOLD_MM = 10.0

# Cuboid姿态误差通过阈值。
YAW_PASS_THRESHOLD_DEG = 10.0

# 是否显示预测中心与GT中心的Open3D验证图。
SHOW_VISUALIZATION = os.environ.get("ROBOT_GRASP_HEADLESS") != "1"


# ============================================================
# 2. 基础工具
# ============================================================

def wrap_angle_deg(angle: float) -> float:
    """归一化到[-180,180)。"""
    return (angle + 180.0) % 360.0 - 180.0


def symmetry_aware_yaw_error_deg(
    predicted_deg: float,
    gt_deg: float,
    period_deg: float,
) -> float:
    """
    考虑几何对称性的yaw误差。

    Cuboid:
        180°周期，因为长轴正反方向等价。

    Cube:
        90°周期，因为正方形footprint每90°等价。
    """
    delta = predicted_deg - gt_deg

    best = float("inf")

    # 多枚举几个等价周期，足够覆盖[-360,360]
    for k in range(-6, 7):
        candidate = abs(
            wrap_angle_deg(
                delta + k * period_deg
            )
        )
        best = min(best, candidate)

    return best


def _proper_box_symmetries() -> list[np.ndarray]:
    """Return the 24 proper rotation symmetries of a cube."""
    result: list[np.ndarray] = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            matrix = np.zeros((3, 3), dtype=np.float64)
            for row, column in enumerate(permutation):
                matrix[row, column] = signs[row]
            if np.linalg.det(matrix) > 0.5:
                result.append(matrix)
    return result


def _rotation_angle_deg(predicted: np.ndarray, expected: np.ndarray) -> float:
    relative = predicted.T @ expected
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))


def symmetry_aware_rotation_error_deg(
    predicted: np.ndarray | None,
    expected: np.ndarray | None,
    object_class: str,
) -> float | None:
    if predicted is None or expected is None:
        return None
    predicted = np.asarray(predicted, dtype=np.float64).reshape(3, 3)
    expected = np.asarray(expected, dtype=np.float64).reshape(3, 3)
    if object_class == "sphere":
        return None
    if object_class in {"cylinder", "spheroid"}:
        # Cylinder axis and spheroid major axis are observable, but rotation
        # around that axis is not. Both endpoint directions are equivalent.
        axis_column = 2 if object_class == "cylinder" else 0
        cosine = abs(float(np.dot(predicted[:, axis_column], expected[:, axis_column])))
        return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))
    if object_class == "cone":
        cosine = float(np.dot(predicted[:, 2], expected[:, 2]))
        return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))

    if object_class == "cuboid":
        symmetries = [
            np.diag(signs)
            for signs in itertools.product((-1.0, 1.0), repeat=3)
            if np.linalg.det(np.diag(signs)) > 0.5
        ]
    else:
        symmetries = _proper_box_symmetries()
    return min(
        _rotation_angle_deg(predicted, expected @ symmetry)
        for symmetry in symmetries
    )


def matrix12_to_numpy(matrix: list[float]) -> np.ndarray:
    """
    CoppeliaSim 12元素矩阵:
    [Vx0 Vy0 Vz0 P0,
     Vx1 Vy1 Vz1 P1,
     Vx2 Vy2 Vz2 P2]
    """
    return np.asarray(
        matrix,
        dtype=np.float64,
    ).reshape(3, 4)


def infer_gt_class(alias: str) -> str | None:
    """
    根据场景Shape alias判断GT类别。

    推荐命名：
        target_cube
        cube_01
        cuboid_01
        cylinder_01
    """
    name = alias.lower()

    if "spheroid" in name:
        return "spheroid"

    if "sphere" in name:
        return "sphere"

    if "cone" in name:
        return "cone"

    if "cylinder" in name:
        return "cylinder"

    if (
        "cuboid" in name
        or "rectangular_prism" in name
        or "rect_prism" in name
    ):
        return "cuboid"

    # 最后判断cube，避免未来某些名称冲突
    if "cube" in name:
        return "cube"

    return None


# ============================================================
# 3. 读取识别结果
# ============================================================

def load_predictions(expected_scene_id: str) -> list[dict]:
    if not RECOGNITION_JSON.exists():
        raise RuntimeError(
            f"找不到：{RECOGNITION_JSON}\n"
            "请先运行：\n"
            "python recognize_objects.py"
        )

    with open(
        RECOGNITION_JSON,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    if not isinstance(data, dict) or "metadata" not in data:
        raise RuntimeError(
            "recognition_results.json是旧格式，缺少scene_id。"
            "请重新运行scene_randomizer、segment和recognize。"
        )

    actual_scene_id = data["metadata"].get("scene_id")
    if actual_scene_id != expected_scene_id:
        raise RuntimeError(
            "识别结果与当前Ground Truth不属于同一个scene_id。"
            "请按完整流水线重新运行。"
        )

    data = data.get("objects", [])

    predictions: list[dict] = []

    for item in data:
        center = np.asarray(
            item["center_m"],
            dtype=np.float64,
        )

        if center.shape != (3,):
            raise RuntimeError(
                f"Object {item.get('id')} 的 center_m 格式错误。"
            )

        rotation_matrix = None
        if item.get("rotation_matrix") is not None:
            candidate_rotation = np.asarray(
                item["rotation_matrix"],
                dtype=np.float64,
            )
            if candidate_rotation.shape == (3, 3):
                rotation_matrix = candidate_rotation

        predictions.append(
            {
                "id": int(item["id"]),
                "class": str(item["class"]).lower(),
                "confidence": float(item.get("confidence", 0.0)),
                "center": center,
                "yaw_deg": (
                    None
                    if item.get("yaw_deg") is None
                    else float(item["yaw_deg"])
                ),
                "geometry": item.get("geometry", {}),
                "rotation_matrix": rotation_matrix,
                "candidates": item.get("candidates", []),
                "source": item.get("source", ""),
            }
        )

    if not predictions:
        raise RuntimeError(
            "recognition_results.json中没有识别结果。"
        )

    return predictions


def load_scene_manifest() -> dict:
    if not SCENE_GT_FILE.exists():
        raise RuntimeError(
            "找不到random_scene_ground_truth.json。"
            "请先运行scene_randomizer.py。"
        )

    with open(SCENE_GT_FILE, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if not manifest.get("scene_id"):
        raise RuntimeError(
            "Ground Truth缺少scene_id，请重新运行scene_randomizer.py。"
        )

    return manifest


# ============================================================
# 4. 获取Robot Base
# ============================================================

def find_robot_base(sim: Any) -> int:
    gripper_tip = find_unique_object_by_alias(
        sim,
        sim.sceneobject_dummy,
        "gripper_tip",
    )

    joints = get_kuka_joints_from_tip(
        sim,
        gripper_tip,
    )

    if len(joints) != 7:
        raise RuntimeError(
            f"预期找到7个KUKA关节，实际找到{len(joints)}个。"
        )

    robot_base = int(
        sim.getObjectParent(
            joints[0]
        )
    )

    if robot_base == -1:
        raise RuntimeError(
            "无法确定Robot Base。"
        )

    return robot_base


# ============================================================
# 5. Shape bounding box中心与姿态
# ============================================================

def get_bbox_transform_in_base(
    sim: Any,
    shape: int,
    robot_base: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    返回：
        size = [Sx,Sy,Sz]
        T_base_bbox = 3x4

    getShapeBB的bbox pose是相对于Shape自身pose。
    """
    size, bbox_pose = sim.getShapeBB(shape)

    size = np.asarray(
        size,
        dtype=np.float64,
    )

    shape_matrix = sim.getObjectMatrix(
        shape,
        robot_base,
    )

    bbox_local_matrix = sim.poseToMatrix(
        bbox_pose
    )

    bbox_matrix = sim.multiplyMatrices(
        shape_matrix,
        bbox_local_matrix,
    )

    return (
        size,
        matrix12_to_numpy(
            bbox_matrix
        ),
    )


def bbox_yaw_deg(
    bbox_matrix: np.ndarray,
    bbox_size: np.ndarray,
    object_class: str,
) -> float | None:
    """
    对当前“基本直立于桌面”的场景计算Ground Truth yaw。

    bbox_matrix的前3x3是bbox坐标轴在Robot Base中的方向。
    """
    if object_class in {"cylinder", "sphere", "spheroid", "cone"}:
        return None

    rotation = bbox_matrix[:, :3]

    # Cube：X/Y都等价，选bbox X轴，之后按90°对称处理。
    if object_class == "cube":
        axis = rotation[:, 0]

    else:
        # Cuboid：选择水平footprint中更长的局部轴。
        # 当前假设物体基本直立，所以比较bbox的X/Y尺寸。
        if bbox_size[0] >= bbox_size[1]:
            axis = rotation[:, 0]
        else:
            axis = rotation[:, 1]

    return math.degrees(
        math.atan2(
            float(axis[1]),
            float(axis[0]),
        )
    )


# ============================================================
# 6. 读取所有Ground Truth工件
# ============================================================

def load_ground_truth(
    sim: Any,
    robot_base: int,
) -> list[dict]:
    shapes = sim.getObjectsInTree(
        sim.handle_scene,
        sim.sceneobject_shape,
        0,
    )

    objects: list[dict] = []

    for handle in shapes:
        alias = str(
            sim.getObjectAlias(handle)
        )

        object_class = infer_gt_class(
            alias
        )

        if object_class is None:
            continue

        size, bbox_matrix = get_bbox_transform_in_base(
            sim,
            int(handle),
            robot_base,
        )

        # bbox几何中心在Robot Base中的位置。
        center = bbox_matrix[:, 3].copy()

        gt_yaw = bbox_yaw_deg(
            bbox_matrix,
            size,
            object_class,
        )

        objects.append(
            {
                "handle": int(handle),
                "alias": alias,
                "class": object_class,
                "center": center,
                "bbox_size": size,
                "yaw_deg": gt_yaw,
                "rotation_matrix": bbox_matrix[:, :3].tolist(),
                "path": get_full_path(
                    sim,
                    int(handle),
                ),
            }
        )

    if not objects:
        raise RuntimeError(
            "场景中没有找到Ground Truth工件。\n\n"
            "请让工件alias中包含以下关键词之一：\n"
            "cube / cuboid / cylinder / sphere / cone\n\n"
            "例如：cube_01、cuboid_01、cylinder_01、sphere_01、cone_01。"
        )

    return objects


def verify_ground_truth_manifest(
    ground_truth: list[dict],
    manifest: dict,
) -> None:
    """Ensure the live scene still corresponds to the randomized scene."""

    manifest_objects = {
        str(item["alias"]): item
        for item in manifest.get("objects", [])
    }

    if len(ground_truth) != int(manifest.get("object_count", -1)):
        raise RuntimeError(
            "当前场景工件数量与Ground Truth清单不一致："
            f"{len(ground_truth)} != {manifest.get('object_count')}。"
        )

    for item in ground_truth:
        expected = manifest_objects.get(item["alias"])
        if expected is None:
            raise RuntimeError(
                f"当前场景中的工件{item['alias']}不在当前scene清单中。"
            )

        expected_center = np.asarray(
            expected["position"],
            dtype=np.float64,
        )
        error = float(
            np.linalg.norm(item["center"] - expected_center)
        )
        if error > SCENE_STATE_TOLERANCE_M:
            raise RuntimeError(
                f"工件{item['alias']}的位置已被修改，"
                f"与清单相差{error * 1000.0:.2f} mm。"
            )

        if item["class"] != str(expected["class"]).lower():
            raise RuntimeError(
                f"工件{item['alias']}的类别与Ground Truth清单不一致。"
            )

        item["stack_level"] = int(expected.get("stack_level", 0))
        item["scenario_role"] = expected.get("scenario_role", "single")
        item["support_alias"] = expected.get("support_alias")


# ============================================================
# 7. Prediction ↔ GT全局匹配
# ============================================================

def match_predictions_to_gt(
    predictions: list[dict],
    ground_truth: list[dict],
):
    pred_centers = np.vstack(
        [x["center"] for x in predictions]
    )

    gt_centers = np.vstack(
        [x["center"] for x in ground_truth]
    )

    # NxM距离矩阵
    distances = np.linalg.norm(
        pred_centers[:, None, :]
        - gt_centers[None, :, :],
        axis=2,
    )

    pred_indices, gt_indices = (
        linear_sum_assignment(
            distances
        )
    )

    matches = []
    matched_pred = set()
    matched_gt = set()

    for pred_i, gt_i in zip(
        pred_indices,
        gt_indices,
    ):
        distance = float(
            distances[pred_i, gt_i]
        )

        if distance > MAX_MATCH_DISTANCE_M:
            continue

        matches.append(
            {
                "prediction": predictions[pred_i],
                "gt": ground_truth[gt_i],
                "position_error_m": distance,
            }
        )

        matched_pred.add(
            int(pred_i)
        )

        matched_gt.add(
            int(gt_i)
        )

    unmatched_predictions = [
        predictions[i]
        for i in range(len(predictions))
        if i not in matched_pred
    ]

    unmatched_gt = [
        ground_truth[i]
        for i in range(len(ground_truth))
        if i not in matched_gt
    ]

    return (
        matches,
        unmatched_predictions,
        unmatched_gt,
    )


# ============================================================
# 8. 尺寸误差
# ============================================================

def predicted_dimensions(
    prediction: dict,
) -> np.ndarray | None:
    geometry = prediction["geometry"]
    cls = prediction["class"]

    try:
        if cls in {"cylinder", "sphere"}:
            if cls == "sphere":
                diameter = float(geometry["diameter_m"])
                return np.array([diameter, diameter, diameter], dtype=np.float64)
            diameter = float(
                geometry["diameter_m"]
            )
            height = float(
                geometry["height_m"]
            )

            return np.array(
                [diameter, diameter, height],
                dtype=np.float64,
            )

        if cls == "spheroid":
            return np.array(
                [
                    float(geometry["axis_x_m"]),
                    float(geometry["axis_y_m"]),
                    float(geometry["axis_z_m"]),
                ],
                dtype=np.float64,
            )

        if cls == "cone":
            base = float(geometry["base_diameter_m"])
            height = float(geometry["height_m"])
            return np.array([base, base, height], dtype=np.float64)

        return np.array(
            [
                float(geometry["length_m"]),
                float(geometry["width_m"]),
                float(geometry["height_m"]),
            ],
            dtype=np.float64,
        )

    except (KeyError, TypeError, ValueError):
        return None


def dimension_error_percent(
    prediction: dict,
    gt: dict,
) -> tuple[float | None, np.ndarray | None]:
    predicted = predicted_dimensions(
        prediction
    )

    if predicted is None:
        return None, None

    gt_size = np.asarray(
        gt["bbox_size"],
        dtype=np.float64,
    )

    # 比较排序后的3个尺度，避免对象朝向/轴命名影响。
    pred_sorted = np.sort(predicted)
    gt_sorted = np.sort(gt_size)

    relative = np.abs(
        pred_sorted - gt_sorted
    ) / np.maximum(
        gt_sorted,
        1e-9,
    )

    return (
        float(relative.mean() * 100.0),
        relative * 100.0,
    )


# ============================================================
# 9. 单个Match评价
# ============================================================

def evaluate_match(match: dict) -> dict:
    pred = match["prediction"]
    gt = match["gt"]

    error_vector = (
        pred["center"]
        - gt["center"]
    )

    error_mm = (
        match["position_error_m"]
        * 1000.0
    )

    class_correct = (
        pred["class"]
        == gt["class"]
    )

    yaw_error = None

    # 只有类别识别正确时，姿态误差才有明确意义。
    if (
        class_correct
        and pred["yaw_deg"] is not None
        and gt["yaw_deg"] is not None
    ):
        if gt["class"] == "cube":
            yaw_error = (
                symmetry_aware_yaw_error_deg(
                    pred["yaw_deg"],
                    gt["yaw_deg"],
                    90.0,
                )
            )

        elif gt["class"] == "cuboid":
            yaw_error = (
                symmetry_aware_yaw_error_deg(
                    pred["yaw_deg"],
                    gt["yaw_deg"],
                    180.0,
                )
            )

    size_error, size_axis_error = (
        dimension_error_percent(
            pred,
            gt,
        )
    )

    rotation_error = None
    if class_correct:
        rotation_error = symmetry_aware_rotation_error_deg(
            pred.get("rotation_matrix"),
            gt.get("rotation_matrix"),
            gt["class"],
        )

    return {
        "pred_id": pred["id"],
        "pred_class": pred["class"],
        "pred_confidence": pred["confidence"],
        "gt_alias": gt["alias"],
        "gt_class": gt["class"],
        "class_correct": class_correct,
        "pred_center": pred["center"],
        "gt_center": gt["center"],
        "dx_mm": float(
            error_vector[0] * 1000.0
        ),
        "dy_mm": float(
            error_vector[1] * 1000.0
        ),
        "dz_mm": float(
            error_vector[2] * 1000.0
        ),
        "position_error_mm": float(
            error_mm
        ),
        "pred_yaw_deg": pred["yaw_deg"],
        "gt_yaw_deg": gt["yaw_deg"],
        "yaw_error_deg": yaw_error,
        "rotation_error_deg": rotation_error,
        "mean_dimension_error_percent": size_error,
        "dimension_axis_error_percent": size_axis_error,
        "gt_stack_level": int(gt.get("stack_level", 0)),
        "gt_scenario_role": gt.get("scenario_role", "single"),
    }


# ============================================================
# 10. 打印结果
# ============================================================

def print_evaluation(
    rows: list[dict],
    unmatched_predictions: list[dict],
    unmatched_gt: list[dict],
):
    print(
        "\n\n"
        "============================================================"
    )
    print(
        "                   多工件 Ground Truth 评价"
    )
    print(
        "============================================================"
    )

    for row in rows:
        print(
            f"\nPrediction {row['pred_id']} "
            f"↔ GT {row['gt_alias']}"
        )

        print(
            f"类别："
            f"{row['pred_class']} → {row['gt_class']}  "
            f"{'✓' if row['class_correct'] else '✗'}"
        )

        print(
            "Pred center: "
            f"({row['pred_center'][0]:.5f}, "
            f"{row['pred_center'][1]:.5f}, "
            f"{row['pred_center'][2]:.5f}) m"
        )

        print(
            "GT center  : "
            f"({row['gt_center'][0]:.5f}, "
            f"{row['gt_center'][1]:.5f}, "
            f"{row['gt_center'][2]:.5f}) m"
        )

        print(
            "XYZ error  : "
            f"({row['dx_mm']:+.2f}, "
            f"{row['dy_mm']:+.2f}, "
            f"{row['dz_mm']:+.2f}) mm"
        )

        print(
            "Position error = "
            f"{row['position_error_mm']:.3f} mm"
        )

        if row["yaw_error_deg"] is not None:
            print(
                "Yaw error      = "
                f"{row['yaw_error_deg']:.3f}°"
            )
        elif row["gt_class"] in {"cylinder", "sphere", "spheroid", "cone"}:
            print(
                "Yaw error      = N/A "
                "(该primitive绕对称轴不可直接用Yaw评价)"
            )
        else:
            print(
                "Yaw error      = N/A"
            )

        if row["rotation_error_deg"] is not None:
            print(
                "Rotation error = "
                f"{row['rotation_error_deg']:.3f}° (symmetry-aware)"
            )

        if (
            row["mean_dimension_error_percent"]
            is not None
        ):
            print(
                "Mean size error = "
                f"{row['mean_dimension_error_percent']:.2f}%"
            )

    # --------------------------------------------------------
    # 总体指标
    # --------------------------------------------------------

    matched_count = len(rows)

    if matched_count == 0:
        raise RuntimeError(
            "没有任何Prediction成功匹配Ground Truth。"
        )

    correct_count = sum(
        int(x["class_correct"])
        for x in rows
    )

    class_accuracy = (
        correct_count / matched_count
    )

    position_errors = np.asarray(
        [
            x["position_error_mm"]
            for x in rows
        ],
        dtype=np.float64,
    )

    yaw_errors = np.asarray(
        [
            x["yaw_error_deg"]
            for x in rows
            if x["yaw_error_deg"] is not None
        ],
        dtype=np.float64,
    )

    rotation_errors = np.asarray(
        [
            x["rotation_error_deg"]
            for x in rows
            if x["rotation_error_deg"] is not None
        ],
        dtype=np.float64,
    )

    size_errors = np.asarray(
        [
            x["mean_dimension_error_percent"]
            for x in rows
            if x["mean_dimension_error_percent"] is not None
        ],
        dtype=np.float64,
    )

    print(
        "\n\n==================== 汇总 ===================="
    )

    print(
        f"GT工件数量         : "
        f"{matched_count + len(unmatched_gt)}"
    )

    print(
        f"识别Prediction数量 : "
        f"{matched_count + len(unmatched_predictions)}"
    )

    print(
        f"成功匹配数量       : {matched_count}"
    )

    print(
        f"分类准确率         : "
        f"{class_accuracy * 100:.2f}%"
    )

    print(
        f"平均定位误差       : "
        f"{position_errors.mean():.3f} mm"
    )

    print(
        f"定位误差中位数     : "
        f"{np.median(position_errors):.3f} mm"
    )

    print(
        f"最大定位误差       : "
        f"{position_errors.max():.3f} mm"
    )

    if len(yaw_errors) > 0:
        print(
            f"平均Yaw误差        : "
            f"{yaw_errors.mean():.3f}°"
        )

        print(
            f"最大Yaw误差        : "
            f"{yaw_errors.max():.3f}°"
        )

    if len(rotation_errors) > 0:
        print(
            f"平均对称旋转误差   : "
            f"{rotation_errors.mean():.3f}°"
        )
        print(
            f"最大对称旋转误差   : "
            f"{rotation_errors.max():.3f}°"
        )

    if len(size_errors) > 0:
        print(
            f"平均尺寸误差       : "
            f"{size_errors.mean():.2f}%"
        )

    if unmatched_gt:
        print(
            "\n未检测到的GT："
        )

        for obj in unmatched_gt:
            print(
                f"  - {obj['alias']} "
                f"({obj['class']})"
            )

    if unmatched_predictions:
        print(
            "\n无法匹配的Prediction："
        )

        for obj in unmatched_predictions:
            print(
                f"  - Object {obj['id']} "
                f"({obj['class']})"
            )

    return {
        "gt_count": (
            matched_count
            + len(unmatched_gt)
        ),
        "prediction_count": (
            matched_count
            + len(unmatched_predictions)
        ),
        "matched_count": matched_count,
        "classification_accuracy_percent": (
            class_accuracy * 100.0
        ),
        "mean_position_error_mm": float(
            position_errors.mean()
        ),
        "median_position_error_mm": float(
            np.median(position_errors)
        ),
        "max_position_error_mm": float(
            position_errors.max()
        ),
        "mean_yaw_error_deg": (
            None
            if len(yaw_errors) == 0
            else float(yaw_errors.mean())
        ),
        "max_yaw_error_deg": (
            None
            if len(yaw_errors) == 0
            else float(yaw_errors.max())
        ),
        "mean_rotation_error_deg": (
            None
            if len(rotation_errors) == 0
            else float(rotation_errors.mean())
        ),
        "max_rotation_error_deg": (
            None
            if len(rotation_errors) == 0
            else float(rotation_errors.max())
        ),
        "mean_dimension_error_percent": (
            None
            if len(size_errors) == 0
            else float(size_errors.mean())
        ),
        "unmatched_gt": [
            x["alias"]
            for x in unmatched_gt
        ],
        "unmatched_predictions": [
            x["id"]
            for x in unmatched_predictions
        ],
    }


# ============================================================
# 11. 保存CSV / JSON
# ============================================================

def save_outputs(
    rows: list[dict],
    summary: dict,
):
    fieldnames = [
        "pred_id",
        "pred_class",
        "pred_confidence",
        "gt_alias",
        "gt_class",
        "gt_stack_level",
        "gt_scenario_role",
        "class_correct",
        "dx_mm",
        "dy_mm",
        "dz_mm",
        "position_error_mm",
        "pred_yaw_deg",
        "gt_yaw_deg",
        "yaw_error_deg",
        "rotation_error_deg",
        "mean_dimension_error_percent",
    ]

    with open(
        CSV_FILE,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    key: row.get(key)
                    for key in fieldnames
                }
            )

    with open(
        SUMMARY_JSON,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "\n评价结果已保存："
    )

    print(
        f"CSV  : {CSV_FILE.resolve()}"
    )

    print(
        f"JSON : {SUMMARY_JSON.resolve()}"
    )


# ============================================================
# 12. Open3D可视化
# ============================================================

def create_sphere(
    center: np.ndarray,
    radius: float,
    color: list[float],
):
    sphere = (
        o3d.geometry.TriangleMesh
        .create_sphere(
            radius=radius
        )
    )

    sphere.compute_vertex_normals()
    sphere.translate(center)
    sphere.paint_uniform_color(color)

    return sphere


def visualize_matches(
    rows: list[dict],
):
    geometries: list[Any] = []

    line_points = []
    line_indices = []

    for row in rows:
        pred = row["pred_center"]
        gt = row["gt_center"]

        # 预测：红色稍大球
        geometries.append(
            create_sphere(
                pred,
                radius=0.006,
                color=[1.0, 0.0, 0.0],
            )
        )

        # GT：绿色稍小球
        geometries.append(
            create_sphere(
                gt,
                radius=0.004,
                color=[0.0, 1.0, 0.0],
            )
        )

        start = len(line_points)
        line_points.extend(
            [
                pred.tolist(),
                gt.tolist(),
            ]
        )

        line_indices.append(
            [start, start + 1]
        )

    if line_points:
        lines = o3d.geometry.LineSet()

        lines.points = (
            o3d.utility.Vector3dVector(
                np.asarray(
                    line_points,
                    dtype=np.float64,
                )
            )
        )

        lines.lines = (
            o3d.utility.Vector2iVector(
                np.asarray(
                    line_indices,
                    dtype=np.int32,
                )
            )
        )

        lines.paint_uniform_color(
            [0.0, 0.0, 0.0]
        )

        geometries.append(lines)

    base_frame = (
        o3d.geometry.TriangleMesh
        .create_coordinate_frame(
            size=0.10,
            origin=[0.0, 0.0, 0.0],
        )
    )

    geometries.append(base_frame)

    print(
        "\nOpen3D Ground Truth验证："
    )
    print(
        "红球 = 点云识别中心"
    )
    print(
        "绿球 = CoppeliaSim Ground Truth中心"
    )
    print(
        "黑线 = 两者之间的定位误差"
    )

    o3d.visualization.draw_geometries(
        geometries,
        window_name=(
            "Recognition vs CoppeliaSim Ground Truth"
        ),
        width=1200,
        height=800,
    )


# ============================================================
# 13. 主程序
# ============================================================

def main():
    print(
        "正在连接 CoppeliaSim..."
    )

    client = RemoteAPIClient()
    sim = client.require("sim")

    print(
        "连接成功。"
    )

    scene_manifest = load_scene_manifest()
    scene_mode = str(scene_manifest.get("scene_mode", "separated")).lower()
    simulation_state = sim.getSimulationState()
    paused_allowed = scene_mode in {"physics", "dynamic", "settled", "drop"}

    if simulation_state != sim.simulation_stopped and not (
        paused_allowed and simulation_state == sim.simulation_paused
    ):
        raise RuntimeError(
            "请保持CoppeliaSim为STOP状态，或让physics场景保持暂停。"
        )

    # --------------------------------------------------------
    # A. Prediction
    # --------------------------------------------------------

    scene_id = str(scene_manifest["scene_id"])
    print(f"当前Scene ID：{scene_id}")

    predictions = load_predictions(scene_id)

    print(
        "\n========== Prediction =========="
    )

    for item in predictions:
        print(
            f"Object {item['id']}: "
            f"{item['class']:8s} "
            f"P={np.round(item['center'], 5)} "
            f"conf={item['confidence']:.2f}"
        )

    # --------------------------------------------------------
    # B. Robot Base + GT
    # --------------------------------------------------------

    robot_base = find_robot_base(
        sim
    )

    print(
        "\nRobot Base："
        f"{get_full_path(sim, robot_base)}"
    )

    ground_truth = load_ground_truth(
        sim,
        robot_base,
    )
    verify_ground_truth_manifest(
        ground_truth,
        scene_manifest,
    )

    print(
        "\n========== CoppeliaSim Ground Truth =========="
    )

    for item in ground_truth:
        print(
            f"{item['alias']:20s} | "
            f"{item['class']:8s} | "
            f"P={np.round(item['center'], 5)} | "
            f"BB={np.round(item['bbox_size'] * 1000, 1)} mm"
        )

    # --------------------------------------------------------
    # C. Global assignment
    # --------------------------------------------------------

    (
        matches,
        unmatched_predictions,
        unmatched_gt,
    ) = match_predictions_to_gt(
        predictions,
        ground_truth,
    )

    # --------------------------------------------------------
    # D. Evaluate
    # --------------------------------------------------------

    rows = [
        evaluate_match(match)
        for match in matches
    ]

    summary = print_evaluation(
        rows,
        unmatched_predictions,
        unmatched_gt,
    )
    summary["scene_id"] = scene_id
    scene_mode = str(scene_manifest.get("scene_mode", "separated")).lower()
    summary["scene_mode"] = scene_mode

    stacked_gt = [
        item
        for item in ground_truth
        if int(item.get("stack_level", 0)) > 0
    ]
    stacked_rows = [
        row
        for row in rows
        if int(row.get("gt_stack_level", 0)) > 0
    ]
    stacked_correct = sum(
        int(row["class_correct"])
        for row in stacked_rows
    )
    summary["stacked_gt_count"] = len(stacked_gt)
    summary["stacked_matched_count"] = len(stacked_rows)
    summary["stacked_recall_percent"] = (
        100.0
        if not stacked_gt
        else 100.0 * len(stacked_rows) / len(stacked_gt)
    )
    summary["stacked_class_accuracy_percent"] = (
        None
        if not stacked_rows
        else 100.0 * stacked_correct / len(stacked_rows)
    )
    summary["stacked_mean_position_error_mm"] = (
        None
        if not stacked_rows
        else float(
            np.mean([row["position_error_mm"] for row in stacked_rows])
        )
    )

    # --------------------------------------------------------
    # E. Pass/Fail参考
    # --------------------------------------------------------

    print(
        "\n========== 当前阶段判定参考 =========="
    )

    print(
        "分类准确率："
        f"{summary['classification_accuracy_percent']:.2f}%"
    )

    print(
        "平均定位误差："
        f"{summary['mean_position_error_mm']:.3f} mm"
    )

    if scene_mode == "level4":
        print(
            "堆叠工件召回率："
            f"{summary['stacked_recall_percent']:.2f}%"
        )
        if summary["stacked_mean_position_error_mm"] is not None:
            print(
                "堆叠工件平均定位误差："
                f"{summary['stacked_mean_position_error_mm']:.3f} mm"
            )

    common_pass = (
        summary["classification_accuracy_percent"]
        >= 99.9
        and summary["mean_position_error_mm"]
        <= POSITION_PASS_THRESHOLD_MM
        and len(unmatched_gt) == 0
        and len(unmatched_predictions) == 0
    )
    level4_pass = (
        common_pass
        and summary["stacked_gt_count"] >= 1
        and summary["stacked_recall_percent"] >= 99.9
        and summary["stacked_class_accuracy_percent"] is not None
        and summary["stacked_class_accuracy_percent"] >= 99.9
        and summary["stacked_mean_position_error_mm"] is not None
        and summary["stacked_mean_position_error_mm"]
        <= POSITION_PASS_THRESHOLD_MM
    )
    summary["level4_pass"] = bool(level4_pass)

    if scene_mode == "level4" and level4_pass:
        print(
            "\n✓ Level 4单视角轻度接触/遮挡/两层堆叠实验通过。"
        )
    elif scene_mode != "level4" and common_pass:
        print("\n✓ 当前场景的实例识别与几何6D定位实验通过。")
    else:
        print(
            "\n当前链路已经完成评价，但还有指标需要调优。"
        )

    save_outputs(
        rows,
        summary,
    )

    if SHOW_VISUALIZATION:
        visualize_matches(
            rows
        )


if __name__ == "__main__":
    main()
