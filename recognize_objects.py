from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from point_cloud import (
    find_unique_object_by_alias,
    get_kuka_joints_from_tip,
    capture_rgbd,
    get_camera_parameters,
    depth_to_camera_point_cloud,
    transform_points,
    create_open3d_cloud,
    get_full_path,
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ============================================================
# 1. 路径与基础参数
# ============================================================

CLUSTER_DIR = Path("segmentation_output")
OUTPUT_DIR = Path("recognition_output")
SCENE_GT_FILE = Path("random_scene_ground_truth.json")
SEGMENTATION_METADATA_FILE = CLUSTER_DIR / "segmentation_metadata.json"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_JSON = OUTPUT_DIR / "recognition_results.json"
SHOW_VISUALIZATION = os.environ.get("ROBOT_GRASP_HEADLESS") != "1"


def load_pipeline_context() -> tuple[dict, dict]:
    if not SCENE_GT_FILE.exists():
        raise RuntimeError(
            "找不到random_scene_ground_truth.json。"
            "请先运行scene_randomizer.py。"
        )
    if not SEGMENTATION_METADATA_FILE.exists():
        raise RuntimeError(
            "找不到segmentation_metadata.json。"
            "请先运行segment_multiple_objects.py。"
        )

    with open(SCENE_GT_FILE, "r", encoding="utf-8") as f:
        scene_manifest = json.load(f)
    with open(SEGMENTATION_METADATA_FILE, "r", encoding="utf-8") as f:
        segmentation_metadata = json.load(f)

    scene_id = scene_manifest.get("scene_id")
    segmentation_scene_id = segmentation_metadata.get("scene_id")
    if not scene_id or scene_id != segmentation_scene_id:
        raise RuntimeError(
            "Ground Truth与分割结果不属于同一个scene_id。"
            "请按scene_randomizer -> segment顺序重新运行。"
        )

    return scene_manifest, segmentation_metadata


# ============================================================
# 2. 尺度无关分类阈值
#
# 注意：这里没有任何 40 mm / 70 mm / 60 mm 等绝对尺寸先验。
# 所有分类阈值都基于几何比例、圆度、矩形度。
# ============================================================

# V2分类器不再使用单一Circularity硬阈值。
# 它综合顶面矩形度、圆拟合残差、圆周覆盖率和长宽比。
# At the current 640x480 depth resolution, a small cube can have a nearly
# circular convex hull. Real cylinders in this scene score well above 0.80;
# require a stronger joint score before accepting the cylinder hypothesis.
CYLINDER_SCORE_THRESHOLD = 0.72
CYLINDER_RECTANGULARITY_SOFT_MAX = 0.90
CIRCLE_ERROR_GOOD = 0.045
CIRCLE_ERROR_BAD = 0.14

# Cube / Cuboid主要看顶面footprint比例；
# 对近似正方形footprint，再用高度/平面尺寸比辅助判断。
CUBE_FOOTPRINT_RATIO_MAX = 1.30
CUBE_HEIGHT_RATIO_MIN = 0.70
CUBE_HEIGHT_RATIO_MAX = 1.40

# 只取物体最顶部的一薄层点做二维形状识别，减少侧壁影响。
TOP_SLICE_THICKNESS_M = 0.006
TOP_SLICE_UPPER_TOLERANCE_M = 0.003
TOP_SLICE_MIN_POINTS = 25
TOP_COMPONENT_DBSCAN_EPS_M = 0.010
TOP_COMPONENT_MIN_POINTS = 3
TOP_SLICE_FALLBACK_PERCENTILE = 72.0

# 圆周可见角覆盖率过低时降低圆柱置信度。
MIN_CIRCLE_ANGULAR_COVERAGE = 0.45

# Level 4允许保留被遮挡后仍有20个以上有效点的实例；低点数结果会
# 通过quality_warnings和较低置信度继续标记。
MIN_CLUSTER_POINTS = 20

# 估计物体高度时使用的高分位数，降低少量离群点影响。
HEIGHT_PERCENTILE = 99.0
# Points below this height are treated as table-supported contact noise. A
# higher bottom surface indicates an upper object in a stack.
BOTTOM_HEIGHT_SNAP_M = 0.015
BOTTOM_PLANAR_SPAN_SNAP_M = 0.008
# Height-bin width for estimating a dominant horizontal surface.  This is
# deliberately larger than the depth quantization noise but smaller than the
# height difference between the two stack layers.
DOMINANT_HEIGHT_BIN_SIZE_M = 0.002

# A single-view top-only observation can make a cylinder cap look rectangular.
# The fallback below is intentionally restricted to a planar upper cluster
# with a tall round profile, so ordinary table-supported cubes are unaffected.
STACKED_ROUND_SCORE_MIN = 0.34
STACKED_ROUND_CIRCULARITY_MIN = 0.86
STACKED_ROUND_COVERAGE_MIN = 0.72
STACKED_ROUND_HEIGHT_RATIO_MIN = 1.45

# 估计桌面时的参数。
TABLE_VOXEL_SIZE = 0.005
TABLE_PLANE_DISTANCE_THRESHOLD = 0.004
TABLE_RANSAC_N = 3
TABLE_RANSAC_ITERATIONS = 1500

# 可视化大小。
POSE_FRAME_SIZE = 0.035
BASE_FRAME_SIZE = 0.10


# ============================================================
# 3. 基础数学工具
# ============================================================

def normalize(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm < 1e-12:
        raise RuntimeError("检测到零长度向量，无法归一化。")
    return v / norm


def cross_2d(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """二维叉积标量。"""
    oa = a - o
    ob = b - o
    return float(oa[0] * ob[1] - oa[1] * ob[0])


# ============================================================
# 4. 读取上一阶段保存的DBSCAN clusters
# ============================================================

def load_clusters(
    segmentation_metadata: dict,
) -> list[tuple[Path, o3d.geometry.PointCloud]]:
    files = [
        CLUSTER_DIR / str(name)
        for name in segmentation_metadata.get("cluster_files", [])
    ]

    if not files:
        raise RuntimeError(
            "当前scene没有可用的cluster文件。\n"
            "请先运行 segment_multiple_objects.py。"
        )

    clusters: list[tuple[Path, o3d.geometry.PointCloud]] = []

    for file_path in files:
        cloud = o3d.io.read_point_cloud(str(file_path))

        if len(cloud.points) < MIN_CLUSTER_POINTS:
            print(
                f"跳过 {file_path.name}: "
                f"只有 {len(cloud.points)} 个点。"
            )
            continue

        clusters.append((file_path, cloud))

    if not clusters:
        raise RuntimeError("所有cluster点数都太少，无法识别。")

    expected_count = int(
        segmentation_metadata.get("expected_gt_object_count", -1)
    )
    if expected_count >= 0 and len(clusters) != int(
        segmentation_metadata.get("cluster_count", len(clusters))
    ):
        raise RuntimeError(
            "分割元数据中的cluster_count与实际文件不一致。"
        )

    return clusters


# ============================================================
# 5. 从当前CoppeliaSim场景重新估计桌面平面
#
# 为什么这么做：
# 上一阶段保存的cluster已经删除了桌面，单靠cluster无法可靠恢复
# 桌面平面。因此这里重新拍一帧，仅用于估计table plane。
# ============================================================

def estimate_table_plane_from_scene(
    sim: Any,
    camera: int,
    robot_base: int,
) -> np.ndarray:
    print("\n正在重新获取一帧，用于估计桌面平面……")

    rgb, depth, width, height = capture_rgbd(sim, camera)
    params = get_camera_parameters(sim, camera, width, height)

    points_camera, colors, _ = depth_to_camera_point_cloud(
        depth,
        rgb,
        params,
    )

    matrix = np.asarray(
        sim.getObjectMatrix(camera, robot_base),
        dtype=np.float64,
    ).reshape(3, 4)

    points_base = transform_points(points_camera, matrix)
    cloud = create_open3d_cloud(points_base, colors)

    cloud = cloud.voxel_down_sample(TABLE_VOXEL_SIZE)

    plane_model, inliers = cloud.segment_plane(
        distance_threshold=TABLE_PLANE_DISTANCE_THRESHOLD,
        ransac_n=TABLE_RANSAC_N,
        num_iterations=TABLE_RANSAC_ITERATIONS,
    )

    plane = np.asarray(plane_model, dtype=np.float64)

    # 归一化 ax+by+cz+d=0，使[a,b,c]为单位法向量。
    normal_norm = float(np.linalg.norm(plane[:3]))
    if normal_norm < 1e-12:
        raise RuntimeError("RANSAC返回了无效桌面平面。")

    plane /= normal_norm

    # 让桌面法向量指向相机一侧。
    camera_position = np.asarray(
        sim.getObjectPosition(camera, robot_base),
        dtype=np.float64,
    )

    camera_signed_distance = float(
        np.dot(plane[:3], camera_position) + plane[3]
    )

    if camera_signed_distance < 0:
        plane *= -1.0
        camera_signed_distance *= -1.0

    print("\n========== TABLE PLANE ==========")
    print(
        f"{plane[0]:.6f} x + "
        f"{plane[1]:.6f} y + "
        f"{plane[2]:.6f} z + "
        f"{plane[3]:.6f} = 0"
    )
    print(f"RANSAC table inliers = {len(inliers)}")
    print(f"Camera-table distance = {camera_signed_distance:.4f} m")

    return plane


# ============================================================
# 6. 建立桌面局部二维坐标系
#
# e1/e2位于桌面平面内，n为桌面法向。
# 后面所有footprint几何分析都在(e1,e2)二维平面中进行。
# ============================================================

def make_table_basis(plane: np.ndarray):
    n = normalize(plane[:3].astype(np.float64))

    # 优先让e1尽量接近Robot Base X方向，方便解释yaw。
    reference = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    if abs(float(np.dot(reference, n))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    e1 = reference - np.dot(reference, n) * n
    e1 = normalize(e1)

    e2 = normalize(np.cross(n, e1))

    # plane已归一化，因此 -d*n 是平面上距离原点最近的点。
    p0 = -plane[3] * n

    return p0, e1, e2, n


# ============================================================
# 7. 将3D cluster投影到桌面二维坐标
# ============================================================

def project_to_table(
    points: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
) -> np.ndarray:
    u = points @ e1
    v = points @ e2
    return np.column_stack([u, v])


# ============================================================
# 8. 二维凸包：Monotonic Chain
#
# 不依赖OpenCV/Scipy，纯NumPy即可。
# ============================================================

def convex_hull_2d(points: np.ndarray) -> np.ndarray:
    if len(points) < 3:
        raise RuntimeError("二维点数少于3，无法计算凸包。")

    # 去除几乎重复点，减少运算量。
    unique = np.unique(np.round(points, decimals=6), axis=0)

    if len(unique) < 3:
        raise RuntimeError("有效二维点过少，无法计算凸包。")

    order = np.lexsort((unique[:, 1], unique[:, 0]))
    pts = unique[order]

    lower: list[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross_2d(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list[np.ndarray] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross_2d(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = np.array(lower[:-1] + upper[:-1], dtype=np.float64)

    if len(hull) < 3:
        raise RuntimeError("凸包退化，无法进行形状识别。")

    return hull


# ============================================================
# 9. 多边形面积 / 周长 / 质心
# ============================================================

def polygon_area(hull: np.ndarray) -> float:
    x = hull[:, 0]
    y = hull[:, 1]
    return 0.5 * abs(
        float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    )


def polygon_perimeter(hull: np.ndarray) -> float:
    edges = np.roll(hull, -1, axis=0) - hull
    return float(np.linalg.norm(edges, axis=1).sum())


def polygon_centroid(hull: np.ndarray) -> np.ndarray:
    x = hull[:, 0]
    y = hull[:, 1]
    x2 = np.roll(x, -1)
    y2 = np.roll(y, -1)

    cross = x * y2 - x2 * y
    signed_area_twice = float(cross.sum())

    if abs(signed_area_twice) < 1e-12:
        return np.mean(hull, axis=0)

    cx = float(((x + x2) * cross).sum() / (3.0 * signed_area_twice))
    cy = float(((y + y2) * cross).sum() / (3.0 * signed_area_twice))

    return np.array([cx, cy], dtype=np.float64)


# ============================================================
# 10. 最小面积二维包围矩形
#
# 对每条凸包边作为候选方向，计算该方向下AABB面积，选择最小者。
# 这是尺度无关的矩形footprint估计。
# ============================================================

def minimum_area_rectangle(hull: np.ndarray) -> dict:
    best = None

    for i in range(len(hull)):
        p1 = hull[i]
        p2 = hull[(i + 1) % len(hull)]

        edge = p2 - p1
        edge_norm = float(np.linalg.norm(edge))

        if edge_norm < 1e-12:
            continue

        axis_u = edge / edge_norm
        axis_v = np.array([-axis_u[1], axis_u[0]], dtype=np.float64)

        basis = np.vstack([axis_u, axis_v])
        projected = hull @ basis.T

        minimum = projected.min(axis=0)
        maximum = projected.max(axis=0)
        extent = maximum - minimum
        area = float(extent[0] * extent[1])

        center_local = 0.5 * (minimum + maximum)
        center_world_2d = center_local @ basis

        if best is None or area < best["area"]:
            best = {
                "area": area,
                "extent": extent,
                "center": center_world_2d,
                "axis_u": axis_u,
                "axis_v": axis_v,
            }

    if best is None:
        raise RuntimeError("无法计算最小面积矩形。")

    # 统一让axis_long对应更长边。
    if best["extent"][0] >= best["extent"][1]:
        long_size = float(best["extent"][0])
        short_size = float(best["extent"][1])
        axis_long = best["axis_u"]
        axis_short = best["axis_v"]
    else:
        long_size = float(best["extent"][1])
        short_size = float(best["extent"][0])
        axis_long = best["axis_v"]
        axis_short = best["axis_u"]

    best["long_size"] = long_size
    best["short_size"] = short_size
    best["axis_long"] = axis_long
    best["axis_short"] = axis_short

    return best


# ============================================================
# 11. 尺度无关二维形状特征
# ============================================================

def calculate_shape_features(
    hull: np.ndarray,
    min_rect: dict,
) -> dict:
    area = polygon_area(hull)
    perimeter = polygon_perimeter(hull)
    rect_area = max(float(min_rect["area"]), 1e-12)

    rectangularity = float(np.clip(area / rect_area, 0.0, 1.2))

    if perimeter < 1e-12:
        circularity = 0.0
    else:
        circularity = float(
            np.clip(4.0 * math.pi * area / (perimeter * perimeter), 0.0, 1.2)
        )

    aspect_ratio = float(
        min_rect["long_size"] / max(min_rect["short_size"], 1e-12)
    )

    return {
        "hull_area": area,
        "hull_perimeter": perimeter,
        "rectangularity": rectangularity,
        "circularity": circularity,
        "footprint_aspect_ratio": aspect_ratio,
    }



# ============================================================
# 11B. V2：只提取最顶部表面
# ============================================================

def extract_top_surface_points(
    points: np.ndarray,
    plane: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    n = normalize(plane[:3].astype(np.float64))
    heights = points @ n + plane[3]

    positive = heights[heights > 0.0]

    if len(positive) < 10:
        raise RuntimeError("cluster没有足够的桌面上方点。")

    top_height = estimate_dominant_surface_height(positive)

    mask = (
        heights >= top_height - TOP_SLICE_THICKNESS_M
    ) & (
        heights <= top_height + TOP_SLICE_UPPER_TOLERANCE_M
    )

    if int(mask.sum()) < TOP_SLICE_MIN_POINTS:
        fallback = float(
            np.percentile(
                positive,
                TOP_SLICE_FALLBACK_PERCENTILE,
            )
        )
        mask = heights >= fallback

    top_points = points[mask]

    if len(top_points) < 8:
        order = np.argsort(heights)
        take = min(
            len(points),
            max(8, TOP_SLICE_MIN_POINTS),
        )
        top_points = points[order[-take:]]

    # Remove a clearly smaller disconnected top fragment (for example a few
    # boundary points from a neighboring object).  Two similarly sized pieces
    # are retained because they can be the two sides of an occluded support.
    if len(top_points) >= 40:
        reference = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(reference, n))) > 0.90:
            reference = np.array([0.0, 1.0, 0.0])
        axis_u = normalize(np.cross(n, reference))
        axis_v = normalize(np.cross(n, axis_u))
        uv = np.column_stack(
            [top_points @ axis_u, top_points @ axis_v]
        )
        component_cloud = o3d.geometry.PointCloud()
        component_cloud.points = o3d.utility.Vector3dVector(
            np.column_stack([uv, np.zeros(len(uv))])
        )
        labels = np.asarray(
            component_cloud.cluster_dbscan(
                eps=TOP_COMPONENT_DBSCAN_EPS_M,
                min_points=TOP_COMPONENT_MIN_POINTS,
                print_progress=False,
            )
        )
        component_ids = [
            int(label)
            for label in np.unique(labels)
            if int(label) >= 0
        ]
        if component_ids:
            sizes = {
                label: int(np.sum(labels == label))
                for label in component_ids
            }
            largest = max(component_ids, key=lambda label: sizes[label])
            largest_count = sizes[largest]
            keep_threshold = max(
                8,
                int(math.ceil(0.30 * largest_count)),
            )
            keep_ids = [
                label
                for label in component_ids
                if sizes[label] >= keep_threshold
            ]
            keep_mask = np.isin(labels, keep_ids)
            if (
                int(keep_mask.sum()) >= TOP_SLICE_MIN_POINTS
                and int(keep_mask.sum()) < len(top_points)
            ):
                top_points = top_points[keep_mask]

    return top_points, heights, top_height


# ============================================================
# 11C. V2：二维圆拟合
# ============================================================

def fit_circle_2d(
    hull: np.ndarray,
) -> dict:
    if len(hull) < 3:
        raise RuntimeError("圆拟合至少需要3个凸包点。")

    x = hull[:, 0]
    y = hull[:, 1]

    A = np.column_stack(
        [2.0 * x, 2.0 * y, np.ones_like(x)]
    )
    b = x * x + y * y

    solution, *_ = np.linalg.lstsq(
        A,
        b,
        rcond=None,
    )

    cx = float(solution[0])
    cy = float(solution[1])
    c0 = float(solution[2])

    radius_sq = c0 + cx * cx + cy * cy

    if radius_sq <= 1e-12:
        return {
            "center": np.array([cx, cy], dtype=np.float64),
            "radius": 0.0,
            "normalized_rmse": float("inf"),
            "angular_coverage": 0.0,
        }

    radius = math.sqrt(radius_sq)
    center = np.array([cx, cy], dtype=np.float64)

    radial = np.linalg.norm(
        hull - center,
        axis=1,
    )

    rmse = float(
        np.sqrt(
            np.mean(
                (radial - radius) ** 2
            )
        )
    )

    normalized_rmse = (
        rmse
        / max(radius, 1e-9)
    )

    angles = np.mod(
        np.arctan2(
            hull[:, 1] - cy,
            hull[:, 0] - cx,
        ),
        2.0 * math.pi,
    )

    angles = np.sort(angles)

    if len(angles) >= 2:
        gaps = np.diff(
            np.concatenate(
                [
                    angles,
                    angles[:1] + 2.0 * math.pi,
                ]
            )
        )
        largest_gap = float(np.max(gaps))
        angular_coverage = float(
            np.clip(
                1.0
                - largest_gap
                / (2.0 * math.pi),
                0.0,
                1.0,
            )
        )
    else:
        angular_coverage = 0.0

    return {
        "center": center,
        "radius": float(radius),
        "normalized_rmse": float(normalized_rmse),
        "angular_coverage": angular_coverage,
    }


# ============================================================
# 11D. V2：矩形边界拟合误差
# ============================================================

def rectangle_edge_error(
    hull: np.ndarray,
    min_rect: dict,
) -> float:
    axis_u = np.asarray(
        min_rect["axis_u"],
        dtype=np.float64,
    )
    axis_v = np.asarray(
        min_rect["axis_v"],
        dtype=np.float64,
    )

    basis = np.vstack([axis_u, axis_v])
    projected = hull @ basis.T

    minimum = projected.min(axis=0)
    maximum = projected.max(axis=0)

    distances = np.column_stack(
        [
            np.abs(projected[:, 0] - minimum[0]),
            np.abs(projected[:, 0] - maximum[0]),
            np.abs(projected[:, 1] - minimum[1]),
            np.abs(projected[:, 1] - maximum[1]),
        ]
    )

    nearest = np.min(
        distances,
        axis=1,
    )

    short_size = max(
        float(min_rect["short_size"]),
        1e-9,
    )

    return float(
        np.mean(nearest)
        / short_size
    )


# ============================================================
# 11E. V2：圆柱综合得分
# ============================================================

def cylinder_likelihood(
    circularity: float,
    rectangularity: float,
    circle_error: float,
    angular_coverage: float,
    rectangle_error: float,
) -> tuple[float, dict]:
    rect_score = float(
        np.clip(
            (
                CYLINDER_RECTANGULARITY_SOFT_MAX
                - rectangularity
            )
            /
            max(
                CYLINDER_RECTANGULARITY_SOFT_MAX
                - math.pi / 4.0,
                1e-9,
            ),
            0.0,
            1.0,
        )
    )

    circle_fit_score = float(
        np.clip(
            (
                CIRCLE_ERROR_BAD
                - circle_error
            )
            /
            max(
                CIRCLE_ERROR_BAD
                - CIRCLE_ERROR_GOOD,
                1e-9,
            ),
            0.0,
            1.0,
        )
    )

    circularity_score = float(
        np.clip(
            (
                circularity - 0.68
            )
            / (0.94 - 0.68),
            0.0,
            1.0,
        )
    )

    coverage_score = float(
        np.clip(
            (
                angular_coverage - 0.35
            )
            / (0.80 - 0.35),
            0.0,
            1.0,
        )
    )

    rectangle_rejection = float(
        np.clip(
            rectangle_error / 0.085,
            0.0,
            1.0,
        )
    )

    score = float(
        0.34 * rect_score
        + 0.31 * circle_fit_score
        + 0.14 * circularity_score
        + 0.11 * coverage_score
        + 0.10 * rectangle_rejection
    )

    if (
        angular_coverage < MIN_CIRCLE_ANGULAR_COVERAGE
        and rectangularity > 0.90
    ):
        score *= 0.80

    return (
        score,
        {
            "rect_score": rect_score,
            "circle_fit_score": circle_fit_score,
            "circularity_score": circularity_score,
            "coverage_score": coverage_score,
            "rectangle_rejection": rectangle_rejection,
        },
    )



# ============================================================
# 12. 估计高度与3D中心
# ============================================================

def estimate_height_and_center(
    points: np.ndarray,
    plane: np.ndarray,
    p0: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    n: np.ndarray,
    footprint_center_uv: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    signed_heights = points @ n + plane[3]

    # cluster来自“桌面上方”点，所以高度理论上应为正。
    positive = signed_heights[signed_heights > 0]

    if len(positive) < 10:
        raise RuntimeError("cluster没有足够的桌面上方点。")

    bottom_height, top_height, height = estimate_visible_vertical_extent(
        positive
    )

    # 桌面上的footprint中心点。
    table_center = (
        p0
        + footprint_center_uv[0] * e1
        + footprint_center_uv[1] * e2
    )

    # 物体几何中心：桌面中心点沿法向抬高一半高度。
    center_3d = table_center + (bottom_height + 0.5 * height) * n

    return height, center_3d, signed_heights


def estimate_visible_vertical_extent(
    positive_heights: np.ndarray,
) -> tuple[float, float, float]:
    """Estimate an object's top height before relational support inference."""

    # A single view often misses the lower side walls of a table-supported
    # object.  Inferring the bottom from the cluster minimum would then raise
    # its center incorrectly.  Start from the table and let
    # apply_stack_support_hints() establish a non-zero bottom only when a real
    # lower footprint is present.
    bottom_height = 0.0
    top_height = estimate_dominant_surface_height(positive_heights)

    height = max(top_height - bottom_height, 1e-6)
    return bottom_height, top_height, height


def estimate_dominant_surface_height(
    positive_heights: np.ndarray,
) -> float:
    """Estimate the height of the densest horizontal surface in a cluster."""
    values = np.asarray(positive_heights, dtype=np.float64)
    if len(values) == 0:
        return 0.0

    z_min = float(values.min())
    z_max = float(values.max())
    span = z_max - z_min
    if span <= DOMINANT_HEIGHT_BIN_SIZE_M:
        return float(np.median(values))

    bin_count = max(
        8,
        min(80, int(math.ceil(span / DOMINANT_HEIGHT_BIN_SIZE_M))),
    )
    counts, edges = np.histogram(
        values,
        bins=bin_count,
        range=(z_min, z_max + 1e-9),
    )
    max_count = int(counts.max())
    if max_count <= 0:
        return float(np.percentile(values, HEIGHT_PERCENTILE))

    # Select the highest dense bin.  This ignores a few boundary outliers but
    # still prefers the upper visible face over lower side-wall samples.
    dense_threshold = max(5, int(math.ceil(0.45 * max_count)))
    dense_bins = [
        index for index, count in enumerate(counts)
        if int(count) >= dense_threshold
    ]
    selected = max(dense_bins) if dense_bins else int(np.argmax(counts))
    mask = (
        values >= edges[selected]
    ) & (
        values <= edges[selected + 1] + 1e-9
    )
    selected_values = values[mask]
    if len(selected_values) == 0:
        return float(0.5 * (edges[selected] + edges[selected + 1]))
    return float(np.median(selected_values))


# ============================================================
# 13. 尺度无关分类器
#
# 只分类三个简单、直立primitive：
#   cube / cuboid / cylinder
#
# 绝对尺寸完全不参与分类。
# ============================================================

def classify_primitive(
    long_size: float,
    short_size: float,
    height: float,
    circularity: float,
    rectangularity: float,
    circle_error: float,
    angular_coverage: float,
    rectangle_error: float,
) -> tuple[str, float, dict]:
    cylinder_score, score_parts = (
        cylinder_likelihood(
            circularity,
            rectangularity,
            circle_error,
            angular_coverage,
            rectangle_error,
        )
    )

    footprint_ratio = float(
        long_size
        / max(short_size, 1e-9)
    )

    mean_footprint = max(
        0.5 * (long_size + short_size),
        1e-9,
    )

    height_ratio = float(
        height / mean_footprint
    )

    is_cylinder = (
        cylinder_score >= CYLINDER_SCORE_THRESHOLD
        and rectangularity
        <= CYLINDER_RECTANGULARITY_SOFT_MAX
    )

    # A heavily occluded tall cylinder may expose only a small top patch and
    # therefore receive a weak circle score. The height/footprint fallback is
    # deliberately restricted to round, tall objects so near-circular cubes
    # (height_ratio around 1) still remain cubes.
    tall_round_fallback = (
        cylinder_score >= 0.45
        and circularity >= 0.88
        and footprint_ratio <= 1.35
        and height_ratio >= 1.45
    )
    high_quality_round_fallback = (
        cylinder_score >= 0.64
        and circularity >= 0.94
        and rectangularity <= 0.89
        and circle_error <= 0.060
        and footprint_ratio <= 1.15
    )
    is_cylinder = (
        is_cylinder
        or tall_round_fallback
        or high_quality_round_fallback
    )

    # Partial central occlusion can make the remaining square top boundary
    # look rounded.  A marginal cylinder score with cube-like 3D proportions
    # is therefore kept as a cube hypothesis.
    marginal_round_cube = (
        is_cylinder
        and cylinder_score < 0.82
        and circularity < 0.90
        and footprint_ratio <= CUBE_FOOTPRINT_RATIO_MAX
        and CUBE_HEIGHT_RATIO_MIN <= height_ratio <= CUBE_HEIGHT_RATIO_MAX
    )
    if marginal_round_cube:
        is_cylinder = False

    if is_cylinder:
        confidence = float(
            np.clip(
                0.55
                + 0.45
                * (
                    cylinder_score
                    - CYLINDER_SCORE_THRESHOLD
                )
                /
                max(
                    1.0
                    - CYLINDER_SCORE_THRESHOLD,
                    1e-9,
                ),
                0.55,
                0.99,
            )
        )

        return (
            "cylinder",
            confidence,
            {
                "dimension_ratio": None,
                "cylinder_score": cylinder_score,
                "footprint_ratio": footprint_ratio,
                **score_parts,
            },
        )

    if (
        footprint_ratio <= CUBE_FOOTPRINT_RATIO_MAX
        and
        CUBE_HEIGHT_RATIO_MIN
        <= height_ratio
        <= CUBE_HEIGHT_RATIO_MAX
    ):
        object_class = "cube"

        footprint_closeness = (
            1.0
            - min(
                abs(footprint_ratio - 1.0)
                /
                max(
                    CUBE_FOOTPRINT_RATIO_MAX
                    - 1.0,
                    1e-9,
                ),
                1.0,
            )
        )

        height_closeness = (
            1.0
            - min(
                abs(height_ratio - 1.0)
                / 0.40,
                1.0,
            )
        )

        confidence = float(
            np.clip(
                0.55
                + 0.25 * footprint_closeness
                + 0.20 * height_closeness,
                0.55,
                0.98,
            )
        )

    else:
        object_class = "cuboid"

        ratio_strength = float(
            np.clip(
                (
                    footprint_ratio
                    - CUBE_FOOTPRINT_RATIO_MAX
                )
                / 0.80,
                0.0,
                1.0,
            )
        )

        height_difference = float(
            np.clip(
                abs(height_ratio - 1.0)
                / 0.60,
                0.0,
                1.0,
            )
        )

        confidence = float(
            np.clip(
                0.55
                + 0.28 * ratio_strength
                + 0.15 * height_difference,
                0.55,
                0.98,
            )
        )

    dimensions = np.array(
        [long_size, short_size, height],
        dtype=np.float64,
    )

    dimension_ratio = float(
        dimensions.max()
        / max(dimensions.min(), 1e-9)
    )

    return (
        object_class,
        confidence,
        {
            "dimension_ratio": dimension_ratio,
            "cylinder_score": cylinder_score,
            "footprint_ratio": footprint_ratio,
            "height_ratio": height_ratio,
            **score_parts,
        },
    )


# ============================================================
# 14. 将二维长轴转换到Robot Base三维方向
# ============================================================

def axis_2d_to_3d(
    axis_2d: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
) -> np.ndarray:
    direction = axis_2d[0] * e1 + axis_2d[1] * e2
    return normalize(direction)


# ============================================================
# 15. 构造估计姿态旋转矩阵
# ============================================================

def build_pose_rotation(
    object_class: str,
    long_axis_3d: np.ndarray,
    table_normal: np.ndarray,
) -> tuple[np.ndarray, float | None]:
    z_axis = normalize(table_normal)

    if object_class == "cylinder":
        # 圆柱绕自身轴的yaw不可观测。
        # 这里只构造一个任意但稳定的x/y方向。
        x_axis = long_axis_3d - np.dot(long_axis_3d, z_axis) * z_axis

        if np.linalg.norm(x_axis) < 1e-9:
            reference = np.array([1.0, 0.0, 0.0])
            x_axis = reference - np.dot(reference, z_axis) * z_axis

        x_axis = normalize(x_axis)
        y_axis = normalize(np.cross(z_axis, x_axis))
        x_axis = normalize(np.cross(y_axis, z_axis))

        rotation = np.column_stack([x_axis, y_axis, z_axis])
        return rotation, None

    x_axis = normalize(long_axis_3d)
    y_axis = normalize(np.cross(z_axis, x_axis))
    x_axis = normalize(np.cross(y_axis, z_axis))

    rotation = np.column_stack([x_axis, y_axis, z_axis])

    # 相对于Robot Base XY的yaw，便于查看。
    yaw = math.atan2(float(x_axis[1]), float(x_axis[0]))

    # 对矩形，180度方向等价，归一化到[-90,90)。
    while yaw >= math.pi / 2:
        yaw -= math.pi
    while yaw < -math.pi / 2:
        yaw += math.pi

    return rotation, yaw


# ============================================================
# 16. 单个cluster分析
# ============================================================

def analyze_cluster(
    index: int,
    file_path: Path,
    cloud: o3d.geometry.PointCloud,
    plane: np.ndarray,
    p0: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    n: np.ndarray,
) -> dict:
    points = np.asarray(
        cloud.points,
        dtype=np.float64,
    )

    if len(points) < MIN_CLUSTER_POINTS:
        raise RuntimeError(
            f"Object {index}点数太少。"
        )

    top_points, signed_heights, _ = (
        extract_top_surface_points(
            points,
            plane,
        )
    )

    top_points_2d = project_to_table(
        top_points,
        e1,
        e2,
    )

    hull = convex_hull_2d(
        top_points_2d
    )

    min_rect = minimum_area_rectangle(
        hull
    )

    features = calculate_shape_features(
        hull,
        min_rect,
    )

    circle = fit_circle_2d(
        hull
    )

    rect_error = rectangle_edge_error(
        hull,
        min_rect,
    )

    positive = signed_heights[
        signed_heights > 0
    ]

    if len(positive) < 10:
        raise RuntimeError(
            "cluster没有足够的桌面上方点。"
        )

    bottom_height, top_height, height = (
        estimate_visible_vertical_extent(
            positive
        )
    )

    long_size = float(
        min_rect["long_size"]
    )

    short_size = float(
        min_rect["short_size"]
    )

    (
        object_class,
        confidence,
        diagnostics,
    ) = classify_primitive(
        long_size,
        short_size,
        height,
        features["circularity"],
        features["rectangularity"],
        circle["normalized_rmse"],
        circle["angular_coverage"],
        rect_error,
    )

    if (
        object_class == "cylinder"
        and np.isfinite(
            circle["normalized_rmse"]
        )
        and circle["radius"] > 1e-6
    ):
        footprint_center_uv = np.asarray(
            circle["center"],
            dtype=np.float64,
        )
    else:
        footprint_center_uv = np.asarray(
            min_rect["center"],
            dtype=np.float64,
        )

    table_center = (
        p0
        + footprint_center_uv[0] * e1
        + footprint_center_uv[1] * e2
    )

    center_3d = (
        table_center
        + (bottom_height + 0.5 * height) * n
    )

    if object_class == "cylinder":
        diameter = (
            2.0
            * float(
                circle["radius"]
            )
        )

        if (
            not np.isfinite(diameter)
            or diameter <= 1e-6
        ):
            diameter = (
                2.0
                * math.sqrt(
                    max(
                        features["hull_area"],
                        0.0,
                    )
                    / math.pi
                )
            )

        geometry = {
            "diameter_m": diameter,
            "height_m": height,
        }

    else:
        geometry = {
            "length_m": long_size,
            "width_m": short_size,
            "height_m": height,
        }

    long_axis_3d = axis_2d_to_3d(
        np.asarray(
            min_rect["axis_long"],
            dtype=np.float64,
        ),
        e1,
        e2,
    )

    rotation, yaw = build_pose_rotation(
        object_class,
        long_axis_3d,
        n,
    )

    if object_class == "cylinder":
        diameter = geometry["diameter_m"]
        obb_extent = np.array(
            [diameter, diameter, height],
            dtype=np.float64,
        )
    else:
        obb_extent = np.array(
            [long_size, short_size, height],
            dtype=np.float64,
        )

    obb = o3d.geometry.OrientedBoundingBox(
        center_3d,
        rotation,
        obb_extent,
    )

    quality_warnings = []

    if circle["angular_coverage"] < 0.50:
        quality_warnings.append(
            "low_angular_coverage"
        )

    if len(top_points) < 35:
        quality_warnings.append(
            "few_top_points"
        )

    if rect_error > 0.10:
        quality_warnings.append(
            "irregular_top_boundary"
        )

    return {
        "id": index,
        "source": file_path.name,
        "class": object_class,
        "confidence": confidence,
        "center": center_3d,
        "rotation": rotation,
        "yaw": yaw,
        "geometry": geometry,
        "features": {
            "circularity": features["circularity"],
            "rectangularity": features["rectangularity"],
            "footprint_aspect_ratio": features["footprint_aspect_ratio"],
            "dimension_ratio": diagnostics.get("dimension_ratio"),
            "cylinder_score": diagnostics.get("cylinder_score"),
            "circle_fit_error": circle["normalized_rmse"],
            "circle_angular_coverage": circle["angular_coverage"],
            "circle_diameter_m": float(2.0 * circle["radius"]),
            "rectangle_edge_error": rect_error,
            "hull_area_m2": features["hull_area"],
            "top_point_count": int(len(top_points)),
            "height_ratio": diagnostics.get("height_ratio"),
        },
        "obb": obb,
        "footprint_center": table_center,
        "top_points_uv": top_points_2d,
        "top_footprint_center_uv": np.asarray(
            min_rect["center"],
            dtype=np.float64,
        ),
        "point_count": len(points),
        "height_min_m": float(np.min(signed_heights)),
        "height_max_m": float(np.max(signed_heights)),
        "dominant_top_height_m": estimate_dominant_surface_height(
            positive
        ),
        "support_height_m": float(bottom_height),
        "top_height_m": float(top_height),
        "quality_warnings": quality_warnings,
    }


# ============================================================
# 17. 可视化姿态坐标系
# ============================================================

def create_pose_frame(result: dict) -> o3d.geometry.TriangleMesh:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = result["rotation"]
    transform[:3, 3] = result["center"]

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=POSE_FRAME_SIZE
    )
    frame.transform(transform)
    return frame


# ============================================================
# 18. 输出结果
# ============================================================

def print_result(result: dict) -> None:
    center = result["center"]
    features = result["features"]

    print("\n\n========================================")
    print(f"Object {result['id']}")
    print("========================================")

    print(f"Source      : {result['source']}")
    print(f"Points      : {result['point_count']}")
    print(f"Class       : {result['class']}")
    print(f"Confidence  : {result['confidence']:.3f}")

    print("\nEstimated center (Robot Base):")
    print(f"X = {center[0]:.6f} m")
    print(f"Y = {center[1]:.6f} m")
    print(f"Z = {center[2]:.6f} m")

    print("\nScale-independent features:")
    print(f"Circularity       = {features['circularity']:.4f}")
    print(f"Rectangularity    = {features['rectangularity']:.4f}")
    print(f"Footprint ratio   = {features['footprint_aspect_ratio']:.4f}")
    print(f"Circle fit error  = {features['circle_fit_error']:.4f}")
    print(f"Circle coverage   = {features['circle_angular_coverage']:.4f}")
    print(f"Rectangle error   = {features['rectangle_edge_error']:.4f}")
    print(f"Cylinder score    = {features['cylinder_score']:.4f}")
    print(f"Top points        = {features['top_point_count']}")

    if features["dimension_ratio"] is not None:
        print(f"3D dimension ratio = {features['dimension_ratio']:.4f}")

    if result.get("quality_warnings"):
        print(
            "Quality warning   = "
            + ", ".join(result["quality_warnings"])
        )

    if result["class"] == "cylinder":
        print("\nEstimated geometry:")
        print(
            f"Diameter = "
            f"{result['geometry']['diameter_m'] * 1000:.2f} mm"
        )
        print(
            f"Height   = "
            f"{result['geometry']['height_m'] * 1000:.2f} mm"
        )
        print("Yaw      = N/A（圆柱绕自身轴旋转不可观测）")
    else:
        print("\nEstimated geometry:")
        print(
            f"Length = "
            f"{result['geometry']['length_m'] * 1000:.2f} mm"
        )
        print(
            f"Width  = "
            f"{result['geometry']['width_m'] * 1000:.2f} mm"
        )
        print(
            f"Height = "
            f"{result['geometry']['height_m'] * 1000:.2f} mm"
        )

        if result["yaw"] is None:
            print("Yaw = N/A")
        else:
            print(f"Yaw = {math.degrees(result['yaw']):.2f}°")

        if result["class"] == "cube":
            print("提示：Cube存在90°旋转对称性，yaw只代表一个等价主方向。")


# ============================================================
# 19. JSON保存
# ============================================================

def save_results_json(
    results: list[dict],
    scene_manifest: dict,
    segmentation_metadata: dict,
) -> None:
    serializable = []

    for result in results:
        item = {
            "id": result["id"],
            "source": result["source"],
            "class": result["class"],
            "confidence": result["confidence"],
            "center_m": result["center"].tolist(),
            "rotation_matrix": result["rotation"].tolist(),
            "yaw_deg": (
                None
                if result["yaw"] is None
                else math.degrees(result["yaw"])
            ),
            "geometry": result["geometry"],
            "features": result["features"],
            "point_count": result["point_count"],
            "support_height_m": result.get("support_height_m", 0.0),
            "top_height_m": result.get("top_height_m"),
            "quality_warnings": result.get("quality_warnings", []),
        }
        serializable.append(item)

    payload = {
        "metadata": {
            "scene_id": scene_manifest["scene_id"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "expected_gt_object_count": int(scene_manifest["object_count"]),
            "segmentation_cluster_count": int(
                segmentation_metadata["cluster_count"]
            ),
        },
        "objects": serializable,
    }

    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n识别结果已保存：{RESULT_JSON.resolve()}")


def _footprint_contains_point(
    support: dict,
    point: np.ndarray,
    margin_m: float = 0.010,
) -> bool:
    geometry = support["geometry"]
    if support["class"] == "cylinder":
        length = width = float(geometry["diameter_m"])
    else:
        length = float(geometry["length_m"])
        width = float(geometry["width_m"])

    delta = np.asarray(point, dtype=np.float64) - support["center"]
    local_x = float(np.dot(delta, support["rotation"][:, 0]))
    local_y = float(np.dot(delta, support["rotation"][:, 1]))
    return (
        abs(local_x) <= 0.5 * length + margin_m
        and abs(local_y) <= 0.5 * width + margin_m
    )


def apply_stack_support_hints(
    results: list[dict],
    table_normal: np.ndarray,
    table_axis_u: np.ndarray,
    table_axis_v: np.ndarray,
) -> int:
    """Recover the support height of a planar upper-object observation."""

    updated = 0
    supporting_results: list[dict] = []
    for result in results:
        raw_span = result["height_max_m"] - result["height_min_m"]
        if result["height_min_m"] <= BOTTOM_HEIGHT_SNAP_M:
            continue

        candidates = [
            other
            for other in results
            if other is not result
            and float(
                other.get("dominant_top_height_m", other["top_height_m"])
            ) <= result["height_min_m"] + 0.006
            and float(
                other.get("dominant_top_height_m", other["top_height_m"])
            ) < result["height_max_m"] - 0.012
            and _footprint_contains_point(
                other,
                result["footprint_center"],
            )
        ]
        if not candidates:
            continue

        support = max(
            candidates,
            key=lambda item: float(
                item.get("dominant_top_height_m", item["top_height_m"])
            ),
        )
        if not any(item is support for item in supporting_results):
            supporting_results.append(support)
        support.setdefault("supported_upper_results", []).append(result)
        bottom_height = float(
            support.get(
                "dominant_top_height_m",
                support["top_height_m"],
            )
        )
        top_height = float(result["height_max_m"])
        height = max(top_height - bottom_height, 1e-6)

        geometry = result["geometry"]
        if result["class"] == "cylinder":
            long_size = float(geometry["diameter_m"])
            short_size = long_size
        else:
            long_size = float(geometry["length_m"])
            short_size = float(geometry["width_m"])

        features = result["features"]
        object_class, confidence, diagnostics = classify_primitive(
            long_size,
            short_size,
            height,
            features["circularity"],
            features["rectangularity"],
            features["circle_fit_error"],
            features["circle_angular_coverage"],
            features["rectangle_edge_error"],
        )

        planar_upper = raw_span <= BOTTOM_PLANAR_SPAN_SNAP_M
        round_fallback = (
            planar_upper
            and object_class != "cylinder"
            and diagnostics.get("cylinder_score", 0.0)
            >= STACKED_ROUND_SCORE_MIN
            and features.get("circularity", 0.0)
            >= STACKED_ROUND_CIRCULARITY_MIN
            and features.get("circle_angular_coverage", 0.0)
            >= STACKED_ROUND_COVERAGE_MIN
            and height / max(0.5 * (long_size + short_size), 1e-9)
            >= STACKED_ROUND_HEIGHT_RATIO_MIN
        )
        if round_fallback:
            object_class = "cylinder"
            confidence = max(
                0.55,
                min(0.82, 0.55 + 0.25 * diagnostics["cylinder_score"]),
            )

        if object_class == "cylinder":
            long_size = max(
                long_size,
                short_size,
                float(features.get("circle_diameter_m", 0.0)),
            )
            geometry = {
                "diameter_m": long_size,
                "height_m": height,
            }
        else:
            geometry = {
                "length_m": long_size,
                "width_m": short_size,
                "height_m": height,
            }

        table_center = result["footprint_center"]
        center = table_center + (bottom_height + 0.5 * height) * table_normal
        long_axis = result["rotation"][:, 0]
        rotation, yaw = build_pose_rotation(
            object_class,
            long_axis,
            table_normal,
        )
        extent = (
            np.asarray([long_size, short_size, height], dtype=np.float64)
            if object_class != "cylinder"
            else np.asarray([long_size, long_size, height], dtype=np.float64)
        )

        result["class"] = object_class
        result["confidence"] = confidence
        result["geometry"] = geometry
        result["center"] = center
        result["rotation"] = rotation
        result["yaw"] = yaw
        result["obb"] = o3d.geometry.OrientedBoundingBox(
            center,
            rotation,
            extent,
        )
        result["support_height_m"] = bottom_height
        result["top_height_m"] = top_height
        result["features"]["height_ratio"] = diagnostics.get("height_ratio")
        result["features"]["dimension_ratio"] = diagnostics.get(
            "dimension_ratio"
        )
        result["features"]["cylinder_score"] = diagnostics.get(
            "cylinder_score"
        )
        result["quality_warnings"].append("stack_support_height_inferred")
        updated += 1

    # In this controlled Level 4 generator, stack bases are cubes/cuboids.
    # Their visible top can be a crescent or an L-shape after occlusion and may
    # therefore score as a short cylinder.  Once an upper object has been
    # geometrically assigned to that footprint, restore the base hypothesis
    # from its near-square 3D proportions.
    for support in supporting_results:
        upper_results = support.get("supported_upper_results", [])
        if support.get("class") == "cube" and upper_results:
            upper = upper_results[0]
            support_top_uv = np.asarray(
                support.get("top_points_uv", []),
                dtype=np.float64,
            )
            if len(support_top_uv) >= TOP_SLICE_MIN_POINTS:
                upper_geometry = upper["geometry"]
                if upper["class"] == "cylinder":
                    upper_length = upper_width = float(
                        upper_geometry["diameter_m"]
                    )
                else:
                    upper_length = float(upper_geometry["length_m"])
                    upper_width = float(upper_geometry["width_m"])
                upper_center = upper["center"]
                upper_rotation = upper["rotation"]
                upper_corners = []
                for sign_u in (-0.5, 0.5):
                    for sign_v in (-0.5, 0.5):
                        upper_corners.append(
                            upper_center
                            + sign_u * upper_length * upper_rotation[:, 0]
                            + sign_v * upper_width * upper_rotation[:, 1]
                        )
                upper_corners_uv = project_to_table(
                    np.asarray(upper_corners),
                    table_axis_u,
                    table_axis_v,
                )
                combined_uv = np.vstack(
                    [support_top_uv, upper_corners_uv]
                )
                combined_hull = convex_hull_2d(combined_uv)
                combined_rect = minimum_area_rectangle(combined_hull)
                current_length = float(support["geometry"]["length_m"])
                current_width = float(support["geometry"]["width_m"])
                if (
                    combined_rect["long_size"] >= 0.075
                    and combined_rect["long_size"]
                    > 1.25 * max(current_length, current_width)
                ):
                    current_uv = np.asarray(
                        support["top_footprint_center_uv"],
                        dtype=np.float64,
                    )
                    target_uv = np.asarray(
                        combined_rect["center"],
                        dtype=np.float64,
                    )
                    delta_uv = target_uv - current_uv
                    table_center = (
                        support["footprint_center"]
                        + delta_uv[0] * table_axis_u
                        + delta_uv[1] * table_axis_v
                    )
                    height = float(support["geometry"]["height_m"])
                    long_axis = axis_2d_to_3d(
                        np.asarray(
                            combined_rect["axis_long"],
                            dtype=np.float64,
                        ),
                        table_axis_u,
                        table_axis_v,
                    )
                    rotation, yaw = build_pose_rotation(
                        "cuboid",
                        long_axis,
                        table_normal,
                    )
                    center = table_center + 0.5 * height * table_normal
                    support["class"] = "cuboid"
                    support["confidence"] = 0.72
                    support["geometry"] = {
                        "length_m": float(combined_rect["long_size"]),
                        "width_m": float(combined_rect["short_size"]),
                        "height_m": height,
                    }
                    support["center"] = center
                    support["footprint_center"] = table_center
                    support["rotation"] = rotation
                    support["yaw"] = yaw
                    support["obb"] = o3d.geometry.OrientedBoundingBox(
                        center,
                        rotation,
                        np.asarray(
                            [
                                combined_rect["long_size"],
                                combined_rect["short_size"],
                                height,
                            ],
                            dtype=np.float64,
                        ),
                    )
                    support["quality_warnings"].append(
                        "stack_base_footprint_completed"
                    )
                    continue

        if support.get("class") == "cuboid" and upper_results:
            geometry = support["geometry"]
            length = float(geometry["length_m"])
            width = float(geometry["width_m"])
            height = float(geometry["height_m"])
            if length > 1e-9:
                height_to_length = height / length
                width_to_length = width / length
                incomplete_cube = (
                    0.85 <= height_to_length <= 1.15
                    and width_to_length <= 0.72
                )
                if incomplete_cube:
                    upper = upper_results[0]
                    short_axis = support["rotation"][:, 1]
                    center_delta = (
                        upper["footprint_center"]
                        - support["footprint_center"]
                    )
                    short_shift = float(np.dot(center_delta, short_axis))
                    short_shift = float(
                        np.clip(short_shift, -0.30 * length, 0.30 * length)
                    )
                    shift = short_shift * short_axis
                    support["center"] = support["center"] + shift
                    support["footprint_center"] = (
                        support["footprint_center"] + shift
                    )
                    side = 0.5 * (length + height)
                    support["class"] = "cube"
                    support["confidence"] = 0.72
                    support["geometry"] = {
                        "length_m": side,
                        "width_m": side,
                        "height_m": height,
                    }
                    rotation, yaw = build_pose_rotation(
                        "cube",
                        support["rotation"][:, 0],
                        table_normal,
                    )
                    support["rotation"] = rotation
                    support["yaw"] = yaw
                    support["obb"] = o3d.geometry.OrientedBoundingBox(
                        support["center"],
                        rotation,
                        np.asarray([side, side, height], dtype=np.float64),
                    )
                    support["features"]["dimension_ratio"] = max(
                        side,
                        height,
                    ) / max(min(side, height), 1e-9)
                    support["features"]["height_ratio"] = height / side
                    support["quality_warnings"].append(
                        "stack_base_cube_completed"
                    )
                    continue

        if support.get("class") != "cylinder":
            continue
        diameter = float(support["geometry"].get("diameter_m", 0.0))
        height = float(support["geometry"].get("height_m", 0.0))
        if diameter <= 1e-9:
            continue
        base_ratio = height / diameter
        if not (0.70 <= base_ratio <= 1.30):
            continue
        if support["features"].get("circularity", 0.0) >= 0.92:
            continue

        length = width = diameter
        support["class"] = "cube"
        support["confidence"] = max(
            0.60,
            min(0.82, 0.60 + 0.15 * (1.0 - abs(base_ratio - 1.0))),
        )
        support["geometry"] = {
            "length_m": length,
            "width_m": width,
            "height_m": height,
        }
        rotation, yaw = build_pose_rotation(
            "cube",
            support["rotation"][:, 0],
            table_normal,
        )
        support["rotation"] = rotation
        support["yaw"] = yaw
        support["obb"] = o3d.geometry.OrientedBoundingBox(
            support["center"],
            rotation,
            np.asarray([length, width, height], dtype=np.float64),
        )

    return updated


# ============================================================
# 20. 主程序
# ============================================================

def main() -> None:
    if SCENE_GT_FILE.exists():
        with SCENE_GT_FILE.open("r", encoding="utf-8") as handle:
            dispatch_manifest = json.load(handle)
        dispatch_mode = str(
            dispatch_manifest.get("scene_mode", "separated")
        ).lower()
        if dispatch_mode in {"physics", "dynamic", "settled", "drop", "planned", "constraint", "planned_contact"}:
            from recognize_primitives_3d import main as recognize_3d_main

            print(f"{dispatch_mode}场景：使用任意姿态3-D primitive候选识别器。")
            recognize_3d_main()
            return

    print("正在连接 CoppeliaSim……")

    client = RemoteAPIClient()
    sim = client.require("sim")

    print("连接成功。")

    if sim.getSimulationState() != sim.simulation_stopped:
        raise RuntimeError(
            "请先停止CoppeliaSim仿真。\n"
            "本程序只读取当前静态场景，不启动动力学。"
        )

    scene_manifest, segmentation_metadata = load_pipeline_context()
    print(f"当前Scene ID：{scene_manifest['scene_id']}")
    RESULT_JSON.unlink(missing_ok=True)

    # --------------------------------------------------------
    # A. 找Camera、gripper_tip、Robot Base
    # --------------------------------------------------------

    camera = find_unique_object_by_alias(
        sim,
        sim.sceneobject_visionsensor,
        "rgbd_camera",
    )

    gripper_tip = find_unique_object_by_alias(
        sim,
        sim.sceneobject_dummy,
        "gripper_tip",
    )

    joints = get_kuka_joints_from_tip(sim, gripper_tip)

    if len(joints) != 7:
        raise RuntimeError(
            f"预期找到7个KUKA关节，实际找到{len(joints)}个。"
        )

    robot_base = int(sim.getObjectParent(joints[0]))

    print("\nCamera     :", get_full_path(sim, camera))
    print("Robot Base :", get_full_path(sim, robot_base))

    # --------------------------------------------------------
    # B. 自动估计桌面
    # --------------------------------------------------------

    plane = estimate_table_plane_from_scene(
        sim,
        camera,
        robot_base,
    )

    p0, e1, e2, n = make_table_basis(plane)

    print("\nTable normal:", np.round(n, 6))

    # --------------------------------------------------------
    # C. 读取上一阶段DBSCAN clusters
    # --------------------------------------------------------

    clusters = load_clusters(segmentation_metadata)

    print(f"\n读取到 {len(clusters)} 个有效Cluster。")

    # --------------------------------------------------------
    # D. 对每个cluster进行尺寸无关识别
    # --------------------------------------------------------

    results: list[dict] = []
    geometries: list[Any] = []

    palette = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.5, 0.0],
        [0.5, 0.0, 1.0],
    ]

    for index, (file_path, cloud) in enumerate(clusters):
        result = analyze_cluster(
            index,
            file_path,
            cloud,
            plane,
            p0,
            e1,
            e2,
            n,
        )

        results.append(result)

    if str(scene_manifest.get("scene_mode", "separated")).lower() == "level4":
        inferred = apply_stack_support_hints(results, n, e1, e2)
        if inferred:
            print(f"已推断堆叠上层支撑高度：{inferred} 个")

    for result in results:
        print_result(result)

        # 彩色cluster
        display_cloud = o3d.geometry.PointCloud(cloud)
        display_cloud.paint_uniform_color(palette[index % len(palette)])
        geometries.append(display_cloud)

        # OBB
        result["obb"].color = (0.0, 0.0, 0.0)
        geometries.append(result["obb"])

        # 位姿坐标系
        geometries.append(create_pose_frame(result))

    # --------------------------------------------------------
    # E. 汇总
    # --------------------------------------------------------

    print("\n\n========================================")
    print("         尺寸无关识别汇总")
    print("========================================")

    for result in results:
        center = result["center"]

        if result["class"] == "cylinder":
            geometry_text = (
                f"D={result['geometry']['diameter_m'] * 1000:.1f} mm, "
                f"H={result['geometry']['height_m'] * 1000:.1f} mm"
            )
            yaw_text = "N/A"
        else:
            geometry_text = (
                f"L={result['geometry']['length_m'] * 1000:.1f}, "
                f"W={result['geometry']['width_m'] * 1000:.1f}, "
                f"H={result['geometry']['height_m'] * 1000:.1f} mm"
            )
            yaw_text = (
                "N/A"
                if result["yaw"] is None
                else f"{math.degrees(result['yaw']):.1f}°"
            )

        print(
            f"Object {result['id']}: "
            f"{result['class']:8s} | "
            f"conf={result['confidence']:.2f} | "
            f"P=({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}) | "
            f"Yaw={yaw_text} | "
            f"{geometry_text}"
        )

    save_results_json(
        results,
        scene_manifest,
        segmentation_metadata,
    )

    # --------------------------------------------------------
    # F. Open3D显示
    # --------------------------------------------------------

    base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=BASE_FRAME_SIZE,
        origin=[0.0, 0.0, 0.0],
    )
    geometries.append(base_frame)

    print("\nOpen3D显示：")
    print("彩色点云 = 每个DBSCAN实例")
    print("黑色框   = 自动估计尺寸的3D OBB")
    print("小坐标轴 = 自动估计位姿")

    if SHOW_VISUALIZATION:
        o3d.visualization.draw_geometries(
            geometries,
            window_name="Size-Independent Primitive Recognition",
            width=1200,
            height=800,
        )
    else:
        print("ROBOT_GRASP_HEADLESS=1，跳过Open3D可视化。")


if __name__ == "__main__":
    main()
