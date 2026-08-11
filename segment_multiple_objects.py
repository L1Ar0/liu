from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import open3d as o3d

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from geometric_segmentation import refine_geometric_clusters

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
# 1. 参数
# ============================================================

OUTPUT_DIR = Path(
    "segmentation_output"
)
SCENE_GT_FILE = Path("random_scene_ground_truth.json")
SEGMENTATION_METADATA_FILE = OUTPUT_DIR / "segmentation_metadata.json"
VIEW_01_OBJECT_CLOUD_FILE = OUTPUT_DIR / "view_01_object_cloud.ply"
SHOW_VISUALIZATION = os.environ.get("ROBOT_GRASP_HEADLESS") != "1"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def load_scene_manifest() -> dict:
    if not SCENE_GT_FILE.exists():
        raise RuntimeError(
            "找不到random_scene_ground_truth.json。"
            "请先运行scene_randomizer.py。"
        )

    with open(SCENE_GT_FILE, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    scene_id = manifest.get("scene_id")
    if not scene_id:
        raise RuntimeError(
            "Ground Truth缺少scene_id，说明它是旧格式。"
            "请重新运行scene_randomizer.py。"
        )

    return manifest


def clear_old_clusters() -> None:
    for path in OUTPUT_DIR.glob("cluster_*.ply"):
        if path.is_file():
            path.unlink()
    SEGMENTATION_METADATA_FILE.unlink(missing_ok=True)


# ------------------------------------------------------------
# 点云降采样
# ------------------------------------------------------------

# 4 mm voxel。
#
# 对40~70 mm工件来说比较合适。
VOXEL_SIZE = 0.003


# ------------------------------------------------------------
# 离群点过滤
# ------------------------------------------------------------

OUTLIER_NB_NEIGHBORS = 20
OUTLIER_STD_RATIO = 2.0


# ------------------------------------------------------------
# RANSAC桌面
# ------------------------------------------------------------

PLANE_DISTANCE_THRESHOLD = 0.004
PLANE_RANSAC_N = 3
PLANE_ITERATIONS = 1500


# ------------------------------------------------------------
# 桌面上方工件区域
# ------------------------------------------------------------

# 距离桌面至少6 mm才认为属于工件。
#
# 可以去掉桌面残留噪声。
MIN_OBJECT_HEIGHT = 0.006

# 只保留桌面上方12 cm以内的点。
#
# 你的工件最高约6 cm，
# RG2通常远高于12 cm，
# 因此这个参数能帮助去掉夹爪点云。
MAX_OBJECT_HEIGHT = 0.12


# ------------------------------------------------------------
# DBSCAN
# ------------------------------------------------------------

# 邻域搜索半径9 mm。随机化器保证工件之间至少保留8 mm间隔；使用
# 小于12 mm的半径，避免相邻工件在深度采样后被DBSCAN串成一个簇。
DBSCAN_EPS = 0.009

# 一个核心区域最少需要多少点。
DBSCAN_MIN_POINTS = 15

# DBSCAN完成后，
# 少于这么多点的cluster直接视为噪声。
MIN_CLUSTER_POINTS = 40
MIN_CLUSTER_POINTS_LEVEL4 = 20

# DBSCAN can split a sparsely observed side face from the top surface. Only
# small fragments whose XY support overlaps another cluster are merged.
FRAGMENT_MAX_POINTS = 100
FRAGMENT_XY_OVERLAP_MARGIN_M = 0.004
# When an upper object hides the middle of a lower top plane, the two visible
# pieces can be separated by the full footprint of the upper object.  Keep
# this merge limited to same-height fragments and only use it while the
# controlled benchmark still has more clusters than expected objects.
COPLANAR_FRAGMENT_MAX_GAP_M = 0.030
STACK_SPLIT_MIN_SPAN_M = 0.030
STACK_SPLIT_BIN_SIZE_M = 0.003
STACK_SPLIT_MIN_PEAK_POINTS = 45
STACK_SPLIT_SUPPORT_TOLERANCE_M = 0.003


# ============================================================
# 2. 打印点云信息
# ============================================================

def print_cloud_info(
    title: str,
    cloud: o3d.geometry.PointCloud,
):
    points = np.asarray(
        cloud.points
    )

    print(
        f"\n========== {title} =========="
    )

    print(
        f"Point count = {len(points)}"
    )

    if len(points) == 0:
        return

    minimum = points.min(
        axis=0
    )

    maximum = points.max(
        axis=0
    )

    print(
        f"X: {minimum[0]:.4f}"
        f" ~ {maximum[0]:.4f} m"
    )

    print(
        f"Y: {minimum[1]:.4f}"
        f" ~ {maximum[1]:.4f} m"
    )

    print(
        f"Z: {minimum[2]:.4f}"
        f" ~ {maximum[2]:.4f} m"
    )


# ============================================================
# 3. Voxel Downsample
# ============================================================

def voxel_downsample(
    cloud: o3d.geometry.PointCloud,
    voxel_size: float | None = None,
):
    print(
        "\n正在进行Voxel Downsample..."
    )

    downsampled = (
        cloud.voxel_down_sample(
            voxel_size=float(voxel_size if voxel_size is not None else VOXEL_SIZE)
        )
    )

    print(
        f"降采样前：{len(cloud.points)}"
    )

    print(
        f"降采样后：{len(downsampled.points)}"
    )

    return downsampled


# ============================================================
# 4. Statistical Outlier Removal
# ============================================================

def remove_outliers(
    cloud: o3d.geometry.PointCloud,
):
    print(
        "\n正在进行离群点过滤..."
    )

    filtered_cloud, indices = (
        cloud.remove_statistical_outlier(
            nb_neighbors=
                OUTLIER_NB_NEIGHBORS,

            std_ratio=
                OUTLIER_STD_RATIO,
        )
    )

    print(
        f"过滤前：{len(cloud.points)}"
    )

    print(
        f"过滤后：{len(filtered_cloud.points)}"
    )

    return filtered_cloud


# ============================================================
# 5. RANSAC寻找桌面
# ============================================================

def detect_table_plane(
    cloud: o3d.geometry.PointCloud,
):

    print(
        "\n正在使用RANSAC检测桌面..."
    )

    plane_model, inliers = (
        cloud.segment_plane(
            distance_threshold=
                PLANE_DISTANCE_THRESHOLD,

            ransac_n=
                PLANE_RANSAC_N,

            num_iterations=
                PLANE_ITERATIONS,
        )
    )

    plane_model = np.asarray(
        plane_model,
        dtype=np.float64,
    )

    a, b, c, d = plane_model

    print(
        "\n========== RANSAC平面 =========="
    )

    print(
        f"{a:.6f} x + "
        f"{b:.6f} y + "
        f"{c:.6f} z + "
        f"{d:.6f} = 0"
    )

    print(
        f"Plane points = {len(inliers)}"
    )

    table_cloud = (
        cloud.select_by_index(
            inliers
        )
    )

    non_table_cloud = (
        cloud.select_by_index(
            inliers,
            invert=True,
        )
    )

    return (
        plane_model,
        table_cloud,
        non_table_cloud,
    )


# ============================================================
# 6. 统一桌面法向方向
# ============================================================

def orient_plane_toward_camera(
    sim,
    plane_model: np.ndarray,
    camera: int,
    robot_base: int,
):
    """
    RANSAC平面：

        ax + by + cz + d = 0

    法向量方向可能朝上，也可能朝下。

    我们利用Camera的位置判断哪一面是“桌面上方”。

    RGB-D Camera显然应该位于桌面的上方，
    所以令Camera所在侧的signed distance为正。
    """

    plane = plane_model.copy()

    normal = plane[:3]

    norm = np.linalg.norm(
        normal
    )

    if norm < 1e-9:
        raise RuntimeError(
            "RANSAC返回了无效平面。"
        )

    plane = plane / norm

    camera_position = np.asarray(
        sim.getObjectPosition(
            camera,
            robot_base,
        ),
        dtype=np.float64,
    )

    camera_signed_distance = (
        np.dot(
            plane[:3],
            camera_position,
        )
        + plane[3]
    )

    # 如果camera在负半空间，
    # 把整个平面方程乘-1。
    if camera_signed_distance < 0:
        plane *= -1.0

    print(
        "\nCamera距离桌面："
        f"{abs(camera_signed_distance):.4f} m"
    )

    print(
        "已经统一桌面法向量："
        "朝向Camera所在一侧。"
    )

    return plane


# ============================================================
# 7. 根据桌面高度提取工件
# ============================================================

def extract_objects_above_table(
    cloud: o3d.geometry.PointCloud,
    plane_model: np.ndarray,
):
    """
    只保留：

        6 mm < 点到桌面高度 < 120 mm

    因此：

        桌面 → 去掉
        很高处RG2 → 去掉
        桌面上的工件 → 保留
    """

    points = np.asarray(
        cloud.points
    )

    colors = np.asarray(
        cloud.colors
    )

    normal = (
        plane_model[:3]
    )

    d = plane_model[3]

    heights = (
        points @ normal
        + d
    )

    mask = (
        (heights > MIN_OBJECT_HEIGHT)
        &
        (heights < MAX_OBJECT_HEIGHT)
    )

    object_points = points[
        mask
    ]

    if len(colors) == len(points):
        object_colors = colors[
            mask
        ]
    else:
        object_colors = np.zeros(
            (
                len(object_points),
                3,
            ),
            dtype=np.float64,
        )

    object_cloud = (
        o3d.geometry.PointCloud()
    )

    object_cloud.points = (
        o3d.utility.Vector3dVector(
            object_points
        )
    )

    object_cloud.colors = (
        o3d.utility.Vector3dVector(
            object_colors
        )
    )

    print(
        "\n========== 桌面上方区域 =========="
    )

    print(
        f"Candidate object points = "
        f"{len(object_points)}"
    )

    return (
        object_cloud,
        heights,
        mask,
    )


# ============================================================
# 8. DBSCAN
# ============================================================

def _cluster_xy_bounds(
    cluster: o3d.geometry.PointCloud,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(cluster.points, dtype=np.float64)
    return points[:, :2].min(axis=0), points[:, :2].max(axis=0)


def _cluster_z_bounds(
    cluster: o3d.geometry.PointCloud,
) -> tuple[float, float]:
    points = np.asarray(cluster.points, dtype=np.float64)
    return float(points[:, 2].min()), float(points[:, 2].max())


def _xy_overlap_score(
    first: o3d.geometry.PointCloud,
    second: o3d.geometry.PointCloud,
) -> float:
    first_min, first_max = _cluster_xy_bounds(first)
    second_min, second_max = _cluster_xy_bounds(second)
    overlap = np.maximum(
        0.0,
        np.minimum(first_max, second_max)
        - np.maximum(first_min, second_min),
    )
    return float(overlap[0] * overlap[1])


def _xy_bbox_gap(
    first: o3d.geometry.PointCloud,
    second: o3d.geometry.PointCloud,
) -> float:
    first_min, first_max = _cluster_xy_bounds(first)
    second_min, second_max = _cluster_xy_bounds(second)
    gap = np.maximum(
        0.0,
        np.maximum(first_min - second_max, second_min - first_max),
    )
    return float(np.linalg.norm(gap))


def _cluster_median_z(
    cluster: o3d.geometry.PointCloud,
) -> float:
    return float(np.median(np.asarray(cluster.points)[:, 2]))


def _can_merge_fragment(
    fragment: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
) -> bool:
    fragment_min, fragment_max = _cluster_xy_bounds(fragment)
    target_min, target_max = _cluster_xy_bounds(target)
    target_min = target_min - FRAGMENT_XY_OVERLAP_MARGIN_M
    target_max = target_max + FRAGMENT_XY_OVERLAP_MARGIN_M
    xy_overlap = np.minimum(fragment_max, target_max) - np.maximum(
        fragment_min,
        target_min,
    )
    if np.any(xy_overlap <= 0.0):
        return False

    fragment_z_min, fragment_z_max = _cluster_z_bounds(fragment)
    target_z_min, target_z_max = _cluster_z_bounds(target)
    return not (
        fragment_z_min > target_z_max + 0.025
        or target_z_min > fragment_z_max + 0.025
    )


def _merge_clouds(
    first: o3d.geometry.PointCloud,
    second: o3d.geometry.PointCloud,
) -> o3d.geometry.PointCloud:
    merged = o3d.geometry.PointCloud()
    merged.points = o3d.utility.Vector3dVector(
        np.vstack([np.asarray(first.points), np.asarray(second.points)])
    )

    first_colors = np.asarray(first.colors)
    second_colors = np.asarray(second.colors)
    if (
        len(first_colors) == len(first.points)
        and len(second_colors) == len(second.points)
    ):
        merged.colors = o3d.utility.Vector3dVector(
            np.vstack([first_colors, second_colors])
        )
    return merged


def merge_split_fragments(
    clusters: list[o3d.geometry.PointCloud],
    expected_count: int,
) -> tuple[list[o3d.geometry.PointCloud], int]:
    """Merge small overlapping fragments until the manifest count is met."""

    clusters = list(clusters)
    merged_count = 0
    while len(clusters) > expected_count:
        fragment_index = min(
            range(len(clusters)),
            key=lambda index: len(clusters[index].points),
        )
        fragment = clusters[fragment_index]
        if len(fragment.points) > FRAGMENT_MAX_POINTS:
            break

        candidates = [
            index
            for index, target in enumerate(clusters)
            if index != fragment_index
            and len(target.points) >= len(fragment.points)
            and _can_merge_fragment(fragment, target)
        ]
        if not candidates:
            break

        target_index = max(
            candidates,
            key=lambda index: _xy_overlap_score(fragment, clusters[index]),
        )
        clusters[target_index] = _merge_clouds(
            clusters[target_index],
            fragment,
        )
        clusters.pop(fragment_index)
        merged_count += 1

    return clusters, merged_count


def merge_coplanar_fragments(
    clusters: list[o3d.geometry.PointCloud],
    expected_count: int,
) -> tuple[list[o3d.geometry.PointCloud], int]:
    """Merge adjacent pieces of one horizontal support surface."""

    clusters = list(clusters)
    merged_count = 0
    while len(clusters) > expected_count:
        candidates: list[tuple[float, float, int, int]] = []
        for first_index in range(len(clusters)):
            for second_index in range(first_index + 1, len(clusters)):
                first = clusters[first_index]
                second = clusters[second_index]
                if min(len(first.points), len(second.points)) > 180:
                    continue
                z_delta = abs(_cluster_median_z(first) - _cluster_median_z(second))
                gap = _xy_bbox_gap(first, second)
                if z_delta <= 0.004 and gap <= COPLANAR_FRAGMENT_MAX_GAP_M:
                    candidates.append((z_delta, gap, first_index, second_index))

        if not candidates:
            break

        _, _, first_index, second_index = min(candidates)
        clusters[first_index] = _merge_clouds(
            clusters[first_index],
            clusters[second_index],
        )
        clusters.pop(second_index)
        merged_count += 1

    return clusters, merged_count


def _cloud_from_points(
    points: np.ndarray,
    colors: np.ndarray,
) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    if len(colors) == len(points):
        cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud


def split_height_layers(
    clusters: list[o3d.geometry.PointCloud],
    min_cluster_points: int = MIN_CLUSTER_POINTS,
) -> tuple[list[o3d.geometry.PointCloud], int]:
    """Split a Level 4 stack cluster at a strong horizontal surface peak."""

    result: list[o3d.geometry.PointCloud] = []
    split_count = 0
    for cluster in clusters:
        points = np.asarray(cluster.points, dtype=np.float64)
        colors = np.asarray(cluster.colors)
        if len(points) < min_cluster_points:
            result.append(cluster)
            continue

        z_min = float(points[:, 2].min())
        z_max = float(points[:, 2].max())
        span = z_max - z_min
        if span < STACK_SPLIT_MIN_SPAN_M:
            result.append(cluster)
            continue

        bin_count = max(
            16,
            min(40, int(math.ceil(span / STACK_SPLIT_BIN_SIZE_M))),
        )
        counts, edges = np.histogram(
            points[:, 2],
            bins=bin_count,
            range=(z_min, z_max + 1e-9),
        )
        median_count = float(np.median(counts))
        candidates = [
            index
            for index, count in enumerate(counts)
            if z_min + 0.02 * span
            <= 0.5 * (edges[index] + edges[index + 1])
            <= z_max - 0.15 * span
            and int(count) >= STACK_SPLIT_MIN_PEAK_POINTS
            and float(count) >= max(2.0 * median_count, 1.0)
        ]
        if not candidates:
            result.append(cluster)
            continue

        peak_index = max(candidates, key=lambda index: int(counts[index]))
        # Keep the complete lower horizontal support surface together.  Using
        # only the histogram edge can send points exactly on that plane into
        # the upper cluster because of floating-point/depth quantization.
        lower_surface = float(
            0.5 * (edges[peak_index] + edges[peak_index + 1])
        )
        threshold = lower_surface + STACK_SPLIT_SUPPORT_TOLERANCE_M
        low_mask = points[:, 2] <= threshold
        high_mask = ~low_mask
        if (
            int(low_mask.sum()) < min_cluster_points
            or int(high_mask.sum()) < min_cluster_points
        ):
            result.append(cluster)
            continue
        # A single cone/cylinder can also contain a dense cap histogram peak.
        # A true stack interface has a substantial lower object below that
        # plane; reject cap fragments whose lower layer is only a thin band.
        if (
            float(np.ptp(points[low_mask, 2])) < 0.012
            or float(np.ptp(points[high_mask, 2])) < 0.035
        ):
            result.append(cluster)
            continue

        result.append(
            _cloud_from_points(
                points[low_mask],
                colors[low_mask] if len(colors) == len(points) else colors,
            )
        )
        result.append(
            _cloud_from_points(
                points[high_mask],
                colors[high_mask] if len(colors) == len(points) else colors,
            )
        )
        split_count += 1

    return result, split_count


def reassign_shared_support_plane_points(
    clusters: list[o3d.geometry.PointCloud],
    min_cluster_points: int = MIN_CLUSTER_POINTS,
) -> tuple[list[o3d.geometry.PointCloud], int]:
    """Return support-plane points left in an upper stack cluster."""

    clusters = list(clusters)
    reassigned = 0
    for lower_index, lower in enumerate(clusters):
        lower_points = np.asarray(lower.points, dtype=np.float64)
        if len(lower_points) < min_cluster_points:
            continue
        if float(np.ptp(lower_points[:, 2])) > 0.010:
            continue

        lower_height = float(np.median(lower_points[:, 2]))
        for upper_index, upper in enumerate(clusters):
            if upper_index == lower_index:
                continue
            upper_points = np.asarray(upper.points, dtype=np.float64)
            if len(upper_points) < min_cluster_points:
                continue
            if float(np.ptp(upper_points[:, 2])) < 0.020:
                continue
            if _xy_bbox_gap(lower, upper) > 0.020:
                continue
            if float(upper_points[:, 2].min()) > lower_height + 0.006:
                continue
            if float(upper_points[:, 2].max()) < lower_height + 0.020:
                continue

            transfer_mask = np.abs(
                upper_points[:, 2] - lower_height
            ) <= 0.0035
            transfer_count = int(transfer_mask.sum())
            if transfer_count < 5:
                continue
            if len(upper_points) - transfer_count < min_cluster_points:
                continue

            upper_colors = np.asarray(upper.colors)
            transfer_cloud = _cloud_from_points(
                upper_points[transfer_mask],
                upper_colors[transfer_mask]
                if len(upper_colors) == len(upper_points)
                else upper_colors,
            )
            clusters[lower_index] = _merge_clouds(
                clusters[lower_index],
                transfer_cloud,
            )
            clusters[upper_index] = _cloud_from_points(
                upper_points[~transfer_mask],
                upper_colors[~transfer_mask]
                if len(upper_colors) == len(upper_points)
                else upper_colors,
            )
            lower = clusters[lower_index]
            lower_points = np.asarray(lower.points, dtype=np.float64)
            reassigned += transfer_count

    return clusters, reassigned


def run_dbscan(
    cloud: o3d.geometry.PointCloud,
    expected_count: int | None = None,
    enable_layer_split: bool = False,
    min_cluster_points: int = MIN_CLUSTER_POINTS,
):
    print(
        "\n正在执行DBSCAN实例分割..."
    )

    labels = np.asarray(
        cloud.cluster_dbscan(
            eps=DBSCAN_EPS,
            min_points=
                DBSCAN_MIN_POINTS,
            print_progress=True,
        )
    )

    if len(labels) == 0:
        raise RuntimeError(
            "DBSCAN没有收到点。"
        )

    valid_labels = labels[
        labels >= 0
    ]

    if len(valid_labels) == 0:
        raise RuntimeError(
            "\nDBSCAN没有找到任何cluster。\n"
            "优先尝试：\n"
            "DBSCAN_EPS = 0.015"
        )

    raw_cluster_count = (
        int(valid_labels.max())
        + 1
    )

    print(
        f"\nDBSCAN原始Cluster数量："
        f"{raw_cluster_count}"
    )

    # --------------------------------------------------------
    # 删除太小的cluster
    # --------------------------------------------------------

    clusters = []

    for cluster_id in range(
        raw_cluster_count
    ):

        indices = np.where(
            labels == cluster_id
        )[0]

        if (
            len(indices)
            < min_cluster_points
        ):
            print(
                f"Cluster {cluster_id}: "
                f"{len(indices)} points "
                "→ 视为小噪声，删除"
            )

            continue

        cluster = (
            cloud.select_by_index(
                indices
            )
        )

        clusters.append(
            cluster
        )

    coplanar_merges = 0
    if expected_count is not None and len(clusters) > expected_count:
        clusters, coplanar_merges = merge_coplanar_fragments(
            clusters,
            expected_count,
        )

    layer_splits = 0
    if enable_layer_split:
        clusters, layer_splits = split_height_layers(
            clusters,
            min_cluster_points=min_cluster_points,
        )

    reassigned_support_points = 0
    if enable_layer_split:
        clusters, reassigned_support_points = (
            reassign_shared_support_plane_points(
                clusters,
                min_cluster_points=min_cluster_points,
            )
        )

    merged_fragments = 0
    if expected_count is not None and len(clusters) > expected_count:
        clusters, merged_fragments = merge_split_fragments(
            clusters,
            expected_count,
        )

    # Fragment merging can create the final planar support cluster, so run the
    # shared-plane correction once more after that merge.
    if enable_layer_split:
        clusters, post_merge_reassigned = (
            reassign_shared_support_plane_points(
                clusters,
                min_cluster_points=min_cluster_points,
            )
        )
        reassigned_support_points += post_merge_reassigned

    print(
        "\n有效工件Cluster数量："
        f"{len(clusters)}"
    )
    if merged_fragments:
        print(f"已合并DBSCAN小碎片：{merged_fragments} 个")
    if coplanar_merges:
        print(f"已合并共面遮挡碎片：{coplanar_merges} 个")
    if layer_splits:
        print(f"已按高度层拆分堆叠簇：{layer_splits} 个")

    return (
        labels,
        clusters,
    )


# ============================================================
# 9. 分析每一个Cluster
# ============================================================

def analyze_clusters(
    clusters,
):

    results = []

    print(
        "\n\n========================================"
    )

    print(
        "             工件实例"
    )

    print(
        "========================================"
    )

    for index, cluster in enumerate(
        clusters
    ):

        points = np.asarray(
            cluster.points
        )

        center = np.mean(
            points,
            axis=0,
        )

        aabb = (
            cluster
            .get_axis_aligned_bounding_box()
        )

        extent = np.asarray(
            aabb.get_extent()
        )

        result = {
            "id": index,
            "points": len(points),
            "center": center,
            "aabb_extent": extent,
        }

        results.append(
            result
        )

        print(
            f"\nObject {index}"
        )

        print(
            f"Points = {len(points)}"
        )

        print(
            "可见点云中心："
        )

        print(
            f"X = {center[0]:.4f} m"
        )

        print(
            f"Y = {center[1]:.4f} m"
        )

        print(
            f"Z = {center[2]:.4f} m"
        )

        print(
            "AABB extent："
        )

        print(
            f"{extent[0] * 1000:.1f} × "
            f"{extent[1] * 1000:.1f} × "
            f"{extent[2] * 1000:.1f} mm"
        )

    return results


# ============================================================
# 10. 保存Cluster
# ============================================================

def save_clusters(
    clusters,
    results: list[dict],
    scene_manifest: dict,
    plane_model: np.ndarray,
    segmentation_method: str = "dbscan",
    view_metadata: dict | None = None,
    geometric_diagnostics: list[dict] | None = None,
):

    print(
        "\n========== 保存Cluster =========="
    )

    cluster_files = []

    for index, cluster in enumerate(
        clusters
    ):

        file_path = (
            OUTPUT_DIR
            /
            f"cluster_{index:02d}.ply"
        )

        success = (
            o3d.io.write_point_cloud(
                str(file_path),
                cluster,
            )
        )

        print(
            f"{file_path} "
            f"→ {success}"
        )

        if not success:
            raise RuntimeError(f"保存Cluster失败：{file_path}")

        cluster_files.append(file_path.name)

    metadata = {
        "scene_id": scene_manifest["scene_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cluster_count": len(clusters),
        "cluster_files": cluster_files,
        "segmentation_method": segmentation_method,
        "uses_color_features": False,
        "table_plane": np.asarray(plane_model, dtype=float).tolist(),
        "views": view_metadata or {"view_count": 1},
        "geometric_split_diagnostics": geometric_diagnostics or [],
        "parameters": {
            "voxel_size_m": VOXEL_SIZE,
            "plane_distance_threshold_m": PLANE_DISTANCE_THRESHOLD,
            "min_object_height_m": MIN_OBJECT_HEIGHT,
            "max_object_height_m": MAX_OBJECT_HEIGHT,
            "dbscan_eps_m": DBSCAN_EPS,
            "dbscan_min_points": DBSCAN_MIN_POINTS,
            "min_cluster_points": int(
                MIN_CLUSTER_POINTS_LEVEL4
                if str(scene_manifest.get("scene_mode", "separated")).lower()
                == "level4"
                else MIN_CLUSTER_POINTS
            ),
            "level4_height_layer_split": (
                str(scene_manifest.get("scene_mode", "separated")).lower()
                == "level4"
            ),
        },
        "clusters": [
            {
                "id": int(item["id"]),
                "points": int(item["points"]),
                "center_m": np.asarray(item["center"], dtype=float).tolist(),
                "aabb_extent_m": np.asarray(
                    item["aabb_extent"], dtype=float
                ).tolist(),
            }
            for item in results
        ],
    }

    with open(
        SEGMENTATION_METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"分割元数据：{SEGMENTATION_METADATA_FILE.resolve()}")

    if False:  # segmentation never uses Ground Truth object count
        print(
            "\n警告：有效Cluster数量与GT工件数量不一致："
            f"{len(clusters)} != {scene_manifest['object_count']}。"
        )


# ============================================================
# 11. 给Cluster上不同颜色
# ============================================================

def colorize_clusters(
    clusters,
):

    # 只是为了可视化。
    palette = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.5, 0.0],
        [0.5, 0.0, 1.0],
        [0.0, 0.5, 1.0],
        [0.5, 1.0, 0.0],
    ]

    colored_clusters = []

    for index, cluster in enumerate(
        clusters
    ):

        display_cluster = (
            o3d.geometry.PointCloud(
                cluster
            )
        )

        color = palette[
            index % len(palette)
        ]

        display_cluster.paint_uniform_color(
            color
        )

        colored_clusters.append(
            display_cluster
        )

    return colored_clusters


# ============================================================
# 12. 主程序
# ============================================================

def main():

    print(
        "正在连接CoppeliaSim..."
    )

    client = RemoteAPIClient()

    sim = client.require("sim")

    print("连接成功。")

    scene_manifest = load_scene_manifest()
    scene_mode = str(scene_manifest.get("scene_mode", "separated")).lower()
    simulation_state = sim.getSimulationState()
    paused_allowed = scene_mode in {"physics", "dynamic", "settled", "drop"}

    if simulation_state != sim.simulation_stopped and not (
        paused_allowed and simulation_state == sim.simulation_paused
    ):
        raise RuntimeError(
            "请先停止CoppeliaSim仿真，或让physics场景保持暂停。"
        )

    print(f"当前Scene ID：{scene_manifest['scene_id']}")
    clear_old_clusters()

    # ========================================================
    # A. 找Camera
    # ========================================================

    camera = (
        find_unique_object_by_alias(
            sim,
            sim.sceneobject_visionsensor,
            "rgbd_camera",
        )
    )

    gripper_tip = (
        find_unique_object_by_alias(
            sim,
            sim.sceneobject_dummy,
            "gripper_tip",
        )
    )

    joints = get_kuka_joints_from_tip(
        sim,
        gripper_tip,
    )

    if len(joints) != 7:
        raise RuntimeError(
            f"预期7个KUKA关节，"
            f"实际找到{len(joints)}个。"
        )

    robot_base = int(
        sim.getObjectParent(
            joints[0]
        )
    )

    print(
        "\nCamera：",
        get_full_path(
            sim,
            camera,
        )
    )

    print(
        "Robot Base：",
        get_full_path(
            sim,
            robot_base,
        )
    )

    # ========================================================
    # B. RGB-D → Base Point Cloud
    # ========================================================

    (
        rgb,
        depth,
        width,
        height,
    ) = capture_rgbd(
        sim,
        camera,
    )

    parameters = (
        get_camera_parameters(
            sim,
            camera,
            width,
            height,
        )
    )

    (
        points_camera,
        colors,
        valid_mask,
    ) = depth_to_camera_point_cloud(
        depth,
        rgb,
        parameters,
    )

    transform = np.asarray(
        sim.getObjectMatrix(
            camera,
            robot_base,
        ),
        dtype=np.float64,
    ).reshape(
        3,
        4,
    )

    points_base = transform_points(
        points_camera,
        transform,
    )

    raw_cloud = create_open3d_cloud(
        points_base,
        colors,
    )

    print_cloud_info(
        "RAW POINT CLOUD",
        raw_cloud,
    )

    # ========================================================
    # C. Voxel
    # ========================================================

    cloud = voxel_downsample(
        raw_cloud
    )

    # ========================================================
    # D. Outlier
    # ========================================================

    cloud = remove_outliers(
        cloud
    )

    # ========================================================
    # E. RANSAC TABLE
    # ========================================================

    (
        raw_plane_model,
        table_cloud,
        non_table_cloud,
    ) = detect_table_plane(
        cloud
    )

    # 统一法向方向
    plane_model = (
        orient_plane_toward_camera(
            sim,
            raw_plane_model,
            camera,
            robot_base,
        )
    )

    # ========================================================
    # F. 提取桌面上方物体
    # ========================================================

    (
        object_cloud,
        heights,
        object_mask,
    ) = extract_objects_above_table(
        cloud,
        plane_model,
    )

    if (
        len(object_cloud.points)
        < 100
    ):
        raise RuntimeError(
            "\n桌面上方工件点太少。\n"
            "可能原因：\n"
            "1. MAX_OBJECT_HEIGHT太小；\n"
            "2. RANSAC找错平面；\n"
            "3. 工件不在视野内。"
        )

    # ========================================================
    # G. DBSCAN
    # ========================================================

    min_cluster_points = (
        MIN_CLUSTER_POINTS_LEVEL4
        if scene_mode == "level4"
        else MIN_CLUSTER_POINTS
    )
    labels, initial_clusters = run_dbscan(
        object_cloud,
        expected_count=None,
        enable_layer_split=scene_mode in {"planned", "level4"},
        min_cluster_points=min_cluster_points,
    )
    clusters, geometric_diagnostics = refine_geometric_clusters(
        initial_clusters,
        min_cluster_points=min_cluster_points,
    )
    segmentation_method = "dbscan_normal_curvature_primitive"
    print(
        "Geometry-only refinement: "
        f"{len(initial_clusters)} spatial clusters -> {len(clusters)} instances"
    )

    if len(clusters) == 0:
        raise RuntimeError(
            "没有得到有效工件Cluster。"
        )

    # ========================================================
    # H. 分析
    # ========================================================

    results = analyze_clusters(
        clusters
    )

    # ========================================================
    # I. 保存
    # ========================================================

    if not o3d.io.write_point_cloud(
        str(VIEW_01_OBJECT_CLOUD_FILE),
        object_cloud,
    ):
        raise RuntimeError("Unable to save first-view object point cloud")

    save_clusters(
        clusters,
        results,
        scene_manifest,
        plane_model,
        segmentation_method=segmentation_method,
        view_metadata={
            "view_count": 1,
            "view_id": "view_01",
            "camera_matrix_base": transform.tolist(),
            "camera_intrinsics": {
                "fx": float(parameters["fx"]),
                "fy": float(parameters["fy"]),
                "cx": float(parameters["cx"]),
                "cy": float(parameters["cy"]),
                "K": np.asarray(parameters["K"], dtype=float).tolist(),
                "width": int(width),
                "height": int(height),
            },
            "object_cloud_file": VIEW_01_OBJECT_CLOUD_FILE.name,
        },
        geometric_diagnostics=geometric_diagnostics,
    )

    # ========================================================
    # J. 可视化
    # ========================================================

    colored_clusters = (
        colorize_clusters(
            clusters
        )
    )

    coordinate_frame = (
        o3d.geometry.TriangleMesh
        .create_coordinate_frame(
            size=0.10,
            origin=[
                0.0,
                0.0,
                0.0,
            ],
        )
    )

    print(
        "\n========================================"
    )

    print(
        "Open3D显示说明："
    )

    print(
        "灰白平面 = RANSAC桌面"
    )

    print(
        "不同颜色 = DBSCAN分出的不同工件"
    )

    print(
        "========================================"
    )

    geometries = [
        table_cloud,
        coordinate_frame,
    ]

    geometries.extend(
        colored_clusters
    )

    if SHOW_VISUALIZATION:
        o3d.visualization.draw_geometries(
            geometries,
            window_name=(
                "Multi-Object Point Cloud Segmentation"
            ),
            width=1200,
            height=800,
        )
    else:
        print("ROBOT_GRASP_HEADLESS=1，跳过Open3D可视化。")

    print(
        "\n========== 当前阶段完成 =========="
    )

    print(
        f"检测到 {len(clusters)} 个"
        "独立3D工件实例。"
    )


if __name__ == "__main__":
    main()
