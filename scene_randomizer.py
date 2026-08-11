from __future__ import annotations

import json
import math
import os
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from point_cloud import (
    find_unique_object_by_alias,
    get_camera_parameters,
    get_kuka_joints_from_tip,
    get_full_path,
)
from shape_catalog import class_from_alias, get_shape_spec


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ============================================================
# 1. 实验参数
# ============================================================

# 先用固定5个工件建立稳定基线。
# 多场景通过后，再依次提高到8和10。
MIN_OBJECTS = 5
MAX_OBJECTS = 5

# 第一次运行时，会根据你“当前已经验证成功的工件分布”
# 自动估计一个安全工作区中心，并写入REFERENCE_FILE。
#
# 后续运行直接复用这个中心，防止随机场景逐次漂移。
REFERENCE_FILE = Path("randomizer_reference.json")

# 每次随机场景的真实参数记录。
SCENE_GT_FILE = Path("random_scene_ground_truth.json")

# 相对于自动估计工作区中心的随机范围。
#
# 如果后续发现边缘工件出相机视野：
# 把这两个数稍微减小。
WORKSPACE_HALF_X = 0.110
WORKSPACE_HALF_Y = 0.090

# 工件之间额外保留的平面间隔。
SEPARATION_MARGIN_M = 0.008

# 每个工件最多尝试多少次随机位置。
MAX_PLACEMENT_TRIES = 3000

# A layout can fail because an early random placement blocks the remaining
# objects even when a valid arrangement exists. Retry the complete layout
# while keeping the already generated object dimensions and classes fixed.
MAX_LAYOUT_ATTEMPTS = 100

# 随机数种子。
#
# None：
#   每次真正随机。
#
# 例如 12345：
#   每次生成完全一样的场景，方便debug。
RANDOM_SEED: int | None = (
    int(os.environ["ROBOT_GRASP_RANDOM_SEED"])
    if os.environ.get("ROBOT_GRASP_RANDOM_SEED")
    else None
)

# Keep the validated separated baseline as the default. Set
# ROBOT_GRASP_SCENE_MODE=level4 to generate a mild contact/stacking scene.
SCENE_MODE = os.environ.get("ROBOT_GRASP_SCENE_MODE", "separated").lower()

# Level 4 deliberately starts with one two-layer stack. The upper object is
# smaller and offset, so part of the lower object remains visible.
LEVEL4_STACK_CLEARANCE_M = 0.001
LEVEL4_TOP_OFFSET_RATIO = (0.75, 0.90)
LEVEL4_INDEPENDENT_CLEARANCE_M = 0.014

# 每次生成新场景时清理这些阶段的旧结果，避免跨场景串数据。
SEGMENTATION_DIR = Path("segmentation_output")
RECOGNITION_DIR = Path("recognition_output")
EVALUATION_DIR = Path("evaluation_output")

# 当前相机只有320x240。提高到640x480后，小工件和局部遮挡区域
# 能保留更多有效点。设置失败时只警告，不中断场景生成。
TARGET_CAMERA_RESOLUTION = (640, 480)

# RG2左右钩爪会遮挡图像边缘，因此只把中央区域作为随机放置的
# 安全成像区域。比例形式可以自动适应不同分辨率。
SAFE_IMAGE_U_MIN_RATIO = 0.20
SAFE_IMAGE_U_MAX_RATIO = 0.80
SAFE_IMAGE_V_MIN_RATIO = 0.08
SAFE_IMAGE_V_MAX_RATIO = 0.92

CAMERA_NEAR_MARGIN_M = 0.010
CAMERA_FAR_MARGIN_M = 0.020


# ============================================================
# 2. 随机尺寸范围
#
# 注意：
# 这些尺寸只属于“场景生成器”。
#
# recognize_objects.py不会读取这里，
# 因此识别器依然不知道Ground Truth尺寸。
# ============================================================

# Cube边长：
# 30~60 mm
CUBE_SIDE_RANGE = (
    0.030,
    0.060,
)

# Cuboid：
# 长 55~90 mm
# 宽 28~48 mm
# 高 25~55 mm
CUBOID_LENGTH_RANGE = (
    0.055,
    0.090,
)

CUBOID_WIDTH_RANGE = (
    0.028,
    0.048,
)

CUBOID_HEIGHT_RANGE = (
    0.025,
    0.055,
)

# Cylinder：
# 直径 30~58 mm
# 高度 35~75 mm
CYLINDER_DIAMETER_RANGE = (
    0.030,
    0.058,
)

CYLINDER_HEIGHT_RANGE = (
    0.035,
    0.075,
)


# ============================================================
# 3. 颜色
#
# 颜色只用于视觉效果。
# 当前识别模块不依赖颜色。
# ============================================================

COLOR_PALETTE = [
    [0.85, 0.25, 0.25],
    [0.25, 0.70, 0.30],
    [0.25, 0.40, 0.90],
    [0.90, 0.75, 0.20],
    [0.75, 0.30, 0.80],
    [0.20, 0.75, 0.75],
    [0.90, 0.50, 0.20],
    [0.55, 0.55, 0.85],
]


# ============================================================
# 4. 工件Alias规则
# ============================================================

TEST_OBJECT_PATTERN = re.compile(
    r"^(?:"
    r"target_cube"
    r"|cube(?:_\d+)?"
    r"|cuboid(?:_\d+)?"
    r"|cylinder(?:_\d+)?"
    r"|sphere(?:_\d+)?"
    r"|spheroid(?:_\d+)?"
    r"|cone(?:_\d+)?"
    r"|rand_cube_\d+"
    r"|rand_cuboid_\d+"
    r"|rand_cylinder_\d+"
    r"|rand_sphere_\d+"
    r"|rand_spheroid_\d+"
    r"|rand_cone_\d+"
    r")$",
    re.IGNORECASE,
)


def infer_object_class(alias: str) -> str | None:
    name = alias.lower()

    if not TEST_OBJECT_PATTERN.match(name):
        return None
    try:
        return class_from_alias(name)
    except ValueError:
        return None


# ============================================================
# 5. CoppeliaSim / 坐标工具
# ============================================================

def matrix12_to_numpy(matrix: list[float]) -> np.ndarray:
    return np.asarray(
        matrix,
        dtype=np.float64,
    ).reshape(
        3,
        4,
    )


def clear_previous_pipeline_outputs() -> None:
    """Remove only generated pipeline files from the previous scene."""

    targets = [
        (SEGMENTATION_DIR, "cluster_*.ply"),
        (SEGMENTATION_DIR, "segmentation_metadata.json"),
        (RECOGNITION_DIR, "recognition_results.json"),
        (EVALUATION_DIR, "evaluation_results.csv"),
        (EVALUATION_DIR, "evaluation_summary.json"),
    ]
    contact_file = Path("random_scene_contacts.json")
    if contact_file.exists():
        contact_file.unlink()

    removed = 0
    for directory, pattern in targets:
        if not directory.exists():
            continue
        for path in directory.glob(pattern):
            if path.is_file():
                path.unlink()
                removed += 1

    print(f"已清理上一场景的流水线文件：{removed} 个")


def configure_camera_resolution(sim: Any, camera: int) -> tuple[int, int]:
    """Try to use the high-resolution RGB-D baseline and return actual size."""

    target_width, target_height = TARGET_CAMERA_RESOLUTION
    try:
        sim.setObjectInt32Param(
            camera,
            sim.visionintparam_resolution_x,
            int(target_width),
        )
        sim.setObjectInt32Param(
            camera,
            sim.visionintparam_resolution_y,
            int(target_height),
        )
    except Exception as exc:
        print(f"警告：无法设置Vision Sensor分辨率：{exc}")

    width = int(
        sim.getObjectInt32Param(
            camera,
            sim.visionintparam_resolution_x,
        )
    )
    height = int(
        sim.getObjectInt32Param(
            camera,
            sim.visionintparam_resolution_y,
        )
    )

    print(f"RGB-D分辨率：{width} x {height}")
    return width, height


def build_camera_model(
    sim: Any,
    camera: int,
    robot_base: int,
) -> dict[str, Any]:
    """Build the projection model used to reject unsafe random placements."""

    width, height = configure_camera_resolution(sim, camera)
    params = get_camera_parameters(sim, camera, width, height)
    base_camera = np.asarray(
        sim.getObjectMatrix(camera, robot_base),
        dtype=np.float64,
    ).reshape(3, 4)

    safe_bounds = {
        "u_min": (width - 1) * SAFE_IMAGE_U_MIN_RATIO,
        "u_max": (width - 1) * SAFE_IMAGE_U_MAX_RATIO,
        "v_min": (height - 1) * SAFE_IMAGE_V_MIN_RATIO,
        "v_max": (height - 1) * SAFE_IMAGE_V_MAX_RATIO,
    }

    return {
        "width": width,
        "height": height,
        "fov_x": float(params["fov_x"]),
        "fov_y": float(params["fov_y"]),
        "near": float(params["near"]),
        "far": float(params["far"]),
        "base_camera": base_camera,
        "safe_bounds": safe_bounds,
    }


def bbox_corners_in_base(
    x: float,
    y: float,
    z: float,
    size_xyz: tuple[float, float, float],
    yaw: float,
) -> np.ndarray:
    """Return the eight candidate object-box corners in robot-base frame."""

    sx, sy, sz = size_xyz
    local = np.asarray(
        [
            [ix * sx / 2.0, iy * sy / 2.0, iz * sz / 2.0]
            for ix in (-1.0, 1.0)
            for iy in (-1.0, 1.0)
            for iz in (-1.0, 1.0)
        ],
        dtype=np.float64,
    )

    c = math.cos(yaw)
    s = math.sin(yaw)
    rotation_z = np.asarray(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    return local @ rotation_z.T + np.asarray([x, y, z])


def project_base_points_to_image(
    points_base: np.ndarray,
    camera_model: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Project robot-base points using CoppeliaSim's camera convention."""

    transform = camera_model["base_camera"]
    rotation = transform[:, :3]
    translation = transform[:, 3]
    points_camera = (points_base - translation) @ rotation

    z = points_camera[:, 2]
    if np.any(z <= camera_model["near"] + CAMERA_NEAR_MARGIN_M):
        return None
    if np.any(z >= camera_model["far"] - CAMERA_FAR_MARGIN_M):
        return None

    tan_x = math.tan(camera_model["fov_x"] / 2.0)
    tan_y = math.tan(camera_model["fov_y"] / 2.0)
    u = (
        1.0 - points_camera[:, 0] / (z * tan_x)
    ) * 0.5 * (camera_model["width"] - 1)
    v = (
        1.0 - points_camera[:, 1] / (z * tan_y)
    ) * 0.5 * (camera_model["height"] - 1)

    return points_camera, np.column_stack([u, v])


def check_candidate_camera_visibility(
    x: float,
    y: float,
    z: float,
    size_xyz: tuple[float, float, float],
    yaw: float,
    camera_model: dict[str, Any],
) -> dict[str, Any] | None:
    """Reject candidates outside the usable view or inside the jaw mask."""

    corners = bbox_corners_in_base(x, y, z, size_xyz, yaw)
    projected = project_base_points_to_image(corners, camera_model)
    if projected is None:
        return None

    points_camera, pixels = projected
    bounds = camera_model["safe_bounds"]
    if (
        float(pixels[:, 0].min()) < bounds["u_min"]
        or float(pixels[:, 0].max()) > bounds["u_max"]
        or float(pixels[:, 1].min()) < bounds["v_min"]
        or float(pixels[:, 1].max()) > bounds["v_max"]
    ):
        return None

    return {
        "image_bbox_px": [
            float(pixels[:, 0].min()),
            float(pixels[:, 1].min()),
            float(pixels[:, 0].max()),
            float(pixels[:, 1].max()),
        ],
        "camera_depth_range_m": [
            float(points_camera[:, 2].min()),
            float(points_camera[:, 2].max()),
        ],
    }


def find_robot_base(sim: Any) -> int:
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
            f"预期从gripper_tip回溯得到7个KUKA关节，"
            f"实际得到{len(joints)}个。"
        )

    robot_base = int(
        sim.getObjectParent(
            joints[0]
        )
    )

    if robot_base == -1:
        raise RuntimeError(
            "无法确定KUKA Robot Base。"
        )

    return robot_base


def get_bbox_center_and_size_in_base(
    sim: Any,
    shape: int,
    robot_base: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    返回：
        bbox center in Robot Base
        bbox size
    """

    size, bbox_pose = (
        sim.getShapeBB(shape)
    )

    object_matrix = (
        sim.getObjectMatrix(
            shape,
            robot_base,
        )
    )

    bbox_local_matrix = (
        sim.poseToMatrix(
            bbox_pose
        )
    )

    bbox_matrix = (
        sim.multiplyMatrices(
            object_matrix,
            bbox_local_matrix,
        )
    )

    matrix_np = matrix12_to_numpy(
        bbox_matrix
    )

    return (
        matrix_np[:, 3].copy(),
        np.asarray(
            size,
            dtype=np.float64,
        ),
    )


# ============================================================
# 6. 查找当前测试工件
# ============================================================

def find_existing_test_objects(
    sim: Any,
) -> list[dict]:
    shapes = sim.getObjectsInTree(
        sim.handle_scene,
        sim.sceneobject_shape,
        0,
    )

    result = []

    for shape in shapes:
        alias = str(
            sim.getObjectAlias(shape)
        )

        object_class = (
            infer_object_class(
                alias
            )
        )

        if object_class is None:
            continue

        result.append(
            {
                "handle": int(shape),
                "alias": alias,
                "class": object_class,
            }
        )

    return result


# ============================================================
# 7. 建立/读取随机化参考坐标
# ============================================================

def create_reference_from_current_scene(
    sim: Any,
    robot_base: int,
    objects: list[dict],
) -> dict:
    """
    利用当前已经验证成功的工件场景估计：
        工作区中心X/Y
        桌面高度Z

    因此不需要你手动输入机器人的具体坐标。
    """

    if len(objects) < 1:
        raise RuntimeError(
            "\n第一次运行scene_randomizer.py时，"
            "场景里至少需要保留一个当前已经验证成功的"
            "cube/cuboid/cylinder工件。\n"
            "程序会利用它推断桌面高度和安全工作区中心。"
        )

    centers = []
    bottom_heights = []

    print(
        "\n正在根据当前场景建立随机化参考..."
    )

    for item in objects:
        center, size = (
            get_bbox_center_and_size_in_base(
                sim,
                item["handle"],
                robot_base,
            )
        )

        centers.append(
            center
        )

        # 当前阶段工件均基本直立。
        bottom_z = (
            center[2]
            - size[2] / 2.0
        )

        bottom_heights.append(
            bottom_z
        )

        print(
            f"{item['alias']:18s} "
            f"P={np.round(center, 4)} "
            f"BB={np.round(size * 1000, 1)} mm"
        )

    centers_np = np.vstack(
        centers
    )

    workspace_center_x = float(
        np.median(
            centers_np[:, 0]
        )
    )

    workspace_center_y = float(
        np.median(
            centers_np[:, 1]
        )
    )

    table_z = float(
        np.median(
            np.asarray(
                bottom_heights,
                dtype=np.float64,
            )
        )
    )

    reference = {
        "workspace_center_x": (
            workspace_center_x
        ),
        "workspace_center_y": (
            workspace_center_y
        ),
        "table_z": table_z,
        "workspace_half_x": (
            WORKSPACE_HALF_X
        ),
        "workspace_half_y": (
            WORKSPACE_HALF_Y
        ),
    }

    with open(
        REFERENCE_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            reference,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "\n已建立随机化参考："
    )

    print(
        f"Workspace center = "
        f"({workspace_center_x:.4f}, "
        f"{workspace_center_y:.4f}) m"
    )

    print(
        f"Table Z = {table_z:.5f} m"
    )

    print(
        f"已保存：{REFERENCE_FILE.resolve()}"
    )

    return reference


def load_or_create_reference(
    sim: Any,
    robot_base: int,
    existing_objects: list[dict],
) -> dict:
    if REFERENCE_FILE.exists():
        with open(
            REFERENCE_FILE,
            "r",
            encoding="utf-8",
        ) as f:
            reference = json.load(f)

        # 旧reference可能保存过更大的随机范围。每次运行都使用当前
        # 配置，避免仅修改常量却仍沿用旧的0.145/0.120 m范围。
        reference["workspace_half_x"] = WORKSPACE_HALF_X
        reference["workspace_half_y"] = WORKSPACE_HALF_Y

        print(
            "\n使用已有随机化参考："
            f"{REFERENCE_FILE.resolve()}"
        )

        print(
            "Workspace center = "
            f"({reference['workspace_center_x']:.4f}, "
            f"{reference['workspace_center_y']:.4f}) m"
        )

        print(
            f"Table Z = "
            f"{reference['table_z']:.5f} m"
        )

        return reference

    return create_reference_from_current_scene(
        sim,
        robot_base,
        existing_objects,
    )


# ============================================================
# 8. 删除旧测试工件
# ============================================================

def remove_existing_test_objects(
    sim: Any,
    objects: list[dict],
) -> None:
    if not objects:
        return

    print(
        "\n正在删除旧测试工件："
    )

    handles = []

    for item in objects:
        print(
            f"  {item['alias']}"
        )

        handles.append(
            item["handle"]
        )

    sim.removeObjects(
        handles
    )


# ============================================================
# 9. 随机生成尺寸
# ============================================================

def random_dimensions(
    object_class: str,
) -> tuple[float, float, float]:
    if object_class == "cube":
        side = random.uniform(
            *CUBE_SIDE_RANGE
        )

        return (
            side,
            side,
            side,
        )

    if object_class == "cuboid":
        length = random.uniform(
            *CUBOID_LENGTH_RANGE
        )

        width = random.uniform(
            *CUBOID_WIDTH_RANGE
        )

        height = random.uniform(
            *CUBOID_HEIGHT_RANGE
        )

        # 确保footprint确实具有明显长宽差异。
        if length / width < 1.40:
            length = min(
                CUBOID_LENGTH_RANGE[1],
                width * 1.45,
            )

        return (
            length,
            width,
            height,
        )

    if object_class == "cylinder":
        diameter = random.uniform(
            *CYLINDER_DIAMETER_RANGE
        )

        height = random.uniform(
            *CYLINDER_HEIGHT_RANGE
        )

        return (
            diameter,
            diameter,
            height,
        )

    raise ValueError(
        f"未知类别：{object_class}"
    )


# ============================================================
# 10. 非重叠随机放置
# ============================================================

def footprint_radius(
    size_xyz: tuple[float, float, float],
) -> float:
    """
    用XY footprint的外接圆作为保守碰撞检查。

    好处：
    即使Cuboid发生任意yaw旋转，
    也能保证平面上不会与其他工件重叠。
    """

    sx, sy, _ = size_xyz

    return (
        0.5
        * math.sqrt(
            sx * sx
            + sy * sy
        )
    )


def footprint_half_extents(
    size_xyz: tuple[float, float, float],
) -> np.ndarray:
    """Return the XY half extents used by the placement collision test."""

    return np.asarray(
        [size_xyz[0] / 2.0, size_xyz[1] / 2.0],
        dtype=np.float64,
    )


def footprint_axes(yaw: float) -> np.ndarray:
    """Return the two unit axes of an XY oriented bounding rectangle."""

    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.asarray(
        [[c, s], [-s, c]],
        dtype=np.float64,
    )


def oriented_footprints_overlap(
    center_a: np.ndarray,
    size_a: tuple[float, float, float],
    yaw_a: float,
    center_b: np.ndarray,
    size_b: tuple[float, float, float],
    yaw_b: float,
    clearance_m: float | None = None,
) -> bool:
    """Conservative 2-D SAT test with the configured clearance margin."""

    clearance = (
        SEPARATION_MARGIN_M
        if clearance_m is None
        else float(clearance_m)
    )
    axes_a = footprint_axes(yaw_a)
    axes_b = footprint_axes(yaw_b)
    half_a = footprint_half_extents(size_a) + clearance / 2.0
    half_b = footprint_half_extents(size_b) + clearance / 2.0
    center_delta = center_b - center_a

    for axis in np.vstack([axes_a, axes_b]):
        distance = abs(float(np.dot(center_delta, axis)))
        radius_a = float(
            half_a[0] * abs(np.dot(axes_a[0], axis))
            + half_a[1] * abs(np.dot(axes_a[1], axis))
        )
        radius_b = float(
            half_b[0] * abs(np.dot(axes_b[0], axis))
            + half_b[1] * abs(np.dot(axes_b[1], axis))
        )
        if distance >= radius_a + radius_b:
            return False

    return True


def sample_non_overlapping_xy(
    reference: dict,
    placed_objects: list[dict],
    size_xyz: tuple[float, float, float],
    yaw: float,
    table_z: float,
    camera_model: dict[str, Any],
    clearance_m: float | None = None,
) -> tuple[float, float, dict[str, Any]]:
    center_x = float(
        reference[
            "workspace_center_x"
        ]
    )

    center_y = float(
        reference[
            "workspace_center_y"
        ]
    )

    half_x = float(
        reference.get(
            "workspace_half_x",
            WORKSPACE_HALF_X,
        )
    )

    half_y = float(
        reference.get(
            "workspace_half_y",
            WORKSPACE_HALF_Y,
        )
    )

    sx, sy, _ = size_xyz
    c = abs(math.cos(yaw))
    s = abs(math.sin(yaw))
    extent_x = c * sx / 2.0 + s * sy / 2.0
    extent_y = s * sx / 2.0 + c * sy / 2.0

    x_min = (
        center_x - half_x + extent_x
    )

    x_max = (
        center_x + half_x - extent_x
    )

    y_min = (
        center_y - half_y + extent_y
    )

    y_max = (
        center_y + half_y - extent_y
    )

    if (
        x_min >= x_max
        or y_min >= y_max
    ):
        raise RuntimeError(
            "随机工作区过小，无法容纳当前工件。"
        )

    overlap_rejections = 0
    visibility_rejections = 0

    for _ in range(
        MAX_PLACEMENT_TRIES
    ):
        x = random.uniform(
            x_min,
            x_max,
        )

        y = random.uniform(
            y_min,
            y_max,
        )

        okay = True

        for other in placed_objects:
            if oriented_footprints_overlap(
                np.asarray([x, y], dtype=np.float64),
                size_xyz,
                yaw,
                np.asarray(other["position"], dtype=np.float64),
                other["size_xyz"],
                other["yaw"],
                clearance_m=clearance_m,
            ):
                okay = False
                overlap_rejections += 1
                break

        if okay:
            visibility = check_candidate_camera_visibility(
                x,
                y,
                table_z + size_xyz[2] / 2.0,
                size_xyz,
                yaw,
                camera_model,
            )
            if visibility is not None:
                return x, y, visibility
            visibility_rejections += 1

    raise RuntimeError(
        "\n在当前工作区内无法找到足够的非重叠位置。\n"
        "解决方法任选一个：\n"
        "1. 把MAX_OBJECTS暂时改小；\n"
        "2. 减小SEPARATION_MARGIN_M；\n"
        "3. 调整WORKSPACE_HALF_X/Y或安全图像区域；\n"
        "4. 检查相机是否被RG2钩爪大面积遮挡。\n"
        f"本次拒绝统计：重叠={overlap_rejections}，"
        f"视场={visibility_rejections}。"
    )


# ============================================================
# 11. 创建Primitive Shape
# ============================================================

def create_object(
    sim: Any,
    object_class: str,
    size_xyz: tuple[float, float, float],
) -> int:
    primitive_name = get_shape_spec(object_class).primitive_name
    parameter_name = {
        "cuboid": "primitiveshape_cuboid",
        "cylinder": "primitiveshape_cylinder",
        "spheroid": "primitiveshape_spheroid",
        "cone": "primitiveshape_cone",
    }[primitive_name]
    primitive_type = getattr(sim, parameter_name)

    handle = int(
        sim.createPrimitiveShape(
            primitive_type,
            list(size_xyz),
            0,
        )
    )

    # 当前视觉实验全部保持静态。
    # 即使误点Play，这些测试物体也不会自由掉落。
    try:
        sim.setObjectInt32Param(
            handle,
            sim.shapeintparam_static,
            1,
        )
    except Exception:
        # 新版本虽然更推荐properties API，
        # 但如果该兼容接口不可用，也不影响STOP状态下视觉实验。
        pass

    return handle


# ============================================================
# 12. 设置颜色
# ============================================================

def set_shape_color(
    sim: Any,
    handle: int,
    color: list[float],
) -> None:
    try:
        sim.setShapeColor(
            handle,
            "",
            sim.colorcomponent_ambient_diffuse,
            color,
        )
    except Exception:
        # 颜色不影响几何识别，
        # 所以失败也可以继续。
        pass


# ============================================================
# 13. 创建一整个随机场景
# ============================================================

def generate_random_scene(
    sim: Any,
    robot_base: int,
    reference: dict,
    camera_model: dict[str, Any],
) -> list[dict]:
    if SCENE_MODE in {"physics", "dynamic", "settled", "drop"}:
        # Import lazily to keep the validated static baseline independent from
        # the dynamics-only helpers and avoid a module import cycle.
        from physics_scene_randomizer import generate_physics_scene

        return generate_physics_scene(
            sim,
            robot_base,
            reference,
            camera_model,
        )

    if SCENE_MODE in {"planned", "constraint", "planned_contact"}:
        from planned_scene_randomizer import generate_planned_scene

        return generate_planned_scene(
            sim,
            robot_base,
            reference,
            camera_model,
        )

    if SCENE_MODE in {"level4", "stack", "stacked", "contact", "occlusion"}:
        return generate_level4_scene(
            sim,
            robot_base,
            reference,
            camera_model,
        )

    return _generate_separated_scene(
        sim,
        robot_base,
        reference,
        camera_model,
    )


def _generate_separated_scene(
    sim: Any,
    robot_base: int,
    reference: dict,
    camera_model: dict[str, Any],
) -> list[dict]:
    object_count = random.randint(
        MIN_OBJECTS,
        MAX_OBJECTS,
    )

    # 至少保证三类各出现一次。
    classes = [
        "cube",
        "cuboid",
        "cylinder",
    ]

    while len(classes) < object_count:
        classes.append(
            random.choice(
                [
                    "cube",
                    "cuboid",
                    "cylinder",
                ]
            )
        )

    random.shuffle(
        classes
    )

    # 先生成全部尺寸并按占地半径从大到小规划，避免随机顺序
    # 先放小物体导致后续大物体无法找到位置。
    specs = []
    for object_class in classes:
        size_xyz = random_dimensions(object_class)
        specs.append(
            {
                "class": object_class,
                "size_xyz": size_xyz,
                "radius": footprint_radius(size_xyz),
                "yaw": random.uniform(-math.pi, math.pi),
            }
        )
    specs.sort(key=lambda item: item["radius"], reverse=True)

    table_z = float(
        reference["table_z"]
    )

    class_counters = {
        "cube": 0,
        "cuboid": 0,
        "cylinder": 0,
    }

    print(
        "\n"
        "============================================================"
    )

    print(
        f"正在生成随机场景：{object_count} 个工件"
    )

    print(
        "============================================================"
    )

    planned_objects: list[dict] | None = None
    for layout_attempt in range(1, MAX_LAYOUT_ATTEMPTS + 1):
        placed_objects: list[dict] = []
        candidate_layout: list[dict] = []

        try:
            for spec in specs:
                object_class = spec["class"]
                size_xyz = spec["size_xyz"]
                radius = spec["radius"]
                yaw = spec["yaw"]
                _, _, sz = size_xyz
                z = table_z + sz / 2.0

                x, y, visibility = sample_non_overlapping_xy(
                    reference,
                    placed_objects,
                    size_xyz,
                    yaw,
                    table_z,
                    camera_model,
                )

                candidate_layout.append(
                    {
                        "class": object_class,
                        "size_xyz": size_xyz,
                        "radius": radius,
                        "yaw": yaw,
                        "x": x,
                        "y": y,
                        "z": z,
                        "visibility": visibility,
                    }
                )
                placed_objects.append(
                    {
                        "position": [x, y],
                        "size_xyz": size_xyz,
                        "yaw": yaw,
                    }
                )

            planned_objects = candidate_layout
            if layout_attempt > 1:
                print(f"布局随机重试成功：第 {layout_attempt} 次")
            break
        except RuntimeError:
            if layout_attempt == MAX_LAYOUT_ATTEMPTS:
                raise

    if planned_objects is None:
        raise RuntimeError("无法规划随机工件布局。")

    # 只有全部位置规划成功后才创建场景对象，避免失败时留下半个场景。
    created_handles: list[int] = []
    result_objects: list[dict] = []
    try:
        for scene_index, plan in enumerate(planned_objects):
            object_class = plan["class"]
            size_xyz = plan["size_xyz"]
            radius = plan["radius"]
            yaw = plan["yaw"]
            x = plan["x"]
            y = plan["y"]
            z = plan["z"]
            visibility = plan["visibility"]
            sx, sy, sz = size_xyz

            # CoppeliaSim quaternion: [qx, qy, qz, qw]
            quaternion = [
                0.0,
                0.0,
                math.sin(yaw / 2.0),
                math.cos(yaw / 2.0),
            ]

            handle = create_object(
                sim,
                object_class,
                size_xyz,
            )
            created_handles.append(handle)

            class_counters[object_class] += 1

            alias = (
                f"rand_{object_class}_"
                f"{class_counters[object_class]:02d}"
            )

            sim.setObjectAlias(handle, alias)

            pose = [x, y, z, *quaternion]

            sim.setObjectPose(handle, pose, robot_base)

            color = COLOR_PALETTE[scene_index % len(COLOR_PALETTE)]
            set_shape_color(sim, handle, color)

            object_record = {
                "handle": handle,
                "alias": alias,
                "class": object_class,
                "size_m": [float(sx), float(sy), float(sz)],
                "position": [float(x), float(y), float(z)],
                "yaw_deg": float(math.degrees(yaw)),
                "footprint_radius": float(radius),
                "image_bbox_px": visibility["image_bbox_px"],
                "camera_depth_range_m": visibility[
                    "camera_depth_range_m"
                ],
            }
            result_objects.append(object_record)

            print(
                f"{alias:20s} | {object_class:8s} | "
                f"size=({sx*1000:5.1f}, {sy*1000:5.1f}, "
                f"{sz*1000:5.1f}) mm | "
                f"P=({x:.3f}, {y:.3f}, {z:.3f}) | "
                f"yaw={math.degrees(yaw):7.2f}°"
            )
    except Exception:
        if created_handles:
            sim.removeObjects(created_handles)
        raise

    return result_objects


def _level4_dimensions(object_class: str, role: str) -> tuple[float, float, float]:
    """Generate conservative dimensions for a stable, partially visible stack."""

    if role == "stack_base":
        if object_class == "cube":
            side = random.uniform(0.052, 0.060)
            return side, side, side
        return (
            random.uniform(0.080, 0.090),
            random.uniform(0.046, 0.048),
            random.uniform(0.035, 0.050),
        )

    if role == "stack_top":
        if object_class == "cube":
            side = random.uniform(0.030, 0.034)
            return side, side, side
        diameter = random.uniform(0.030, 0.034)
        return diameter, diameter, random.uniform(0.038, 0.052)

    return random_dimensions(object_class)


def generate_level4_scene(
    sim: Any,
    robot_base: int,
    reference: dict,
    camera_model: dict[str, Any],
) -> list[dict]:
    """Create a mild single-view contact/occlusion/two-layer scene."""

    table_z = float(reference["table_z"])
    base_class = random.choice(["cube", "cuboid"])
    top_class = random.choice(["cube", "cylinder"])
    specs = [
        {
            "plan_id": "stack_base",
            "role": "stack_base",
            "class": base_class,
            "size_xyz": _level4_dimensions(base_class, "stack_base"),
            "yaw": random.uniform(-math.pi, math.pi),
        },
        {
            "plan_id": "stack_top",
            "role": "stack_top",
            "class": top_class,
            "size_xyz": _level4_dimensions(top_class, "stack_top"),
            "yaw": random.uniform(-math.pi, math.pi),
        },
    ]

    for index, object_class in enumerate(["cube", "cuboid", "cylinder"]):
        specs.append(
            {
                "plan_id": f"single_{index}",
                "role": "single",
                "class": object_class,
                "size_xyz": _level4_dimensions(object_class, "single"),
                "yaw": random.uniform(-math.pi, math.pi),
            }
        )

    print("\n============================================================")
    print("正在生成Level 4单视角轻度堆叠场景：5 个工件")
    print("============================================================")

    planned_objects: list[dict] | None = None
    for layout_attempt in range(1, MAX_LAYOUT_ATTEMPTS + 1):
        base_spec = specs[0]
        top_spec = specs[1]
        placed_objects: list[dict] = []

        # Position-only retries can remain impossible for one unlucky set of
        # large dimensions and fixed orientations.  Refresh orientations on
        # every attempt and dimensions periodically while keeping the class
        # composition and random seed reproducible.
        base_spec["yaw"] = random.uniform(-math.pi, math.pi)
        for spec in specs[2:]:
            spec["yaw"] = random.uniform(-math.pi, math.pi)
        if layout_attempt > 1 and (layout_attempt - 1) % 20 == 0:
            base_spec["size_xyz"] = _level4_dimensions(
                str(base_spec["class"]),
                "stack_base",
            )
            top_spec["size_xyz"] = _level4_dimensions(
                str(top_spec["class"]),
                "stack_top",
            )
            for spec in specs[2:]:
                spec["size_xyz"] = _level4_dimensions(
                    str(spec["class"]),
                    "single",
                )
        top_spec["yaw"] = float(base_spec["yaw"]) + random.uniform(
            -math.pi / 6.0,
            math.pi / 6.0,
        )

        try:
            base_size = base_spec["size_xyz"]
            base_yaw = float(base_spec["yaw"])
            base_x, base_y, base_visibility = sample_non_overlapping_xy(
                reference,
                placed_objects,
                base_size,
                base_yaw,
                table_z,
                camera_model,
            )
            base_z = table_z + base_size[2] / 2.0
            base_plan = {
                **base_spec,
                "x": base_x,
                "y": base_y,
                "z": base_z,
                "visibility": base_visibility,
                "stack_level": 0,
                "contact_group": 1,
                "support_plan_id": None,
                "expected_partial_occlusion": True,
            }
            placed_objects.append(
                {
                    "position": [base_x, base_y],
                    "size_xyz": base_size,
                    "yaw": base_yaw,
                }
            )

            single_plans: list[dict] = []
            for spec in specs[2:]:
                size_xyz = spec["size_xyz"]
                yaw = float(spec["yaw"])
                x, y, visibility = sample_non_overlapping_xy(
                    reference,
                    placed_objects,
                    size_xyz,
                    yaw,
                    table_z,
                    camera_model,
                    clearance_m=LEVEL4_INDEPENDENT_CLEARANCE_M,
                )
                single_plans.append(
                    {
                        **spec,
                        "x": x,
                        "y": y,
                        "z": table_z + size_xyz[2] / 2.0,
                        "visibility": visibility,
                        "stack_level": 0,
                        "contact_group": None,
                        "support_plan_id": None,
                        "expected_partial_occlusion": False,
                    }
                )
                placed_objects.append(
                    {
                        "position": [x, y],
                        "size_xyz": size_xyz,
                        "yaw": yaw,
                    }
                )

            top_size = top_spec["size_xyz"]
            top_yaw = float(top_spec["yaw"])
            base_axes = footprint_axes(base_yaw)
            relative_yaw = top_yaw - base_yaw
            c = abs(math.cos(relative_yaw))
            s = abs(math.sin(relative_yaw))
            top_extent_u = c * top_size[0] / 2.0 + s * top_size[1] / 2.0
            top_extent_v = s * top_size[0] / 2.0 + c * top_size[1] / 2.0
            available_u = base_size[0] / 2.0 - top_extent_u - 0.002
            available_v = base_size[1] / 2.0 - top_extent_v - 0.002
            if available_u <= 0.0 or available_v <= 0.0:
                raise RuntimeError("上层工件无法稳定放入底层工件顶面。")

            offset_ratio = random.uniform(*LEVEL4_TOP_OFFSET_RATIO)
            offset_u = random.choice([-1.0, 1.0]) * available_u * offset_ratio
            offset_v = random.choice([-1.0, 1.0]) * available_v * offset_ratio
            top_xy = (
                np.asarray([base_x, base_y], dtype=np.float64)
                + offset_u * base_axes[0]
                + offset_v * base_axes[1]
            )
            top_z = table_z + base_size[2] + top_size[2] / 2.0
            top_visibility = check_candidate_camera_visibility(
                float(top_xy[0]),
                float(top_xy[1]),
                top_z,
                top_size,
                top_yaw,
                camera_model,
            )
            if top_visibility is None:
                raise RuntimeError("上层工件不在安全成像区域内。")

            top_plan = {
                **top_spec,
                "x": float(top_xy[0]),
                "y": float(top_xy[1]),
                "z": top_z,
                "visibility": top_visibility,
                "stack_level": 1,
                "contact_group": 1,
                "support_plan_id": "stack_base",
                "expected_partial_occlusion": False,
            }
            planned_objects = [base_plan, top_plan, *single_plans]
            if layout_attempt > 1:
                print(f"Level 4布局随机重试成功：第 {layout_attempt} 次")
            break
        except RuntimeError:
            if layout_attempt == MAX_LAYOUT_ATTEMPTS:
                raise

    if planned_objects is None:
        raise RuntimeError("无法规划Level 4轻度堆叠场景。")

    class_counters = {"cube": 0, "cuboid": 0, "cylinder": 0}
    alias_by_plan_id: dict[str, str] = {}
    created_handles: list[int] = []
    result_objects: list[dict] = []

    try:
        for scene_index, plan in enumerate(planned_objects):
            object_class = str(plan["class"])
            size_xyz = plan["size_xyz"]
            yaw = float(plan["yaw"])
            sx, sy, sz = size_xyz
            handle = create_object(sim, object_class, size_xyz)
            created_handles.append(handle)
            class_counters[object_class] += 1
            alias = f"rand_{object_class}_{class_counters[object_class]:02d}"
            sim.setObjectAlias(handle, alias)
            sim.setObjectPose(
                handle,
                [
                    plan["x"],
                    plan["y"],
                    plan["z"],
                    0.0,
                    0.0,
                    math.sin(yaw / 2.0),
                    math.cos(yaw / 2.0),
                ],
                robot_base,
            )
            set_shape_color(
                sim,
                handle,
                COLOR_PALETTE[scene_index % len(COLOR_PALETTE)],
            )
            alias_by_plan_id[str(plan["plan_id"])] = alias
            support_alias = (
                None
                if plan["support_plan_id"] is None
                else alias_by_plan_id[str(plan["support_plan_id"])]
            )
            visibility = plan["visibility"]
            record = {
                "handle": handle,
                "alias": alias,
                "class": object_class,
                "size_m": [float(sx), float(sy), float(sz)],
                "position": [float(plan["x"]), float(plan["y"]), float(plan["z"])],
                "yaw_deg": float(math.degrees(yaw)),
                "footprint_radius": float(footprint_radius(size_xyz)),
                "image_bbox_px": visibility["image_bbox_px"],
                "camera_depth_range_m": visibility["camera_depth_range_m"],
                "scenario_role": plan["role"],
                "stack_level": int(plan["stack_level"]),
                "support_alias": support_alias,
                "contact_group": plan["contact_group"],
                "expected_partial_occlusion": bool(
                    plan["expected_partial_occlusion"]
                ),
            }
            result_objects.append(record)
            print(
                f"{alias:20s} | {object_class:8s} | role={plan['role']:10s} | "
                f"P=({plan['x']:.3f}, {plan['y']:.3f}, {plan['z']:.3f}) | "
                f"size=({sx*1000:.1f}, {sy*1000:.1f}, {sz*1000:.1f}) mm"
            )
    except Exception:
        if created_handles:
            sim.removeObjects(created_handles)
        raise

    return result_objects


# ============================================================
# 14. 保存场景Ground Truth
# ============================================================

def save_scene_ground_truth(
    objects: list[dict],
    reference: dict,
    scene_id: str,
    random_seed: int,
    camera_model: dict[str, Any],
) -> None:
    serializable = {
        "scene_id": scene_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": int(random_seed),
        "scene_mode": SCENE_MODE,
        "object_count": len(objects),
        "reference": reference,
        "camera": {
            "width": int(camera_model["width"]),
            "height": int(camera_model["height"]),
            "fov_x_deg": math.degrees(camera_model["fov_x"]),
            "fov_y_deg": math.degrees(camera_model["fov_y"]),
            "near_m": float(camera_model["near"]),
            "far_m": float(camera_model["far"]),
            "safe_image_bounds_px": camera_model["safe_bounds"],
        },
        "objects": [],
    }

    contact_file = Path("random_scene_contacts.json")
    if contact_file.exists():
        serializable["contact_file"] = str(contact_file)

    layout_types = sorted(
        {
            str(item.get("planned_layout"))
            for item in objects
            if item.get("planned_layout")
        }
    )
    if layout_types:
        serializable["planned_layout_types"] = layout_types
        serializable["requires_contact_instance_split"] = any(
            str(item.get("scenario_role", "table_only")) != "table_only"
            for item in objects
        )

    for item in objects:
        serializable[
            "objects"
        ].append(
            {
                key: value
                for key, value
                in item.items()
                if key not in (
                    "handle",
                    "footprint_radius",
                )
            }
        )

    with open(
        SCENE_GT_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            serializable,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "\n随机场景Ground Truth已保存："
    )

    print(
        SCENE_GT_FILE.resolve()
    )


# ============================================================
# 15. 相机基本检查
# ============================================================

def check_camera_exists(
    sim: Any,
) -> int:
    camera = (
        find_unique_object_by_alias(
            sim,
            sim.sceneobject_visionsensor,
            "rgbd_camera",
        )
    )

    print(
        "\nRGB-D Camera："
        f"{get_full_path(sim, camera)}"
    )

    return camera


# ============================================================
# 16. 主程序
# ============================================================

def main() -> None:
    random_seed = (
        int(RANDOM_SEED)
        if RANDOM_SEED is not None
        else random.SystemRandom().randrange(0, 2**32)
    )
    random.seed(random_seed)
    np.random.seed(random_seed)
    scene_id = (
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        f"_seed_{random_seed}"
    )

    print(f"Scene ID：{scene_id}")
    print(f"Scene mode：{SCENE_MODE}")

    print(
        "正在连接CoppeliaSim..."
    )

    client = RemoteAPIClient()
    sim = client.require("sim")

    print(
        "连接成功。"
    )

    if (
        sim.getSimulationState()
        != sim.simulation_stopped
    ):
        raise RuntimeError(
            "请从STOP状态启动场景生成。"
        )

    camera = check_camera_exists(
        sim
    )

    robot_base = find_robot_base(
        sim
    )

    print(
        "Robot Base："
        f"{get_full_path(sim, robot_base)}"
    )

    camera_model = build_camera_model(
        sim,
        camera,
        robot_base,
    )

    clear_previous_pipeline_outputs()

    # --------------------------------------------------------
    # 先读取当前工件。
    #
    # 第一次运行时用来建立table/workspace reference。
    # 之后则只是为了删除上一轮工件。
    # --------------------------------------------------------

    existing_objects = (
        find_existing_test_objects(
            sim
        )
    )

    print(
        f"\n当前找到测试工件："
        f"{len(existing_objects)}"
    )

    for item in existing_objects:
        print(
            f"  {item['alias']} "
            f"({item['class']})"
        )

    reference = (
        load_or_create_reference(
            sim,
            robot_base,
            existing_objects,
        )
    )

    # 建立参考之后再删旧物体。
    remove_existing_test_objects(
        sim,
        existing_objects,
    )

    # --------------------------------------------------------
    # 新随机场景
    # --------------------------------------------------------

    objects = generate_random_scene(
        sim,
        robot_base,
        reference,
        camera_model,
    )

    save_scene_ground_truth(
        objects,
        reference,
        scene_id,
        random_seed,
        camera_model,
    )

    print(
        "\n"
        "============================================================"
    )

    print(
        "随机场景生成完成。"
    )

    print(
        "============================================================"
    )

    print(
        f"工件数量：{len(objects)}"
    )

    if SCENE_MODE in {"physics", "dynamic", "settled", "drop"}:
        print(
            "\n物理场景已经稳定并保持PAUSED。"
            "在完成分割、识别和评价之前不要点击Stop，"
            "否则CoppeliaSim可能恢复仿真初始状态。"
        )

    print(
        "\n现在回到CoppeliaSim检查："
    )

    print(
        "1. 所有物体是否都在相机工作区域内；"
    )

    print("2. 计划接触面是否贴合且不存在明显穿透；")

    print("3. 堆叠、斜靠或桥接物体是否符合预期支撑关系；")

    print(
        "\n确认正常后依次运行："
    )

    print(
        "python segment_multiple_objects.py"
    )

    print(
        "python recognize_objects.py"
    )

    print(
        "python evaluate_ground_truth.py"
    )


if __name__ == "__main__":
    main()
