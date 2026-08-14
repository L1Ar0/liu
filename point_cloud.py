from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    import open3d as o3d
except ImportError as exc:
    raise RuntimeError(
        "没有安装 Open3D。\n"
        "请在虚拟环境中运行：\n"
        "python -m pip install open3d"
    ) from exc

from remote_session import RemoteAPIClient


# ============================================================
# 1. 基本配置
# ============================================================

OUTPUT_DIR = Path("camera_output")

# 点云保存文件
CAMERA_PCD_FILE = OUTPUT_DIR / "point_cloud_camera.ply"
BASE_PCD_FILE = OUTPUT_DIR / "point_cloud_base.ply"

# 是否显示 Open3D 窗口
SHOW_POINT_CLOUD = True

# 是否保存点云
SAVE_POINT_CLOUD = True

# 去掉刚好落在 far clipping plane 的背景像素
FAR_PLANE_MARGIN_M = 0.002


# ============================================================
# 2. CoppeliaSim 对象查找
# ============================================================

def get_full_path(sim: Any, handle: int) -> str:
    """读取对象完整路径。"""

    try:
        return str(sim.getObjectAlias(handle, 2))
    except Exception:
        return str(sim.getObjectAlias(handle))


def find_unique_object_by_alias(
    sim: Any,
    object_type: int,
    alias: str,
) -> int:
    """
    在整个场景中搜索指定类型且 alias 匹配的对象。

    这样不用担心：
        /rgbd_camera
        /iiwa/RG2/rgbd_camera
        /iiwa/RG2/gripper_tip/rgbd_camera

    等不同绝对路径问题。
    """

    objects = sim.getObjectsInTree(
        sim.handle_scene,
        object_type,
        0,
    )

    matches = []

    for handle in objects:
        object_alias = str(
            sim.getObjectAlias(handle)
        )

        if object_alias == alias:
            matches.append(int(handle))

    if len(matches) == 0:
        raise RuntimeError(
            f"场景中没有找到 {alias}。"
        )

    if len(matches) > 1:
        print(
            f"警告：找到 {len(matches)} 个 "
            f"{alias}，使用第一个："
        )

        for handle in matches:
            print(
                "  ",
                get_full_path(sim, handle)
            )

    return matches[0]


def get_kuka_joints_from_tip(
    sim: Any,
    gripper_tip: int,
) -> list[int]:
    """
    从 gripper_tip 沿父级向上回溯，
    找到 KUKA 的七个关节。

    因为 RG2 内部运动关节不是 gripper_tip 的祖先，
    所以通常不会混入这里。
    """

    joints_reverse = []

    current = gripper_tip

    while current != -1:

        if (
            sim.getObjectType(current)
            == sim.sceneobject_joint
        ):
            joints_reverse.append(
                int(current)
            )

        current = int(
            sim.getObjectParent(current)
        )

    joints = list(
        reversed(joints_reverse)
    )

    return joints


# ============================================================
# 3. 获取 RGB-D
# ============================================================

def capture_rgbd(
    sim: Any,
    camera: int,
):
    """
    主动处理 Vision Sensor，
    然后读取 RGB 和米制 Depth。
    """

    print("\n正在处理 RGB-D Vision Sensor...")

    # Explicit handling = ON 时需要这一句. Physics scene generation renders
    # once before pausing, and this direct call safely refreshes the frame.
    sim.handleVisionSensor(camera)
    # Never start/stop a paused simulation from this low-level capture helper.
    # Physics scene generation already renders all explicit-handling sensors
    # before pausing.  Starting a new simulation here used to require a second
    # stepping client and could strand the ZMQ add-on coroutine when a stage
    # failed or was interrupted.  A direct handle call is safe in both the
    # stopped and paused states.

    # --------------------------------------------------------
    # RGB
    # --------------------------------------------------------

    rgb_buffer, resolution = (
        sim.getVisionSensorImg(
            camera,
            0,
        )
    )

    width = int(resolution[0])
    height = int(resolution[1])

    rgb = np.frombuffer(
        rgb_buffer,
        dtype=np.uint8,
    )

    expected_rgb_size = (
        width * height * 3
    )

    if rgb.size != expected_rgb_size:
        raise RuntimeError(
            "RGB Buffer 长度异常："
            f"{rgb.size} != "
            f"{expected_rgb_size}"
        )

    rgb = rgb.reshape(
        height,
        width,
        3,
    )

    # CoppeliaSim 图像：
    # 左下角为原点
    #
    # NumPy / 常规图像：
    # 左上角为原点
    #
    # 所以统一上下翻转。
    rgb = np.flipud(rgb)

    # --------------------------------------------------------
    # Depth
    # --------------------------------------------------------

    depth_buffer, depth_resolution = (
        sim.getVisionSensorDepth(
            camera,
            1,
        )
    )

    depth_width = int(
        depth_resolution[0]
    )

    depth_height = int(
        depth_resolution[1]
    )

    if (
        depth_width != width
        or depth_height != height
    ):
        raise RuntimeError(
            "RGB 与 Depth 分辨率不一致。"
        )

    # ZeroMQ Python API 返回 packed float32 bytes
    if isinstance(
        depth_buffer,
        (bytes, bytearray, memoryview),
    ):
        depth = np.frombuffer(
            depth_buffer,
            dtype=np.float32,
        ).copy()

    else:
        depth = np.asarray(
            depth_buffer,
            dtype=np.float32,
        )

    expected_depth_size = (
        width * height
    )

    if depth.size != expected_depth_size:
        raise RuntimeError(
            "Depth Buffer 长度异常："
            f"{depth.size} != "
            f"{expected_depth_size}"
        )

    depth = depth.reshape(
        height,
        width,
    )

    depth = np.flipud(depth)

    return (
        rgb,
        depth,
        width,
        height,
    )


# ============================================================
# 4. 获取相机参数
# ============================================================

def get_camera_parameters(
    sim: Any,
    camera: int,
    width: int,
    height: int,
):
    """
    获取：
        perspective angle
        horizontal FOV
        vertical FOV
        near
        far

    并构造近似针孔相机内参矩阵 K。
    """

    perspective_angle = float(
        sim.getObjectFloatParam(
            camera,
            sim.visionfloatparam_perspective_angle,
        )
    )

    near_clip = float(
        sim.getObjectFloatParam(
            camera,
            sim.visionfloatparam_near_clipping,
        )
    )

    far_clip = float(
        sim.getObjectFloatParam(
            camera,
            sim.visionfloatparam_far_clipping,
        )
    )

    ratio = (
        width / height
    )

    # CoppeliaSim 的 Perspective angle
    # 总是对应分辨率较大的那个方向。
    if width >= height:

        fov_x = perspective_angle

        fov_y = 2.0 * math.atan(
            math.tan(
                perspective_angle / 2.0
            )
            / ratio
        )

    else:

        fov_y = perspective_angle

        fov_x = 2.0 * math.atan(
            math.tan(
                perspective_angle / 2.0
            )
            * ratio
        )

    # --------------------------------------------------------
    # 标准计算机视觉形式的内参
    #
    # 注意：
    # K 本身按照常见 CV 图像坐标定义；
    # 后面转换到 CoppeliaSim Camera frame
    # 时会处理 X/Y 方向符号。
    # --------------------------------------------------------

    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0

    fx = (
        (width - 1)
        / (
            2.0
            * math.tan(
                fov_x / 2.0
            )
        )
    )

    fy = (
        (height - 1)
        / (
            2.0
            * math.tan(
                fov_y / 2.0
            )
        )
    )

    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    return {
        "perspective_angle": perspective_angle,
        "fov_x": fov_x,
        "fov_y": fov_y,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "K": K,
        "near": near_clip,
        "far": far_clip,
    }


# ============================================================
# 5. Depth → Camera Point Cloud
# ============================================================

def depth_to_camera_point_cloud(
    depth: np.ndarray,
    rgb: np.ndarray,
    params: dict,
):
    """
    将米制 Depth 反投影到 CoppeliaSim
    Vision Sensor 坐标系。

    CoppeliaSim Vision Sensor 坐标：

        +X = 图像左
        +Y = 图像上
        +Z = 看向前方

    因为图像已经 np.flipud，
    当前数组坐标：

        u 向右增加
        v 向下增加

    所以：
        左边 X 为正
        上边 Y 为正
    """

    height, width = depth.shape

    near = params["near"]
    far = params["far"]

    fov_x = params["fov_x"]
    fov_y = params["fov_y"]

    # --------------------------------------------------------
    # 每个像素坐标
    # --------------------------------------------------------

    v, u = np.indices(
        (height, width),
        dtype=np.float64,
    )

    z = depth.astype(
        np.float64
    )

    # --------------------------------------------------------
    # 直接按照 CoppeliaSim 相机坐标系反投影
    #
    # u = 0          → 图像左边 → +X
    # u = width - 1  → 图像右边 → -X
    #
    # v = 0           → 图像顶部 → +Y
    # v = height - 1  → 图像底部 → -Y
    # --------------------------------------------------------

    if width > 1:
        x_normalized = (
            1.0
            - 2.0
            * u
            / (width - 1)
        )
    else:
        x_normalized = np.zeros_like(u)

    if height > 1:
        y_normalized = (
            1.0
            - 2.0
            * v
            / (height - 1)
        )
    else:
        y_normalized = np.zeros_like(v)

    x = (
        z
        * math.tan(
            fov_x / 2.0
        )
        * x_normalized
    )

    y = (
        z
        * math.tan(
            fov_y / 2.0
        )
        * y_normalized
    )

    points = np.stack(
        [
            x,
            y,
            z,
        ],
        axis=-1,
    )

    # --------------------------------------------------------
    # 有效点过滤
    # --------------------------------------------------------

    valid = (
        np.isfinite(z)
        & (z > near)
        & (
            z
            < far
            - FAR_PLANE_MARGIN_M
        )
    )

    points_camera = points[
        valid
    ]

    colors = (
        rgb.astype(np.float64)
        / 255.0
    )[valid]

    return (
        points_camera,
        colors,
        valid,
    )


# ============================================================
# 6. Camera frame → Robot base frame
# ============================================================

def transform_points(
    points: np.ndarray,
    matrix_3x4: np.ndarray,
) -> np.ndarray:
    """
    p_base = R_base_camera * p_camera + t
    """

    rotation = matrix_3x4[
        :3,
        :3,
    ]

    translation = matrix_3x4[
        :3,
        3,
    ]

    return (
        points @ rotation.T
        + translation
    )


# ============================================================
# 7. Open3D Point Cloud
# ============================================================

def create_open3d_cloud(
    points: np.ndarray,
    colors: np.ndarray,
):
    """构造 Open3D PointCloud。"""

    cloud = (
        o3d.geometry.PointCloud()
    )

    cloud.points = (
        o3d.utility.Vector3dVector(
            points
        )
    )

    cloud.colors = (
        o3d.utility.Vector3dVector(
            colors
        )
    )

    return cloud


# ============================================================
# 8. 主程序
# ============================================================

def main():

    print(
        "正在连接 CoppeliaSim..."
    )

    client = RemoteAPIClient()
    sim = client.require("sim")

    print("连接成功。")

    # --------------------------------------------------------
    # 保持仿真停止
    # --------------------------------------------------------

    if (
        sim.getSimulationState()
        != sim.simulation_stopped
    ):
        raise RuntimeError(
            "请先停止 CoppeliaSim 仿真。\n"
            "本程序不需要启动动力学。"
        )

    # --------------------------------------------------------
    # 找到相机
    # --------------------------------------------------------

    camera = (
        find_unique_object_by_alias(
            sim,
            sim.sceneobject_visionsensor,
            "rgbd_camera",
        )
    )

    print(
        "\n找到 RGB-D Camera："
    )

    print(
        get_full_path(
            sim,
            camera,
        )
    )

    # --------------------------------------------------------
    # 找 gripper_tip
    # --------------------------------------------------------

    gripper_tip = (
        find_unique_object_by_alias(
            sim,
            sim.sceneobject_dummy,
            "gripper_tip",
        )
    )

    print(
        "\n找到 gripper_tip："
    )

    print(
        get_full_path(
            sim,
            gripper_tip,
        )
    )

    # --------------------------------------------------------
    # 从 gripper_tip 寻找 KUKA 7 关节
    # --------------------------------------------------------

    joints = get_kuka_joints_from_tip(
        sim,
        gripper_tip,
    )

    print(
        "\n========== KUKA 运动链 =========="
    )

    for index, joint in enumerate(
        joints,
        start=1,
    ):
        print(
            f"J{index}: "
            f"{get_full_path(sim, joint)}"
        )

    if len(joints) != 7:
        raise RuntimeError(
            f"\n预期找到7个KUKA关节，"
            f"实际找到{len(joints)}个。\n"
            "请检查 gripper_tip 是否确实位于"
            "机械臂末端层级下。"
        )

    # --------------------------------------------------------
    # Joint 1 的父对象作为机器人基座参考坐标系
    # --------------------------------------------------------

    robot_base = int(
        sim.getObjectParent(
            joints[0]
        )
    )

    if robot_base == -1:
        raise RuntimeError(
            "无法找到 KUKA 基座参考对象。"
        )

    print(
        "\n机器人 Base frame："
    )

    print(
        get_full_path(
            sim,
            robot_base,
        )
    )

    # --------------------------------------------------------
    # Capture RGB-D
    # --------------------------------------------------------

    (
        rgb,
        depth,
        width,
        height,
    ) = capture_rgbd(
        sim,
        camera,
    )

    print(
        "\n========== RGB-D =========="
    )

    print(
        f"Resolution = "
        f"{width} x {height}"
    )

    print(
        f"RGB shape  = "
        f"{rgb.shape}"
    )

    print(
        f"Depth shape = "
        f"{depth.shape}"
    )

    # --------------------------------------------------------
    # Camera intrinsics
    # --------------------------------------------------------

    params = get_camera_parameters(
        sim,
        camera,
        width,
        height,
    )

    print(
        "\n========== Camera Parameters =========="
    )

    print(
        "Perspective angle = "
        f"{math.degrees(params['perspective_angle']):.3f}°"
    )

    print(
        "Horizontal FOV = "
        f"{math.degrees(params['fov_x']):.3f}°"
    )

    print(
        "Vertical FOV = "
        f"{math.degrees(params['fov_y']):.3f}°"
    )

    print(
        f"Near = "
        f"{params['near']:.4f} m"
    )

    print(
        f"Far  = "
        f"{params['far']:.4f} m"
    )

    print(
        "\nIntrinsic Matrix K:"
    )

    print(
        params["K"]
    )

    # --------------------------------------------------------
    # Depth → Camera point cloud
    # --------------------------------------------------------

    (
        points_camera,
        colors,
        valid_mask,
    ) = depth_to_camera_point_cloud(
        depth,
        rgb,
        params,
    )

    print(
        "\n========== Camera Point Cloud =========="
    )

    print(
        f"有效点数量："
        f"{len(points_camera)}"
    )

    print(
        "有效像素比例："
        f"{100.0 * valid_mask.mean():.2f}%"
    )

    if len(points_camera) < 100:
        raise RuntimeError(
            "有效点云太少。\n"
            "请检查相机方向、Depth、"
            "Near/Far clipping plane。"
        )

    # --------------------------------------------------------
    # Camera → Base transformation
    # --------------------------------------------------------

    matrix = sim.getObjectMatrix(
        camera,
        robot_base,
    )

    T_base_camera = np.asarray(
        matrix,
        dtype=np.float64,
    ).reshape(
        3,
        4,
    )

    print(
        "\n========== T_base_camera =========="
    )

    print(
        T_base_camera
    )

    points_base = transform_points(
        points_camera,
        T_base_camera,
    )

    # --------------------------------------------------------
    # Open3D
    # --------------------------------------------------------

    cloud_camera = (
        create_open3d_cloud(
            points_camera,
            colors,
        )
    )

    cloud_base = (
        create_open3d_cloud(
            points_base,
            colors,
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if SAVE_POINT_CLOUD:

        success1 = o3d.io.write_point_cloud(
            str(CAMERA_PCD_FILE),
            cloud_camera,
        )

        success2 = o3d.io.write_point_cloud(
            str(BASE_PCD_FILE),
            cloud_base,
        )

        print(
            "\n========== 保存 =========="
        )

        print(
            f"Camera frame PLY："
            f"{CAMERA_PCD_FILE.resolve()}"
        )

        print(
            f"保存状态：{success1}"
        )

        print(
            f"\nBase frame PLY："
            f"{BASE_PCD_FILE.resolve()}"
        )

        print(
            f"保存状态：{success2}"
        )

    # --------------------------------------------------------
    # 点云范围检查
    # --------------------------------------------------------

    min_xyz = np.min(
        points_base,
        axis=0,
    )

    max_xyz = np.max(
        points_base,
        axis=0,
    )

    print(
        "\n========== Base Frame 点云范围 =========="
    )

    print(
        "X: "
        f"{min_xyz[0]:.4f} "
        f"~ {max_xyz[0]:.4f} m"
    )

    print(
        "Y: "
        f"{min_xyz[1]:.4f} "
        f"~ {max_xyz[1]:.4f} m"
    )

    print(
        "Z: "
        f"{min_xyz[2]:.4f} "
        f"~ {max_xyz[2]:.4f} m"
    )

    # --------------------------------------------------------
    # Open3D 可视化
    # --------------------------------------------------------

    if SHOW_POINT_CLOUD:

        print(
            "\n正在打开 Open3D..."
        )

        print(
            "鼠标左键：旋转"
        )

        print(
            "鼠标滚轮：缩放"
        )

        print(
            "Shift + 左键：平移"
        )

        # Base frame 坐标轴
        base_frame = (
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

        # Camera frame
        camera_frame = (
            o3d.geometry.TriangleMesh
            .create_coordinate_frame(
                size=0.06,
            )
        )

        T4 = np.eye(
            4,
            dtype=np.float64,
        )

        T4[:3, :4] = (
            T_base_camera
        )

        camera_frame.transform(
            T4
        )

        o3d.visualization.draw_geometries(
            [
                cloud_base,
                base_frame,
                camera_frame,
            ],
            window_name=(
                "KUKA RGB-D Point Cloud "
                "- Robot Base Frame"
            ),
            width=1200,
            height=800,
        )

    print(
        "\n========== 本阶段完成 =========="
    )

    print(
        "RGB-D 已成功转换为 3D 点云。"
    )

    print(
        "点云已经表达在 KUKA "
        "机器人基座坐标系中。"
    )


if __name__ == "__main__":
    main()
