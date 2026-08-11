import json
from pathlib import Path

import numpy as np
from PIL import Image

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from point_cloud import get_camera_parameters


# ============================================================
# 配置
# ============================================================

# 你截图中显示的实际完整路径
CAMERA_PATH = "/iiwa/RG2/rgbd_camera"

OUTPUT_DIR = Path("camera_output")

NEAR_CLIP = 0.05
FAR_CLIP = 1.20


def main():
    print("正在连接 CoppeliaSim...")

    client = RemoteAPIClient()
    sim = client.require("sim")

    print("连接成功。")

    # ========================================================
    # 1. 获取 Vision Sensor
    # ========================================================

    try:
        camera = sim.getObject(CAMERA_PATH)
    except Exception as exc:
        raise RuntimeError(
            f"找不到相机：{CAMERA_PATH}\n"
            "请检查 Scene hierarchy 中的完整路径。"
        ) from exc

    if sim.getObjectType(camera) != sim.sceneobject_visionsensor:
        raise RuntimeError(
            f"{CAMERA_PATH} 不是 Vision Sensor。"
        )

    print(f"Camera handle = {camera}")

    # ========================================================
    # 2. 主动处理 Vision Sensor
    # ========================================================

    print("\n正在主动处理 Vision Sensor...")

    handle_result = sim.handleVisionSensor(camera)

    # 千万不要：
    #
    # print(handle_result)
    #
    # 某些版本/配置下其中可能含二进制辅助数据。

    if isinstance(handle_result, (list, tuple)):
        detection_count = handle_result[0]
    else:
        detection_count = handle_result

    print(
        f"handleVisionSensor 完成，"
        f"detection count = {detection_count}"
    )

    # ========================================================
    # 3. 获取 RGB
    # ========================================================

    print("\n正在读取 RGB...")

    rgb_buffer, rgb_resolution = sim.getVisionSensorImg(
        camera,
        0
    )

    width = int(rgb_resolution[0])
    height = int(rgb_resolution[1])

    print(
        f"RGB resolution = {width} x {height}"
    )

    print(
        f"RGB buffer type = {type(rgb_buffer).__name__}"
    )

    print(
        f"RGB buffer bytes = {len(rgb_buffer)}"
    )

    # RGB 本来就是 uint8 byte buffer
    rgb = np.frombuffer(
        rgb_buffer,
        dtype=np.uint8
    )

    expected_rgb_size = (
        width * height * 3
    )

    if rgb.size != expected_rgb_size:
        raise RuntimeError(
            "\nRGB Buffer 长度异常。\n"
            f"实际：{rgb.size}\n"
            f"预计：{expected_rgb_size}"
        )

    rgb = rgb.reshape(
        height,
        width,
        3
    )

    # CoppeliaSim图像方向与普通图像显示方向不同
    rgb = np.flipud(rgb)

    camera_parameters = get_camera_parameters(
        sim,
        camera,
        width,
        height,
    )

    print("\n========== RGB 检查 ==========")

    print(
        f"RGB shape   = {rgb.shape}"
    )

    print(
        f"RGB min     = {rgb.min()}"
    )

    print(
        f"RGB max     = {rgb.max()}"
    )

    print(
        f"RGB mean    = {rgb.mean():.3f}"
    )

    center_rgb = rgb[
        height // 2,
        width // 2,
        :
    ]

    print(
        f"中心像素 RGB = "
        f"{center_rgb.tolist()}"
    )

    # ========================================================
    # 4. 获取米制 Depth
    # ========================================================

    print("\n正在读取 Depth...")

    depth_buffer, depth_resolution = (
        sim.getVisionSensorDepth(
            camera,
            1
        )
    )

    depth_width = int(
        depth_resolution[0]
    )

    depth_height = int(
        depth_resolution[1]
    )

    print(
        f"Depth resolution = "
        f"{depth_width} x {depth_height}"
    )

    print(
        f"Depth buffer type = "
        f"{type(depth_buffer).__name__}"
    )

    print(
        f"Depth buffer bytes = "
        f"{len(depth_buffer)}"
    )

    # ========================================================
    # 关键：
    # Depth 是 float32 编码的二进制数据
    # 不应该直接打印
    # ========================================================

    if isinstance(
        depth_buffer,
        (bytes, bytearray, memoryview)
    ):
        # 优先使用 CoppeliaSim 官方的解包接口
        depth_values = sim.unpackFloatTable(
            depth_buffer
        )

        depth = np.asarray(
            depth_values,
            dtype=np.float32
        )

    else:
        # 兼容某些版本直接返回 float list
        depth = np.asarray(
            depth_buffer,
            dtype=np.float32
        )

    expected_depth_size = (
        depth_width * depth_height
    )

    if depth.size != expected_depth_size:
        raise RuntimeError(
            "\nDepth Buffer 长度异常。\n"
            f"实际：{depth.size}\n"
            f"预计：{expected_depth_size}"
        )

    depth = depth.reshape(
        depth_height,
        depth_width
    )

    depth = np.flipud(depth)

    # ========================================================
    # 5. Depth统计
    # ========================================================

    finite_mask = np.isfinite(depth)

    if not finite_mask.any():
        raise RuntimeError(
            "Depth 中没有任何有效浮点数。"
        )

    valid_depth = depth[
        finite_mask
    ]

    print("\n========== Depth 检查 ==========")

    print(
        f"Depth shape = {depth.shape}"
    )

    print(
        f"Depth min   = "
        f"{valid_depth.min():.4f} m"
    )

    print(
        f"Depth max   = "
        f"{valid_depth.max():.4f} m"
    )

    print(
        f"Depth mean  = "
        f"{valid_depth.mean():.4f} m"
    )

    center_depth = depth[
        depth_height // 2,
        depth_width // 2
    ]

    print(
        f"中心像素深度 = "
        f"{center_depth:.4f} m"
    )

    # ========================================================
    # 6. 保存 RGB / Depth
    # ========================================================

    OUTPUT_DIR.mkdir(
        exist_ok=True
    )

    rgb_path = OUTPUT_DIR / "rgb_test.png"

    Image.fromarray(
        rgb,
        mode="RGB"
    ).save(
        rgb_path
    )

    # --------------------------------------------------------
    # Depth可视化
    #
    # 注意：
    # 保存的是用于观察的8-bit图片，
    # 真正深度数据仍然是depth这个float32数组。
    # --------------------------------------------------------

    depth_clipped = np.clip(
        depth,
        NEAR_CLIP,
        FAR_CLIP
    )

    depth_visual = (
        (
            depth_clipped - NEAR_CLIP
        )
        /
        (
            FAR_CLIP - NEAR_CLIP
        )
        * 255.0
    )

    depth_visual = (
        255.0 - depth_visual
    )

    depth_visual = np.clip(
        depth_visual,
        0,
        255
    ).astype(
        np.uint8
    )

    depth_path = (
        OUTPUT_DIR
        / "depth_test.png"
    )

    Image.fromarray(
        depth_visual,
        mode="L"
    ).save(
        depth_path
    )

    # 保存真正米制深度
    np.save(
        OUTPUT_DIR / "depth_meters.npy",
        depth
    )

    # SAM-6D expects an integer depth image in millimetres. This is separate
    # from the 8-bit visualization above.
    depth_mm = np.clip(
        np.rint(depth.astype(np.float64) * 1000.0),
        0,
        np.iinfo(np.uint16).max,
    ).astype(np.uint16)
    depth_mm_path = OUTPUT_DIR / "depth_mm.png"
    Image.fromarray(depth_mm, mode="I;16").save(depth_mm_path)

    camera_path = OUTPUT_DIR / "camera.json"
    camera_path.write_text(
        json.dumps(
            {
                "cam_K": camera_parameters["K"].reshape(-1).tolist(),
                "depth_scale": 1.0,
                "width": width,
                "height": height,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n========== 文件输出 ==========")

    print(
        f"RGB：{rgb_path.resolve()}"
    )

    print(
        f"Depth预览："
        f"{depth_path.resolve()}"
    )

    print(
        "米制Depth："
        f"{(OUTPUT_DIR / 'depth_meters.npy').resolve()}"
    )

    print(f"SAM-6D Depth：{depth_mm_path.resolve()}")
    print(f"SAM-6D Camera：{camera_path.resolve()}")

    # ========================================================
    # 7. 自动诊断
    # ========================================================

    print("\n========== 自动诊断 ==========")

    if rgb.max() == 0:
        print(
            "❌ RGB 完全为黑色。"
        )

        print(
            "需要继续检查 Vision Sensor "
            "的渲染设置。"
        )

    elif rgb.mean() < 3:
        print(
            "⚠ RGB数据存在，但非常暗。"
        )

    else:
        print(
            "✓ RGB数据有效。"
        )

    near_count = np.count_nonzero(
        depth < FAR_CLIP - 0.01
    )

    total_count = depth.size

    visible_ratio = (
        near_count / total_count
    )

    print(
        f"非Far-plane像素比例："
        f"{visible_ratio * 100:.2f}%"
    )

    if visible_ratio < 0.01:
        print(
            "❌ Depth几乎全部位于Far plane。"
        )

        print(
            "这说明相机基本没有看到物体，"
            "需要检查相机方向或 Entity to render。"
        )

    else:
        print(
            "✓ Depth中检测到了实际场景表面。"
        )

    print("\n测试结束。")


if __name__ == "__main__":
    main()
