import math

from coppeliasim_zmqremoteapi_client import RemoteAPIClient


def format_vector(name: str, vector: list[float]) -> None:
    """同时以米和毫米打印三维向量。"""
    print(f"\n{name}")
    print(
        f"  米:   x={vector[0]: .4f}, "
        f"y={vector[1]: .4f}, "
        f"z={vector[2]: .4f}"
    )
    print(
        f"  毫米: x={vector[0] * 1000: .1f}, "
        f"y={vector[1] * 1000: .1f}, "
        f"z={vector[2] * 1000: .1f}"
    )


def main() -> None:
    print("正在连接 CoppeliaSim...")

    client = RemoteAPIClient()
    sim = client.require("sim")

    print("连接成功。")

    # 根据对象的 Alias 获取句柄
    tip_handle = sim.getObject("/iiwa_tip")
    cube_handle = sim.getObject("/target_cube")

    print(f"iiwa_tip handle: {tip_handle}")
    print(f"target_cube handle: {cube_handle}")

    # 分别读取二者在世界坐标系中的位置
    tip_world = sim.getObjectPosition(
        tip_handle,
        sim.handle_world
    )

    cube_world = sim.getObjectPosition(
        cube_handle,
        sim.handle_world
    )

    # 关键操作：
    # 读取 target_cube 在 iiwa_tip 坐标系中的位置
    cube_relative_to_tip = sim.getObjectPosition(
        cube_handle,
        tip_handle
    )

    format_vector("iiwa_tip 在世界坐标系中的位置", tip_world)
    format_vector("target_cube 在世界坐标系中的位置", cube_world)
    format_vector(
        "target_cube 在 iiwa_tip 坐标系中的位置",
        cube_relative_to_tip
    )

    # 计算末端原点到方块原点的直线距离
    distance = math.sqrt(
        cube_relative_to_tip[0] ** 2
        + cube_relative_to_tip[1] ** 2
        + cube_relative_to_tip[2] ** 2
    )

    print("\n末端到方块中心的直线距离：")
    print(f"  {distance:.4f} m")
    print(f"  {distance * 1000:.1f} mm")

    print("\n读取完成。")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\n程序运行失败：")
        print(error)

        print("\n请检查：")
        print("1. CoppeliaSim 是否已经打开")
        print("2. 场景中是否存在 iiwa_tip")
        print("3. 场景中是否存在 target_cube")
        print("4. 对象 Alias 是否完全一致")