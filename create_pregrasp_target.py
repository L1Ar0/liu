from remote_session import RemoteAPIClient


# 初始测试时，让目标点位于方块中心上方 30 cm
PREGRASP_HEIGHT = 0.30


def find_object(sim, paths: list[str]) -> int:
    """尝试多个对象路径，返回找到的第一个对象句柄。"""
    for path in paths:
        handle = sim.getObject(path, {"noError": True})

        if handle >= 0:
            print(f"找到对象：{path}")
            return handle

    raise RuntimeError(
        "没有找到对象，尝试过以下路径：\n"
        + "\n".join(paths)
    )


def main() -> None:
    print("正在连接 CoppeliaSim……")

    client = RemoteAPIClient()
    sim = client.require("sim")

    print("连接成功。")

    # 1. 找到末端 TCP
    tip_handle = find_object(
        sim,
        [
            "/iiwa_tip",
            "/iiwa/connection/iiwa_tip",
            "/iiwa/iiwa_tip",
        ],
    )

    # 2. 找到目标方块
    cube_handle = find_object(
        sim,
        [
            "/target_cube",
            "/iiwa/target_cube",
        ],
    )

    # 3. 检查场景中是否已经存在 iiwa_target
    target_handle = sim.getObject(
        "/iiwa_target",
        {"noError": True},
    )

    if target_handle < 0:
        print("\n场景中没有 iiwa_target，正在自动创建……")

        # 创建尺寸为 4 cm 的 Dummy
        target_handle = sim.createDummy(0.04)
        sim.setObjectAlias(target_handle, "iiwa_target")

        print("iiwa_target 创建成功。")
    else:
        print("\n场景中已经存在 iiwa_target，将更新其位置。")

    # 保证 target 不是 connection 或 iiwa_tip 的子对象
    parent_handle = sim.getObjectParent(target_handle)

    if parent_handle != sim.handle_world:
        print("检测到 iiwa_target 存在父对象，正在移动到场景根节点……")

        sim.setObjectParent(
            target_handle,
            sim.handle_world,
            True,
        )

    # 4. 读取 iiwa_tip 的完整世界位姿
    #
    # pose 的结构为：
    # [x, y, z, qx, qy, qz, qw]
    tip_pose = sim.getObjectPose(
        tip_handle,
        sim.handle_world,
    )

    # 5. 读取方块中心的世界坐标
    cube_position = sim.getObjectPosition(
        cube_handle,
        sim.handle_world,
    )

    # 6. 先复制 iiwa_tip 当前朝向
    target_pose = tip_pose.copy()

    # 再把位置改到方块正上方
    target_pose[0] = cube_position[0]
    target_pose[1] = cube_position[1]
    target_pose[2] = cube_position[2] + PREGRASP_HEIGHT

    # 7. 设置 iiwa_target 世界位姿
    sim.setObjectPose(
        target_handle,
        target_pose,
        sim.handle_world,
    )

    target_position = sim.getObjectPosition(
        target_handle,
        sim.handle_world,
    )

    relative_to_cube = sim.getObjectPosition(
        target_handle,
        cube_handle,
    )

    print("\n========== 创建结果 ==========")

    print("\ntarget_cube 世界坐标：")
    print(f"x = {cube_position[0]:.4f} m")
    print(f"y = {cube_position[1]:.4f} m")
    print(f"z = {cube_position[2]:.4f} m")

    print("\niiwa_target 世界坐标：")
    print(f"x = {target_position[0]:.4f} m")
    print(f"y = {target_position[1]:.4f} m")
    print(f"z = {target_position[2]:.4f} m")

    print("\niiwa_target 相对 target_cube 的位置：")
    print(f"x = {relative_to_cube[0]:.4f} m")
    print(f"y = {relative_to_cube[1]:.4f} m")
    print(f"z = {relative_to_cube[2]:.4f} m")

    print("\n预期结果应接近：")
    print("x = 0")
    print("y = 0")
    print(f"z = {PREGRASP_HEIGHT:.2f} m")

    print("\n本阶段完成：iiwa_target 已位于方块正上方。")


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print("\n程序运行失败：")
        print(error)

        print("\n请检查：")
        print("1. CoppeliaSim 是否已打开")
        print("2. iiwa_tip 是否存在")
        print("3. target_cube 是否存在")
        print("4. 对象 Alias 是否正确")
