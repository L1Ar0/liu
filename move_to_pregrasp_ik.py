import math
import time

from remote_session import RemoteAPIClient


# 机械臂运动被分成多少个小步
MOVE_STEPS = 150

# 每一步之间暂停时间，单位为秒
STEP_DELAY = 0.025

# 阻尼最小二乘参数
IK_DAMPING = 0.1
IK_MAX_ITERATIONS = 100

# 最终允许的位置误差：5 mm
MAX_FINAL_ERROR = 0.005


def find_object(sim, possible_paths):
    """按照给定路径依次查找场景对象。"""
    for path in possible_paths:
        handle = sim.getObject(path, {"noError": True})

        if handle >= 0:
            print(f"找到对象：{path}")
            return handle

    raise RuntimeError(
        "无法找到对象，尝试过：\n"
        + "\n".join(possible_paths)
    )


def collect_chain_joints(sim, tip_handle, base_handle):
    """
    从 iiwa_tip 沿父对象向上查找，直到 iiwa 基座，
    收集这条运动链上的所有关节。
    """
    joints_from_tip_to_base = []
    current_handle = tip_handle

    while current_handle != base_handle:
        parent_handle = sim.getObjectParent(current_handle)

        if parent_handle < 0:
            raise RuntimeError(
                "从 iiwa_tip 向上查找时到达了场景根节点，"
                "但没有找到 /iiwa。\n"
                "请检查 iiwa_tip 是否确实位于机械臂层级结构下。"
            )

        object_type = sim.getObjectType(parent_handle)

        if object_type == sim.sceneobject_joint:
            joints_from_tip_to_base.append(parent_handle)

        current_handle = parent_handle

    # 当前顺序是末端到基座，反转后变成基座到末端
    joints_from_tip_to_base.reverse()
    return joints_from_tip_to_base


def distance_between(position_a, position_b):
    """计算两个三维点之间的欧氏距离。"""
    return math.sqrt(
        (position_a[0] - position_b[0]) ** 2
        + (position_a[1] - position_b[1]) ** 2
        + (position_a[2] - position_b[2]) ** 2
    )


def interpolate_position(start_position, end_position, ratio):
    """在两个三维位置之间进行线性插值。"""
    return [
        start_position[0]
        + (end_position[0] - start_position[0]) * ratio,

        start_position[1]
        + (end_position[1] - start_position[1]) * ratio,

        start_position[2]
        + (end_position[2] - start_position[2]) * ratio,
    ]


def print_position(title, position):
    print(title)
    print(
        f"  x = {position[0]:.4f} m\n"
        f"  y = {position[1]:.4f} m\n"
        f"  z = {position[2]:.4f} m"
    )


def main():
    print("正在连接 CoppeliaSim……")

    client = RemoteAPIClient()
    sim = client.require("sim")
    simIK = client.require("simIK")

    print("连接成功。")

    # 本阶段要求仿真保持停止
    simulation_state = sim.getSimulationState()

    if simulation_state != sim.simulation_stopped:
        raise RuntimeError(
            "当前仿真没有停止。\n"
            "请先点击 CoppeliaSim 顶部的停止按钮，"
            "然后重新运行程序。"
        )

    # -----------------------------
    # 1. 获取基座、Tip 和 Target
    # -----------------------------

    base_handle = find_object(
        sim,
        [
            "/iiwa",
        ],
    )

    tip_handle = find_object(
        sim,
        [
            "/iiwa/connection/iiwa_tip",
            "/iiwa_tip",
        ],
    )

    target_handle = find_object(
        sim,
        [
            "/iiwa_target",
        ],
    )

    # -----------------------------
    # 2. 检查机械臂关节链
    # -----------------------------

    joint_handles = collect_chain_joints(
        sim,
        tip_handle,
        base_handle,
    )

    print("\n检测到的机械臂关节数量：", len(joint_handles))

    for index, joint_handle in enumerate(joint_handles, start=1):
        try:
            alias = sim.getObjectAlias(joint_handle)
        except Exception:
            alias = f"handle={joint_handle}"

        print(f"  关节 {index}：{alias}")

    if len(joint_handles) != 7:
        raise RuntimeError(
            f"预期找到 7 个关节，实际找到 {len(joint_handles)} 个。\n"
            "请检查 iiwa_tip 是否确实位于第七个关节之后。"
        )

    # 保存初始关节状态，出错时用于恢复
    original_joint_positions = [
        sim.getJointPosition(joint_handle)
        for joint_handle in joint_handles
    ]

    original_joint_modes = [
        sim.getJointMode(joint_handle)
        for joint_handle in joint_handles
    ]

    # 切换为运动学模式
    print("\n正在把七个关节切换为 kinematic 模式……")

    for joint_handle in joint_handles:
        sim.setJointMode(
            joint_handle,
            sim.jointmode_kinematic,
        )

    # -----------------------------
    # 3. 记录起点和终点
    # -----------------------------

    start_tip_position = sim.getObjectPosition(
        tip_handle,
        sim.handle_world,
    )

    final_target_position = sim.getObjectPosition(
        target_handle,
        sim.handle_world,
    )

    print()
    print_position(
        "机械臂末端当前位置：",
        start_tip_position,
    )

    print()
    print_position(
        "预抓取目标位置：",
        final_target_position,
    )

    total_distance = distance_between(
        start_tip_position,
        final_target_position,
    )

    print(
        f"\n末端需要移动的直线距离："
        f"{total_distance:.4f} m "
        f"({total_distance * 1000:.1f} mm)"
    )

    # 目标点过远时先停止，避免机械臂发生大幅动作
    if total_distance > 1.2:
        raise RuntimeError(
            "iiwa_tip 与 iiwa_target 之间距离超过 1.2 m。\n"
            "这通常表示目标点位置不合理，请先检查 target_cube "
            "和 iiwa_target 的世界坐标。"
        )

    ik_environment = None
    movement_completed = False

    try:
        # -----------------------------
        # 4. 建立 IK 环境
        # -----------------------------

        print("\n正在建立位置约束 IK 环境……")

        ik_environment = simIK.createEnvironment()

        ik_group = simIK.createGroup(
            ik_environment,
            "iiwa_position_ik_group",
        )

        # 使用阻尼最小二乘法，提高接近奇异位置时的稳定性
        simIK.setGroupCalculation(
            ik_environment,
            ik_group,
            simIK.method_damped_least_squares,
            IK_DAMPING,
            IK_MAX_ITERATIONS,
        )

        (
            ik_element,
            sim_to_ik_map,
            ik_to_sim_map,
        ) = simIK.addElementFromScene(
            ik_environment,
            ik_group,
            base_handle,
            tip_handle,
            target_handle,
            simIK.constraint_position,
        )

        print(f"IK 元素创建成功，句柄：{ik_element}")

        # -----------------------------
        # 5. 把 target 暂时放到 tip 当前位置
        # -----------------------------

        sim.setObjectPosition(
            target_handle,
            start_tip_position,
            sim.handle_world,
        )

        result, flags, precision = simIK.handleGroup(
            ik_environment,
            ik_group,
            {
                "syncWorlds": True,
            },
        )

        if result != simIK.result_success:
            raise RuntimeError(
                "IK 初始化失败。\n"
                f"flags = {flags}\n"
                f"precision = {precision}"
            )

        print("\nIK 初始化成功。")
        print("机械臂即将开始移动……")
        time.sleep(1.0)

        # -----------------------------
        # 6. 分步移动目标并求解 IK
        # -----------------------------

        for step in range(1, MOVE_STEPS + 1):
            ratio = step / MOVE_STEPS

            current_target_position = interpolate_position(
                start_tip_position,
                final_target_position,
                ratio,
            )

            sim.setObjectPosition(
                target_handle,
                current_target_position,
                sim.handle_world,
            )

            result, flags, precision = simIK.handleGroup(
                ik_environment,
                ik_group,
                {
                    "syncWorlds": True,
                },
            )

            if result != simIK.result_success:
                raise RuntimeError(
                    f"IK 在第 {step}/{MOVE_STEPS} 步求解失败。\n"
                    f"flags = {flags}\n"
                    f"precision = {precision}\n"
                    "机械臂将恢复到运行前的位置。"
                )

            # 每移动约 10% 输出一次进度
            report_interval = max(1, MOVE_STEPS // 10)

            if (
                step == 1
                or step == MOVE_STEPS
                or step % report_interval == 0
            ):
                print(
                    f"移动进度："
                    f"{step}/{MOVE_STEPS} "
                    f"({ratio * 100:.0f}%)"
                )

            time.sleep(STEP_DELAY)

        movement_completed = True

        # -----------------------------
        # 7. 检查最终误差
        # -----------------------------

        final_tip_position = sim.getObjectPosition(
            tip_handle,
            sim.handle_world,
        )

        actual_target_position = sim.getObjectPosition(
            target_handle,
            sim.handle_world,
        )

        final_error = distance_between(
            final_tip_position,
            actual_target_position,
        )

        print("\n========== IK 运动结果 ==========")

        print()
        print_position(
            "最终 iiwa_tip 位置：",
            final_tip_position,
        )

        print()
        print_position(
            "最终 iiwa_target 位置：",
            actual_target_position,
        )

        print(
            f"\n最终位置误差："
            f"{final_error:.6f} m "
            f"({final_error * 1000:.2f} mm)"
        )

        if final_error <= MAX_FINAL_ERROR:
            print("\n位置约束 IK 测试成功。")
        else:
            print(
                "\n机械臂已经接近目标，"
                "但误差超过 5 mm，需要进一步调整 IK 参数。"
            )

    except Exception:
        # 出错后恢复到程序运行前的状态
        print("\n检测到错误，正在恢复机械臂初始姿态……")

        for joint_handle, joint_position in zip(
            joint_handles,
            original_joint_positions,
        ):
            sim.setJointPosition(
                joint_handle,
                joint_position,
            )

        for joint_handle, joint_mode in zip(
            joint_handles,
            original_joint_modes,
        ):
            sim.setJointMode(
                joint_handle,
                joint_mode,
            )

        # iiwa_target 恢复到原来的预抓取位置
        sim.setObjectPosition(
            target_handle,
            final_target_position,
            sim.handle_world,
        )

        raise

    finally:
        if ik_environment is not None:
            simIK.eraseEnvironment(ik_environment)

    if movement_completed:
        print("\nIK 环境已经清理。")
        print("机械臂保持在预抓取位置。")
        print("七个关节保持为 kinematic 模式。")


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print("\n程序运行失败：")
        print(error)

        print("\n请检查：")
        print("1. CoppeliaSim 仿真是否已经停止")
        print("2. 场景中是否存在 /iiwa")
        print("3. iiwa_tip 是否位于完整的七关节链末端")
        print("4. iiwa_target 是否位于机械臂可达范围内")
        print("5. simIK 插件是否正常加载")
