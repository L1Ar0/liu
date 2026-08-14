import math
import re
import time
from typing import List

from remote_session import RemoteAPIClient


def get_joint_number(sim, joint_handle: int) -> int:
    """
    从关节名称末尾提取编号。

    例如：
    LBR_iiwa_7_R800_joint1 -> 1
    LBR_iiwa_7_R800_joint7 -> 7
    """

    name = sim.getObjectAlias(joint_handle)

    match = re.search(
        r"joint[_ ]?(\d+)$",
        name,
        re.IGNORECASE,
    )

    if match:
        return int(match.group(1))

    return 999


def move_smoothly(
    sim,
    joint_handles: List[int],
    target_degrees: List[float],
    steps: int = 100,
    delay: float = 0.02,
) -> None:
    """
    在仿真停止状态下，让七个关节平滑移动到目标角度。
    """

    if len(joint_handles) != 7:
        raise ValueError(
            f"预期7个关节，实际得到{len(joint_handles)}个。"
        )

    if len(target_degrees) != 7:
        raise ValueError(
            "目标姿态必须包含7个关节角度。"
        )

    # 读取起始关节角，单位为弧度。
    start_positions = [
        sim.getJointPosition(joint)
        for joint in joint_handles
    ]

    # 将目标角度由“度”转换成“弧度”。
    target_positions = [
        math.radians(angle)
        for angle in target_degrees
    ]

    for step_index in range(steps + 1):
        progress = step_index / steps

        # Smoothstep插值：
        # 开始和结束时较慢，中间较快。
        alpha = 3 * progress**2 - 2 * progress**3

        for joint, start, target in zip(
            joint_handles,
            start_positions,
            target_positions,
        ):
            current_position = (
                start + alpha * (target - start)
            )

            sim.setJointPosition(
                joint,
                current_position,
            )

        time.sleep(delay)


def get_object_pose(sim, object_handle: int):
    """
    获取对象相对于世界坐标系的位置和姿态。
    """

    position = sim.getObjectPosition(
        object_handle,
        sim.handle_world,
    )

    orientation = sim.getObjectOrientation(
        object_handle,
        sim.handle_world,
    )

    return position, orientation


def print_pose(
    sim,
    object_handle: int,
    title: str,
):
    """
    打印对象的位置和欧拉角。
    """

    position, orientation = get_object_pose(
        sim,
        object_handle,
    )

    # 位置由米转换成厘米。
    position_cm = [
        value * 100
        for value in position
    ]

    # 姿态由弧度转换成角度。
    orientation_deg = [
        math.degrees(value)
        for value in orientation
    ]

    print(f"\n========== {title} ==========")

    print("世界坐标系下的位置：")

    print(
        f"X = {position[0]:.4f} m "
        f"({position_cm[0]:.2f} cm)"
    )

    print(
        f"Y = {position[1]:.4f} m "
        f"({position_cm[1]:.2f} cm)"
    )

    print(
        f"Z = {position[2]:.4f} m "
        f"({position_cm[2]:.2f} cm)"
    )

    print("世界坐标系下的姿态：")

    print(f"Alpha = {orientation_deg[0]:.2f}°")
    print(f"Beta  = {orientation_deg[1]:.2f}°")
    print(f"Gamma = {orientation_deg[2]:.2f}°")

    return position


def calculate_distance(
    position_a,
    position_b,
) -> float:
    """
    计算两个三维位置之间的直线距离。
    """

    dx = position_b[0] - position_a[0]
    dy = position_b[1] - position_a[1]
    dz = position_b[2] - position_a[2]

    return math.sqrt(
        dx**2 + dy**2 + dz**2
    )


def main() -> None:
    print("正在连接 CoppeliaSim...")

    client = RemoteAPIClient()
    sim = client.require("sim")

    print("连接成功。")

    # 当前程序只能在仿真停止状态下运行。
    simulation_state = sim.getSimulationState()

    if simulation_state != sim.simulation_stopped:
        raise RuntimeError(
            "请先点击CoppeliaSim顶部的停止按钮，"
            "不要启动仿真，然后重新运行程序。"
        )

    # 获取机械臂根对象。
    robot_handle = sim.getObject("/iiwa")

    # 获取机械臂中的全部关节。
    joint_handles = sim.getObjectsInTree(
        robot_handle,
        sim.sceneobject_joint,
        0,
    )

    # 按joint1至joint7排序。
    joint_handles.sort(
        key=lambda handle: get_joint_number(
            sim,
            handle,
        )
    )

    print(f"找到 {len(joint_handles)} 个关节。")

    if len(joint_handles) != 7:
        print("\n当前找到的关节：")

        for joint in joint_handles:
            print(sim.getObjectAlias(joint))

        raise RuntimeError(
            "没有正确找到7个关节。"
        )

    print("\n关节顺序：")

    for index, joint in enumerate(
        joint_handles,
        start=1,
    ):
        name = sim.getObjectAlias(joint)

        print(
            f"joint{index}: {name}"
        )

    # 暂时使用第七个关节作为末端参考点。
    end_reference = joint_handles[-1]

    end_name = sim.getObjectAlias(
        end_reference
    )

    print(
        f"\n当前末端参考对象：{end_name}"
    )

    # 保存初始关节姿态。
    initial_pose = [
        math.degrees(
            sim.getJointPosition(joint)
        )
        for joint in joint_handles
    ]

    # 读取初始末端位置。
    initial_position = print_pose(
        sim,
        end_reference,
        "初始姿态的末端坐标",
    )

    # 第一组测试姿态。
    pose_a = [
        0.0,
        -20.0,
        0.0,
        35.0,
        0.0,
        -20.0,
        0.0,
    ]

    print("\n正在移动到姿态 A...")

    move_smoothly(
        sim,
        joint_handles,
        pose_a,
    )

    print("已到达姿态 A。")

    position_a = print_pose(
        sim,
        end_reference,
        "姿态 A 的末端坐标",
    )

    distance_a = calculate_distance(
        initial_position,
        position_a,
    )

    print(
        f"\n末端相对初始位置移动了："
        f"{distance_a:.4f} m "
        f"({distance_a * 100:.2f} cm)"
    )

    time.sleep(1)

    # 第二组测试姿态。
    pose_b = [
        25.0,
        -30.0,
        15.0,
        45.0,
        -15.0,
        -25.0,
        10.0,
    ]

    print("\n正在移动到姿态 B...")

    move_smoothly(
        sim,
        joint_handles,
        pose_b,
    )

    print("已到达姿态 B。")

    position_b = print_pose(
        sim,
        end_reference,
        "姿态 B 的末端坐标",
    )

    distance_b = calculate_distance(
        position_a,
        position_b,
    )

    print(
        f"\n末端从姿态A到姿态B移动了："
        f"{distance_b:.4f} m "
        f"({distance_b * 100:.2f} cm)"
    )

    time.sleep(1)

    print("\n正在返回初始姿态...")

    move_smoothly(
        sim,
        joint_handles,
        initial_pose,
    )

    print("已返回初始姿态。")

    print_pose(
        sim,
        end_reference,
        "返回后的末端坐标",
    )

    print("\n本次测试完成。")


if __name__ == "__main__":
    main()
