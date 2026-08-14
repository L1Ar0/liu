import math
import re
import time

from remote_session import RemoteAPIClient


def get_joint_number(sim, joint_handle: int) -> int:
    """从关节名称末尾读取 joint1～joint7 的编号。"""

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
    joint_handles: list[int],
    target_degrees: list[float],
    steps: int = 100,
    delay: float = 0.02,
) -> None:
    """
    在仿真停止状态下，平滑修改七个关节的位置。
    不启动动力学，不计算重力和碰撞。
    """

    if len(joint_handles) != 7:
        raise ValueError(
            f"应找到7个关节，实际找到{len(joint_handles)}个。"
        )

    if len(target_degrees) != 7:
        raise ValueError("目标姿态必须包含7个关节角。")

    start_positions = [
        sim.getJointPosition(joint)
        for joint in joint_handles
    ]

    target_positions = [
        math.radians(angle)
        for angle in target_degrees
    ]

    for step_index in range(steps + 1):
        progress = step_index / steps

        # Smoothstep插值，使开始和结束阶段更加平滑。
        alpha = 3 * progress**2 - 2 * progress**3

        for joint, start, target in zip(
            joint_handles,
            start_positions,
            target_positions,
        ):
            current_position = (
                start + alpha * (target - start)
            )

            # 直接修改关节几何位置。
            # 注意：整个程序没有启动动力学仿真。
            sim.setJointPosition(
                joint,
                current_position,
            )

        time.sleep(delay)


def print_joint_angles(sim, joint_handles: list[int]) -> None:
    """打印七个关节当前角度。"""

    print("\n七个关节当前角度：")

    for index, joint in enumerate(joint_handles, start=1):
        name = sim.getObjectAlias(joint)
        angle = math.degrees(
            sim.getJointPosition(joint)
        )

        print(
            f"joint{index}: "
            f"{name}, "
            f"{angle:.2f}°"
        )


def main() -> None:
    print("正在连接 CoppeliaSim...")

    client = RemoteAPIClient()
    sim = client.require("sim")

    print("连接成功。")

    # 本程序要求仿真处于停止状态。
    simulation_state = sim.getSimulationState()

    if simulation_state != sim.simulation_stopped:
        raise RuntimeError(
            "请先点击CoppeliaSim顶部的停止按钮，"
            "再运行本程序。"
        )

    robot_handle = sim.getObject("/iiwa")

    joint_handles = sim.getObjectsInTree(
        robot_handle,
        sim.sceneobject_joint,
        0,
    )

    joint_handles.sort(
        key=lambda handle: get_joint_number(sim, handle)
    )

    print(f"找到 {len(joint_handles)} 个关节。")

    if len(joint_handles) != 7:
        print("\n当前找到的关节名称：")

        for joint in joint_handles:
            print(sim.getObjectAlias(joint))

        raise RuntimeError(
            "没有准确找到7个关节，请检查机械臂根对象。"
        )

    print("\n关节顺序：")

    for index, joint in enumerate(joint_handles, start=1):
        print(
            f"joint{index}: "
            f"{sim.getObjectAlias(joint)}"
        )

    initial_pose = [
        math.degrees(sim.getJointPosition(joint))
        for joint in joint_handles
    ]

    print_joint_angles(sim, joint_handles)

    pose_a = [
        0.0,
        -20.0,
        0.0,
        35.0,
        0.0,
        -20.0,
        0.0,
    ]

    pose_b = [
        25.0,
        -30.0,
        15.0,
        45.0,
        -15.0,
        -25.0,
        10.0,
    ]

    print("\n正在移动到姿态A……")

    move_smoothly(
        sim,
        joint_handles,
        pose_a,
    )

    print("已到达姿态A。")
    print_joint_angles(sim, joint_handles)

    time.sleep(1)

    print("\n正在移动到姿态B……")

    move_smoothly(
        sim,
        joint_handles,
        pose_b,
    )

    print("已到达姿态B。")
    print_joint_angles(sim, joint_handles)

    time.sleep(1)

    print("\n正在返回初始姿态……")

    move_smoothly(
        sim,
        joint_handles,
        initial_pose,
    )

    print("已返回初始姿态。")
    print_joint_angles(sim, joint_handles)

    print("\n运动测试结束。")


if __name__ == "__main__":
    main()
