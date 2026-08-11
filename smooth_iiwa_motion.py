import math
import re
import time

from coppeliasim_zmqremoteapi_client import RemoteAPIClient


def get_joint_number(sim, joint_handle: int) -> int:
    """从关节名称末尾提取 joint1～joint7 的编号。"""

    joint_name = sim.getObjectAlias(joint_handle)

    match = re.search(
        r"joint[_ ]?(\d+)$",
        joint_name,
        re.IGNORECASE,
    )

    if match:
        return int(match.group(1))

    return 999


def move_smoothly(
    sim,
    joint_handles: list[int],
    target_degrees: list[float],
    steps: int = 120,
) -> None:
    """
    通过逐步改变动力学关节的目标位置，
    让机械臂平滑移动到指定姿态。
    """

    if len(joint_handles) != 7:
        raise ValueError("机械臂应当包含7个关节。")

    if len(target_degrees) != 7:
        raise ValueError("目标姿态必须包含7个角度。")

    # 读取当前实际关节角。
    start_positions = [
        sim.getJointPosition(joint)
        for joint in joint_handles
    ]

    # 度转换为弧度。
    target_positions = [
        math.radians(angle)
        for angle in target_degrees
    ]

    for step_index in range(steps + 1):
        progress = step_index / steps

        # 平滑插值，运动开始和结束时更慢。
        alpha = 3 * progress**2 - 2 * progress**3

        for joint, start, target in zip(
            joint_handles,
            start_positions,
            target_positions,
        ):
            interpolated_target = (
                start + alpha * (target - start)
            )

            # 给动力学位置控制器发送“目标角度”，
            # 而不是直接修改关节实际角度。
            sim.setJointTargetPosition(
                joint,
                interpolated_target,
            )

        # 仿真推进一帧。
        sim.step()

        # 只用于减慢显示速度，方便观察。
        time.sleep(0.015)

    # 让控制器额外运行一段时间，使关节稳定在目标附近。
    for _ in range(20):
        sim.step()
        time.sleep(0.015)


def print_joint_states(sim, joint_handles: list[int]) -> None:
    """输出七个关节的实际角度。"""

    print("\n当前实际关节角：")

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

    robot_handle = sim.getObject("/iiwa")

    joint_handles = sim.getObjectsInTree(
        robot_handle,
        sim.sceneobject_joint,
        0,
    )

    joint_handles.sort(
        key=lambda handle: get_joint_number(sim, handle)
    )

    print(f"共找到 {len(joint_handles)} 个关节。")

    if len(joint_handles) != 7:
        raise RuntimeError(
            f"预期找到7个关节，实际找到"
            f"{len(joint_handles)}个。"
        )

    print("\n关节控制顺序：")

    for index, joint in enumerate(joint_handles, start=1):
        print(
            f"joint{index}: "
            f"{sim.getObjectAlias(joint)}"
        )

    # 只开启同步步进，不改变关节模式。
    sim.setStepping(True)
    sim.startSimulation()

    try:
        # 先推进几帧，让动力学模型稳定。
        for _ in range(10):
            sim.step()
            time.sleep(0.02)

        initial_pose = [
            math.degrees(
                sim.getJointPosition(joint)
            )
            for joint in joint_handles
        ]

        print_joint_states(sim, joint_handles)

        # 使用幅度较小、较安全的测试姿态。
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

        print("\n正在移动到姿态 A...")

        move_smoothly(
            sim,
            joint_handles,
            pose_a,
        )

        print("已经到达姿态 A。")
        print_joint_states(sim, joint_handles)

        time.sleep(1)

        print("\n正在移动到姿态 B...")

        move_smoothly(
            sim,
            joint_handles,
            pose_b,
        )

        print("已经到达姿态 B。")
        print_joint_states(sim, joint_handles)

        time.sleep(1)

        print("\n正在返回初始姿态...")

        move_smoothly(
            sim,
            joint_handles,
            initial_pose,
        )

        print("已经返回初始姿态。")
        print_joint_states(sim, joint_handles)

        time.sleep(1)

    finally:
        sim.stopSimulation()

    print("\n测试结束，仿真已经停止。")


if __name__ == "__main__":
    main()