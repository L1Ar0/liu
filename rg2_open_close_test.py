from __future__ import annotations

import math
import re
import time
from typing import Any, Sequence

from remote_session import RemoteAPIClient


# ============================================================
# 参数
# ============================================================

GRIPPER_TIP_PATH = "/gripper_tip"
RG2_SIGNAL_NAME = "RG2_open"

# CoppeliaSim 自带 RG2 模型通常读取 int32 信号 RG2_open：
# 1 表示打开，0 表示关闭。
OPEN_SIGNAL_VALUE = 1
CLOSE_SIGNAL_VALUE = 0

# 每个动作持续的仿真步数。
# 默认仿真步长通常约为 0.05 s，70 步约为 3.5 s。
OPEN_STEPS = 70
CLOSE_STEPS = 70
HOLD_STEPS = 10

# 判断主驱动关节是否发生有效运动的阈值。
JOINT_MOVE_EPS = 1e-4


# ============================================================
# 基础工具
# ============================================================

def get_object_or_raise(sim: Any, path: str) -> int:
    try:
        return int(sim.getObject(path))
    except Exception as exc:
        raise RuntimeError(
            f"场景中找不到对象：{path}\n"
            "请检查对象别名是否完全一致。"
        ) from exc


def get_alias(sim: Any, handle: int) -> str:
    try:
        return str(sim.getObjectAlias(handle))
    except Exception:
        return f"handle_{handle}"


def get_full_path(sim: Any, handle: int) -> str:
    try:
        return str(sim.getObjectAlias(handle, 2))
    except Exception:
        return get_alias(sim, handle)


def get_joint_mode_compat(sim: Any, joint: int) -> int:
    """
    兼容不同 CoppeliaSim / ZeroMQ Remote API 版本。

    有的版本返回整数 mode；有的版本会返回
    (mode, options) 元组。options 当前未使用。
    """
    raw_mode = sim.getJointMode(joint)

    if isinstance(raw_mode, (tuple, list)):
        if len(raw_mode) == 0:
            raise RuntimeError(
                f"getJointMode({joint}) 返回了空结果。"
            )
        return int(raw_mode[0])

    return int(raw_mode)


def set_joint_mode_compat(
    sim: Any,
    joint: int,
    mode: int,
) -> None:
    """
    兼容 setJointMode 的两参数和三参数接口。
    旧式底层接口的第三个 options 参数设为 0。
    """
    try:
        sim.setJointMode(joint, mode)
    except TypeError:
        sim.setJointMode(joint, mode, 0)


def extract_joint_number(name: str) -> int | None:
    """从 joint1、iiwa_joint1 等名称中提取 1~7。"""
    match = re.search(r"joint[_ ]?(\d+)$", name, re.IGNORECASE)
    if not match:
        return None
    number = int(match.group(1))
    return number if 1 <= number <= 7 else None


def get_iiwa_joints_from_tip(sim: Any, tip: int) -> list[int]:
    """
    从 gripper_tip 向父级回溯，寻找 KUKA iiwa 的七个关节。
    优先根据 joint1~joint7 名称排序。
    """
    chain_joints: list[int] = []
    current = tip

    while current != -1:
        if sim.getObjectType(current) == sim.sceneobject_joint:
            chain_joints.append(int(current))
        current = int(sim.getObjectParent(current))

    numbered: list[tuple[int, int]] = []
    for joint in chain_joints:
        number = extract_joint_number(get_alias(sim, joint))
        if number is not None:
            numbered.append((number, joint))

    numbered.sort(key=lambda item: item[0])
    joints = [joint for _, joint in numbered]

    if len(joints) == 7:
        return joints

    # 兼容关节名称没有 joint1~joint7 的情况：
    # 父级回溯顺序为末端到基座，反转后是基座到末端。
    fallback = list(reversed(chain_joints))
    if len(fallback) >= 7:
        return fallback[-7:]

    return fallback


def find_rg2_motor_joint(sim: Any) -> int:
    """
    在场景全部关节中自动寻找 RG2 的主驱动关节。
    常见名称为 openCloseJoint。
    """
    joints = sim.getObjectsInTree(
        sim.handle_scene,
        sim.sceneobject_joint,
        0,
    )

    candidates: list[int] = []

    for joint in joints:
        alias = get_alias(sim, int(joint)).lower()

        if "openclose" in alias or "open_close" in alias:
            candidates.append(int(joint))

    print("\n========== RG2 关节搜索 ==========")

    if not candidates:
        print("没有找到名称包含 openClose 的关节。")
        print("场景中的全部关节如下：")
        for joint in joints:
            print(f"- {get_full_path(sim, int(joint))}")
        raise RuntimeError(
            "无法自动找到 RG2 主驱动关节。\n"
            "请在对象树中确认夹爪内部是否存在 openCloseJoint。"
        )

    print("候选主驱动关节：")
    for handle in candidates:
        print(f"- {get_full_path(sim, handle)}")

    if len(candidates) > 1:
        print(
            "检测到多个 openCloseJoint，默认使用第一个。\n"
            "如果场景中有多个 RG2，请只保留当前测试夹爪，"
            "或在代码中指定正确句柄。"
        )

    return candidates[0]


def set_rg2_signal(sim: Any, value: int) -> None:
    """向 RG2 模型脚本发送 int32 打开/关闭信号。"""
    try:
        sim.setInt32Signal(RG2_SIGNAL_NAME, int(value))
    except Exception as exc:
        raise RuntimeError(
            "当前 CoppeliaSim 无法调用 sim.setInt32Signal。\n"
            "请确认 RG2 模型脚本使用 "
            "sim.getInt32Signal('RG2_open')。"
        ) from exc


def clear_wrong_float_signal(sim: Any) -> None:
    """清除旧程序可能遗留的同名 float 信号。"""
    try:
        sim.clearFloatSignal(RG2_SIGNAL_NAME)
    except Exception:
        pass


def run_steps(
    client: RemoteAPIClient,
    sim: Any,
    motor_joint: int,
    step_count: int,
    label: str,
) -> float:
    """推进指定数量的同步仿真步，并定期输出主关节位置。"""
    print(f"\n开始：{label}")

    report_interval = max(1, step_count // 5)

    for index in range(step_count):
        client.step()

        if index % report_interval == 0 or index == step_count - 1:
            position = float(sim.getJointPosition(motor_joint))
            print(
                f"  step {index + 1:3d}/{step_count}: "
                f"openCloseJoint = {position:.6f}"
            )

    return float(sim.getJointPosition(motor_joint))


def wait_until_stopped(
    client: RemoteAPIClient,
    sim: Any,
    max_steps: int = 50,
) -> None:
    """在 stepping 模式下推进停止流程。"""
    for _ in range(max_steps):
        if sim.getSimulationState() == sim.simulation_stopped:
            return
        client.step()
        time.sleep(0.005)

    raise RuntimeError("仿真未能在预期时间内停止。")


# ============================================================
# 主程序
# ============================================================

def main() -> None:
    print("正在连接 CoppeliaSim……")

    client = RemoteAPIClient()
    sim = client.require("sim")

    print("连接成功。")

    if sim.getSimulationState() != sim.simulation_stopped:
        raise RuntimeError(
            "运行脚本前请先停止 CoppeliaSim 仿真。"
        )

    gripper_tip = get_object_or_raise(sim, GRIPPER_TIP_PATH)

    if sim.getObjectType(gripper_tip) != sim.sceneobject_dummy:
        raise RuntimeError("/gripper_tip 不是 Dummy 对象。")

    arm_joints = get_iiwa_joints_from_tip(sim, gripper_tip)

    print("\n========== KUKA iiwa 关节链 ==========")
    for index, joint in enumerate(arm_joints, start=1):
        print(f"J{index}: {get_full_path(sim, joint)}")

    if len(arm_joints) != 7:
        raise RuntimeError(
            f"从 /gripper_tip 父级链中应找到 7 个机械臂关节，"
            f"实际找到 {len(arm_joints)} 个。\n"
            "请检查 RG2 是否正确挂在机械臂末端。"
        )

    motor_joint = find_rg2_motor_joint(sim)

    print("\n========== RG2 主驱动关节 ==========")
    print(f"路径：{get_full_path(sim, motor_joint)}")
    print(f"初始位置：{sim.getJointPosition(motor_joint):.6f}")
    print(f"初始模式：{get_joint_mode_compat(sim, motor_joint)}")

    # 保存机械臂状态。将机械臂关节暂时设置为运动学模式，
    # 避免启动动力学后机械臂因为原模型约束问题而散架。
    saved_arm_modes = {
        joint: get_joint_mode_compat(sim, joint)
        for joint in arm_joints
    }
    saved_arm_positions = {
        joint: float(sim.getJointPosition(joint))
        for joint in arm_joints
    }

    initial_motor_position = float(
        sim.getJointPosition(motor_joint)
    )

    simulation_started = False

    try:
        print("\n正在将机械臂七个关节临时切换为运动学模式……")

        for joint in arm_joints:
            set_joint_mode_compat(
                sim,
                joint,
                sim.jointmode_kinematic,
            )
            sim.setJointPosition(joint, saved_arm_positions[joint])

        print("机械臂关节已锁定在当前姿态。")

        # 清除旧程序可能遗留的 float 类型同名信号。
        clear_wrong_float_signal(sim)

        # 先发送关闭命令。默认 RG2 脚本中：0=关闭，1=打开。
        # 这样启动仿真后先归一化到明确的关闭状态。
        set_rg2_signal(sim, CLOSE_SIGNAL_VALUE)

        # 外部 Python 以 stepping 模式逐步推进仿真。
        client.setStepping(True)
        sim.startSimulation()
        simulation_started = True

        # 1. 先归一化到关闭端，消除初始位置不确定性。
        normalized_closed_position = run_steps(
            client,
            sim,
            motor_joint,
            CLOSE_STEPS,
            "归一化到关闭状态",
        )

        # 2. 打开：int32 信号设为 1。
        set_rg2_signal(sim, OPEN_SIGNAL_VALUE)
        opened_position = run_steps(
            client,
            sim,
            motor_joint,
            OPEN_STEPS,
            "打开夹爪",
        )

        # 保持打开：继续保持信号为 1，而不是设为 0。
        run_steps(
            client,
            sim,
            motor_joint,
            HOLD_STEPS,
            "保持打开状态",
        )

        # 3. 关闭：int32 信号设为 0。
        set_rg2_signal(sim, CLOSE_SIGNAL_VALUE)
        closed_position = run_steps(
            client,
            sim,
            motor_joint,
            CLOSE_STEPS,
            "关闭夹爪",
        )

        # 保持关闭：继续保持信号为 0。
        run_steps(
            client,
            sim,
            motor_joint,
            HOLD_STEPS,
            "保持关闭状态",
        )

        open_change = opened_position - normalized_closed_position
        close_change = closed_position - opened_position

        print("\n========== 开合测试结果 ==========")
        print(f"脚本启动前主关节位置：{initial_motor_position:.6f}")
        print(f"归一化关闭后位置：{normalized_closed_position:.6f}")
        print(f"打开后主关节位置：{opened_position:.6f}")
        print(f"再次关闭后位置：{closed_position:.6f}")
        print(f"打开阶段变化量：{open_change:.6f}")
        print(f"关闭阶段变化量：{close_change:.6f}")

        opened_ok = abs(open_change) > JOINT_MOVE_EPS
        closed_ok = abs(close_change) > JOINT_MOVE_EPS
        reverse_ok = open_change * close_change < 0.0

        if opened_ok and closed_ok and reverse_ok:
            print("\n结果：RG2 已完成完整的双向打开和关闭测试。")
        else:
            print("\n结果：RG2 仍未表现出完整双向开合。")
            print(
                "请打开 RG2 的 simulation script，确认其中读取的是：\n"
                "sim.getInt32Signal('RG2_open')\n"
                "并确认 1=打开、0=关闭。"
            )

    finally:
        try:
            set_rg2_signal(sim, 0.0)
        except Exception:
            pass

        if simulation_started:
            try:
                sim.stopSimulation()
                wait_until_stopped(client, sim)
            except Exception as exc:
                print(f"停止仿真时出现提示：{exc}")

        # 停止后恢复机械臂原模式和原关节角。
        if sim.getSimulationState() == sim.simulation_stopped:
            print("\n正在恢复机械臂原有关节模式和姿态……")

            for joint in arm_joints:
                set_joint_mode_compat(
                    sim,
                    joint,
                    saved_arm_modes[joint],
                )
                sim.setJointPosition(
                    joint,
                    saved_arm_positions[joint],
                )

            print("机械臂状态已恢复。")

        try:
            client.setStepping(False)
        except Exception:
            pass


if __name__ == "__main__":
    main()
