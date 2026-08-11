from __future__ import annotations

import time
from typing import Any

from coppeliasim_zmqremoteapi_client import RemoteAPIClient


# ============================================================
# 配置
# ============================================================

# 如果场景中的夹爪根对象不是 /RG2，请改成实际路径，
# 例如 /RG2#0 或 /RG2_gripper。
RG2_ROOT_PATH = "/RG2"

MOTION_STEPS = 80
STEP_DELAY_SECONDS = 0.02

# 测试结束后是否让夹爪保持打开。
LEAVE_GRIPPER_OPEN = True


# ============================================================
# 基础工具
# ============================================================

def get_object_or_raise(sim: Any, path: str) -> int:
    try:
        return int(sim.getObject(path))
    except Exception as exc:
        raise RuntimeError(
            f"找不到对象：{path}\n"
            "请检查 CoppeliaSim 左侧对象树中的名称。"
        ) from exc


def get_full_path(sim: Any, handle: int) -> str:
    try:
        return str(sim.getObjectAlias(handle, 2))
    except Exception:
        return str(sim.getObjectAlias(handle))


def get_joint_mode_compat(sim: Any, joint: int) -> int:
    """兼容 getJointMode 返回 mode 或 (mode, options) 的版本。"""
    result = sim.getJointMode(joint)

    if isinstance(result, (tuple, list)):
        if not result:
            raise RuntimeError(
                f"getJointMode({joint}) 返回了空结果。"
            )
        return int(result[0])

    return int(result)


def set_joint_mode_compat(
    sim: Any,
    joint: int,
    mode: int,
) -> None:
    """兼容 setJointMode 的两参数和三参数形式。"""
    try:
        sim.setJointMode(joint, mode)
    except TypeError:
        sim.setJointMode(joint, mode, 0)


def joint_mode_name(sim: Any, mode: int) -> str:
    if mode == sim.jointmode_kinematic:
        return "kinematic"
    if mode == sim.jointmode_dependent:
        return "dependent"
    if mode == sim.jointmode_dynamic:
        return "dynamic"
    return f"unknown({mode})"


def joint_type_name(sim: Any, joint_type: int) -> str:
    if joint_type == sim.joint_revolute:
        return "revolute"
    if joint_type == sim.joint_prismatic:
        return "prismatic"
    if joint_type == sim.joint_spherical:
        return "spherical"
    return f"unknown({joint_type})"


# ============================================================
# RG2 关节查找
# ============================================================

def find_open_close_joint(
    sim: Any,
    rg2_root: int,
) -> tuple[int, list[int]]:
    """
    枚举 RG2 内部关节，并寻找名称包含 openClose 的
    非球形主驱动关节。

    关键修复：球形关节不能调用 sim.getJointPosition，
    因此只对 revolute/prismatic 关节读取单一位置值。
    """

    joints = [
        int(joint)
        for joint in sim.getObjectsInTree(
            rg2_root,
            sim.sceneobject_joint,
            0,
        )
    ]

    if not joints:
        raise RuntimeError(
            "RG2 层级下没有找到任何关节。"
        )

    print("\n========== RG2 内部关节 ==========")

    main_joint = -1

    for index, joint in enumerate(joints, start=1):
        alias = str(sim.getObjectAlias(joint))
        joint_type = int(sim.getJointType(joint))
        mode = get_joint_mode_compat(sim, joint)

        if joint_type == sim.joint_spherical:
            position_text = "N/A（球形关节）"
        else:
            position = float(sim.getJointPosition(joint))
            position_text = f"{position:.6f}"

        print(
            f"{index:2d}. "
            f"{alias:35s} "
            f"type={joint_type_name(sim, joint_type):10s} "
            f"mode={joint_mode_name(sim, mode):10s} "
            f"position={position_text}"
        )

        normalized_name = (
            alias.lower()
            .replace("_", "")
            .replace("-", "")
            .replace(" ", "")
        )

        name_matches = (
            "openclosejoint" in normalized_name
            or "openclose" in normalized_name
        )

        is_single_dof = joint_type in (
            sim.joint_revolute,
            sim.joint_prismatic,
        )

        if name_matches and is_single_dof:
            main_joint = joint

    if main_joint == -1:
        raise RuntimeError(
            "\n没有找到名称包含 openClose 的非球形主驱动关节。\n"
            "请查看上方列表，把实际主驱动关节名称发来。"
        )

    return main_joint, joints


def print_dependencies(
    sim: Any,
    joints: list[int],
) -> None:
    """打印非球形 dependent 关节的依赖关系。"""

    print("\n========== 从属关节关系 ==========")

    found_dependency = False

    for joint in joints:
        joint_type = int(sim.getJointType(joint))

        # 球形关节不是本次单自由度开合控制对象，直接跳过。
        if joint_type == sim.joint_spherical:
            continue

        mode = get_joint_mode_compat(sim, joint)

        if mode != sim.jointmode_dependent:
            continue

        found_dependency = True

        try:
            result = sim.getJointDependency(joint)

            master = int(result[0])
            offset = float(result[1])
            coefficient = float(result[2])

            joint_name = str(sim.getObjectAlias(joint))

            if master != -1:
                master_name = str(sim.getObjectAlias(master))
            else:
                master_name = "None"

            print(
                f"{joint_name} = "
                f"{offset:.6f} + "
                f"{coefficient:.6f} × "
                f"{master_name}"
            )

        except Exception as exc:
            print(
                f"{sim.getObjectAlias(joint)}："
                f"无法读取依赖关系，{exc}"
            )

    if not found_dependency:
        print("没有检测到非球形 dependent 关节。")


# ============================================================
# 运动学开合
# ============================================================

def animate_joint(
    sim: Any,
    joint: int,
    start: float,
    target: float,
    title: str,
) -> None:
    """在仿真停止状态下平滑设置主关节位置。"""

    print(f"\n开始：{title}")

    for step_index in range(MOTION_STEPS + 1):
        progress = step_index / MOTION_STEPS

        # Smoothstep 插值。
        alpha = 3.0 * progress**2 - 2.0 * progress**3
        position = start + alpha * (target - start)

        sim.setJointPosition(joint, position)

        if (
            step_index == 0
            or step_index == MOTION_STEPS
            or step_index % 20 == 0
        ):
            print(
                f"  step {step_index:3d}/{MOTION_STEPS}: "
                f"{position:.6f}"
            )

        time.sleep(STEP_DELAY_SECONDS)


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
            "请先停止仿真。\n"
            "本程序不会启动动力学仿真。"
        )

    rg2_root = get_object_or_raise(sim, RG2_ROOT_PATH)

    print(
        f"\nRG2 根对象：{get_full_path(sim, rg2_root)}"
    )

    main_joint, all_rg2_joints = find_open_close_joint(
        sim,
        rg2_root,
    )

    main_joint_name = str(sim.getObjectAlias(main_joint))
    main_joint_type = int(sim.getJointType(main_joint))

    print(
        f"\n检测到主驱动关节：{main_joint_name} "
        f"[{joint_type_name(sim, main_joint_type)}]"
    )

    print_dependencies(sim, all_rg2_joints)

    # 主关节必须是单自由度关节。
    if main_joint_type == sim.joint_spherical:
        raise RuntimeError(
            "检测到的主驱动关节是球形关节，"
            "不能作为 RG2 开合主关节。"
        )

    cyclic, interval = sim.getJointInterval(main_joint)

    if cyclic:
        raise RuntimeError(
            "openCloseJoint 被设置为循环关节，"
            "无法确定开合端点。"
        )

    lower_limit = float(interval[0])
    upper_limit = lower_limit + float(interval[1])
    current_position = float(sim.getJointPosition(main_joint))

    # 结合你之前的实测：-0.048 为关闭端，0 为打开端。
    closed_position = lower_limit
    open_position = upper_limit

    print("\n========== 主关节范围 ==========")
    print(f"下限：{lower_limit:.6f}")
    print(f"上限：{upper_limit:.6f}")
    print(f"当前位置：{current_position:.6f}")
    print(f"设定关闭位置：{closed_position:.6f}")
    print(f"设定打开位置：{open_position:.6f}")

    original_mode = get_joint_mode_compat(sim, main_joint)

    print(
        f"\n主关节原模式："
        f"{joint_mode_name(sim, original_mode)}"
    )

    set_joint_mode_compat(
        sim,
        main_joint,
        sim.jointmode_kinematic,
    )

    print("主关节已切换为 kinematic 模式。")

    current_position = float(sim.getJointPosition(main_joint))

    animate_joint(
        sim,
        main_joint,
        current_position,
        open_position,
        "运动学打开夹爪",
    )

    time.sleep(1.0)

    current_position = float(sim.getJointPosition(main_joint))

    animate_joint(
        sim,
        main_joint,
        current_position,
        closed_position,
        "运动学关闭夹爪",
    )

    time.sleep(1.0)

    if LEAVE_GRIPPER_OPEN:
        current_position = float(sim.getJointPosition(main_joint))

        animate_joint(
            sim,
            main_joint,
            current_position,
            open_position,
            "重新打开夹爪",
        )

    final_position = float(sim.getJointPosition(main_joint))

    print("\n========== 测试结果 ==========")
    print(f"主关节最终位置：{final_position:.6f}")
    print("测试期间未启动动力学仿真。")
    print("请观察 RG2 两根手指是否实际张合。")
    print("主驱动关节保持 kinematic 模式。")


if __name__ == "__main__":
    main()