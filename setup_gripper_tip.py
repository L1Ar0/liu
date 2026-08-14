from __future__ import annotations

from typing import Any

from remote_session import RemoteAPIClient


FLANGE_TIP_PATH = "/iiwa_tip"
GRIPPER_TIP_PATH = "/gripper_tip"


def get_object_or_raise(
    sim: Any,
    path: str,
) -> int:
    """读取场景对象，不存在时抛出明确错误。"""

    try:
        return int(sim.getObject(path))
    except Exception as exc:
        raise RuntimeError(
            f"找不到场景对象：{path}"
        ) from exc


def get_full_path(
    sim: Any,
    handle: int,
) -> str:
    """读取对象完整路径。"""

    try:
        return str(
            sim.getObjectAlias(handle, 2)
        )
    except Exception:
        return str(
            sim.getObjectAlias(handle)
        )


def print_parent_chain(
    sim: Any,
    handle: int,
) -> None:
    """输出对象的父级链，检查其是否挂在夹爪下。"""

    print("\n========== gripper_tip 父级链 ==========")

    current = handle

    while current != -1:
        print(get_full_path(sim, current))
        current = int(
            sim.getObjectParent(current)
        )


def main() -> None:
    print("正在连接 CoppeliaSim……")

    client = RemoteAPIClient()
    sim = client.require("sim")

    print("连接成功。")

    if (
        sim.getSimulationState()
        != sim.simulation_stopped
    ):
        raise RuntimeError(
            "请先停止仿真，再运行本程序。"
        )

    flange_tip = get_object_or_raise(
        sim,
        FLANGE_TIP_PATH,
    )

    gripper_tip = get_object_or_raise(
        sim,
        GRIPPER_TIP_PATH,
    )

    if (
        sim.getObjectType(flange_tip)
        != sim.sceneobject_dummy
    ):
        raise RuntimeError(
            "/iiwa_tip 不是 Dummy 对象。"
        )

    if (
        sim.getObjectType(gripper_tip)
        != sim.sceneobject_dummy
    ):
        raise RuntimeError(
            "/gripper_tip 不是 Dummy 对象。"
        )

    # 保存 gripper_tip 当前的位置。
    gripper_position = sim.getObjectPosition(
        gripper_tip,
        sim.handle_world,
    )

    # 读取法兰 tip 的完整姿态。
    flange_pose = list(
        sim.getObjectPose(
            flange_tip,
            sim.handle_world,
        )
    )

    # 将位置替换成 gripper_tip 已经手动设置的位置，
    # 方向保持与 iiwa_tip 完全一致。
    flange_pose[0] = gripper_position[0]
    flange_pose[1] = gripper_position[1]
    flange_pose[2] = gripper_position[2]

    sim.setObjectPose(
        gripper_tip,
        flange_pose,
        sim.handle_world,
    )

    print("\ngripper_tip 的方向已与 iiwa_tip 对齐。")

    pose = sim.getObjectPose(
        gripper_tip,
        sim.handle_world,
    )

    print("\n========== gripper_tip 位姿 ==========")

    print(
        f"位置：X={pose[0]:.5f} m，"
        f"Y={pose[1]:.5f} m，"
        f"Z={pose[2]:.5f} m"
    )

    print(
        f"四元数："
        f"qx={pose[3]:.6f}，"
        f"qy={pose[4]:.6f}，"
        f"qz={pose[5]:.6f}，"
        f"qw={pose[6]:.6f}"
    )

    print_parent_chain(
        sim,
        gripper_tip,
    )

    print(
        "\n请检查父级链中是否包含 RG2。"
    )

    print(
        "完成后保存 CoppeliaSim 场景。"
    )


if __name__ == "__main__":
    main()
