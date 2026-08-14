from __future__ import annotations

import math
import time
from typing import Any, Sequence

from remote_session import RemoteAPIClient


# ============================================================
# 一、场景对象名称
# ============================================================

# 必须存在的两个 Dummy。
TIP_PATH = "/iiwa_tip"
TARGET_PATH = "/iiwa_target"

# 是否自动把 target 放到方块上方。
#
# False：
#   直接使用你在 CoppeliaSim 里设置好的 iiwa_target
#   位置和方向。
#
# True：
#   自动读取 target_cube 的位置，将 target 放在方块上方。
AUTO_PLACE_TARGET_ABOVE_CUBE = False

CUBE_PATH = "/target_cube"

# 只有 AUTO_PLACE_TARGET_ABOVE_CUBE=True 时使用。
CUBE_HALF_HEIGHT_M = 0.025
PREGRASP_CLEARANCE_M = 0.12


# ============================================================
# 二、IK 搜索与运动参数
# ============================================================

# 搜索 IK 解的最长时间。
IK_SEARCH_TIME_SECONDS = 10.0

# 最终关节至少离边界多少度。
FINAL_LIMIT_MARGIN_DEG = 5.0

# IK 接近边界多少度时开始主动回避。
IK_AVOID_MARGIN_DEG = 10.0

# IK 单次迭代关节最大变化。
IK_MAX_STEP_DEG = 3.0

# 平滑运动插值步数。
MOTION_STEPS = 180

# 每一步的显示等待时间。
MOTION_DELAY_SECONDS = 0.015

# 最终成功判定阈值。
POSITION_TOLERANCE_M = 0.003
ORIENTATION_TOLERANCE_DEG = 2.0


# ============================================================
# 三、对象读取与层级检查
# ============================================================

def get_object_or_raise(
    sim: Any,
    object_path: str,
) -> int:
    """根据绝对路径读取场景对象。"""

    handle = sim.getObject(
        object_path,
        {"noError": True},
    )

    if handle == -1:
        raise RuntimeError(
            f"场景中找不到对象：{object_path}\n"
            "请检查左侧对象树中的名称是否完全一致。"
        )

    return handle


def object_type_name(
    sim: Any,
    object_handle: int,
) -> str:
    """将对象类型编号转换成便于查看的名称。"""

    object_type = sim.getObjectType(object_handle)

    type_map = {
        sim.sceneobject_shape: "shape",
        sim.sceneobject_joint: "joint",
        sim.sceneobject_dummy: "dummy",
        sim.sceneobject_camera: "camera",
        sim.sceneobject_visionsensor: "vision sensor",
        sim.sceneobject_forcesensor: "force sensor",
    }

    return type_map.get(
        object_type,
        f"other({object_type})",
    )


def get_full_path(
    sim: Any,
    object_handle: int,
) -> str:
    """读取对象完整层级路径。"""

    try:
        return sim.getObjectAlias(
            object_handle,
            2,
        )
    except Exception:
        return sim.getObjectAlias(
            object_handle
        )


def print_parent_chain(
    sim: Any,
    tip_handle: int,
) -> None:
    """打印从 iiwa_tip 到场景根节点的父级链。"""

    print("\n========== iiwa_tip 父级链 ==========")

    current = tip_handle
    level = 0

    while current != -1:
        indent = "  " * level

        print(
            f"{indent}↑ "
            f"{get_full_path(sim, current)} "
            f"[{object_type_name(sim, current)}]"
        )

        current = sim.getObjectParent(current)
        level += 1


def get_joints_from_tip(
    sim: Any,
    tip_handle: int,
) -> list[int]:
    """
    从 iiwa_tip 开始向父级回溯。

    回溯得到的顺序：
        joint7 → joint6 → ... → joint1

    最后反转为：
        joint1 → joint2 → ... → joint7
    """

    joints_from_tip: list[int] = []

    current = tip_handle

    while current != -1:
        if (
            sim.getObjectType(current)
            == sim.sceneobject_joint
        ):
            joints_from_tip.append(current)

        current = sim.getObjectParent(current)

    joints = list(reversed(joints_from_tip))

    return joints


def print_joint_chain(
    sim: Any,
    joints: Sequence[int],
) -> None:
    """打印检测到的七关节运动链。"""

    print("\n========== 检测到的运动链 ==========")

    for index, joint in enumerate(
        joints,
        start=1,
    ):
        print(
            f"J{index}: "
            f"{get_full_path(sim, joint)}"
        )


# ============================================================
# 四、IK 世界句柄映射
# ============================================================

def get_ik_handle(
    sim_to_ik_map: Any,
    scene_handle: int,
) -> int:
    """
    将场景世界的对象句柄转换成 IK 世界句柄。

    CoppeliaSim 的场景世界和 simIK 世界拥有不同句柄。
    """

    # 常见情况：返回列表，可直接用场景句柄作为索引。
    try:
        value = sim_to_ik_map[scene_handle]

        if value is not None:
            return int(value)
    except (IndexError, KeyError, TypeError):
        pass

    # 有些客户端可能解码为字典。
    if isinstance(sim_to_ik_map, dict):
        if scene_handle in sim_to_ik_map:
            return int(
                sim_to_ik_map[scene_handle]
            )

        string_key = str(scene_handle)

        if string_key in sim_to_ik_map:
            return int(
                sim_to_ik_map[string_key]
            )

    raise RuntimeError(
        f"无法将场景句柄 {scene_handle} "
        "转换为 IK 世界句柄。"
    )


# ============================================================
# 五、关节范围检查
# ============================================================

def get_joint_limits(
    sim: Any,
    joints: Sequence[int],
) -> list[tuple[float, float]]:
    """
    读取每个关节的下限和上限。

    CoppeliaSim 返回：
        interval[0] = 下限
        interval[1] = 关节总范围

    所以：
        上限 = 下限 + 总范围
    """

    limits: list[tuple[float, float]] = []

    print("\n========== 关节物理范围 ==========")

    for index, joint in enumerate(
        joints,
        start=1,
    ):
        cyclic, interval = sim.getJointInterval(
            joint
        )

        if cyclic:
            raise RuntimeError(
                f"J{index} 被设置为循环关节。\n"
                "KUKA iiwa 的关节通常应具有明确角度范围，"
                "请检查该关节属性。"
            )

        lower = float(interval[0])
        upper = lower + float(interval[1])

        current = float(
            sim.getJointPosition(joint)
        )

        clearance = min(
            current - lower,
            upper - current,
        )

        limits.append((lower, upper))

        print(
            f"J{index}: "
            f"[{math.degrees(lower):8.2f}°, "
            f"{math.degrees(upper):8.2f}°]  "
            f"当前={math.degrees(current):8.2f}°  "
            f"边界余量={math.degrees(clearance):6.2f}°"
        )

    return limits


def validate_config(
    config: Sequence[float],
    limits: Sequence[tuple[float, float]],
    tolerance: float = 1e-7,
) -> None:
    """检查一组关节角是否越界。"""

    if len(config) != len(limits):
        raise ValueError(
            "关节配置数量和关节限制数量不一致。"
        )

    for index, (
        value,
        limit,
    ) in enumerate(
        zip(config, limits),
        start=1,
    ):
        lower, upper = limit

        if (
            value < lower - tolerance
            or value > upper + tolerance
        ):
            raise RuntimeError(
                f"J{index} 关节越界："
                f"{math.degrees(value):.3f}°；"
                f"允许范围为 "
                f"[{math.degrees(lower):.3f}°, "
                f"{math.degrees(upper):.3f}°]"
            )


def minimum_limit_clearance(
    config: Sequence[float],
    limits: Sequence[tuple[float, float]],
) -> float:
    """计算整组关节中最小的边界余量。"""

    return min(
        min(
            value - lower,
            upper - value,
        )
        for value, (lower, upper) in zip(
            config,
            limits,
        )
    )


def print_joint_config(
    title: str,
    config: Sequence[float],
    limits: Sequence[tuple[float, float]],
) -> None:
    """打印关节角及安全余量。"""

    print(f"\n========== {title} ==========")

    for index, (
        value,
        limit,
    ) in enumerate(
        zip(config, limits),
        start=1,
    ):
        lower, upper = limit

        clearance = min(
            value - lower,
            upper - value,
        )

        print(
            f"J{index}: "
            f"{math.degrees(value):8.2f}°  "
            f"边界余量="
            f"{math.degrees(clearance):6.2f}°"
        )


# ============================================================
# 六、IK 解处理
# ============================================================

def normalize_solutions(
    raw_solutions: Any,
    joint_count: int,
) -> list[list[float]]:
    """统一 findConfigs 返回结果格式。"""

    if raw_solutions is None:
        return []

    if len(raw_solutions) == 0:
        return []

    first = raw_solutions[0]

    # 正常情况：二维列表。
    if isinstance(first, (list, tuple)):
        return [
            [
                float(value)
                for value in solution
            ]
            for solution in raw_solutions
        ]

    # 兼容某些客户端返回扁平数组。
    flat = [
        float(value)
        for value in raw_solutions
    ]

    if len(flat) % joint_count != 0:
        raise RuntimeError(
            "findConfigs 返回的数据长度异常。"
        )

    solutions: list[list[float]] = []

    for start in range(
        0,
        len(flat),
        joint_count,
    ):
        solutions.append(
            flat[start:start + joint_count]
        )

    return solutions


def choose_best_solution(
    solutions: Sequence[Sequence[float]],
    current_config: Sequence[float],
    limits: Sequence[tuple[float, float]],
) -> list[float] | None:
    """
    选择较安全的 IK 解。

    条件：
    1. 所有关节都不越界；
    2. 离边界至少保留 FINAL_LIMIT_MARGIN_DEG；
    3. 尽量靠近当前姿态；
    4. 尽量远离关节极限。
    """

    required_margin = math.radians(
        FINAL_LIMIT_MARGIN_DEG
    )

    candidates: list[
        tuple[float, list[float]]
    ] = []

    for raw_solution in solutions:
        solution = [
            float(value)
            for value in raw_solution
        ]

        if len(solution) != len(current_config):
            continue

        try:
            validate_config(
                solution,
                limits,
            )
        except RuntimeError:
            continue

        clearance = minimum_limit_clearance(
            solution,
            limits,
        )

        if clearance < required_margin:
            continue

        movement_score = 0.0

        for (
            target_value,
            current_value,
            limit,
        ) in zip(
            solution,
            current_config,
            limits,
        ):
            lower, upper = limit

            joint_range = max(
                upper - lower,
                1e-6,
            )

            difference = (
                target_value - current_value
            ) / joint_range

            movement_score += difference**2

        # 越靠近边界，惩罚越大。
        limit_penalty = (
            0.05
            / (
                clearance
                + math.radians(0.5)
            )
        )

        score = (
            movement_score
            + limit_penalty
        )

        candidates.append(
            (score, solution)
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: item[0]
    )

    return candidates[0][1]


# ============================================================
# 七、机械臂平滑运动
# ============================================================

def move_joints_smoothly(
    sim: Any,
    joints: Sequence[int],
    target_config: Sequence[float],
    limits: Sequence[tuple[float, float]],
) -> None:
    """
    在仿真停止状态下进行关节空间平滑插值。

    不启动动力学，所以不会出现机械臂散架。
    """

    start_config = [
        float(sim.getJointPosition(joint))
        for joint in joints
    ]

    validate_config(
        start_config,
        limits,
    )

    validate_config(
        target_config,
        limits,
    )

    for step_index in range(
        MOTION_STEPS + 1
    ):
        progress = (
            step_index / MOTION_STEPS
        )

        # Smoothstep 插值。
        alpha = (
            3.0 * progress**2
            - 2.0 * progress**3
        )

        current_config = [
            start
            + alpha * (target - start)
            for start, target in zip(
                start_config,
                target_config,
            )
        ]

        validate_config(
            current_config,
            limits,
        )

        for joint, value in zip(
            joints,
            current_config,
        ):
            sim.setJointPosition(
                joint,
                value,
            )

        time.sleep(
            MOTION_DELAY_SECONDS
        )


# ============================================================
# 八、目标位置设置
# ============================================================

def place_target_above_cube(
    sim: Any,
    target_handle: int,
    cube_handle: int,
) -> None:
    """
    将 iiwa_target 移动到方块正上方。

    仅修改目标位置，不修改目标方向。
    """

    cube_position = sim.getObjectPosition(
        cube_handle,
        sim.handle_world,
    )

    target_pose = sim.getObjectPose(
        target_handle,
        sim.handle_world,
    )

    target_pose[0] = cube_position[0]
    target_pose[1] = cube_position[1]

    target_pose[2] = (
        cube_position[2]
        + CUBE_HALF_HEIGHT_M
        + PREGRASP_CLEARANCE_M
    )

    sim.setObjectPose(
        target_handle,
        target_pose,
        sim.handle_world,
    )

    print("\n已自动设置预抓取目标：")

    print(
        f"方块中心："
        f"X={cube_position[0]:.4f} m，"
        f"Y={cube_position[1]:.4f} m，"
        f"Z={cube_position[2]:.4f} m"
    )

    print(
        f"目标位置："
        f"X={target_pose[0]:.4f} m，"
        f"Y={target_pose[1]:.4f} m，"
        f"Z={target_pose[2]:.4f} m"
    )

    print(
        "iiwa_target 原来的方向保持不变。"
    )


# ============================================================
# 九、位姿读取与误差计算
# ============================================================

def print_pose(
    sim: Any,
    object_handle: int,
    title: str,
) -> None:
    """
    打印位置和四元数。

    pose 格式：
        [x, y, z, qx, qy, qz, qw]
    """

    pose = sim.getObjectPose(
        object_handle,
        sim.handle_world,
    )

    print(f"\n========== {title} ==========")

    print(
        f"位置："
        f"X={pose[0]:.5f} m，"
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


def calculate_position_error(
    sim: Any,
    tip_handle: int,
    target_handle: int,
) -> float:
    """计算 tip 和 target 的三维位置误差。"""

    tip_position = sim.getObjectPosition(
        tip_handle,
        sim.handle_world,
    )

    target_position = sim.getObjectPosition(
        target_handle,
        sim.handle_world,
    )

    return math.sqrt(
        sum(
            (tip_value - target_value) ** 2
            for tip_value, target_value in zip(
                tip_position,
                target_position,
            )
        )
    )


def normalize_quaternion(
    quaternion: Sequence[float],
) -> list[float]:
    """归一化四元数。"""

    norm = math.sqrt(
        sum(
            value**2
            for value in quaternion
        )
    )

    if norm < 1e-12:
        raise RuntimeError(
            "检测到无效的零四元数。"
        )

    return [
        value / norm
        for value in quaternion
    ]


def calculate_orientation_error(
    sim: Any,
    tip_handle: int,
    target_handle: int,
) -> float:
    """
    计算 tip 与 target 的完整方向误差。

    返回值单位：弧度。
    """

    tip_pose = sim.getObjectPose(
        tip_handle,
        sim.handle_world,
    )

    target_pose = sim.getObjectPose(
        target_handle,
        sim.handle_world,
    )

    tip_quaternion = normalize_quaternion(
        tip_pose[3:7]
    )

    target_quaternion = normalize_quaternion(
        target_pose[3:7]
    )

    dot_product = sum(
        tip_value * target_value
        for tip_value, target_value in zip(
            tip_quaternion,
            target_quaternion,
        )
    )

    # q 和 -q 表示同一个方向。
    dot_product = abs(dot_product)

    dot_product = max(
        -1.0,
        min(1.0, dot_product),
    )

    return 2.0 * math.acos(
        dot_product
    )


# ============================================================
# 十、主程序
# ============================================================

def main() -> None:
    print("正在连接 CoppeliaSim……")

    client = RemoteAPIClient()

    sim = client.require("sim")
    simIK = client.require("simIK")

    print("连接成功。")

    # --------------------------------------------------------
    # 1. 检查仿真状态
    # --------------------------------------------------------

    if (
        sim.getSimulationState()
        != sim.simulation_stopped
    ):
        raise RuntimeError(
            "请先点击 CoppeliaSim 顶部的停止按钮。\n"
            "本程序必须在仿真停止状态下运行。"
        )

    # --------------------------------------------------------
    # 2. 读取 tip 和 target
    # --------------------------------------------------------

    tip = get_object_or_raise(
        sim,
        TIP_PATH,
    )

    target = get_object_or_raise(
        sim,
        TARGET_PATH,
    )

    if (
        sim.getObjectType(tip)
        != sim.sceneobject_dummy
    ):
        raise RuntimeError(
            f"{TIP_PATH} 不是 Dummy 对象。"
        )

    if (
        sim.getObjectType(target)
        != sim.sceneobject_dummy
    ):
        raise RuntimeError(
            f"{TARGET_PATH} 不是 Dummy 对象。"
        )

    # --------------------------------------------------------
    # 3. 从 tip 向上寻找七个关节
    # --------------------------------------------------------

    print_parent_chain(
        sim,
        tip,
    )

    joints = get_joints_from_tip(
        sim,
        tip,
    )

    print(
        f"\n从 iiwa_tip 父级链中找到 "
        f"{len(joints)} 个关节。"
    )

    if len(joints) != 7:
        raise RuntimeError(
            "\n无法从 iiwa_tip 回溯出完整的七关节运动链。\n"
            f"实际找到：{len(joints)} 个关节。\n\n"
            "正确层级应类似：\n"
            "base\n"
            "└── joint1\n"
            "    └── link1\n"
            "        └── joint2\n"
            "            └── ...\n"
            "                └── joint7\n"
            "                    └── flange/link7\n"
            "                        └── iiwa_tip\n\n"
            "请检查 iiwa_tip 是否真正挂在机械臂末端，"
            "而不是独立的顶层对象。"
        )

    print_joint_chain(
        sim,
        joints,
    )

    # Joint 1 的父对象是 IK 运动链基座。
    ik_base = sim.getObjectParent(
        joints[0]
    )

    if ik_base == -1:
        print(
            "\nJoint 1 没有父对象，"
            "将世界坐标系作为 IK 基座。"
        )
    else:
        print(
            "\nIK 基座："
            f"{get_full_path(sim, ik_base)}"
        )

    # --------------------------------------------------------
    # 4. 可选：自动设置 target 到方块上方
    # --------------------------------------------------------

    if AUTO_PLACE_TARGET_ABOVE_CUBE:
        cube = get_object_or_raise(
            sim,
            CUBE_PATH,
        )

        place_target_above_cube(
            sim,
            target,
            cube,
        )
    else:
        print(
            "\nAUTO_PLACE_TARGET_ABOVE_CUBE=False"
        )

        print(
            "程序将直接使用场景中当前 "
            "iiwa_target 的位置和方向。"
        )

    print_pose(
        sim,
        tip,
        "运动前 iiwa_tip 位姿",
    )

    print_pose(
        sim,
        target,
        "iiwa_target 目标位姿",
    )

    # --------------------------------------------------------
    # 5. 读取关节范围
    # --------------------------------------------------------

    limits = get_joint_limits(
        sim,
        joints,
    )

    current_config = [
        float(sim.getJointPosition(joint))
        for joint in joints
    ]

    validate_config(
        current_config,
        limits,
    )

    # --------------------------------------------------------
    # 6. 创建 IK 环境
    # --------------------------------------------------------

    ik_environment = simIK.createEnvironment()
    ik_group = simIK.createGroup(
        ik_environment
    )

    try:
        # constraint_pose 同时约束：
        # x、y、z 和完整方向 alpha、beta、gamma。
        (
            ik_element,
            sim_to_ik_map,
            _,
        ) = simIK.addElementFromScene(
            ik_environment,
            ik_group,
            ik_base,
            tip,
            target,
            simIK.constraint_pose,
        )

        # 将七个场景关节句柄转换成 IK 世界句柄。
        ik_joints = [
            get_ik_handle(
                sim_to_ik_map,
                joint,
            )
            for joint in joints
        ]

        # ----------------------------------------------------
        # 7. 配置 IK 求解器
        # ----------------------------------------------------

        simIK.setGroupCalculation(
            ik_environment,
            ik_group,
            simIK.method_damped_least_squares,
            0.05,
            120,
        )

        group_flags = simIK.getGroupFlags(
            ik_environment,
            ik_group,
        )

        group_flags |= (
            simIK.group_avoidlimits
            | simIK.group_restoreonbadlintol
            | simIK.group_restoreonbadangtol
        )

        simIK.setGroupFlags(
            ik_environment,
            ik_group,
            group_flags,
        )

        simIK.setElementPrecision(
            ik_environment,
            ik_group,
            ik_element,
            [
                0.001,
                math.radians(1.0),
            ],
        )

        # 将场景中的真实关节限制复制到 IK 世界。
        for (
            ik_joint,
            limit,
        ) in zip(
            ik_joints,
            limits,
        ):
            lower, upper = limit

            simIK.setJointInterval(
                ik_environment,
                ik_joint,
                False,
                [
                    lower,
                    upper - lower,
                ],
            )

            simIK.setJointLimitMargin(
                ik_environment,
                ik_joint,
                math.radians(
                    IK_AVOID_MARGIN_DEG
                ),
            )

            simIK.setJointMaxStepSize(
                ik_environment,
                ik_joint,
                math.radians(
                    IK_MAX_STEP_DEG
                ),
            )

        # 将当前场景状态同步到 IK 世界。
        simIK.syncFromSim(
            ik_environment,
            [ik_group],
        )

        # ----------------------------------------------------
        # 8. 搜索多个完整位姿 IK 解
        # ----------------------------------------------------

        print(
            "\n正在搜索满足位置、方向和关节范围的 "
            "IK 解……"
        )

        raw_solutions = simIK.findConfigs(
            ik_environment,
            ik_group,
            ik_joints,
            {
                "maxDist": 1.5,
                "maxTime": IK_SEARCH_TIME_SECONDS,
                "findMultiple": True,
                "pMetric": [
                    1.0,
                    1.0,
                    1.0,
                    0.25,
                ],
                "cMetric": [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ],
            },
        )

        solutions = normalize_solutions(
            raw_solutions,
            len(joints),
        )

        print(
            f"搜索到 {len(solutions)} 组候选解。"
        )

        best_solution = choose_best_solution(
            solutions,
            current_config,
            limits,
        )

        if best_solution is None:
            raise RuntimeError(
                "\n没有找到同时满足以下条件的 IK 解：\n"
                "1. iiwa_tip 与 iiwa_target 位置对齐；\n"
                "2. iiwa_tip 与 iiwa_target 方向对齐；\n"
                "3. 所有关节均在物理范围内；\n"
                f"4. 每个关节至少保留 "
                f"{FINAL_LIMIT_MARGIN_DEG:.1f}° 安全余量。\n\n"
                "可以尝试：\n"
                "1. 把 iiwa_target 稍微靠近机械臂；\n"
                "2. 将 target 高度提高 2～5 cm；\n"
                "3. 稍微改变 target 绕竖直轴的角度；\n"
                "4. 临时将 FINAL_LIMIT_MARGIN_DEG 改为 2.0。"
            )

        print_joint_config(
            "选择的安全 IK 解",
            best_solution,
            limits,
        )

        # ----------------------------------------------------
        # 9. 平滑执行
        # ----------------------------------------------------

        print(
            "\n开始移动机械臂，使 iiwa_tip "
            "与 iiwa_target 完整对齐……"
        )

        move_joints_smoothly(
            sim,
            joints,
            best_solution,
            limits,
        )

        print("机械臂运动完成。")

        # ----------------------------------------------------
        # 10. 最终验证
        # ----------------------------------------------------

        final_config = [
            float(sim.getJointPosition(joint))
            for joint in joints
        ]

        validate_config(
            final_config,
            limits,
        )

        position_error = calculate_position_error(
            sim,
            tip,
            target,
        )

        orientation_error = (
            calculate_orientation_error(
                sim,
                tip,
                target,
            )
        )

        minimum_clearance = (
            minimum_limit_clearance(
                final_config,
                limits,
            )
        )

        print_pose(
            sim,
            tip,
            "运动后 iiwa_tip 位姿",
        )

        print_pose(
            sim,
            target,
            "iiwa_target 位姿",
        )

        print_joint_config(
            "最终关节角",
            final_config,
            limits,
        )

        print(
            "\n========== 最终对齐结果 =========="
        )

        print(
            f"位置误差："
            f"{position_error * 1000:.3f} mm"
        )

        print(
            f"完整方向误差："
            f"{math.degrees(orientation_error):.3f}°"
        )

        print(
            f"最小关节边界余量："
            f"{math.degrees(minimum_clearance):.3f}°"
        )

        position_success = (
            position_error
            <= POSITION_TOLERANCE_M
        )

        orientation_success = (
            orientation_error
            <= math.radians(
                ORIENTATION_TOLERANCE_DEG
            )
        )

        clearance_success = (
            minimum_clearance
            >= math.radians(
                FINAL_LIMIT_MARGIN_DEG
            )
        )

        print("\n========== 判定 ==========")

        print(
            "位置对齐："
            + (
                "通过"
                if position_success
                else "未通过"
            )
        )

        print(
            "方向对齐："
            + (
                "通过"
                if orientation_success
                else "未通过"
            )
        )

        print(
            "关节安全余量："
            + (
                "通过"
                if clearance_success
                else "未通过"
            )
        )

        if (
            position_success
            and orientation_success
            and clearance_success
        ):
            print(
                "\n结果：iiwa_tip 已与 iiwa_target "
                "完成完整位置和方向对齐。"
            )
        else:
            print(
                "\n结果：机械臂已经移动到候选解，"
                "但部分误差没有达到设置的阈值。"
            )

    finally:
        simIK.eraseEnvironment(
            ik_environment
        )


if __name__ == "__main__":
    main()
