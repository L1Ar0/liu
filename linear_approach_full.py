from __future__ import annotations

import math
import time
from typing import Any, Sequence

from coppeliasim_zmqremoteapi_client import RemoteAPIClient


# ============================================================
# 1. 场景对象与运行参数
# ============================================================

TIP_PATH = "/iiwa_tip"
TARGET_PATH = "/iiwa_target"

# 从预抓取点沿世界坐标系 -Z 方向下降的距离。
# 第一轮建议 0.03~0.05 m，确认安全后再增加。
APPROACH_DISTANCE_M = 0.05

# 测试完成后是否沿原路径返回预抓取点。
RETURN_TO_PREGRASP = True

# 若脚本开始时 tip 尚未与 target 对齐，
# 是否先自动完成预抓取对齐。
ALIGN_PREGRASP_FIRST = True

# IK 搜索参数。
IK_SEARCH_TIME_SECONDS = 10.0
FINAL_LIMIT_MARGIN_DEG = 5.0
IK_AVOID_MARGIN_DEG = 10.0
IK_MAX_STEP_DEG = 3.0

# 预抓取点的关节空间平滑运动参数。
ALIGN_MOTION_STEPS = 180
ALIGN_STEP_DELAY_SECONDS = 0.015

# 直线下降路径参数。
CARTESIAN_PATH_POINTS = 120
CARTESIAN_STEP_DELAY_SECONDS = 0.015

# 对齐成功判定阈值。
POSITION_TOLERANCE_M = 0.003
ORIENTATION_TOLERANCE_DEG = 2.0


# ============================================================
# 2. 场景对象与运动链
# ============================================================

def get_object_or_raise(sim: Any, path: str) -> int:
    """读取场景对象；找不到时给出清晰报错。"""

    try:
        return int(sim.getObject(path))
    except Exception as exc:
        raise RuntimeError(
            f"场景中找不到对象 {path}，"
            "请检查对象名称和路径。"
        ) from exc


def get_full_path(sim: Any, handle: int) -> str:
    """尽量返回对象的完整层级路径。"""

    try:
        return str(sim.getObjectAlias(handle, 2))
    except Exception:
        return str(sim.getObjectAlias(handle))


def get_joints_from_tip(sim: Any, tip: int) -> list[int]:
    """
    从 iiwa_tip 沿父级向上回溯关节。

    回溯顺序是 joint7 -> joint1，
    最后反转为 joint1 -> joint7。
    """

    reverse_joints: list[int] = []
    current = tip

    while current != -1:
        if sim.getObjectType(current) == sim.sceneobject_joint:
            reverse_joints.append(int(current))

        current = int(
            sim.getObjectParent(current)
        )

    return list(reversed(reverse_joints))


def get_ik_handle(
    mapping: Any,
    scene_handle: int,
) -> int:
    """把场景句柄转换成 simIK 环境中的句柄。"""

    if isinstance(mapping, dict):
        if scene_handle in mapping:
            return int(mapping[scene_handle])

        if str(scene_handle) in mapping:
            return int(
                mapping[str(scene_handle)]
            )

    if isinstance(mapping, (list, tuple)):
        # 常见格式：场景句柄直接作为列表索引。
        if 0 <= scene_handle < len(mapping):
            value = mapping[scene_handle]

            if isinstance(value, (int, float)):
                return int(value)

        # 兼容 [[sceneHandle, ikHandle], ...]。
        for item in mapping:
            if (
                isinstance(item, (list, tuple))
                and len(item) == 2
                and int(item[0]) == scene_handle
            ):
                return int(item[1])

    raise RuntimeError(
        f"无法把场景句柄 {scene_handle} "
        "映射到 simIK 环境。"
    )


# ============================================================
# 3. 关节限制与配置选择
# ============================================================

def get_joint_limits(
    sim: Any,
    joints: Sequence[int],
) -> list[tuple[float, float]]:
    """读取每个关节的下限与上限，单位为弧度。"""

    limits: list[tuple[float, float]] = []

    print("\n========== 关节范围 ==========")

    for index, joint in enumerate(
        joints,
        start=1,
    ):
        cyclic, interval = sim.getJointInterval(
            joint
        )

        if cyclic:
            raise RuntimeError(
                f"J{index} 被设置为循环关节，"
                "无法按有限安全范围检查。"
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
            f"边界余量="
            f"{math.degrees(clearance):6.2f}°"
        )

    return limits


def validate_config(
    config: Sequence[float],
    limits: Sequence[tuple[float, float]],
    tolerance: float = 1e-7,
) -> None:
    """检查一组关节角是否超出物理范围。"""

    if len(config) != len(limits):
        raise ValueError(
            "关节配置数量与关节限制数量不一致。"
        )

    for index, (
        value,
        (lower, upper),
    ) in enumerate(
        zip(config, limits),
        start=1,
    ):
        if (
            value < lower - tolerance
            or value > upper + tolerance
        ):
            raise RuntimeError(
                f"J{index} 越界："
                f"{math.degrees(value):.3f}°，"
                f"允许范围为 "
                f"[{math.degrees(lower):.3f}°, "
                f"{math.degrees(upper):.3f}°]。"
            )


def minimum_limit_clearance(
    config: Sequence[float],
    limits: Sequence[tuple[float, float]],
) -> float:
    """返回整组配置中最小的关节边界余量。"""

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


def normalize_configs(
    raw_configs: Any,
    joint_count: int,
) -> list[list[float]]:
    """统一 findConfigs 返回的二维或扁平数据格式。"""

    if raw_configs is None or len(raw_configs) == 0:
        return []

    first = raw_configs[0]

    if isinstance(first, (list, tuple)):
        return [
            [float(v) for v in config]
            for config in raw_configs
        ]

    flat = [
        float(v)
        for v in raw_configs
    ]

    if len(flat) % joint_count != 0:
        raise RuntimeError(
            "findConfigs 返回的数据长度异常。"
        )

    return [
        flat[start:start + joint_count]
        for start in range(
            0,
            len(flat),
            joint_count,
        )
    ]


def normalize_path(
    raw_path: Any,
    joint_count: int,
) -> list[list[float]]:
    """统一 generatePath 返回的扁平或二维路径格式。"""

    if raw_path is None or len(raw_path) == 0:
        return []

    first = raw_path[0]

    if isinstance(first, (list, tuple)):
        path = [
            [float(v) for v in config]
            for config in raw_path
        ]
    else:
        flat = [
            float(v)
            for v in raw_path
        ]

        if len(flat) % joint_count != 0:
            raise RuntimeError(
                "generatePath 返回的数据长度异常。"
            )

        path = [
            flat[start:start + joint_count]
            for start in range(
                0,
                len(flat),
                joint_count,
            )
        ]

    for index, config in enumerate(
        path,
        start=1,
    ):
        if len(config) != joint_count:
            raise RuntimeError(
                f"路径点 {index} 的关节数量不正确。"
            )

    return path


def choose_best_solution(
    solutions: Sequence[Sequence[float]],
    current_config: Sequence[float],
    limits: Sequence[tuple[float, float]],
) -> list[float] | None:
    """
    选择不越界、留有安全余量且接近当前姿态的IK解。
    """

    required_margin = math.radians(
        FINAL_LIMIT_MARGIN_DEG
    )

    candidates: list[
        tuple[float, list[float]]
    ] = []

    for raw_solution in solutions:
        solution = [
            float(v)
            for v in raw_solution
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
            target,
            current,
            (lower, upper),
        ) in zip(
            solution,
            current_config,
            limits,
        ):
            joint_range = max(
                upper - lower,
                1e-6,
            )

            movement_score += (
                (target - current)
                / joint_range
            ) ** 2

        # 越接近关节边界，惩罚越大。
        limit_penalty = (
            0.05
            / (
                clearance
                + math.radians(0.5)
            )
        )

        candidates.append(
            (
                movement_score + limit_penalty,
                solution,
            )
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: item[0]
    )

    return candidates[0][1]


# ============================================================
# 4. 位姿误差与运动执行
# ============================================================

def normalize_quaternion(
    quaternion: Sequence[float],
) -> list[float]:
    """归一化四元数。"""

    norm = math.sqrt(
        sum(
            value * value
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


def position_error(
    sim: Any,
    tip: int,
    target: int,
) -> float:
    """计算 tip 与 target 的位置误差。"""

    tip_position = sim.getObjectPosition(
        tip,
        sim.handle_world,
    )

    target_position = sim.getObjectPosition(
        target,
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


def orientation_error(
    sim: Any,
    tip: int,
    target: int,
) -> float:
    """计算 tip 与 target 的完整方向误差。"""

    tip_pose = sim.getObjectPose(
        tip,
        sim.handle_world,
    )

    target_pose = sim.getObjectPose(
        target,
        sim.handle_world,
    )

    tip_quaternion = normalize_quaternion(
        tip_pose[3:7]
    )

    target_quaternion = normalize_quaternion(
        target_pose[3:7]
    )

    dot_product = abs(
        sum(
            tip_value * target_value
            for tip_value, target_value in zip(
                tip_quaternion,
                target_quaternion,
            )
        )
    )

    dot_product = max(
        -1.0,
        min(1.0, dot_product),
    )

    return 2.0 * math.acos(
        dot_product
    )


def print_pose(
    sim: Any,
    handle: int,
    title: str,
) -> None:
    """打印对象的世界坐标位姿。"""

    pose = sim.getObjectPose(
        handle,
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


def move_joint_space_smoothly(
    sim: Any,
    joints: Sequence[int],
    target_config: Sequence[float],
    limits: Sequence[tuple[float, float]],
) -> None:
    """
    在仿真停止状态下，
    平滑执行关节空间运动。
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
        ALIGN_MOTION_STEPS + 1
    ):
        progress = (
            step_index
            / ALIGN_MOTION_STEPS
        )

        alpha = (
            3.0 * progress**2
            - 2.0 * progress**3
        )

        config = [
            start
            + alpha * (target - start)
            for start, target in zip(
                start_config,
                target_config,
            )
        ]

        validate_config(
            config,
            limits,
        )

        for joint, value in zip(
            joints,
            config,
        ):
            sim.setJointPosition(
                joint,
                value,
            )

        time.sleep(
            ALIGN_STEP_DELAY_SECONDS
        )


def execute_cartesian_path(
    sim: Any,
    joints: Sequence[int],
    path: Sequence[Sequence[float]],
    limits: Sequence[tuple[float, float]],
    reverse: bool = False,
) -> None:
    """逐点执行 generatePath 产生的关节配置。"""

    configurations = (
        list(reversed(path))
        if reverse
        else list(path)
    )

    for config in configurations:
        validate_config(
            config,
            limits,
        )

        for joint, value in zip(
            joints,
            config,
        ):
            sim.setJointPosition(
                joint,
                value,
            )

        time.sleep(
            CARTESIAN_STEP_DELAY_SECONDS
        )


# ============================================================
# 5. IK 建立、预抓取对齐和直线下降
# ============================================================

def configure_ik(
    sim: Any,
    simIK: Any,
    tip: int,
    target: int,
    joints: Sequence[int],
    limits: Sequence[tuple[float, float]],
) -> tuple[int, int, int, list[int]]:
    """创建完整六维位姿约束的IK环境。"""

    ik_base = int(
        sim.getObjectParent(joints[0])
    )

    ik_environment = int(
        simIK.createEnvironment()
    )

    ik_group = int(
        simIK.createGroup(ik_environment)
    )

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

    ik_tip = get_ik_handle(
        sim_to_ik_map,
        tip,
    )

    ik_joints = [
        get_ik_handle(
            sim_to_ik_map,
            joint,
        )
        for joint in joints
    ]

    simIK.setGroupCalculation(
        ik_environment,
        ik_group,
        simIK.method_damped_least_squares,
        0.05,
        120,
    )

    try:
        flags = simIK.getGroupFlags(
            ik_environment,
            ik_group,
        )

        flags |= (
            simIK.group_avoidlimits
            | simIK.group_restoreonbadlintol
            | simIK.group_restoreonbadangtol
        )

        simIK.setGroupFlags(
            ik_environment,
            ik_group,
            flags,
        )

    except Exception:
        print(
            "提示：当前版本未启用额外group flags，"
            "继续使用关节限制求解。"
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

    for ik_joint, (
        lower,
        upper,
    ) in zip(
        ik_joints,
        limits,
    ):
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

    return (
        ik_environment,
        ik_group,
        ik_tip,
        ik_joints,
    )


def align_pregrasp_if_needed(
    sim: Any,
    simIK: Any,
    ik_environment: int,
    ik_group: int,
    ik_joints: Sequence[int],
    tip: int,
    target: int,
    joints: Sequence[int],
    limits: Sequence[tuple[float, float]],
) -> None:
    """
    如果当前尚未对齐，
    则先搜索安全IK解并完成预抓取对齐。
    """

    initial_position_error = position_error(
        sim,
        tip,
        target,
    )

    initial_orientation_error = orientation_error(
        sim,
        tip,
        target,
    )

    print(
        "\n========== 预抓取点初始误差 =========="
    )

    print(
        f"位置误差："
        f"{initial_position_error * 1000:.3f} mm"
    )

    print(
        f"方向误差："
        f"{math.degrees(initial_orientation_error):.3f}°"
    )

    already_aligned = (
        initial_position_error
        <= POSITION_TOLERANCE_M
        and initial_orientation_error
        <= math.radians(
            ORIENTATION_TOLERANCE_DEG
        )
    )

    if already_aligned:
        print(
            "预抓取点已经对齐，无需重新求解。"
        )
        return

    if not ALIGN_PREGRASP_FIRST:
        raise RuntimeError(
            "tip尚未与target对齐，"
            "且ALIGN_PREGRASP_FIRST=False。"
        )

    simIK.syncFromSim(
        ik_environment,
        [ik_group],
    )

    current_config = [
        float(sim.getJointPosition(joint))
        for joint in joints
    ]

    raw_solutions = simIK.findConfigs(
        ik_environment,
        ik_group,
        list(ik_joints),
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
                1.0
            ] * len(joints),
        },
    )

    solutions = normalize_configs(
        raw_solutions,
        len(joints),
    )

    print(
        f"搜索到 {len(solutions)} 组"
        "预抓取候选IK解。"
    )

    best_solution = choose_best_solution(
        solutions,
        current_config,
        limits,
    )

    if best_solution is None:
        raise RuntimeError(
            "没有找到满足预抓取位姿和"
            "安全余量的IK解。"
        )

    move_joint_space_smoothly(
        sim,
        joints,
        best_solution,
        limits,
    )

    final_position_error = position_error(
        sim,
        tip,
        target,
    )

    final_orientation_error = orientation_error(
        sim,
        tip,
        target,
    )

    print(
        "\n========== 预抓取对齐结果 =========="
    )

    print(
        f"位置误差："
        f"{final_position_error * 1000:.3f} mm"
    )

    print(
        f"方向误差："
        f"{math.degrees(final_orientation_error):.3f}°"
    )

    if (
        final_position_error
        > POSITION_TOLERANCE_M
        or final_orientation_error
        > math.radians(
            ORIENTATION_TOLERANCE_DEG
        )
    ):
        raise RuntimeError(
            "预抓取位姿对齐未达到设定精度，"
            "取消下降。"
        )


def main() -> None:
    print("正在连接 CoppeliaSim……")

    client = RemoteAPIClient()

    sim = client.require("sim")
    simIK = client.require("simIK")

    print("连接成功。")

    if (
        sim.getSimulationState()
        != sim.simulation_stopped
    ):
        raise RuntimeError(
            "请先停止CoppeliaSim仿真，"
            "再运行本程序。"
        )

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
            f"{TIP_PATH} 不是Dummy。"
        )

    if (
        sim.getObjectType(target)
        != sim.sceneobject_dummy
    ):
        raise RuntimeError(
            f"{TARGET_PATH} 不是Dummy。"
        )

    joints = get_joints_from_tip(
        sim,
        tip,
    )

    if len(joints) != 7:
        print(
            "\n从tip回溯得到的关节："
        )

        for joint in joints:
            print(
                get_full_path(
                    sim,
                    joint,
                )
            )

        raise RuntimeError(
            f"应找到7个关节，"
            f"实际找到{len(joints)}个。"
        )

    print(
        "\n========== 七关节运动链 =========="
    )

    for index, joint in enumerate(
        joints,
        start=1,
    ):
        print(
            f"J{index}: "
            f"{get_full_path(sim, joint)}"
        )

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

    ik_environment = -1

    try:
        (
            ik_environment,
            ik_group,
            ik_tip,
            ik_joints,
        ) = configure_ik(
            sim,
            simIK,
            tip,
            target,
            joints,
            limits,
        )

        # ====================================================
        # A. 确保当前位于预抓取点
        # ====================================================

        align_pregrasp_if_needed(
            sim,
            simIK,
            ik_environment,
            ik_group,
            ik_joints,
            tip,
            target,
            joints,
            limits,
        )

        print_pose(
            sim,
            tip,
            "下降前iiwa_tip位姿",
        )

        print_pose(
            sim,
            target,
            "预抓取iiwa_target位姿",
        )

        pregrasp_target_pose = list(
            sim.getObjectPose(
                target,
                sim.handle_world,
            )
        )

        pregrasp_tip_pose = list(
            sim.getObjectPose(
                tip,
                sim.handle_world,
            )
        )

        # ====================================================
        # B. 只降低目标Z，不改变四元数方向
        # ====================================================

        approach_target_pose = list(
            pregrasp_target_pose
        )

        approach_target_pose[2] -= (
            APPROACH_DISTANCE_M
        )

        sim.setObjectPose(
            target,
            approach_target_pose,
            sim.handle_world,
        )

        print(
            "\n========== 下降目标 =========="
        )

        print(
            f"计划下降距离："
            f"{APPROACH_DISTANCE_M * 100:.2f} cm"
        )

        print(
            f"目标位置："
            f"X={approach_target_pose[0]:.5f} m，"
            f"Y={approach_target_pose[1]:.5f} m，"
            f"Z={approach_target_pose[2]:.5f} m"
        )

        print(
            "目标四元数保持不变。"
        )

        # 把新的target位姿与当前关节状态
        # 同步到IK环境。
        simIK.syncFromSim(
            ik_environment,
            [ik_group],
        )

        print(
            "\n正在生成保持完整方向的"
            "笛卡尔直线路径……"
        )

        raw_path = simIK.generatePath(
            ik_environment,
            ik_group,
            list(ik_joints),
            ik_tip,
            CARTESIAN_PATH_POINTS,
        )

        path = normalize_path(
            raw_path,
            len(joints),
        )

        if not path:
            sim.setObjectPose(
                target,
                pregrasp_target_pose,
                sim.handle_world,
            )

            raise RuntimeError(
                "未能生成直线下降路径。"
                "请先把APPROACH_DISTANCE_M"
                "改为0.03，或提高预抓取点。"
            )

        print(
            f"路径生成成功，共"
            f"{len(path)}个路径点。"
        )

        path_minimum_clearance = float(
            "inf"
        )

        for config in path:
            validate_config(
                config,
                limits,
            )

            path_minimum_clearance = min(
                path_minimum_clearance,
                minimum_limit_clearance(
                    config,
                    limits,
                ),
            )

        print(
            "整条路径的最小关节边界余量："
            f"{math.degrees(path_minimum_clearance):.3f}°"
        )

        if (
            path_minimum_clearance
            < math.radians(
                FINAL_LIMIT_MARGIN_DEG
            )
        ):
            sim.setObjectPose(
                target,
                pregrasp_target_pose,
                sim.handle_world,
            )

            raise RuntimeError(
                "下降路径存在关节过度接近边界的情况，"
                "已取消执行。"
            )

        # ====================================================
        # C. 执行下降
        # ====================================================

        print(
            "\n开始沿笛卡尔直线路径下降……"
        )

        execute_cartesian_path(
            sim,
            joints,
            path,
            limits,
            reverse=False,
        )

        print("下降完成。")

        final_tip_pose = list(
            sim.getObjectPose(
                tip,
                sim.handle_world,
            )
        )

        final_position_error = position_error(
            sim,
            tip,
            target,
        )

        final_orientation_error = orientation_error(
            sim,
            tip,
            target,
        )

        actual_descent = (
            pregrasp_tip_pose[2]
            - final_tip_pose[2]
        )

        print(
            "\n========== 直线下降结果 =========="
        )

        print(
            f"实际下降距离："
            f"{actual_descent * 100:.3f} cm"
        )

        print(
            f"下降后位置误差："
            f"{final_position_error * 1000:.3f} mm"
        )

        print(
            f"下降后方向误差："
            f"{math.degrees(final_orientation_error):.3f}°"
        )

        print(
            f"路径最小关节边界余量："
            f"{math.degrees(path_minimum_clearance):.3f}°"
        )

        success = (
            final_position_error
            <= POSITION_TOLERANCE_M
            and final_orientation_error
            <= math.radians(
                ORIENTATION_TOLERANCE_DEG
            )
            and path_minimum_clearance
            >= math.radians(
                FINAL_LIMIT_MARGIN_DEG
            )
        )

        if not success:
            raise RuntimeError(
                "下降已执行，但最终误差"
                "未达到设定阈值。"
            )

        print(
            "结果：末端已保持完整方向，"
            "沿直线到达下降目标。"
        )

        # ====================================================
        # D. 测试后沿原路径返回
        # ====================================================

        if RETURN_TO_PREGRASP:
            print(
                "\n正在沿原路径返回预抓取点……"
            )

            sim.setObjectPose(
                target,
                pregrasp_target_pose,
                sim.handle_world,
            )

            execute_cartesian_path(
                sim,
                joints,
                path,
                limits,
                reverse=True,
            )

            return_position_error = position_error(
                sim,
                tip,
                target,
            )

            return_orientation_error = orientation_error(
                sim,
                tip,
                target,
            )

            print("返回完成。")

            print(
                f"返回后位置误差："
                f"{return_position_error * 1000:.3f} mm"
            )

            print(
                f"返回后方向误差："
                f"{math.degrees(return_orientation_error):.3f}°"
            )

            if (
                return_position_error
                > POSITION_TOLERANCE_M
                or return_orientation_error
                > math.radians(
                    ORIENTATION_TOLERANCE_DEG
                )
            ):
                raise RuntimeError(
                    "返回预抓取点后的误差超过阈值。"
                )

            print(
                "结果：已安全返回预抓取点。"
            )

    finally:
        if ik_environment != -1:
            simIK.eraseEnvironment(
                ik_environment
            )


if __name__ == "__main__":
    main()