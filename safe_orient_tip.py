import math
import re
import time
from typing import Any, Sequence

from remote_session import RemoteAPIClient


# ============================================================
# 用户可修改参数
# ============================================================

ROBOT_PATH = "/iiwa"
TIP_PATH = "/iiwa_tip"
TARGET_PATH = "/iiwa_target"

# False：只要求末端接近方向与目标一致，允许绕接近轴旋转。
# True：要求完整姿态完全一致。
LOCK_YAW = True

# 强制让 target 的位置等于当前 tip 的位置。
# 这样本阶段只改变方向，不向方块下降。
KEEP_CURRENT_PREGRASP_POSITION = True

# 优先尝试严格保持末端位置的笛卡尔路径。
TRY_CARTESIAN_PATH_FIRST = True

# 笛卡尔路径失败后，是否允许使用关节空间路径。
# 关节空间路径最终位姿准确，但运动中末端可能稍微画弧。
ALLOW_JOINT_SPACE_FALLBACK = True

# 最终关节角与物理边界之间至少保留的角度。
FINAL_LIMIT_MARGIN_DEG = 5.0

# IK 在接近边界多少度时开始主动回避。
IK_AVOID_MARGIN_DEG = 10.0

# 单次 IK 迭代允许的最大关节变化。
IK_MAX_STEP_DEG = 3.0

# 笛卡尔路径离散点数量。
CARTESIAN_PATH_POINTS = 150

# 关节空间回退路径的插值步数。
JOINT_INTERPOLATION_STEPS = 180

# 随机搜索 IK 解的最长时间。
IK_SEARCH_TIME_SECONDS = 8.0


# ============================================================
# 工具函数
# ============================================================

def extract_joint_number(name: str) -> int:
    """
    从关节名称末尾提取编号。

    支持：
    joint1
    iiwa_joint1
    LBR_iiwa_7_R800_joint1
    """
    match = re.search(r"joint[_ ]?(\d+)$", name, re.IGNORECASE)

    if match:
        return int(match.group(1))

    return 999


def get_scene_joints(sim: Any, robot_handle: int) -> list[int]:
    """读取并按 joint1～joint7 排序。"""

    joints = sim.getObjectsInTree(
        robot_handle,
        sim.sceneobject_joint,
        0,
    )

    joints.sort(
        key=lambda handle: extract_joint_number(
            sim.getObjectAlias(handle)
        )
    )

    return joints


def is_descendant(sim: Any, child: int, ancestor: int) -> bool:
    """判断 child 是否位于 ancestor 的对象层级之下。"""

    current = child

    while current != -1:
        if current == ancestor:
            return True

        current = sim.getObjectParent(current)

    return False


def get_mapped_handle(
    mapping: Any,
    scene_handle: int,
) -> int:
    """
    从 addElementFromScene 返回的映射中，
    获取场景对象对应的 IK 环境对象句柄。

    不同远程API版本可能将映射解码为dict或列表，
    所以这里做兼容处理。
    """

    if isinstance(mapping, dict):
        if scene_handle in mapping:
            return int(mapping[scene_handle])

        string_key = str(scene_handle)

        if string_key in mapping:
            return int(mapping[string_key])

    if isinstance(mapping, list):
        # 某些版本可能以 [sceneHandle, ikHandle] 对的形式返回。
        for item in mapping:
            if (
                isinstance(item, (list, tuple))
                and len(item) == 2
                and int(item[0]) == scene_handle
            ):
                return int(item[1])

        # 极少数版本可能直接使用句柄作为列表索引。
        if 0 <= scene_handle < len(mapping):
            value = mapping[scene_handle]

            if isinstance(value, int):
                return value

    raise RuntimeError(
        f"无法在 simToIkMap 中找到场景句柄 {scene_handle}。"
    )


def get_joint_limits(
    sim: Any,
    joints: Sequence[int],
) -> list[tuple[float, float]]:
    """
    读取场景中的实际关节限制。

    返回：
    [
        (joint1_lower, joint1_upper),
        ...
    ]
    """

    limits: list[tuple[float, float]] = []

    print("\n================ 关节限制 ================")

    for index, joint in enumerate(joints, start=1):
        cyclic, interval = sim.getJointInterval(joint)

        if cyclic:
            raise RuntimeError(
                f"Joint {index} 被设置为无限循环关节。"
                "iiwa关节应当具有明确限制，请检查模型属性。"
            )

        lower = float(interval[0])
        upper = lower + float(interval[1])
        current = float(sim.getJointPosition(joint))

        limits.append((lower, upper))

        lower_deg = math.degrees(lower)
        upper_deg = math.degrees(upper)
        current_deg = math.degrees(current)

        distance_lower = current_deg - lower_deg
        distance_upper = upper_deg - current_deg
        nearest = min(distance_lower, distance_upper)

        print(
            f"J{index}: "
            f"[{lower_deg:8.2f}°, {upper_deg:8.2f}°]  "
            f"当前={current_deg:8.2f}°  "
            f"距最近边界={nearest:6.2f}°"
        )

        if nearest < FINAL_LIMIT_MARGIN_DEG:
            print(
                f"  警告：J{index} 当前已经非常接近边界，"
                "程序将搜索另一组冗余IK解。"
            )

    return limits


def validate_physical_limits(
    config: Sequence[float],
    limits: Sequence[tuple[float, float]],
    tolerance: float = 1e-7,
) -> None:
    """确保配置没有超出任何物理关节限制。"""

    for index, (value, limit) in enumerate(
        zip(config, limits),
        start=1,
    ):
        lower, upper = limit

        if value < lower - tolerance or value > upper + tolerance:
            raise RuntimeError(
                f"J{index} 超出物理限制："
                f"{math.degrees(value):.3f}°，"
                f"允许范围为 "
                f"[{math.degrees(lower):.3f}°, "
                f"{math.degrees(upper):.3f}°]"
            )


def minimum_limit_clearance(
    config: Sequence[float],
    limits: Sequence[tuple[float, float]],
) -> float:
    """返回配置中距最近关节边界的最小角距离，单位弧度。"""

    clearances = []

    for value, (lower, upper) in zip(config, limits):
        clearances.append(
            min(value - lower, upper - value)
        )

    return min(clearances)


def print_config(
    title: str,
    config: Sequence[float],
    limits: Sequence[tuple[float, float]],
) -> None:
    """打印关节配置以及每个关节的边界余量。"""

    print(f"\n================ {title} ================")

    for index, (value, limit) in enumerate(
        zip(config, limits),
        start=1,
    ):
        lower, upper = limit

        clearance_deg = math.degrees(
            min(value - lower, upper - value)
        )

        print(
            f"J{index}: "
            f"{math.degrees(value):8.2f}°  "
            f"边界余量={clearance_deg:6.2f}°"
        )


def normalize_configurations(
    raw_configs: Any,
    joint_count: int,
) -> list[list[float]]:
    """
    兼容 findConfigs 可能返回的嵌套列表或扁平列表。
    """

    if raw_configs is None or len(raw_configs) == 0:
        return []

    first = raw_configs[0]

    if isinstance(first, (list, tuple)):
        return [
            [float(value) for value in config]
            for config in raw_configs
        ]

    flat = [float(value) for value in raw_configs]

    if len(flat) % joint_count != 0:
        raise RuntimeError(
            "findConfigs 返回的数据长度不能被关节数量整除。"
        )

    return [
        flat[index:index + joint_count]
        for index in range(0, len(flat), joint_count)
    ]


def normalize_path(
    raw_path: Any,
    joint_count: int,
) -> list[list[float]]:
    """将 generatePath 返回结果转换为逐点关节配置。"""

    if raw_path is None or len(raw_path) == 0:
        return []

    first = raw_path[0]

    if isinstance(first, (list, tuple)):
        return [
            [float(value) for value in config]
            for config in raw_path
        ]

    flat_path = [float(value) for value in raw_path]

    if len(flat_path) % joint_count != 0:
        raise RuntimeError(
            "generatePath 返回的数据长度不正确。"
        )

    return [
        flat_path[index:index + joint_count]
        for index in range(0, len(flat_path), joint_count)
    ]


def choose_best_configuration(
    solutions: Sequence[Sequence[float]],
    current: Sequence[float],
    limits: Sequence[tuple[float, float]],
) -> list[float] | None:
    """
    从多个IK解中选择：
    1. 不越界；
    2. 最终至少保留指定边界余量；
    3. 距离当前姿态较近；
    4. 尽量位于关节范围中间。
    """

    required_margin = math.radians(
        FINAL_LIMIT_MARGIN_DEG
    )

    candidates: list[tuple[float, list[float]]] = []

    for solution in solutions:
        try:
            validate_physical_limits(solution, limits)
        except RuntimeError:
            continue

        clearance = minimum_limit_clearance(
            solution,
            limits,
        )

        if clearance < required_margin:
            continue

        normalized_travel = 0.0

        for value, start, (lower, upper) in zip(
            solution,
            current,
            limits,
        ):
            joint_range = max(upper - lower, 1e-6)

            normalized_travel += (
                (value - start) / joint_range
            ) ** 2

        # clearance越大，惩罚越小。
        clearance_penalty = 0.08 / (
            clearance + math.radians(0.5)
        )

        score = normalized_travel + clearance_penalty

        candidates.append(
            (score, list(solution))
        )

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])

    return candidates[0][1]


def execute_joint_space_motion(
    sim: Any,
    joints: Sequence[int],
    target_config: Sequence[float],
    limits: Sequence[tuple[float, float]],
) -> None:
    """
    在仿真停止状态下，平滑执行关节空间轨迹。

    最终末端位姿由IK解保证；
    中间过程末端可能产生轻微弧线运动。
    """

    start_config = [
        float(sim.getJointPosition(joint))
        for joint in joints
    ]

    for step_index in range(
        JOINT_INTERPOLATION_STEPS + 1
    ):
        progress = (
            step_index / JOINT_INTERPOLATION_STEPS
        )

        # Smoothstep。
        alpha = (
            3.0 * progress**2
            - 2.0 * progress**3
        )

        current_config = [
            start + alpha * (target - start)
            for start, target in zip(
                start_config,
                target_config,
            )
        ]

        validate_physical_limits(
            current_config,
            limits,
        )

        for joint, value in zip(
            joints,
            current_config,
        ):
            sim.setJointPosition(joint, value)

        time.sleep(0.015)


def execute_cartesian_path(
    sim: Any,
    joints: Sequence[int],
    path: Sequence[Sequence[float]],
    limits: Sequence[tuple[float, float]],
) -> None:
    """
    执行 simIK.generatePath 产生的路径。

    该路径尝试保持末端笛卡尔约束，
    本阶段主要表现为原地调整方向。
    """

    for config in path:
        validate_physical_limits(config, limits)

        for joint, value in zip(joints, config):
            sim.setJointPosition(joint, value)

        time.sleep(0.015)


def position_error(
    sim: Any,
    tip: int,
    target: int,
) -> float:
    """计算tip与target之间的位置误差。"""

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
            (a - b) ** 2
            for a, b in zip(
                tip_position,
                target_position,
            )
        )
    )


def quaternion_angle_error(
    sim: Any,
    tip: int,
    target: int,
) -> float:
    """计算两个完整姿态之间的旋转角误差。"""

    tip_pose = sim.getObjectPose(
        tip,
        sim.handle_world,
    )

    target_pose = sim.getObjectPose(
        target,
        sim.handle_world,
    )

    q_tip = tip_pose[3:7]
    q_target = target_pose[3:7]

    norm_tip = math.sqrt(
        sum(value * value for value in q_tip)
    )

    norm_target = math.sqrt(
        sum(value * value for value in q_target)
    )

    dot = sum(
        a * b
        for a, b in zip(q_tip, q_target)
    ) / max(norm_tip * norm_target, 1e-12)

    # q和-q表示相同姿态。
    dot = abs(dot)
    dot = max(-1.0, min(1.0, dot))

    return 2.0 * math.acos(dot)


def local_z_axis(
    sim: Any,
    handle: int,
) -> list[float]:
    """读取对象局部Z轴在世界坐标系中的方向。"""

    matrix = sim.getObjectMatrix(
        handle,
        sim.handle_world,
    )

    # CoppeliaSim 3x4矩阵的第三列是局部Z轴。
    axis = [
        float(matrix[2]),
        float(matrix[6]),
        float(matrix[10]),
    ]

    norm = math.sqrt(
        sum(value * value for value in axis)
    )

    return [
        value / max(norm, 1e-12)
        for value in axis
    ]


def approach_axis_error(
    sim: Any,
    tip: int,
    target: int,
) -> float:
    """
    计算tip与target局部Z轴的夹角。
    用于只约束自上而下接近方向的情况。
    """

    axis_tip = local_z_axis(sim, tip)
    axis_target = local_z_axis(sim, target)

    dot = sum(
        a * b
        for a, b in zip(axis_tip, axis_target)
    )

    dot = max(-1.0, min(1.0, dot))

    return math.acos(dot)


# ============================================================
# 主程序
# ============================================================

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
            "请先停止CoppeliaSim仿真。"
            "本程序不能在动力学仿真运行时执行。"
        )

    robot = sim.getObject(ROBOT_PATH)
    tip = sim.getObject(TIP_PATH)
    target = sim.getObject(TARGET_PATH)

    if sim.getObjectType(tip) != sim.sceneobject_dummy:
        raise RuntimeError(
            f"{TIP_PATH} 不是Dummy对象。"
        )

    if sim.getObjectType(target) != sim.sceneobject_dummy:
        raise RuntimeError(
            f"{TARGET_PATH} 不是Dummy对象。"
        )

    joints = get_scene_joints(sim, robot)

    if len(joints) != 7:
        print("找到的关节如下：")

        for joint in joints:
            print(sim.getObjectAlias(joint))

        raise RuntimeError(
            f"预期找到7个关节，实际找到{len(joints)}个。"
        )

    if not is_descendant(sim, tip, joints[-1]):
        raise RuntimeError(
            "iiwa_tip不在Joint 7之后的对象层级中。"
            "请把iiwa_tip挂到末端法兰或最后一个连杆下。"
        )

    print("\n关节顺序：")

    for index, joint in enumerate(joints, start=1):
        print(
            f"J{index}: {sim.getObjectAlias(joint)}"
        )

    limits = get_joint_limits(sim, joints)

    current_config = [
        float(sim.getJointPosition(joint))
        for joint in joints
    ]

    validate_physical_limits(
        current_config,
        limits,
    )

    if KEEP_CURRENT_PREGRASP_POSITION:
        current_tip_position = sim.getObjectPosition(
            tip,
            sim.handle_world,
        )

        # 只复制位置，保留target原有方向。
        sim.setObjectPosition(
            target,
            current_tip_position,
            sim.handle_world,
        )

        print(
            "\n已将iiwa_target的位置锁定为"
            "当前iiwa_tip的位置。"
        )

    initial_position_error = position_error(
        sim,
        tip,
        target,
    )

    print(
        f"初始位置误差："
        f"{initial_position_error * 1000:.3f} mm"
    )

    # 自上而下通常只需要接近轴一致。
    constraints = (
        simIK.constraint_position
        | simIK.constraint_alpha_beta
    )

    if LOCK_YAW:
        constraints |= simIK.constraint_gamma

        print(
            "姿态模式：完整方向约束，"
            "包括绕接近轴的旋转角。"
        )
    else:
        print(
            "姿态模式：自上而下接近方向约束，"
            "绕接近轴的旋转角保持自由。"
        )

    ik_environment = simIK.createEnvironment()
    ik_group = simIK.createGroup(ik_environment)

    try:
        (
            ik_element,
            sim_to_ik_map,
            _,
        ) = simIK.addElementFromScene(
            ik_environment,
            ik_group,
            robot,
            tip,
            target,
            constraints,
        )

        ik_tip = get_mapped_handle(
            sim_to_ik_map,
            tip,
        )

        ik_joints = [
            get_mapped_handle(
                sim_to_ik_map,
                joint,
            )
            for joint in joints
        ]

        # 使用阻尼最小二乘，提高奇异位形附近的稳定性。
        simIK.setGroupCalculation(
            ik_environment,
            ik_group,
            simIK.method_damped_least_squares,
            0.05,
            120,
        )

        # 求解失败时恢复状态，并主动避开关节边界。
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
                0.001,                 # 1 mm
                math.radians(1.0),     # 1 degree
            ],
        )

        for scene_joint, ik_joint, limit in zip(
            joints,
            ik_joints,
            limits,
        ):
            lower, upper = limit

            # 明确把场景中的物理限制复制到IK环境。
            simIK.setJointInterval(
                ik_environment,
                ik_joint,
                False,
                [lower, upper - lower],
            )

            # 让求解器在接近边界时主动回避。
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

        # 把当前关节状态和target状态同步到IK环境。
        simIK.syncFromSim(
            ik_environment,
            [ik_group],
        )

        selected_path: list[list[float]] = []

        # ----------------------------------------------------
        # 路线A：严格笛卡尔路径
        # ----------------------------------------------------
        if TRY_CARTESIAN_PATH_FIRST:
            print(
                "\n正在尝试生成保持预抓取位置的"
                "笛卡尔方向调整路径……"
            )

            raw_path = simIK.generatePath(
                ik_environment,
                ik_group,
                ik_joints,
                ik_tip,
                CARTESIAN_PATH_POINTS,
            )

            candidate_path = normalize_path(
                raw_path,
                len(joints),
            )

            if candidate_path:
                path_is_valid = True

                try:
                    for config in candidate_path:
                        validate_physical_limits(
                            config,
                            limits,
                        )
                except RuntimeError as error:
                    print(
                        f"笛卡尔路径越界：{error}"
                    )

                    path_is_valid = False

                if path_is_valid:
                    final_clearance = (
                        minimum_limit_clearance(
                            candidate_path[-1],
                            limits,
                        )
                    )

                    if final_clearance >= math.radians(
                        FINAL_LIMIT_MARGIN_DEG
                    ):
                        selected_path = candidate_path

                        print(
                            f"笛卡尔路径生成成功，"
                            f"共{len(selected_path)}个配置点。"
                        )
                    else:
                        print(
                            "笛卡尔路径虽然可达，"
                            "但最终关节离边界太近，"
                            "改用多解搜索。"
                        )
            else:
                print(
                    "未生成笛卡尔路径。"
                    "常见原因是奇异位形、"
                    "中间姿态不可达或关节边界。"
                )

        # ----------------------------------------------------
        # 路线B：搜索多个最终IK解
        # ----------------------------------------------------
        selected_solution: list[float] | None = None

        if not selected_path:
            print(
                "\n正在随机搜索多组安全IK解……"
            )

            raw_solutions = simIK.findConfigs(
                ik_environment,
                ik_group,
                ik_joints,
                {
                    "maxTime": IK_SEARCH_TIME_SECONDS,
                    "maxDist": 1.5,
                    "findMultiple": True,
                    "pMetric": [1.0, 1.0, 1.0, 0.25],
                    "cMetric": [1.0] * len(joints),
                },
            )

            solutions = normalize_configurations(
                raw_solutions,
                len(joints),
            )

            print(
                f"共搜索到 {len(solutions)} 组候选IK解。"
            )

            selected_solution = (
                choose_best_configuration(
                    solutions,
                    current_config,
                    limits,
                )
            )

            if selected_solution is None:
                mode_hint = (
                    "请先把 LOCK_YAW 改为 False。"
                    if LOCK_YAW
                    else
                    "可尝试将预抓取点上移2～5 cm，"
                    "或改变目标绕竖直轴的角度。"
                )

                raise RuntimeError(
                    "没有找到同时满足目标位姿和"
                    f"{FINAL_LIMIT_MARGIN_DEG:.1f}°"
                    "安全余量的关节解。\n"
                    + mode_hint
                )

            print_config(
                "选择的安全IK解",
                selected_solution,
                limits,
            )

        # ----------------------------------------------------
        # 执行
        # ----------------------------------------------------
        if selected_path:
            print(
                "\n开始严格保持预抓取位置调整方向……"
            )

            execute_cartesian_path(
                sim,
                joints,
                selected_path,
                limits,
            )

        elif (
            selected_solution is not None
            and ALLOW_JOINT_SPACE_FALLBACK
        ):
            print(
                "\n警告：正在使用关节空间回退路径。"
            )
            print(
                "最终末端位置和方向会满足目标，"
                "但运动过程中末端可能出现轻微弧线。"
            )

            execute_joint_space_motion(
                sim,
                joints,
                selected_solution,
                limits,
            )

        else:
            raise RuntimeError(
                "已找到安全最终IK解，"
                "但关节空间回退路径被禁用。"
            )

        # ----------------------------------------------------
        # 最终验证
        # ----------------------------------------------------
        final_config = [
            float(sim.getJointPosition(joint))
            for joint in joints
        ]

        validate_physical_limits(
            final_config,
            limits,
        )

        print_config(
            "最终关节角",
            final_config,
            limits,
        )

        final_position_error = position_error(
            sim,
            tip,
            target,
        )

        approach_error = approach_axis_error(
            sim,
            tip,
            target,
        )

        full_orientation_error = (
            quaternion_angle_error(
                sim,
                tip,
                target,
            )
        )

        print("\n================ 最终误差 ================")

        print(
            f"位置误差："
            f"{final_position_error * 1000:.3f} mm"
        )

        print(
            f"接近轴方向误差："
            f"{math.degrees(approach_error):.3f}°"
        )

        print(
            f"完整姿态误差："
            f"{math.degrees(full_orientation_error):.3f}°"
        )

        minimum_clearance_deg = math.degrees(
            minimum_limit_clearance(
                final_config,
                limits,
            )
        )

        print(
            f"所有关节中的最小边界余量："
            f"{minimum_clearance_deg:.3f}°"
        )

        if final_position_error > 0.003:
            print(
                "警告：最终位置误差超过3 mm。"
            )

        if approach_error > math.radians(2.0):
            print(
                "警告：接近方向误差超过2°。"
            )

        if (
            LOCK_YAW
            and full_orientation_error
            > math.radians(2.0)
        ):
            print(
                "警告：完整姿态误差超过2°。"
            )

        print("\n末端方向调整阶段完成。")

    finally:
        simIK.eraseEnvironment(
            ik_environment
        )


if __name__ == "__main__":
    main()
