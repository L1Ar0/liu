import math
import os
import time
import traceback

from remote_session import RemoteAPIClient


# ============================================================
# 用户可调参数
# ============================================================

# 搜索得到的最终构型至少距离真实机械限位多少角度
# 例如实际范围 [-120°, 120°]，搜索范围会缩小到 [-115°, 115°]
SAFETY_MARGIN_DEG = 5.0

# 执行运动时，每一步允许的最大关节变化
MAX_COMMAND_STEP_DEG = 0.5

# 运动最少分成多少步
MIN_MOTION_STEPS = 240

# 每步显示间隔
STEP_DELAY = 0.012

# IK 搜索时间
FIRST_SEARCH_TIME = 8.0
SECOND_SEARCH_TIME = 20.0

# 最终精度要求
MAX_FINAL_POSITION_ERROR = 0.005        # 5 mm
MAX_FINAL_Z_ERROR = math.radians(3.0)  # 3°

# 运行前，tip 与预抓取 target 的最大允许位置差
MAX_INITIAL_POSITION_ERROR = 0.03      # 30 mm

# 工具局部 Z 轴最终朝向世界坐标系下方
TARGET_Z_AXIS = [0.0, 0.0, -1.0]

# 数值边界容差，仅用于浮点误差
BOUNDARY_TOLERANCE = 1e-6


# ============================================================
# 场景对象与机械臂链
# ============================================================

def find_object(sim, possible_paths):
    """依次尝试多个路径，返回第一个有效对象句柄。"""
    for path in possible_paths:
        handle = sim.getObject(
            path,
            {"noError": True},
        )

        if handle is not None and handle >= 0:
            print(f"找到对象：{path}")
            return int(handle)

    raise RuntimeError(
        "无法找到对象，尝试过：\n"
        + "\n".join(possible_paths)
    )


def collect_chain_joints(sim, tip_handle, base_handle):
    """
    从 iiwa_tip 沿父节点向上查找，收集 base 到 tip 之间的关节。
    返回顺序：J1 → J7。
    """
    joints = []
    current = tip_handle
    visited = set()

    while current != base_handle:
        if current in visited:
            raise RuntimeError(
                "机械臂场景层级中检测到循环父子关系。"
            )

        visited.add(current)

        parent = sim.getObjectParent(current)

        if parent is None or parent < 0:
            raise RuntimeError(
                "从 iiwa_tip 向上没有找到 /iiwa。\n"
                "请检查 iiwa_tip 是否确实位于机械臂末端。"
            )

        if sim.getObjectType(parent) == sim.sceneobject_joint:
            joints.append(int(parent))

        current = parent

    joints.reverse()
    return joints


def get_alias(sim, handle):
    """读取对象 Alias。"""
    try:
        return sim.getObjectAlias(handle)
    except Exception:
        return f"handle={handle}"


# ============================================================
# 向量、位置和姿态
# ============================================================

def vector_norm(vector):
    return math.sqrt(sum(value * value for value in vector))


def normalize_vector(vector):
    length = vector_norm(vector)

    if length < 1e-12:
        raise ValueError(f"无法归一化零向量：{vector}")

    return [value / length for value in vector]


def dot_product(vector_a, vector_b):
    return sum(a * b for a, b in zip(vector_a, vector_b))


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def angle_between_vectors(vector_a, vector_b):
    unit_a = normalize_vector(vector_a)
    unit_b = normalize_vector(vector_b)

    cosine_value = clamp(
        dot_product(unit_a, unit_b),
        -1.0,
        1.0,
    )

    return math.acos(cosine_value)


def position_distance(position_a, position_b):
    return math.sqrt(
        sum(
            (a - b) ** 2
            for a, b in zip(position_a, position_b)
        )
    )


def get_local_axis(sim, object_handle, axis_name):
    """
    获取对象局部坐标轴在世界坐标系中的方向。

    CoppeliaSim 变换矩阵排列：
    [Xx, Yx, Zx, Px,
     Xy, Yy, Zy, Py,
     Xz, Yz, Zz, Pz]
    """
    matrix = sim.getObjectMatrix(
        object_handle,
        sim.handle_world,
    )

    axis_name = axis_name.lower()

    if axis_name == "x":
        axis = [matrix[0], matrix[4], matrix[8]]
    elif axis_name == "y":
        axis = [matrix[1], matrix[5], matrix[9]]
    elif axis_name == "z":
        axis = [matrix[2], matrix[6], matrix[10]]
    else:
        raise ValueError(f"未知坐标轴：{axis_name}")

    return normalize_vector(axis)


def get_horizontal_tool_axis(sim, tip_handle):
    """
    取当前工具 X 轴的水平投影，用于构造目标姿态。

    由于本程序只约束 alpha/beta，绕工具 Z 轴仍然自由，
    因此这个 X 轴只用于生成一个合法姿态。
    """
    x_axis = get_local_axis(sim, tip_handle, "x")

    horizontal_axis = [
        x_axis[0],
        x_axis[1],
        0.0,
    ]

    if vector_norm(horizontal_axis) < 1e-6:
        y_axis = get_local_axis(sim, tip_handle, "y")

        horizontal_axis = [
            y_axis[0],
            y_axis[1],
            0.0,
        ]

    if vector_norm(horizontal_axis) < 1e-6:
        horizontal_axis = [1.0, 0.0, 0.0]

    return normalize_vector(horizontal_axis)


def print_position(title, position):
    print(title)
    print(f"  x = {position[0]: .5f} m")
    print(f"  y = {position[1]: .5f} m")
    print(f"  z = {position[2]: .5f} m")


# ============================================================
# 关节边界
# ============================================================

def read_joint_limits(sim, scene_joint_handles):
    """
    读取场景中七个关节的真实机械范围。

    返回：
    [
        {
            "lower": 下限,
            "upper": 上限,
            "safe_lower": 带安全余量的下限,
            "safe_upper": 带安全余量的上限,
            ...
        },
        ...
    ]
    """
    joint_limits = []

    print("\n" + "=" * 72)
    print("场景关节范围检查")
    print("=" * 72)

    for index, joint_handle in enumerate(
        scene_joint_handles,
        start=1,
    ):
        cyclic, interval = sim.getJointInterval(joint_handle)

        if cyclic:
            raise RuntimeError(
                f"J{index} 被配置成循环关节，没有有限机械边界。\n"
                "为了严格执行关节限位，请先在 CoppeliaSim 中"
                "为该关节设置真实上下限。"
            )

        if not isinstance(interval, (list, tuple)) or len(interval) < 2:
            raise RuntimeError(
                f"J{index} 的关节范围返回异常：{interval!r}"
            )

        lower = float(interval[0])
        joint_range = float(interval[1])
        upper = lower + joint_range

        if joint_range <= 0:
            raise RuntimeError(
                f"J{index} 的关节范围无效：{joint_range}"
            )

        # 安全余量最多采用设定值，同时不超过关节范围的 5%
        safety_margin = min(
            math.radians(SAFETY_MARGIN_DEG),
            joint_range * 0.05,
        )

        safe_lower = lower + safety_margin
        safe_upper = upper - safety_margin

        if safe_lower >= safe_upper:
            raise RuntimeError(
                f"J{index} 的安全区间无效。"
            )

        current_position = float(
            sim.getJointPosition(joint_handle)
        )

        if (
            current_position < lower - BOUNDARY_TOLERANCE
            or current_position > upper + BOUNDARY_TOLERANCE
        ):
            raise RuntimeError(
                f"J{index} 当前已经越过真实机械边界：\n"
                f"当前={math.degrees(current_position):.3f}°，"
                f"范围=[{math.degrees(lower):.3f}°, "
                f"{math.degrees(upper):.3f}°]"
            )

        distance_to_lower = current_position - lower
        distance_to_upper = upper - current_position

        state = "正常"

        if current_position < safe_lower:
            state = "接近下限，需要向安全区移动"
        elif current_position > safe_upper:
            state = "接近上限，需要向安全区移动"

        print(
            f"J{index} {get_alias(sim, joint_handle)}\n"
            f"  当前：{math.degrees(current_position):8.3f}°\n"
            f"  真实范围："
            f"[{math.degrees(lower):8.3f}°, "
            f"{math.degrees(upper):8.3f}°]\n"
            f"  搜索安全范围："
            f"[{math.degrees(safe_lower):8.3f}°, "
            f"{math.degrees(safe_upper):8.3f}°]\n"
            f"  距下限：{math.degrees(distance_to_lower):8.3f}°，"
            f"距上限：{math.degrees(distance_to_upper):8.3f}°\n"
            f"  状态：{state}"
        )

        joint_limits.append(
            {
                "lower": lower,
                "upper": upper,
                "range": joint_range,
                "safe_lower": safe_lower,
                "safe_upper": safe_upper,
                "safe_range": safe_upper - safe_lower,
                "safety_margin": safety_margin,
            }
        )

    return joint_limits


def assert_configuration_within_hard_limits(
    configuration,
    joint_limits,
    stage_name,
):
    """保证一组关节角严格位于场景真实边界内。"""
    if len(configuration) != len(joint_limits):
        raise RuntimeError(
            f"{stage_name}：关节数与限位数不一致。"
        )

    for index, (position, limit) in enumerate(
        zip(configuration, joint_limits),
        start=1,
    ):
        position = float(position)

        if (
            position < limit["lower"] - BOUNDARY_TOLERANCE
            or position > limit["upper"] + BOUNDARY_TOLERANCE
        ):
            raise RuntimeError(
                f"{stage_name}：J{index} 将越过真实机械边界。\n"
                f"准备写入：{math.degrees(position):.5f}°\n"
                f"真实范围："
                f"[{math.degrees(limit['lower']):.5f}°, "
                f"{math.degrees(limit['upper']):.5f}°]"
            )


def configuration_within_safe_limits(
    configuration,
    joint_limits,
):
    """判断一组构型是否位于收缩后的安全范围内。"""
    if len(configuration) != len(joint_limits):
        return False

    for position, limit in zip(configuration, joint_limits):
        if (
            position < limit["safe_lower"] - BOUNDARY_TOLERANCE
            or position > limit["safe_upper"] + BOUNDARY_TOLERANCE
        ):
            return False

    return True


def read_current_configuration(sim, joint_handles):
    return [
        float(sim.getJointPosition(joint_handle))
        for joint_handle in joint_handles
    ]


# ============================================================
# IK 搜索
# ============================================================

def normalize_configurations(raw_configurations, joint_count):
    """
    兼容不同版本可能返回的：
    - [[q1...q7], [q1...q7]]
    - [q1...q7, q1...q7]
    """
    if raw_configurations is None:
        return []

    if not isinstance(raw_configurations, (list, tuple)):
        raise RuntimeError(
            f"findConfigs 返回类型异常："
            f"{type(raw_configurations)}"
        )

    if len(raw_configurations) == 0:
        return []

    # 嵌套列表形式
    if isinstance(raw_configurations[0], (list, tuple)):
        configurations = []

        for item in raw_configurations:
            if len(item) != joint_count:
                continue

            configurations.append(
                [float(value) for value in item]
            )

        return configurations

    # 平铺形式
    flat_values = [
        float(value)
        for value in raw_configurations
    ]

    if len(flat_values) % joint_count != 0:
        raise RuntimeError(
            "findConfigs 返回的平铺数组长度"
            "不能被关节数量整除。"
        )

    return [
        flat_values[index:index + joint_count]
        for index in range(
            0,
            len(flat_values),
            joint_count,
        )
    ]


def configuration_score(
    candidate,
    current_configuration,
    joint_limits,
):
    """
    分数越低越好。

    同时考虑：
    1. 与当前构型的距离；
    2. 距离安全区间中心的程度；
    3. 是否过于靠近安全边界。
    """
    movement_score = 0.0
    center_score = 0.0
    boundary_score = 0.0

    for candidate_value, current_value, limit in zip(
        candidate,
        current_configuration,
        joint_limits,
    ):
        safe_range = limit["safe_range"]
        safe_center = (
            limit["safe_lower"] + limit["safe_upper"]
        ) / 2.0

        normalized_delta = (
            candidate_value - current_value
        ) / safe_range

        movement_score += normalized_delta ** 2

        normalized_center_distance = (
            candidate_value - safe_center
        ) / (safe_range / 2.0)

        center_score += normalized_center_distance ** 2

        distance_to_safe_edge = min(
            candidate_value - limit["safe_lower"],
            limit["safe_upper"] - candidate_value,
        )

        normalized_edge_distance = max(
            distance_to_safe_edge / safe_range,
            1e-6,
        )

        boundary_score += 1.0 / normalized_edge_distance

    return (
        movement_score
        + 0.08 * center_score
        + 0.002 * boundary_score
    )


def configure_ik_joint_limits(
    simIK,
    ik_environment,
    ik_group,
    ik_joint_handles,
    scene_current_configuration,
    joint_limits,
):
    """
    在 IK 环境内部设置收缩后的安全限位。

    不修改 CoppeliaSim 场景中原始关节范围。
    """
    if len(ik_joint_handles) != len(joint_limits):
        raise RuntimeError(
            "IK 环境中的关节数量与场景不一致。"
        )

    for index, (
        ik_joint,
        current_value,
        limit,
    ) in enumerate(
        zip(
            ik_joint_handles,
            scene_current_configuration,
            joint_limits,
        ),
        start=1,
    ):
        simIK.setJointInterval(
            ik_environment,
            ik_joint,
            False,
            [
                limit["safe_lower"],
                limit["safe_range"],
            ],
        )

        # IK 避限触发阈值
        avoid_margin = min(
            math.radians(SAFETY_MARGIN_DEG),
            limit["safe_range"] * 0.10,
        )

        simIK.setJointLimitMargin(
            ik_environment,
            ik_joint,
            avoid_margin,
        )

        # 限制 IK 内部单步关节变化
        simIK.setJointMaxStepSize(
            ik_environment,
            ik_joint,
            math.radians(2.0),
        )

        # 当前姿态可能正好在真实边界附近。
        # 为保证 IK 环境自身合法，将其夹到安全区间内。
        safe_initial_value = clamp(
            current_value,
            limit["safe_lower"],
            limit["safe_upper"],
        )

        simIK.setJointPosition(
            ik_environment,
            ik_joint,
            safe_initial_value,
        )

        print(
            f"IK J{index} 安全范围："
            f"[{math.degrees(limit['safe_lower']):.2f}°, "
            f"{math.degrees(limit['safe_upper']):.2f}°]"
        )

    # 在原有 flags 基础上启用：
    # - 主动远离关节限位
    # - 碰到限位立即停止
    # - 精度不合格时不保留坏的 IK 状态
    group_flags = int(
        simIK.getGroupFlags(
            ik_environment,
            ik_group,
        )
    )

    for flag_name in [
        "group_avoidlimits",
        "group_stoponlimithit",
        "group_restoreonbadlintol",
        "group_restoreonbadangtol",
    ]:
        flag_value = getattr(
            simIK,
            flag_name,
            None,
        )

        if flag_value is not None:
            group_flags |= int(flag_value)

    simIK.setGroupFlags(
        ik_environment,
        ik_group,
        group_flags,
    )

    print(f"IK Group flags：{group_flags}")


def search_safe_configuration(
    simIK,
    ik_environment,
    ik_group,
    ik_joint_handles,
    current_configuration,
    joint_limits,
):
    """搜索若干安全 IK 构型并选择最合适的一组。"""
    search_attempts = [
        {
            "maxTime": FIRST_SEARCH_TIME,
            "maxDist": 1.5,
            "findMultiple": True,
            "pMetric": [1.0, 1.0, 1.0, 0.2],
            "cMetric": [1.0] * len(ik_joint_handles),
        },
        {
            "maxTime": SECOND_SEARCH_TIME,
            "maxDist": 3.0,
            "findMultiple": True,
            "pMetric": [1.0, 1.0, 1.0, 0.15],
            "cMetric": [1.0] * len(ik_joint_handles),
        },
    ]

    all_candidates = []

    for attempt_index, parameters in enumerate(
        search_attempts,
        start=1,
    ):
        print(
            f"\n开始第 {attempt_index} 次安全构型搜索，"
            f"最长 {parameters['maxTime']:.1f} 秒……"
        )

        raw_configurations = simIK.findConfigs(
            ik_environment,
            ik_group,
            ik_joint_handles,
            parameters,
        )

        configurations = normalize_configurations(
            raw_configurations,
            len(ik_joint_handles),
        )

        print(
            f"本次返回 {len(configurations)} 组候选构型。"
        )

        for configuration in configurations:
            if configuration_within_safe_limits(
                configuration,
                joint_limits,
            ):
                all_candidates.append(configuration)

        if all_candidates:
            break

    if not all_candidates:
        raise RuntimeError(
            "没有找到同时满足以下要求的机械臂构型：\n"
            "1. iiwa_tip 位于 iiwa_target；\n"
            "2. 工具 Z 轴竖直向下；\n"
            f"3. 每个关节距离真实边界至少约 "
            f"{SAFETY_MARGIN_DEG:.1f}°。\n\n"
            "程序没有移动机械臂。可以尝试把 target_cube "
            "移到更靠近机械臂前方的位置，或者提高预抓取点。"
        )

    # 删除重复或近似重复候选
    unique_candidates = []

    for candidate in all_candidates:
        is_duplicate = any(
            max(
                abs(a - b)
                for a, b in zip(candidate, existing)
            ) < math.radians(0.2)
            for existing in unique_candidates
        )

        if not is_duplicate:
            unique_candidates.append(candidate)

    scored_candidates = [
        (
            configuration_score(
                candidate,
                current_configuration,
                joint_limits,
            ),
            candidate,
        )
        for candidate in unique_candidates
    ]

    scored_candidates.sort(key=lambda item: item[0])

    print(
        f"\n共有 {len(scored_candidates)} 组有效安全候选。"
    )

    for rank, (score, candidate) in enumerate(
        scored_candidates[:5],
        start=1,
    ):
        angle_text = ", ".join(
            f"J{index + 1}={math.degrees(value):.1f}°"
            for index, value in enumerate(candidate)
        )

        print(
            f"候选 {rank}：score={score:.5f}\n"
            f"  {angle_text}"
        )

    return scored_candidates[0][1]


# ============================================================
# 安全关节运动
# ============================================================

def smoothstep(value):
    """三次平滑插值，起点和终点速度均为零。"""
    return value * value * (3.0 - 2.0 * value)


def restore_scene(
    sim,
    scene_joint_handles,
    original_configuration,
    target_handle,
    original_target_pose,
):
    """失败时恢复运行前状态。"""
    print("\n正在恢复运行前的机械臂构型……")

    for joint_handle, position in zip(
        scene_joint_handles,
        original_configuration,
    ):
        try:
            sim.setJointPosition(
                joint_handle,
                float(position),
            )
        except Exception as error:
            print(
                f"关节 {get_alias(sim, joint_handle)} "
                f"恢复失败：{error}"
            )

    try:
        sim.setObjectPose(
            target_handle,
            original_target_pose,
            sim.handle_world,
        )
    except Exception as error:
        print(f"iiwa_target 恢复失败：{error}")

    print("恢复操作结束。")


def execute_safe_joint_motion(
    sim,
    scene_joint_handles,
    start_configuration,
    goal_configuration,
    joint_limits,
):
    """
    从当前构型平滑移动到安全目标构型。

    每一步都在写入前和写入后检查真实机械边界。
    """
    assert_configuration_within_hard_limits(
        start_configuration,
        joint_limits,
        "运动起点",
    )

    assert_configuration_within_hard_limits(
        goal_configuration,
        joint_limits,
        "运动终点",
    )

    if not configuration_within_safe_limits(
        goal_configuration,
        joint_limits,
    ):
        raise RuntimeError(
            "搜索得到的最终构型不在安全范围内，拒绝运动。"
        )

    maximum_delta = max(
        abs(goal - start)
        for start, goal in zip(
            start_configuration,
            goal_configuration,
        )
    )

    steps_from_delta = math.ceil(
        maximum_delta
        / math.radians(MAX_COMMAND_STEP_DEG)
    )

    motion_steps = max(
        MIN_MOTION_STEPS,
        steps_from_delta,
    )

    print("\n" + "=" * 72)
    print("开始执行安全关节运动")
    print("=" * 72)

    print(
        f"最大单关节总变化："
        f"{math.degrees(maximum_delta):.3f}°"
    )
    print(f"运动步数：{motion_steps}")
    print(
        f"理论最大单步变化不超过："
        f"{MAX_COMMAND_STEP_DEG:.3f}°"
    )

    report_interval = max(1, motion_steps // 20)

    for step in range(1, motion_steps + 1):
        ratio = step / motion_steps
        interpolation_ratio = smoothstep(ratio)

        proposed_configuration = [
            start
            + (goal - start) * interpolation_ratio
            for start, goal in zip(
                start_configuration,
                goal_configuration,
            )
        ]

        # 写入前检查
        assert_configuration_within_hard_limits(
            proposed_configuration,
            joint_limits,
            f"第 {step}/{motion_steps} 步写入前",
        )

        # 所有值确认合法后再写入
        for joint_handle, position in zip(
            scene_joint_handles,
            proposed_configuration,
        ):
            sim.setJointPosition(
                joint_handle,
                float(position),
            )

        # 写入后重新读取实际关节值
        actual_configuration = read_current_configuration(
            sim,
            scene_joint_handles,
        )

        assert_configuration_within_hard_limits(
            actual_configuration,
            joint_limits,
            f"第 {step}/{motion_steps} 步写入后",
        )

        if (
            step == 1
            or step == motion_steps
            or step % report_interval == 0
        ):
            minimum_hard_margin = min(
                min(
                    actual - limit["lower"],
                    limit["upper"] - actual,
                )
                for actual, limit in zip(
                    actual_configuration,
                    joint_limits,
                )
            )

            angle_text = ", ".join(
                f"J{index + 1}={math.degrees(value):.1f}°"
                for index, value in enumerate(
                    actual_configuration
                )
            )

            print(
                f"{step:4d}/{motion_steps} "
                f"({ratio * 100:5.1f}%)，"
                f"最小真实边界余量="
                f"{math.degrees(minimum_hard_margin):.2f}°\n"
                f"  {angle_text}"
            )

        time.sleep(STEP_DELAY)


# ============================================================
# 主程序
# ============================================================

def main():
    print("=" * 72)
    print("实际运行文件：")
    print(os.path.abspath(__file__))
    print()
    print("本程序不会调用 sim.setJointMode。")
    print("程序会在每一个运动步骤检查七个关节的真实边界。")
    print("=" * 72)

    client = RemoteAPIClient()
    sim = client.require("sim")
    simIK = client.require("simIK")

    print("\n已连接 CoppeliaSim。")

    if sim.getSimulationState() != sim.simulation_stopped:
        raise RuntimeError(
            "请先停止 CoppeliaSim 仿真，再运行本程序。"
        )

    base_handle = find_object(
        sim,
        ["/iiwa"],
    )

    tip_handle = find_object(
        sim,
        [
            "/iiwa_tip",
            "/iiwa/connection/iiwa_tip",
        ],
    )

    target_handle = find_object(
        sim,
        ["/iiwa_target"],
    )

    scene_joint_handles = collect_chain_joints(
        sim,
        tip_handle,
        base_handle,
    )

    print(
        f"\n检测到 {len(scene_joint_handles)} 个关节。"
    )

    for index, joint_handle in enumerate(
        scene_joint_handles,
        start=1,
    ):
        print(
            f"  J{index}：{get_alias(sim, joint_handle)}"
        )

    if len(scene_joint_handles) != 7:
        raise RuntimeError(
            f"预期找到 7 个关节，实际找到 "
            f"{len(scene_joint_handles)} 个。"
        )

    original_configuration = read_current_configuration(
        sim,
        scene_joint_handles,
    )

    original_target_pose = sim.getObjectPose(
        target_handle,
        sim.handle_world,
    )

    joint_limits = read_joint_limits(
        sim,
        scene_joint_handles,
    )

    # 再次检查当前构型没有越界
    assert_configuration_within_hard_limits(
        original_configuration,
        joint_limits,
        "当前初始构型",
    )

    tip_position = sim.getObjectPosition(
        tip_handle,
        sim.handle_world,
    )

    pregrasp_position = sim.getObjectPosition(
        target_handle,
        sim.handle_world,
    )

    initial_position_error = position_distance(
        tip_position,
        pregrasp_position,
    )

    print()
    print_position(
        "当前 iiwa_tip 位置：",
        tip_position,
    )

    print()
    print_position(
        "预抓取 iiwa_target 位置：",
        pregrasp_position,
    )

    print(
        f"\n初始位置误差："
        f"{initial_position_error * 1000:.3f} mm"
    )

    if initial_position_error > MAX_INITIAL_POSITION_ERROR:
        raise RuntimeError(
            "iiwa_tip 与 iiwa_target 的距离超过 30 mm。\n"
            "请先恢复或重新完成位置 IK。"
        )

    # 构造工具 Z 轴向下的目标姿态
    horizontal_x_axis = get_horizontal_tool_axis(
        sim,
        tip_handle,
    )

    final_target_pose = sim.buildPose(
        pregrasp_position,
        TARGET_Z_AXIS,
        6,
        horizontal_x_axis,
    )

    # 让场景中的 target 显示最终目标姿态
    sim.setObjectPose(
        target_handle,
        final_target_pose,
        sim.handle_world,
    )

    ik_environment = None
    movement_started = False

    try:
        # ----------------------------------------------------
        # 创建独立 IK 环境
        # ----------------------------------------------------

        ik_environment = simIK.createEnvironment()

        ik_group = simIK.createGroup(
            ik_environment,
            "iiwa_safe_top_down_group",
        )

        simIK.setGroupCalculation(
            ik_environment,
            ik_group,
            simIK.method_damped_least_squares,
            0.20,
            300,
        )

        constraints = (
            simIK.constraint_position
            | simIK.constraint_alpha_beta
        )

        (
            ik_element,
            _sim_to_ik_map,
            _ik_to_sim_map,
        ) = simIK.addElementFromScene(
            ik_environment,
            ik_group,
            base_handle,
            tip_handle,
            target_handle,
            constraints,
        )

        simIK.setElementPrecision(
            ik_environment,
            ik_group,
            ik_element,
            [
                0.003,
                math.radians(3.0),
            ],
        )

        # getGroupJoints 对普通串联 7R 机械臂返回基座到末端顺序
        ik_joint_handles = simIK.getGroupJoints(
            ik_environment,
            ik_group,
        )

        print(
            f"\nIK 环境检测到 "
            f"{len(ik_joint_handles)} 个关节。"
        )

        if len(ik_joint_handles) != 7:
            raise RuntimeError(
                "IK 环境中的关节数量不是 7。"
            )

        configure_ik_joint_limits(
            simIK,
            ik_environment,
            ik_group,
            ik_joint_handles,
            original_configuration,
            joint_limits,
        )

        # ----------------------------------------------------
        # 在安全区间内搜索最终构型
        # ----------------------------------------------------

        goal_configuration = search_safe_configuration(
            simIK,
            ik_environment,
            ik_group,
            ik_joint_handles,
            original_configuration,
            joint_limits,
        )

        assert_configuration_within_hard_limits(
            goal_configuration,
            joint_limits,
            "搜索得到的目标构型",
        )

        if not configuration_within_safe_limits(
            goal_configuration,
            joint_limits,
        ):
            raise RuntimeError(
                "目标构型未通过安全边界检查。"
            )

        print("\n" + "=" * 72)
        print("选中的安全目标构型")
        print("=" * 72)

        for index, (value, limit) in enumerate(
            zip(goal_configuration, joint_limits),
            start=1,
        ):
            lower_margin = value - limit["lower"]
            upper_margin = limit["upper"] - value

            print(
                f"J{index}: "
                f"{math.degrees(value):8.3f}°，"
                f"距真实下限 "
                f"{math.degrees(lower_margin):7.3f}°，"
                f"距真实上限 "
                f"{math.degrees(upper_margin):7.3f}°"
            )

        print(
            "\n搜索成功。1 秒后开始执行关节空间平滑运动……"
        )

        time.sleep(1.0)
        movement_started = True

        execute_safe_joint_motion(
            sim,
            scene_joint_handles,
            original_configuration,
            goal_configuration,
            joint_limits,
        )

        # ----------------------------------------------------
        # 最终检查
        # ----------------------------------------------------

        final_configuration = read_current_configuration(
            sim,
            scene_joint_handles,
        )

        assert_configuration_within_hard_limits(
            final_configuration,
            joint_limits,
            "最终构型",
        )

        if not configuration_within_safe_limits(
            final_configuration,
            joint_limits,
        ):
            raise RuntimeError(
                "最终构型虽然没有越过真实边界，"
                "但没有保持所要求的安全余量。"
            )

        final_tip_position = sim.getObjectPosition(
            tip_handle,
            sim.handle_world,
        )

        final_target_position = sim.getObjectPosition(
            target_handle,
            sim.handle_world,
        )

        final_position_error = position_distance(
            final_tip_position,
            final_target_position,
        )

        final_tip_z_axis = get_local_axis(
            sim,
            tip_handle,
            "z",
        )

        final_z_error = angle_between_vectors(
            final_tip_z_axis,
            TARGET_Z_AXIS,
        )

        print("\n" + "=" * 72)
        print("最终检查结果")
        print("=" * 72)

        print(
            f"最终位置误差："
            f"{final_position_error * 1000:.3f} mm"
        )

        print(
            f"最终工具 Z 轴方向误差："
            f"{math.degrees(final_z_error):.3f}°"
        )

        print("\n最终工具 Z 轴世界方向：")
        print(
            f"  X = {final_tip_z_axis[0]: .6f}\n"
            f"  Y = {final_tip_z_axis[1]: .6f}\n"
            f"  Z = {final_tip_z_axis[2]: .6f}"
        )

        print("\n最终七关节状态：")

        for index, (value, limit) in enumerate(
            zip(final_configuration, joint_limits),
            start=1,
        ):
            minimum_margin = min(
                value - limit["lower"],
                limit["upper"] - value,
            )

            print(
                f"  J{index}="
                f"{math.degrees(value):8.3f}°，"
                f"最小真实边界余量="
                f"{math.degrees(minimum_margin):7.3f}°"
            )

        if final_position_error > MAX_FINAL_POSITION_ERROR:
            raise RuntimeError(
                "最终末端位置误差超过 5 mm。"
            )

        if final_z_error > MAX_FINAL_Z_ERROR:
            raise RuntimeError(
                "最终工具 Z 轴方向误差超过 3°。"
            )

        print("\n安全姿态调整成功。")
        print("所有关节均位于真实机械边界内。")
        print(
            f"最终目标构型同时保留了约 "
            f"{SAFETY_MARGIN_DEG:.1f}° 的限位安全余量。"
        )

    except Exception:
        if movement_started:
            restore_scene(
                sim,
                scene_joint_handles,
                original_configuration,
                target_handle,
                original_target_pose,
            )
        else:
            # 未开始运动时只恢复 target
            try:
                sim.setObjectPose(
                    target_handle,
                    original_target_pose,
                    sim.handle_world,
                )
            except Exception:
                pass

        raise

    finally:
        if ik_environment is not None:
            try:
                simIK.eraseEnvironment(
                    ik_environment
                )
                print("\nIK 环境已清理。")
            except Exception as cleanup_error:
                print(
                    f"\nIK 环境清理失败：{cleanup_error}"
                )


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print("\n" + "=" * 72)
        print("程序运行失败")
        print("=" * 72)
        print(error)

        print("\n完整错误调用位置：")
        traceback.print_exc()

        print("\n检查事项：")
        print("1. CoppeliaSim 仿真必须处于停止状态")
        print("2. iiwa_tip 必须位于完整的七关节链末端")
        print("3. iiwa_target 必须位于合理的预抓取位置")
        print("4. 七个关节必须设置有限的真实上下限")
        print("5. 如果没有安全解，请移动方块或提高预抓取高度")
