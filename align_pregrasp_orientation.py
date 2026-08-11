import math
import time
from typing import Any, Iterable

from coppeliasim_zmqremoteapi_client import RemoteAPIClient


# ============================================================
# 可调整参数
# ============================================================

# 将姿态调整分成更多小步，降低单步旋转量
ROTATION_STEPS = 360

# 每一步暂停时间，单位：秒
STEP_DELAY = 0.015

# 阻尼最小二乘 IK 参数
IK_DAMPING = 0.25
IK_MAX_ITERATIONS = 300

# 开始姿态调整前，tip 与 target 最大允许距离
MAX_INITIAL_POSITION_ERROR = 0.03       # 30 mm

# 最终允许误差
MAX_FINAL_POSITION_ERROR = 0.005        # 5 mm
MAX_Z_ALIGNMENT_ERROR = math.radians(3.0)

# 工具局部 Z 轴最终指向世界坐标系 -Z
TARGET_Z_AXIS = [0.0, 0.0, -1.0]


# ============================================================
# 通用辅助函数
# ============================================================

def find_object(sim, possible_paths: Iterable[str]) -> int:
    """尝试多个对象路径，返回第一个找到的对象句柄。"""
    attempted_paths = []

    for path in possible_paths:
        attempted_paths.append(path)

        handle = sim.getObject(
            path,
            {"noError": True},
        )

        if handle is not None and handle >= 0:
            print(f"找到对象：{path}")
            return int(handle)

    raise RuntimeError(
        "无法找到对象，尝试过以下路径：\n"
        + "\n".join(attempted_paths)
    )


def convert_to_int(value: Any, value_name: str) -> int:
    """
    将 Remote API 返回的数值转换为普通 Python int。

    某些版本或接口包装可能返回：
        2
        [2]
        (2,)
        (2, 0)
        {"mode": 2}
        {"jointMode": 2}

    sim.setJointMode 的第二个参数必须是数值。
    """
    original_value = value

    # 处理 NumPy 等标量类型
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass

    # 处理列表或元组
    while isinstance(value, (list, tuple)):
        if len(value) == 0:
            raise TypeError(
                f"{value_name} 返回了空列表或空元组。"
            )

        # 如果返回多个值，第一个通常是主返回值 jointMode
        value = value[0]

    # 处理字典
    if isinstance(value, dict):
        possible_keys = [
            "jointMode",
            "mode",
            "value",
            "result",
        ]

        extracted = False

        for key in possible_keys:
            if key in value:
                value = value[key]
                extracted = True
                break

        if not extracted:
            raise TypeError(
                f"无法从字典中提取 {value_name}：{value}"
            )

    try:
        return int(value)

    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(
            f"{value_name} 无法转换成整数。\n"
            f"原始值：{original_value!r}\n"
            f"原始类型：{type(original_value)}"
        ) from error


def collect_chain_joints(
    sim,
    tip_handle: int,
    base_handle: int,
) -> list[int]:
    """
    从 iiwa_tip 沿父对象向上查找，直到 iiwa 基座，
    收集运动链中的全部关节。
    """
    joints_from_tip_to_base = []
    current_handle = tip_handle

    visited_handles = set()

    while current_handle != base_handle:
        if current_handle in visited_handles:
            raise RuntimeError(
                "场景层级中检测到循环父子关系。"
            )

        visited_handles.add(current_handle)

        parent_handle = sim.getObjectParent(
            current_handle
        )

        if parent_handle is None or parent_handle < 0:
            raise RuntimeError(
                "从 iiwa_tip 向上查找时到达了场景根节点，"
                "但没有找到 /iiwa。\n"
                "请检查 iiwa_tip 是否位于机械臂末端层级下。"
            )

        object_type = sim.getObjectType(
            parent_handle
        )

        if object_type == sim.sceneobject_joint:
            joints_from_tip_to_base.append(
                int(parent_handle)
            )

        current_handle = parent_handle

    # 原顺序是末端到基座，反转为基座到末端
    joints_from_tip_to_base.reverse()

    return joints_from_tip_to_base


def vector_norm(vector: list[float]) -> float:
    """计算三维向量长度。"""
    return math.sqrt(
        vector[0] ** 2
        + vector[1] ** 2
        + vector[2] ** 2
    )


def normalize_vector(
    vector: list[float],
    fallback: list[float] | None = None,
) -> list[float]:
    """归一化三维向量。"""
    norm = vector_norm(vector)

    if norm < 1e-10:
        if fallback is not None:
            return fallback.copy()

        raise ValueError(
            f"无法归一化接近零的向量：{vector}"
        )

    return [
        vector[0] / norm,
        vector[1] / norm,
        vector[2] / norm,
    ]


def clamp(value: float, minimum: float, maximum: float) -> float:
    """把数值限制在指定区间。"""
    return max(
        minimum,
        min(maximum, value),
    )


def dot_product(
    vector_a: list[float],
    vector_b: list[float],
) -> float:
    """计算两个三维向量的点积。"""
    return (
        vector_a[0] * vector_b[0]
        + vector_a[1] * vector_b[1]
        + vector_a[2] * vector_b[2]
    )


def angle_between_vectors(
    vector_a: list[float],
    vector_b: list[float],
) -> float:
    """计算两个向量之间的夹角，返回弧度。"""
    normalized_a = normalize_vector(vector_a)
    normalized_b = normalize_vector(vector_b)

    cosine_value = dot_product(
        normalized_a,
        normalized_b,
    )

    cosine_value = clamp(
        cosine_value,
        -1.0,
        1.0,
    )

    return math.acos(cosine_value)


def position_distance(
    position_a: list[float],
    position_b: list[float],
) -> float:
    """计算两个三维位置之间的距离。"""
    return math.sqrt(
        (position_a[0] - position_b[0]) ** 2
        + (position_a[1] - position_b[1]) ** 2
        + (position_a[2] - position_b[2]) ** 2
    )


def get_local_axis(
    sim,
    object_handle: int,
    axis_name: str,
) -> list[float]:
    """
    获取对象局部坐标轴在世界坐标系中的方向。

    CoppeliaSim 12元素矩阵排列：
        [Xx, Yx, Zx, Px,
         Xy, Yy, Zy, Py,
         Xz, Yz, Zz, Pz]
    """
    matrix = sim.getObjectMatrix(
        object_handle,
        sim.handle_world,
    )

    if axis_name.lower() == "x":
        axis = [
            matrix[0],
            matrix[4],
            matrix[8],
        ]

    elif axis_name.lower() == "y":
        axis = [
            matrix[1],
            matrix[5],
            matrix[9],
        ]

    elif axis_name.lower() == "z":
        axis = [
            matrix[2],
            matrix[6],
            matrix[10],
        ]

    else:
        raise ValueError(
            f"不支持的坐标轴名称：{axis_name}"
        )

    return normalize_vector(axis)


def get_horizontal_x_axis(
    sim,
    tip_handle: int,
) -> list[float]:
    """
    获取当前 iiwa_tip 的局部 X 轴，并将其投影到世界水平面。

    这样能够尽量保留当前腕部绕竖直轴的方向，
    避免强制对准世界 X 轴产生不必要的腕部旋转。
    """
    current_x_axis = get_local_axis(
        sim,
        tip_handle,
        "x",
    )

    horizontal_projection = [
        current_x_axis[0],
        current_x_axis[1],
        0.0,
    ]

    horizontal_length = vector_norm(
        horizontal_projection
    )

    if horizontal_length < 1e-6:
        # 如果当前 X 轴几乎竖直，则尝试使用当前 Y 轴的水平投影
        current_y_axis = get_local_axis(
            sim,
            tip_handle,
            "y",
        )

        horizontal_projection = [
            current_y_axis[0],
            current_y_axis[1],
            0.0,
        ]

        horizontal_length = vector_norm(
            horizontal_projection
        )

    if horizontal_length < 1e-6:
        # 最后的备用方向
        return [1.0, 0.0, 0.0]

    return normalize_vector(
        horizontal_projection
    )


def print_position(
    title: str,
    position: list[float],
) -> None:
    """打印三维位置。"""
    print(title)
    print(f"  x = {position[0]: .5f} m")
    print(f"  y = {position[1]: .5f} m")
    print(f"  z = {position[2]: .5f} m")


def format_precision(precision: Any) -> str:
    """格式化 IK 返回的精度信息。"""
    if isinstance(precision, (list, tuple)):
        if len(precision) >= 2:
            try:
                linear_error = float(precision[0])
                angular_error = float(precision[1])

                return (
                    f"线性精度={linear_error:.6f} m，"
                    f"角度精度="
                    f"{math.degrees(angular_error):.3f}°"
                )
            except (TypeError, ValueError):
                pass

    return repr(precision)


def decode_ik_flags(
    simIK,
    flags: Any,
) -> list[str]:
    """把 IK 失败标志转换为可读信息。"""
    try:
        flags_number = convert_to_int(
            flags,
            "IK flags",
        )
    except Exception:
        return [f"无法解析 flags：{flags!r}"]

    flag_names = [
        ("calc_notperformed", "IK 未执行"),
        ("calc_cannotinvert", "雅可比矩阵无法求逆"),
        ("calc_notwithintolerance", "未达到设定精度"),
        ("calc_stepstoobig", "IK 单步变化过大"),
        ("calc_limithit", "关节达到或超过限制"),
    ]

    results = []

    for constant_name, description in flag_names:
        constant_value = getattr(
            simIK,
            constant_name,
            None,
        )

        if constant_value is None:
            continue

        try:
            constant_number = convert_to_int(
                constant_value,
                constant_name,
            )
        except Exception:
            continue

        if flags_number & constant_number:
            results.append(description)

    if not results:
        results.append(
            f"未识别的 IK flags={flags_number}"
        )

    return results


def get_joint_alias(sim, joint_handle: int) -> str:
    """获取关节名称，失败时返回句柄。"""
    try:
        return sim.getObjectAlias(
            joint_handle
        )
    except Exception:
        return f"handle={joint_handle}"


def describe_joint_limit_hits(
    simIK,
    ik_environment: int,
    ik_group: int,
) -> str:
    """读取最近一次 IK 中可能出现的关节限制问题。"""
    try:
        ik_joint_handles, overshoots = (
            simIK.getGroupJointLimitHits(
                ik_environment,
                ik_group,
            )
        )

        if not ik_joint_handles:
            return "未检测到明确的关节限制碰撞。"

        lines = [
            "检测到以下 IK 关节达到限制："
        ]

        for index, ik_joint_handle in enumerate(
            ik_joint_handles
        ):
            overshoot = None

            if index < len(overshoots):
                overshoot = overshoots[index]

            lines.append(
                f"  IK关节句柄={ik_joint_handle}, "
                f"越界量={overshoot}"
            )

        return "\n".join(lines)

    except Exception as error:
        return (
            "无法读取关节限制信息："
            f"{error}"
        )


def restore_scene_state(
    sim,
    joint_handles: list[int],
    original_joint_positions: list[float],
    original_joint_modes: list[int],
    target_handle: int,
    original_target_pose: list[float],
) -> None:
    """
    尽可能恢复机械臂和 target。
    恢复过程中的错误只打印，不覆盖原始 IK 错误。
    """
    print("\n正在恢复运行前的场景状态……")

    # 此时关节通常仍为 kinematic，
    # 先恢复位置比较安全
    for index, (
        joint_handle,
        joint_position,
    ) in enumerate(
        zip(
            joint_handles,
            original_joint_positions,
        ),
        start=1,
    ):
        try:
            sim.setJointPosition(
                joint_handle,
                float(joint_position),
            )

        except Exception as restore_error:
            print(
                f"关节 {index} 位置恢复失败："
                f"{restore_error}"
            )

    # 再恢复关节模式
    for index, (
        joint_handle,
        joint_mode,
    ) in enumerate(
        zip(
            joint_handles,
            original_joint_modes,
        ),
        start=1,
    ):
        try:
            safe_mode = convert_to_int(
                joint_mode,
                f"关节 {index} 模式",
            )

            sim.setJointMode(
                joint_handle,
                safe_mode,
            )

        except Exception as restore_error:
            print(
                f"关节 {index} 模式恢复失败："
                f"value={joint_mode!r}, "
                f"type={type(joint_mode)}, "
                f"error={restore_error}"
            )

    try:
        sim.setObjectPose(
            target_handle,
            original_target_pose,
            sim.handle_world,
        )

    except Exception as restore_error:
        print(
            "iiwa_target 位姿恢复失败："
            f"{restore_error}"
        )

    print("场景恢复操作结束。")


# ============================================================
# 主程序
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

    simulation_state = sim.getSimulationState()

    if simulation_state != sim.simulation_stopped:
        raise RuntimeError(
            "当前仿真未停止。\n"
            "请先点击 CoppeliaSim 顶部的停止按钮，"
            "再运行本程序。"
        )

    # --------------------------------------------------------
    # 2. 获取对象
    # --------------------------------------------------------

    base_handle = find_object(
        sim,
        [
            "/iiwa",
        ],
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
        [
            "/iiwa_target",
        ],
    )

    # --------------------------------------------------------
    # 3. 收集七个关节
    # --------------------------------------------------------

    joint_handles = collect_chain_joints(
        sim,
        tip_handle,
        base_handle,
    )

    print(
        f"\n检测到的机械臂关节数量："
        f"{len(joint_handles)}"
    )

    for index, joint_handle in enumerate(
        joint_handles,
        start=1,
    ):
        print(
            f"  关节 {index}："
            f"{get_joint_alias(sim, joint_handle)}"
        )

    if len(joint_handles) != 7:
        raise RuntimeError(
            f"预期检测到 7 个关节，"
            f"实际检测到 {len(joint_handles)} 个。\n"
            "请检查 iiwa_tip 是否位于完整七关节链末端。"
        )

    # --------------------------------------------------------
    # 4. 保存原始状态
    # --------------------------------------------------------

    original_joint_positions = []

    for joint_handle in joint_handles:
        joint_position = sim.getJointPosition(
            joint_handle
        )

        original_joint_positions.append(
            float(joint_position)
        )

    original_joint_modes = []

    for index, joint_handle in enumerate(
        joint_handles,
        start=1,
    ):
        raw_mode = sim.getJointMode(
            joint_handle
        )

        safe_mode = convert_to_int(
            raw_mode,
            f"关节 {index} 的 getJointMode 返回值",
        )

        original_joint_modes.append(
            safe_mode
        )

        print(
            f"  关节 {index} 原始模式："
            f"{safe_mode}"
        )

    original_target_pose = sim.getObjectPose(
        target_handle,
        sim.handle_world,
    )

    # --------------------------------------------------------
    # 5. 切换为运动学模式
    # --------------------------------------------------------

    kinematic_mode = convert_to_int(
        sim.jointmode_kinematic,
        "sim.jointmode_kinematic",
    )

    print("\n正在将七个关节切换为 kinematic 模式……")

    for joint_handle in joint_handles:
        sim.setJointMode(
            joint_handle,
            kinematic_mode,
        )

    # --------------------------------------------------------
    # 6. 获取起始状态
    # --------------------------------------------------------

    start_tip_pose = sim.getObjectPose(
        tip_handle,
        sim.handle_world,
    )

    start_tip_position = start_tip_pose[:3]

    target_position = sim.getObjectPosition(
        target_handle,
        sim.handle_world,
    )

    initial_position_error = position_distance(
        start_tip_position,
        target_position,
    )

    print()
    print_position(
        "iiwa_tip 当前世界坐标：",
        start_tip_position,
    )

    print()
    print_position(
        "iiwa_target 当前世界坐标：",
        target_position,
    )

    print(
        "\n当前 tip 与 target 的位置误差："
        f"{initial_position_error * 1000:.3f} mm"
    )

    if initial_position_error > MAX_INITIAL_POSITION_ERROR:
        restore_scene_state(
            sim,
            joint_handles,
            original_joint_positions,
            original_joint_modes,
            target_handle,
            original_target_pose,
        )

        raise RuntimeError(
            "iiwa_tip 距离 iiwa_target 超过 30 mm。\n"
            "请先重新运行上一阶段的位置 IK。"
        )

    # --------------------------------------------------------
    # 7. 构造自上而下目标姿态
    # --------------------------------------------------------

    current_tip_z_axis = get_local_axis(
        sim,
        tip_handle,
        "z",
    )

    final_x_axis = get_horizontal_x_axis(
        sim,
        tip_handle,
    )

    final_target_pose = sim.buildPose(
        target_position,
        TARGET_Z_AXIS,
        6,
        final_x_axis,
    )

    required_z_rotation = angle_between_vectors(
        current_tip_z_axis,
        TARGET_Z_AXIS,
    )

    print("\n当前 iiwa_tip 局部 Z 轴：")
    print(
        f"  [{current_tip_z_axis[0]: .5f}, "
        f"{current_tip_z_axis[1]: .5f}, "
        f"{current_tip_z_axis[2]: .5f}]"
    )

    print("最终目标 Z 轴：")
    print("  [0.00000, 0.00000, -1.00000]")

    print("用于构造目标姿态的水平 X 轴：")
    print(
        f"  [{final_x_axis[0]: .5f}, "
        f"{final_x_axis[1]: .5f}, "
        f"{final_x_axis[2]: .5f}]"
    )

    print(
        "末端 Z 轴预计需要转动："
        f"{math.degrees(required_z_rotation):.3f}°"
    )

    # --------------------------------------------------------
    # 8. 创建 IK 环境
    # --------------------------------------------------------

    ik_environment = None
    movement_completed = False

    try:
        ik_environment = simIK.createEnvironment()

        ik_group = simIK.createGroup(
            ik_environment,
            "iiwa_top_down_ik_group",
        )

        simIK.setGroupCalculation(
            ik_environment,
            ik_group,
            simIK.method_damped_least_squares,
            IK_DAMPING,
            IK_MAX_ITERATIONS,
        )

        # 关键修改：
        #
        # 位置约束：
        #   X、Y、Z 必须跟随 target
        #
        # Alpha/Beta 约束：
        #   tip 和 target 的 Z 轴必须重合
        #
        # 不加入 Gamma：
        #   允许腕部绕工具 Z 轴自由转动
        constraints = (
            simIK.constraint_position
            | simIK.constraint_alpha_beta
        )

        (
            ik_element,
            sim_to_ik_map,
            ik_to_sim_map,
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

        print(
            "\nIK 元素创建成功："
            f"{ik_element}"
        )

        print(
            "当前约束："
            "位置 XYZ + Z轴方向，"
            "不强制绕Z轴角度。"
        )

        # ----------------------------------------------------
        # 9. 让 target 从 tip 当前位姿开始
        # ----------------------------------------------------

        sim.setObjectPose(
            target_handle,
            start_tip_pose,
            sim.handle_world,
        )

        result, flags, precision = simIK.handleGroup(
            ik_environment,
            ik_group,
            {
                "syncWorlds": True,
            },
        )

        if result != simIK.result_success:
            failure_reasons = decode_ik_flags(
                simIK,
                flags,
            )

            raise RuntimeError(
                "姿态 IK 初始化失败。\n"
                f"flags={flags}\n"
                f"原因：{'；'.join(failure_reasons)}\n"
                f"{format_precision(precision)}"
            )

        print("姿态 IK 初始化成功。")
        print("1 秒后开始调整末端方向……")

        time.sleep(1.0)

        # ----------------------------------------------------
        # 10. 分步插值姿态
        # ----------------------------------------------------

        report_interval = max(
            1,
            ROTATION_STEPS // 10,
        )

        for step in range(
            1,
            ROTATION_STEPS + 1,
        ):
            ratio = step / ROTATION_STEPS

            current_target_pose = sim.interpolatePoses(
                start_tip_pose,
                final_target_pose,
                ratio,
            )

            sim.setObjectPose(
                target_handle,
                current_target_pose,
                sim.handle_world,
            )

            result, flags, precision = simIK.handleGroup(
                ik_environment,
                ik_group,
                {
                    "syncWorlds": True,
                },
            )

            if result != simIK.result_success:
                failure_reasons = decode_ik_flags(
                    simIK,
                    flags,
                )

                joint_limit_description = (
                    describe_joint_limit_hits(
                        simIK,
                        ik_environment,
                        ik_group,
                    )
                )

                raise RuntimeError(
                    "\n姿态 IK 求解失败。\n"
                    f"失败步骤：{step}/{ROTATION_STEPS}\n"
                    f"完成比例：{ratio * 100:.2f}%\n"
                    f"flags={flags}\n"
                    f"原因：{'；'.join(failure_reasons)}\n"
                    f"{format_precision(precision)}\n"
                    f"{joint_limit_description}"
                )

            if (
                step == 1
                or step == ROTATION_STEPS
                or step % report_interval == 0
            ):
                current_tip_z_axis = get_local_axis(
                    sim,
                    tip_handle,
                    "z",
                )

                current_z_error = (
                    angle_between_vectors(
                        current_tip_z_axis,
                        TARGET_Z_AXIS,
                    )
                )

                print(
                    f"姿态调整进度："
                    f"{step}/{ROTATION_STEPS} "
                    f"({ratio * 100:.0f}%)，"
                    f"Z轴误差="
                    f"{math.degrees(current_z_error):.2f}°"
                )

            time.sleep(STEP_DELAY)

        movement_completed = True

        # ----------------------------------------------------
        # 11. 检查最终结果
        # ----------------------------------------------------

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

        final_z_alignment_error = angle_between_vectors(
            final_tip_z_axis,
            TARGET_Z_AXIS,
        )

        print("\n========== 姿态 IK 最终结果 ==========")

        print()
        print_position(
            "最终 iiwa_tip 位置：",
            final_tip_position,
        )

        print()
        print_position(
            "最终 iiwa_target 位置：",
            final_target_position,
        )

        print(
            "\n最终位置误差："
            f"{final_position_error * 1000:.3f} mm"
        )

        print(
            "最终 Z 轴方向误差："
            f"{math.degrees(final_z_alignment_error):.3f}°"
        )

        print("\n最终 iiwa_tip 局部 Z 轴世界方向：")
        print(
            f"  X = {final_tip_z_axis[0]: .6f}\n"
            f"  Y = {final_tip_z_axis[1]: .6f}\n"
            f"  Z = {final_tip_z_axis[2]: .6f}"
        )

        print("\n预期接近：")
        print("  X = 0")
        print("  Y = 0")
        print("  Z = -1")

        if (
            final_position_error
            <= MAX_FINAL_POSITION_ERROR
            and final_z_alignment_error
            <= MAX_Z_ALIGNMENT_ERROR
        ):
            print(
                "\n自上而下抓取姿态设置成功。"
            )

        else:
            print(
                "\n机械臂已经接近目标姿态，"
                "但最终误差超过设定范围。"
            )

    except Exception as original_error:
        print("\n========== 原始失败原因 ==========")
        print(repr(original_error))
        print(str(original_error))

        restore_scene_state(
            sim,
            joint_handles,
            original_joint_positions,
            original_joint_modes,
            target_handle,
            original_target_pose,
        )

        # 保留真正的 IK 错误，
        # 不让恢复阶段的错误将其覆盖
        raise RuntimeError(
            "姿态调整失败，详情见上方的原始失败原因。"
        ) from original_error

    finally:
        if ik_environment is not None:
            try:
                simIK.eraseEnvironment(
                    ik_environment
                )
                print("\nIK 环境已经清理。")

            except Exception as cleanup_error:
                print(
                    "\nIK 环境清理失败："
                    f"{cleanup_error}"
                )

    if movement_completed:
        print(
            "机械臂保持在自上而下的预抓取姿态。"
        )
        print(
            "七个关节保持为 kinematic 模式。"
        )
        print(
            "iiwa_target 保持在最终预抓取位姿。"
        )


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print("\n程序运行失败：")
        print(error)

        print("\n请检查：")
        print("1. CoppeliaSim 仿真是否已经停止")
        print("2. 上一阶段的位置 IK 是否已经完成")
        print("3. 场景中是否存在 /iiwa")
        print("4. 场景中是否存在 iiwa_tip")
        print("5. 场景中是否存在 iiwa_target")
        print("6. iiwa_tip 是否位于完整的七关节链末端")
        print("7. iiwa_target 是否位于机械臂可达范围内")
        print("8. simIK 插件是否正常加载")