import math
import time

from remote_session import RemoteAPIClient


def main() -> None:
    client = RemoteAPIClient()
    sim = client.require("sim")

    robot_handle = sim.getObject("/iiwa")

    joint_handles = sim.getObjectsInTree(
        robot_handle,
        sim.sceneobject_joint,
        0,
    )

    print(f"找到 {len(joint_handles)} 个关节。")

    for index, joint_handle in enumerate(joint_handles, start=1):
        name = sim.getObjectAlias(joint_handle)
        angle = math.degrees(sim.getJointPosition(joint_handle))
        print(f"关节 {index}：{name}，当前角度：{angle:.2f}°")

    if len(joint_handles) < 2:
        raise RuntimeError("没有找到足够的机械臂关节。")

    joint1 = joint_handles[0]
    joint2 = joint_handles[1]

    # 先让机械臂弯曲，便于观察底座旋转。
    sim.setJointPosition(joint2, math.radians(-35.0))
    sim.setJointPosition(joint1, math.radians(0.0))

    print("\n初始姿态：joint1 = 0°")
    time.sleep(3)

    # joint1 绕底座转动。
    sim.setJointPosition(joint1, math.radians(60.0))

    actual_angle = math.degrees(sim.getJointPosition(joint1))
    print(f"转动后：joint1 = {actual_angle:.2f}°")

    time.sleep(3)


if __name__ == "__main__":
    main()
