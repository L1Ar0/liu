import os

from remote_session import RemoteAPIClient


print("当前运行文件：")
print(os.path.abspath(__file__))

client = RemoteAPIClient()
sim = client.require("sim")

joint = sim.getObject(
    "/iiwa/joint",
    {"noError": True},
)

if joint < 0:
    raise RuntimeError(
        "没有找到 /iiwa/joint，请根据场景中的第一个关节名称修改路径。"
    )

raw_mode = sim.getJointMode(joint)
kinematic_constant = sim.jointmode_kinematic

print("\ngetJointMode 返回值：")
print("值：", repr(raw_mode))
print("类型：", type(raw_mode))

print("\nsim.jointmode_kinematic：")
print("值：", repr(kinematic_constant))
print("类型：", type(kinematic_constant))

safe_mode = int(kinematic_constant)

print("\n转换后的 safe_mode：")
print("值：", safe_mode)
print("类型：", type(safe_mode))

sim.setJointMode(
    int(joint),
    safe_mode,
)

print("\nsetJointMode 测试成功。")
