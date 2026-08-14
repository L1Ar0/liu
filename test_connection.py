from remote_session import RemoteAPIClient


def main() -> None:
    """测试 Python 是否能够连接并控制 CoppeliaSim。"""

    print("正在连接 CoppeliaSim...")

    client = RemoteAPIClient()
    sim = client.require("sim")

    print("连接成功，准备启动仿真。")

    # 使用同步步进模式：
    # Python 每调用一次 sim.step()，仿真才前进一步。
    sim.setStepping(True)
    sim.startSimulation()

    try:
        while sim.getSimulationTime() < 1.0:
            simulation_time = sim.getSimulationTime()
            print(f"当前仿真时间：{simulation_time:.2f} 秒")
            sim.step()
    finally:
        # 即使程序中途报错，也尝试停止仿真。
        sim.stopSimulation()

    print("测试完成，仿真已经停止。")


if __name__ == "__main__":
    main()
