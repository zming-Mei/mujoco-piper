import ikpy.chain
import ikpy.utils.plot as plot_utils
import matplotlib.pyplot as plt
import numpy as np

def main():
    # 加载机器人模型
    my_chain = ikpy.chain.Chain.from_urdf_file(
        "/home/zming/mujoco/piper/piper_description/urdf/piper_no_gripper_description.urdf"
    )

    # 目标末端位置
    ee_pos = [-0.13, 0.5, 0.1] 

    # 初始关节角（全零）
    initial_angles = [0] * len(my_chain.links)

    # 计算目标关节角（仅位置约束）
    target_angles = my_chain.inverse_kinematics(target_position=ee_pos)

    # 插值参数
    num_steps = 50

    # 在关节空间线性插值
    interpolated_path = np.linspace(initial_angles, target_angles, num_steps)

    # 创建一个窗口和坐标轴（只创建一次）
    fig, ax = plot_utils.init_3d_figure()
    plt.ion()

    # 动画循环
    for i, joint_angles in enumerate(interpolated_path):
        ax.clear()
        my_chain.plot(joint_angles, ax)

        # 添加红色目标点（固定位置）
        ax.scatter([ee_pos[0]], [ee_pos[1]], [ee_pos[2]], color='red', s=100, label='Target' if i == 0 else "")

        ax.set_title(f"Step {i+1}/{num_steps}")
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_zlim(0, 0.6)

        # 只在第一帧添加图例，避免重复
        if i == 0:
            ax.legend()

        plt.draw()
        plt.pause(0.05)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()