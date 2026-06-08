import numpy as np
import matplotlib.pyplot as plt
import os

# 1. 读取保存的投影数据
data_path = "./projection_frames/trajectories_2d.npy"
if not os.path.exists(data_path):
    print(f"未找到数据文件：{data_path}，请先运行 Taichi 仿真代码生成数据！")
    exit()

data = np.load(data_path)  # Shape: (100, 100, 2)
n_frames, n_particles, _ = data.shape

# 2. 创建画布 (对齐当时 800x800 的正方形视口)
plt.figure(figsize=(8, 8))
plt.title("Particle Trajectories in Camera 2D Projection", fontsize=14)

# 3. 遍历每一个粒子，画出它的运动轨迹
for p_idx in range(n_particles):
    # 提取第 p_idx 个粒子在所有帧的 x 和 y 坐标
    # 注意：Taichi 的屏幕坐标 Y 轴向上，而 Matplotlib 默认 Y 轴也向上，刚好契合
    x_traj = data[:, p_idx, 0]
    y_traj = data[:, p_idx, 1]
    
    # 过滤掉不可见（标记为 -1）的帧
    valid_mask = (x_traj >= 0) & (y_traj >= 0)
    if not np.any(valid_mask):
        continue
        
    # 画出轨迹折线，alpha 设置透明度防止 100 根线糊在一起
    plt.plot(x_traj[valid_mask], y_traj[valid_mask], color='cyan', alpha=0.4, linewidth=1.5)
    
    # 顺便用红点标出它们的终点位置（最后一帧）
    plt.scatter(x_traj[-1], y_traj[-1], color='red', s=5, alpha=0.6)

# 4. 美化图表
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("X Screen Coordinate")
plt.ylabel("Y Screen Coordinate")
plt.grid(True, linestyle="--", alpha=0.5)
plt.gca().set_facecolor('#111122') # 设置成和 Taichi 渲染类似的深色背景

# 保存图像
plt.savefig("./projection_frames/trajectories_static.png", dpi=300)
plt.show()