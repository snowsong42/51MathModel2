import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# 读取数据
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "Attachment 2.xlsx")
df = pd.read_excel(file_path, sheet_name=0)
displacement = df["Surface Displacement (mm)"].values

# 采样间隔：10分钟 = 1/6 小时
dt = 1/6  # hour
# 计算速度 (mm/h)
velocity = np.diff(displacement) / dt

# 对速度进行平滑（窗口51点，多项式阶数3），减少噪声对加速度的影响
window = 51    # 必须为奇数
polyorder = 3
velocity_smooth = savgol_filter(velocity, window, polyorder)

# 计算加速度 (mm/h²)
acceleration = np.diff(velocity_smooth) / dt

# 生成时间轴（单位：小时），从第一个速度点开始
time_velocity = np.arange(len(velocity)) * dt
time_acc = np.arange(len(acceleration)) * dt

# 绘图
plt.figure(figsize=(12, 5))
plt.plot(time_acc, acceleration, linewidth=0.8, color='red')
plt.xlabel("Time (hours)", fontsize=12)
plt.ylabel("Acceleration (mm/h²)", fontsize=12)
plt.title("Acceleration Level of Surface Displacement (Question 2)", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
