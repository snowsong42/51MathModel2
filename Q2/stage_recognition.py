import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt, savgol_filter

# ==================== 1. 数据读取与预处理 ====================
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "Attachment 2.xlsx")
df = pd.read_excel(file_path, sheet_name=0)   # 第一行为列名
d = df["Surface Displacement (mm)"].values    # 位移 (mm)
N = len(d)
dt = 10 / 60          # 采样间隔 10 min = 1/6 h
time_h = np.arange(N) * dt                     # 时间 (小时)
# 将零值视为缺失值，进行线性插值（避免速度计算中的虚假跳变）
d_interp = d.copy()
zero_idx = np.where(d == 0)[0]
if len(zero_idx) > 0:
    x = np.arange(N)
    mask = d != 0
    d_interp = np.interp(x, x[mask], d[mask])

# ==================== 2. 速度与加速度计算 ====================
v = np.diff(d_interp) / dt                     # 速度 (mm/h), 长度 N-1
a = np.diff(v) / dt                            # 加速度 (mm/h^2), 长度 N-2
# 中位数滤波平滑加速度（窗口51点，消除高频噪声）
a_smooth = medfilt(a, kernel_size=51)
# 将加速度序列补齐到与原始位移相同长度（便于绘图）
a_full = np.full(N, np.nan)
a_full[1:-1] = a_smooth

# ==================== 3. 阶段转换节点识别（基于平滑速度） ====================
L = 72                    # 窗口大小 (12小时)
H = 36                    # 持久性检验窗口 (6小时)
C_jump = 5.0
C_trans = 0.6
theta_back = 0.5

# 计算滑动窗口的中位数速度与MAD
v_med = np.full_like(v, np.nan)
v_mad = np.full_like(v, np.nan)
for i in range(L, len(v)):
    win = v[i-L+1 : i+1]
    v_med[i] = np.median(win)
    v_mad[i] = np.median(np.abs(win - v_med[i]))

z_score = np.abs(v - v_med) / (v_mad + 1e-6)   # 跳变指标

# 计算阶跃量 R_i 与持久性检验
R = np.zeros(len(v))
retreat = np.ones(len(v))
for i in range(L, len(v)-L):
    if z_score[i] > C_jump:
        continue
    mu1 = np.median(v[i-L+1 : i+1])
    mu2 = np.median(v[i+1 : i+L+1])
    if mu1 < 1e-6:
        continue
    R[i] = (mu2 - mu1) / mu1
    # 持久性：检查未来 H/2 ~ H 个点的速度是否显著回落
    if mu2 > mu1:
        future_win = v[i+H//2 : i+H+1]
        mu_future = np.median(future_win) if len(future_win) > 0 else mu2
        retreat[i] = (mu2 - mu_future) / (mu2 - mu1 + 1e-6)

S = R * (R > 0) * (retreat <= theta_back)     # 综合转换强度
candidate_nodes = np.where(S > C_trans)[0]

# 根据速度水平区分两个阶段节点
def get_stage_nodes(candidates, v, L=72):
    nodes = {"slow2acc": None, "acc2fast": None}
    v_smooth = savgol_filter(v, window_length=151, polyorder=2)   # 用于判读速度水平
    speed_thresh = [0.6, 5.0]   # 低速阈值、快速阈值 (mm/h)
    for i in candidates:
        if i < L or i > len(v)-L:
            continue
        v_before = np.median(v[i-L+1 : i+1])
        v_after  = np.median(v[i+1 : i+L+1])
        if nodes["slow2acc"] is None and v_before <= speed_thresh[0] and v_after <= speed_thresh[1]:
            nodes["slow2acc"] = i   # 节点对应速度序列的索引（对应位移点 i+1? 统一用位移点坐标）
        elif nodes["acc2fast"] is None and v_after > speed_thresh[1]:
            nodes["acc2fast"] = i
            break
    return nodes

stage_nodes = get_stage_nodes(candidate_nodes, v)
node1 = stage_nodes["slow2acc"]    # 速度序列索引，对应位移点索引 = node1+1
node2 = stage_nodes["acc2fast"]

if node1 is not None:
    t_node1 = (node1+1) * dt   # 时间 (h)
    idx_node1 = node1 + 1      # 位移点序号（1-index）
else:
    t_node1, idx_node1 = None, None
if node2 is not None:
    t_node2 = (node2+1) * dt
    idx_node2 = node2 + 1
else:
    t_node2, idx_node2 = None, None



# ==================== 4. 加速度水平图 ====================
plt.figure(figsize=(14, 5))
plt.plot(time_h[1:-1], a_smooth, 'b-', linewidth=0.8, label='Smoothed acceleration')
plt.axhline(0, color='k', linestyle='--', linewidth=0.5)
plt.xlabel('Time (hours)', fontsize=12)
plt.ylabel('Acceleration (mm/h²)', fontsize=12)
plt.title('Acceleration Level of Surface Displacement (Smoothed)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'acceleration_level.png'), dpi=300)
plt.show()

# ==================== 5. 阶段划分直观图 ====================
# 平滑速度用于展示趋势
v_smooth = savgol_filter(v, window_length=151, polyorder=2)
# 创建双轴图
fig, ax1 = plt.subplots(figsize=(14, 7))
ax1.plot(time_h, d_interp, 'k-', linewidth=0.5, label='Displacement (raw)')
ax1.set_xlabel('Time (hours)', fontsize=12)
ax1.set_ylabel('Displacement (mm)', fontsize=12, color='k')
ax1.tick_params(axis='y', labelcolor='k')
# 速度曲线（右轴）
ax2 = ax1.twinx()
ax2.plot(time_h[:-1], v_smooth, 'r-', linewidth=1.2, label='Smoothed velocity (mm/h)')
ax2.set_ylabel('Velocity (mm/h)', fontsize=12, color='r')
ax2.tick_params(axis='y', labelcolor='r')
# 标记转换节点
if node1 is not None:
    ax1.axvline(x=t_node1, color='g', linestyle='--', linewidth=1.5, alpha=0.7, label='Node 1: Slow→Acc')
if node2 is not None:
    ax1.axvline(x=t_node2, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='Node 2: Acc→Fast')
# 阶段背景着色
if node1 is not None and node2 is not None:
    ax1.axvspan(0, t_node1, alpha=0.1, color='blue', label='Slow stage')
    ax1.axvspan(t_node1, t_node2, alpha=0.1, color='green', label='Acceleration stage')
    ax1.axvspan(t_node2, time_h[-1], alpha=0.1, color='red', label='Fast stage')
# 合并图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
plt.title('Three-stage Deformation and Transition Nodes', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'stage_division.png'), dpi=300)
plt.show()
