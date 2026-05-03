"""
爆破影响分析 (独立模块)
======================
从 ap4.xlsx 提取数据，独立运行指数衰减累积爆破模型

爆破位移模型:
  每次爆破产生初始位移 A0 = beta * Q / R^3
  之后以速率 lambda 指数衰减: D_blast(t) = sum(A0_j * exp(-lambda * (t - t_j)))
  等效速率: v_blast = lambda * D_blast
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ============================================================
# 自动确定路径
# ============================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# ============================================================
# 参数
# ============================================================
filename = '../ap4.xlsx'
sheet_name = '训练集'

# 阶段划分参数 (与 simulate.py 一致)
AUTO_PHASE = True
SMOOTH_WIN = 50
RATE_THRESH1 = 0.02
RATE_THRESH2 = 0.10
CONTINUOUS_N = 10

# 爆破参数
beta_1, beta_2, beta_3 = 0.1, 0.5, 1.0
lambda_1 = 1.0 / 200
lambda_2 = 1.0 / 300
lambda_3 = 1.0 / 500
beta_list  = [beta_1, beta_2, beta_3]
lambda_list = [lambda_1, lambda_2, lambda_3]

# ============================================================
# 数据读取
# ============================================================
print('=' * 72)
print('  爆破影响分析 (独立模块)')
print('=' * 72)

df = pd.read_excel(filename, sheet_name=sheet_name)
data_mat = df.iloc[:, 1:].values
n = len(data_mat)

D_real = data_mat[:, 0].copy()
dist   = data_mat[:, 4].copy()
charge = data_mat[:, 5].copy()

# 缺失值处理
dist[np.isnan(dist)] = 0
charge[np.isnan(charge)] = 0

t_axis = np.arange(n, dtype=float)

# ============================================================
# 阶段划分
# ============================================================
if AUTO_PHASE:
    vel = np.diff(D_real, prepend=D_real[0])
    vel_smooth = pd.Series(vel).rolling(window=SMOOTH_WIN, center=True, min_periods=1).mean().values

    b1, b2 = n, n
    exceed1 = vel_smooth > RATE_THRESH1
    for i in range(n - CONTINUOUS_N + 1):
        if np.all(exceed1[i:i + CONTINUOUS_N]):
            b1 = i
            break
    exceed2 = vel_smooth > RATE_THRESH2
    for i in range(n - CONTINUOUS_N + 1):
        if np.all(exceed2[i:i + CONTINUOUS_N]):
            b2 = i
            break
    if b1 >= b2:
        b1 = max(1, b2 - 200)

b1 = max(1, min(n, b1))
b2 = max(b1 + 10, min(n, b2))
slices = [slice(0, b1), slice(b1, b2), slice(b2, n)]

print(f'阶段划分: 0~{b1-1} | {b1}~{b2-1} | {b2}~{n-1}')

# ============================================================
# 爆破模型计算
# ============================================================
blast_accum = np.zeros(n)   # 累积位移
v_blast_all = np.zeros(n)   # 等效速率
A0_all = np.zeros(n)        # 每次爆破的初始幅值

for ph in range(3):
    idx   = slices[ph]
    Nph   = idx.stop - idx.start
    lam   = lambda_list[ph]
    beta  = beta_list[ph]

    dist_ph   = dist[idx]
    charge_ph = charge[idx]

    state = 0.0
    for i in range(Nph):
        state *= np.exp(-lam)          # 先衰减

        if dist_ph[i] > 0 and charge_ph[i] > 0:
            A0 = beta * charge_ph[i] / (max(dist_ph[i], 0.1) ** 3)
            state += A0
            A0_all[idx.start + i] = A0

        blast_accum[idx.start + i] = state
        v_blast_all[idx.start + i] = state * lam

# 统计
total_blast_disp = blast_accum[-1]
max_blast_disp = np.max(blast_accum)
max_blast_rate = np.max(v_blast_all)
blast_event_count = np.sum(A0_all > 0)

print(f'\n--- 爆破统计 ---')
print(f'  爆破事件数: {blast_event_count}')
print(f'  最终累积位移: {total_blast_disp:.4f} mm')
print(f'  最大累积位移: {max_blast_disp:.4f} mm')
print(f'  最大等效速率: {max_blast_rate:.6f} mm/10min')

# ============================================================
# 绘图
# ============================================================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle('爆破效应分析 (指数衰减累积模型)', fontsize=14, fontweight='bold')

# (a) 爆破累积位移
ax_a = axes[0, 0]
ax_a.plot(t_axis, blast_accum, 'g-', linewidth=1.2, label='爆破累积位移')
ax_a.fill_between(t_axis, 0, blast_accum, color='green', alpha=0.15)
ax_a.axvline(b1 - 0.5, color='gray', linestyle='--', linewidth=0.8, label='阶段边界')
ax_a.axvline(b2 - 0.5, color='gray', linestyle='--', linewidth=0.8)
ax_a.set_xlabel('时间步 (10min/步)')
ax_a.set_ylabel('累积位移 (mm)')
ax_a.set_title('(a) 爆破累积位移')
ax_a.legend(loc='upper left')
ax_a.grid(True)

# (b) 爆破等效速率
ax_b = axes[0, 1]
ax_b.plot(t_axis, v_blast_all, 'g-', linewidth=1.2, label='爆破等效速率')
ax_b.axvline(b1 - 0.5, color='gray', linestyle='--', linewidth=0.8)
ax_b.axvline(b2 - 0.5, color='gray', linestyle='--', linewidth=0.8)
ax_b.set_xlabel('时间步 (10min/步)')
ax_b.set_ylabel('速率 (mm/10min)')
ax_b.set_title('(b) 爆破等效速率 v = lambda * D_blast')
ax_b.legend(loc='upper left')
ax_b.grid(True)

# (c) 爆破事件初始幅值散点 (分阶段着色)
ax_c = axes[1, 0]
colors_ph = ['#4472C4', '#ED7D31', '#70AD47']
labels_ph = ['阶段一', '阶段二', '阶段三']
event_mask = A0_all > 0
if np.any(event_mask):
    for ph in range(3):
        idx_ph = slices[ph]
        mask_ph = event_mask & (t_axis >= idx_ph.start) & (t_axis < idx_ph.stop)
        if np.any(mask_ph):
            ax_c.scatter(t_axis[mask_ph], A0_all[mask_ph], s=20,
                         c=colors_ph[ph], label=labels_ph[ph], alpha=0.7, edgecolors='none')
    ax_c.axvline(b1 - 0.5, color='gray', linestyle='--', linewidth=0.8)
    ax_c.axvline(b2 - 0.5, color='gray', linestyle='--', linewidth=0.8)
ax_c.set_xlabel('时间步 (10min/步)')
ax_c.set_ylabel('初始幅值 A0 = beta * Q / R^3')
ax_c.set_title('(c) 爆破事件初始幅值')
ax_c.legend(loc='upper left')
ax_c.grid(True)

# (d) 爆破位移占总位移比例
# 注：这里用实测位移 D_real 代替预测位移估算占比
ax_d = axes[1, 1]
ratio = np.zeros(n)
mask_nonzero = D_real > 0.1
ratio[mask_nonzero] = blast_accum[mask_nonzero] / D_real[mask_nonzero] * 100
ax_d.plot(t_axis, ratio, 'g-', linewidth=0.8, label='占比')
ax_d.axvline(b1 - 0.5, color='gray', linestyle='--', linewidth=0.8)
ax_d.axvline(b2 - 0.5, color='gray', linestyle='--', linewidth=0.8)
ax_d.set_xlabel('时间步 (10min/步)')
ax_d.set_ylabel('占比 (%)')
ax_d.set_title('(d) 爆破位移占总位移比例 (估算)')
ax_d.set_ylim(bottom=0)
ax_d.legend(loc='upper left')
ax_d.grid(True)

plt.tight_layout()
save_path = 'blast_analysis.png'
plt.savefig(save_path, dpi=200, bbox_inches='tight')
print(f'\n图像已保存: {save_path}')

# 输出统计数据到文本框
text_str = (
    f'爆破事件数: {blast_event_count}\n'
    f'最终累积位移: {total_blast_disp:.4f} mm\n'
    f'最大累积位移: {max_blast_disp:.4f} mm\n'
    f'最大等效速率: {max_blast_rate:.6f} mm/10min'
)
print(text_str)

plt.show()

print('=' * 72)
print('  爆破影响分析完成!')
print('=' * 72)
