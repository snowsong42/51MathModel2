"""
Q4 速率叠加模型 — Python 仿真版 (simulate.m 的 Python 移植)
============================================================
读取 ap4.xlsx 的训练集数据，运行速率叠加模型

速率叠加公式:
  v(t) = v_basic + kr*R_eff(t) + kp*max(0,P-Pcrit) + km*M_cum(t)
  D(t) = D(0) + cumsum(v)    (积分求位移)
  爆破脉冲项 beta * (Q / R^3) 单独叠加
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

# --- 自动确定脚本所在目录，构建相对于脚本的路径 ---
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)  # 切换到脚本目录

# ============================================================
# 用户调参区
# ============================================================

# --- 文件路径 ---
filename = '../ap4.xlsx'
sheet_name = '训练集'

# --- 阶段划分参数 ---
AUTO_PHASE = True
MANUAL_BREAK1 = 200  # 手动阶段1→2的数据索引（AUTO_PHASE=False时有效）
MANUAL_BREAK2 = 800  # 手动阶段2→3的数据索引
SMOOTH_WIN = 50      # 计算速率时的滑动平均窗口（数据点数）
RATE_THRESH1 = 0.02  # 阶段一最大速率 (mm/10min)，超此值进入阶段二
RATE_THRESH2 = 0.10  # 阶段二最大速率，超此值进入阶段三
CONTINUOUS_N = 10    # 连续多少个点超过阈值才认定进入下一阶段

# --- 模型参数（分阶段）---
# 阶段一
v0_1 = 0.002
kr_1 = 0.0005
tau_r1 = 30
Lr_1 = 50
kp_1 = 0.0001
Pcrit_1 = 50
km_1 = 0
Lm_1 = 50
beta_1 = 0.1

# 阶段二
v0_2 = 0.01
kr_2 = 0.003
tau_r2 = 150
Lr_2 = 200
kp_2 = 0.002
Pcrit_2 = 40
km_2 = 0.00005
Lm_2 = 200
beta_2 = 0.5

# 阶段三
v0_3 = 0.05
kr_3 = 0.008
tau_r3 = 80
Lr_3 = 150
kp_3 = 0.01
Pcrit_3 = 35
km_3 = 0.0005
Lm_3 = 300
beta_3 = 1.0

# --- 爆破衰减系数 (1 / tau_blast) ---
lambda_1 = 1.0 / 200   # 阶段一：衰减时间常数 200 步（约 33 小时）
lambda_2 = 1.0 / 300   # 阶段二
lambda_3 = 1.0 / 500   # 阶段三
lambda_list = [lambda_1, lambda_2, lambda_3]

# 全局参数
P0 = 30  # 参考孔压 (kPa)


# ============================================================
# 数据读取
# ============================================================
print('=' * 72)
print('  Q4 速率叠加模型 — Python 仿真版')
print('=' * 72)

df = pd.read_excel(filename, sheet_name=sheet_name)

# 时间列在第1列, 数值列在第2~7列
data_mat = df.iloc[:, 1:].values  # 跳过时间列，取所有数值列
n = len(data_mat)

print(f'数据加载完成: {n} 个样本点')
print('变量: Surface Displacement, Rainfall, Pore Pressure, Microseismic, Blast Dist, Charge')

# 提取各列 (与ap4.xlsx列顺序完全对应)
D_real = data_mat[:, 0].copy()  # Surface Displacement (mm)
rain   = data_mat[:, 1].copy()  # Rainfall (mm)
pore   = data_mat[:, 2].copy()  # Pore Water Pressure (kPa)
micro  = data_mat[:, 3].copy()  # Microseismic Event Count
dist   = data_mat[:, 4].copy()  # Blasting Point Distance (m)
charge = data_mat[:, 5].copy()  # Maximum Charge per Segment (kg)

print(f'位移范围: {np.min(D_real):.3f} ~ {np.max(D_real):.3f} mm')

# 缺失值处理
rain[np.isnan(rain)] = 0
micro[np.isnan(micro)] = 0
dist[np.isnan(dist)] = 0
charge[np.isnan(charge)] = 0
# 孔压用前向填充
pore_series = pd.Series(pore)
pore = pore_series.ffill().values

t_axis = np.arange(n, dtype=float)

# ============================================================
# 自动阶段划分（基于速率阈值）
# ============================================================
if AUTO_PHASE:
    vel = np.diff(D_real, prepend=D_real[0])
    vel_smooth = pd.Series(vel).rolling(window=SMOOTH_WIN, center=True, min_periods=1).mean().values

    b1 = n
    b2 = n
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
else:
    b1 = MANUAL_BREAK1
    b2 = MANUAL_BREAK2

b1 = max(1, min(n, b1))
b2 = max(b1 + 10, min(n, b2))

# Python 0-indexed: slices for each phase
slices = [slice(0, b1), slice(b1, b2), slice(b2, n)]

print(f'\n阶段划分:')
print(f'  阶段一: 0 ~ {b1-1}')
print(f'  阶段二: {b1} ~ {b2-1}')
print(f'  阶段三: {b2} ~ {n-1}')

# ============================================================
# 分阶段速率叠加计算
# ============================================================
v0_list   = [v0_1, v0_2, v0_3]
kr_list   = [kr_1, kr_2, kr_3]
tau_r_list = [tau_r1, tau_r2, tau_r3]
Lr_list   = [Lr_1, Lr_2, Lr_3]
kp_list   = [kp_1, kp_2, kp_3]
Pcrit_list = [Pcrit_1, Pcrit_2, Pcrit_3]
km_list   = [km_1, km_2, km_3]
Lm_list   = [Lm_1, Lm_2, Lm_3]
beta_list = [beta_1, beta_2, beta_3]

D_pred = np.zeros(n)
current_start_disp = D_real[0]
v_components = np.zeros((n, 5))  # [v_base, v_rain, v_pore, v_micro, v_blast]
blast_accum = np.zeros(n)        # 全时段爆破累积位移
v_blast_all = np.zeros(n)        # 全时段爆破等效速率

for ph in range(3):
    idx = slices[ph]
    Nph = idx.stop - idx.start

    rain_ph   = rain[idx]
    pore_ph   = pore[idx]
    micro_ph  = micro[idx]
    dist_ph   = dist[idx]
    charge_ph = charge[idx]
    lam       = lambda_list[ph]

    # 1. 基础蠕变
    v_base = v0_list[ph] * np.ones(Nph)

    # ---------- 降雨有效入渗 (滑动窗口指数加权) ----------
    Lr  = Lr_list[ph]
    tau = tau_r_list[ph]

    # 预计算衰减权重 (长度 Lr+1, 对应 d=0 ... Lr)
    w = np.exp(-np.arange(Lr + 1) / tau)
    w /= w.sum()          # 归一化，使得窗口内权重和为1

    Reff = np.zeros(Nph)
    for i in range(Nph):
        # 当前时刻 i 能看到的过去 Lr 个历史降雨 (包括当前)
        start = max(0, i - Lr)
        local_rain = rain_ph[start:i+1]         # 长度可能 < Lr+1
        # 取权重序列的后 local_rain 个元素 (对应最近的历史)
        w_local = w[-len(local_rain):]          # 从后截取，保证最新降雨权重最高
        w_local /= w_local.sum()               # 重新归一化
        Reff[i] = np.dot(local_rain, w_local)

    v_rain = kr_list[ph] * Reff

    # 3. 孔压项
    exceed = np.maximum(0, pore_ph - Pcrit_list[ph])
    v_pore = kp_list[ph] * exceed * (pore_ph / P0)

    # 4. 微震累积 (滑动窗口求和)
    Lm = Lm_list[ph]
    Mcum = pd.Series(micro_ph).rolling(window=Lm + 1, min_periods=1).sum().values
    v_micro = km_list[ph] * Mcum

    # ---------- 5. 爆破脉冲 (指数衰减累积) ----------
    # 每次爆破产生初始位移 A0 = beta * Q / R^3
    # 之后以速率 lambda 指数衰减: 对时刻 t, 爆破位移 = A0 * exp(-lambda * (t - t0))
    # 总爆破位移 = 所有历史爆破的残余叠加
    blast_state = 0.0  # 累积的爆破位移状态（指数衰减后）
    v_blast_ph = np.zeros(Nph)
    blast_accum_ph = np.zeros(Nph)  # 爆破累计位移分量

    for i in range(Nph):
        # 先衰减: 每步乘衰减因子 exp(-lambda)
        blast_state *= np.exp(-lam)

        # 再叠加新的爆破增量
        if dist_ph[i] > 0 and charge_ph[i] > 0:
            A0 = beta_list[ph] * charge_ph[i] / (max(dist_ph[i], 0.1) ** 3)
            blast_state += A0

        blast_accum_ph[i] = blast_state
        v_blast_ph[i] = blast_state * lam  # 爆破等效速率: v = lambda * D_blast

    # 总速率 (五分量叠加)
    v_total = v_base + v_rain + v_pore + v_micro + v_blast_ph
    v_total = np.maximum(0, v_total)

    # 积分求位移: 总位移 = 四分量位移 + 爆破累积位移
    D_trend = np.zeros(Nph)
    D_trend[0] = current_start_disp
    for i in range(1, Nph):
        D_trend[i] = D_trend[i - 1] + v_total[i - 1]

    D_pred[idx] = D_trend + blast_accum_ph  # 将爆破位移作为独立附加项

    v_components[idx, :] = np.column_stack([v_base, v_rain, v_pore, v_micro, v_blast_ph])
    blast_accum[idx] = blast_accum_ph
    v_blast_all[idx] = v_blast_ph
    current_start_disp = D_trend[-1]  # 四分量部分末尾的位移（爆破单独加了）

# ============================================================
# 评估指标
# ============================================================
res = D_real - D_pred
rmse = np.sqrt(np.mean(res ** 2))
mae = np.mean(np.abs(res))
r2 = 1 - np.sum(res ** 2) / np.sum((D_real - np.mean(D_real)) ** 2)
nrmse = rmse / (np.max(D_real) - np.min(D_real)) * 100

print(f'\n{"=" * 30} 评估指标 {"=" * 30}')
print(f'RMSE  = {rmse:.4f} mm')
print(f'MAE   = {mae:.4f} mm')
print(f'R^2   = {r2:.4f}')
print(f'NRMSE = {nrmse:.2f} %')
print('=' * 72)

# ============================================================
# 绘图
# ============================================================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(16, 9))
fig.suptitle('Q4 速率叠加模型拟合效果 (Python 仿真)', fontsize=14, fontweight='bold')

# (1) 位移拟合
ax1 = plt.subplot(2, 3, 1)
ax1.plot(t_axis, D_real, 'b-', linewidth=1, label='实测')
ax1.plot(t_axis, D_pred, 'r-', linewidth=2, label='模型预测')
ax1.axvline(b1 - 0.5, color='g', linestyle='--', linewidth=1.2, label='阶段边界1')
ax1.axvline(b2 - 0.5, color='m', linestyle='--', linewidth=1.2, label='阶段边界2')
ax1.set_xlabel('时间步 (10min/步)')
ax1.set_ylabel('表面位移 (mm)')
ax1.set_title('速率叠加模型拟合效果')
ax1.legend(loc='upper left')
ax1.grid(True)

# (2) 残差时序
ax2 = plt.subplot(2, 3, 2)
ax2.plot(t_axis, res, 'k-', linewidth=0.8, label='残差')
ax2.axhline(0, color='r', linestyle='--', linewidth=1)
ax2.axhline(2 * np.std(res), color='gray', linestyle='--', linewidth=0.8)
ax2.axhline(-2 * np.std(res), color='gray', linestyle='--', linewidth=0.8)
ax2.set_xlabel('时间步')
ax2.set_ylabel('残差 (mm)')
ax2.set_title(f'残差时序 (RMSE={rmse:.3f}mm)')
ax2.grid(True)

# (3) 残差分布
ax3 = plt.subplot(2, 3, 3)
ax3.hist(res, bins=50, density=True, facecolor=[0.3, 0.6, 0.9], edgecolor='white', alpha=0.7)
x_vals = np.linspace(np.min(res), np.max(res), 200)
y_fit = norm.pdf(x_vals, np.mean(res), np.std(res))
ax3.plot(x_vals, y_fit, 'r-', linewidth=2)
ax3.set_xlabel('残差 (mm)')
ax3.set_ylabel('概率密度')
ax3.set_title(f'残差分布 (MAE={mae:.3f}mm)')
ax3.grid(True)

# (4) 速率分量堆叠 (5分量)
ax4 = plt.subplot(2, 3, 4)
v_comp_nonneg = np.maximum(0, v_components)
ax4.stackplot(t_axis,
              v_comp_nonneg[:, 0], v_comp_nonneg[:, 1],
              v_comp_nonneg[:, 2], v_comp_nonneg[:, 3],
              v_comp_nonneg[:, 4],
              labels=['基础蠕变', '降雨', '孔压', '微震', '爆破'],
              alpha=0.8)
ax4.set_xlabel('时间步')
ax4.set_ylabel('速率 (mm/10min)')
ax4.set_title('速率分量堆叠 (5分量)')
ax4.legend(loc='upper left')
ax4.grid(True)

# (5) 各阶段平均贡献 (5分量)
ax5 = plt.subplot(2, 3, 5)
mean_comp = np.zeros((3, 5))
for ph in range(3):
    idx = slices[ph]
    mean_comp[ph, :] = np.mean(v_components[idx, :], axis=0)

x_pos = np.arange(3)
width = 0.15
colors = ['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000', '#70AD47']
labels = ['基础', '降雨', '孔压', '微震', '爆破']
for i in range(5):
    ax5.bar(x_pos + i * width - 2 * width, mean_comp[:, i], width,
            color=colors[i], label=labels[i], alpha=0.85)
ax5.set_xticks(x_pos)
ax5.set_xticklabels(['阶段一', '阶段二', '阶段三'])
ax5.set_ylabel('平均速率 (mm/10min)')
ax5.set_title('各阶段平均速率贡献 (5分量)')
ax5.legend(loc='upper right')
ax5.grid(True, axis='y')

# (6) 实测vs预测散点图
ax6 = plt.subplot(2, 3, 6)
ax6.scatter(D_real, D_pred, s=5, c='blue', alpha=0.5, edgecolors='none')
m_val = min(np.min(D_real), np.min(D_pred))
M_val = max(np.max(D_real), np.max(D_pred))
ax6.plot([m_val, M_val], [m_val, M_val], 'r--', linewidth=1.5)
ax6.set_xlabel('实测位移 (mm)')
ax6.set_ylabel('预测位移 (mm)')
ax6.set_title(f'实测 vs 预测 (R²={r2:.4f})')
ax6.set_aspect('equal')
ax6.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.95])

# 保存图像
save_path = 'simulate_results.png'
plt.savefig(save_path, dpi=200, bbox_inches='tight')
print(f'图像已保存: {save_path}')

plt.show()

# ============================================================
# 爆破分量单独图像
# ============================================================
fig2, axes = plt.subplots(2, 2, figsize=(14, 8))
fig2.suptitle('爆破效应分析 (指数衰减累积模型)', fontsize=14, fontweight='bold')

# --- (a) 爆破累积位移 ---
ax_a = axes[0, 0]
ax_a.plot(t_axis, blast_accum, 'g-', linewidth=1.2, label='爆破累积位移')
ax_a.fill_between(t_axis, 0, blast_accum, color='green', alpha=0.15)
ax_a.axvline(b1 - 0.5, color='gray', linestyle='--', linewidth=0.8, label='阶段边界')
ax_a.axvline(b2 - 0.5, color='gray', linestyle='--', linewidth=0.8)
ax_a.set_xlabel('时间步 (10min/步)')
ax_a.set_ylabel('累积位移 (mm)')
ax_a.set_title('(a) 爆破累积位移')
ax_a.legend()
ax_a.grid(True)

# --- (b) 爆破等效速率 ---
ax_b = axes[0, 1]
ax_b.plot(t_axis, v_blast_all, 'g-', linewidth=1.2, label='爆破等效速率')
ax_b.axvline(b1 - 0.5, color='gray', linestyle='--', linewidth=0.8)
ax_b.axvline(b2 - 0.5, color='gray', linestyle='--', linewidth=0.8)
ax_b.set_xlabel('时间步 (10min/步)')
ax_b.set_ylabel('速率 (mm/10min)')
ax_b.set_title('(b) 爆破等效速率 v = λ·D_blast')
ax_b.legend()
ax_b.grid(True)

# --- (c) 爆破事件散点 (Q/R³ 幅值) ---
ax_c = axes[1, 0]
blast_events = (dist > 0) & (charge > 0)
if np.any(blast_events):
    A0_all = np.zeros(n)
    A0_all[blast_events] = beta_list[0] * charge[blast_events] / (np.maximum(dist[blast_events], 0.1) ** 3)
    # 分阶段用不同颜色
    for ph in range(3):
        idx_ph = slices[ph]
        mask = blast_events & np.array([i in range(idx_ph.start, idx_ph.stop) for i in range(n)])
        if np.any(mask):
            ax_c.scatter(t_axis[mask], A0_all[mask], s=15,
                        c=['#4472C4', '#ED7D31', '#70AD47'][ph],
                        label=f'阶段{["一","二","三"][ph]}', alpha=0.7)
    ax_c.axvline(b1 - 0.5, color='gray', linestyle='--', linewidth=0.8)
    ax_c.axvline(b2 - 0.5, color='gray', linestyle='--', linewidth=0.8)
ax_c.set_xlabel('时间步 (10min/步)')
ax_c.set_ylabel('初始幅值 A₀ = β·Q/R³')
ax_c.set_title('(c) 爆破事件初始幅值')
ax_c.legend()
ax_c.grid(True)

# --- (d) 爆破位移占总位移比例 ---
ax_d = axes[1, 1]
non_zero_disp = D_pred > 0.1
ratio = np.zeros(n)
ratio[non_zero_disp] = blast_accum[non_zero_disp] / D_pred[non_zero_disp] * 100
ax_d.plot(t_axis, ratio, 'g-', linewidth=0.8, label='占比')
ax_d.axvline(b1 - 0.5, color='gray', linestyle='--', linewidth=0.8)
ax_d.axvline(b2 - 0.5, color='gray', linestyle='--', linewidth=0.8)
ax_d.set_xlabel('时间步 (10min/步)')
ax_d.set_ylabel('占比 (%)')
ax_d.set_title('(d) 爆破位移占总位移比例')
ax_d.set_ylim(bottom=0)
ax_d.legend()
ax_d.grid(True)

plt.tight_layout()
save_path2 = 'blast_analysis.png'
plt.savefig(save_path2, dpi=200, bbox_inches='tight')
print(f'爆破分析图像已保存: {save_path2}')

plt.show()

# ============================================================
# 保存结果
# ============================================================
T_out = pd.DataFrame({
    'Step': t_axis.astype(int),
    'Actual_Displacement': D_real,
    'Predicted_Displacement': D_pred,
    'Residual': res
})
T_out.to_csv('simulate_output.csv', index=False)
print(f'结果已保存: simulate_output.csv')
print('=' * 72)
print('  Q4 速率叠加模型 (Python 仿真) 完成!')
print('=' * 72)
