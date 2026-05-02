"""
============================================================
四段匀加速运动假设检验脚本（含逐段残差分析）
核心：分段线性拟合检测速度的线性趋势突变点
对每段独立线性拟合 → 速度分段直线 → 积分得位移 → 与实测位移对比
新增：每段速度拟合残差统计、每段位移残差统计
============================================================
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 找线性趋势突变点（findchangepts线性版替代） ====================
def find_linear_changepts(y, max_cp=3, min_dist=50):
    """
    寻找 y 中线性趋势的突变点（等价于 MATLAB findchangepts with 'linear' statistic）
    使用二分分割 + 线性拟合残差平方和最小化
    """
    N = len(y)
    x = np.arange(N)

    def seg_cost(i, j):
        """计算区间 [i, j] 的线性拟合代价（残差平方和）"""
        if j - i < 2:
            return 0.0
        seg_x = x[i:j+1]
        seg_y = y[i:j+1]
        A = np.vstack([seg_x, np.ones_like(seg_x)]).T
        coeff, resid, _, _ = np.linalg.lstsq(A, seg_y, rcond=None)
        if resid.size == 0:
            return 0.0
        return resid[0]

    def total_cost(cp_list):
        """给定突变点列表，计算总代价"""
        points = [0] + sorted(cp_list) + [N-1]
        cost = 0.0
        for i in range(len(points)-1):
            cost += seg_cost(points[i], points[i+1])
        return cost

    # 使用贪心法逐个添加突变点
    cp_idx = []
    candidates = np.arange(min_dist, N - min_dist)

    for _ in range(max_cp):
        best_cp = None
        best_cost = float('inf')
        for cp in candidates:
            if cp in cp_idx:
                continue
            test_cps = cp_idx + [cp]
            c = total_cost(test_cps)
            if c < best_cost:
                best_cost = c
                best_cp = cp
        if best_cp is not None:
            cp_idx.append(int(best_cp))

    return np.sort(cp_idx)


# ==================== 0. 读取数据 ====================
script_dir = os.path.dirname(os.path.abspath(__file__))
# 数据文件在 Q2_py/Filtered 2.xlsx 中
q2_py_dir = os.path.join(script_dir, "../../Q2_py")
data_path = os.path.join(q2_py_dir, "Filtered 2.xlsx")

df = pd.read_excel(data_path, sheet_name=0)
t_idx = df["Serial No."].values              # 序号
x_filt = df["Surface Displacement (mm)"].values   # 滤波后位移 (mm)
v = df["Smoothed Velocity (mm/h)"].values         # 平滑速度 (mm/h)

N = len(x_filt)
dt = 10 / 60                                     # 采样间隔 (h) — 10分钟
t_h = (t_idx - 1) * dt                           # 时间轴 (h)

print(f"数据长度: {N}, 时间跨度: {t_h[-1]:.1f} h")

# ==================== 1. 检测速度的线性趋势突变点 ====================
max_cp = 3                      # 四段需要 3 个拐点
min_dist = round(N * 0.005)     # 最小间隔，防止过于密集

# 处理 NaN 值
v_clean = np.nan_to_num(v, nan=np.nanmedian(v[~np.isnan(v)]))
cp_idx = find_linear_changepts(v_clean, max_cp=max_cp, min_dist=min_dist)

cp_time = t_h[cp_idx]            # 拐点对应的时间 (h)

print(f"\n检测到的拐点序号: {cp_idx}")
print(f"对应时间(小时): {np.round(cp_time, 2)}")

# ==================== 2. 分段线性拟合速度（四段独立直线） ====================
# 定义分段区间：[1, cp1], [cp1+1, cp2], [cp2+1, cp3], [cp3+1, N]
segments = []
prev = 0
for cp in cp_idx:
    segments.append([prev, cp])
    prev = cp + 1
segments.append([prev, N-1])

# 确保边界
segments[0][0] = 0
segments[-1][1] = N-1

nSeg = len(segments)        # 应为 4

print(f"\n======== 四段匀加速（速度线性）拟合结果与残差分析 ========")
v_fit = np.zeros(N)         # 拟合速度
coeffs = np.zeros((nSeg, 2))  # 每段的 [截距 b, 斜率 k] (v = b + k*t)
v_residual = np.zeros(N)    # 速度拟合残差

for i in range(nSeg):
    idx_start, idx_end = segments[i][0], segments[i][1]
    if idx_start >= idx_end:
        continue
    seg_t = t_h[idx_start:idx_end+1]
    seg_v = v_clean[idx_start:idx_end+1]

    # 线性拟合 polyfit(seg_t, seg_v, 1)
    p = np.polyfit(seg_t, seg_v, 1)        # p[0] = 斜率（加速度），p[1] = 截距
    seg_v_fit = np.polyval(p, seg_t)
    v_fit[idx_start:idx_end+1] = seg_v_fit
    v_residual[idx_start:idx_end+1] = seg_v - seg_v_fit
    coeffs[i, :] = [p[1], p[0]]            # b, k

    # 本段速度残差统计
    seg_res = seg_v - seg_v_fit
    rmse_v = np.sqrt(np.mean(seg_res**2))
    mae_v = np.mean(np.abs(seg_res))
    max_res_v = np.max(np.abs(seg_res))

    print(f"\n阶段 {i+1}（t = {seg_t[0]:.1f} ~ {seg_t[-1]:.1f} h，样本点 {idx_start+1} ~ {idx_end+1}）:")
    print(f"  速度公式: v(t) = {p[1]:.6f} + {p[0]:.6f} * t  (加速度 = {p[0]:.6f} mm/h²)")
    print(f"  速度残差分析: RMSE = {rmse_v:.6f} mm/h, MAE = {mae_v:.6f} mm/h, 最大绝对残差 = {max_res_v:.6f} mm/h")

# ==================== 3. 由拟合速度积分得到拟合位移 ====================
x_model = np.zeros(N)
x_model[0] = x_filt[0]      # 初始位移与滤波数据对齐
for i in range(1, N):
    # 梯形积分
    x_model[i] = x_model[i-1] + 0.5 * (v_fit[i-1] + v_fit[i]) * dt

# 位移残差
residual_disp = x_filt - x_model

# ==================== 4. 绘图对比 ====================
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# 子图1：速度
ax1.plot(t_h, v_clean, 'k.', markersize=2, label='平滑速度')
ax1.plot(t_h, v_fit, 'r-', linewidth=1.5, label='拟合速度 (分段直线)')
for ct in cp_time:
    ax1.axvline(ct, color='gray', linestyle='--', linewidth=1.0)
ax1.set_ylabel("速度 (mm/h)")
ax1.set_title("速度：平滑测量值 vs 四段线性拟合（匀加速）")
ax1.legend(loc='best')
ax1.grid(alpha=0.3)

# 子图2：位移
ax2.plot(t_h, x_filt, 'b-', linewidth=1.2, label='滤波后位移')
ax2.plot(t_h, x_model, 'r--', linewidth=1.5, label='拟合位移 (四段匀加速)')
for ct in cp_time:
    ax2.axvline(ct, color='gray', linestyle='--', linewidth=1.0)
ax2.set_ylabel("位移 (mm)")
ax2.set_title("位移：滤波测量值 vs 由分段匀加速度积分得到的位移")
ax2.legend(loc='best')
ax2.grid(alpha=0.3)

# 子图3：位移残差
ax3.plot(t_h, residual_disp, 'k-', linewidth=0.8)
ax3.axhline(0, color='gray', linestyle='-', linewidth=0.5)
ax3.set_xlabel("时间 t (h)")
ax3.set_ylabel("位移残差 (mm)")
ax3.set_title("位移拟合残差（实测 - 模型）")
ax3.grid(alpha=0.3)

plt.suptitle("四段匀加速运动模型验证")
plt.tight_layout()
fig_path = os.path.join(script_dir, "stage_fitting_results.png")
plt.savefig(fig_path, dpi=200)
print(f"\n模型验证图已保存至 {fig_path}")
plt.close()

# ==================== 5. 逐段位移残差分析 ====================
print(f"\n======== 各阶段位移拟合残差分析 ========")
for i in range(nSeg):
    idx_start, idx_end = segments[i][0], segments[i][1]
    if idx_start >= idx_end:
        continue
    seg_res_disp = residual_disp[idx_start:idx_end+1]
    rmse_d = np.sqrt(np.mean(seg_res_disp**2))
    mae_d = np.mean(np.abs(seg_res_disp))
    max_d = np.max(np.abs(seg_res_disp))

    print(f"阶段 {i+1}（t = {t_h[idx_start]:.1f} ~ {t_h[idx_end]:.1f} h）: "
          f"位移 RMSE = {rmse_d:.6f} mm, MAE = {mae_d:.6f} mm, 最大绝对残差 = {max_d:.6f} mm")

# ==================== 6. 整体拟合优度评价 ====================
SS_res = np.sum((x_filt - x_model)**2)
SS_tot = np.sum((x_filt - np.mean(x_filt))**2)
R_sq = 1 - SS_res / SS_tot
RMSE = np.sqrt(np.mean((x_filt - x_model)**2))
MAE = np.mean(np.abs(x_filt - x_model))

print(f"\n======== 整体位移拟合评价 ========")
print(f"决定系数 R²  : {R_sq:.6f}")
print(f"均方根误差 RMSE : {RMSE:.4f} mm")
print(f"平均绝对误差 MAE : {MAE:.4f} mm")
print(f"最大绝对残差     : {np.max(np.abs(x_filt - x_model)):.4f} mm")

# 定性结论
if R_sq > 0.999 and np.max(np.abs(x_filt - x_model)) < 0.5:
    print(f"\n>>> 评估：四段匀加速模型对位移的拟合精度极高，")
    print(f"    该系统可近似为分阶段匀加速运动。")
else:
    print(f"\n>>> 评估：位移残差明显，四段匀加速模型不能完美描述实际运动，")
    print(f"    系统可能存在加速度连续变化的过程（非恒定加速度）。")

# ==================== 7. 前三段整体位移拟合评价 ====================
first3_mask = np.zeros(N, dtype=bool)
for i in range(min(3, nSeg)):
    idx_start, idx_end = segments[i][0], segments[i][1]
    if idx_start <= idx_end:
        first3_mask[idx_start:idx_end+1] = True

residual_first3 = residual_disp[first3_mask]
x_filt_first3 = x_filt[first3_mask]

SS_res3 = np.sum(residual_first3**2)
SS_tot3 = np.sum((x_filt_first3 - np.mean(x_filt_first3))**2)
R_sq3 = 1 - SS_res3 / SS_tot3
RMSE3 = np.sqrt(np.mean(residual_first3**2))
MAE3 = np.mean(np.abs(residual_first3))

print(f"\n======== 前三段整体位移拟合评价 ========")
print(f"决定系数 R²  : {R_sq3:.6f}")
print(f"均方根误差 RMSE : {RMSE3:.4f} mm")
print(f"平均绝对误差 MAE : {MAE3:.4f} mm")
print(f"最大绝对残差     : {np.max(np.abs(residual_first3)):.4f} mm")
