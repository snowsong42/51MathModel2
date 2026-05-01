import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt, savgol_filter
from sklearn.metrics import r2_score, mean_squared_error

# ==================== 1. 数据读取与预处理 ====================
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "Attachment 2.xlsx")
df = pd.read_excel(file_path, sheet_name=0)
d_orig = df["Surface Displacement (mm)"].values
N = len(d_orig)
dt = 10 / 60               # 采样间隔 10分钟 = 1/6 小时
time_h = np.arange(N) * dt # 时间 (小时)

# 零值视为缺失，线性插值
d = d_orig.copy()
zero_idx = np.where(d == 0)[0]
if len(zero_idx) > 0:
    x = np.arange(N)
    mask = d != 0
    d = np.interp(x, x[mask], d[mask])

# 中值滤波抑制跳变（窗口31点）
d_filt = medfilt(d, kernel_size=31)

# ==================== 2. 速度计算与平滑 ====================
v = np.diff(d_filt) / dt                # 速度 (mm/h)
# 平滑速度用于阶段识别
v_smooth = savgol_filter(v, window_length=151, polyorder=2)

# ==================== 3. 阶段转换节点识别 ====================
L = 144                     # 滑动窗口长度 (24小时)
min_gap = 500               # 两节点最小间隔点数
thresh_slow = 0.15          # 匀速阶段速度上界 (mm/h)
thresh_fast = 2.0           # 快速阶段速度下界 (mm/h)

# 计算滑动窗口速度中位数
v_med = np.full_like(v, np.nan)
for i in range(L, len(v)-L):
    v_med[i] = np.median(v[i-L+1 : i+L])

# 识别节点1 (匀速→加速)
node1 = None
for i in range(L, len(v)-L):
    if v_med[i] > thresh_slow:
        # 要求后续至少100点持续大于阈值
        if np.all(v_med[i:i+100] > thresh_slow):
            node1 = i
            break

# 识别节点2 (加速→快速)
node2 = None
if node1 is not None:
    start = node1 + min_gap
    for i in range(start, len(v)-L):
        if v_med[i] > thresh_fast:
            if np.all(v_med[i:i+50] > thresh_fast):
                node2 = i
                break

# 将速度索引转换为位移点序号（1-index）
idx1 = node1 + 1 if node1 is not None else None
idx2 = node2 + 1 if node2 is not None else None
t1 = (idx1 - 1) * dt if idx1 else None
t2 = (idx2 - 1) * dt if idx2 else None

print("="*50)
print("阶段转换节点识别结果")
print(f"节点1 (匀速→加速): 序号 {idx1}, 时间 {t1/24:.2f} 天")
print(f"节点2 (加速→快速): 序号 {idx2}, 时间 {t2/24:.2f} 天")
print("="*50)

# ==================== 4. 阶段划分与建模 ====================
# 定义各阶段的数据切片
if idx1 is not None and idx2 is not None:
    # 阶段I: 1 ~ idx1
    tI = time_h[:idx1]
    dI = d_filt[:idx1]
    # 阶段II: idx1+1 ~ idx2   (注意索引: 位移点 idx1 属于阶段I, idx1+1 开始阶段II)
    tII = time_h[idx1:idx2]
    dII = d_filt[idx1:idx2]
    # 阶段III: idx2+1 ~ end
    tIII = time_h[idx2:]
    dIII = d_filt[idx2:]
else:
    raise ValueError("未能识别出两个有效转换节点，请检查阈值或数据处理。")

# 模型拟合函数
def fit_linear(t, d):
    """线性回归: d = a*t + b"""
    A = np.vstack([t, np.ones_like(t)]).T
    a, b = np.linalg.lstsq(A, d, rcond=None)[0]
    d_pred = a*t + b
    r2 = r2_score(d, d_pred)
    rmse = np.sqrt(mean_squared_error(d, d_pred))
    return a, b, d_pred, r2, rmse

def fit_quadratic(t, d):
    """二次回归: d = a*t^2 + b*t + c"""
    A = np.vstack([t**2, t, np.ones_like(t)]).T
    a, b, c = np.linalg.lstsq(A, d, rcond=None)[0]
    d_pred = a*t**2 + b*t + c
    r2 = r2_score(d, d_pred)
    rmse = np.sqrt(mean_squared_error(d, d_pred))
    return a, b, c, d_pred, r2, rmse

# 阶段 I: 线性
a1, b1, dI_pred, r2_1, rmse_1 = fit_linear(tI, dI)

# 阶段 II: 二次
a2, b2, c2, dII_pred, r2_2, rmse_2 = fit_quadratic(tII, dII)

# 阶段 III: 线性
a3, b3, dIII_pred, r2_3, rmse_3 = fit_linear(tIII, dIII)

# 计算各阶段平均速度（位移变化量 / 持续时间）
dur_I = tI[-1] - tI[0] if len(tI) > 1 else 0
delta_d_I = dI[-1] - dI[0]
v_mean_I = delta_d_I / dur_I if dur_I > 0 else 0

dur_II = tII[-1] - tII[0]
delta_d_II = dII[-1] - dII[0]
v_mean_II = delta_d_II / dur_II

dur_III = tIII[-1] - tIII[0]
delta_d_III = dIII[-1] - dIII[0]
v_mean_III = delta_d_III / dur_III

# 打印结果
print("\n阶段 I (缓慢匀速形变)")
print(f"  模型: d = {a1:.6f} t + {b1:.4f}")
print(f"  R² = {r2_1:.4f}, RMSE = {rmse_1:.3f} mm")
print(f"  持续时间: {dur_I:.1f} h, 位移变化: {delta_d_I:.2f} mm")
print(f"  平均速度: {v_mean_I:.4f} mm/h")

print("\n阶段 II (加速形变)")
print(f"  模型: d = {a2:.6e} t² + {b2:.6f} t + {c2:.4f}")
print(f"  R² = {r2_2:.4f}, RMSE = {rmse_2:.3f} mm")
print(f"  持续时间: {dur_II:.1f} h, 位移变化: {delta_d_II:.2f} mm")
print(f"  平均速度: {v_mean_II:.4f} mm/h")

print("\n阶段 III (快速形变)")
print(f"  模型: d = {a3:.6f} t + {b3:.4f}")
print(f"  R² = {r2_3:.4f}, RMSE = {rmse_3:.3f} mm")
print(f"  持续时间: {dur_III:.1f} h, 位移变化: {delta_d_III:.2f} mm")
print(f"  平均速度: {v_mean_III:.4f} mm/h")

# ==================== 5. 绘制直观图形 ====================
# 5.1 原始位移 + 分段拟合曲线
plt.figure(figsize=(12, 6))
plt.plot(time_h, d_filt, 'k-', linewidth=0.6, alpha=0.7, label='Filtered displacement')
plt.plot(tI, dI_pred, 'b--', linewidth=2, label='Stage I (linear fit)')
plt.plot(tII, dII_pred, 'g--', linewidth=2, label='Stage II (quadratic fit)')
plt.plot(tIII, dIII_pred, 'r--', linewidth=2, label='Stage III (linear fit)')
# 标记转换节点
if idx1: plt.axvline(x=t1, color='blue', linestyle=':', linewidth=1.5, label=f'Node1 t={t1/24:.1f}d')
if idx2: plt.axvline(x=t2, color='red', linestyle=':', linewidth=1.5, label=f'Node2 t={t2/24:.1f}d')
plt.xlabel('Time (hours)')
plt.ylabel('Displacement (mm)')
plt.title('Three-stage Modeling of Surface Displacement')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'stage_fitting_curve.png'), dpi=300)
plt.show()

# 5.2 残差图
plt.figure(figsize=(12, 10))
# 残差 I
plt.subplot(3,1,1)
resI = dI - dI_pred
plt.plot(tI, resI, 'b.', markersize=2, alpha=0.5)
plt.axhline(0, color='k', linestyle='-')
plt.ylabel('Residual (mm)')
plt.title(f'Stage I Residuals (R²={r2_1:.4f}, RMSE={rmse_1:.3f}mm)')
plt.grid(True, alpha=0.3)
# 残差 II
plt.subplot(3,1,2)
resII = dII - dII_pred
plt.plot(tII, resII, 'g.', markersize=2, alpha=0.5)
plt.axhline(0, color='k', linestyle='-')
plt.ylabel('Residual (mm)')
plt.title(f'Stage II Residuals (R²={r2_2:.4f}, RMSE={rmse_2:.3f}mm)')
plt.grid(True, alpha=0.3)
# 残差 III
plt.subplot(3,1,3)
resIII = dIII - dIII_pred
plt.plot(tIII, resIII, 'r.', markersize=2, alpha=0.5)
plt.axhline(0, color='k', linestyle='-')
plt.xlabel('Time (hours)')
plt.ylabel('Residual (mm)')
plt.title(f'Stage III Residuals (R²={r2_3:.4f}, RMSE={rmse_3:.3f}mm)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'stage_residuals.png'), dpi=300)
plt.show()

# 5.3 速度曲线 + 转换节点
plt.figure(figsize=(12, 5))
plt.plot(time_h[:-1], v_smooth, 'r-', linewidth=1.2, label='Smoothed velocity')
plt.axhline(thresh_slow, color='b', linestyle='--', label=f'Slow threshold = {thresh_slow} mm/h')
plt.axhline(thresh_fast, color='g', linestyle='--', label=f'Fast threshold = {thresh_fast} mm/h')
if idx1: plt.axvline(x=t1, color='blue', linestyle=':', linewidth=1.5)
if idx2: plt.axvline(x=t2, color='red', linestyle=':', linewidth=1.5)
plt.xlabel('Time (hours)')
plt.ylabel('Velocity (mm/h)')
plt.title('Velocity Trend and Stage Transition Nodes')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'velocity_transition.png'), dpi=300)
plt.show()

print(f"\n所有图形已保存至: {script_dir}")
