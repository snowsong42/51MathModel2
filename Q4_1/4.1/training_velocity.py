"""
====================================================================
训练集数据滤波预处理 -> 四维度变量拟合 -> 残差分析
数据源: Q4_1/ap4.xlsx -> 训练集 Sheet

流程:
  1) 读取训练集 Sheet，提取各维变量
  2) 位移: 零值插值 -> 中值滤波 -> 小波去噪 -> S-G 平滑
  3) 提取四维度特征变量并预处理
  4) 多元回归/带交互项拟合 -> 对比预测 vs 实际滤波后位移
  5) 残差分析 & 可视化
====================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import medfilt, savgol_filter
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import pywt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
#  1. 读取 Excel 数据
# ============================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "../ap4.xlsx")
df = pd.read_excel(data_path, sheet_name='训练集')

# 时间列
time_raw = df.iloc[:, 0]
try:
    time_dt = pd.to_datetime(time_raw)
except Exception:
    time_dt = None

N = len(df)
serial_no = np.arange(1, N + 1, dtype=float)

# === 核心变量 ===
raw_displacement = df.iloc[:, 1].values.astype(float)
rainfall     = df.iloc[:, 2].fillna(0).values.astype(float)
pore         = df.iloc[:, 3].ffill().bfill().values.astype(float)
micro        = df.iloc[:, 4].fillna(0).values.astype(float)
blast_dist   = df.iloc[:, 5].fillna(0).values.astype(float)
blast_charge = df.iloc[:, 6].fillna(0).values.astype(float)

print("=" * 72)
print("  Training Data: Filter + 4D Variable Fitting Analysis")
print(f"  Data: ap4.xlsx -> Train Sheet")
print(f"  Length: {N} points")
print(f"  Time: {str(time_raw.iloc[0])} ~ {str(time_raw.iloc[-1])}")
print(f"  Disp: {raw_displacement.min():.4f} ~ {raw_displacement.max():.4f} mm")
print("=" * 72)


# ============================================================
#  2. 位移滤波预处理
# ============================================================
zero_mask = (raw_displacement == 0)
n_zero = int(np.sum(zero_mask))
interp_data = raw_displacement.copy()
if n_zero > 0:
    valid_idx = np.where(~zero_mask)[0]
    interp_data[zero_mask] = np.interp(
        np.where(zero_mask)[0], valid_idx, raw_displacement[valid_idx])

print(f"\n[Filter] Zero values: {n_zero}")

win_median = 9
median_filtered = medfilt(interp_data, kernel_size=win_median)
print(f"[Filter] Median k={win_median}")


def wavelet_denoise(signal, wavelet='db4', level=4):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))
    coeffs_thresh = [coeffs[0]]
    for i in range(1, len(coeffs)):
        coeffs_thresh.append(pywt.threshold(coeffs[i], threshold, mode='soft'))
    return pywt.waverec(coeffs_thresh, wavelet)[:len(signal)]


wavelet_denoised = wavelet_denoise(median_filtered)
print(f"[Filter] Wavelet db4, level=4")

order_sg, framelen_sg = 2, 21
sg_filtered = savgol_filter(wavelet_denoised, window_length=framelen_sg, polyorder=order_sg)
print(f"[Filter] S-G order={order_sg}, framelen={framelen_sg}")

y_true = sg_filtered.copy()

# ============================================================
#  计算表面位移速度 (mm/min)
# ============================================================
# 采样间隔 dt = 10 min (数据中每10分钟一个点)
dt = 10.0  # minutes
velocity = np.gradient(y_true, dt)  # mm/min
velocity_raw = np.gradient(raw_displacement, dt)  # 原始速度用于对比

print(f"\n[Velocity] Velocity range: {velocity.min():.4f} ~ {velocity.max():.4f} mm/min")


# ============================================================
#  3. 四维度特征工程
# ============================================================
rain_cumsum = np.cumsum(rainfall)
rain_decay = np.zeros(N)
decay_factor = 0.95
cum = 0.0
for i in range(N):
    cum = cum * decay_factor + rainfall[i]
    rain_decay[i] = cum

micro_roll = pd.Series(micro).rolling(window=12, min_periods=1, center=True).mean().values

X_feat = np.column_stack([
    rainfall, rain_cumsum, rain_decay,
    pore, micro, micro_roll,
    blast_dist, blast_charge,
])

feat_labels = [
    'Rainfall', 'Rainfall_CumSum', 'Rainfall_Decay',
    'PorePressure', 'Microseismic', 'Microseismic_Roll',
    'BlastDist', 'BlastCharge'
]

print(f"\n[Feature] {len(feat_labels)} features built:")
for i, name in enumerate(feat_labels):
    print(f"        {i+1}. {name}")


# ============================================================
#  4. 模型拟合: 多项式回归 (degree=2) + Ridge
# ============================================================
poly_degree = 2
model = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(degree=poly_degree, include_bias=False, interaction_only=False),
    Ridge(alpha=1.0)
)

model.fit(X_feat, y_true)
y_pred = model.predict(X_feat)

residual = y_true - y_pred

rmse = float(np.sqrt(np.mean(residual ** 2)))
mae = float(np.mean(np.abs(residual)))
r2 = r2_score(y_true, y_pred)
nrmse = rmse / (np.max(y_true) - np.min(y_true)) * 100

print(f"\n{'='*60}")
print(f"  4D Variable Fitting Evaluation")
print(f"{'='*60}")
print(f"  R^2  = {r2:.6f}")
print(f"  RMSE = {rmse:.4f} mm")
print(f"  MAE  = {mae:.4f} mm")
print(f"  NRMSE= {nrmse:.2f} %")
print(f"{'='*60}")


# ============================================================
#  5. 可视化: 2x3 subplot
# ============================================================
x_disp = time_dt if time_dt is not None else serial_no

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Training Set: 4D Variable Fitting Analysis (ap4.xlsx -> Train)',
             fontsize=16, fontweight='bold', y=0.98)

ax1, ax2 = axes[0, 0], axes[0, 1]
ax3, ax4 = axes[0, 2], axes[1, 0]
ax5, ax6 = axes[1, 1], axes[1, 2]

# --- subplot 1: filtered vs predicted ---
ax1.plot(x_disp, y_true, 'b-', lw=1.2, label='Filtered (Actual)')
ax1.plot(x_disp, y_pred, 'r-', lw=1.0, alpha=0.8, label=f'4D Predicted (R^2={r2:.4f})')
ax1.fill_between(np.arange(N), y_true, y_pred,
                 color='gray', alpha=0.15, label='Residual area')
ax1.set_ylabel('Surface Displacement (mm)', fontsize=12)
ax1.set_title('Filtered vs 4D Variable Prediction', fontsize=13)
ax1.legend(fontsize=9, loc='upper left')
ax1.grid(alpha=0.3)

# --- subplot 2: residual time series ---
ax2.plot(x_disp, residual, 'k-', lw=0.6, alpha=0.7, label='Residual')
ax2.axhline(0, color='gray', lw=0.8)
ax2.axhline(2 * np.std(residual), color='red', ls='--', lw=1, alpha=0.6,
            label=f'+/-2sigma = +/-{2*np.std(residual):.2f}mm')
ax2.axhline(-2 * np.std(residual), color='red', ls='--', lw=1, alpha=0.6)
ax2.fill_between(np.arange(N), -2*np.std(residual), 2*np.std(residual),
                 color='red', alpha=0.04)
ax2.set_ylabel('Residual (mm)', fontsize=12)
ax2.set_title(f'Residual Time Series (RMSE={rmse:.3f}mm, MAE={mae:.3f}mm)', fontsize=13)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# --- subplot 3: residual histogram ---
ax3.hist(residual, bins=80, color='steelblue', edgecolor='white', alpha=0.7, density=True)
mu_res, std_res = np.mean(residual), np.std(residual)
x_norm = np.linspace(mu_res - 4*std_res, mu_res + 4*std_res, 200)
ax3.plot(x_norm, 1/(std_res*np.sqrt(2*np.pi))*np.exp(-(x_norm-mu_res)**2/(2*std_res**2)),
         'r-', lw=2, label=f'Normal N({mu_res:.3f},{std_res:.3f})')
ax3.axvline(0, color='gray', ls='--', lw=1)
ax3.set_xlabel('Residual (mm)', fontsize=12)
ax3.set_ylabel('Density', fontsize=12)
ax3.set_title(f'Residual Distribution (skew={pd.Series(residual).skew():.2f})', fontsize=13)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)

# --- subplot 4: predicted vs actual scatter ---
ax4.scatter(y_true, y_pred, s=1, c='steelblue', alpha=0.4)
min_val = min(y_true.min(), y_pred.min())
max_val = max(y_true.max(), y_pred.max())
ax4.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1.5, label='Ideal 1:1')
ax4.set_xlabel('Actual (mm)', fontsize=12)
ax4.set_ylabel('Predicted (mm)', fontsize=12)
ax4.set_title(f'Predicted vs Actual (R^2={r2:.4f})', fontsize=13)
ax4.legend(fontsize=9)
ax4.grid(alpha=0.3)
ax4.axis('equal')

# --- subplot 5: 4D variables time series ---
ax5_twin = ax5.twinx()
ax5.plot(x_disp, rainfall, 'b-', lw=0.5, alpha=0.5, label='Rainfall')
ax5.plot(x_disp, pore, 'g-', lw=0.6, alpha=0.6, label='PorePressure')
ax5_twin.plot(x_disp, micro, 'o', markersize=0.8, color='purple', alpha=0.3, label='Microseismic')
ax5_twin.plot(x_disp, blast_dist, 's', markersize=0.8, color='orange', alpha=0.3, label='BlastDist')
lines1, labels1 = ax5.get_legend_handles_labels()
lines2, labels2 = ax5_twin.get_legend_handles_labels()
ax5.legend(lines1+lines2, labels1+labels2, fontsize=8, loc='upper left')
ax5.set_ylabel('Rainfall / PorePressure', fontsize=11)
ax5_twin.set_ylabel('Microseismic / Blast', fontsize=11)
ax5.set_title('4D Variables Time Series', fontsize=13)
ax5.grid(alpha=0.2)

# --- subplot 6: feature importance ---
poly = model.named_steps['polynomialfeatures']
ridge = model.named_steps['ridge']
coef = ridge.coef_
n_orig = len(feat_labels)
imp = np.zeros(n_orig)
for i in range(n_orig):
    for j in range(len(coef)):
        if poly.powers_[j][i] > 0:
            imp[i] += abs(coef[j])

imp_norm = imp / imp.max() if imp.max() > 0 else imp
idx_sort = np.argsort(imp_norm)
ax6.barh(range(n_orig), imp_norm[idx_sort], color='steelblue', alpha=0.7)
ax6.set_yticks(range(n_orig))
ax6.set_yticklabels([feat_labels[i] for i in idx_sort], fontsize=9)
ax6.set_xlabel('Relative Importance', fontsize=12)
ax6.set_title('Feature Importance (Poly Coef Weighted)', fontsize=13)
ax6.grid(alpha=0.3, axis='x')

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig_path = os.path.join(script_dir, "4.1_surface_displacement_analysis.png")
plt.savefig(fig_path, dpi=200, bbox_inches='tight')
print(f"\n[Chart] Saved: {fig_path}")
plt.close()


# ============================================================
#  6. 四维度对速度的影响：部分依赖图 (Partial Dependence)
# ============================================================
print(f"\n{'='*60}")
print(f"  4D Partial Dependence on Velocity分析")
print(f"{'='*60}")

# 定义四维度核心特征（原始+最相关衍生）映射到模型特征空间
# 核心4个物理维度:
#   Dim1: Rainfall -> features: rainfall(0), rain_cumsum(1), rain_decay(2)
#   Dim2: PorePressure -> feature: pore(3)
#   Dim3: Microseismic -> features: micro(4), micro_roll(5)
#   Dim4: Blasting -> features: blast_dist(6), blast_charge(7)

dim_config = [
    ("Rainfall (mm)", 0, [0, 1, 2], '#1f77b4', 'Rainfall'),
    ("Pore Pressure (kPa)", 1, [3], '#2ca02c', 'PorePressure'),
    ("Microseismic (count)", 2, [4, 5], '#9467bd', 'Microseismic'),
    ("Blasting (dist/charge)", 3, [6, 7], '#d62728', 'Blasting'),
]

# 为每个维度计算部分依赖
n_grid = 50
dim_pdp_data = []

for dim_name, dim_idx, feat_idxs, color, short_name in dim_config:
    # 取该维度的原始数据（第一个特征索引作为代表性横轴）
    if dim_idx == 0:  # Rainfall
        x_vals = rainfall.copy()
    elif dim_idx == 1:  # Pore
        x_vals = pore.copy()
    elif dim_idx == 2:  # Micro
        x_vals = micro.copy()
    elif dim_idx == 3:  # Blast
        x_vals = blast_dist.copy()  # use blast_dist as representative
    
    # 构建grid点
    x_grid = np.linspace(np.percentile(x_vals, 2), np.percentile(x_vals, 98), n_grid)
    
    # 对每个grid点：替换该维度特征，其他保持均值
    v_pred_grid = np.zeros(n_grid)
    X_base = np.tile(np.mean(X_feat, axis=0), (n_grid, 1))
    
    for g in range(n_grid):
        for fi in feat_idxs:
            X_base[g, fi] = X_base[g, fi] + (x_grid[g] - X_base[g, fi])
    
    v_pred_grid = model.predict(X_base)
    
    dim_pdp_data.append((dim_name, x_grid, v_pred_grid, color, x_vals, short_name))

# 绘制四维度对速度的部分依赖图
fig_v, axes_v = plt.subplots(2, 2, figsize=(18, 14))
fig_v.suptitle('Four-Dimension Partial Dependence on Surface Displacement Velocity',
               fontsize=16, fontweight='bold', y=0.98)

for idx, (dim_name, x_grid, v_pred_grid, color, x_vals, short_name) in enumerate(dim_pdp_data):
    ax = axes_v[idx // 2, idx % 2]
    
    # 实际速度散点（颜色映射到第三维）
    ax.scatter(x_vals, velocity, s=1, c='gray', alpha=0.2, label='Actual velocity')
    
    # 拟合曲线
    ax.plot(x_grid, v_pred_grid, '-', color=color, lw=3.0, label=f'PD fit ({short_name})')
    
    # 置信带
    v_std = np.std(velocity) / 4
    ax.fill_between(x_grid, v_pred_grid - v_std, v_pred_grid + v_std,
                    color=color, alpha=0.1)
    
    ax.set_xlabel(dim_name, fontsize=13)
    ax.set_ylabel('Velocity (mm/min)', fontsize=13)
    ax.set_title(f'Effect of {short_name} on Surface Velocity', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
vel_fig_path = os.path.join(script_dir, "4.1_velocity_partial_dependence.png")
plt.savefig(vel_fig_path, dpi=200, bbox_inches='tight')
print(f"[Chart] Velocity PD saved: {vel_fig_path}")
plt.close()

# 速度拟合评估
print(f"\n[Velocity PD] 4D partial dependence on velocity computed")
print(f"[Velocity PD] Velocity range: {velocity.min():.4f} ~ {velocity.max():.4f} mm/min")

# ============================================================
#  7. 分段残差统计
# ============================================================
print(f"\n{'='*60}")
print(f"  Residual Statistics by Displacement Level")
print(f"{'='*60}")
seg_boundaries = [0, 10, 50, 100, 200, 500, 1000]
for i in range(len(seg_boundaries)-1):
    lo, hi = seg_boundaries[i], seg_boundaries[i+1]
    mask = (y_true >= lo) & (y_true < hi)
    if mask.sum() > 0:
        seg_rmse = np.sqrt(np.mean(residual[mask]**2))
        seg_mae  = np.mean(np.abs(residual[mask]))
        seg_max  = np.max(np.abs(residual[mask]))
        print(f"  Disp [{lo:>4}, {hi:>4})mm, n={mask.sum():>4}]  "
              f"RMSE={seg_rmse:.3f}  MAE={seg_mae:.3f}  Max|Res|={seg_max:.3f}")

# ============================================================
#  7. 保存结果
# ============================================================
result_df = pd.DataFrame({
    'SerialNo': serial_no.astype(int),
    'Time': time_raw,
    'RawDisplacement': raw_displacement,
    'FilteredDisplacement': y_true,
    'PredictedDisplacement': y_pred,
    'Residual': residual,
    'Rainfall': rainfall,
    'PorePressure': pore,
    'Microseismic': micro,
    'BlastDist': blast_dist,
    'BlastCharge': blast_charge
})
csv_path = os.path.join(script_dir, "training_velocity_results.csv")
result_df.to_csv(csv_path, index=False)
print(f"\n[Save] Results saved: {csv_path}")

print(f"\n{'='*60}")
print(f"  [Done] Training data processing complete")
print(f"  Method: Polynomial Regression (deg=2, Ridge alpha=1.0)")
print(f"  Metrics: R^2={r2:.4f}, RMSE={rmse:.3f}mm, MAE={mae:.3f}mm")
print(f"{'='*60}")
