import pandas as pd
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import os
# ===== matplotlib 显示设置 =====
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

script_dir = os.path.dirname(os.path.abspath(__file__))

# ==================== 1. TV去噪算法 (ADMM) ====================
def tv_denoise_admm(y, lam=1.0, rho=1.0, max_iter=100, tol=1e-4):
    y = np.asarray(y, dtype=float)
    N = len(y)
    e = np.ones(N)
    D = sparse.diags([-e, e], [0, 1], shape=(N-1, N)).tocsc()
    x = y.copy()
    z = np.zeros(N-1)
    u = np.zeros(N-1)
    DTD = D.T @ D
    I = sparse.eye(N)
    A = I + rho * DTD
    from scipy.sparse.linalg import factorized
    solve_A = factorized(A.tocsc())
    for k in range(max_iter):
        rhs = y + rho * (D.T @ (z - u))
        x_new = solve_A(rhs)
        d = D @ x_new + u
        z_new = np.maximum(0, d - lam/rho) - np.maximum(0, -d - lam/rho)
        u_new = u + d - z_new
        if np.linalg.norm(x_new - x) < tol * np.linalg.norm(x):
            break
        x, z, u = x_new, z_new, u_new
    return x

# ==================== 2. 读取数据 ====================
file_path = os.path.join(script_dir, "../ap3.xlsx")
df_train = pd.read_excel(file_path, sheet_name="训练集")

cols = {
    'a': 'a: Rainfall (mm)',
    'b': 'b: Pore Water Pressure (kPa)',
    'c': 'c: Microseismic Event Count',
    'd': 'd: Deep Displacement (mm)',
    'e': 'e: Surface Displacement (mm)'
}
data_raw = {}
for key, col in cols.items():
    series = pd.to_numeric(df_train[col].copy(), errors='coerce')
    data_raw[key] = series

# ==================== 3. 缺失值补齐 ====================
data_filled = {}
for key, series in data_raw.items():
    filled = series.interpolate(method='linear', limit_direction='both', limit_area='inside')
    filled = filled.bfill().ffill().fillna(0)
    data_filled[key] = filled.values

# ==================== 4. TV去噪 ====================
lambda_dict = {'a': 0.3, 'b': 0.8, 'c': 0.2, 'd': 0.5, 'e': 0.5}
data_denoised = {}
for key, y in data_filled.items():
    y_den = tv_denoise_admm(y, lam=lambda_dict[key], max_iter=200, tol=1e-5)
    if key == 'c':
        y_den = np.round(y_den).astype(int)
    data_denoised[key] = y_den

# ==================== 5. 可视化 ====================
key_names = {
    'a': '降雨量 (mm)', 'b': '孔隙水压力 (kPa)',
    'c': '微震事件数', 'd': '深部位移 (mm)', 'e': '表面位移 (mm)'
}

# ---- 5a. 常规版本 ----
fig, axes = plt.subplots(5, 1, figsize=(16, 14))
for i, key in enumerate(['a', 'b', 'c', 'd', 'e']):
    ax = axes[i]
    orig = data_raw[key].values
    filled = data_filled[key]
    denoised = data_denoised[key]
    t = np.arange(len(orig))
    ax.plot(t, orig, 'gray', alpha=0.3, linewidth=0.5, label='原始数据（含缺失）')
    ax.plot(t, filled, 'b--', alpha=0.6, linewidth=0.8, label='插值填充')
    ax.plot(t, denoised, 'r-', linewidth=1.5, label='TV去噪结果')
    ax.set_ylabel(key_names[key], fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.2)
axes[0].set_title('训练集各变量 TV 去噪效果（ADMM 求解）', fontsize=14)
axes[-1].set_xlabel('时间序号', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "denoising_result.png"), dpi=300)
print(f"去噪效果图 v1 已保存至 {os.path.join(script_dir, 'denoising_result.png')}")
plt.close()

# ---- 5b. 高对比度版本：淡化原始数据，突出优化后数据集 ----
fig, axes = plt.subplots(5, 1, figsize=(16, 14))
for i, key in enumerate(['a', 'b', 'c', 'd', 'e']):
    ax = axes[i]
    orig = data_raw[key].values
    filled = data_filled[key]
    denoised = data_denoised[key]
    t = np.arange(len(orig))
    # 原始数据极淡 → 仅作背景参考
    ax.plot(t, orig, 'gray', alpha=0.08, linewidth=0.4, label='原始数据（背景参考）')
    # 插值填充区域仅作位置提示
    ax.plot(t, filled, 'blue', alpha=0.15, linewidth=0.4, label='插值填充')
    # TV去噪结果：粗实线突出显示
    ax.plot(t, denoised, 'red', linewidth=2.2, alpha=0.95, label='★ TV去噪结果（优化后）')
    ax.set_ylabel(key_names[key], fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.12)
axes[0].set_title('训练集各变量 TV 去噪效果（高对比度版 — 突出优化后数据）', fontsize=14)
axes[-1].set_xlabel('时间序号', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "denoising_result_highlight.png"), dpi=300)
print(f"去噪效果图 v2（高对比度）已保存至 {os.path.join(script_dir, 'denoising_result_highlight.png')}")
plt.close()
