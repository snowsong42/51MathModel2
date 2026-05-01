import pandas as pd
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, factorized
import matplotlib.pyplot as plt
import os

# ==================== 1. TV去噪函数 (ADMM) ====================
def tv_denoise_admm(y, lam=1.0, rho=1.0, max_iter=100, tol=1e-4):
    y = np.asarray(y, dtype=float)
    N = len(y)
    e = np.ones(N)
    D = sparse.diags([-e, e], [0, 1], shape=(N-1, N)).tocsc()
    DTD = D.T @ D
    I = sparse.eye(N)
    A = I + rho * DTD
    solve_A = factorized(A.tocsc())
    x = y.copy()
    z = np.zeros(N-1)
    u = np.zeros(N-1)
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

# ==================== 2. 数据读取 (从 ap3.xlsx) ====================
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "ap3.xlsx")
df_train = pd.read_excel(file_path, sheet_name="训练集")

# ap3.xlsx 中的实际列名
cols = {
    'a': 'a: Rainfall (mm)',
    'b': 'b: Pore Water Pressure (kPa)',
    'c': 'c: Microseismic Event Count',
    'd': 'd: Deep Displacement (mm)',
    'e': 'e: Surface Displacement (mm)'
}

# 提取并转为数值，空值变为NaN
raw_data = {}
for key, col in cols.items():
    series = pd.to_numeric(df_train[col], errors='coerce')
    raw_data[key] = series

# ==================== 3. 缺失值补齐 ====================
filled_data = {}
for key, series in raw_data.items():
    filled = series.interpolate(method='linear', limit_direction='both')
    filled = filled.bfill().ffill().fillna(0)
    filled_data[key] = filled.values

# ==================== 4. TV去噪 (不同变量lambda不同) ====================
lambda_dict = {'a':0.3, 'b':0.8, 'c':0.2, 'd':0.5, 'e':0.5}
denoised_data = {}
for key, y in filled_data.items():
    y_den = tv_denoise_admm(y, lam=lambda_dict[key], max_iter=200)
    if key == 'c':   # 微震事件数取整
        y_den = np.round(y_den).astype(int)
    denoised_data[key] = y_den

# ==================== 5. 保存结果 ====================
df_cleaned = df_train[['Serial No. ']].copy()
# 输出时使用简洁的列名
clean_col_names = {
    'a': 'Rainfall (mm)',
    'b': 'Pore Water Pressure (kPa)',
    'c': 'Microseismic Event Count',
    'd': 'Deep Displacement (mm)',
    'e': 'Surface Displacement (mm)'
}
for key in cols:
    df_cleaned[clean_col_names[key]] = denoised_data[key]
output_path = os.path.join(script_dir, "training_set_cleaned.xlsx")
df_cleaned.to_excel(output_path, index=False)

# ==================== 6. 输出各变量的统计摘要 ====================
print("="*60)
print("各变量原始缺失数、补齐后均值、去噪后均值")
for key in cols:
    orig_missing = raw_data[key].isna().sum()
    filled_mean = np.mean(filled_data[key])
    den_mean = np.mean(denoised_data[key])
    print(f"{key}: 原始缺失 {orig_missing:4d} 个, 补齐后均值 = {filled_mean:.4f}, 去噪后均值 = {den_mean:.4f}")

# ==================== 7. 绘制五变量去噪前后对比图 ====================
fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
for idx, key in enumerate(['a','b','c','d','e']):
    ax = axes[idx]
    t = np.arange(len(raw_data[key]))
    ax.plot(t, raw_data[key].values, 'gray', alpha=0.4, label='Raw (with gaps)')
    ax.plot(t, filled_data[key], 'b--', alpha=0.7, label='After interpolation')
    ax.plot(t, denoised_data[key], 'r-', linewidth=1.2, label='TV denoised')
    ax.set_ylabel(cols[key].split('(')[0].strip(), fontsize=10)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(alpha=0.3)
axes[-1].set_xlabel('Time point (10-min interval)')
plt.suptitle('Missing imputation and TV denoising results for five variables', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "five_variables_denoising.png"), dpi=300)
plt.show()

print("\n所有变量均已处理并保存，上图展示五个维度的缺失补齐与去噪对比。")
