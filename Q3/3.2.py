import pandas as pd
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import factorized
from collections import defaultdict
import os
import matplotlib.pyplot as plt

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

# ==================== 5. 计算残差并检测异常 ====================
def detect_outliers(residuals, k=3.0):
    """
    基于MAD检测异常值
    residuals: 残差数组
    k: 阈值倍数，通常3.0
    返回布尔数组，True表示异常
    """
    if len(residuals) == 0:
        return np.array([], dtype=bool)
    median = np.median(residuals)
    mad = np.median(np.abs(residuals - median))
    if mad == 0:
        std = np.std(residuals)
        threshold = k * std
    else:
        threshold = k * mad
    return np.abs(residuals - median) > threshold

# 存储每个变量的异常标记
outlier_flags = {}
outlier_counts = {}
total_length = None

for key in ['a','b','c','d','e']:
    raw_filled = filled_data[key]
    clean_signal = denoised_data[key]
    residuals = raw_filled - clean_signal
    flags = detect_outliers(residuals, k=3.0)
    outlier_flags[key] = flags
    outlier_counts[key] = np.sum(flags)
    total_length = len(flags)

# ==================== 6. 输出表3.1 ====================
print("="*60)
print("表3.1 训练集单变量异常点检出结果")
print("-"*60)
print(f"{'数据集变量':<20} {'异常点数量':>10}")
for key in ['a','b','c','d','e']:
    var_name = {'a':'a：降雨量','b':'b：孔隙水压力','c':'c：微震事件数','d':'d：深部位移','e':'e：表面位移'}[key]
    print(f"{var_name:<20} {outlier_counts[key]:>10}")
print(f"{'总数':<20} {sum(outlier_counts.values()):>10}")
print("="*60)

# ==================== 7. 找出共同异常点 (≥2个变量异常) ====================
common_outliers = defaultdict(list)
for t in range(total_length):
    abnormal_vars = [key for key in ['a','b','c','d','e'] if outlier_flags[key][t]]
    if len(abnormal_vars) >= 2:
        var_str = ''.join(sorted(abnormal_vars))
        common_outliers[t+1] = var_str   # 序号从1开始

# ==================== 8. 输出表3.2 (前20行示例) ====================
print("\n表3.2 训练集多变量共同异常点变量清单")
print("-"*50)
print(f"{'时间点对应编号':<12} {'共同异常点处的异常变量':<10}")
for i, (idx, vars_) in enumerate(sorted(common_outliers.items())):
    if i >= 20:
        print(f"... 共 {len(common_outliers)} 个共同异常点，已输出前20个 ...")
        break
    print(f"{idx:<12} {vars_}")
if len(common_outliers) <= 20:
    print(f"共 {len(common_outliers)} 个共同异常点")

# 保存完整表3.2到Excel
output_path = os.path.join(script_dir, "common_outliers_table3.2.xlsx")
df_common = pd.DataFrame(list(common_outliers.items()), columns=['Serial No.', 'Common Abnormals'])
df_common.to_excel(output_path, index=False)
print(f"\n完整表3.2已保存至 {output_path}")

# ==================== 9. 绘图展示五变量残差与异常点 ====================
var_labels = {
    'a': 'Rainfall (mm)',
    'b': 'Pore Water Pressure (kPa)',
    'c': 'Microseismic Event Count',
    'd': 'Deep Displacement (mm)',
    'e': 'Surface Displacement (mm)'
}

fig, axes = plt.subplots(5, 1, figsize=(16, 14), sharex=True)
t = np.arange(total_length)

for idx, key in enumerate(['a','b','c','d','e']):
    ax = axes[idx]
    raw_filled = filled_data[key]
    clean_signal = denoised_data[key]
    residuals = raw_filled - clean_signal
    flags = outlier_flags[key]

    # 绘制原始数据、去噪信号、残差
    ax.plot(t, raw_filled, 'gray', alpha=0.3, linewidth=0.6, label='Filled raw')
    ax.plot(t, clean_signal, 'b-', linewidth=1.0, alpha=0.7, label='TV denoised')
    ax.plot(t, residuals, 'orange', linewidth=0.5, alpha=0.5, label='Residual')

    # 用红色散点标记异常点
    outlier_idx = np.where(flags)[0]
    if len(outlier_idx) > 0:
        ax.scatter(outlier_idx, raw_filled[outlier_idx],
                   color='red', s=8, alpha=0.6, label=f'Outliers ({len(outlier_idx)})', zorder=5)

    ax.set_ylabel(var_labels[key], fontsize=9)
    ax.legend(loc='upper left', fontsize=7, ncol=4)
    ax.grid(alpha=0.2)
    ax.set_xlim(0, total_length)

axes[-1].set_xlabel('Time point (10-min interval)')
plt.suptitle('Outlier detection based on TV denoising residuals (MAD, k=3.0)', fontsize=14)
plt.tight_layout()
plot_path = os.path.join(script_dir, "outlier_detection_results.png")
plt.savefig(plot_path, dpi=300)
print(f"异常检测结果图已保存至 {plot_path}")
plt.show()
