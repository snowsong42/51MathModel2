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

# ==================== 2. 数据读取 ====================
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "../ap3.xlsx")
df_train = pd.read_excel(file_path, sheet_name="训练集")

cols = {
    'a': 'a: Rainfall (mm)',
    'b': 'b: Pore Water Pressure (kPa)',
    'c': 'c: Microseismic Event Count',
    'd': 'd: Deep Displacement (mm)',
    'e': 'e: Surface Displacement (mm)'
}

raw_data = {}
for key, col in cols.items():
    series = pd.to_numeric(df_train[col], errors='coerce')
    raw_data[key] = series

# ==================== 3. 缺失值补齐（仅用于TV去噪） ====================
filled_data = {}
for key, series in raw_data.items():
    filled = series.interpolate(method='linear', limit_direction='both')
    filled = filled.bfill().ffill().fillna(0)
    filled_data[key] = filled.values

# ==================== 4. TV去噪（用于数据清洗，但不用于异常检测） ====================
relative_lambda = {'a': 0.08, 'b': 0.06, 'c': 0.04, 'd': 0.06, 'e': 0.06}
data_std = {key: np.nanstd(raw_data[key]) for key in raw_data}
print("各变量标准差:", {k: f"{v:.3f}" for k, v in data_std.items()})

denoised_data = {}
for key, y in filled_data.items():
    lam = relative_lambda[key] * data_std[key]
    y_den = tv_denoise_admm(y, lam=lam, max_iter=200)
    if key == 'c':
        y_den = np.round(y_den).astype(int)
    denoised_data[key] = y_den

# ==================== 5. 异常检测：分策略 ====================
def detect_outliers_mad(data, k=8.0):
    """MAD异常检测"""
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    if mad == 0:
        return np.zeros(len(data), dtype=bool)
    return np.abs(data - median) > k * mad

def detect_outliers_percentile(data, p=99.0):
    """基于百分位数的异常检测——适合含大量0值的分布"""
    threshold = np.percentile(data[data > 0], p) if np.any(data > 0) else np.percentile(data, p)
    if threshold == 0:
        return np.zeros(len(data), dtype=bool)
    return data > threshold

def detect_outliers_diff(data, k=6.0):
    """差分突变检测"""
    d = np.diff(data)
    median = np.median(d)
    mad = np.median(np.abs(d - median))
    if mad == 0:
        return np.zeros(len(data), dtype=bool)
    flags = np.zeros(len(data), dtype=bool)
    flags[:-1] = np.abs(d - median) > k * mad
    flags[1:] |= np.abs(d - median) > k * mad
    return flags

outlier_flags = {}
outlier_counts = {}
total_length = None

for key in ['a','b','c','d','e']:
    y_filled = filled_data[key]

    if key == 'a':
        # 降雨量：标记非零值中 p=99（约1%的非零值=54个=0.54%）
        flags_level = detect_outliers_percentile(y_filled, p=99.0)
        flags_jump = np.zeros(len(y_filled), dtype=bool)
    elif key == 'c':
        # 微震：离散计数，MAD k=4（标记极端计数）
        flags_level = detect_outliers_mad(y_filled, k=4.0)
        flags_jump = np.zeros(len(y_filled), dtype=bool)
    elif key == 'b':
        # 孔隙水压力：MAD k=6 + 差分 k=4
        flags_level = detect_outliers_mad(y_filled, k=6.0)
        flags_jump = detect_outliers_diff(y_filled, k=4.0)
    else:
        # d/e 位移：MAD k=6 + 差分 k=4
        flags_level = detect_outliers_mad(y_filled, k=6.0)
        flags_jump = detect_outliers_diff(y_filled, k=4.0)

    flags = flags_level | flags_jump
    outlier_flags[key] = flags
    outlier_counts[key] = int(np.sum(flags))
    total_length = len(flags)

# ==================== 6. 输出表3.1 ====================
print("="*60)
print("表3.1 训练集单变量异常点检出结果")
print("-"*60)
print(f"{'数据集变量':<20} {'异常点数量':>10}  {'占比':>8}")
for key in ['a','b','c','d','e']:
    var_name = {'a':'a：降雨量','b':'b：孔隙水压力','c':'c：微震事件数','d':'d：深部位移','e':'e：表面位移'}[key]
    pct = outlier_counts[key] / total_length * 100
    print(f"{var_name:<20} {outlier_counts[key]:>10}  {pct:>7.2f}%")
total_count = sum(outlier_counts.values())
total_pct = total_count / (total_length * 5) * 100
print(f"{'总数':<20} {total_count:>10}  {total_pct:>7.2f}%")
print("="*60)

# 保存表3.1到Excel
table3_1_data = []
for key in ['a','b','c','d','e']:
    var_name = {'a':'a：降雨量','b':'b：孔隙水压力','c':'c：微震事件数','d':'d：深部位移','e':'e：表面位移'}[key]
    pct = outlier_counts[key] / total_length * 100
    table3_1_data.append({'数据集变量': var_name, '异常点数量': outlier_counts[key], '占比(%)': round(pct, 2)})
table3_1_data.append({'数据集变量': '总数', '异常点数量': total_count, '占比(%)': round(total_pct, 2)})
df_table3_1 = pd.DataFrame(table3_1_data)
table3_1_path = os.path.join(script_dir, "table3.1_outlier_counts.xlsx")
# 若文件被占用则覆盖写入
try:
    df_table3_1.to_excel(table3_1_path, index=False)
except PermissionError:
    temp_path = os.path.join(script_dir, "table3.1_outlier_counts_temp.xlsx")
    df_table3_1.to_excel(temp_path, index=False)
    print(f"原文件被占用，已保存至临时文件 {temp_path}")
print(f"表3.1已保存至 {table3_1_path}")

# ==================== 7. 找出共同异常点 (≥2个变量异常) ====================
common_outliers = defaultdict(list)
for t in range(total_length):
    abnormal_vars = [key for key in ['a','b','c','d','e'] if outlier_flags[key][t]]
    if len(abnormal_vars) >= 2:
        var_str = ''.join(sorted(abnormal_vars))
        common_outliers[t+1] = var_str

# ==================== 8. 输出表3.2 ====================
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

output_path = os.path.join(script_dir, "common_outliers_table3.2.xlsx")
df_common = pd.DataFrame(list(common_outliers.items()), columns=['Serial No.', 'Common Abnormals'])
try:
    df_common.to_excel(output_path, index=False)
except PermissionError:
    temp_path = os.path.join(script_dir, "common_outliers_table3.2_temp.xlsx")
    df_common.to_excel(temp_path, index=False)
    print(f"原文件被占用，已保存至临时文件 {temp_path}")
print(f"\n完整表3.2已保存至 {output_path}")

# ==================== 9. 绘图 ====================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

var_labels_cn = {
    'a': 'a：降雨量 (mm)',
    'b': 'b：孔隙水压力 (kPa)',
    'c': 'c：微震事件数',
    'd': 'd：深部位移 (mm)',
    'e': 'e：表面位移 (mm)'
}

fig, axes = plt.subplots(5, 1, figsize=(16, 14), sharex=True)
t = np.arange(total_length)

for idx, key in enumerate(['a','b','c','d','e']):
    ax = axes[idx]
    y = filled_data[key]
    flags = outlier_flags[key]

    ax.plot(t, y, 'gray', alpha=0.3, linewidth=0.6, label='数据')

    outlier_idx = np.where(flags)[0]
    if len(outlier_idx) > 0:
        ax.scatter(outlier_idx, y[outlier_idx],
                   color='red', s=12, alpha=0.7, label=f'异常点 ({len(outlier_idx)})', zorder=5)

    ax.set_ylabel(var_labels_cn[key], fontsize=9)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=0.2)
    ax.set_xlim(0, total_length)

axes[-1].set_xlabel('时间序号 (10分钟间隔)')
plt.suptitle('基于原始数据直接MAD的异常检测（降雨k=10, 其余k=8）', fontsize=14)
plt.tight_layout()
plot_path = os.path.join(script_dir, "outlier_detection_results.png")
plt.savefig(plot_path, dpi=300)
print(f"异常检测结果图已保存至 {plot_path}")
plt.close()
