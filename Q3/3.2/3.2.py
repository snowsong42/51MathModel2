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
file_path = os.path.join(script_dir, "../ap3.xlsx")
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

# ==================== 4. TV去噪 ====================
# 【改进1】lambda 按数据标准差缩放，保证平滑力度在所有变量上一致
# 相对强度 relative_strength: a(降雨量) 脉冲强→大些, c(微震) 计数低→小, 其余适中
relative_lambda = {'a': 0.08, 'b': 0.06, 'c': 0.04, 'd': 0.06, 'e': 0.06}
data_std = {key: np.nanstd(raw_data[key]) for key in raw_data}
print("各变量标准差:", {k: f"{v:.3f}" for k, v in data_std.items()})

denoised_data = {}
for key, y in filled_data.items():
    # 实际lambda = 相对强度 × 数据标准差
    lam = relative_lambda[key] * data_std[key]
    y_den = tv_denoise_admm(y, lam=lam, max_iter=200)
    if key == 'c':
        y_den = np.round(y_den).astype(int)
    denoised_data[key] = y_den

# ==================== 改进的异常检测（统一用残差MAD） ====================
def detect_outliers_by_residual(y_filled, y_clean, k=4.0):
    """基于残差的MAD异常检测，适用于所有变量"""
    resid = y_filled - y_clean
    median = np.median(resid)
    mad = np.median(np.abs(resid - median))
    if mad == 0:
        # 极端情况：残差全为0，没有异常
        return np.zeros(len(resid), dtype=bool)
    threshold = k * mad
    return np.abs(resid - median) > threshold

# 存储每个变量的异常标记
outlier_flags = {}
outlier_counts = {}
total_length = None

for key in ['a','b','c','d','e']:
    y_filled = filled_data[key]
    y_clean = denoised_data[key]
    # 对微震事件数使用稍大的阈值，避免正常波动被标异常
    k_value = 5.0 if key == 'c' else 4.0
    flags = detect_outliers_by_residual(y_filled, y_clean, k=k_value)
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
df_table3_1.to_excel(table3_1_path, index=False)
print(f"表3.1已保存至 {table3_1_path}")

# ==================== 7. 找出共同异常点 (≥2个变量异常) ====================
common_outliers = defaultdict(list)
for t in range(total_length):
    abnormal_vars = [key for key in ['a','b','c','d','e'] if outlier_flags[key][t]]
    if len(abnormal_vars) >= 2:
        var_str = ''.join(sorted(abnormal_vars))
        common_outliers[t+1] = var_str

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
    raw_filled = filled_data[key]
    clean_signal = denoised_data[key]
    residuals = raw_filled - clean_signal
    flags = outlier_flags[key]

    # 绘制原始数据、去噪信号、残差
    ax.plot(t, raw_filled, 'gray', alpha=0.25, linewidth=0.6, label='插值后原始数据')
    ax.plot(t, clean_signal, 'b-', linewidth=1.2, alpha=0.8, label='TV去噪信号')
    ax.plot(t, residuals, 'orange', linewidth=0.5, alpha=0.4, label='残差')

    # 用红色散点标记异常点
    outlier_idx = np.where(flags)[0]
    if len(outlier_idx) > 0:
        ax.scatter(outlier_idx, raw_filled[outlier_idx],
                   color='red', s=6, alpha=0.6, label=f'异常点 ({len(outlier_idx)})', zorder=5)

    ax.set_ylabel(var_labels_cn[key], fontsize=9)
    ax.legend(loc='upper left', fontsize=8, ncol=4)
    ax.grid(alpha=0.2)
    ax.set_xlim(0, total_length)

axes[-1].set_xlabel('时间序号 (10分钟间隔)')
plt.suptitle('基于TV去噪残差的异常点检测（归一化MAD, k=3.0）', fontsize=14)
plt.tight_layout()
plot_path = os.path.join(script_dir, "outlier_detection_results.png")
plt.savefig(plot_path, dpi=300)
print(f"异常检测结果图已保存至 {plot_path}")
plt.close()
