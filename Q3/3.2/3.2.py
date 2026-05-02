import pandas as pd
import numpy as np
from collections import defaultdict
import os
import matplotlib.pyplot as plt

# ==================== 1. 读取3.1的去噪结果 ====================
script_dir = os.path.dirname(os.path.abspath(__file__))
denoised_path = os.path.join(script_dir, "../3.1/train_denoised.xlsx")
df_denoised = pd.read_excel(denoised_path)

# 统一列名映射
col_map = {
    'a': 'a: Rainfall (mm)',
    'b': 'b: Pore Water Pressure (kPa)',
    'c': 'c: Microseismic Event Count',
    'd': 'd: Deep Displacement (mm)',
    'e': 'e: Surface Displacement (mm)'
}

df_denoised = df_denoised.rename(columns=lambda c: c.strip())

filled_data = {}
for key, col in col_map.items():
    filled_data[key] = df_denoised[col.strip()].values

print(f"已加载3.1去噪后的训练集数据，长度={len(filled_data['a'])}")

# ==================== 4. Robust标准化 (统一量纲) ====================
# 每个变量 z = (x - median) / MAD → 无量纲的"偏离倍数"
standardized = {}
medians = {}
mads = {}

for key, y in filled_data.items():
    med = np.median(y)
    mad = np.median(np.abs(y - med))
    if mad == 0:
        mad = np.std(y)  # 安全兜底
        if mad == 0:
            mad = 1.0
    medians[key] = med
    mads[key] = mad
    standardized[key] = (y - med) / mad

print("各变量中位数与MAD:", {k: f"med={medians[k]:.3f}, mad={mads[k]:.3f}" for k in standardized})

# ==================== 5. 直接在标准化空间做MAD异常检测（统一方法） ====================
# 不再依赖TV去噪残差——因为对于降雨量(大量0+少量脉冲)，TV去噪会过拟合残差
# 直接在标准化z值上做MAD检测：标记那些偏离中位数超过k*MAD的点
# 对所有5个变量完全相同的流程

def detect_outliers_zscore(z_values, k=4.0):
    """直接在标准化z值上做MAD异常检测——统一方法统一阈值"""
    median = np.median(z_values)
    mad = np.median(np.abs(z_values - median))
    if mad == 0:
        return np.zeros(len(z_values), dtype=bool)
    return np.abs(z_values - median) > k * mad

k_value = 4.0
print(f"\n使用统一阈值 k = {k_value}（直接在标准化z值上检测，所有变量相同）")

# 打印各变量标准化后的统计信息
for key in ['a','b','c','d','e']:
    z = standardized[key]
    print(f"  {key}: median(z)={np.median(z):.3f}  MAD(z)={np.median(np.abs(z-np.median(z))):.3f}  min={z.min():.1f}  max={z.max():.1f}  nonzeros={np.sum(z!=0)}/{len(z)}")

outlier_flags = {}
outlier_counts = {}
total_length = None

for key in ['a','b','c','d','e']:
    flags = detect_outliers_zscore(standardized[key], k=k_value)
    outlier_flags[key] = flags
    outlier_counts[key] = int(np.sum(flags))
    total_length = len(flags)

# ==================== 7. 输出表3.1 ====================
print("="*60)
print("表3.1 训练集单变量异常点检出结果（标准化+统一阈值k=3.0）")
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

# ==================== 8. 共同异常点 (≥2个变量异常) ====================
common_outliers = defaultdict(list)
for t in range(total_length):
    abnormal_vars = [key for key in ['a','b','c','d','e'] if outlier_flags[key][t]]
    if len(abnormal_vars) >= 2:
        var_str = ''.join(sorted(abnormal_vars))
        common_outliers[t+1] = var_str

# ==================== 9. 输出表3.2 ====================
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

# ==================== 10. 绘图 ====================
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
    # 在标准化空间绘图（便于统一标尺对比）
    y_std = standardized[key]
    flags = outlier_flags[key]

    ax.plot(t, y_std, 'gray', alpha=0.25, linewidth=0.6, label='标准化数据 (z-score)')

    outlier_idx = np.where(flags)[0]
    if len(outlier_idx) > 0:
        ax.scatter(outlier_idx, y_std[outlier_idx],
                   color='red', s=10, alpha=0.6, label=f'异常点 ({len(outlier_idx)})', zorder=5)

    # 标记 ±k 阈值线
    ax.axhline(y=k_value, color='r', linestyle='--', alpha=0.3, linewidth=0.5)
    ax.axhline(y=-k_value, color='r', linestyle='--', alpha=0.3, linewidth=0.5)

    ax.set_ylabel(var_labels_cn[key], fontsize=9)
    ax.legend(loc='upper left', fontsize=8, ncol=4)
    ax.grid(alpha=0.2)
    ax.set_xlim(0, total_length)

axes[-1].set_xlabel('时间序号 (10分钟间隔)')
plt.suptitle('公平版异常检测：Robust标准化 + 统一MAD异常检测(k=4.0)', fontsize=13)
plt.tight_layout()
plt.show()
plot_path = os.path.join(script_dir, "outlier_detection_results.png")
plt.savefig(plot_path, dpi=300)
print(f"异常检测结果图已保存至 {plot_path}")
plt.close()
