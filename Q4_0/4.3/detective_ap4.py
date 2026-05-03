"""
对 ap4_denoise.xlsx 中训练集和实验集的 a、b、c 三列进行异常检测（按 Stage 分组内部分别检测）。
输出 ap4_detected.xlsx，在 a b c 列后分别添加 a_ab b_ab c_ab 异常标记列（1=异常，0=正常）。
可视化：3 张图 stage{1/2/3}_dt.png，各含 a/b/c 3 个子图。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

script_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(script_dir, "../4.2/ap4_denoise.xlsx")
output_path = os.path.join(script_dir, "ap4_detected.xlsx")

target_cols = ['a', 'b', 'c']
var_names = {'a': '降雨量 (mm)', 'b': '孔隙水压力 (kPa)', 'c': '微震事件数'}

k_value = 3
def detect_outliers_mad(values, k=k_value):
    """Robust标准化 + MAD异常检测，返回 0/1 标记数组"""
    z = (values - np.median(values)) / max(np.median(np.abs(values - np.median(values))), 1e-10)
    mad_z = np.median(np.abs(z - np.median(z)))
    if mad_z == 0:
        return np.zeros(len(z), dtype=int)
    return (np.abs(z - np.median(z)) > k * mad_z).astype(int)


# ===== 按 Stage 分组检测 =====
results = {}
std_data = {}   # 标准化数据（按 stage 拆分）
flag_data = {}  # 异常标记（按 stage 拆分）

for sheet_name in ["训练集", "实验集"]:
    df = pd.read_excel(input_path, sheet_name=sheet_name)
    df_out = df.copy()

    flags_all = {}  # 临时存储各列的完整标记序列
    for col in target_cols:
        flags_all[col] = np.zeros(len(df), dtype=int)

    # 按 Stage 分组检测
    for stage in sorted(df['Stage'].unique()):
        mask = df['Stage'] == stage
        for col in target_cols:
            values = pd.to_numeric(df.loc[mask, col], errors='coerce').fillna(0).values
            flags_all[col][mask] = detect_outliers_mad(values)

        # 保存该 stage 的标准化数据用于绘图
        for col in target_cols:
            vals = pd.to_numeric(df.loc[mask, col], errors='coerce').fillna(0).values
            key = f"{sheet_name}_{col}_stage{stage}"
            std_data[key] = (vals - np.median(vals)) / max(np.median(np.abs(vals - np.median(vals))), 1e-10)
            flag_data[key] = flags_all[col][mask]

    # 构造输出列
    new_cols = []
    for col in df.columns:
        new_cols.append(col)
        if col in target_cols:
            df_out[f'{col}_ab'] = flags_all[col]
            new_cols.append(f'{col}_ab')

    results[sheet_name] = df_out[new_cols]

# ===== 输出到 Excel =====
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    results["训练集"].to_excel(writer, sheet_name="训练集", index=False)
    results["实验集"].to_excel(writer, sheet_name="实验集", index=False)
print(f"异常检测结果已保存至 {output_path}")
for sn in ["训练集", "实验集"]:
    print(f"{sn} 列: {results[sn].columns.tolist()}")

# ===== 按 Stage 分别绘图 =====
stages = sorted(results["训练集"]['Stage'].unique())

for stage in stages:
    fig, axes = plt.subplots(3, 1, figsize=(14, 9))
    fig.suptitle(f'Stage {stage} 异常检测（Robust标准化 + MAD, k={k_value}）', fontsize=14)

    for i, col in enumerate(target_cols):
        ax = axes[i]

        # 训练集
        z_train = std_data[f"训练集_{col}_stage{stage}"]
        flags_train = flag_data[f"训练集_{col}_stage{stage}"]
        t = np.arange(len(z_train))
        ax.plot(t, z_train, 'gray', alpha=0.25, linewidth=0.6, label='训练集 (z-score)')
        outlier_idx = np.where(flags_train == 1)[0]
        if len(outlier_idx) > 0:
            ax.scatter(outlier_idx, z_train[outlier_idx], color='red', s=10, alpha=0.6,
                       label=f'训练集异常 ({len(outlier_idx)})')

        # 实验集
        z_exp = std_data[f"实验集_{col}_stage{stage}"]
        flags_exp = flag_data[f"实验集_{col}_stage{stage}"]
        t_exp = np.arange(len(z_exp))
        ax.plot(t_exp, z_exp, 'lightblue', alpha=0.25, linewidth=0.6, label='实验集 (z-score)')
        outlier_idx = np.where(flags_exp == 1)[0]
        if len(outlier_idx) > 0:
            ax.scatter(outlier_idx, z_exp[outlier_idx], color='orange', s=10, alpha=0.6,
                       label=f'实验集异常 ({len(outlier_idx)})')

        ax.axhline(y=k_value, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
        ax.axhline(y=-k_value, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
        ax.set_ylabel(var_names[col], fontsize=11)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel('时间序号（阶段内相对时间）', fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(script_dir, f"stage{stage}_dt.png")
    plt.savefig(save_path, dpi=200)
    print(f"已保存 {save_path}")
    plt.show()
    plt.close()

print("全部完成")
