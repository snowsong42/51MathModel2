"""
全局差分 + MAD 异常检测（不再分阶段，差分已消除趋势）
对 ap4_denoise.xlsx 中 a、b、c 三列做异常检测，输出 ap4_detected.xlsx + 异常检测结果图
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
k_value = 6


# (classify_trend 不再使用，保留以备参考)


def detect_outliers_mad(values, k=k_value, use_diff=True, zero_inflated=False):
    """（可选一阶差分）+ Robust标准化 → MAD异常检测，返回 0/1 标记数组
    
    Parameters
    ----------
    zero_inflated : bool
        零膨胀模式，针对降雨这类大量零值的数据。
        只在非零值上估计尺度，使用单侧上阈值，避免零值拉低基线导致误报。
    """
    if use_diff:
        signal = np.insert(np.diff(values), 0, 0)
    else:
        signal = values

    # ---- 零膨胀模式：仅在非零值上估计尺度，单侧阈值 ----
    if zero_inflated:
        if use_diff:
            # 差分后的"非零" = 跳跃发生（降雨开始/结束）
            nonzero = signal[signal != 0]
        else:
            # 原始值的"非零" = 有雨日
            nonzero = values[values > 0]
        if len(nonzero) < 10:
            # 有雨日太少，用高分位数兜底
            thr = np.percentile(signal, 99.5)
            return (signal > thr).astype(int)
        med_pos = np.median(nonzero)
        mad_pos = np.median(np.abs(nonzero - med_pos))
        if mad_pos == 0:
            return np.zeros(len(signal), dtype=int)
        # 单侧上阈值：只关心异常的大值
        threshold = med_pos + k * mad_pos
        return (signal > threshold).astype(int)

    # ---- 标准 MAD 模式（双侧） ----
    if np.all(signal == 0):
        return np.zeros(len(signal), dtype=int)

    med = np.median(signal)
    mad = np.median(np.abs(signal - med))
    if mad == 0:
        return np.zeros(len(signal), dtype=int)

    z = (signal - med) / mad
    med_z = np.median(z)
    mad_z = np.median(np.abs(z - med_z))
    if mad_z == 0:
        return np.zeros(len(z), dtype=int)

    return (np.abs(z - med_z) > k * mad_z).astype(int)


def robust_normalize(vals):
    """Robust 标准化"""
    med = np.median(vals)
    mad = np.median(np.abs(vals - med))
    return (vals - med) / max(mad, 1e-10)


# ===== 固定差分策略 =====
# 根据物理意义硬编码：
#   降雨量 (a)：零膨胀脉冲事件，不差分 use_diff=False
#   孔隙水压力 (b) + 微震事件数 (c)：有趋势/累积成分，差分 use_diff=True
col_use_diff = {'a': False, 'b': True, 'c': True}
print("固定差分策略:")
for col in target_cols:
    print(f"  {var_names[col]:<15} → {'一阶差分' if col_use_diff[col] else '不差分'}")

results = {}
for sheet_name in ["训练集", "实验集"]:
    df = pd.read_excel(input_path, sheet_name=sheet_name)
    df_out = df.copy()
    for col in target_cols:
        vals = pd.to_numeric(df[col], errors='coerce').fillna(0).values
        # 降雨量（a）是零膨胀数据，开启零膨胀 MAD 模式
        zero_inflated = (col == 'a')
        df_out[f'{col}_ab'] = detect_outliers_mad(vals, use_diff=col_use_diff[col],
                                                   zero_inflated=zero_inflated)
    results[sheet_name] = df_out

# ===== 输出到 Excel =====
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    results["训练集"].to_excel(writer, sheet_name="训练集", index=False)
    results["实验集"].to_excel(writer, sheet_name="实验集", index=False)
print(f"异常检测结果已保存至 {output_path}")
for sn in ["训练集", "实验集"]:
    print(f"{sn} 列: {results[sn].columns.tolist()}")

# ===== 统计异常数量 =====
for sn in ["训练集", "实验集"]:
    df = results[sn]
    print(f"\n{sn} 异常点统计:")
    for col in target_cols:
        n_ab = df[f'{col}_ab'].sum()
        print(f"  {var_names[col]}: {int(n_ab)} 个异常点 ({n_ab/len(df)*100:.2f}%)")

# ===== 绘图 =====
fig, axes = plt.subplots(3, 1, figsize=(14, 9))
fig.suptitle(f'异常检测 (k={k_value}) — 降雨量不差分(零膨胀MAD)，水压与微震一阶差分', fontsize=13)

t_train = np.arange(len(results["训练集"]))
t_exp = np.arange(len(results["实验集"]))

for i, col in enumerate(target_cols):
    ax = axes[i]
    use_diff = col_use_diff[col]
    zero_inflated = (col == 'a')
    # 对于零膨胀模式，直接在原始信号上画图，并标注单侧阈值
    for label, color, t in [("训练集", "#e74c3c", t_train), ("实验集", "#3498db", t_exp)]:
        df = results[label]
        vals = pd.to_numeric(df[col], errors='coerce').fillna(0).values
        if zero_inflated:
            # 零膨胀：画原始值（单位：mm），阈值直接在值域上计算
            plot_vals = vals
            nonzero = vals[vals > 0]
            if len(nonzero) >= 10:
                med_pos = np.median(nonzero)
                mad_pos = np.median(np.abs(nonzero - med_pos))
                thr = med_pos + k_value * mad_pos
            else:
                thr = np.percentile(vals, 99.5)
            flags = df[f'{col}_ab'].values
            ax.plot(t, plot_vals, color=color, alpha=0.7, linewidth=0.8, label=label)
            ox = np.where(flags == 1)[0]
            if len(ox) > 0:
                ax.scatter(ox, plot_vals[ox], color=color, s=15, edgecolors='k',
                           linewidths=0.3, zorder=5, label=f'{label}异常({len(ox)})')
            # 单侧阈值线
            ax.axhline(thr, color=color, ls='--', alpha=0.4, linewidth=0.8)
        else:
            # 非零膨胀：画 robust 标准化后的值
            plot_vals = np.insert(np.diff(vals), 0, 0) if use_diff else vals
            z = robust_normalize(plot_vals)
            flags = df[f'{col}_ab'].values
            ax.plot(t, z, color=color, alpha=0.7, linewidth=0.8, label=label)
            ox = np.where(flags == 1)[0]
            if len(ox) > 0:
                ax.scatter(ox, z[ox], color=color, s=15, edgecolors='k',
                           linewidths=0.3, zorder=5, label=f'{label}异常({len(ox)})')
            ax.axhline(k_value, color=color, ls='--', alpha=0.3)
            ax.axhline(-k_value, color=color, ls='--', alpha=0.3)

    ax.set_ylabel(var_names[col], fontsize=11)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.2)

axes[-1].set_xlabel('时间序号', fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.96])
save_path = os.path.join(script_dir, "detection_result.png")
plt.savefig(save_path, dpi=200)
print(f"\n检测结果图已保存至 {save_path}")
plt.show()
plt.close()
print("全部完成")
