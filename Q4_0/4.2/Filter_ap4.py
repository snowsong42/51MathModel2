"""
对 ap4_stage.xlsx 中训练集和实验集的 a(降雨量)、b(孔隙水压力)、c(微震事件数) 三列
进行去噪处理：
  1. 三次样条插值补缺失
  2. SG滤波（各列独立调参，仅削弱毛刺，保留显著异常）
输出 ap4_denoise.xlsx，保留原始格式，仅替换 a、b、c 数值。
可视化：3 个子图（a、b、c 的训练集+实验集平滑对比）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
import os

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

script_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(script_dir, "../4.1/ap4_stage.xlsx")
output_path = os.path.join(script_dir, "ap4_denoise.xlsx")

# ===== SG滤波参数（各列独立调参）=====
sg_params = {
    'a': {'window_length': 3,  'polyorder': 2},   # 降雨量：多点0值，轻度平滑
    'b': {'window_length': 11, 'polyorder': 3},   # 孔隙水压力：连续信号，适中平滑
    'c': {'window_length': 21,  'polyorder': 1},   # 微震事件数：离散计数，几乎不处理
}

# ===== 核心函数 =====
def cubic_spline_fill(series):
    """三次样条插值填充缺失值"""
    arr = series.values.astype(float)
    idx = np.arange(len(arr))
    valid_mask = ~np.isnan(arr)
    valid_idx = idx[valid_mask]
    valid_vals = arr[valid_mask]
    if len(valid_vals) == 0:
        return np.zeros_like(arr)
    elif len(valid_vals) < 4:
        return np.interp(idx, valid_idx, valid_vals)
    else:
        cs = CubicSpline(valid_idx, valid_vals, bc_type='natural')
        return cs(idx)


def process_column(series, window_length, polyorder):
    """三次样条插值 → SG滤波"""
    filled = cubic_spline_fill(series)
    # 确保 window_length 不超过序列长度且为奇数
    wl = min(window_length, len(filled) - 1) if len(filled) % 2 == 0 else min(window_length, len(filled))
    if wl % 2 == 0:
        wl -= 1
    wl = max(3, wl)
    smoothed = savgol_filter(filled, window_length=wl, polyorder=polyorder)
    return filled, smoothed


# ===== 数据读取与处理 =====
df_train = pd.read_excel(input_path, sheet_name="训练集")
df_exp = pd.read_excel(input_path, sheet_name="实验集")

target_cols = ['a', 'b', 'c']
col_names = {'a': '降雨量 (mm)', 'b': '孔隙水压力 (kPa)', 'c': '微震事件数'}

results = {}

for sheet_name, df in [("训练集", df_train), ("实验集", df_exp)]:
    df_out = df.copy()
    for col in target_cols:
        series = pd.to_numeric(df_out[col], errors='coerce')
        params = sg_params[col]
        _, smooth = process_column(series, params['window_length'], params['polyorder'])
        df_out[col] = smooth
    results[sheet_name] = df_out

# ===== 输出到 Excel =====
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    results["训练集"].to_excel(writer, sheet_name="训练集", index=False)
    results["实验集"].to_excel(writer, sheet_name="实验集", index=False)
print(f"去噪结果已保存至 {output_path}")

# ===== 可视化：3 个子图 =====
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

for i, col in enumerate(target_cols):
    ax = axes[i]

    train_raw = pd.to_numeric(df_train[col], errors='coerce').values
    train_smooth = results["训练集"][col].values
    exp_raw = pd.to_numeric(df_exp[col], errors='coerce').values
    exp_smooth = results["实验集"][col].values

    t = np.arange(len(train_raw))
    ax.plot(t, train_raw, 'gray', alpha=0.2, linewidth=0.5, label='训练集原始')
    ax.plot(t, train_smooth, 'r-', linewidth=1.5, label='训练集平滑')

    t_exp = np.arange(len(exp_raw))
    ax.plot(t_exp, exp_raw, 'orange', alpha=0.2, linewidth=0.5, label='实验集原始')
    ax.plot(t_exp, exp_smooth, 'b-', linewidth=1.5, label='实验集平滑')

    ax.set_ylabel(col_names[col], fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

axes[0].set_title(f'a/b/c 列去噪效果（三次样条插值 + SG滤波）', fontsize=13)
axes[-1].set_xlabel('时间序号', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "denoising_result.png"), dpi=200)
print(f"平滑效果图已保存至 {os.path.join(script_dir, 'denoising_result.png')}")
plt.show()
plt.close()
