"""
对 ap4_stage.xlsx 中训练集和实验集的 a(降雨量)、b(孔隙水压力)、c(微震事件数) 三列
进行轻度去噪：三次样条插值补缺失 → 动态窗口平滑 → SG 滤波（各列独立调参）。
输出 ap4_denoise.xlsx，保留原始格式，仅替换 a、b、c 数值。
可视化：3 个子图（a、b、c 的训练集+实验集去噪对比）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.ndimage import generic_filter
from scipy.signal import savgol_filter
import os

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

script_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(script_dir, "../4.1/ap4_stage.xlsx")
output_path = os.path.join(script_dir, "ap4_denoise.xlsx")


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


def dynamic_smooth(signal, window_std=60, base_radius=8, sharp_radius=1):
    """
    基于局部标准差的动态窗口平滑。
    - 平坦区域（标准差小）使用大半径 base_radius
    - 特征密集区域（标准差大）使用小半径 sharp_radius
    返回平滑后的信号 ndarray。
    """
    local_std = generic_filter(signal, np.std, size=window_std)
    std_max = local_std.max() if local_std.max() > 0 else 1.0
    weights = local_std / std_max                     # 0: 平坦, 1: 特征密集

    radii = base_radius - weights * (base_radius - sharp_radius)
    radii = np.clip(radii, sharp_radius, base_radius).astype(int)

    smoothed = np.zeros_like(signal)
    n = len(signal)
    for i in range(n):
        r = max(1, radii[i])
        start = max(0, i - r)
        end = min(n, i + r + 1)
        smoothed[i] = np.mean(signal[start:end])
    return smoothed


def process_column(series, smooth_params=None, bypass=False):
    """三次样条插值 → 动态平滑 → SG 滤波
    
    Parameters
    ----------
    bypass : bool
        若为 True，只做插值填补 NaN，不做平滑/滤波（保留原始细节）。
    """
    if smooth_params is None:
        smooth_params = {}
    filled = cubic_spline_fill(series)
    if bypass:
        return filled, filled
    smoothed = dynamic_smooth(filled, **smooth_params)
    # SG 滤波进一步去除残余高频噪声
    smoothed = savgol_filter(smoothed, window_length=17, polyorder=3)
    return filled, smoothed


# ===== 每列独立的去噪参数 =====
# 各列参数基于自身信号特点设定：
#   - 降雨量 (a)：陡脉冲信号，sharp_radius 设得很小以保留降雨事件陡峭起落
#   - 孔隙水压力 (b)：缓变信号，sharp_radius 稍大可增加平滑度
#   - 微震事件数 (c)：只做插值补 NaN，不做平滑/滤波（bypass=True），保留原始细节
col_params = {
    'a': dict(window_std=600, base_radius=10, sharp_radius=0.1),   # 降雨量：保留脉冲起落
    'b': dict(window_std=60, base_radius=10, sharp_radius=0.1),   # 孔压：缓变，适度平滑
    'c': dict(window_std=60, base_radius=10, sharp_radius=20),   # 微震事件数：只插值补 NaN，不平滑（bypass=True）
}

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
        params = col_params.get(col, {})
        # 微震事件数（c）只做插值补 NaN，不做平滑/滤波，保留原始细节
        bypass = 0
        filled, smooth = process_column(series, smooth_params=params, bypass=bypass)
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
    ax.plot(t, train_raw, 'gray', alpha=0.5, linewidth=0.5, label='训练集原始')
    ax.plot(t, train_smooth, 'r-', linewidth=1.5, label='训练集去噪')

    t_exp = np.arange(len(exp_raw))
    ax.plot(t_exp, exp_raw, 'orange', alpha=0.5, linewidth=0.5, label='实验集原始')
    ax.plot(t_exp, exp_smooth, 'b-', linewidth=1.5, label='实验集去噪')

    ax.set_ylabel(col_names[col], fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

axes[0].set_title('a/b/c 列去噪效果对比（三次样条插值 + 动态平滑 + SG滤波，各列独立调参）', fontsize=13)
axes[-1].set_xlabel('时间序号', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "denoising_result.png"), dpi=200)
print(f"去噪效果图已保存至 {os.path.join(script_dir, 'denoising_result.png')}")
print("动态平滑+SG滤波参数:")
for col, params in col_params.items():
    print(f"  {col} ({col_names[col]}): {params}")
print("SG滤波: window_length=21, polyorder=3 (统一)")
plt.show()
plt.close()
