import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.ndimage import generic_filter
from scipy.signal import savgol_filter
import os

# ===== matplotlib 显示设置 =====
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

script_dir = os.path.dirname(os.path.abspath(__file__))

# ==================== 1. 数据读取 ====================
file_path = os.path.join(script_dir, "../ap3.xlsx")
df_train = pd.read_excel(file_path, sheet_name="训练集")
df_exp = pd.read_excel(file_path, sheet_name="实验集")

train_cols = {
    'a': 'a: Rainfall (mm)',
    'b': 'b: Pore Water Pressure (kPa)',
    'c': 'c: Microseismic Event Count',
    'd': 'd: Deep Displacement (mm)',
    'e': 'e: Surface Displacement (mm)'
}

exp_cols = {
    'a': 'Rainfall (mm)',
    'b': 'Pore Water Pressure (kPa)',
    'c': 'Microseismic Event Count',
    'd': 'Deep Displacement (mm)',
    'e': 'Surface Displacement (mm)'
}

# ==================== 2. 动态平滑函数（取代 TV 去噪） ====================
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
    # 计算局部标准差，归一化到 [0, 1]
    local_std = generic_filter(signal, np.std, size=window_std)
    std_max = local_std.max() if local_std.max() > 0 else 1.0
    weights = local_std / std_max                     # 0: 平坦, 1: 特征密集

    # 根据权重线性插值出每个点的平滑半径
    radii = base_radius - weights * (base_radius - sharp_radius)
    radii = np.clip(radii, sharp_radius, base_radius).astype(int)

    # 动态平滑：以半径为半宽做滑动平均
    smoothed = np.zeros_like(signal)
    n = len(signal)
    for i in range(n):
        r = max(1, radii[i])                 # 半径至少为 1
        start = max(0, i - r)
        end = min(n, i + r + 1)
        smoothed[i] = np.mean(signal[start:end])
    return smoothed


def load_fill_and_smooth(df, col_map, keys, smooth_params=None):
    """
    读取数据 -> 三次样条插值 -> 动态平滑 -> SG滤波
    返回两个字典：
        data_filled : 插值后的数据（用于绘图）
        data_smooth : 动态平滑+SG滤波后的数据（用于输出）
    """
    if smooth_params is None:
        smooth_params = {}   # 可针对每列传入不同的参数，这里全部使用全局默认值
    data_filled = {}
    data_smooth = {}
    for key in keys:
        series = pd.to_numeric(df[col_map[key]].copy(), errors='coerce')
        filled = cubic_spline_fill(series)
        data_filled[key] = filled

        # 第一步：动态平滑（GAUSS）
        params = smooth_params.get(key, {})
        smoothed = dynamic_smooth(filled, **params)
        # 第二步：SG滤波平滑，进一步去除残余高频噪声
        smoothed = savgol_filter(smoothed, window_length=21, polyorder=3)
        data_smooth[key] = smoothed
    return data_filled, data_smooth


train_keys = ['a', 'b', 'c', 'd', 'e']
exp_keys = ['a', 'b', 'c', 'd', 'e']

# 动态平滑参数（可根据每列特点单独设置，这里全部统一）
smooth_params = {
    'a': dict(window_std=500, base_radius=10, sharp_radius=0.1),
    'b': dict(window_std=500, base_radius=10, sharp_radius=3),
    'c': dict(window_std=500, base_radius=10, sharp_radius=3),
    'd': dict(window_std=500, base_radius=10, sharp_radius=3),
    'e': dict(window_std=500, base_radius=10, sharp_radius=3),
}

# 处理训练集
data_filled_train, data_smooth_train = load_fill_and_smooth(
    df_train, train_cols, train_keys, smooth_params
)
# 处理实验集
data_filled_exp, data_smooth_exp = load_fill_and_smooth(
    df_exp, exp_cols, exp_keys, smooth_params
)

# ==================== 3. 输出去噪结果到 xlsx ====================
# 训练集
serial_train = df_train['Serial No. '].values if 'Serial No. ' in df_train.columns else df_train['Serial No.'].values
df_train_out = pd.DataFrame({
    'Serial No.': serial_train,
    'a: Rainfall (mm)': data_smooth_train['a'],
    'b: Pore Water Pressure (kPa)': data_smooth_train['b'],
    'c: Microseismic Event Count': data_smooth_train['c'],
    'd: Deep Displacement (mm)': data_smooth_train['d'],
    'e: Surface Displacement (mm)': data_smooth_train['e'],
})
train_out_path = os.path.join(script_dir, "train_denoised.xlsx")
df_train_out.to_excel(train_out_path, index=False)
print(f"训练集平滑结果已保存至 {train_out_path}")

# 实验集
df_exp_out = pd.DataFrame({
    'Serial No. ': df_exp['Serial No. '].values,
    'Rainfall (mm)': data_smooth_exp['a'],
    'Pore Water Pressure (kPa)': data_smooth_exp['b'],
    'Microseismic Event Count': data_smooth_exp['c'],
    'Deep Displacement (mm)': data_smooth_exp['d'],
    'Surface Displacement (mm)': data_smooth_exp['e'],
})
exp_out_path = os.path.join(script_dir, "exp_denoised.xlsx")
df_exp_out.to_excel(exp_out_path, index=False)
print(f"实验集平滑结果已保存至 {exp_out_path}")

# ==================== 4. 可视化（仅训练集） ====================
key_names = {
    'a': '降雨量 (mm)', 'b': '孔隙水压力 (kPa)',
    'c': '微震事件数', 'd': '深部位移 (mm)', 'e': '表面位移 (mm)'
}

# 训练集原始数据（含缺失）用于对比
data_raw_train = {}
for key in train_keys:
    series = pd.to_numeric(df_train[train_cols[key]].copy(), errors='coerce')
    data_raw_train[key] = series

fig, axes = plt.subplots(5, 1, figsize=(16, 14))
for i, key in enumerate(['a', 'b', 'c', 'd', 'e']):
    ax = axes[i]
    orig = data_raw_train[key].values
    filled = data_filled_train[key]
    smoothed = data_smooth_train[key]
    t = np.arange(len(orig))
    ax.plot(t, orig, 'gray', alpha=0.3, linewidth=0.5, label='原始数据（含缺失）')
    ax.plot(t, filled, 'b--', alpha=0.6, linewidth=0.8, label='三次样条插值')
    ax.plot(t, smoothed, 'r-', linewidth=1.5, label='动态平滑 + SG滤波')
    ax.set_ylabel(key_names[key], fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.2)
axes[0].set_title('训练集各变量 三次样条插值 + 动态平滑 + SG滤波 效果', fontsize=14)
axes[-1].set_xlabel('时间序号', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "smoothing_result.png"), dpi=300)
print(f"平滑效果图已保存至 {os.path.join(script_dir, 'smoothing_result.png')}")
print("GAUSS参数: window_std=80, base_radius=15, sharp_radius=0.5")
print("SG滤波参数: window_length=21, polyorder=3")
plt.show()
plt.close()
