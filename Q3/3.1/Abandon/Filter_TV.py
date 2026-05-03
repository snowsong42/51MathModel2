import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import CubicSpline
from skimage.restoration import denoise_tv_chambolle

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

# ==================== 2. 三次样条插值 + TV去噪 ====================
def cubic_spline_fill(series):
    """
    用三次样条插值填补缺失值；
    有效点少于4个时退化为线性插值；
    全部缺失时返回全 0。
    返回插值后的np.ndarray。
    """
    arr = series.values.astype(float)
    idx = np.arange(len(arr))
    valid_mask = ~np.isnan(arr)
    valid_idx = idx[valid_mask]
    valid_vals = arr[valid_mask]

    if len(valid_vals) == 0:
        # 全部缺失 → 返回全 0
        return np.zeros_like(arr)
    elif len(valid_vals) < 4:
        # 点数不够，线性插值
        filled = np.interp(idx, valid_idx, valid_vals)
    else:
        cs = CubicSpline(valid_idx, valid_vals, bc_type='natural')
        filled = cs(idx)
    return filled

def load_fill_and_denoise(df, col_map, keys, lambda_dict):
    """
    读取数据 -> 三次样条插值填充 -> TV去噪
    返回两个字典：
        data_filled : 插值后的数据（用于绘图）
        data_denoised: TV去噪后的数据（用于输出）
    """
    data_filled = {}
    data_denoised = {}
    for key in keys:
        series = pd.to_numeric(df[col_map[key]].copy(), errors='coerce')
        # 三次样条插值
        filled = cubic_spline_fill(series)
        data_filled[key] = filled

        # TV去噪
        denoised = denoise_tv_chambolle(
            filled,
            weight=lambda_dict[key],
            eps=1e-5,
            max_num_iter=200
        )

        data_denoised[key] = denoised
    return data_filled, data_denoised

train_keys = ['a', 'b', 'c', 'd', 'e']
exp_keys = ['a', 'b', 'c', 'd', 'e']

# 正则化参数（与原来完全一致）
lambda_dict = {'a': 10, 'b': 12, 'c': 6, 'd': 2, 'e': 3}

# 处理训练集
data_filled_train, data_denoised_train = load_fill_and_denoise(
    df_train, train_cols, train_keys, lambda_dict
)
# 处理实验集
data_filled_exp, data_denoised_exp = load_fill_and_denoise(
    df_exp, exp_cols, exp_keys, lambda_dict
)

# ==================== 3. 输出去噪结果到 xlsx ====================
# 训练集
df_train_out = pd.DataFrame({
    'Serial No.': df_train['Serial No. '].values if 'Serial No. ' in df_train.columns else df_train['Serial No.'].values,
    'a: Rainfall (mm)': data_denoised_train['a'],
    'b: Pore Water Pressure (kPa)': data_denoised_train['b'],
    'c: Microseismic Event Count': data_denoised_train['c'],
    'd: Deep Displacement (mm)': data_denoised_train['d'],
    'e: Surface Displacement (mm)': data_denoised_train['e'],
})
train_out_path = os.path.join(script_dir, "train_denoised.xlsx")
df_train_out.to_excel(train_out_path, index=False)
print(f"训练集去噪结果已保存至 {train_out_path}")

# 实验集
df_exp_out = pd.DataFrame({
    'Serial No. ': df_exp['Serial No. '].values,
    'Rainfall (mm)': data_denoised_exp['a'],
    'Pore Water Pressure (kPa)': data_denoised_exp['b'],
    'Microseismic Event Count': data_denoised_exp['c'],
    'Deep Displacement (mm)': data_denoised_exp['d'],
    'Surface Displacement (mm)': data_denoised_exp['e'],
})
exp_out_path = os.path.join(script_dir, "exp_denoised.xlsx")
df_exp_out.to_excel(exp_out_path, index=False)
print(f"实验集去噪结果已保存至 {exp_out_path}")

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
    filled = data_filled_train[key]        # 三次样条插值结果
    denoised = data_denoised_train[key]    # TV去噪结果
    t = np.arange(len(orig))
    ax.plot(t, orig, 'gray', alpha=0.3, linewidth=0.5, label='原始数据（含缺失）')
    ax.plot(t, filled, 'b--', alpha=0.6, linewidth=0.8, label='三次样条插值')
    ax.plot(t, denoised, 'r-', linewidth=1.5, label='TV去噪结果')
    ax.set_ylabel(key_names[key], fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.2)
axes[0].set_title('训练集各变量 三次样条插值 + TV 去噪 效果', fontsize=14)
axes[-1].set_xlabel('时间序号', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "denoising_result.png"), dpi=300)
print(f"去噪效果图已保存至 {os.path.join(script_dir, 'denoising_result.png')}")
plt.show()
plt.close()
