import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.ndimage import generic_filter, uniform_filter1d
import os
# ===== matplotlib 显示设置 =====
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ----- 1. 数据读取与插值（与原脚本一致） -----
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "../ap3.xlsx")
df = pd.read_excel(file_path, sheet_name="训练集")
col = 'a: Rainfall (mm)'
raw = pd.to_numeric(df[col], errors='coerce').values

def cubic_fill(series):
    arr = series.astype(float)
    idx = np.arange(len(arr))
    valid = ~np.isnan(arr)
    if valid.sum() < 4:
        return np.interp(idx, idx[valid], arr[valid])
    cs = CubicSpline(idx[valid], arr[valid], bc_type='natural')
    return cs(idx)

filled = cubic_fill(raw)

# ----- 2. 动态平滑（基于局部标准差） -----
window_std = 70          # 计算局部标准差的窗口
base_smooth = 5         # 平滑基础窗口（平坦区）
sharp_smooth = 1         # 特征区保留细节的最小窗口

local_std = generic_filter(filled, np.std, size=window_std)
std_max = local_std.max() if local_std.max()>0 else 1
weights = local_std / std_max     # 0~1，1表示特征密集

# 根据权重线性插值出每个点的平滑窗口大小（取整）
smooth_wins = (base_smooth - weights * (base_smooth - sharp_smooth)).astype(int)
smooth_wins = np.clip(smooth_wins, sharp_smooth, base_smooth)

# 动态平滑：对每个点用不同大小的均匀滤波
smoothed = np.zeros_like(filled)
for i in range(len(filled)):
    half = max(1, smooth_wins[i])
    start = max(0, i - half)
    end = min(len(filled), i + half + 1)
    smoothed[i] = np.mean(filled[start:end])

# ----- 3. 可视化 -----
plt.figure(figsize=(14,4))
plt.plot(filled, 'gray', alpha=0.4, label='插值后信号')
plt.plot(smoothed, 'r', linewidth=1.2, label='动态平滑')
plt.legend(); plt.title('动态窗口平滑（保留尖锐特征）')
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "dynamic_smooth.png"), dpi=150)
plt.show()