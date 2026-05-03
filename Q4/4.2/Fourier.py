"""
Q4 降噪前后频谱对比（只对 a、b、c 进行了降噪，因此仅对比这三列）
输入：ap4_stage.xlsx（原始，用于插值补缺失） + ap4_denoise.xlsx（降噪后）
输出：Fourier/spectrum_{a,b,c}.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.signal import find_peaks, welch

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "Fourier")
os.makedirs(output_dir, exist_ok=True)

# 只处理降噪的三列 a、b、c
keys = ['a', 'b', 'c']
key_names = {'a': '降雨量', 'b': '孔隙水压力', 'c': '微震事件数'}
col_names = {'a': 'a', 'b': 'b', 'c': 'c'}


def cubic_spline_fill(series):
    arr = series.values.astype(float)
    idx = np.arange(len(arr))
    mask = ~np.isnan(arr)
    valid_idx = idx[mask]
    valid_vals = arr[mask]
    if len(valid_vals) < 4:
        return np.interp(idx, valid_idx, valid_vals)
    cs = CubicSpline(valid_idx, valid_vals, bc_type='natural')
    return cs(idx)


# 1. 读取原始数据 & 三次样条插值
file_path = os.path.join(script_dir, "../4.1/ap4_stage.xlsx")
df_raw = pd.read_excel(file_path, sheet_name="训练集")

pre_denoise = {}
for key in keys:
    series = pd.to_numeric(df_raw[col_names[key]].copy(), errors='coerce')
    pre_denoise[key] = cubic_spline_fill(series)

# 2. 读取降噪后数据
df_den = pd.read_excel(os.path.join(script_dir, "ap4_denoise.xlsx"))
post_denoise = {}
for key in keys:
    post_denoise[key] = df_den[col_names[key]].values.astype(float)

print(f"已加载 Q4 训练集数据，长度={len(pre_denoise['a'])}")
print(f"降噪前 NaN 填充: a={np.isnan(pre_denoise['a']).sum()}, "
      f"b={np.isnan(pre_denoise['b']).sum()}, c={np.isnan(pre_denoise['c']).sum()}")


def get_envelope(f, Pxx, mode='upper'):
    min_dist = max(3, len(f) // 50)
    if mode == 'upper':
        peaks, _ = find_peaks(Pxx, distance=min_dist)
    else:
        peaks, _ = find_peaks(-Pxx, distance=min_dist)
    locs = f[peaks]
    pks = Pxx[peaks]
    if len(locs) < 2:
        return f, np.full_like(f, np.mean(Pxx))
    locs = np.unique(np.concatenate([[f[0]], locs, [f[-1]]]))
    pks = np.interp(locs, f[peaks], Pxx[peaks])
    pchip = PchipInterpolator(locs, pks, extrapolate=True)
    return f, pchip(f)


# 3. 画频谱及包络图（a、b、c 各一张）
for key in keys:
    sig_pre = pre_denoise[key]
    sig_post = post_denoise[key]
    N = len(sig_pre)

    # 信噪比
    residual = sig_pre - sig_post
    SNR = 20 * np.log10(np.std(sig_pre) / np.std(residual))
    print(f'{key_names[key]}: SNR = {SNR:.2f} dB')

    # 功率谱估计（Welch 方法）
    window = min(256, N)
    noverlap = window // 2
    nfft = 1024
    f, Pxx_pre = welch(sig_pre - np.mean(sig_pre), fs=1.0, window='hamming',
                       nperseg=window, noverlap=noverlap, nfft=nfft)
    _, Pxx_post = welch(sig_post - np.mean(sig_post), fs=1.0, window='hamming',
                        nperseg=window, noverlap=noverlap, nfft=nfft)

    # 包络线
    f_upper_pre, P_upper_pre = get_envelope(f, Pxx_pre, 'upper')
    f_lower_pre, P_lower_pre = get_envelope(f, Pxx_pre, 'lower')
    f_upper_post, P_upper_post = get_envelope(f, Pxx_post, 'upper')
    f_lower_post, P_lower_post = get_envelope(f, Pxx_post, 'lower')

    # 绘图
    plt.figure(figsize=(10, 6.5))
    ax = plt.gca()
    ax.set_yscale('log')

    plt.fill_between(f_upper_pre, P_upper_pre, P_lower_pre, color='b', alpha=0.08)
    plt.plot(f_upper_pre, P_upper_pre, 'b--', linewidth=1.0, label='降噪前上包络')
    plt.plot(f_lower_pre, P_lower_pre, 'b--', linewidth=1.0, label='降噪前下包络')
    plt.plot(f, Pxx_pre, 'royalblue', linewidth=1.2, label='降噪前功率谱')

    plt.fill_between(f_upper_post, P_upper_post, P_lower_post, color='r', alpha=0.08)
    plt.plot(f_upper_post, P_upper_post, 'r--', linewidth=1.0, label='降噪后上包络')
    plt.plot(f_lower_post, P_lower_post, 'r--', linewidth=1.0, label='降噪后下包络')
    plt.plot(f, Pxx_post, 'tomato', linewidth=1.2, label='降噪后功率谱')

    plt.xlabel('频率 (周期/序号)')
    plt.ylabel('功率谱密度')
    plt.title(f'{key_names[key]} 降噪前后频谱对比 (信噪比 = {SNR:.2f} dB)')
    plt.legend(loc='best', fontsize=8)
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'spectrum_{key}.png'), dpi=300)
    plt.close()

print(f'频谱图已保存至 {output_dir}')
