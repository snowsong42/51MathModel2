import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, find_peaks
from scipy.interpolate import CubicSpline
import os

# ===== 设置 =====
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "Fourier")
os.makedirs(output_dir, exist_ok=True)

# ===== 读取数据 =====
df = pd.read_csv(os.path.join(script_dir, 'Filtered_Result.csv'))
raw = df['RawDisplacement'].values.astype(float)
filtered = df['FilteredDisplacement'].values.astype(float)

valid = ~np.isnan(raw) & ~np.isnan(filtered)
raw = raw[valid]
filtered = filtered[valid]
N = len(raw)
print(f'有效数据点数：{N}')

# ===== 信噪比 =====
residual = raw - filtered
SNR = 20 * np.log10(np.std(raw) / np.std(residual))
print(f'信噪比 SNR = {SNR:.2f} dB')

# ===== 功率谱 =====
window = min(256, N)
noverlap = window // 2
nfft = 1024

f, Pxx_raw = welch(raw - np.mean(raw), fs=1.0, window='hamming',
                   nperseg=window, noverlap=noverlap, nfft=nfft)
_, Pxx_filt = welch(filtered - np.mean(filtered), fs=1.0, window='hamming',
                    nperseg=window, noverlap=noverlap, nfft=nfft)

# ===== 包络函数 =====
from scipy.interpolate import PchipInterpolator

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
    # 强制包含首尾频率点，避免外推
    locs = np.unique(np.concatenate([[f[0]], locs, [f[-1]]]))
    pks = np.interp(locs, f[peaks], Pxx[peaks])  # 用线性插值获取端点的包络值
    # 改用 Pchip 保持单调不振荡
    pchip = PchipInterpolator(locs, pks, extrapolate=True)
    return f, pchip(f)

f_upper_raw, P_upper_raw = get_envelope(f, Pxx_raw, 'upper')
f_lower_raw, P_lower_raw = get_envelope(f, Pxx_raw, 'lower')
f_upper_filt, P_upper_filt = get_envelope(f, Pxx_filt, 'upper')
f_lower_filt, P_lower_filt = get_envelope(f, Pxx_filt, 'lower')

# ===== 绘图 =====
plt.figure(figsize=(10, 6.5))
ax = plt.gca()
ax.set_yscale('log')

# 降噪后填充
plt.fill_between(f_upper_filt, P_upper_filt, P_lower_filt, color='r', alpha=0.08)
plt.plot(f_upper_filt, P_upper_filt, 'r--', linewidth=1.0, label='降噪后上包络')
plt.plot(f_lower_filt, P_lower_filt, 'r--', linewidth=1.0, label='降噪后下包络')
plt.plot(f, Pxx_filt, 'r-', linewidth=1.2, label='降噪后功率谱')

# 降噪前填充
plt.fill_between(f_upper_raw, P_upper_raw, P_lower_raw, color='b', alpha=0.08)
plt.plot(f_upper_raw, P_upper_raw, 'b--', linewidth=1.0, label='降噪前上包络')
plt.plot(f_lower_raw, P_lower_raw, 'b--', linewidth=1.0, label='降噪前下包络')
plt.plot(f, Pxx_raw, 'b-', linewidth=1.2, label='降噪前功率谱')

plt.xlabel('频率 (周期/序号)')
plt.ylabel('功率谱密度')
plt.title(f'降噪前后傅里叶频谱对比 (信噪比 = {SNR:.2f} dB)')
plt.legend(loc='best')
plt.grid(True, which='both', alpha=0.3)
plt.tight_layout()

save_path = os.path.join(output_dir, 'spectrum_comparison.png')
plt.savefig(save_path, dpi=300)
plt.close()
print(f'图片已保存至 {save_path}')