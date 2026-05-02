"""
============================================================
三阶段形变识别 (优化版) — 基于速度变化与持续性约束
适用数据集: Q3/ap3.xlsx (表面位移)
============================================================

算法步骤:
1. 预处理: 中值滤波去噪、缺失值插值
2. 计算平滑速度序列
3. 脉冲检测与抑制: 剔除短时速度尖峰, 避免误判为阶段转换
4. 第一阶段粗分: 利用速度CUSUM/PELT找到主加速起点 (缓慢→非缓慢)
5. 第二阶段细分: 在非缓慢段内, 检测速度从线性增长到指数/加速增长的转折点
6. 输出两个转换节点及其对应时间
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, medfilt
from scipy.interpolate import interp1d
from scipy.stats import linregress

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 工具函数 ====================
def moving_average(x, window):
    """简单移动平均"""
    return np.convolve(x, np.ones(window)/window, mode='same')

def remove_velocity_spikes(v, v_smooth, threshold_factor=5.0, min_duration=6):
    """
    检测并移除速度脉冲:
    若某点原始速度偏离平滑速度超过 threshold_factor * std, 
    且该高值持续点数少于 min_duration, 则替换为平滑值。
    """
    std = np.std(np.abs(v - v_smooth))
    spike_mask = np.abs(v - v_smooth) > threshold_factor * std
    v_clean = v.copy()
    # 找到连续脉冲段
    i = 0
    while i < len(spike_mask):
        if spike_mask[i]:
            start = i
            while i < len(spike_mask) and spike_mask[i]:
                i += 1
            end = i
            if end - start < min_duration:  # 短脉冲, 替换
                v_clean[start:end] = v_smooth[start:end]
        else:
            i += 1
    return v_clean

def cusum_change_point(series, threshold_factor=1.5):
    """
    基于速度序列的CUSUM双向检测, 寻找显著均值上跳点。
    返回变化点索引, 若无显著变化则返回 None。
    """
    series = np.asarray(series)
    n = len(series)
    if n < 10:
        return None
    # 计算正向和反向累计和
    cumsum_pos = np.cumsum(series - np.mean(series))
    cumsum_neg = np.cumsum(np.mean(series) - series)
    # 取正向累计和的最大值位置 (上跳)
    idx_up = np.argmax(cumsum_pos)
    # 简单阈值判断: 若最大累计和超过 threshold_factor * std of series * sqrt(n)
    threshold = threshold_factor * np.std(series) * np.sqrt(n)
    if cumsum_pos[idx_up] > threshold:
        return idx_up
    else:
        return None

def find_acceleration_shift(t, v_smooth, start_idx, end_idx):
    """
    在给定速度区间 [start_idx, end_idx] 内, 寻找从线性增长到加速增长的转折点。
    方法: 滑窗拟合二次函数, 寻找二次项系数 a 的突变点。
    """
    window = max(30, (end_idx - start_idx) // 3)
    half = window // 2
    a_profile = []
    for i in range(start_idx + half, end_idx - half):
        win = slice(i - half, i + half)
        t_win = t[win]
        v_win = v_smooth[win]
        # 二次拟合 v = a*t^2 + b*t + c
        coeffs = np.polyfit(t_win, v_win, 2)
        a_profile.append(coeffs[0])  # a 值
    a_profile = np.array(a_profile)
    # 找到 a 开始持续为正且显著增大的点
    # 简单策略: 连续若干点 a > 阈值
    mask = (a_profile > np.quantile(a_profile[:max(1,len(a_profile)//3)], 0.8))  # 后段a突然大
    # 连续区域
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return None
    # 取第一个连续区段
    diff = np.diff(indices)
    breaks = np.where(diff > 1)[0]
    if len(breaks) > 0:
        first_end = indices[breaks[0]]
    else:
        first_end = indices[-1]
    first_start = indices[0]
    # 返回该区段的中点对应的原始时间索引
    mid_local = (first_start + first_end) // 2
    real_idx = start_idx + half + mid_local
    return max(start_idx + half, min(real_idx, end_idx - half))


# ==================== 主程序 ====================
if __name__ == "__main__":
    # 1. 读取数据
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "../../Q3/ap3.xlsx")

    df = pd.read_excel(data_path, sheet_name=0)
    t_idx = df["Serial No. "].values
    x_raw = df["e: Surface Displacement (mm)"].values

    N = len(x_raw)
    dt = 10 / 60  # 小时
    t_h = (t_idx - t_idx[0]) * dt
    t_days = t_h / 24

    print("=" * 60)
    print("三阶段形变识别 (优化版)")
    print("=" * 60)
    print(f"数据长度: {N}, 时间跨度: {t_h[-1]:.1f} h ({t_days[-1]:.1f} days)")

    # 2. 预处理: 缺失值插值 + 中值滤波
    mask_nan = np.isnan(x_raw)
    if np.sum(mask_nan) > 0:
        x_filled = pd.Series(x_raw).interpolate(method='linear').values
        print(f"已插值 {np.sum(mask_nan)} 个缺失值")
    else:
        x_filled = x_raw.copy()

    # 中值滤波去除孤立野点
    x_smooth = medfilt(x_filled, kernel_size=13)

    # 3. 速度计算与平滑
    v_raw = np.gradient(x_smooth, t_h)
    # 对原始速度再做一次中值滤波
    v_raw = medfilt(v_raw, kernel_size=7)
    # 使用 Savitzky-Golay 滤波器平滑速度
    window_sg = min(151, len(v_raw) - 1 if len(v_raw) % 2 == 0 else len(v_raw))
    window_sg = window_sg if window_sg % 2 == 1 else window_sg - 1
    v_smooth = savgol_filter(v_raw, window_length=window_sg, polyorder=3)

    # 4. 脉冲抑制: 移除非物理的短时速度尖峰
    v_clean = remove_velocity_spikes(v_raw, v_smooth, threshold_factor=5.0, min_duration=6)
    # 再次平滑
    v_clean_smooth = savgol_filter(v_clean, window_length=window_sg, polyorder=3)

    # 5. 第一阶段粗分: 使用速度CUSUM寻找缓慢→加速的转换点
    # 为减小噪声影响，使用短窗口均值
    v_ma_short = moving_average(v_clean_smooth, 24)  # 4小时窗口
    cp1 = cusum_change_point(v_ma_short, threshold_factor=1.5)

    # 若 CUSUM 未检测到显著变化, 则使用位移-速度联合判断
    if cp1 is None or cp1 < 20 or cp1 > N - 50:
        # 备选: 基于位移的斜率突变检测
        # 使用位移滑窗斜率
        slope_windows = np.zeros(N - 20)
        for i in range(20, N):
            seg = x_smooth[i-20:i]
            slope = linregress(t_h[i-20:i], seg).slope
            slope_windows[i-20] = slope
        # 找到斜率从平稳转为持续升高的点
        threshold = np.mean(slope_windows[:max(1, len(slope_windows)//3)]) + 2*np.std(slope_windows[:max(1, len(slope_windows)//3)])
        candidates = np.where(slope_windows > threshold)[0]
        if len(candidates) > 0:
            cp1 = candidates[0] + 20
        else:
            cp1 = len(t_h) // 2  # 如果全部失败，假定一半处

    print(f"\n第一转换点 (缓慢→加速) 初步定位在索引 {cp1} (t = {t_h[cp1]:.2f} h)")

    # 6. 第二阶段细分: 在 cp1 之后的区域内寻找加速→快速的转换点
    if cp1 >= len(t_h) - 50:
        cp2 = cp1 + 1  # 无效情况
    else:
        cp2 = find_acceleration_shift(t_h, v_clean_smooth, cp1, len(t_h)-1)
        if cp2 is None or cp2 <= cp1:
            # 备选: 直接对速度做二分段
            cp2 = cp1 + (len(t_h) - cp1) // 3  # 取后1/3处

    print(f"第二转换点 (加速→快速) 初步定位在索引 {cp2} (t = {t_h[cp2]:.2f} h)")

    # 7. 强制输出两个断点，并保证间距合理
    cp1 = max(20, min(cp1, N-40))
    cp2 = max(cp1 + 20, min(cp2, N-10))

    # 8. 可视化
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # 图1: 位移
    ax = axes[0]
    ax.plot(t_days, x_smooth, 'b-', lw=0.8, alpha=0.8, label='中值滤波位移')
    ax.axvline(t_days[cp1], color='orange', ls='--', lw=2, label=f'转换点1: {t_days[cp1]:.2f} d')
    ax.axvline(t_days[cp2], color='red', ls='--', lw=2, label=f'转换点2: {t_days[cp2]:.2f} d')
    ax.axvspan(t_days[0], t_days[cp1], alpha=0.1, color='green')
    ax.axvspan(t_days[cp1], t_days[cp2], alpha=0.1, color='yellow')
    ax.axvspan(t_days[cp2], t_days[-1], alpha=0.1, color='red')
    ax.set_ylabel('Displacement (mm)')
    ax.set_title('三阶段形变划分 (优化算法)')
    ax.legend()

    # 图2: 速度
    ax = axes[1]
    ax.plot(t_days, v_raw, 'gray', lw=0.5, alpha=0.4, label='原始速度')
    ax.plot(t_days, v_clean_smooth, 'b', lw=1.5, label='平滑速度 (脉冲抑制后)')
    ax.axvline(t_days[cp1], color='orange', ls='--', lw=2)
    ax.axvline(t_days[cp2], color='red', ls='--', lw=2)
    ax.set_ylabel('Velocity (mm/h)')
    ax.set_title('速度序列与转换点')
    ax.legend()

    # 图3: 阶段示意 (加速度或速度分段均值)
    ax = axes[2]
    # 分阶段显示速度均值水平
    phases = [(0, cp1, 'Phase1: 缓慢'), (cp1, cp2, 'Phase2: 加速'), (cp2, N-1, 'Phase3: 快速')]
    for s, e, label in phases:
        ax.axvspan(t_days[s], t_days[e], alpha=0.2, label=label)
    ax.plot(t_days, v_clean_smooth, 'k', lw=1.0)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Velocity (mm/h)')
    ax.set_title('阶段速度示意')
    ax.legend()

    plt.tight_layout()
    fig_path = os.path.join(script_dir, "phase_identification_optimized.png")
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    print(f"\n图像已保存: {fig_path}")
    plt.show()

    # 9. 输出最终结果
    print("\n" + "=" * 60)
    print("最终阶段划分结果")
    print("=" * 60)
    print(f"缓慢匀速形变阶段:   索引 1 ~ {cp1}   (时间: {t_days[0]:.2f} ~ {t_days[cp1-1]:.2f} 天)")
    print(f"加速形变阶段:       索引 {cp1+1} ~ {cp2}   (时间: {t_days[cp1]:.2f} ~ {t_days[cp2-1]:.2f} 天)")
    print(f"快速形变阶段:       索引 {cp2+1} ~ {N}   (时间: {t_days[cp2]:.2f} ~ {t_days[-1]:.2f} 天)")
    print(f"转换节点1: 索引 {cp1} (时间 {t_days[cp1]:.2f} 天, {t_h[cp1]:.2f} h)")
    print(f"转换节点2: 索引 {cp2} (时间 {t_days[cp2]:.2f} 天, {t_h[cp2]:.2f} h)")
