"""
原始数据探索性分析（EDA）
对标 Q1 filter.m 的可视化部分 + 补充分析
输出：
  - 4 张可视化图（原始时序、速度分布、加速度水平、清洗前后对比）
  - 命令窗口：数据基本统计信息
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import medfilt, savgol_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

def main():
    # ===== 路径设置 =====
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config.SCRIPT_DIR = script_dir
    in_path = os.path.join(script_dir, config.RAW_DATA)
    clean_path = os.path.join(script_dir, config.CLEAN_DATA)

    print("=" * 55)
    print("原始数据探索性分析 (EDA)")
    print("=" * 55)

    # ===== 1. 读取原始数据 =====
    df = pd.read_excel(in_path, sheet_name=0)
    d_orig = df["Surface Displacement (mm)"].values
    N = len(d_orig)
    dt = config.DT
    t = np.arange(N) * dt  # 小时

    # 基本统计
    zero_count = np.sum(d_orig == 0)
    print(f"\n数据点总数: {N}")
    print(f"采样间隔: {dt*60:.0f} 分钟")
    print(f"时间跨度: {t[-1]/24:.2f} 天 ({t[-1]:.1f} 小时)")
    print(f"零值点数: {zero_count} ({zero_count/N*100:.2f}%)")
    print(f"位移范围: [{np.min(d_orig):.2f}, {np.max(d_orig):.2f}] mm")
    print(f"均值±标准差: {np.mean(d_orig):.4f} ± {np.std(d_orig):.4f} mm")

    # 读取清洗后数据（如果有）
    try:
        df_clean = pd.read_excel(clean_path, sheet_name=0)
        d_clean = df_clean["Surface Displacement (mm)"].values
        has_clean = True
    except FileNotFoundError:
        # 如果还没有清洗数据，手动做一遍轻量清洗用于对比
        d_interp = d_orig.copy()
        zero_idx = np.where(d_interp == 0)[0]
        if len(zero_idx) > 0:
            x = np.arange(N)
            mask = d_interp != 0
            d_interp = np.interp(x, x[mask], d_interp[mask])
        d_clean = medfilt(d_interp, kernel_size=config.MEDFILT_KERNEL)
        has_clean = False

    # ===== 2. 计算速度 =====
    v = np.diff(d_orig) / dt

    # ===== 3. 加速度 =====
    v_smooth = savgol_filter(v, window_length=config.SAVGOL_WINDOW,
                              polyorder=config.SAVGOL_POLY)
    a = np.diff(v_smooth) / dt

    # ===== 4. 绘图 =====

    # 图1：原始位移时序 + 标记零值 + 3σ异常候选
    plt.figure(figsize=(14, 5))
    plt.plot(t / 24, d_orig, 'k-', linewidth=0.5, alpha=0.7, label='Raw displacement')

    # 标记零值位置
    if zero_count > 0:
        zero_pts = np.where(d_orig == 0)[0]
        plt.scatter(t[zero_pts] / 24, d_orig[zero_pts],
                    color='red', s=15, alpha=0.6, label=f'Zero value ({zero_count} pts)')

    # 标记3σ异常候选
    mean_d = np.mean(d_orig)
    std_d = np.std(d_orig)
    outliers = d_orig > (mean_d + 3 * std_d)
    if np.any(outliers):
        plt.scatter(t[outliers] / 24, d_orig[outliers],
                    color='orange', s=10, alpha=0.5, label='3sigma outlier candidates')

    plt.xlabel("Time (days)", fontsize=12)
    plt.ylabel("Displacement (mm)", fontsize=12)
    plt.title("Original Displacement Time Series", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, "eda_raw_series.png"), dpi=200)
    plt.show()
    print("[图1] 原始位移时序 → eda_raw_series.png")

    # 图2：速度分布直方图 + 时序
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # 速度直方图
    axes[0].hist(v, bins=100, color='steelblue', edgecolor='none', alpha=0.7)
    axes[0].axvline(np.median(v), color='red', linestyle='--', label=f'Median={np.median(v):.3f}')
    axes[0].axvline(np.mean(v), color='green', linestyle='--', label=f'Mean={np.mean(v):.3f}')
    axes[0].set_xlabel("Velocity (mm/h)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Velocity Distribution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 速度时序
    axes[1].plot(t[:-1] / 24, v, 'b-', linewidth=0.4, alpha=0.5, label='Raw velocity')
    axes[1].axhline(config.THRESH_SLOW, color='green', linestyle='--',
                    label=f'Slow threshold={config.THRESH_SLOW}')
    axes[1].axhline(config.THRESH_FAST, color='red', linestyle='--',
                    label=f'Fast threshold={config.THRESH_FAST}')
    axes[1].set_xlabel("Time (days)")
    axes[1].set_ylabel("Velocity (mm/h)")
    axes[1].set_title("Velocity Time Series")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, "eda_velocity.png"), dpi=200)
    plt.show()
    print("[图2] 速度分析 → eda_velocity.png")

    # 图3：加速度水平
    plt.figure(figsize=(14, 4))
    plt.plot(t[1:-1] / 24, a, 'r-', linewidth=0.6, alpha=0.7, label='Acceleration')
    plt.axhline(0, color='k', linestyle='--', linewidth=0.5)
    plt.xlabel("Time (days)", fontsize=12)
    plt.ylabel("Acceleration (mm/h^2)", fontsize=12)
    plt.title("Acceleration Level", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, "eda_acceleration.png"), dpi=200)
    plt.show()
    print("[图3] 加速度水平 → eda_acceleration.png")

    # 图4：原始 vs 清洗后对比
    plt.figure(figsize=(14, 6))
    plt.plot(t / 24, d_orig, 'gray', linewidth=0.4, alpha=0.5, label='Raw')
    plt.plot(t / 24, d_clean, 'b-', linewidth=0.8, label='Cleaned')
    plt.xlabel("Time (days)", fontsize=12)
    plt.ylabel("Displacement (mm)", fontsize=12)
    plt.title("Raw vs Cleaned Displacement", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, "eda_raw_vs_clean.png"), dpi=200)
    plt.show()
    print("[图4] 原始 vs 清洗后 → eda_raw_vs_clean.png")

    print("\n" + "=" * 55)
    print("EDA 完成！共生成 4 张图表")
    print("=" * 55)

if __name__ == "__main__":
    main()
