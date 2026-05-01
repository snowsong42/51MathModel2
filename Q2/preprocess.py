"""
数据清洗脚本：Attachment 2 → Filtered 2
清洗规则：
  1) 零值视为缺失值，线性插值补齐
  2) 中值滤波抑制瞬时跳变（窗口可调）
输出：
  - Filtered 2.xlsx      清洗后数据（序号、时间、位移）
  - 图4 清洗前后对比.png  原始 vs 清洗后对比图（含差值时序）
  - 命令窗口：清洗前后统计对比
对标 Q1 filter.m 的数据清洗流程
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()
from scipy.signal import medfilt

# ===== matplotlib 显示设置 =====
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']  # 中文字体 + 英文字体fallback
plt.rcParams['axes.unicode_minus'] = False           # 修复负号显示为方块

# 确保能找到同目录下的 config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

def main():
    # ===== 路径设置 =====
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config.SCRIPT_DIR = script_dir
    in_path = os.path.join(script_dir, config.RAW_DATA)
    out_path = os.path.join(script_dir, config.CLEAN_DATA)

    print("=" * 55)
    print("数据清洗：Attachment 2  →  Filtered 2")
    print("=" * 55)

    # ===== 1. 读取原始数据 =====
    print(f"\n>>> 正在读取: {config.RAW_DATA} ...")
    df = pd.read_excel(in_path, sheet_name=0)
    d_orig = df["Surface Displacement (mm)"].values
    N = len(d_orig)
    dt = config.DT
    time_h = np.arange(N) * dt
    print(f"读取完成，共 {N} 个数据点，采样间隔 {dt*60:.0f} 分钟")

    # ===== 2. 零值插值 =====
    d_interp = d_orig.copy()
    zero_idx = np.where(d_interp == 0)[0]
    n_zero = len(zero_idx)
    if n_zero > 0:
        x = np.arange(N)
        mask = d_interp != 0
        d_interp = np.interp(x, x[mask], d_interp[mask])
        print(f"零值插值: 发现 {n_zero} 个零值点，已线性插值补齐")
    else:
        print("零值插值: 无零值点，无需插值")

    # ===== 3. 中值滤波 =====
    kernel = config.MEDFILT_KERNEL
    d_filt = medfilt(d_interp, kernel_size=kernel)
    print(f"中值滤波: 窗口 {kernel} 点，已抑制瞬时跳变")

    # ===== 4. 保存 Filtered 2.xlsx =====
    output_df = pd.DataFrame({
        "Serial No.": np.arange(1, N + 1),
        "Time (hours)": np.round(time_h, 4),
        "Surface Displacement (mm)": np.round(d_filt, 4)
    })
    output_df.to_excel(out_path, index=False, sheet_name="Sheet1")
    print(f"\n>>> 清洗后数据已保存: {config.CLEAN_DATA}")

    # ===== 5. 清洗前后统计对比 =====
    print("\n" + "=" * 55)
    print("清洗前后统计对比")
    print("-" * 55)
    print(f"{'指标':<20} {'清洗前':>12} {'清洗后':>12}")
    print("-" * 55)
    print(f"{'均值 (mm)':<20} {np.mean(d_orig):>12.4f} {np.mean(d_filt):>12.4f}")
    print(f"{'标准差 (mm)':<20} {np.std(d_orig):>12.4f} {np.std(d_filt):>12.4f}")
    print(f"{'最小值 (mm)':<20} {np.min(d_orig):>12.4f} {np.min(d_filt):>12.4f}")
    print(f"{'最大值 (mm)':<20} {np.max(d_orig):>12.4f} {np.max(d_filt):>12.4f}")
    print(f"{'零值点数':<20} {n_zero:>12} {0:>12}")
    print("=" * 55)
    print("预处理完成！")

    # ===== 6. 图4：清洗前后对比 =====
    print("\n>>> 生成图4：清洗前后对比...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # 上：原始 vs 清洗后叠加
    ax1.plot(time_h / 24, d_orig, 'gray', linewidth=0.4, alpha=0.6, label='Raw')
    ax1.plot(time_h / 24, d_filt, 'b-', linewidth=0.8, label='Cleaned')
    ax1.set_ylabel("Displacement (mm)", fontsize=12)
    ax1.set_title("Raw vs Cleaned Displacement", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 下：差值时序（清洗 - 原始）
    diff = d_filt - d_orig
    ax2.plot(time_h / 24, diff, 'r-', linewidth=0.5, alpha=0.7, label='Difference (Cleaned - Raw)')
    ax2.axhline(0, color='k', linestyle='--', linewidth=0.5)
    ax2.set_xlabel("Time (days)", fontsize=12)
    ax2.set_ylabel("Difference (mm)", fontsize=12)
    ax2.set_title("Cleaning Effect (Difference)", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, "图4 清洗前后对比.png"), dpi=200)
    plt.show()
    print("[图4] 清洗前后对比 → 图4 清洗前后对比.png")
    plt.show(block=True)

if __name__ == "__main__":
    main()
