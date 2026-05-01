"""
数据清洗脚本：Attachment 2 → Filtered 2
清洗规则：
  1) 零值视为缺失值，线性插值补齐
  2) 中值滤波抑制瞬时跳变（窗口可调）
输出：
  - Filtered 2.xlsx      清洗后数据（序号、时间、位移）
  - 命令窗口：清洗前后统计对比
对标 Q1 filter.m 的数据清洗流程
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.signal import medfilt

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

if __name__ == "__main__":
    main()
