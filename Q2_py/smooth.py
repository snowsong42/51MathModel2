"""
数据平滑脚本：基于已清洗位移（Filtered 2.xlsx）计算速度/加速度并平顺
流程：
  清洗位移 → 计算速度 → Savgol平滑 → 计算加速度 → Savgol平滑
  → 追加到 Filtered 2.xlsx

输出：
  - Filtered 2.xlsx      追加4列（原始速度、平滑速度、原始加速度、平滑加速度）
  - 命令窗口：平滑参数统计

注意：必须先运行 preprocess.py 生成清洗后位移，再运行本脚本。
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config.SCRIPT_DIR = script_dir
    clean_path = os.path.join(script_dir, config.CLEAN_DATA)

    print("=" * 55)
    print("数据平滑：Filtered 2  →  追加平顺结果")
    print("=" * 55)

    # ===== 1. 读取已清洗数据 =====
    df = pd.read_excel(clean_path, sheet_name=0)
    d = df["Surface Displacement (mm)"].values
    N = len(d)
    dt = config.DT

    # ===== 2. 计算速度并平滑 =====
    v_raw = np.diff(d) / dt
    v_smooth = savgol_filter(v_raw, window_length=config.SAVGOL_WINDOW,
                              polyorder=config.SAVGOL_POLY)

    # ===== 3. 计算加速度并平滑 =====
    a_raw = np.diff(v_smooth) / dt
    a_smooth = savgol_filter(a_raw, window_length=config.SAVGOL_WINDOW,
                              polyorder=config.SAVGOL_POLY)

    # 补齐长度（速度 N-1，加速度 N-2，首尾用 NaN 填充）
    v_raw_pad = np.concatenate([v_raw, [np.nan]])
    v_smooth_pad = np.concatenate([v_smooth, [np.nan]])
    a_raw_pad = np.concatenate([[np.nan], a_raw, [np.nan]])
    a_smooth_pad = np.concatenate([[np.nan], a_smooth, [np.nan]])

    # ===== 4. 追加到 Filtered 2.xlsx =====
    df["Velocity (mm/h)"] = np.round(v_raw_pad, 6)
    df["Smoothed Velocity (mm/h)"] = np.round(v_smooth_pad, 6)
    df["Acceleration (mm/h²)"] = np.round(a_raw_pad, 8)
    df["Smoothed Acceleration (mm/h²)"] = np.round(a_smooth_pad, 8)

    df.to_excel(clean_path, index=False, sheet_name="Sheet1")
    print(f"\n>>> 平顺结果已追加到: {config.CLEAN_DATA}")
    print(f"    新增列: Velocity, Smoothed Velocity, Acceleration, Smoothed Acceleration")

    # ===== 5. 统计输出 =====
    v_mean = np.nanmean(v_smooth)
    v_std = np.nanstd(v_smooth)
    a_mean = np.nanmean(a_smooth)
    a_std = np.nanstd(a_smooth)
    print(f"\n平滑速度: 均值={v_mean:.4f} mm/h, 标准差={v_std:.4f} mm/h")
    print(f"平滑加速度: 均值={a_mean:.6f} mm/h², 标准差={a_std:.6f} mm/h²")
    print(f"Savgol参数: 窗口={config.SAVGOL_WINDOW}, 阶数={config.SAVGOL_POLY}")
    print("\n平滑完成！")

if __name__ == "__main__":
    main()
