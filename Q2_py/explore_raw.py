"""
原始数据探索性分析（EDA）
对标 Q1 filter.m 的可视化部分 + 补充分析
输出：
  - 3 张可视化图（原始时序、速度分布与平顺结果、加速度分布与平顺结果）
  - 命令窗口：数据基本统计信息

注意：清洗前后对比图（图4）已移至 preprocess.py
速度、加速度及其平顺结果直接从 Filtered 2.xlsx 读取
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()  # 交互模式：所有plt.show()不阻塞，多图同时弹出

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# ===== matplotlib 显示设置 =====
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']  # 中文字体 + 英文字体fallback
plt.rcParams['axes.unicode_minus'] = False           # 修复负号显示为方块

def main():
    # ===== 路径设置 =====
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config.SCRIPT_DIR = script_dir
    clean_path = os.path.join(script_dir, config.CLEAN_DATA)

    print("=" * 55)
    print("原始数据探索性分析 (EDA)")
    print("=" * 55)

    # ===== 1. 读取清洗后数据（含平顺结果） =====
    df = pd.read_excel(clean_path, sheet_name=0)
    d_orig = df["Surface Displacement (mm)"].values
    N = len(d_orig)
    dt = config.DT
    t = np.arange(N) * dt  # 小时

    # 从 xlsx 中直接读取速度、加速度及其平顺结果
    v = df["Velocity (mm/h)"].values
    v_smooth = df["Smoothed Velocity (mm/h)"].values
    a = df["Acceleration (mm/h²)"].values
    a_smooth = df["Smoothed Acceleration (mm/h²)"].values

    # 基本统计
    zero_count = np.sum(d_orig == 0)
    print(f"\n数据点总数: {N}")
    print(f"采样间隔: {dt*60:.0f} 分钟")
    print(f"时间跨度: {t[-1]/24:.2f} 天 ({t[-1]:.1f} 小时)")
    print(f"零值点数: {zero_count} ({zero_count/N*100:.2f}%)")
    print(f"位移范围: [{np.min(d_orig):.2f}, {np.max(d_orig):.2f}] mm")
    print(f"均值±标准差: {np.mean(d_orig):.4f} ± {np.std(d_orig):.4f} mm")
    print(f"(速度、加速度及平顺结果从 {config.CLEAN_DATA} 读取)")

    # ===== 4. 绘图 =====

    # 图1：原始位移时序 + 标记零值 + 3σ异常候选
    plt.figure(figsize=(14, 5))
    plt.plot(t / 24, d_orig, 'k-', linewidth=1.0, alpha=0.8, label='原始位移')

    # 标记零值位置
    if zero_count > 0:
        zero_pts = np.where(d_orig == 0)[0]
        plt.scatter(t[zero_pts] / 24, d_orig[zero_pts],
                    color='red', s=15, alpha=0.6, label=f'零值 ({zero_count} 点)')

    # 标记3σ异常候选
    mean_d = np.mean(d_orig)
    std_d = np.std(d_orig)
    outliers = d_orig > (mean_d + 3 * std_d)
    if np.any(outliers):
        plt.scatter(t[outliers] / 24, d_orig[outliers],
                    color='orange', s=10, alpha=0.5, label='3σ 异常候选')

    plt.xlabel("时间 (天)", fontsize=12)
    plt.ylabel("位移 (mm)", fontsize=12)
    plt.title("原始位移时序", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, "图1 原始位移时序.png"), dpi=200)
    plt.show()
    print("[图1] 原始位移时序 → 图1 原始位移时序.png")

    # 图2：速度分布直方图 + 时序（平顺结果）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    # 速度直方图 (bin=60)
    ax1.hist(v, bins=60, color='steelblue', edgecolor='none', alpha=0.7)
    ax1.axvline(np.median(v), color='red', linestyle='--', label=f'中位数={np.median(v):.3f}')
    ax1.axvline(np.mean(v), color='green', linestyle='--', label=f'均值={np.mean(v):.3f}')
    ax1.set_xlabel("速度 (mm/h)")
    ax1.set_ylabel("频数")
    ax1.set_title("速度分布")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 速度时序：原始半透明 + 平滑叠加（直接读取 xlsx 中的平顺结果）
    v_plot = v[:len(t)-1]  # 去掉末尾 NaN
    v_smooth_plot = v_smooth[:len(t)-1]
    ax2.plot(t[:-1] / 24, v_plot, 'b-', linewidth=0.3, alpha=0.3, label='原始速度')
    ax2.plot(t[:-1] / 24, v_smooth_plot, 'b-', linewidth=1.5, alpha=0.9, label='平滑速度')
    ax2.axhline(config.THRESH_SLOW, color='green', linestyle='--',
                linewidth=1.2, label=f'慢速阈值={config.THRESH_SLOW}')
    ax2.axhline(config.THRESH_FAST, color='red', linestyle='-.',
                linewidth=1.2, label=f'快速阈值={config.THRESH_FAST}')
    ax2.set_xlabel("时间 (天)")
    ax2.set_ylabel("速度 (mm/h)")
    ax2.set_title("速度平顺结果")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, "图2 速度分布与平顺结果.png"), dpi=200)
    plt.show()
    print("[图2] 速度分析 → 图2 速度分布与平顺结果.png")

    # 图3：加速度分布直方图 + 时序（平顺结果，无阈值线）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    # 加速度直方图 (bin=60)
    ax1.hist(a, bins=60, color='coral', edgecolor='none', alpha=0.7)
    ax1.axvline(np.median(a), color='red', linestyle='--', label=f'中位数={np.median(a):.3f}')
    ax1.axvline(np.mean(a), color='green', linestyle='--', label=f'均值={np.mean(a):.3f}')
    ax1.set_xlabel("加速度 (mm/h²)")
    ax1.set_ylabel("频数")
    ax1.set_title("加速度分布")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 加速度时序：原始半透明 + 平滑叠加（直接读取 xlsx 中的平顺结果）
    a_plot = a[1:-1]  # 去掉首尾 NaN
    a_smooth_plot = a_smooth[1:-1]
    ax2.plot(t[1:-1] / 24, a_plot, 'r-', linewidth=0.3, alpha=0.3, label='原始加速度')
    ax2.plot(t[1:-1] / 24, a_smooth_plot, 'r-', linewidth=1.5, alpha=0.9, label='平滑加速度')
    ax2.axhline(0, color='k', linestyle='-', linewidth=0.5)
    ax2.set_xlabel("时间 (天)")
    ax2.set_ylabel("加速度 (mm/h²)")
    ax2.set_title("加速度平顺结果")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, "图3 加速度分布与平顺结果.png"), dpi=200)
    plt.show()
    print("[图3] 加速度分析 → 图3 加速度分布与平顺结果.png")

    print("\n" + "=" * 55)
    print("EDA 完成！共生成 3 张图表（原始时序、速度平顺、加速度平顺）")
    print("=" * 55)
    plt.show(block=True)  # 保持所有图窗口打开

if __name__ == "__main__":
    main()
