"""
阶段转换节点检测
提供两种算法：
  1) sliding_window_detect()  — 简化滑动窗口方案（原 velocity.py）
  2) mad_persistence_detect() — MAD+阶跃+持久性方案（原 stage_recognition.py）
输入：清洗后数据 (Filtered 2.xlsx, 包含 Smoothed Velocity 列)
输出：节点1、节点2的位移点序号，以及对应时间
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# ===== matplotlib 显示设置 =====
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']  # 中文字体 + 英文字体fallback
plt.rcParams['axes.unicode_minus'] = False           # 修复负号显示为方块

def _load_clean_data():
    """读取清洗后数据，返回位移数组、时间数组、dt、Smoothed Velocity"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, config.CLEAN_DATA)
    df = pd.read_excel(path, sheet_name=0)
    d = df["Surface Displacement (mm)"].values
    v_smooth = df["Smoothed Velocity (mm/h)"].values
    dt = config.DT
    t = np.arange(len(d)) * dt
    return d, t, dt, v_smooth


def sliding_window_detect():
    """
    简化滑动窗口方案（原 velocity.py）
    基于速度中位数窗口 + 双阈值 + 持续性检验
    使用 Filtered 2.xlsx 中预计算的 Smoothed Velocity
    """
    print("-" * 50)
    print("检测方法: 滑动窗口法（简化方案）")
    print("-" * 50)

    d, _, dt, v_smooth = _load_clean_data()
    N = len(d)

    # 使用已平滑的速度数据（从 xlsx 读取，末尾有 NaN，要去掉）
    v = v_smooth[~np.isnan(v_smooth)]

    # 参数
    L = config.SLIDING_WINDOW
    thresh_slow = config.THRESH_SLOW
    thresh_fast = config.THRESH_FAST
    min_gap = config.MIN_GAP
    persist_slow = config.PERSIST_SLOW
    persist_fast = config.PERSIST_FAST

    # 滑动窗口中位数速度
    v_med = np.full_like(v, np.nan)
    for i in range(L, len(v) - L):
        v_med[i] = np.median(v[i - L + 1 : i + L])

    # 节点1：匀速→加速
    node1 = None
    for i in range(L, len(v) - L):
        if v_med[i] > thresh_slow:
            if np.all(v_med[i : i + persist_slow] > thresh_slow):
                node1 = i
                break

    # 节点2：加速→快速
    node2 = None
    if node1 is not None:
        start = node1 + min_gap
        for i in range(start, len(v) - L):
            if v_med[i] > thresh_fast:
                if np.all(v_med[i : i + persist_fast] > thresh_fast):
                    node2 = i
                    break

    # 转换到位移点序号（1-index）
    idx1 = node1 + 1 if node1 is not None else None
    idx2 = node2 + 1 if node2 is not None else None
    t1 = (idx1 - 1) * dt if idx1 else None
    t2 = (idx2 - 1) * dt if idx2 else None

    _print_nodes(idx1, t1, idx2, t2)
    return idx1, t1, idx2, t2


def mad_persistence_detect():
    """
    MAD+阶跃量+持久性检验方案（原 stage_recognition.py，对应问题2.md文档算法）
    完整实现文档中的3层过滤机制
    使用 Filtered 2.xlsx 中预计算的速度数据
    """
    print("-" * 50)
    print("检测方法: MAD+持久性法（完整文档算法）")
    print("-" * 50)

    d, _, dt, v_smooth = _load_clean_data()

    # 使用原始速度（从 xlsx 读取，末尾有 NaN）
    df = pd.read_excel(os.path.join(os.path.dirname(os.path.abspath(__file__)), config.CLEAN_DATA), sheet_name=0)
    v_raw = df["Velocity (mm/h)"].values
    v = v_raw[~np.isnan(v_raw)]  # 去掉末尾 NaN

    N = len(d)

    # 参数
    L = config.MAD_WINDOW
    H = config.MAD_HOLD
    C_jump = config.C_JUMP
    C_trans = config.C_TRANS
    theta_back = config.THETA_BACK
    speed_low = config.SPEED_THRESH_LOW
    speed_high = config.SPEED_THRESH_HIGH

    # ---- 第2层：MAD跳变指标 ----
    v_med = np.full_like(v, np.nan)
    v_mad = np.full_like(v, np.nan)
    for i in range(L, len(v)):
        win = v[i - L + 1 : i + 1]
        v_med[i] = np.median(win)
        v_mad[i] = np.median(np.abs(win - v_med[i]))

    z_score = np.abs(v - v_med) / (v_mad + 1e-6)

    # ---- 第3层：阶跃量 + 持久性检验 ----
    R = np.zeros(len(v))
    retreat = np.ones(len(v))
    for i in range(L, len(v) - L):
        if z_score[i] > C_jump:
            continue  # 跳过瞬时跳变点
        mu1 = np.median(v[i - L + 1 : i + 1])
        mu2 = np.median(v[i + 1 : i + L + 1])
        if mu1 < 1e-6:
            continue
        R[i] = (mu2 - mu1) / mu1
        # 持久性检验
        if mu2 > mu1:
            future_win = v[i + H // 2 : i + H + 1]
            mu_future = np.median(future_win) if len(future_win) > 0 else mu2
            retreat[i] = (mu2 - mu_future) / (mu2 - mu1 + 1e-6)

    S = R * (R > 0) * (retreat <= theta_back)
    candidate_nodes = np.where(S > C_trans)[0]

    # 根据绝对速度水平区分两个节点
    nodes = {"slow2acc": None, "acc2fast": None}
    # 已平滑速度用于速度判断（已在 v_smooth 中，去掉 NaN）
    v_for_judge = v_smooth[~np.isnan(v_smooth)]
    for i in candidate_nodes:
        if i < L or i > len(v) - L:
            continue
        v_before = np.median(v[i - L + 1 : i + 1])
        v_after = np.median(v[i + 1 : i + L + 1])
        if nodes["slow2acc"] is None and v_before <= speed_low and v_after <= speed_high:
            nodes["slow2acc"] = i
        elif nodes["acc2fast"] is None and v_after > speed_high:
            nodes["acc2fast"] = i
            break

    node1 = nodes["slow2acc"]
    node2 = nodes["acc2fast"]

    # 转换到位移点序号（1-index）
    idx1 = node1 + 1 if node1 is not None else None
    idx2 = node2 + 1 if node2 is not None else None
    t1 = (idx1 - 1) * dt if idx1 else None
    t2 = (idx2 - 1) * dt if idx2 else None

    _print_nodes(idx1, t1, idx2, t2)
    return idx1, t1, idx2, t2


def _print_nodes(idx1, t1, idx2, t2):
    """打印节点识别结果"""
    print("\n阶段转换节点识别结果")
    if idx1 is not None:
        print(f"  节点1 (匀速→加速): 序号 {idx1}, 时间 {t1/24:.2f} 天 ({t1:.1f} h)")
    else:
        print("  节点1 (匀速→加速): 未识别")
    if idx2 is not None:
        print(f"  节点2 (加速→快速): 序号 {idx2}, 时间 {t2/24:.2f} 天 ({t2:.1f} h)")
    else:
        print("  节点2 (加速→快速): 未识别")
    if idx1 is None or idx2 is None:
        print("  ⚠ 警告：未能识别出两个有效转换节点！")
    print()


def plot_velocity_nodes(d, t, dt, idx1, t1, idx2, t2, save_dir):
    """图5：速度曲线 + 节点标记 + 阈值线 + 三阶段色块背景"""
    # 从 xlsx 直接读取已平滑速度
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, config.CLEAN_DATA)
    df = pd.read_excel(path, sheet_name=0)
    v_smooth = df["Smoothed Velocity (mm/h)"].values
    v_smooth = v_smooth[~np.isnan(v_smooth)]

    plt.figure(figsize=(12, 5))
    ax = plt.gca()

    # 三阶段色块背景
    t_end = t[-1] / 24
    t1_day = t1 / 24
    t2_day = t2 / 24
    ax.axvspan(0, t1_day, alpha=0.06, color='blue', label='阶段 I (匀速)')
    ax.axvspan(t1_day, t2_day, alpha=0.06, color='green', label='阶段 II (加速)')
    ax.axvspan(t2_day, t_end, alpha=0.06, color='red', label='阶段 III (快速)')

    plt.plot(t[:-1] / 24, v_smooth, 'r-', linewidth=1.5,
             label='速度平滑曲线 (Savgol)')

    # 阈值线：不同线型
    plt.axhline(config.THRESH_SLOW, color='blue', linestyle='--', alpha=0.7,
                linewidth=1.2, label=f'慢速阈值 = {config.THRESH_SLOW} mm/h')
    plt.axhline(config.THRESH_FAST, color='green', linestyle='-.', alpha=0.7,
                linewidth=1.2, label=f'快速阈值 = {config.THRESH_FAST} mm/h')

    # 节点竖线 + 顶部标记 + 交叉圆圈
    if idx1:
        plt.axvline(x=t1_day, color='blue', linestyle=':', linewidth=2)
        # 顶部箭头标注
        plt.annotate(f'节点1\n{t1_day:.1f} 天', xy=(t1_day, ax.get_ylim()[1]),
                     xytext=(t1_day, ax.get_ylim()[1] * 1.05),
                     ha='center', fontsize=10, color='blue', fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
        # 曲线交叉处圆圈
        v_at_node = v_smooth[int(np.round(idx1 - 1))]
        plt.scatter(t1_day, v_at_node, s=80, facecolors='none',
                    edgecolors='blue', linewidth=2, zorder=5)
    if idx2:
        plt.axvline(x=t2_day, color='red', linestyle=':', linewidth=2)
        plt.annotate(f'节点2\n{t2_day:.1f} 天', xy=(t2_day, ax.get_ylim()[1]),
                     xytext=(t2_day, ax.get_ylim()[1] * 1.05),
                     ha='center', fontsize=10, color='red', fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
        v_at_node = v_smooth[int(np.round(idx2 - 1))]
        plt.scatter(t2_day, v_at_node, s=80, facecolors='none',
                    edgecolors='red', linewidth=2, zorder=5)

    plt.xlabel('时间 (天)', fontsize=12)
    plt.ylabel('速度 (mm/h)', fontsize=12)
    plt.title('速度趋势与阶段转换节点', fontsize=14)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, '图5 速度过渡与节点.png')
    plt.savefig(path, dpi=300)
    plt.show()
    print(f"[图5] 速度过渡曲线 → {path}")


if __name__ == "__main__":
    print("=" * 50)
    print("阶段转换节点检测")
    print("=" * 50)

    # 用默认的滑动窗口法检测并画图
    d, t, dt, _ = _load_clean_data()
    idx1, t1, idx2, t2 = sliding_window_detect()
    if idx1 is None or idx2 is None:
        print("\n>>> 滑动窗口法失败，回退到MAD+持久性法")
        idx1, t1, idx2, t2 = mad_persistence_detect()

    if idx1 is not None and idx2 is not None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        plot_velocity_nodes(d, t, dt, idx1, t1, idx2, t2, script_dir)
        plt.show(block=True)
    else:
        print("\n⚠ 两种检测方法均失败，无法画图5")
