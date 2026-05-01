"""
阶段转换节点检测
提供两种算法：
  1) sliding_window_detect()  — 简化滑动窗口方案（原 velocity.py）
  2) mad_persistence_detect() — MAD+阶跃+持久性方案（原 stage_recognition.py）
输入：清洗后数据 (Filtered 2.xlsx)
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
from scipy.signal import savgol_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# ===== matplotlib 显示设置 =====
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']  # 中文字体 + 英文字体fallback
plt.rcParams['axes.unicode_minus'] = False           # 修复负号显示为方块

def _load_clean_data():
    """读取清洗后数据，返回位移数组和时间数组"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, config.CLEAN_DATA)
    df = pd.read_excel(path, sheet_name=0)
    d = df["Surface Displacement (mm)"].values
    dt = config.DT
    t = np.arange(len(d)) * dt
    return d, t, dt


def sliding_window_detect():
    """
    简化滑动窗口方案（原 velocity.py）
    基于速度中位数窗口 + 双阈值 + 持续性检验
    """
    print("-" * 50)
    print("检测方法: 滑动窗口法（简化方案）")
    print("-" * 50)

    d, _, dt = _load_clean_data()
    N = len(d)

    # 速度
    v = np.diff(d) / dt

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
    """
    print("-" * 50)
    print("检测方法: MAD+持久性法（完整文档算法）")
    print("-" * 50)

    d, _, dt = _load_clean_data()
    N = len(d)

    # 速度
    v = np.diff(d) / dt

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
    v_smooth = savgol_filter(v, window_length=config.SAVGOL_WINDOW,
                              polyorder=config.SAVGOL_POLY)
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
    """图5：速度曲线 + 节点标记 + 阈值线"""
    v = np.diff(d) / dt
    v_smooth = savgol_filter(v, window_length=config.SAVGOL_WINDOW,
                              polyorder=config.SAVGOL_POLY)

    plt.figure(figsize=(12, 5))
    plt.plot(t[:-1] / 24, v_smooth, 'r-', linewidth=1.2,
             label='Smoothed velocity (Savgol)')

    # 阈值线
    plt.axhline(config.THRESH_SLOW, color='b', linestyle='--', alpha=0.7,
                label=f'Slow threshold = {config.THRESH_SLOW} mm/h')
    plt.axhline(config.THRESH_FAST, color='g', linestyle='--', alpha=0.7,
                label=f'Fast threshold = {config.THRESH_FAST} mm/h')

    # 节点标记
    if idx1:
        plt.axvline(x=t1 / 24, color='blue', linestyle=':', linewidth=1.5,
                    label=f'Node1 (t={t1/24:.1f}d)')
    if idx2:
        plt.axvline(x=t2 / 24, color='red', linestyle=':', linewidth=1.5,
                    label=f'Node2 (t={t2/24:.1f}d)')

    plt.xlabel('Time (days)', fontsize=12)
    plt.ylabel('Velocity (mm/h)', fontsize=12)
    plt.title('Velocity Trend and Stage Transition Nodes', fontsize=14)
    plt.legend()
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
    d, t, dt = _load_clean_data()
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
