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
from scipy.signal import savgol_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


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


if __name__ == "__main__":
    print("=" * 50)
    print("阶段转换节点检测")
    print("=" * 50)

    print("\n>>> 方法1：滑动窗口法")
    sliding_window_detect()

    print("\n>>> 方法2：MAD+持久性法")
    mad_persistence_detect()
