"""
结果可视化 —— 生成三张主图
对标 Q1 correct_and_test.m 的 §5 可视化部分
输入：清洗后数据 + 节点 + 各阶段模型
输出：3 张 PNG 图（stage_fitting_curve, stage_residuals, velocity_transition）
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def _load_data_and_nodes():
    """读取清洗后数据并检测节点"""
    from detect_nodes import sliding_window_detect, mad_persistence_detect

    idx1, t1, idx2, t2 = sliding_window_detect()
    if idx1 is None or idx2 is None:
        idx1, t1, idx2, t2 = mad_persistence_detect()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, config.CLEAN_DATA)
    import pandas as pd
    df = pd.read_excel(path, sheet_name=0)
    d = df["Surface Displacement (mm)"].values
    dt = config.DT
    t = np.arange(len(d)) * dt

    return d, t, dt, idx1, t1, idx2, t2


def plot_fitting_curve(stages, t_full, d_full, idx1, t1, idx2, t2, save_dir):
    """图1：原始位移 + 分段拟合曲线"""
    plt.figure(figsize=(12, 6))
    plt.plot(t_full / 24, d_full, 'k-', linewidth=0.6, alpha=0.7,
             label='Filtered displacement')

    colors = {'I': 'b', 'II': 'g', 'III': 'r'}
    for label in ['I', 'II', 'III']:
        s = stages[label]
        plt.plot(s['t'] / 24, s['d_pred'], f'{colors[label]}--',
                 linewidth=2, label=f'Stage {label} ({s["model"]} fit)')

    # 标记节点
    if idx1:
        plt.axvline(x=t1 / 24, color='blue', linestyle=':', linewidth=1.5,
                    label=f'Node1 t={t1/24:.1f}d')
    if idx2:
        plt.axvline(x=t2 / 24, color='red', linestyle=':', linewidth=1.5,
                    label=f'Node2 t={t2/24:.1f}d')

    plt.xlabel('Time (days)', fontsize=12)
    plt.ylabel('Displacement (mm)', fontsize=12)
    plt.title('Three-stage Modeling of Surface Displacement', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, '图6 三段拟合曲线.png')
    plt.savefig(path, dpi=300)
    plt.show()
    print(f"[图6] 三段拟合曲线 → {path}")


def plot_residuals(stages, save_dir):
    """图2：各阶段残差图"""
    plt.figure(figsize=(12, 10))
    colors = {'I': 'b', 'II': 'g', 'III': 'r'}
    labels = {'I': 'Slow Constant', 'II': 'Accelerating', 'III': 'Rapid'}

    for i, label in enumerate(['I', 'II', 'III'], 1):
        plt.subplot(3, 1, i)
        s = stages[label]
        res = s['d'] - s['d_pred']
        plt.plot(s['t'] / 24, res, f'{colors[label]}.', markersize=2, alpha=0.5)
        plt.axhline(0, color='k', linestyle='-')
        plt.ylabel('Residual (mm)', fontsize=10)
        plt.title(f'Stage {label} ({labels[label]}) Residuals '
                  f'(R^2={s["r2"]:.4f}, RMSE={s["rmse"]:.3f}mm)')
        plt.grid(True, alpha=0.3)

    plt.xlabel('Time (days)', fontsize=12)
    plt.tight_layout()
    path = os.path.join(save_dir, '图7 各阶段残差.png')
    plt.savefig(path, dpi=300)
    plt.show()
    print(f"[图7] 各阶段残差 → {path}")


def plot_velocity_transition(d, t, dt, idx1, t1, idx2, t2, save_dir):
    """图3：速度曲线 + 转换节点 + 阈值线"""
    v = np.diff(d) / dt
    v_smooth = savgol_filter(v, window_length=config.SAVGOL_WINDOW,
                              polyorder=config.SAVGOL_POLY)

    plt.figure(figsize=(12, 5))
    plt.plot(t[:-1] / 24, v_smooth, 'r-', linewidth=1.2,
             label='Smoothed velocity (Savgol)')

    # 阈值线（滑动窗口法参数）
    plt.axhline(config.THRESH_SLOW, color='b', linestyle='--', alpha=0.7,
                label=f'Slow threshold = {config.THRESH_SLOW} mm/h')
    plt.axhline(config.THRESH_FAST, color='g', linestyle='--', alpha=0.7,
                label=f'Fast threshold = {config.THRESH_FAST} mm/h')

    # 节点
    if idx1:
        plt.axvline(x=t1 / 24, color='blue', linestyle=':', linewidth=1.5)
    if idx2:
        plt.axvline(x=t2 / 24, color='red', linestyle=':', linewidth=1.5)

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


def plot_all(save_dir=None):
    """一键生成全部3张图"""
    if save_dir is None:
        save_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 55)
    print("生成结果可视化图表")
    print("=" * 55)

    # 读取数据 + 检测节点
    d, t, dt, idx1, t1, idx2, t2 = _load_data_and_nodes()

    if idx1 is None or idx2 is None:
        print("✗ 无法生成图表：节点识别失败")
        return

    # 建模
    from modeling import modeling_with_nodes
    stages = modeling_with_nodes(idx1, idx2)

    # 生成三张图
    plot_fitting_curve(stages, t, d, idx1, t1, idx2, t2, save_dir)
    plot_residuals(stages, save_dir)
    plot_velocity_transition(d, t, dt, idx1, t1, idx2, t2, save_dir)

    print(f"\n所有图形已保存至: {save_dir}")
    print("=" * 55)


if __name__ == "__main__":
    plot_all()
