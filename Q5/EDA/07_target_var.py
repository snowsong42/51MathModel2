"""07 — 目标变量（表面位移/Delta_D）
图10: 位移速度时序与阶段划分（一阶差分/速度 + 阶段底色标记 + 水平参考线）
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common.plot_utils import setup_zh, phase_name
from common.data_utils import load_pipeline
from common.eda_utils import save_and_show, OUT_DIR, ensure_out_dir

def main():
    setup_zh()
    out_dir = ensure_out_dir()

    # 加载数据
    df, base_vars, (b1, b2) = load_pipeline()
    print(f"数据形状: {df.shape}, b1={b1}, b2={b2}")

    timestamps = df['Time'] if 'Time' in df.columns else df.index
    disp = df['Displacement'].values if 'Displacement' in df.columns else None
    delta_d = df['Delta_D'].values if 'Delta_D' in df.columns else None

    if disp is None and delta_d is None:
        print("⚠ 未找到位移相关列，跳过")
        return

    # 计算速度（一阶差分）
    if disp is not None:
        velocity = np.diff(disp, prepend=disp[0])
    else:
        velocity = delta_d

    # 计算各阶段速度均值
    n = len(df)
    phase_labels = np.zeros(n, dtype=int)
    phase_labels[:b1] = 0
    phase_labels[b1:b2] = 1
    phase_labels[b2:] = 2

    # ===== 图10: 位移速度时序与阶段划分 =====
    fig, ax = plt.subplots(figsize=(16, 7))

    # 阶段底色
    colors_phase = {0: '#d6e8f0', 1: '#f0d6d6', 2: '#f0e0d6'}
    phase_names = {0: '缓慢变形', 1: '加速变形', 2: '快速变形'}
    for phase_id in [0, 1, 2]:
        mask = phase_labels == phase_id
        if mask.any():
            idx = np.where(mask)[0]
            ax.axvspan(timestamps.iloc[idx[0]] if hasattr(timestamps, 'iloc') else timestamps[idx[0]],
                       timestamps.iloc[idx[-1]] if hasattr(timestamps, 'iloc') else timestamps[idx[-1]],
                       alpha=0.2, color=colors_phase[phase_id],
                       label=f'阶段{phase_id+1}: {phase_names[phase_id]}')

    # 速度曲线
    ax.plot(timestamps, velocity, color='#2c3e50', linewidth=0.8, alpha=0.7, label='位移速度 (一阶差分)')

    # 各阶段速度均值水平线
    mean_colors = {0: '#4a8db7', 1: '#c0392b', 2: '#d35400'}
    for phase_id in [0, 1, 2]:
        mask = phase_labels == phase_id
        if mask.any():
            mean_v = np.mean(velocity[mask])
            ax.axhline(mean_v, color=mean_colors[phase_id],
                       linewidth=2, linestyle='--', alpha=0.8,
                       label=f'{phase_names[phase_id]} 均值={mean_v:.4f}')

    # 阶段分界线
    for boundary, label in [(b1, f'阶段1→2 (b1={b1})'), (b2, f'阶段2→3 (b2={b2})')]:
        if boundary < n:
            ax.axvline(timestamps.iloc[boundary] if hasattr(timestamps, 'iloc') else timestamps[boundary],
                       color='black', linestyle=':', alpha=0.6, linewidth=1.2)

    ax.set_title('位移速度时序与阶段划分', fontsize=15)
    ax.set_xlabel('时间')
    ax.set_ylabel('位移速度 (每步位移增量)')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_and_show(fig, os.path.join(out_dir, 'displacement_velocity_phases.png'))

    print("\n✅ 目标变量可视化完成")

if __name__ == '__main__':
    main()
