"""05 — 微震事件数
图5: 事件间隔与密度（事件间隔直方图 + 滑动时间窗事件计数）
图6: 累积事件数与位移双 y 轴时序图
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common.plot_utils import setup_zh
from common.data_utils import load_pipeline
from common.eda_utils import save_and_show, OUT_DIR, ensure_out_dir

def main():
    setup_zh()
    out_dir = ensure_out_dir()

    # 加载数据
    df, base_vars, (b1, b2) = load_pipeline()
    print(f"数据形状: {df.shape}")

    timestamps = df['Time'] if 'Time' in df.columns else df.index
    micro = df['Microseismic'].values if 'Microseismic' in df.columns else None
    delta_d = df['Delta_D'].values if 'Delta_D' in df.columns else None
    disp = df['Displacement'].values if 'Displacement' in df.columns else delta_d

    if micro is None:
        print("⚠ 未找到 Microseismic 列，跳过")
        return

    # 微震事件时刻（微震计数 > 0）
    event_idx = np.where(micro > 0)[0]
    print(f"微震事件总数: {len(event_idx)}")

    # ===== 图5: 事件间隔与密度 =====
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 左图: 事件间隔直方图
    if len(event_idx) > 1:
        intervals = np.diff(event_idx)
        axes[0].hist(intervals, bins=80, color='steelblue', edgecolor='white', alpha=0.8)
        axes[0].axvline(np.median(intervals), color='red', linestyle='--',
                        label=f'中位数={np.median(intervals):.0f}步')
        axes[0].set_xlabel('事件间隔 (步数)')
        axes[0].set_ylabel('频次')
        axes[0].set_title('微震事件间隔分布', fontsize=13)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, '事件不足', ha='center', va='center')

    # 右图: 滑动时间窗事件计数
    windows = [36, 72, 144]  # 6h, 12h, 24h
    win_labels = ['6h (36步)', '12h (72步)', '24h (144步)']
    colors = ['#e41a1c', '#4daf4a', '#377eb8']
    for w, lab, col in zip(windows, win_labels, colors):
        count = pd.Series(micro).rolling(w, min_periods=1).sum()
        axes[1].plot(timestamps, count, color=col, linewidth=1.0, label=lab, alpha=0.8)
    axes[1].set_title('滑动时间窗微震事件计数', fontsize=13)
    axes[1].set_xlabel('时间')
    axes[1].set_ylabel('事件计数')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    save_and_show(fig, os.path.join(out_dir, 'microseismic_event_analysis.png'))

    # ===== 图6: 累积事件数与位移双 y 轴时序图 =====
    if disp is not None:
        fig, ax1 = plt.subplots(figsize=(14, 6))
        ax2 = ax1.twinx()

        # 累积微震事件数（左 y 轴）
        cum_events = np.cumsum(micro)
        ax1.plot(timestamps, cum_events, color='#e41a1c', linewidth=1.5, label='累积微震事件数')
        ax1.set_ylabel('累积微震事件数', color='#e41a1c', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='#e41a1c')

        # 位移（右 y 轴）
        ax2.plot(timestamps, disp, color='#377eb8', linewidth=1.0, label='位移', alpha=0.8)
        ax2.set_ylabel('位移 / 位移增量', color='#377eb8', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='#377eb8')

        # 阶段分界线
        ax1.axvline(timestamps.iloc[b1] if hasattr(timestamps, 'iloc') else timestamps[b1],
                    color='gray', linestyle='--', alpha=0.5, label=f'阶段1/{b1}')
        ax1.axvline(timestamps.iloc[b2] if hasattr(timestamps, 'iloc') else timestamps[b2],
                    color='gray', linestyle='--', alpha=0.5, label=f'阶段2/{b2}')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

        ax1.set_title('累积微震事件数与位移时序对比', fontsize=14)
        ax1.set_xlabel('时间')
        ax1.grid(True, alpha=0.3)
        fig.tight_layout()
        save_and_show(fig, os.path.join(out_dir, 'cum_microseismic_vs_displacement.png'))

    print("\n✅ 微震变量可视化完成")

if __name__ == '__main__':
    main()
