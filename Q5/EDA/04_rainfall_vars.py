"""04 — 降雨相关变量
图3: 降雨量多窗口累积与有效降雨
图4: 降雨与位移的互相关函数 (CCF)
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common.plot_utils import setup_zh
from common.data_utils import load_pipeline
from common.eda_utils import ccf_compute, effective_rainfall, save_and_show, OUT_DIR, ensure_out_dir

def main():
    setup_zh()
    out_dir = ensure_out_dir()

    # 加载数据
    df, base_vars, (b1, b2) = load_pipeline()
    print(f"数据形状: {df.shape}")

    timestamps = df['Time'] if 'Time' in df.columns else df.index
    rain = df['Rainfall'].values if 'Rainfall' in df.columns else None
    delta_d = df['Delta_D'].values if 'Delta_D' in df.columns else None

    if rain is None:
        print("⚠ 未找到 Rainfall 列，跳过")
        return

    # ===== 图3: 降雨量多窗口累积与有效降雨 =====
    fig, ax = plt.subplots(figsize=(14, 6))

    # 原降雨柱状图
    ax.bar(timestamps, rain, width=0.001, alpha=0.3, color='royalblue', label='原始降雨量', zorder=2)

    # 滑动求和窗口：3h(18步), 6h(36步), 12h(72步), 24h(144步)
    cum_windows = [18, 36, 72, 144]
    cum_labels = ['3h累积', '6h累积', '12h累积', '24h累积']
    colors = ['#e41a1c', '#ff7f00', '#4daf4a', '#984ea3']
    for w, lab, col in zip(cum_windows, cum_labels, colors):
        cum = pd.Series(rain).rolling(w, min_periods=1).sum()
        ax.plot(timestamps, cum, color=col, linewidth=1.0, label=lab, alpha=0.8)

    # 有效降雨曲线
    eff_rain = effective_rainfall(rain, decay=0.85)
    ax.plot(timestamps, eff_rain, color='black', linewidth=1.5, linestyle='--', label='有效降雨 (衰减0.85)', alpha=0.9)

    ax.set_title('降雨量多窗口累积与有效降雨', fontsize=14)
    ax.set_ylabel('降雨量 / 累积量')
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_and_show(fig, os.path.join(out_dir, 'rainfall_cumulative.png'))

    # ===== 图4: 降雨与位移的互相关函数（CCF） =====
    if delta_d is not None:
        fig, ax = plt.subplots(figsize=(12, 5))
        max_lag = 72  # ±12h
        lags, corrs = ccf_compute(rain, delta_d, max_lag)

        ax.stem(lags, corrs, basefmt=' ', linefmt='steelblue', markerfmt='o')
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')

        # 标注最大正相关对应的滞后
        valid_mask = ~np.isnan(corrs)
        if valid_mask.any():
            max_idx = np.argmax(corrs[valid_mask])
            max_lag_val = lags[valid_mask][max_idx]
            max_corr = corrs[valid_mask][max_idx]
            ax.annotate(f'最大正相关: lag={max_lag_val}, r={max_corr:.3f}',
                        xy=(max_lag_val, max_corr),
                        xytext=(max_lag_val+10, max_corr+0.05),
                        arrowprops=dict(arrowstyle='->', color='red'),
                        fontsize=11, color='red', fontweight='bold')

        ax.set_title('降雨与位移增量 (Delta_D) 的互相关函数 (CCF)', fontsize=13)
        ax.set_xlabel('滞后步数 (每步10分钟)')
        ax.set_ylabel('相关系数')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        save_and_show(fig, os.path.join(out_dir, 'ccf_rainfall_displacement.png'))
    else:
        print("⚠ 未找到 Delta_D 列，跳过 CCF")

    print("\n✅ 降雨变量可视化完成")

if __name__ == '__main__':
    main()
