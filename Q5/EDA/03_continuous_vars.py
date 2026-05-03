"""03 — 连续传感器变量：孔隙水压力、干湿入渗系数
图1: 原始序列与多窗口滑动平均对比
图2: 分布图（小提琴+KDE）
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common.plot_utils import setup_zh
from common.data_utils import load_pipeline
from common.eda_utils import plot_rolling_mean, save_and_show, OUT_DIR, ensure_out_dir, classify_vars

def main():
    setup_zh()
    out_dir = ensure_out_dir()

    # 加载数据
    df, base_vars, (b1, b2) = load_pipeline()
    print(f"数据形状: {df.shape}")

    # 获取连续变量
    var_dict = classify_vars(list(df.columns), base_vars)
    continuous = var_dict['continuous']
    print(f"连续传感器变量: {continuous}")

    # 确保有 Time 列
    timestamps = df['Time'] if 'Time' in df.columns else df.index

    # 滑动窗口（步数）：3h=18, 6h=36, 12h=72, 24h=144
    windows = [18, 36, 72, 144]
    window_labels = ['3h(18步)', '6h(36步)', '12h(72步)', '24h(144步)']
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']

    for var in continuous:
        if var not in df.columns:
            print(f"  跳过 {var}: 不存在")
            continue
        series = df[var].values

        # ===== 图1: 滑动平均对比 =====
        fig, ax = plt.subplots(figsize=(14, 5))
        plot_rolling_mean(ax, timestamps, series, windows, colors, window_labels)
        ax.set_title(f'{var} — 原始序列与多窗口滑动平均', fontsize=13)
        ax.set_ylabel(var)
        fig.tight_layout()
        fname = f'{var}_rolling_mean.png'
        save_and_show(fig, os.path.join(out_dir, fname))

        # ===== 图2: 分布图（直方图+KDE） =====
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 直方图 + KDE
        sns.histplot(series, kde=True, ax=axes[0], color='steelblue', bins=50)
        axes[0].set_title(f'{var} — 分布直方图+KDE')
        axes[0].set_xlabel(var)

        # 小提琴图
        # 创建临时 DataFrame 给 seaborn
        tmp_df = pd.DataFrame({var: series})
        sns.violinplot(data=tmp_df, y=var, ax=axes[1], color='lightcoral')
        axes[1].set_title(f'{var} — 小提琴图')

        fig.tight_layout()
        fname2 = f'{var}_distribution.png'
        save_and_show(fig, os.path.join(out_dir, fname2))

    print("\n✅ 连续传感器变量可视化完成")

if __name__ == '__main__':
    main()
