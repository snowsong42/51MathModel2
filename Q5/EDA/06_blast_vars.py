"""06 — 爆破相关变量
图7: 爆破事件散点图（BlastDist vs BlastCharge，按时间着色）
图8: 爆破前后位移响应叠加图（±24h，灰色叠加+红色均值）
图9: 爆破能量衰减特征示意（指数衰减示例）
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
from common.eda_utils import blast_response_extract, save_and_show, OUT_DIR, ensure_out_dir

def main():
    setup_zh()
    out_dir = ensure_out_dir()

    # 加载数据
    df, base_vars, (b1, b2) = load_pipeline()
    print(f"数据形状: {df.shape}")

    timestamps = df['Time'] if 'Time' in df.columns else df.index
    blast_dist = df['BlastDist'].values if 'BlastDist' in df.columns else None
    blast_charge = df['BlastCharge'].values if 'BlastCharge' in df.columns else None

    if blast_dist is None:
        print("⚠ 未找到 BlastDist 列，跳过")
        return

    # 爆破事件位置
    blast_mask = blast_dist > 0
    blast_idx = np.where(blast_mask)[0]
    print(f"爆破事件总数: {len(blast_idx)}")

    # ===== 图7: 爆破事件散点图 =====
    fig, ax = plt.subplots(figsize=(10, 8))
    # 按时间顺序着色（彩虹色）
    n = len(blast_idx)
    if n > 1:
        colors = plt.cm.rainbow(np.linspace(0, 1, n))
    else:
        colors = ['red']

    for i, idx in enumerate(blast_idx):
        d = blast_dist[idx]
        c = blast_charge[idx] if blast_charge is not None else 0
        ax.scatter(d, c, color=colors[i], s=80, alpha=0.8, edgecolors='black', linewidth=0.5,
                   label=f'Event {i+1}' if n <= 20 else None)

    ax.set_xlabel('爆破距离 (BlastDist)', fontsize=12)
    ax.set_ylabel('单段药量 (BlastCharge)', fontsize=12)
    ax.set_title('爆破事件散点图 (按时间彩虹色着色)', fontsize=14)
    ax.grid(True, alpha=0.3)
    if n <= 20:
        ax.legend(fontsize=7, ncol=2, loc='upper right')
    fig.tight_layout()
    save_and_show(fig, os.path.join(out_dir, 'blast_scatter.png'))

    # ===== 图8: 爆破前后位移响应叠加图 =====
    if 'Delta_D' in df.columns:
        fig, ax = plt.subplots(figsize=(14, 6))
        # 提取每个爆破事件前后 ±144步 (24h) 的 Delta_D
        sequences = blast_response_extract(df, blast_mask, before=144, after=144)
        print(f"提取爆破响应序列数: {len(sequences)}")

        if len(sequences) > 0:
            # 补齐到统一长度
            max_len = max(len(s) for s in sequences)
            aligned = []
            for s in sequences:
                if len(s) < max_len:
                    s = np.pad(s, (0, max_len - len(s)), mode='constant', constant_values=np.nan)
                aligned.append(s)
            aligned = np.array(aligned)

            x_axis = np.arange(-144, max_len - 144)
            # 灰色半透明叠加
            for seq in aligned:
                ax.plot(x_axis[:len(seq)], seq, color='gray', alpha=0.15, linewidth=0.8)
            # 红色粗均值线
            mean_seq = np.nanmean(aligned, axis=0)
            ax.plot(x_axis[:len(mean_seq)], mean_seq, color='red', linewidth=2.5, label='平均响应')

            # 标注爆破时刻
            ax.axvline(0, color='black', linestyle='--', alpha=0.5, label='爆破时刻')
            ax.set_xlabel('距爆破时刻的步数 (每步10分钟)', fontsize=12)
            ax.set_ylabel('位移增量 Delta_D', fontsize=12)
            ax.set_title('爆破前后位移响应叠加图 (±24h)', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, '无可用爆破响应序列', ha='center', va='center', fontsize=14)

        fig.tight_layout()
        save_and_show(fig, os.path.join(out_dir, 'blast_response_curves.png'))

    # ===== 图9: 爆破能量衰减特征示意 =====
    fig, ax = plt.subplots(figsize=(10, 5))
    # 用前几次爆破事件示意
    n_events = min(5, len(blast_idx))
    if n_events > 0:
        tau = 50  # 衰减时间常数（步数）
        t = np.arange(0, 200)  # 200步衰减曲线
        colors_demo = plt.cm.Set1(np.linspace(0, 1, n_events))

        for i in range(n_events):
            idx = blast_idx[i]
            # 构造简化爆破能量: q/d^2
            d = max(blast_dist[idx], 1)
            q = blast_charge[idx] if blast_charge is not None else 1
            energy = q / (d ** 2)
            decay = energy * np.exp(-t / tau)
            ax.plot(t, decay, color=colors_demo[i], linewidth=1.5,
                    label=f'Event {i+1}: q={q:.1f}, d={d:.1f}')

        ax.set_xlabel('爆破后步数 (每步10分钟)', fontsize=12)
        ax.set_ylabel('衰减能量 (q/d² · exp(-t/τ))', fontsize=12)
        ax.set_title(f'爆破能量衰减特征示意 (τ={tau}步)', fontsize=14)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, '无爆破事件可用于示例', ha='center', va='center', fontsize=14)

    fig.tight_layout()
    save_and_show(fig, os.path.join(out_dir, 'blast_decay_demo.png'))

    print("\n✅ 爆破变量可视化完成")

if __name__ == '__main__':
    main()
