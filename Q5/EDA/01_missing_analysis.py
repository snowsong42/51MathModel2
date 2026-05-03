"""01 — 缺失值分析与缺失矩阵图"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import missingno as msno

# 添加上级目录到 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common.plot_utils import setup_zh
from common.eda_utils import save_and_show, OUT_DIR, ensure_out_dir

def main():
    setup_zh()
    out_dir = ensure_out_dir()

    # 加载原始数据（未清洗前的原始 xlsx）
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             'common', 'Attachment 5.xlsx')
    df_raw = pd.read_excel(data_path)
    print(f"原始数据形状: {df_raw.shape}")
    print(f"原始列名: {list(df_raw.columns)}")

    # 映射列名以匹配 data_utils 的命名
    rules = [
        ('时间', 'Time'), ('表面位移', 'Displacement'), ('降雨', 'Rainfall'),
        ('孔隙', 'PorePressure'), ('微震', 'Microseismic'), ('入渗', 'Infiltration'),
        ('爆破', 'BlastDist'), ('距离', 'BlastDist'), ('单段', 'BlastCharge'),
        ('药量', 'BlastCharge'), ('最大', 'BlastCharge'),
        ('Time', 'Time'), ('Surface Displacement', 'Displacement'),
        ('Rainfall', 'Rainfall'), ('Pore Water Pressure', 'PorePressure'),
        ('Microseismic', 'Microseismic'), ('Dry-Wet Infiltration', 'Infiltration'),
        ('Blasting Point Distance', 'BlastDist'),
        ('Maximum Charge per Segment', 'BlastCharge'),
    ]
    col_map = {}
    for c in df_raw.columns:
        s = str(c)
        for kw, en in rules:
            if kw in s:
                col_map[c] = en
                break
    df = df_raw.rename(columns=col_map)

    # 选择关键数值列做缺失分析
    key_cols = [c for c in ['PorePressure', 'Infiltration', 'Rainfall',
                            'Microseismic', 'BlastDist', 'BlastCharge',
                            'Displacement'] if c in df.columns]
    df_sub = df[key_cols]

    # 1) 缺失比例表
    missing_ratio = df_sub.isna().mean().to_frame('缺失比例')
    missing_ratio['非空数量'] = df_sub.count()
    missing_ratio['总数'] = len(df_sub)
    missing_ratio.to_csv(os.path.join(out_dir, 'missing_ratio.csv'), encoding='utf-8-sig')
    print("\n=== 缺失比例 ===")
    print(missing_ratio)

    # 2) 缺失矩阵图（使用 missingno）
    fig, ax = plt.subplots(figsize=(12, 6))
    msno.matrix(df_sub, ax=ax, fontsize=10, sparkline=False)
    ax.set_title('缺失值矩阵图 (黑色=有值, 白色=缺失)', fontsize=14)
    fig.tight_layout()
    save_and_show(fig, os.path.join(out_dir, 'missing_pattern.png'))

    # 3) 爆破时刻非空检查
    if 'BlastDist' in df.columns and 'BlastCharge' in df.columns:
        blast_nonnull = df_sub.dropna(subset=['BlastDist', 'BlastCharge'])
        print(f"\n爆破时刻(BlastDist非空)记录数: {blast_nonnull.shape[0]}")
        print(f"爆破时刻其他变量的缺失情况:")
        print(blast_nonnull.isna().sum())

    print("\n✅ 缺失分析完成")

if __name__ == '__main__':
    main()
