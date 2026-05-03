"""02 — 变量分类与统计表生成"""

import os, sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common.plot_utils import setup_zh
from common.data_utils import load_pipeline
from common.eda_utils import classify_vars, compute_variable_stats, OUT_DIR, ensure_out_dir, save_and_show

def main():
    setup_zh()
    out_dir = ensure_out_dir()

    # 加载数据
    print("加载数据中...")
    df, base_vars, (b1, b2) = load_pipeline()
    print(f"数据形状: {df.shape}")
    print(f"基础变量: {base_vars}")
    print(f"阶段边界: b1={b1}, b2={b2}")
    print(f"列名列表: {list(df.columns)}")

    # 变量分类
    var_dict = classify_vars(list(df.columns), base_vars)
    print("\n=== 变量分类 ===")
    for k, v in var_dict.items():
        print(f"  {k}: {v}")

    # 保存分类结果
    with open(os.path.join(out_dir, 'variable_classes.txt'), 'w', encoding='utf-8') as f:
        for k, v in var_dict.items():
            f.write(f"{k}: {', '.join(v)}\n")

    # 统计表：对所有数值变量
    all_numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
                   and c not in ['Time', 'Hour', 'Phase']]
    stats_df = compute_variable_stats(df, all_numeric)
    stats_df.to_csv(os.path.join(out_dir, 'variable_statistics.csv'), encoding='utf-8-sig')
    print("\n=== 统计汇总（前20行） ===")
    print(stats_df.head(20))
    print(f"\n统计表已保存，共 {len(stats_df)} 个变量")

    # 保存阶段边界供其他脚本使用
    np.savez(os.path.join(out_dir, 'phase_boundaries.npz'), b1=b1, b2=b2)
    print(f"阶段边界 b1={b1}, b2={b2} 已保存")

    print("\n✅ 变量分类与统计完成")

if __name__ == '__main__':
    main()
