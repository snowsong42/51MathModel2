"""
Q5 特征工程脚本
在 load_pipeline() 返回的已有 36 个基础特征之上，追加高级特征。
输出: Q5/feature/feature_{n}.xlsx

用法: python Q5/feature/feature.py
"""

import pandas as pd
import numpy as np
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common.data_utils import load_pipeline

# ============================================================
# 辅助函数
# ============================================================

def exp_decay_series(event_times, values, tau, total_len):
    """
    指数衰减序列生成器
    event_times: list[int] 事件发生的时间索引
    values:      list[float] 每个事件的权重值
    tau:         float 衰减时间常数
    total_len:   int 总序列长度
    returns:     ndarray shape (total_len,)
    """
    series = np.zeros(total_len)
    for t, v in zip(event_times, values):
        if t >= total_len:
            continue
        decay = v * np.exp(-np.arange(total_len - t) / tau)
        series[t:] += decay
    return series


def effective_rainfall(rain_series, decay):
    """
    计算指数衰减有效降雨
    eff(t) = rain(t) + decay * eff(t-1)
    """
    eff = 0.0
    eff_list = []
    for r in rain_series:
        eff = r + decay * eff
        eff_list.append(eff)
    return np.array(eff_list)


# ============================================================
# 主流程
# ============================================================

def main():
    out_dir = os.path.dirname(__file__) or 'Q5/feature'
    os.makedirs(out_dir, exist_ok=True)

    # ---- 1. 加载数据 ----
    print("加载数据...")
    df, base, (b1, b2) = load_pipeline()
    print(f"数据形状: {df.shape}, b1={b1}, b2={b2}")
    print(f"已有列: {list(df.columns)}")

    # 注意: load_pipeline 中 clean() 将 BlastDist/BlastCharge 的 NaN 填充为 0
    # 但由于原始正值保留不变，使用 BlastDist > 0 可正确识别爆破事件
    # （原非爆破时刻 fillna(0) 后为 0，爆破时刻保持正值）
    blast_mask = df['BlastDist'] > 0 if 'BlastDist' in df.columns else df['BlastCharge'] > 0
    print(f"BlastDist > 0 行数: {blast_mask.sum()}")

    # ---- 2. 连续传感器变量高级特征 ----
    print("\n[1/5] 连续传感器变量高级特征...")

    # (a) 差分特征
    if 'PorePressure' in df.columns:
        if 'Pore_diff1' not in df.columns:
            df['Pore_diff1'] = df['PorePressure'].diff(1)
        if 'Pore_diff6' not in df.columns:
            df['Pore_diff6'] = df['PorePressure'].diff(6)

    if 'Infiltration' in df.columns:
        if 'Infilt_diff1' not in df.columns:
            df['Infilt_diff1'] = df['Infiltration'].diff(1)
        if 'Infilt_diff6' not in df.columns:
            df['Infilt_diff6'] = df['Infiltration'].diff(6)

    # (b) 滚动标准差（12h 窗口 = 72 步）
    if 'PorePressure' in df.columns:
        if 'Pore_roll_std_12h' not in df.columns:
            df['Pore_roll_std_12h'] = df['PorePressure'].rolling(72, min_periods=1).std()
    if 'Infiltration' in df.columns:
        if 'Infilt_roll_std_12h' not in df.columns:
            df['Infilt_roll_std_12h'] = df['Infiltration'].rolling(72, min_periods=1).std()

    # (c) 趋势交互差分 (先确保交互项存在，再取 12h 差分)
    if 'PorePressure' in df.columns and 'Infiltration' in df.columns:
        if 'PoreInfilt' not in df.columns:
            df['PoreInfilt'] = df['PorePressure'] * df['Infiltration']
        if 'PoreInfilt_diff12h' not in df.columns:
            df['PoreInfilt_diff12h'] = df['PoreInfilt'].diff(72)

    # ---- 3. 降雨变量高级特征 ----
    print("[2/5] 降雨变量高级特征...")

    if 'Rainfall' in df.columns:

        # (a) 多窗口累积量
        rain_windows = {
            'Rain_3h_sum': 18,   # 3h
            'Rain_6h_sum': 36,   # 6h
            'Rain_12h_sum': 72,  # 12h
            'Rain_24h_sum': 144, # 24h
        }
        for col_name, win in rain_windows.items():
            if col_name not in df.columns:
                df[col_name] = df['Rainfall'].rolling(win, min_periods=1).sum()

        # (b) 多衰减常数有效降雨
        if 'Rain_eff_09' not in df.columns:
            df['Rain_eff_09'] = effective_rainfall(df['Rainfall'].values, 0.9)
        if 'Rain_eff_07' not in df.columns:
            df['Rain_eff_07'] = effective_rainfall(df['Rainfall'].values, 0.7)

        # (c) 降雨强度波动
        if 'Rain_6h_std' not in df.columns:
            df['Rain_6h_std'] = df['Rainfall'].rolling(36, min_periods=1).std()

    # ---- 4. 微震变量高级特征 ----
    print("[3/5] 微震变量高级特征...")

    if 'Microseismic' in df.columns:

        ms = df['Microseismic']

        # (a) 事件密度（滚动窗口内 >0 的计数）
        micro_windows = {
            'Micro_6h_count': 36,
            'Micro_12h_count': 72,
            'Micro_24h_count': 144,
        }
        for col_name, win in micro_windows.items():
            if col_name not in df.columns:
                df[col_name] = ms.rolling(win, min_periods=1).apply(
                    lambda x: (x > 0).sum(), raw=True
                )

        # (b) 最近 24h 累积事件数
        if 'Micro_24h_cum' not in df.columns:
            df['Micro_24h_cum'] = ms.rolling(144, min_periods=1).sum()

        # (c) 能量代理特征
        if 'Micro_energy_sqrt' not in df.columns:
            df['Micro_energy_sqrt'] = np.sqrt(ms)
        if 'Micro_energy_sq' not in df.columns:
            df['Micro_energy_sq'] = ms ** 2

    # ---- 5. 爆破变量高级特征 ----
    print("[4/5] 爆破变量高级特征...")

    tau = 50  # 衰减时间常数（~8.3h）

    if blast_mask.sum() > 0:
        # 找到爆破事件索引（使用 BlastDist > 0 识别）
        blast_idx = df.index[blast_mask].tolist()
        n_events = len(blast_idx)
        print(f"    爆破事件数: {n_events}")

        if n_events > 0:

            # (a) 爆破距离影响
            if 'BlastDist' in df.columns and 'BlastDist_impact' not in df.columns:
                blast_dist_vals = df.loc[blast_idx, 'BlastDist'].values
                # 权重 v = 1.0 / (d^2 + 1)
                w_dist = 1.0 / (blast_dist_vals ** 2 + 1)
                df['BlastDist_impact'] = exp_decay_series(
                    blast_idx, w_dist, tau, len(df)
                )

            # (b) 爆破药量影响
            if 'BlastCharge' in df.columns and 'BlastCharge_impact' not in df.columns:
                blast_q_vals = df.loc[blast_idx, 'BlastCharge'].values
                df['BlastCharge_impact'] = exp_decay_series(
                    blast_idx, blast_q_vals, tau, len(df)
                )

            # (c) 联合能量衰减影响  v = q / (d^2 + 1)
            if ('BlastDist' in df.columns and 'BlastCharge' in df.columns
                    and 'BlastEnergy_decay_impact' not in df.columns):
                blast_dist_vals = df.loc[blast_idx, 'BlastDist'].values
                blast_q_vals = df.loc[blast_idx, 'BlastCharge'].values
                w_energy = blast_q_vals / (blast_dist_vals ** 2 + 1)
                df['BlastEnergy_decay_impact'] = exp_decay_series(
                    blast_idx, w_energy, tau, len(df)
                )

            # (d) 爆破间隔
            if 'Blast_interval' not in df.columns:
                intervals = np.diff(blast_idx)  # 长度 n_events-1
                df['Blast_interval'] = np.nan
                # 第一个事件之前填 0
                df.loc[:blast_idx[0], 'Blast_interval'] = 0
                for i in range(len(intervals)):
                    df.loc[blast_idx[i]:blast_idx[i+1]-1, 'Blast_interval'] = intervals[i]
                # 最后一个事件之后延续最后一个间隔
                if len(intervals) > 0:
                    df.loc[blast_idx[-1]:, 'Blast_interval'] = intervals[-1]
                else:
                    df['Blast_interval'] = 0
                # 剩余 NaN 填 0
                df['Blast_interval'] = df['Blast_interval'].fillna(0)

        else:
            # 无爆破事件，填零
            for col in ['BlastDist_impact', 'BlastCharge_impact',
                        'BlastEnergy_decay_impact', 'Blast_interval']:
                if col not in df.columns:
                    df[col] = 0.0

    # 补充 Time_since_blast（如果基特征中没有）
    if 'Time_since_blast' not in df.columns and 'BlastCharge' in df.columns:
        from common.data_utils import _time_since_event
        df['Time_since_blast'] = _time_since_event(df['BlastCharge'].values)

    # ---- 6. 清理异常值 ----
    print("[5/5] 清理异常值...")
    # 将 inf/-inf 替换为 NaN，然后 NaN 填 0
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # ---- 7. 整理列顺序 ----
    print("整理列顺序...")

    # 目标变量列
    target_cols = ['Delta_D']
    if 'Displacement' in df.columns:
        target_cols.append('Displacement')
    target_cols = [c for c in target_cols if c in df.columns]

    # Phase 列
    phase_col = 'Phase'

    # 所有其他列（特征列）：排除 Phase 和目标变量
    exclude = set([phase_col] + target_cols)
    feature_cols = [c for c in df.columns if c not in exclude
                    and c != 'Time' and c != 'Hour']

    n_features = len(feature_cols)
    print(f"特征总数（不含 Phase 和目标）: {n_features}")

    # 重新排列
    final_cols = [phase_col] + feature_cols + target_cols
    # 确保所有列名都存在
    final_cols = [c for c in final_cols if c in df.columns]
    df_out = df[final_cols].copy()

    # ---- 8. 保存 ----
    out_path = os.path.join(out_dir, f'feature_{n_features}.xlsx')
    df_out.to_excel(out_path, index=False)
    print(f'[特征表] 已保存，特征总数 = {n_features}')
    print(f'    保存路径: {out_path}')
    print(f'    表结构: {list(df_out.columns)}')


if __name__ == '__main__':
    main()
