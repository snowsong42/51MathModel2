"""Q4 特征工程——从 Q5/feature/feature.py 移植并适配（去掉 Infiltration 相关特征）"""

import numpy as np
import pandas as pd


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


def add_pressure_features(df):
    """孔压相关特征（无干湿入渗）"""
    if 'PorePressure' not in df.columns:
        return df
    # 差分
    if 'Pore_diff1' not in df.columns:
        df['Pore_diff1'] = df['PorePressure'].diff(1)
    if 'Pore_diff6' not in df.columns:
        df['Pore_diff6'] = df['PorePressure'].diff(6)
    # 12h 滚动标准差
    if 'Pore_roll_std_12h' not in df.columns:
        df['Pore_roll_std_12h'] = df['PorePressure'].rolling(72, min_periods=1).std()
    # 孔压 × 降雨交互
    if 'Rainfall' in df.columns and 'Pore_Rain' not in df.columns:
        df['Pore_Rain'] = df['PorePressure'] * df['Rainfall']
    return df


def add_rainfall_features(df):
    """降雨相关特征"""
    if 'Rainfall' not in df.columns:
        return df
    # 多窗口累积量
    rain_windows = {
        'Rain_3h_sum': 18,
        'Rain_6h_sum': 36,
        'Rain_12h_sum': 72,
        'Rain_24h_sum': 144,
    }
    for col_name, win in rain_windows.items():
        if col_name not in df.columns:
            df[col_name] = df['Rainfall'].rolling(win, min_periods=1).sum()
    # 有效降雨（多衰减常数）
    if 'Rain_eff_09' not in df.columns:
        df['Rain_eff_09'] = effective_rainfall(df['Rainfall'].values, 0.9)
    if 'Rain_eff_07' not in df.columns:
        df['Rain_eff_07'] = effective_rainfall(df['Rainfall'].values, 0.7)
    # 降雨强度波动
    if 'Rain_6h_std' not in df.columns:
        df['Rain_6h_std'] = df['Rainfall'].rolling(36, min_periods=1).std()
    # 距上次降雨
    if 'Time_since_rain' not in df.columns:
        from Q4_LGBM.common.data_utils import _time_since_event
        df['Time_since_rain'] = _time_since_event(df['Rainfall'].values)
    return df


def add_microseismic_features(df):
    """微震事件特征"""
    if 'Microseismic' not in df.columns:
        return df
    ms = df['Microseismic']
    # 滚动窗口内事件密度（>0 计数）
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
    # 24h 累积事件数
    if 'Micro_24h_cum' not in df.columns:
        df['Micro_24h_cum'] = ms.rolling(144, min_periods=1).sum()
    # 能量代理特征
    if 'Micro_energy_sqrt' not in df.columns:
        df['Micro_energy_sqrt'] = np.sqrt(ms)
    if 'Micro_energy_sq' not in df.columns:
        df['Micro_energy_sq'] = ms ** 2
    return df


def add_blast_features(df, tau=50):
    """爆破相关特征——基于原始 NaN/非空识别事件"""
    # 识别爆破事件：原始数据中爆破列为 NaN→无事件，非空→有事件
    # 注意：此处使用原始值（尚未被填充）
    # 但实际上在 clean_train 中我们已经保留了 NaN，所以用 notna() 识别
    has_blast_dist = 'BlastDist' in df.columns and df['BlastDist'].notna().any()
    has_blast_charge = 'BlastCharge' in df.columns and df['BlastCharge'].notna().any()

    if not has_blast_dist and not has_blast_charge:
        # 无爆破事件，填零
        for col in ['BlastDist_impact', 'BlastCharge_impact',
                     'BlastEnergy_decay_impact', 'Blast_interval',
                     'Blast_PPV', 'Blast_Energy', 'Time_since_blast']:
            if col not in df.columns:
                df[col] = 0.0
        return df

    # 优先用 BlastDist 识别爆破事件
    if has_blast_dist:
        blast_mask = df['BlastDist'].notna() & (df['BlastDist'] > 0)
    else:
        blast_mask = df['BlastCharge'].notna() & (df['BlastCharge'] > 0)

    blast_idx = df.index[blast_mask].tolist()
    n_events = len(blast_idx)
    print(f"    爆破事件数: {n_events}")

    if n_events == 0:
        for col in ['BlastDist_impact', 'BlastCharge_impact',
                     'BlastEnergy_decay_impact', 'Blast_interval',
                     'Blast_PPV', 'Blast_Energy', 'Time_since_blast']:
            if col not in df.columns:
                df[col] = 0.0
        return df

    # (a) 爆破距离影响（权重 = 1/(d^2+1)）
    if 'BlastDist' in df.columns and 'BlastDist_impact' not in df.columns:
        blast_dist_vals = df.loc[blast_idx, 'BlastDist'].values
        w_dist = 1.0 / (blast_dist_vals ** 2 + 1)
        df['BlastDist_impact'] = exp_decay_series(blast_idx, w_dist, tau, len(df))

    # (b) 爆破药量影响
    if 'BlastCharge' in df.columns and 'BlastCharge_impact' not in df.columns:
        blast_q_vals = df.loc[blast_idx, 'BlastCharge'].values
        df['BlastCharge_impact'] = exp_decay_series(blast_idx, blast_q_vals, tau, len(df))

    # (c) 联合能量衰减影响 v = q / (d^2+1)
    if ('BlastDist' in df.columns and 'BlastCharge' in df.columns
            and 'BlastEnergy_decay_impact' not in df.columns):
        blast_dist_vals = df.loc[blast_idx, 'BlastDist'].values
        blast_q_vals = df.loc[blast_idx, 'BlastCharge'].values
        w_energy = blast_q_vals / (blast_dist_vals ** 2 + 1)
        df['BlastEnergy_decay_impact'] = exp_decay_series(blast_idx, w_energy, tau, len(df))

    # (d) 爆破间隔
    if 'Blast_interval' not in df.columns:
        intervals = np.diff(blast_idx)
        df['Blast_interval'] = np.nan
        df.loc[:blast_idx[0], 'Blast_interval'] = 0
        for i in range(len(intervals)):
            df.loc[blast_idx[i]:blast_idx[i + 1] - 1, 'Blast_interval'] = intervals[i]
        if len(intervals) > 0:
            df.loc[blast_idx[-1]:, 'Blast_interval'] = intervals[-1]
        else:
            df['Blast_interval'] = 0
        df['Blast_interval'] = df['Blast_interval'].fillna(0)

    # (e) PPV 瞬时特征（仅在爆破时刻非零）
    if 'Blast_PPV' not in df.columns or 'Blast_Energy' not in df.columns:
        if 'BlastDist' in df.columns and 'BlastCharge' in df.columns:
            d_safe = df['BlastDist'].values.copy()
            d_safe[d_safe < 1] = 1
            # blast 列暂时填充 NaN 以便计算
            q_filled = df['BlastCharge'].fillna(0).values
            d_filled = df['BlastDist'].fillna(0).values
            d_safe = d_filled.copy(); d_safe[d_safe < 1] = 1
            if 'Blast_PPV' not in df.columns:
                df['Blast_PPV'] = np.sqrt(np.abs(q_filled)) / d_safe
                # 只在爆破时刻非零
                df.loc[~blast_mask, 'Blast_PPV'] = 0
            if 'Blast_Energy' not in df.columns:
                df['Blast_Energy'] = q_filled / (d_safe ** 2)
                df.loc[~blast_mask, 'Blast_Energy'] = 0

    # (f) 距上次爆破
    if 'Time_since_blast' not in df.columns:
        if 'BlastCharge' in df.columns:
            from Q4_LGBM.common.data_utils import _time_since_event
            df['Time_since_blast'] = _time_since_event(df['BlastCharge'].fillna(0).values)
        else:
            df['Time_since_blast'] = 0

    return df


def add_time_features(df):
    """时间周期编码"""
    if 'Time' not in df.columns:
        return df
    if 'Hour' in df.columns:
        # 已存在，复用
        df['Day_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['Day_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    else:
        # 从 Time 列提取
        hour = pd.to_datetime(df['Time']).dt.hour
        df['Day_sin'] = np.sin(2 * np.pi * hour / 24)
        df['Day_cos'] = np.cos(2 * np.pi * hour / 24)
    return df


def build_features(df, is_train=True, tau=50):
    """
    统一特征构建入口
    is_train: True→训练集（构造 target）, False→实验集（保留 Displacement NaN）
    """
    # 1. 孔压特征
    df = add_pressure_features(df)
    # 2. 降雨特征
    df = add_rainfall_features(df)
    # 3. 微震特征
    df = add_microseismic_features(df)
    # 4. 爆破特征
    df = add_blast_features(df, tau=tau)
    # 5. 时间特征
    df = add_time_features(df)

    # 清理异常值
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    return df
