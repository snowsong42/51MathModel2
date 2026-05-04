import pandas as pd
import numpy as np

def exp_decay_series(event_times, values, tau, total_len):
    ser = np.zeros(total_len)
    for t, v in zip(event_times, values):
        if t < total_len:
            ser[t:] += v * np.exp(-np.arange(total_len - t) / tau)
    return ser

def effective_rainfall(rain_series, decay):
    eff, res = 0.0, []
    for r in rain_series:
        eff = r + decay * eff
        res.append(eff)
    return np.array(res)

def build_features(df, is_train=True): # 通用特征工程
    df = df.copy()
    n = len(df)
    if 'Rainfall' in df.columns: # 降雨
        R = df['Rainfall'].values
        for w, tag in [(18,'R_3h'),(36,'R_6h'),(72,'R_12h'),(144,'R_24h')]:
            df[tag] = pd.Series(R).rolling(w, min_periods=1).sum()
        df['R_eff_09'] = effective_rainfall(R, 0.9)
        df['R_eff_07'] = effective_rainfall(R, 0.7)
        df['R_6h_std'] = pd.Series(R).rolling(36, min_periods=1).std()
        ts = np.zeros(n); last = -1
        for i in range(n):
            if R[i] > 0: last = i
            ts[i] = i - last if last >= 0 else 0
        df['Time_since_rain'] = ts
    if 'PorePressure' in df.columns: # 水压
        P = df['PorePressure']
        df['P_diff1'] = P.diff(1)
        df['P_diff6'] = P.diff(6)
        df['P_std_12h'] = P.rolling(72, min_periods=1).std()
        if 'Rainfall' in df.columns:
            df['P_Rain'] = P * df['Rainfall']
    if 'Infiltration' in df.columns: # 干湿
        I = df['Infiltration']
        df['I_diff1'] = I.diff(1)
        df['I_diff6'] = I.diff(6)
        df['I_std_12h'] = I.rolling(72, min_periods=1).std()
        if 'PorePressure' in df.columns:
            df['PoreInfilt'] = df['PorePressure'] * I
            df['PoreInfilt_diff12h'] = df['PoreInfilt'].diff(72) if 'PoreInfilt' in df.columns else 0
    if 'Microseismic' in df.columns: # 微震
        M = df['Microseismic']
        for w, tag in [(36,'M_6h_cnt'),(72,'M_12h_cnt'),(144,'M_24h_cnt')]:
            df[tag] = M.rolling(w, min_periods=1).apply(lambda x: (x>0).sum(), raw=True)
        df['M_24h_cum'] = M.rolling(144, min_periods=1).sum()
        df['M_sqrt'] = np.sqrt(M)
        df['M_sq'] = M**2
    blast_events = False # 爆破
    if 'BlastDist' in df.columns and 'BlastCharge' in df.columns:
        mask = df['BlastDist'].notna() & (df['BlastDist'] > 0)
        if mask.sum() > 0:
            blast_events = True
            idx = df.index[mask].tolist()
            dist = df.loc[idx, 'BlastDist'].values
            charge = df.loc[idx, 'BlastCharge'].values
            w_energy = charge / (dist**2 + 1)
            df['Blast_decay'] = exp_decay_series(idx, w_energy, 50, n)
            ts = np.zeros(n); last = -1
            for i in range(n):
                if mask.iloc[i]: last = i
                ts[i] = i - last if last >= 0 else 0
            df['Time_since_blast'] = ts
            df['Blast_interval'] = 0.0
            if len(idx) > 1:
                intervals = np.diff(idx)
                for k in range(len(intervals)):
                    df.loc[idx[k]:idx[k+1]-1, 'Blast_interval'] = intervals[k]
                df.loc[idx[-1]:, 'Blast_interval'] = intervals[-1]
    if not blast_events:
        for c in ['Blast_decay','Time_since_blast','Blast_interval']:
            if c not in df.columns:
                df[c] = 0.0
    if 'Time' in df.columns: # 时间
        hour = pd.to_datetime(df['Time']).dt.hour
        df['Day_sin'] = np.sin(2*np.pi*hour/24)
        df['Day_cos'] = np.cos(2*np.pi*hour/24)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df