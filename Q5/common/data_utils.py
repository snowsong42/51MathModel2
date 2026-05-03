"""数据加载、清洗、特征工程、阶段划分"""

import numpy as np
import pandas as pd
import glob, os, sys
from scipy import signal

def find_data(start_dir='.'):
    """查找目录下的第一个 xlsx 文件"""
    files = glob.glob(os.path.join(start_dir, '*.xlsx')) + glob.glob(os.path.join(start_dir, '*.xls'))
    if files: return files[0]
    # 递归查找
    for root, dirs, _ in os.walk(start_dir):
        for f in glob.glob(os.path.join(root, '*.xlsx')) + glob.glob(os.path.join(root, '*.xls')):
            return f
    return None

def map_columns(df):
    """中/英文列名 → 标准化短名"""
    rules = [
        ('时间','Time'),('表面位移','Displacement'),('降雨','Rainfall'),
        ('孔隙','PorePressure'),('微震','Microseismic'),('入渗','Infiltration'),
        ('爆破','BlastDist'),('距离','BlastDist'),('单段','BlastCharge'),
        ('药量','BlastCharge'),('最大','BlastCharge'),
        # 英文全名匹配
        ('Time','Time'),('Surface Displacement','Displacement'),
        ('Rainfall','Rainfall'),('Pore Water Pressure','PorePressure'),
        ('Microseismic','Microseismic'),('Dry-Wet Infiltration','Infiltration'),
        ('Blasting Point Distance','BlastDist'),
        ('Maximum Charge per Segment','BlastCharge'),
    ]
    col_map = {}
    for c in df.columns:
        s = str(c)
        for kw, en in rules:
            if kw in s: col_map[c] = en; break
    return df.rename(columns=col_map)

def clean(df):
    """基础清洗"""
    if 'Time' in df.columns:
        df['Time'] = pd.to_datetime(df['Time'])
        df = df.sort_values('Time').reset_index(drop=True)
    for c in ['BlastDist','BlastCharge']:
        if c in df.columns: df[c] = df[c].fillna(0)
    for c in ['Rainfall','PorePressure','Microseismic','Infiltration','Displacement']:
        if c in df.columns: df[c] = df[c].ffill().fillna(0)
    return df

def feat_engineer(df):
    """特征工程"""
    base = ['Rainfall','PorePressure','Microseismic','Infiltration','BlastDist','BlastCharge']
    base = [c for c in base if c in df.columns]

    # 位移增量
    if 'Displacement' in df.columns:
        df['Delta_D'] = df['Displacement'].diff()

    # 最优滞后搜索（每6步扫描）
    y = df['Delta_D'].values if 'Delta_D' in df.columns else np.zeros(len(df))
    for feat in base:
        x, best_lag, best_corr = df[feat].values, 0, 0
        for lag in range(0, 289, 6):
            if lag == 0:
                c = abs(np.corrcoef(x, y)[0,1]) if len(x)>1 else 0
            else:
                c = abs(np.corrcoef(x[:-lag], y[lag:])[0,1]) if len(x)>lag else 0
            if c > best_corr: best_corr, best_lag = c, lag
        if best_lag > 0:
            df[f'{feat}_lag{best_lag}'] = df[feat].shift(best_lag)
            df[f'{feat}_lag'] = df[f'{feat}_lag{best_lag}']
        else:
            df[f'{feat}_lag'] = df[feat]

    # 孔压变化
    if 'PorePressure' in df.columns:
        df['Pore_Diff'] = df['PorePressure'].diff()

    # 24h累积
    for c in ['Rainfall','Infiltration']:
        if c in df.columns: df[f'{c}_cum24'] = df[c].rolling(144, min_periods=1).sum()

    # 微震滑动6步
    if 'Microseismic' in df.columns:
        df['Microseismic_roll6'] = df['Microseismic'].rolling(6, min_periods=1).sum()

    # 交叉特征
    if 'PorePressure' in df.columns and 'Rainfall' in df.columns:
        df['Pore_Rain'] = df['PorePressure'] * df['Rainfall']
    if 'PorePressure' in df.columns and 'Infiltration' in df.columns:
        df['Pore_Infilt'] = df['PorePressure'] * df['Infiltration']

    # 爆破PPV
    if 'BlastDist' in df.columns and 'BlastCharge' in df.columns:
        safe = df['BlastDist'].values.copy(); safe[safe<1]=1
        df['Blast_PPV'] = np.sqrt(np.abs(df['BlastCharge'].values)) / safe
        df['Blast_Energy'] = df['BlastCharge'].values / (safe**2)
        df['Time_since_blast'] = _time_since_event(df['BlastCharge'].values)

    # 距上次降雨
    if 'Rainfall' in df.columns:
        df['Time_since_rain'] = _time_since_event(df['Rainfall'].values)

    # 时间特征
    if 'Time' in df.columns:
        df['Hour'] = df['Time'].dt.hour
        df['Day_sin'] = np.sin(2*np.pi*df['Hour']/24)
        df['Day_cos'] = np.cos(2*np.pi*df['Hour']/24)

    df['Disp_cum24'] = df['Delta_D'].rolling(144, min_periods=1).sum() if 'Delta_D' in df.columns else 0

    return df, base

def _time_since_event(vals, thresh=1e-6, max_val=10000):
    """计算距上次事件的时间步数"""
    out = np.zeros(len(vals), dtype=int)
    cnt = 0
    for i in range(1, len(vals)):
        cnt = 0 if vals[i] > thresh else min(cnt+1, max_val)
        out[i] = cnt
    return out

def get_all_features(df, base):
    """获取全部特征列表"""
    feats = [f'{c}_lag' for c in base if f'{c}_lag' in df.columns]
    extra = ['Pore_Diff','Rainfall_cum24','Infiltration_cum24','Microseismic_roll6',
             'Pore_Rain','Pore_Infilt','Blast_PPV','Blast_Energy','Time_since_blast',
             'Time_since_rain','Disp_cum24','Day_sin','Day_cos']
    for e in extra:
        if e in df.columns: feats.append(e)
    return [f for f in feats if f in df.columns]

def divide_phases(df, vel_col='Displacement'):
    """基于速度滑动平均划分三阶段，返回 (b1, b2)"""
    vel = np.diff(df[vel_col].values, prepend=df[vel_col].iloc[0])
    vs = pd.Series(vel).rolling(50, center=True, min_periods=1).mean().values
    n = len(df); b1, b2 = n, n
    for i in range(n-10):
        if vs[i:i+10].mean() > 0.02: b1 = i; break
    for i in range(n-10):
        if vs[i:i+10].mean() > 0.10: b2 = i; break
    if b1 >= b2: b1 = max(1, b2-200)
    return b1, b2

def label_phase(df, b1, b2):
    """添加阶段标签"""
    df['Phase'] = 0
    df.loc[b1:b2-1, 'Phase'] = 1
    df.loc[b2:, 'Phase'] = 2
    return df

def load_pipeline(path=None):
    """一站式加载: path → DataFrame(已清洗+特征+阶段)"""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), 'Attachment 5.xlsx')
    df = pd.read_excel(path)
    df = map_columns(df)
    df = clean(df)
    df, base = feat_engineer(df)
    b1, b2 = divide_phases(df)
    df = label_phase(df, b1, b2)
    df = df.dropna(subset=['Delta_D']).reset_index(drop=True)
    return df, base, (b1, b2)
