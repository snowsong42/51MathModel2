"""数据加载、清洗、阶段划分"""

import numpy as np
import pandas as pd
import os


def map_columns(df):
    """中/英文列名 → 标准化短名"""
    rules = [
        ('时间','Time'),('表面位移','Displacement'),('降雨','Rainfall'),
        ('孔隙','PorePressure'),('微震','Microseismic'),('入渗','Infiltration'),
        ('爆破','BlastDist'),('距离','BlastDist'),('单段','BlastCharge'),
        ('药量','BlastCharge'),('最大','BlastCharge'),('阶段','Phase'),
        ('Stage Label','Phase'),
        # 英文全名
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
            if kw in s:
                col_map[c] = en
                break
    return df.rename(columns=col_map)


def clean_train(df):
    """训练集清洗——不填充爆破列 NaN"""
    if 'Time' in df.columns:
        df['Time'] = pd.to_datetime(df['Time'])
        df = df.sort_values('Time').reset_index(drop=True)
    # 传感器列前向填充（爆破列保留 NaN 用于识别事件）
    for c in ['Rainfall', 'PorePressure', 'Microseismic', 'Displacement']:
        if c in df.columns:
            df[c] = df[c].ffill().fillna(0)
    # 位移增量
    if 'Displacement' in df.columns:
        df['Delta_D'] = df['Displacement'].diff().fillna(0)
    return df


def clean_test(df):
    """实验集清洗——保留 Phase 标签"""
    if 'Time' in df.columns:
        df['Time'] = pd.to_datetime(df['Time'])
        df = df.sort_values('Time').reset_index(drop=True)
    # Stage Label 从 1,2,3 → 0,1,2
    if 'Phase' in df.columns:
        df['Phase'] = df['Phase'].map({1: 0, 2: 1, 3: 2}).fillna(0).astype(int)
    for c in ['Rainfall', 'PorePressure', 'Microseismic']:
        if c in df.columns:
            df[c] = df[c].ffill().fillna(0)
    # Displacement 全为 NaN（待预测），保留原样
    df['Delta_D'] = np.nan  # 占位
    return df


def _time_since_event(vals, thresh=1e-6, max_val=10000):
    """计算距上次事件的时间步数"""
    out = np.zeros(len(vals), dtype=int)
    cnt = 0
    for i in range(1, len(vals)):
        cnt = 0 if vals[i] > thresh else min(cnt + 1, max_val)
        out[i] = cnt
    return out


def load_segment(csv_path):
    """
    从 MATLAB 生成的 segment.csv 中读取断点索引 (b1, b2)
    返回 (b1, b2) 为 0-based Python 索引
    """
    import pandas as pd
    seg_df = pd.read_csv(csv_path)
    # 取第2行(索引1)和第3行(索引2)的'起始索引'列
    b1 = int(seg_df.loc[1, '起始索引'])
    b2 = int(seg_df.loc[2, '起始索引'])
    return b1, b2


def label_phase(df, b1, b2):
    """添加阶段标签"""
    df['Phase'] = 0
    df.loc[b1:b2 - 1, 'Phase'] = 1
    df.loc[b2:, 'Phase'] = 2
    return df


def load_data(data_dir=None):
    """加载 Q4 数据，返回 (train_df, test_df) 均标准化列名并清洗"""
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'common')
    path = os.path.join(data_dir, 'Attachment 4.xlsx')
    if not os.path.exists(path):
        path = os.path.join(os.path.dirname(__file__), 'Attachment 4.xlsx')
    print(f"加载数据: {path}")
    sheets = pd.ExcelFile(path).sheet_names
    print(f"工作表: {sheets}")

    train_raw = pd.read_excel(path, sheet_name='训练集')
    test_raw = pd.read_excel(path, sheet_name='实验集')

    # 列标准化
    train_df = map_columns(train_raw)
    test_df = map_columns(test_raw)

    print(f"训练集列名: {list(train_df.columns)}")
    print(f"实验集列名: {list(test_df.columns)}")

    # 清洗
    train_df = clean_train(train_df)
    test_df = clean_test(test_df)

    # ---- 尝试加载 MATLAB 滤波后的位移并替换 ----
    seg_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'segment'))
    filt_path = os.path.join(seg_dir, 'displacement_filtered.csv')
    if os.path.exists(filt_path):
        filt_df = pd.read_csv(filt_path)
        filt_df['Time'] = pd.to_datetime(filt_df['Time'])
        # 只保留必要的列做 merge
        filt_df = filt_df[['Time', 'Displacement_filtered']].copy()
        # 合并到 train_df（按 Time 左连接）
        train_df = train_df.merge(filt_df, on='Time', how='left')
        # 替换 Displacement 列
        train_df['Displacement'] = train_df['Displacement_filtered']
        train_df.drop(columns=['Displacement_filtered'], inplace=True)
        # 重新计算 Delta_D（滤波后）
        train_df['Delta_D'] = train_df['Displacement'].diff().fillna(0)
        print(f"已加载滤波后位移: {filt_path} ({len(train_df)} 条)")
    else:
        print(f"未找到滤波后位移文件 ({filt_path})，使用原始位移")

    return train_df, test_df
