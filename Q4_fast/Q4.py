import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
import os, warnings
warnings.filterwarnings('ignore')
from feature import build_features # 同目录下通用特征工程

col_map = { # 列名映射
    'Time': 'Time',
    'Surface Displacement (mm)': 'Displacement',
    'Rainfall (mm)': 'Rainfall',
    'Pore Water Pressure (kPa)': 'PorePressure',
    'Microseismic Event Count': 'Microseismic',
    'Blasting Point Distance (m)': 'BlastDist',
    'Maximum Charge per Segment (kg)': 'BlastCharge',
    'Stage Label': 'StageLabel'
}

# ======================= 主流程 =======================
DIR = os.path.dirname(os.path.abspath(__file__))
train_raw = pd.read_excel(os.path.join(DIR, 'Attachment 4.xlsx'), sheet_name='训练集')
test_raw  = pd.read_excel(os.path.join(DIR, 'Attachment 4.xlsx'), sheet_name='实验集')

# ======================= 主流程 =======================
DIR = os.path.dirname(os.path.abspath(__file__))
train_raw = pd.read_excel(os.path.join(DIR, 'Attachment 4.xlsx'), sheet_name='训练集')
test_raw  = pd.read_excel(os.path.join(DIR, 'Attachment 4.xlsx'), sheet_name='实验集')

# 重命名列
train_raw.rename(columns=col_map, inplace=True)
test_raw.rename(columns=col_map, inplace=True)

# 读取分段边界
seg = pd.read_csv(os.path.join(DIR, 'segment', 'segment.csv'))
b1 = int(seg['结束索引'].iloc[0])
b2 = int(seg['结束索引'].iloc[1])
train_raw['Phase'] = 0
train_raw.loc[b1:b2, 'Phase'] = 1
train_raw.loc[b2:, 'Phase'] = 2

# 实验集阶段标签
phase_col = [c for c in test_raw.columns if '阶段' in c or 'Phase' in c]
if phase_col:
    mapping = {'缓慢匀速形变':0, '加速形变':1, '快速形变':2}
    test_raw['Phase'] = test_raw[phase_col[0]].map(mapping).astype(int)
else:
    test_raw['Phase'] = np.digitize(np.arange(len(test_raw)),
                                    [len(test_raw)//3, 2*len(test_raw)//3])

# 特征工程
train_feat = build_features(train_raw)
test_feat  = build_features(test_raw)

# 目标：位移增量
train_feat['Delta_D'] = train_feat['Displacement'].diff().fillna(0)

# 特征列（排除无关列）
exclude = ['Time','Displacement','Delta_D','BlastDist','BlastCharge','Phase']
feat_cols = [c for c in train_feat.columns if c not in exclude]

# 分阶段训练
models = {}
pred_delta = np.zeros(len(train_feat))
for ph in [0,1,2]:
    mask = train_feat['Phase'] == ph
    X = train_feat.loc[mask, feat_cols].values
    y = train_feat.loc[mask, 'Delta_D'].values
    if len(y) < 20: continue
    model = LGBMRegressor(n_estimators=500, learning_rate=0.01, random_state=42, verbosity=-1)
    model.fit(X, y)
    models[ph] = model
    pred_delta[mask] = model.predict(X)

# 训练评估
true_disp = train_feat['Displacement'].values
pred_disp = np.cumsum(pred_delta)
pred_disp = pred_disp + true_disp[0] - pred_disp[0]
r2 = 1 - np.sum((true_disp-pred_disp)**2) / np.sum((true_disp-true_disp.mean())**2)
print(f'训练集整体位移拟合 R² = {r2:.6f}')

# 实验集预测
test_delta = np.zeros(len(test_feat))
for ph in [0,1,2]:
    mask = test_feat['Phase'] == ph
    if mask.sum()==0 or ph not in models: continue
    X = test_feat.loc[mask, feat_cols].values
    test_delta[mask] = models[ph].predict(X)

test_disp = np.cumsum(test_delta)
test_disp = test_disp + true_disp[-1] - test_disp[0]

# 输出指定时间点
targets = ['2025-05-09 12:00','2025-05-27 08:00','2025-06-01 12:00',
           '2025-06-03 22:00','2025-06-04 01:40']
times = pd.to_datetime(test_feat['Time'])
print('\n表4.1 实验集表面位移预测结果')
print('时间点                | 预测位移 (mm)')
for t in targets:
    dt = pd.to_datetime(t)
    idx = (np.abs(times - dt)).idxmin()
    print(f'{t:<20} | {test_disp[idx]:.3f}')