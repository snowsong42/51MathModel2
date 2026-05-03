"""
Q5.1 最优变量组合建模
=====================
通过GBRT对比不同变量组合的预测性能
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings, os, sys, glob
warnings.filterwarnings('ignore')

# ============================================================
# 自动确定路径
# ============================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# ============================================================
# 查找数据文件
# ============================================================
xlsx_files = glob.glob('*.xlsx') + glob.glob('*.xls')
if not xlsx_files:
    print("[错误] 当前目录下未找到 .xlsx 文件！")
    print("请将附件5数据文件放入此目录。")
    sys.exit(1)

data_file = xlsx_files[0]

# ============================================================
# 列名匹配（通过部分字符串匹配）
# ============================================================
# 中文关键词 -> 英文列名
col_keywords = [
    ('时间', 'Time'),
    ('表面位移', 'Displacement'),
    ('降雨', 'Rainfall'),
    ('孔隙', 'PorePressure'),
    ('微震', 'Microseismic'),
    ('入渗', 'Infiltration'),
    ('爆破', 'BlastDist'),
    ('距离', 'BlastDist'),
    ('单段', 'BlastCharge'),
    ('药量', 'BlastCharge'),
    ('最大', 'BlastCharge'),
]

df = pd.read_excel(data_file)

col_map = {}
for cn_col in df.columns:
    cn_str = str(cn_col)
    matched = False
    for kw, en in col_keywords:
        if kw in cn_str:
            col_map[cn_col] = en
            matched = True
            break
    if not matched:
        print(f"  [警告] 未匹配列: {cn_str}")

df = df.rename(columns=col_map)
print(f"数据文件: {data_file}")
print(f"映射后列名: {df.columns.tolist()}")
print(f"数据形状: {df.shape}")

# ============================================================
# 数据读取与清洗
# ============================================================
print('=' * 72)
print('  Q5.1 最优变量组合建模 - GBRT')
print('=' * 72)

# 时间列处理
if 'Time' in df.columns:
    df['Time'] = pd.to_datetime(df['Time'])
    df = df.sort_values('Time').reset_index(drop=True)

# 爆破相关列为空时填充0
for c in ['BlastDist', 'BlastCharge']:
    if c in df.columns:
        df[c] = df[c].fillna(0)

# 其他列缺失处理
for c in ['Rainfall', 'PorePressure', 'Microseismic', 'Infiltration', 'Displacement']:
    if c in df.columns:
        df[c] = df[c].ffill().fillna(0)

# ============================================================
# 特征工程
# ============================================================
# 目标变量：10分钟位移增量
df['Delta_D'] = df['Displacement'].diff()

# 孔压变化率
df['Pore_Diff'] = df['PorePressure'].diff()

# 24小时累积 (144步)
window = 144
df['Rain_cum24'] = df['Rainfall'].rolling(window, min_periods=1).sum()

# 爆破累积
if 'BlastCharge' in df.columns:
    df['Blast_E_cum24'] = df['BlastCharge'].rolling(window, min_periods=1).sum()
    blast_vals = df['BlastCharge'].values
    blast_ts = []
    cnt = 10000
    for v in blast_vals:
        if v > 0:
            cnt = 0
            blast_ts.append(0)
        else:
            blast_ts.append(cnt + 1 if cnt < 10000 else 10000)
            cnt += 1
    df['Time_since_blast'] = blast_ts

# 距上次降雨时间
rain_vals = df['Rainfall'].values
rain_ts = []
cnt = 10000
for v in rain_vals:
    if v > 0:
        cnt = 0
        rain_ts.append(0)
    else:
        rain_ts.append(cnt + 1 if cnt < 10000 else 10000)
        cnt += 1
df['Time_since_rain'] = rain_ts

# 剔除NaN
df = df.dropna(subset=['Delta_D']).reset_index(drop=True)

# 基础特征（原始6个变量）
base_features_raw = ['Rainfall', 'PorePressure', 'Microseismic', 'Infiltration', 'BlastDist', 'BlastCharge']
base_features = [c for c in base_features_raw if c in df.columns]
print(f'基础特征: {base_features}')

# 全部特征
feature_cols_all = base_features + ['Pore_Diff', 'Rain_cum24']
if 'Blast_E_cum24' in df.columns:
    feature_cols_all.append('Blast_E_cum24')
if 'Time_since_blast' in df.columns:
    feature_cols_all.append('Time_since_blast')
if 'Time_since_rain' in df.columns:
    feature_cols_all.append('Time_since_rain')

X_full = df[feature_cols_all]
y = df['Delta_D']

print(f'样本数: {len(df)}')
print(f'特征数: {len(feature_cols_all)}')
print(f'特征列: {feature_cols_all}')

# ============================================================
# 划分训练/测试 (80/20 按时间顺序)
# ============================================================
split_idx = int(len(df) * 0.8)
X_train, X_test = X_full.iloc[:split_idx], X_full.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f'训练集: {len(X_train)} 样本')
print(f'测试集: {len(X_test)} 样本')

# ============================================================
# 评估函数（避免使用特殊字符）
# ============================================================
def evaluate_model(X_train, y_train, X_test, y_test, sel_features, model_name):
    if not sel_features:
        print(f"{model_name}: 无可用特征，跳过")
        return None
    model = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.01,
        max_depth=5, subsample=0.8, random_state=42
    )
    model.fit(X_train[sel_features], y_train)
    y_pred = model.predict(X_test[sel_features])
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name}:")
    print(f"  RMSE = {rmse:.4f} mm")
    print(f"  MAE  = {mae:.4f} mm")
    print(f"  R2   = {r2:.4f}")
    print("-" * 40)
    return (rmse, mae, r2)

# ============================================================
# 定义变量组合 (6个特征中每次去掉1个)
# ============================================================
combos = []; names = []
for removed in base_features:
    combo = [c for c in feature_cols_all if c != removed]
    if removed in ['BlastDist', 'BlastCharge']:
        combo = [c for c in combo if c not in ['Time_since_blast', 'Blast_E_cum24']]
    elif removed == 'Rainfall':
        combo = [c for c in combo if c not in ['Rain_cum24', 'Time_since_rain']]
    elif removed == 'PorePressure':
        combo = [c for c in combo if c not in ['Pore_Diff']]
    combos.append(combo)
    names.append(f"Model (去掉{removed})")

combos.append(feature_cols_all)
names.append("Model (全部变量)")

# ============================================================
# 运行评估
# ============================================================
print("\n========== 模型性能对比 ==========\n")
results = {}
for combo, name in zip(combos, names):
    metrics = evaluate_model(X_train, y_train, X_test, y_test, combo, name)
    if metrics:
        results[name] = metrics

# ============================================================
# 结果汇总
# ============================================================
print("\n========== 结果汇总 ==========")
print(f"{'Model':<35} {'RMSE':>8} {'MAE':>8} {'R2':>8}")
print("-" * 65)
for name, (rmse, mae, r2) in sorted(results.items(), key=lambda x: x[1][0]):
    print(f"{name:<35} {rmse:>8.4f} {mae:>8.4f} {r2:>8.4f}")

if results:
    best = min(results.items(), key=lambda x: x[1][0])
    worst = max(results.items(), key=lambda x: x[1][0])
    print(f"\n最佳模型: {best[0]}  (RMSE = {best[1][0]:.4f} mm)")
    print(f"最差模型: {worst[0]}  (RMSE = {worst[1][0]:.4f} mm)")
    print("\n结论: RMSE最小的组合即为最优变量组合。")
    print("=" * 72)
