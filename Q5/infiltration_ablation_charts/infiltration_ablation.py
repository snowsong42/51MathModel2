"""
Q5.1 干湿入渗系数 消融对比实验
============================
对比：保留 Infiltration 全部特征 vs 去掉 Infiltration 及其衍生特征
生成两张效果对比图
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings, os, sys, glob
warnings.filterwarnings('ignore')

import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 中文字体
plt.rcParams['axes.unicode_minus'] = False
for font_name in ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']:
    try:
        plt.rcParams['font.sans-serif'] = [font_name]
        fig_test = plt.figure()
        fig_test.text(0.5, 0.5, 'test', fontsize=12)
        plt.close(fig_test)
        print(f"[OK] Font: {font_name}")
        break
    except:
        continue

# 路径设置
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = script_dir
parent_dir = os.path.dirname(script_dir)  # Q5_0 目录

os.chdir(parent_dir)
xlsx_files = glob.glob('*.xlsx') + glob.glob('*.xls')
if not xlsx_files:
    print("[ERROR] No .xlsx found!")
    sys.exit(1)
data_file = xlsx_files[0]

# ============================================================
# 数据加载 & 列名匹配
# ============================================================
col_keywords = [
    ('时间', 'Time'), ('表面位移', 'Displacement'),
    ('降雨', 'Rainfall'), ('孔隙', 'PorePressure'),
    ('微震', 'Microseismic'), ('入渗', 'Infiltration'),
    ('爆破', 'BlastDist'), ('距离', 'BlastDist'),
    ('单段', 'BlastCharge'), ('药量', 'BlastCharge'), ('最大', 'BlastCharge'),
]

df = pd.read_excel(data_file)
col_map = {}
for cn_col in df.columns:
    for kw, en in col_keywords:
        if kw in str(cn_col):
            col_map[cn_col] = en
            break
df = df.rename(columns=col_map)

print('=' * 72)
print('  Q5.1 Infiltration Ablation Experiment')
print('=' * 72)
print(f"Data: {data_file}, Shape: {df.shape}")

if 'Time' in df.columns:
    df['Time'] = pd.to_datetime(df['Time'])
    df = df.sort_values('Time').reset_index(drop=True)

for c in ['BlastDist', 'BlastCharge']:
    if c in df.columns:
        df[c] = df[c].fillna(0)

for c in ['Rainfall', 'PorePressure', 'Microseismic', 'Infiltration', 'Displacement']:
    if c in df.columns:
        df[c] = df[c].ffill().fillna(0)

# ============================================================
# 构造特征
# ============================================================
df['Delta_D'] = np.diff(df['Displacement'].values, prepend=df['Displacement'].iloc[0])

base_features = ['Rainfall', 'PorePressure', 'Microseismic', 'Infiltration', 'BlastDist', 'BlastCharge']
base_features = [c for c in base_features if c in df.columns]

def find_optimal_lag(x, y, max_lag=288):
    n = len(x)
    best_lag, best_corr = 0, 0
    for lag in range(0, max_lag + 1, 6):
        if lag == 0:
            c = abs(np.corrcoef(x, y)[0, 1])
        else:
            c = abs(np.corrcoef(x[:-lag], y[lag:])[0, 1])
        if c > best_corr:
            best_corr = c
            best_lag = lag
    return best_lag, best_corr

y_raw = df['Delta_D'].values

lag_info = {}
for feat in base_features:
    x = df[feat].values
    lag, corr = find_optimal_lag(x, y_raw)
    lag_info[feat] = {'lag': lag, 'corr': corr}
    print(f"  {feat}: lag={lag}step({lag*10}min), corr={corr:.4f}")

for feat, info in lag_info.items():
    lag = info['lag']
    if lag > 0:
        df[f'{feat}_lag{lag}'] = df[feat].shift(lag)
        info['lagged_col'] = f'{feat}_lag{lag}'
    else:
        info['lagged_col'] = feat

# 衍生特征
if 'PorePressure' in df.columns:
    df['Pore_Diff'] = df['PorePressure'].diff()
if 'Rainfall' in df.columns:
    df['Rain_cum24'] = df['Rainfall'].rolling(144, min_periods=1).sum()
if 'Infiltration' in df.columns:
    df['Infiltration_Diff'] = df['Infiltration'].diff()
if 'Microseismic' in df.columns:
    df['Microseismic_roll6'] = df['Microseismic'].rolling(6, min_periods=1).sum()

if 'PorePressure' in df.columns and 'Rainfall' in df.columns:
    df['Pore_Rain_Cross'] = df['PorePressure'] * df['Rainfall']
if 'PorePressure' in df.columns and 'Infiltration' in df.columns:
    df['Pore_Infiltration_Cross'] = df['PorePressure'] * df['Infiltration']

if 'BlastCharge' in lag_info:
    bcol = lag_info['BlastCharge']['lagged_col']
    df['Blast_E_cum24'] = df[bcol].rolling(144, min_periods=1).sum()
    blast_vals = df[bcol].values
    blast_ts = np.zeros(len(blast_vals), dtype=int)
    cnt = 0
    for i in range(1, len(blast_vals)):
        cnt = cnt + 1 if blast_vals[i] == 0 else 0
        blast_ts[i] = cnt
    df['Time_since_blast'] = blast_ts

if 'Rainfall' in lag_info:
    rcol = lag_info['Rainfall']['lagged_col']
    rain_vals = df[rcol].values
    rain_ts = np.zeros(len(rain_vals), dtype=int)
    cnt = 0
    for i in range(1, len(rain_vals)):
        cnt = cnt + 1 if rain_vals[i] == 0 else 0
        rain_ts[i] = cnt
    df['Time_since_rain'] = rain_ts

df['Time_idx'] = np.arange(len(df))
if 'Time' in df.columns:
    df['Hour_of_day'] = pd.to_datetime(df['Time']).dt.hour
else:
    df['Hour_of_day'] = 0
df['Day_sin'] = np.sin(2 * np.pi * df['Hour_of_day'] / 24)
df['Day_cos'] = np.cos(2 * np.pi * df['Hour_of_day'] / 24)
df['Disp_cum24'] = df['Delta_D'].rolling(144, min_periods=1).sum()

if 'BlastDist' in df.columns and 'BlastCharge' in df.columns:
    safe_dist = df['BlastDist'].values.copy()
    safe_dist[safe_dist < 1.0] = 1.0
    df['Blast_PPV'] = np.sqrt(np.abs(df['BlastCharge'].values)) / safe_dist
    df['Blast_PPV_log'] = np.log1p(df['Blast_PPV'].clip(lower=0))
    df['Blast_Energy'] = df['BlastCharge'].values / (safe_dist ** 2)
    df['Blast_Energy_log'] = np.log1p(df['Blast_Energy'].clip(lower=0))

# 阶段划分
vel = np.diff(df['Displacement'].values, prepend=df['Displacement'].iloc[0])
vel_smooth = pd.Series(vel).rolling(window=50, center=True, min_periods=1).mean().values

b1, b2 = len(df), len(df)
for i in range(len(df) - 10):
    if vel_smooth[i:i+10].mean() > 0.02:
        b1 = i; break
for i in range(len(df) - 10):
    if vel_smooth[i:i+10].mean() > 0.10:
        b2 = i; break
if b1 >= b2:
    b1 = max(1, b2 - 200)

df['Phase'] = 0
df.loc[b1:b2-1, 'Phase'] = 1
df.loc[b2:, 'Phase'] = 2

print(f"Phase: 0~{b1-1}(Slow) | {b1}~{b2-1}(Accel) | {b2}~{len(df)-1}(Fast)")

# 全部特征列表
all_features = [info['lagged_col'] for info in lag_info.values()] + [
    'Pore_Diff', 'Rain_cum24', 'Infiltration_Diff', 'Microseismic_roll6',
    'Pore_Rain_Cross', 'Pore_Infiltration_Cross',
    'Disp_cum24', 'Time_idx',
]
for blf in ['Blast_E_cum24', 'Time_since_blast',
            'Blast_PPV', 'Blast_PPV_log', 'Blast_Energy', 'Blast_Energy_log']:
    if blf in df.columns:
        all_features.append(blf)
if 'Time_since_rain' in df.columns:
    all_features.append('Time_since_rain')
for tf in ['Day_sin', 'Day_cos']:
    if tf in df.columns:
        all_features.append(tf)
all_features = [c for c in all_features if c in df.columns]

# 去掉 Infiltration 及其衍生特征
infiltration_related = ['Infiltration', 'infiltration', 'Infiltration_Diff',
                        'Pore_Infiltration_Cross']
no_infiltration_features = [f for f in all_features 
                            if not any(kw in f for kw in infiltration_related)]

df = df.dropna(subset=all_features + ['Delta_D']).reset_index(drop=True)

print(f"\nFeatures WITH Infiltration:     {len(all_features)} features")
print(f"Features WITHOUT Infiltration:  {len(no_infiltration_features)} features")

# ============================================================
# 训练函数
# ============================================================
def train_phase_model(df, feature_list, phase_names=None):
    if phase_names is None:
        phase_names = {0: 'Slow', 1: 'Accel', 2: 'Fast'}
    
    y_pred_all = np.zeros(len(df))
    phase_metrics = {}
    
    for phase_id, phase_name in phase_names.items():
        mask = df['Phase'] == phase_id
        if mask.sum() < 50:
            continue
        
        X_ph = df.loc[mask, feature_list].values
        y_ph = df.loc[mask, 'Delta_D'].values
        
        model = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.01,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1, n_jobs=1, force_col_wise=True
        )
        model.fit(X_ph, y_ph)
        y_pred_ph = model.predict(X_ph)
        
        rmse = np.sqrt(mean_squared_error(y_ph, y_pred_ph))
        mae = mean_absolute_error(y_ph, y_pred_ph)
        r2 = r2_score(y_ph, y_pred_ph)
        
        phase_metrics[phase_id] = {'rmse': rmse, 'mae': mae, 'r2': r2, 'n': mask.sum()}
        y_pred_all[mask] = y_pred_ph
    
    overall_rmse = np.sqrt(mean_squared_error(df['Delta_D'].values, y_pred_all))
    overall_mae = mean_absolute_error(df['Delta_D'].values, y_pred_all)
    overall_r2 = r2_score(df['Delta_D'].values, y_pred_all)
    
    return y_pred_all, phase_metrics, overall_rmse, overall_mae, overall_r2

# ============================================================
# 训练两个模型
# ============================================================
print("\n--- Training WITH Infiltration (全变量) ---")
y_pred_with, metrics_with, rmse_with, mae_with, r2_with = train_phase_model(df, all_features)

print("\n--- Training WITHOUT Infiltration (去掉入渗系数) ---")
y_pred_without, metrics_without, rmse_without, mae_without, r2_without = train_phase_model(df, no_infiltration_features)

# ============================================================
# 打印结果
# ============================================================
print("\n" + "=" * 72)
print("  RESULTS: Infiltration Ablation Comparison")
print("=" * 72)
print(f"\n{'Metric':<15} {'WITH Infiltration':>22} {'WITHOUT Infiltration':>24} {'Delta':>12}")
print("-" * 75)
print(f"{'RMSE (mm)':<15} {rmse_with:>22.4f} {rmse_without:>24.4f} {rmse_without - rmse_with:>+12.4f}")
print(f"{'MAE (mm)':<15} {mae_with:>22.4f} {mae_without:>24.4f} {mae_without - mae_with:>+12.4f}")
print(f"{'R2':<15} {r2_with:>22.4f} {r2_without:>24.4f} {r2_without - r2_with:>+12.4f}")

print(f"\n--- Per Phase ---")
print(f"{'Phase':<12} {'WITH RMSE':>12} {'WITHOUT RMSE':>13} {'Change':>10}")
print("-" * 50)
for pid in [0, 1, 2]:
    if pid in metrics_with and pid in metrics_without:
        rw = metrics_with[pid]['rmse']
        rwo = metrics_without[pid]['rmse']
        name = ['Slow', 'Accel', 'Fast'][pid]
        print(f"{name:<12} {rw:>12.4f} {rwo:>13.4f} {rwo - rw:>+10.4f}")

# ============================================================
# 图表 1: 累积位移预测对比
# ============================================================
fig, axes = plt.subplots(3, 1, figsize=(16, 12),
                          gridspec_kw={'height_ratios': [2.5, 0.9, 0.9]})

ax1 = axes[0]
disp_cum_with = df['Displacement'].iloc[0] + y_pred_with.cumsum()
disp_cum_without = df['Displacement'].iloc[0] + y_pred_without.cumsum()

ax1.plot(df.index, df['Displacement'], 'k-', alpha=0.7, linewidth=0.6, label='True Displacement')
ax1.plot(df.index, disp_cum_with, 'b-', alpha=0.6, linewidth=0.8, 
         label=f'With Infiltration (RMSE={rmse_with:.4f}, R2={r2_with:.3f})')
ax1.plot(df.index, disp_cum_without, 'r--', alpha=0.7, linewidth=0.8, 
         label=f'Without Infiltration (RMSE={rmse_without:.4f}, R2={r2_without:.3f})')

for bx, color in [(b1, 'green'), (b2, 'orange')]:
    ax1.axvline(x=bx, color=color, linestyle='--', alpha=0.6, linewidth=1)
ax1.legend(loc='upper left', fontsize=9)
ax1.set_ylabel('Cumulative Displacement (mm)')
ax1.set_title(f'Infiltration Ablation: Cumulative Displacement Prediction\n'
              f'RMSE with Infiltration={rmse_with:.4f} -> without={rmse_without:.4f}  '
              f'(Δ={(rmse_without - rmse_with):.4f} mm, +{(rmse_without/rmse_with - 1)*100:.1f}%)')

ax1.text(b1//2, df['Displacement'].max()*1.06, 'Slow', ha='center', color='green', fontsize=10)
ax1.text((b1+b2)//2, df['Displacement'].max()*1.06, 'Accelerating', ha='center', color='orange', fontsize=10)
ax1.text((b2+len(df))//2, df['Displacement'].max()*1.06, 'Fast', ha='center', color='red', fontsize=10)

ax2 = axes[1]
ax2.plot(df.index, df['Delta_D'], 'k-', alpha=0.3, linewidth=0.4, label='True Delta_D')
ax2.plot(df.index, y_pred_with, 'b-', alpha=0.6, linewidth=0.5, label='With Infiltration')
ax2.plot(df.index, y_pred_without, 'r--', alpha=0.6, linewidth=0.5, label='Without Infiltration')
for bx, color in [(b1, 'green'), (b2, 'orange')]:
    ax2.axvline(x=bx, color=color, linestyle='--', alpha=0.6, linewidth=1)
ax2.set_ylabel('Delta_D (mm)')
ax2.legend(loc='upper right', fontsize=8)

ax3 = axes[2]
resid_with = df['Delta_D'].values - y_pred_with
resid_without = df['Delta_D'].values - y_pred_without
ax3.plot(df.index, resid_with, 'b-', alpha=0.4, linewidth=0.4, 
         label=f'With Infiltration (std={np.std(resid_with):.4f})')
ax3.plot(df.index, resid_without, 'r--', alpha=0.4, linewidth=0.4, 
         label=f'Without Infiltration (std={np.std(resid_without):.4f})')
ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
for bx, color in [(b1, 'green'), (b2, 'orange')]:
    ax3.axvline(x=bx, color=color, linestyle='--', alpha=0.6, linewidth=1)
ax3.set_ylabel('Residual (mm)')
ax3.set_xlabel('Sample Index')
ax3.legend(loc='upper right', fontsize=8)

fig.tight_layout()
fig.savefig(os.path.join(output_dir, 'infiltration_ablation_timeseries.png'), dpi=150)
plt.close(fig)
print(f"\n  [Chart] infiltration_ablation_timeseries.png saved")

# ============================================================
# 图表 2: 散点密度对比
# ============================================================
fig2, axes2 = plt.subplots(1, 2, figsize=(13, 6))

max_val = max(df['Delta_D'].max(), y_pred_with.max(), y_pred_without.max())

ax_left = axes2[0]
ax_left.scatter(df['Delta_D'], y_pred_with, alpha=0.2, s=1.5, c='blue')
ax_left.plot([0, max_val], [0, max_val], 'r--', linewidth=1)
ax_left.set_xlabel('True Delta_D (mm)')
ax_left.set_ylabel('Predicted Delta_D (mm)')
ax_left.set_title(f'With Infiltration ({len(all_features)} features)\n'
                  f'RMSE={rmse_with:.4f}, MAE={mae_with:.4f}, R2={r2_with:.4f}')

ax_right = axes2[1]
ax_right.scatter(df['Delta_D'], y_pred_without, alpha=0.2, s=1.5, c='red')
ax_right.plot([0, max_val], [0, max_val], 'r--', linewidth=1)
ax_right.set_xlabel('True Delta_D (mm)')
ax_right.set_ylabel('Predicted Delta_D (mm)')
ax_right.set_title(f'Without Infiltration ({len(no_infiltration_features)} features)\n'
                   f'RMSE={rmse_without:.4f}, MAE={mae_without:.4f}, R2={r2_without:.4f}')

fig2.suptitle(f'Infiltration Ablation: Scatter Comparison\n'
              f'RMSE change: {rmse_with:.4f} -> {rmse_without:.4f} '
              f'(Δ={(rmse_without - rmse_with):.4f} mm)', fontsize=12, fontweight='bold')
fig2.tight_layout()
fig2.savefig(os.path.join(output_dir, 'infiltration_ablation_scatter.png'), dpi=150)
plt.close(fig2)
print(f"  [Chart] infiltration_ablation_scatter.png saved")

print("\n" + "=" * 72)
print("  DONE: Charts saved to", output_dir)
print("=" * 72)
