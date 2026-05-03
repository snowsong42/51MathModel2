"""
Q5.1 四维度消融实验汇总对比
=========================
对比四个变量维度：Rainfall / PorePressure / Microseismic / Infiltration
统一算法，一次性生成全部对比图表
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

plt.rcParams['axes.unicode_minus'] = False
for font_name in ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']:
    try:
        plt.rcParams['font.sans-serif'] = [font_name]
        fig_test = plt.figure()
        fig_test.text(0.5, 0.5, 'test', fontsize=12)
        plt.close(fig_test)
        break
    except:
        continue

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
os.chdir(parent_dir)
xlsx_files = glob.glob('*.xlsx') + glob.glob('*.xls')
data_file = xlsx_files[0]

# ============================================================
# 数据加载 & 特征工程 (一次完成)
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

if 'Time' in df.columns:
    df['Time'] = pd.to_datetime(df['Time'])
    df = df.sort_values('Time').reset_index(drop=True)

for c in ['BlastDist', 'BlastCharge']:
    if c in df.columns:
        df[c] = df[c].fillna(0)
for c in ['Rainfall', 'PorePressure', 'Microseismic', 'Infiltration', 'Displacement']:
    if c in df.columns:
        df[c] = df[c].ffill().fillna(0)

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

for feat, info in lag_info.items():
    lag = info['lag']
    if lag > 0:
        df[f'{feat}_lag{lag}'] = df[feat].shift(lag)
        info['lagged_col'] = f'{feat}_lag{lag}'
    else:
        info['lagged_col'] = feat

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

df = df.dropna(subset=all_features + ['Delta_D']).reset_index(drop=True)

print('=' * 72)
print('  Q5.1 Four-Dimension Ablation Comparison')
print(f'  Data: {len(df)} rows, {len(all_features)} features')
print('=' * 72)

# ============================================================
# 四组消融定义
# ============================================================
ablation_defs = {
    'Rainfall': {
        'label': 'Rainfall',
        'name_cn': '降雨量',
        'keywords': ['Rainfall', 'rainfall', 'Rain_cum', 'Time_since_rain', 'Pore_Rain_Cross'],
        'color': '#3498db',
    },
    'PorePressure': {
        'label': 'PorePressure',
        'name_cn': '孔隙水压力',
        'keywords': ['PorePressure', 'Pore', 'pore', 'Pore_Diff', 'Pore_Rain_Cross', 'Pore_Infiltration_Cross'],
        'color': '#e74c3c',
    },
    'Microseismic': {
        'label': 'Microseismic',
        'name_cn': '微震',
        'keywords': ['Microseismic', 'microseismic', 'Microseismic_roll6'],
        'color': '#2ecc71',
    },
    'Infiltration': {
        'label': 'Infiltration',
        'name_cn': '干湿入渗系数',
        'keywords': ['Infiltration', 'infiltration', 'Infiltration_Diff', 'Pore_Infiltration_Cross'],
        'color': '#9b59b6',
    },
}

# 训练函数
def train_phase_model(df, feature_list):
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
        phase_metrics[phase_id] = {
            'rmse': np.sqrt(mean_squared_error(y_ph, y_pred_ph)),
            'mae': mean_absolute_error(y_ph, y_pred_ph),
            'r2': r2_score(y_ph, y_pred_ph),
            'n': mask.sum()
        }
        y_pred_all[mask] = y_pred_ph
    overall_rmse = np.sqrt(mean_squared_error(df['Delta_D'].values, y_pred_all))
    overall_mae = mean_absolute_error(df['Delta_D'].values, y_pred_all)
    overall_r2 = r2_score(df['Delta_D'].values, y_pred_all)
    return y_pred_all, phase_metrics, overall_rmse, overall_mae, overall_r2

# ============================================================
# 运行全变量 & 四组消融
# ============================================================
print("\n--- Training FULL MODEL (23 features) ---")
y_full, phase_full, rmse_full, mae_full, r2_full = train_phase_model(df, all_features)
print(f"  FULL: RMSE={rmse_full:.4f}, MAE={mae_full:.4f}, R2={r2_full:.4f}")

results = {}
for abl_key, abl_def in ablation_defs.items():
    print(f"\n--- Training WITHOUT {abl_def['name_cn']} ---")
    keywords = abl_def['keywords']
    ablated_features = [f for f in all_features 
                        if not any(kw in f for kw in keywords)]
    print(f"  Features: {len(ablated_features)} (removed {len(all_features) - len(ablated_features)})")
    y_pred, phase_metrics, rmse, mae, r2 = train_phase_model(df, ablated_features)
    print(f"  RMSE={rmse:.4f} (Δ={rmse - rmse_full:+.4f}), "
          f"MAE={mae:.4f} (Δ={mae - mae_full:+.4f}), "
          f"R2={r2:.4f} (Δ={r2 - r2_full:+.4f})")
    
    # 分阶段
    phase_deltas = {}
    for pid in [0, 1, 2]:
        if pid in phase_metrics and pid in phase_full:
            phase_deltas[pid] = phase_metrics[pid]['rmse'] - phase_full[pid]['rmse']
    
    results[abl_key] = {
        'def': abl_def,
        'rmse_full': rmse_full, 'rmse_ablated': rmse,
        'mae_full': mae_full, 'mae_ablated': mae,
        'r2_full': r2_full, 'r2_ablated': r2,
        'phase_metrics': phase_metrics,
        'phase_full': phase_full,
        'phase_deltas': phase_deltas,
        'y_pred': y_pred,
    }

# ============================================================
# 图表 1: 综合对比柱状图 (RMSE / R2 / 分阶段)
# ============================================================
abl_keys = list(ablation_defs.keys())
abl_labels = [ablation_defs[k]['name_cn'] for k in abl_keys]
abl_colors = [ablation_defs[k]['color'] for k in abl_keys]

fig, axes = plt.subplots(2, 3, figsize=(18, 11))

# (a) 全量对比: RMSE
ax = axes[0, 0]
rmse_deltas = [results[k]['rmse_ablated'] - rmse_full for k in abl_keys]
bars = ax.bar(range(len(abl_keys)), rmse_deltas, color=abl_colors, edgecolor='white', linewidth=1.2)
for i, (bar, delta) in enumerate(zip(bars, rmse_deltas)):
    ax.text(bar.get_x() + bar.get_width()/2., delta + 0.0005 if delta >= 0 else delta - 0.0008,
            f'{delta:+.4f}', ha='center', fontsize=10, fontweight='bold')
ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_xticks(range(len(abl_keys)))
ax.set_xticklabels(abl_labels, fontsize=10)
ax.set_ylabel('Δ RMSE (mm)')
ax.set_title(f'Overall RMSE Change\n(Baseline RMSE = {rmse_full:.4f} mm)', fontsize=11, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# (b) 全量对比: R²
ax = axes[0, 1]
r2_deltas = [results[k]['r2_ablated'] - r2_full for k in abl_keys]
bars = ax.bar(range(len(abl_keys)), r2_deltas, color=abl_colors, edgecolor='white', linewidth=1.2)
for i, (bar, delta) in enumerate(zip(bars, r2_deltas)):
    ax.text(bar.get_x() + bar.get_width()/2., delta + 0.0003 if delta >= 0 else delta - 0.001,
            f'{delta:.4f}', ha='center', fontsize=10, fontweight='bold')
ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_xticks(range(len(abl_keys)))
ax.set_xticklabels(abl_labels, fontsize=10)
ax.set_ylabel('Δ R²')
ax.set_title(f'Overall R² Change\n(Baseline R² = {r2_full:.4f})', fontsize=11, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# (c) RMSE 绝对数值对比
ax = axes[0, 2]
x_pos = np.arange(len(abl_keys))
width = 0.35
bars_full = ax.bar(x_pos - width/2, [rmse_full]*len(abl_keys), width, 
                    label=f'Full ({len(all_features)} features)', color='#34495e', edgecolor='white')
bars_ab = ax.bar(x_pos + width/2, [results[k]['rmse_ablated'] for k in abl_keys], width,
                 label='Ablated', color='#e74c3c', edgecolor='white')
ax.set_xticks(x_pos)
ax.set_xticklabels(abl_labels, fontsize=9)
ax.set_ylabel('RMSE (mm)')
ax.set_title('RMSE: Full vs Ablated', fontsize=11, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(axis='y', alpha=0.3)

# (d) 分阶段 RMSE 变化 (Slow)
ax = axes[1, 0]
phase0_deltas = [results[k]['phase_deltas'].get(0, 0) for k in abl_keys]
bars = ax.bar(range(len(abl_keys)), phase0_deltas, color=abl_colors, edgecolor='white', linewidth=1.2)
for i, (bar, delta) in enumerate(zip(bars, phase0_deltas)):
    ax.text(bar.get_x() + bar.get_width()/2., delta + 0.0003 if delta >= 0 else delta - 0.0005,
            f'{delta:+.4f}', ha='center', fontsize=9, fontweight='bold')
ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_xticks(range(len(abl_keys)))
ax.set_xticklabels(abl_labels, fontsize=9)
ax.set_ylabel('Δ RMSE (mm)')
ax.set_title(f'Slow Phase\n(Baseline RMSE = {phase_full[0]["rmse"]:.4f})', fontsize=10, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# (e) 分阶段 RMSE 变化 (Accel)
ax = axes[1, 1]
phase1_deltas = [results[k]['phase_deltas'].get(1, 0) for k in abl_keys]
bars = ax.bar(range(len(abl_keys)), phase1_deltas, color=abl_colors, edgecolor='white', linewidth=1.2)
for i, (bar, delta) in enumerate(zip(bars, phase1_deltas)):
    ax.text(bar.get_x() + bar.get_width()/2., delta + 0.001 if delta >= 0 else delta - 0.002,
            f'{delta:+.4f}', ha='center', fontsize=9, fontweight='bold')
ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_xticks(range(len(abl_keys)))
ax.set_xticklabels(abl_labels, fontsize=9)
ax.set_ylabel('Δ RMSE (mm)')
ax.set_title(f'Accelerating Phase  ⚠\n(Baseline RMSE = {phase_full[1]["rmse"]:.4f})', fontsize=10, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# (f) 分阶段 RMSE 变化 (Fast)
ax = axes[1, 2]
phase2_deltas = [results[k]['phase_deltas'].get(2, 0) for k in abl_keys]
bars = ax.bar(range(len(abl_keys)), phase2_deltas, color=abl_colors, edgecolor='white', linewidth=1.2)
for i, (bar, delta) in enumerate(zip(bars, phase2_deltas)):
    ax.text(bar.get_x() + bar.get_width()/2., delta + 0.0005 if delta >= 0 else delta - 0.001,
            f'{delta:+.4f}', ha='center', fontsize=9, fontweight='bold')
ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_xticks(range(len(abl_keys)))
ax.set_xticklabels(abl_labels, fontsize=9)
ax.set_ylabel('Δ RMSE (mm)')
ax.set_title(f'Fast Phase\n(Baseline RMSE = {phase_full[2]["rmse"]:.4f})', fontsize=10, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

fig.suptitle('Q5.1 Four-Dimension Ablation Comparison\n'
             f'Full Model: {len(all_features)} features, RMSE={rmse_full:.4f}, MAE={mae_full:.4f}, R²={r2_full:.4f}',
             fontsize=13, fontweight='bold', y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(script_dir, 'comparison_overview.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\n  [Chart] comparison_overview.png saved")

# ============================================================
# 图表 2: 汇总热力图 / 表格
# ============================================================
fig2, ax = plt.subplots(figsize=(14, 5))
ax.axis('off')

col_labels = ['维度', '全变量 RMSE', '消融后 RMSE', 'Δ RMSE', '增幅 %',
              'Slow Δ', 'Accel Δ', 'Fast Δ', '全变量 R²', '消融后 R²', 'Δ R²']
table_data = []
for k in abl_keys:
    r = results[k]
    d = ablation_defs[k]
    row = [
        d['name_cn'],
        f'{rmse_full:.4f}',
        f'{r["rmse_ablated"]:.4f}',
        f'{r["rmse_ablated"] - rmse_full:+.4f}',
        f'{(r["rmse_ablated"]/rmse_full - 1)*100:+.1f}%',
        f'{r["phase_deltas"].get(0, 0):+.4f}',
        f'{r["phase_deltas"].get(1, 0):+.4f}',
        f'{r["phase_deltas"].get(2, 0):+.4f}',
        f'{r2_full:.4f}',
        f'{r["r2_ablated"]:.4f}',
        f'{r["r2_ablated"] - r2_full:+.4f}',
    ]
    table_data.append(row)

table = ax.table(cellText=table_data, colLabels=col_labels,
                 cellLoc='center', loc='center',
                 colColours=['#2c3e50']*11)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 1.8)

# 高亮增幅最大的一行
for i, k in enumerate(abl_keys):
    color = ablation_defs[k]['color']
    for j in range(len(col_labels)):
        cell = table[i+1, j]
        cell.set_facecolor(color + '20')  # 浅色背景
        if j == 4:  # 增幅列加粗
            cell.set_text_props(weight='bold')

# 表头样式
for j in range(len(col_labels)):
    table[0, j].set_text_props(color='white', weight='bold')
    table[0, j].set_facecolor('#2c3e50')

ax.set_title('Q5.1 Ablation Study: Summary Table\n'
             f'Full Model: {len(all_features)} features | Samples: {len(df)} | '
             f'Phases: Slow(0-{b1-1}) Accel({b1}-{b2-1}) Fast({b2}-{len(df)-1})',
             fontsize=11, fontweight='bold', pad=20)

fig2.tight_layout()
fig2.savefig(os.path.join(script_dir, 'comparison_table.png'), dpi=150, bbox_inches='tight')
plt.close(fig2)
print(f"  [Chart] comparison_table.png saved")

# ============================================================
# 图表 3: RMSE 增幅排序 + 阶段热力
# ============================================================
fig3, axes3 = plt.subplots(1, 2, figsize=(16, 6))

# 左: 按总体 RMSE 增幅排序
ax = axes3[0]
sorted_keys = sorted(abl_keys, key=lambda k: results[k]['rmse_ablated'] - rmse_full)
sorted_labels = [ablation_defs[k]['name_cn'] for k in sorted_keys]
sorted_colors = [ablation_defs[k]['color'] for k in sorted_keys]
sorted_deltas = [results[k]['rmse_ablated'] - rmse_full for k in sorted_keys]
sorted_pct = [(results[k]['rmse_ablated']/rmse_full - 1)*100 for k in sorted_keys]

bars = ax.barh(range(len(sorted_keys)), sorted_deltas, color=sorted_colors, edgecolor='white', linewidth=1.5)
for i, (bar, delta, pct) in enumerate(zip(bars, sorted_deltas, sorted_pct)):
    ax.text(bar.get_width() + 0.0003, bar.get_y() + bar.get_height()/2.,
            f'+{delta:.4f} mm ({pct:+.1f}%)', va='center', fontsize=11, fontweight='bold')
ax.axvline(x=0, color='black', linewidth=1)
ax.set_yticks(range(len(sorted_keys)))
ax.set_yticklabels(sorted_labels, fontsize=11)
ax.set_xlabel('Δ RMSE (mm)')
ax.set_title('Overall RMSE Increase After Ablation\n'
             f'(sorted by impact)', fontsize=12, fontweight='bold')
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

# 右: 阶段热力矩阵
ax = axes3[1]
phase_names = ['Slow', 'Accel', 'Fast']
phase_matrix = np.zeros((len(abl_keys), 3))
for i, k in enumerate(abl_keys):
    for j, pid in enumerate([0, 1, 2]):
        phase_matrix[i, j] = results[k]['phase_deltas'].get(pid, 0)

im = ax.imshow(phase_matrix, cmap='RdYlGn_r', aspect='auto', vmin=-0.01, vmax=phase_matrix.max()*1.1)
ax.set_xticks(range(3))
ax.set_xticklabels(phase_names, fontsize=11)
ax.set_yticks(range(len(abl_keys)))
ax.set_yticklabels([ablation_defs[k]['name_cn'] for k in abl_keys], fontsize=11)
for i in range(len(abl_keys)):
    for j in range(3):
        val = phase_matrix[i, j]
        color = 'white' if abs(val) > 0.02 else 'black'
        ax.text(j, i, f'{val:+.4f}', ha='center', va='center', fontsize=10,
                fontweight='bold', color=color)
cbar = plt.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label('Δ RMSE (mm)', fontsize=10)
ax.set_title('Per-Phase RMSE Change Heatmap\n'
             f'(Baseline: Slow={phase_full[0]["rmse"]:.4f} | '
             f'Accel={phase_full[1]["rmse"]:.4f} | Fast={phase_full[2]["rmse"]:.4f})',
             fontsize=11, fontweight='bold')

fig3.suptitle('Q5.1 Ablation Impact Ranking & Phase Sensitivity',
              fontsize=13, fontweight='bold', y=0.99)
fig3.tight_layout(rect=[0, 0, 1, 0.96])
fig3.savefig(os.path.join(script_dir, 'comparison_ranking.png'), dpi=150, bbox_inches='tight')
plt.close(fig3)
print(f"  [Chart] comparison_ranking.png saved")

# ============================================================
# 终端输出汇总
# ============================================================
print("\n" + "=" * 72)
print("  FINAL SUMMARY")
print("=" * 72)
print(f"\nBaseline (Full Model): RMSE={rmse_full:.4f}, MAE={mae_full:.4f}, R2={r2_full:.4f}")
print(f"\n{'维度':<12} {'消融后RMSE':>11} {'Δ RMSE':>10} {'增幅 %':>9} {'消融后R2':>9} {'Δ R2':>9}")
print("-" * 65)
for k in abl_keys:
    r = results[k]
    d = ablation_defs[k]
    print(f"{d['name_cn']:<12} {r['rmse_ablated']:>11.4f} "
          f"{r['rmse_ablated'] - rmse_full:>+10.4f} "
          f"{(r['rmse_ablated']/rmse_full - 1)*100:>+8.1f}% "
          f"{r['r2_ablated']:>9.4f} {r['r2_ablated'] - r2_full:>+9.4f}")

print(f"\n{'Phase':<8}", end='')
for k in abl_keys:
    print(f"{ablation_defs[k]['name_cn']:>12}", end=' ')
print(f"{'(Baseline)':>12}")
for phase_name, pid in [('Slow', 0), ('Accel', 1), ('Fast', 2)]:
    print(f"{phase_name:<8}", end='')
    bl_rmse = phase_full[pid]['rmse']
    for k in abl_keys:
        delta = results[k]['phase_deltas'].get(pid, 0)
        print(f"{delta:>+12.4f}", end=' ')
    print(f"({bl_rmse:.4f})")

print(f"\n  [DONE] All charts saved to {script_dir}")
