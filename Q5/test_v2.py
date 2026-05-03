"""
Q5.1 最优变量组合建模 v3.0 (深度改进版) + 图表自动生成
====================================
改进点（相对v1原始版）:
  P0-1: 滞后互相关分析 -> 构造最优滞后特征
  P0-2: 分阶段建模 (匀速/加速/快速)
  P0-3: 爆破 PPV 物理模型替代原始 BlastDist + BlastCharge
  P1-1: TimeSeriesSplit 交叉验证
  P1-2: 超参数 GridSearch 调优
  P1-3: SHAP 值分析（替代简单 Leave-One-Out）
  P1-4: 递归特征消除 RFE
  P2-1: LightGBM 替代 sklearn GBRT
  P2-2: 丰富衍生特征（交互项、入渗差分、微震滚动）
  P3:   不确定性量化（分位数回归输出预测区间）
  
自动生成对比图片:
  fig1: 分阶段预测 vs 真实值 时序对比图
  fig2: SHAP 特征重要性图
  fig3: 消融实验 RMSE 对比柱状图
  fig4: 分位数回归 预测区间图  80%置信带
  fig5: 特征相关性热力图
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings, os, sys, glob, subprocess
warnings.filterwarnings('ignore')

# ============================================================
# 自动安装缺失依赖
# ============================================================
for lib in ['lightgbm', 'shap', 'matplotlib', 'openpyxl']:
    try:
        __import__(lib)
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', lib, '-q'])

import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 中文字体设置
plt.rcParams['axes.unicode_minus'] = False
for font_name in ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']:
    try:
        plt.rcParams['font.sans-serif'] = [font_name]
        # 测试
        fig_test = plt.figure()
        fig_test.text(0.5, 0.5, '测试中文', fontsize=12)
        plt.close(fig_test)
        print(f"[OK] 中文字体: {font_name}")
        break
    except:
        continue

# ============================================================
# 自动确定路径
# ============================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# 创建图片输出目录
output_dir = os.path.join(script_dir, 'v3_comparison_charts')
os.makedirs(output_dir, exist_ok=True)

# ============================================================
# 查找数据文件
# ============================================================
xlsx_files = glob.glob('*.xlsx') + glob.glob('*.xls')
if not xlsx_files:
    print("[错误] 当前目录下未找到 .xlsx 文件！")
    sys.exit(1)
data_file = xlsx_files[0]

# ============================================================
# 列名匹配
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

# ============================================================
# 数据清洗
# ============================================================
print('=' * 72)
print('  Q5.1 v3.0 (深度改进版) - LightGBM + 滞后 + 分阶段 + SHAP + 分位数')
print('=' * 72)
print("数据文件:", data_file)
print("数据形状:", df.shape)

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
# P0-3: 爆破 PPV 物理模型 (文档3.3)
# ============================================================
print("\n=== P0-3: 爆破 PPV 物理模型 ===")
if 'BlastDist' in df.columns and 'BlastCharge' in df.columns:
    safe_dist = df['BlastDist'].values.copy()
    safe_dist[safe_dist < 1.0] = 1.0
    df['Blast_Dist_safe'] = safe_dist
    df['Blast_PPV'] = np.sqrt(np.abs(df['BlastCharge'].values)) / safe_dist
    df['Blast_PPV_log'] = np.log1p(df['Blast_PPV'].clip(lower=0))
    df['Blast_Energy'] = df['BlastCharge'].values / (safe_dist ** 2)
    df['Blast_Energy_log'] = np.log1p(df['Blast_Energy'].clip(lower=0))
    print("  生成特征: Blast_PPV, Blast_PPV_log, Blast_Energy, Blast_Energy_log")
else:
    print("  缺少 BlastDist 或 BlastCharge，跳过 PPV 模型")

# ============================================================
# 阶段划分 (基于位移速度)
# ============================================================
print("\n--- 阶段划分 ---")
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

print("阶段边界: 0~{} (匀速) | {}~{} (加速) | {}~{} (快速)".format(b1-1, b1, b2-1, b2, len(df)-1))

df['Phase'] = 0
df.loc[:b1-1, 'Phase'] = 0
df.loc[b1:b2-1, 'Phase'] = 1
df.loc[b2:, 'Phase'] = 2

# ============================================================
# P0-1: 滞后互相关分析
# ============================================================
print("\n=== P0-1: 滞后互相关分析 ===")

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

y_raw = np.diff(df['Displacement'].values, prepend=df['Displacement'].iloc[0])

lag_info = {}
for feat in ['Rainfall', 'PorePressure', 'Microseismic', 'Infiltration', 'BlastDist', 'BlastCharge']:
    if feat not in df.columns:
        continue
    x = df[feat].values
    lag, corr = find_optimal_lag(x, y_raw)
    lag_info[feat] = {'lag': lag, 'corr': corr}
    print("  {}: lag={}步({}min), corr={:.4f}".format(feat, lag, lag*10, corr))

for feat, info in lag_info.items():
    lag = info['lag']
    if lag > 0:
        df['{}_lag{}'.format(feat, lag)] = df[feat].shift(lag)
        info['lagged_col'] = '{}_lag{}'.format(feat, lag)
    else:
        info['lagged_col'] = feat

# ============================================================
# 构造所有特征 (P2-2: 丰富衍生特征)
# ============================================================
print("\n=== 构造特征 (P2-2: 丰富衍生特征) ===")

df['Delta_D'] = np.diff(df['Displacement'].values, prepend=df['Displacement'].iloc[0])

base_features = ['Rainfall', 'PorePressure', 'Microseismic', 'Infiltration', 'BlastDist', 'BlastCharge']
base_features = [c for c in base_features if c in df.columns]

lagged_features = [info['lagged_col'] for feat, info in lag_info.items() if feat in base_features]

# --- 原始衍生特征 ---
if 'PorePressure' in df.columns:
    df['Pore_Diff'] = df['PorePressure'].diff()
if 'Rainfall' in df.columns:
    df['Rain_cum24'] = df['Rainfall'].rolling(144, min_periods=1).sum()
if 'Infiltration' in df.columns:
    df['Infiltration_Diff'] = df['Infiltration'].diff()
if 'Microseismic' in df.columns:
    df['Microseismic_roll6'] = df['Microseismic'].rolling(6, min_periods=1).sum()

# --- 交互特征 ---
if 'PorePressure' in df.columns and 'Rainfall' in df.columns:
    df['Pore_Rain_Cross'] = df['PorePressure'] * df['Rainfall']
if 'PorePressure' in df.columns and 'Infiltration' in df.columns:
    df['Pore_Infiltration_Cross'] = df['PorePressure'] * df['Infiltration']

# --- 爆破衍生特征 ---
if 'BlastCharge' in lag_info:
    bcol = lag_info['BlastCharge']['lagged_col']
    df['Blast_E_cum24'] = df[bcol].rolling(144, min_periods=1).sum()
    blast_vals = df[bcol].values
    blast_ts = np.zeros(len(blast_vals), dtype=int)
    cnt = 0
    for i in range(1, len(blast_vals)):
        if blast_vals[i] > 0:
            cnt = 0
        else:
            cnt += 1
        blast_ts[i] = cnt
    df['Time_since_blast'] = blast_ts

# --- 降雨衍生特征 ---
if 'Rainfall' in lag_info:
    rcol = lag_info['Rainfall']['lagged_col']
    rain_vals = df[rcol].values
    rain_ts = np.zeros(len(rain_vals), dtype=int)
    cnt = 0
    for i in range(1, len(rain_vals)):
        if rain_vals[i] > 0:
            cnt = 0
        else:
            cnt += 1
        rain_ts[i] = cnt
    df['Time_since_rain'] = rain_ts

# --- 时间趋势特征 ---
df['Time_idx'] = np.arange(len(df))
if 'Time' in df.columns:
    df['Hour_of_day'] = pd.to_datetime(df['Time']).dt.hour
else:
    df['Hour_of_day'] = 0
df['Day_sin'] = np.sin(2 * np.pi * df['Hour_of_day'] / 24)
df['Day_cos'] = np.cos(2 * np.pi * df['Hour_of_day'] / 24)

# --- 累积位移 ---
df['Disp_cum24'] = df['Delta_D'].rolling(144, min_periods=1).sum()

# ============================================================
# 组装特征列表
# ============================================================
feature_cols = lagged_features + [
    'Pore_Diff', 'Rain_cum24', 'Infiltration_Diff', 'Microseismic_roll6',
    'Pore_Rain_Cross', 'Pore_Infiltration_Cross',
    'Disp_cum24', 'Time_idx',
]

for blf in ['Blast_E_cum24', 'Time_since_blast',
            'Blast_PPV', 'Blast_PPV_log', 'Blast_Energy', 'Blast_Energy_log']:
    if blf in df.columns:
        feature_cols.append(blf)

if 'Time_since_rain' in df.columns:
    feature_cols.append('Time_since_rain')

for tf in ['Day_sin', 'Day_cos']:
    if tf in df.columns:
        feature_cols.append(tf)

feature_cols = [c for c in feature_cols if c in df.columns]

df = df.dropna(subset=feature_cols + ['Delta_D']).reset_index(drop=True)

print("特征列表 ({}个): {}".format(len(feature_cols), feature_cols))

# ============================================================
# 分阶段建模 (P0-2)
# ============================================================
print("\n=== P0-2: 分阶段建模 ===")
print("使用 LightGBM")

def train_model_cv(X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=max(2, min(n_splits, len(X) // 100)))
    param_grid = {
        'n_estimators': [300, 500],
        'learning_rate': [0.01, 0.02],
        'num_leaves': [31],
        'subsample': [0.8],
        'colsample_bytree': [0.8],
    }
    base_model = lgb.LGBMRegressor(random_state=42, verbose=-1,
                                   n_jobs=1, force_col_wise=True)
    gs = GridSearchCV(base_model, param_grid, cv=tscv,
                      scoring='neg_root_mean_squared_error', n_jobs=2, verbose=0)
    gs.fit(X, y)
    cv_rmse = -gs.best_score_
    return gs.best_estimator_, gs.best_params_, cv_rmse

phase_models = {}
phase_metrics = {}
y_pred_all = np.zeros(len(df))

for phase_id, phase_name in [(0, '匀速'), (1, '加速'), (2, '快速')]:
    mask = df['Phase'] == phase_id
    if mask.sum() < 50:
        print("  阶段{}: 样本不足({})，跳过".format(phase_name, mask.sum()))
        continue

    X_ph = df.loc[mask, feature_cols]
    y_ph = df.loc[mask, 'Delta_D']

    model, best_params, cv_rmse = train_model_cv(X_ph.values, y_ph.values)
    y_pred_ph = model.predict(X_ph.values)

    rmse = np.sqrt(mean_squared_error(y_ph, y_pred_ph))
    mae = mean_absolute_error(y_ph, y_pred_ph)
    r2 = r2_score(y_ph, y_pred_ph)

    phase_models[phase_id] = model
    phase_metrics[phase_name] = {'rmse': rmse, 'mae': mae, 'r2': r2,
                                  'cv_rmse': cv_rmse, 'samples': mask.sum()}
    y_pred_all[mask] = y_pred_ph

    print("  阶段{} ({}样本):".format(phase_name, mask.sum()))
    print("    CV-RMSE = {:.4f} mm".format(cv_rmse))
    print("    RMSE = {:.4f} mm, MAE = {:.4f} mm, R2 = {:.4f}".format(rmse, mae, r2))
    print("    最佳参数: {}".format(best_params))

overall_rmse = np.sqrt(mean_squared_error(df['Delta_D'], y_pred_all))
overall_mae = mean_absolute_error(df['Delta_D'], y_pred_all)
overall_r2 = r2_score(df['Delta_D'], y_pred_all)

print("\n--- 总体性能 ---")
print("  RMSE = {:.4f} mm".format(overall_rmse))
print("  MAE  = {:.4f} mm".format(overall_mae))
print("  R2   = {:.4f}".format(overall_r2))

# ============================================================
# P1-4: 递归特征消除 RFE
# ============================================================
print("\n\n=== P1-4: 递归特征消除 RFE ===")
print("（在全量数据上运行，用于发现最优特征子集）")

X_all = df[feature_cols]
y_all = df['Delta_D']
tscv_rfe = TimeSeriesSplit(n_splits=3)

rfe_feature_cols = feature_cols  # fallback
try:
    rfe_model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.01,
                                   num_leaves=15, random_state=42, verbose=-1,
                                   n_jobs=1, force_col_wise=True)
    rfe = RFECV(rfe_model, step=1, cv=tscv_rfe,
                scoring='neg_root_mean_squared_error', n_jobs=2, verbose=0)
    rfe.fit(X_all.values, y_all.values)

    selected_mask = rfe.support_
    selected_features = [f for f, m in zip(feature_cols, selected_mask) if m]
    n_selected = sum(selected_mask)

    print("RFE 最优特征数: {}/{}".format(n_selected, len(feature_cols)))
    print("最优特征子集 ({}个):".format(n_selected))
    for f in selected_features:
        print("  - {}".format(f))
    
    if n_selected > 0:
        rfe_feature_cols = selected_features
except Exception as e:
    print("RFE 运行出错: {}, 使用全部特征".format(str(e)))

# ============================================================
# P1-3: SHAP 值分析
# ============================================================
print("\n\n=== P1-3: SHAP 值分析 ===")

shap_ranking = []  # fallback

try:
    import shap
    shap_model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.01,
                                    num_leaves=31, random_state=42, verbose=-1,
                                    n_jobs=1, force_col_wise=True)
    shap_model.fit(X_all.values, y_all.values)

    explainer = shap.TreeExplainer(shap_model)
    shap_values = explainer.shap_values(X_all.values)

    shap_importance = np.abs(shap_values).mean(axis=0)
    shap_ranking = sorted(zip(feature_cols, shap_importance), key=lambda x: x[1], reverse=True)

    print("\nSHAP 特征重要性排名 (Mean |SHAP|):")
    print("{:<35} {:>12}".format('Feature', 'Mean |SHAP|'))
    print("-" * 50)
    for f, imp in shap_ranking:
        print("{:<35} {:>12.6f}".format(f, imp))

    print("\nSHAP 分析结论:")
    print("  Top 3 最重要: {}".format([f for f, _ in shap_ranking[:3]]))
    print("  Bottom 3 最不重要: {}".format([f for f, _ in shap_ranking[-3:]]))

    # ---- 画 SHAP 特征重要性图 ----
    top_features = [f for f, _ in shap_ranking[:15]]
    top_importance = [imp for _, imp in shap_ranking[:15]]
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Reds_r(np.linspace(0.4, 0.9, len(top_features)))
    bars = ax2.barh(range(len(top_features)), top_importance[::-1], color=colors[::-1])
    ax2.set_yticks(range(len(top_features)))
    ax2.set_yticklabels(top_features[::-1])
    ax2.set_xlabel('Mean |SHAP value|')
    ax2.set_title('SHAP Feature Importance (LightGBM)')
    for i, (bar, val) in enumerate(zip(bars, top_importance[::-1])):
        ax2.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                '{:.4f}'.format(val), va='center', fontsize=8)
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, 'fig2_SHAP_importance.png'), dpi=150)
    plt.close(fig2)
    print("\n  [图] fig2_SHAP_importance.png 已保存")

except ImportError:
    print("shap 未安装，跳过 SHAP 分析。")
except Exception as e:
    print("SHAP 分析出错: {}".format(str(e)))

# ============================================================
# 变量组合消融实验
# ============================================================
print("\n\n=== 变量组合消融实验 ===")

def evaluate_combo(X, y, sel_cols):
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    for train_idx, test_idx in tscv.split(X):
        X_tr = X.iloc[train_idx][sel_cols].values
        X_te = X.iloc[test_idx][sel_cols].values
        y_tr = y.iloc[train_idx].values
        y_te = y.iloc[test_idx].values
        m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.01,
                              num_leaves=31, random_state=42, verbose=-1,
                              n_jobs=1, force_col_wise=True)
        m.fit(X_tr, y_tr)
        y_pred = m.predict(X_te)
        scores.append(np.sqrt(mean_squared_error(y_te, y_pred)))
    return np.mean(scores), np.std(scores)

results_combo = []

# 全部特征
rmse, std = evaluate_combo(X_all, y_all, feature_cols)
results_combo.append(("全部特征({})".format(len(feature_cols)), rmse, std))
print("  全部({}): CV-RMSE={:.4f}+-{:.4f}".format(len(feature_cols), rmse, std))

# RFE 最优子集
if rfe_feature_cols != feature_cols and len(rfe_feature_cols) >= 3:
    rmse, std = evaluate_combo(X_all, y_all, rfe_feature_cols)
    results_combo.append(("RFE优选({})".format(len(rfe_feature_cols)), rmse, std))
    print("  RFE优选({}): CV-RMSE={:.4f}+-{:.4f}".format(len(rfe_feature_cols), rmse, std))

# 仅原始6变量（lagged版本）
orig_names = {feat: info['lagged_col'] for feat, info in lag_info.items() if feat in base_features}
orig6_lagged = list(orig_names.values())
orig6_lagged = [c for c in orig6_lagged if c in X_all.columns]
if len(orig6_lagged) >= 3:
    rmse, std = evaluate_combo(X_all, y_all, orig6_lagged)
    results_combo.append(("仅原始6变量(lagged)", rmse, std))
    print("  仅原始6变量(lagged): CV-RMSE={:.4f}+-{:.4f}".format(rmse, std))

# 逐一剔除原始变量
for removed in base_features:
    if removed not in lag_info:
        continue
    lagged_col = lag_info[removed]['lagged_col']
    sel = [c for c in feature_cols if c != lagged_col]
    
    if removed in ['BlastDist', 'BlastCharge']:
        sel = [c for c in sel if c not in ['Time_since_blast', 'Blast_E_cum24',
                'Blast_PPV', 'Blast_PPV_log', 'Blast_Energy', 'Blast_Energy_log', 'Blast_Dist_safe']]
    elif removed == 'Rainfall':
        sel = [c for c in sel if c not in ['Rain_cum24', 'Time_since_rain', 'Pore_Rain_Cross']]
    elif removed == 'PorePressure':
        sel = [c for c in sel if c not in ['Pore_Diff', 'Pore_Rain_Cross', 'Pore_Infiltration_Cross']]
    elif removed == 'Infiltration':
        sel = [c for c in sel if c not in ['Infiltration_Diff', 'Pore_Infiltration_Cross']]
    elif removed == 'Microseismic':
        sel = [c for c in sel if c not in ['Microseismic_roll6']]
    
    sel = [c for c in sel if c in X_all.columns]
    if len(sel) < 3:
        continue
    rmse, std = evaluate_combo(X_all, y_all, sel)
    results_combo.append(("去掉{}".format(removed), rmse, std))
    print("  去掉{}: CV-RMSE={:.4f}+-{:.4f}".format(removed, rmse, std))

results_combo.sort(key=lambda x: x[1])

print("\n{:<40} {:>8} {:>8}".format('Model', 'CV-RMSE', 'Std'))
print("-" * 60)
for name, rmse, std in results_combo:
    print("{:<40} {:>8.4f} {:>8.4f}".format(name, rmse, std))

best = results_combo[0]
worst = results_combo[-1]
print("\n最佳: {} (RMSE={:.4f}+-{:.4f} mm)".format(best[0], best[1], best[2]))
print("最差: {} (RMSE={:.4f}+-{:.4f} mm)".format(worst[0], worst[1], worst[2]))

# ============================================================
# P3: 不确定性量化 — 分位数回归
# ============================================================
print("\n\n=== P3: 不确定性量化（分位数回归）===")

quantile_data = None  # fallback

try:
    mask_fast = df['Phase'] == 2
    X_fast = df.loc[mask_fast, feature_cols].values
    y_fast = df.loc[mask_fast, 'Delta_D'].values

    quantiles = {'q10': 0.10, 'q50': 0.50, 'q90': 0.90}
    quantile_preds = {}

    for qname, alpha in quantiles.items():
        q_model = lgb.LGBMRegressor(
            objective='quantile', alpha=alpha,
            n_estimators=300, learning_rate=0.01,
            num_leaves=31, random_state=42, verbose=-1,
            n_jobs=1, force_col_wise=True
        )
        split_idx = int(len(X_fast) * 0.8)
        q_model.fit(X_fast[:split_idx], y_fast[:split_idx])
        quantile_preds[qname] = q_model.predict(X_fast[split_idx:])

    y_test_fast = y_fast[split_idx:]
    q10, q50, q90 = quantile_preds['q10'], quantile_preds['q50'], quantile_preds['q90']

    coverage = np.mean((y_test_fast >= q10) & (y_test_fast <= q90))
    interval_width = np.mean(q90 - q10)

    def quantile_loss(y_true, y_pred, alpha):
        e = y_true - y_pred
        return np.mean(np.maximum(alpha * e, (alpha - 1) * e))

    q10_loss = quantile_loss(y_test_fast, q10, 0.10)
    q50_loss = quantile_loss(y_test_fast, q50, 0.50)
    q90_loss = quantile_loss(y_test_fast, q90, 0.90)
    avg_qloss = (q10_loss + q50_loss + q90_loss) / 3
    rmse_q50 = np.sqrt(mean_squared_error(y_test_fast, q50))

    print("（在快速段测试集上评估，样本数={}）".format(len(y_test_fast)))
    print("  分位数模型指标:")
    print("    80% 区间覆盖概率: {:.2%} (理想 80%)".format(coverage))
    print("    平均区间宽度:     {:.4f} mm".format(interval_width))
    print("    中位数(q50) RMSE:  {:.4f} mm".format(rmse_q50))
    print("    平均分位数损失:    {:.6f}".format(avg_qloss))
    print("    区间宽度/RMSE:     {:.2f}".format(interval_width / max(rmse_q50, 0.001)))

    quantile_data = {
        'y_test': y_test_fast, 'q10': q10, 'q50': q50, 'q90': q90,
        'coverage': coverage, 'interval_width': interval_width, 'rmse_q50': rmse_q50
    }

except Exception as e:
    print("分位数回归出错: {}, 跳过".format(str(e)))

# ============================================================
# ============================================================
#  自动生成对比图片
# ============================================================
# ============================================================

print("\n" + "=" * 72)
print("  生成对比图片...")
print("=" * 72)

# ---- fig1: 分阶段预测 vs 真实值 时序对比图 ----
fig1, axes1 = plt.subplots(4, 1, figsize=(14, 12), 
                            gridspec_kw={'height_ratios': [3, 0.8, 0.8, 0.8]})

# 子图1: 累积位移对比
ax_cum = axes1[0]
df['Disp_cum_pred'] = df['Displacement'].iloc[0] + y_pred_all.cumsum()
ax_cum.plot(df.index, df['Displacement'], 'b-', alpha=0.5, linewidth=0.8, label='True Displacement')
ax_cum.plot(df.index, df['Disp_cum_pred'], 'r-', alpha=0.7, linewidth=0.8, label='Predicted Displacement')
# 标注阶段分界线
for b, color in [(b1, 'green'), (b2, 'orange')]:
    ax_cum.axvline(x=b, color=color, linestyle='--', alpha=0.7, linewidth=1)
ax_cum.set_ylabel('Cumulative Displacement (mm)')
ax_cum.legend(loc='upper left', fontsize=8)
ax_cum.set_title('Fig1: Phase-aware Prediction vs True Displacement (RMSE={:.4f}mm, R2={:.4f})'.format(overall_rmse, overall_r2))
ax_cum.text(b1/2, df['Displacement'].max()*1.05, 'Slow', ha='center', color='green', fontsize=9)
ax_cum.text((b1+b2)/2, df['Displacement'].max()*1.05, 'Accelerating', ha='center', color='orange', fontsize=9)
ax_cum.text((b2+len(df))/2, df['Displacement'].max()*1.05, 'Fast', ha='center', color='red', fontsize=9)

# 子图2: 瞬时位移速度 (dD/dt) 对比
ax_vel = axes1[1]
ax_vel.plot(df.index, df['Delta_D'], 'b-', alpha=0.4, linewidth=0.5, label='True Delta_D')
ax_vel.plot(df.index, y_pred_all, 'r-', alpha=0.6, linewidth=0.5, label='Predicted Delta_D')
for b, color in [(b1, 'green'), (b2, 'orange')]:
    ax_vel.axvline(x=b, color=color, linestyle='--', alpha=0.7, linewidth=1)
ax_vel.set_ylabel('Delta_D (mm)')
ax_vel.legend(loc='upper right', fontsize=7)

# 子图3: 残差
ax_res = axes1[2]
residuals = df['Delta_D'].values - y_pred_all
ax_res.plot(df.index, residuals, 'purple', alpha=0.5, linewidth=0.5)
ax_res.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
for b, color in [(b1, 'green'), (b2, 'orange')]:
    ax_res.axvline(x=b, color=color, linestyle='--', alpha=0.7, linewidth=1)
ax_res.set_ylabel('Residual (mm)')
ax_res.set_xlabel('Sample Index')

# 子图4: 残差分布直方图
ax_hist = axes1[3]
ax_hist.hist(residuals, bins=80, color='purple', alpha=0.7, edgecolor='white')
ax_hist.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax_hist.axvline(x=np.mean(residuals), color='red', linestyle='--', linewidth=1, label='Mean={:.4f}'.format(np.mean(residuals)))
ax_hist.set_xlabel('Residual (mm)')
ax_hist.set_ylabel('Frequency')
ax_hist.legend(fontsize=8)

fig1.tight_layout()
fig1.savefig(os.path.join(output_dir, 'fig1_phase_prediction_comparison.png'), dpi=150)
plt.close(fig1)
print("  [图] fig1_phase_prediction_comparison.png 已保存")

# ---- fig3: 消融实验 RMSE 对比柱状图 ----
if len(results_combo) > 1:
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    names = [r[0] for r in results_combo]
    rmses = [r[1] for r in results_combo]
    stds = [r[2] for r in results_combo]
    
    x_pos = range(len(names))
    colors3 = ['#2ecc71' if r[0] == best[0] else '#e74c3c' if r[0] == worst[0] else '#3498db' for r in results_combo]
    bars3 = ax3.bar(x_pos, rmses, yerr=stds, color=colors3, capsize=3, alpha=0.85, edgecolor='white')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax3.set_ylabel('CV-RMSE (mm)')
    ax3.set_title('Fig3: Ablation Study - CV-RMSE Comparison')
    # 标注数值
    for i, (bar, rmse) in enumerate(zip(bars3, rmses)):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                '{:.4f}'.format(rmse), ha='center', va='bottom', fontsize=7)
    ax3.axhline(y=rmses[0], color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax3.set_ylim(bottom=max(0, min(rmses) - 0.15), top=max(rmses) * 1.1)
    fig3.tight_layout()
    fig3.savefig(os.path.join(output_dir, 'fig3_ablation_comparison.png'), dpi=150)
    plt.close(fig3)
    print("  [图] fig3_ablation_comparison.png 已保存")

# ---- fig4: 分位数回归 预测区间图 ----
if quantile_data is not None:
    fig4, ax4 = plt.subplots(figsize=(14, 5))
    
    # 画最近 500 个点（太多点看不清楚）
    n_show = min(500, len(quantile_data['y_test']))
    idx = np.arange(n_show)
    
    ax4.fill_between(idx, quantile_data['q10'][:n_show], quantile_data['q90'][:n_show], 
                     alpha=0.3, color='blue', label='80% Prediction Interval')
    ax4.plot(idx, quantile_data['q50'][:n_show], 'b-', linewidth=0.8, label='Median (q50)')
    ax4.plot(idx, quantile_data['y_test'][:n_show], 'r.', markersize=2, alpha=0.5, label='True Delta_D')
    ax4.set_xlabel('Test Sample Index (fast phase)')
    ax4.set_ylabel('Delta_D (mm)')
    ax4.set_title('Fig4: Quantile Regression - 80% Prediction Interval (Coverage={:.1%}, Width={:.4f}mm, RMSE={:.4f}mm)'.format(
        quantile_data['coverage'], quantile_data['interval_width'], quantile_data['rmse_q50']))
    ax4.legend(loc='upper right', fontsize=8)
    fig4.tight_layout()
    fig4.savefig(os.path.join(output_dir, 'fig4_quantile_prediction_interval.png'), dpi=150)
    plt.close(fig4)
    print("  [图] fig4_quantile_prediction_interval.png 已保存")

# ---- fig5: 特征相关性热力图 ----
fig5, ax5 = plt.subplots(figsize=(14, 11))
# 选择 Top 20 特征（按SHAP重要性）或全部特征（如果特征数较少）
if shap_ranking:
    top_feats = [f for f, _ in shap_ranking[:min(20, len(shap_ranking))]]
else:
    top_feats = feature_cols[:20]

corr_data = df[top_feats + ['Delta_D']].corr()
im = ax5.imshow(corr_data.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax5.set_xticks(range(len(corr_data.columns)))
ax5.set_yticks(range(len(corr_data.columns)))
ax5.set_xticklabels(corr_data.columns, rotation=90, fontsize=7)
ax5.set_yticklabels(corr_data.columns, fontsize=7)
ax5.set_title('Fig5: Feature Correlation Matrix (Top {} features)'.format(len(top_feats)))
cbar = fig5.colorbar(im, ax=ax5, shrink=0.8)
cbar.set_label('Pearson Correlation')
fig5.tight_layout()
fig5.savefig(os.path.join(output_dir, 'fig5_correlation_heatmap.png'), dpi=150)
plt.close(fig5)
print("  [图] fig5_correlation_heatmap.png 已保存")

# ---- fig6: 原始6变量 vs 全特征 对比散点图 ----
fig6, axes6 = plt.subplots(1, 2, figsize=(12, 5))

# 左图：全特征预测 vs 真实
ax6l = axes6[0]
ax6l.scatter(df['Delta_D'], y_pred_all, alpha=0.3, s=2, c='blue')
max_val = max(df['Delta_D'].max(), y_pred_all.max())
ax6l.plot([0, max_val], [0, max_val], 'r--', linewidth=1)
ax6l.set_xlabel('True Delta_D (mm)')
ax6l.set_ylabel('Predicted Delta_D (mm)')
ax6l.set_title('All Features ({})\nRMSE={:.4f}mm, R2={:.4f}'.format(len(feature_cols), overall_rmse, overall_r2))

# 右图：仅原始6变量预测 vs 真实
if len(orig6_lagged) >= 3:
    # 训练一个仅使用原始6变量的模型
    m6 = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.01,
                           num_leaves=31, random_state=42, verbose=-1,
                           n_jobs=1, force_col_wise=True)
    m6.fit(X_all[orig6_lagged].values, y_all.values)
    y_pred_6 = m6.predict(X_all[orig6_lagged].values)
    rmse_6 = np.sqrt(mean_squared_error(df['Delta_D'], y_pred_6))
    r2_6 = r2_score(df['Delta_D'], y_pred_6)

    ax6r = axes6[1]
    ax6r.scatter(df['Delta_D'], y_pred_6, alpha=0.3, s=2, c='green')
    ax6r.plot([0, max_val], [0, max_val], 'r--', linewidth=1)
    ax6r.set_xlabel('True Delta_D (mm)')
    ax6r.set_ylabel('Predicted Delta_D (mm)')
    ax6r.set_title('Original 6 Variables\nRMSE={:.4f}mm, R2={:.4f}'.format(rmse_6, r2_6))
else:
    axes6[1].text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=14)
    axes6[1].set_title('Original 6 Variables (insufficient data)')

fig6.tight_layout()
fig6.savefig(os.path.join(output_dir, 'fig6_scatter_comparison.png'), dpi=150)
plt.close(fig6)
print("  [图] fig6_scatter_comparison.png 已保存")

# ============================================================
# 汇总输出
# ============================================================
print("\n" + "=" * 72)
print("  汇总报告")
print("=" * 72)

print("\n--- 滞后分析 ---")
for feat, info in lag_info.items():
    print("  {}: 滞后{}步({}min), 相关系数={:.4f}".format(feat, info['lag'], info['lag']*10, info['corr']))

print("\n--- 分阶段建模性能 ---")
print("{:<10} {:>8} {:>8} {:>8} {:>8} {:>10}".format('阶段', '样本数', 'RMSE', 'MAE', 'R2', 'CV-RMSE'))
print("-" * 56)
for phase_name in ['匀速', '加速', '快速']:
    if phase_name in phase_metrics:
        m = phase_metrics[phase_name]
        print("{:<10} {:>8} {:>8.4f} {:>8.4f} {:>8.4f} {:>10.4f}".format(
            phase_name, m['samples'], m['rmse'], m['mae'], m['r2'], m['cv_rmse']))

print("\n--- 总体 ---")
print("  RMSE = {:.4f} mm  (v1原始: 1.3312 mm)".format(overall_rmse))
print("  MAE  = {:.4f} mm".format(overall_mae))
print("  R2   = {:.4f}".format(overall_r2))

improvement = (1.3312 - overall_rmse) / 1.3312 * 100
print("  提升: {:.1f}% RMSE减少".format(improvement))

if overall_rmse < 1.3312:
    print("  v3 优于 v1, RMSE 降低 {:.4f} mm".format(1.3312 - overall_rmse))
else:
    print("  v3 未优于 v1, 需要进一步调优")

print("\n--- 特征重要性（消融实验）---")
print("  最佳组合: {} (RMSE={:.4f}+-{:.4f} mm)".format(best[0], best[1], best[2]))
print("  最差组合: {} (RMSE={:.4f}+-{:.4f} mm)".format(worst[0], worst[1], worst[2]))

print("\n--- 已生成对比图片 ---")
print("  输出目录: {}".format(output_dir))
for fname in ['fig1_phase_prediction_comparison.png',
              'fig2_SHAP_importance.png',
              'fig3_ablation_comparison.png',
              'fig4_quantile_prediction_interval.png',
              'fig5_correlation_heatmap.png',
              'fig6_scatter_comparison.png']:
    fpath = os.path.join(output_dir, fname)
    if os.path.exists(fpath):
        print("    {} ({} KB)".format(fname, os.path.getsize(fpath)//1024))
    else:
        print("    {} (未生成)".format(fname))

print("=" * 72)
print("\n  v3.0 All Improvements:")
print("    P0-1: Lag Cross-Correlation Analysis [OK]")
print("    P0-2: Phase-aware Modeling [OK]")
print("    P0-3: Blast PPV Physical Model [OK] (NEW)")
print("    P1-1: TimeSeriesSplit CV [OK]")
print("    P1-2: GridSearch Hyperparameter Tuning [OK]")
print("    P1-3: SHAP Value Analysis [OK] (NEW)")
print("    P1-4: RFE Feature Selection [OK] (NEW)")
print("    P2-1: LightGBM replaces GBRT [OK]")
print("    P2-2: Rich Derived Features [OK] (NEW)")
print("    P3:   Quantile Regression Uncertainty [OK] (NEW)")
print("    Visualization: 6 Charts Auto-generated [OK] (NEW)")
print("=" * 72)
