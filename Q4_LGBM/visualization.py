"""
Q4 LightGBM 增强可视化脚本
=============================================================
对标 Q4/4.5/regression.py + Q4/4.5/predict.py 的输出质量，
为 LightGBM 模型生成相同级别的诊断图表。

输出（14 张图，保存至 Q4_LGBM/）：
  - training_fit_overview.png       — 2×1 位移拟合总览 + 残差（阶段背景色块）
  - seg{1,2,3}/stage_SD_diagnosis.png   — 位移诊断 2×2
  - seg{1,2,3}/stage_delta_diagnosis.png  — 速度诊断 2×2
  - seg{1,2,3}/stage_coefficients.png     — 特征重要性横条图
  - prediction_overview.png          — 实验集三段链式拼接总览
  - seg{1,2,3}/stage_prediction.png       — 各阶段实验集预测

配色对齐 MATLAB Assumption.m：
  阶段1（缓慢变形）：背景浅绿 [0.7,1.0,0.7]  曲线深绿 [0.0,0.6,0.0]
  阶段2（加速变形）：背景浅黄 [1.0,1.0,0.2]  曲线深黄 [1.0,0.8,0.2]
  阶段3（快速破坏）：背景浅红 [1.0,0.7,0.7]  曲线深红 [0.8,0.0,0.0]

用法：python Q4_LGBM/visualization.py
"""

import os, sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import lightgbm as lgb

# ==================== 路径 ====================
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(ROOT))
sys.path.insert(0, ROOT)

from Q4_LGBM.common.data_utils import load_data, label_phase, load_segment
from Q4_LGBM.feature.feature_engineering import build_features
from Q4_LGBM.common.plot_utils import setup_zh

# 阶段配色（对齐 MATLAB）
PHASE_COLORS = {
    'bg':  [(0.7, 1.0, 0.7), (1.0, 1.0, 0.2), (1.0, 0.7, 0.7)],
    'line': [(0.0, 0.6, 0.0), (1.0, 0.8, 0.2), (0.8, 0.0, 0.0)],
}
PHASE_NAMES = ['1缓慢变形', '2加速变形', '3快速破坏']
PHASE_NAMES_SHORT = ['缓慢变形', '加速变形', '快速破坏']

# ==================== 1. 加载数据 ====================
print("=" * 60)
print("步骤 1/7: 加载数据")
print("=" * 60)

train_df_raw, test_df_raw = load_data()

# 分段
segment_csv = os.path.join(ROOT, 'segment', 'segment.csv')
b1, b2 = load_segment(segment_csv)
train_df_raw = label_phase(train_df_raw, b1, b2)
print(f"断点: b1={b1}, b2={b2}")

# 特征工程
print("\n步骤: 构建特征...")
train_feat = build_features(train_df_raw.copy(), is_train=True)
if 'Delta_D' not in train_feat.columns:
    train_feat['Delta_D'] = train_feat['Displacement'].diff().fillna(0)

test_feat = build_features(test_df_raw.copy(), is_train=False)
if 'Displacement' in test_feat.columns:
    test_feat.drop(columns=['Displacement'], inplace=True)
if 'Displacement' in test_feat.columns:
    del test_feat['Displacement']

# 实验集阶段标签（来源于原始数据）
# test_df_reload 直接用 test_df_raw（已在上面从 load_data 获得）
if 'Phase' in test_df_raw.columns and '^2' not in test_feat.columns:
    test_feat['Phase'] = test_df_raw['Phase'].values
    print(f"实验集 Phase 标签分布: {test_feat['Phase'].value_counts().to_dict()}")
else:
    print("警告: test_df_raw 中无 Phase 列，尝试其他方法...")
    # 如果 test_df_raw 也没有 Phase（可能是训练集错误指代）
    # 用 test_df_reload 正确获取
    _, test_df_reload = load_data()
    test_feat['Phase'] = test_df_reload['Phase'].values

# 确定特征列
feature_cols = [c for c in train_feat.columns
                if c not in ['Time', 'Phase', 'Displacement', 'Delta_D',
                             'BlastDist', 'BlastCharge']]
print(f"特征数: {len(feature_cols)}")

# ==================== 2. 重新训练 LightGBM 模型 ====================
print("\n" + "=" * 60)
print("步骤 2/7: 分阶段 LightGBM 训练")
print("=" * 60)

seeds = [42, 43, 44]
lgb_params = {
    'n_estimators': 500,
    'learning_rate': 0.01,
    'num_leaves': 31,
    'max_depth': -1,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'verbosity': -1,
}

models = {}        # phase -> [model_seed0, model_seed1, model_seed2]
train_preds = {}   # phase -> [pred_seed0, ...]
train_fitted = []  # per-phase DataFrame with predictions

for ph in [0, 1, 2]:
    mask = train_feat['Phase'] == ph
    X = train_feat.loc[mask, feature_cols].values
    y = train_feat.loc[mask, 'Delta_D'].values
    print(f"Phase {ph}: {len(y)} 样本")

    if len(y) < 50:
        models[ph] = []
        train_preds[ph] = []
        continue

    ph_models = []
    ph_preds = []
    for seed in seeds:
        params = lgb_params.copy()
        params['random_state'] = seed
        model = lgb.LGBMRegressor(**params)
        model.fit(X, y)
        y_pred = model.predict(X).flatten()
        ph_models.append(model)
        ph_preds.append(y_pred)

    models[ph] = ph_models
    train_preds[ph] = ph_preds

    # 构建 fitted DataFrame
    df_ph = train_feat.loc[mask].copy()
    df_ph['Delta_D_pred'] = np.mean(ph_preds, axis=0)
    df_ph['Residual'] = y - df_ph['Delta_D_pred'].values
    # 累积位移
    sd0 = df_ph['Displacement'].iloc[0]
    df_ph['SD_pred'] = sd0 + df_ph['Delta_D_pred'].cumsum()
    train_fitted.append(df_ph)

# ==================== 3. 训练集预测（全集） ====================
print("\n" + "=" * 60)
print("步骤 3/7: 构建训练集拟合结果")
print("=" * 60)

all_sd_actual = np.concatenate([df['Displacement'].values for df in train_fitted])
all_sd_pred = np.concatenate([df['SD_pred'].values for df in train_fitted])
all_residual = np.concatenate([df['Residual'].values for df in train_fitted])

# 阶段范围（全局索引）
seg_ranges = []
offset = 0
for df_ph in train_fitted:
    n = len(df_ph)
    seg_ranges.append((offset, offset + n - 1))
    offset += n

# 实验集也要知道长度
test_lengths = [len(test_feat[test_feat['Phase'] == ph]) for ph in [0, 1, 2]]
total_len_global = offset

# 整体指标
R2_overall = r2_score(all_sd_actual, all_sd_pred)
RMSE_overall = np.sqrt(mean_squared_error(all_sd_actual, all_sd_pred))
MAE_overall = mean_absolute_error(all_sd_actual, all_sd_pred)
max_res = np.max(np.abs(all_residual))

# 阶段分界线
dividers = [seg_ranges[1][0], seg_ranges[2][0]]

# ==================== 4. 创建输出目录 ====================
seg_dirs = [os.path.join(ROOT, f"seg{i+1}") for i in range(3)]
for d in seg_dirs:
    os.makedirs(d, exist_ok=True)

# ==================== 图表 A: training_fit_overview.png ====================
print("\n生成图表 A: 训练集拟合总览 (training_fit_overview.png)")
setup_zh()

fig = plt.figure(figsize=(16, 9))
fig.suptitle('Q4 LightGBM 分阶段表面位移拟合总览', fontsize=15, fontweight='bold')

gs_total = GridSpec(2, 1, figure=fig, hspace=0.15,
                    left=0.07, right=0.97, top=0.93, bottom=0.06,
                    height_ratios=[2.5, 1])

# ---- 上图：位移 ----
ax1 = fig.add_subplot(gs_total[0])
yl = (all_sd_actual.min() - 5, all_sd_actual.max() + 5)

# 背景色块（仅训练集范围）
for i in range(3):
    x_start = seg_ranges[i][0] - 0.5
    x_end = seg_ranges[i][1] + 0.5
    ax1.fill_between([x_start, x_end], yl[0], yl[1],
                     color=PHASE_COLORS['bg'][i], alpha=0.3, linewidth=0)

for di in dividers:
    ax1.axvline(x=di - 0.5, color='black', linestyle='--', linewidth=1.0)

# 训练集时间轴
t_all = np.arange(len(all_sd_actual))
ax1.plot(t_all, all_sd_actual, 'k-', linewidth=1.5, label='实际位移')

for i in range(3):
    s, e = seg_ranges[i]
    t_tr = np.arange(s, e + 1)
    sd_pred = all_sd_pred[s:e + 1]
    ax1.plot(t_tr, sd_pred, color=PHASE_COLORS['line'][i], linewidth=1.5,
             label=f'拟合位移·{PHASE_NAMES_SHORT[i]}')

# 阶段名标注
for i in range(3):
    x_mid = (seg_ranges[i][0] + seg_ranges[i][1]) / 2
    ax1.text(x_mid, yl[1] - 0.04 * (yl[1] - yl[0]),
             PHASE_NAMES_SHORT[i], ha='center', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.75))

ax1.set_ylabel('表面位移 (mm)', fontsize=12)
ax1.set_title(f'（R^2 = {R2_overall:.6f}，RMSE = {RMSE_overall:.4f} mm）',
              fontsize=13)
ax1.legend(fontsize=8, loc='upper left', ncol=2)
ax1.grid(alpha=0.3)
ax1.set_xlim(-20, total_len_global + 20)

# ---- 下图：残差 ----
ax2 = fig.add_subplot(gs_total[1])
for i in range(3):
    x_start = seg_ranges[i][0] - 0.5
    x_end = seg_ranges[i][1] + 0.5
    ax2.fill_between([x_start, x_end], -10, 10,
                     color=PHASE_COLORS['bg'][i], alpha=0.3, linewidth=0)

for di in dividers:
    ax2.axvline(x=di - 0.5, color='black', linestyle='--', linewidth=1.0)

for i in range(3):
    s, e = seg_ranges[i]
    t_tr = np.arange(s, e + 1)
    res = all_residual[s:e + 1]
    ax2.plot(t_tr, res, color=PHASE_COLORS['line'][i], linewidth=0.8)

ax2.axhline(0, color='black', linestyle='-', linewidth=0.8)
ax2.set_xlabel('时间序号', fontsize=12)
ax2.set_ylabel('位移残差 (mm)', fontsize=12)
ax2.set_title(f'位移拟合残差（MAE = {MAE_overall:.4f} mm，最大 |残差| = {max_res:.4f} mm）',
              fontsize=13)
ax2.grid(alpha=0.3)
ax2.set_xlim(-20, total_len_global + 20)

plt.savefig(os.path.join(ROOT, 'training_fit_overview.png'), dpi=200, bbox_inches='tight')
plt.close(fig)
print("[OK] training_fit_overview.png")

# ==================== 图表 B-D: 每阶段诊断图 ====================
print("\n生成图表 B-D: 各阶段诊断图")

for i in range(3):
    df = train_fitted[i]
    ph = i  # 0,1,2

    y_true_delta = df['Delta_D'].values
    y_pred_delta = df['Delta_D_pred'].values
    delta_residual = df['Residual'].values
    sd_true = df['Displacement'].values
    sd_pred = df['SD_pred'].values
    sd_res = sd_true - sd_pred
    t = np.arange(len(df))

    # 指标
    r2_sd = r2_score(sd_true, sd_pred)
    mae_sd = mean_absolute_error(sd_true, sd_pred)
    rmse_sd = np.sqrt(mean_squared_error(sd_true, sd_pred))
    r2_delta = r2_score(y_true_delta, y_pred_delta)
    mae_delta = mean_absolute_error(y_true_delta, y_pred_delta)
    rmse_delta = np.sqrt(mean_squared_error(y_true_delta, y_pred_delta))

    # 特征重要性
    if ph in models and len(models[ph]) > 0:
        imp = np.mean([m.feature_importances_ for m in models[ph]], axis=0)
        imp_norm = imp / imp.sum() if imp.sum() > 0 else imp
        top_n = 15
        top_idx = np.argsort(imp_norm)[-top_n:][::-1]
        top_feat = [feature_cols[idx] for idx in top_idx]
        top_imp = imp_norm[top_idx]
    else:
        top_feat, top_imp = [], []

    bg_color = PHASE_COLORS['bg'][i]
    line_color = PHASE_COLORS['line'][i]

    # ---------- 图 B: SD 诊断 ----------
    fig_sd = plt.figure(figsize=(14, 10))
    fig_sd.suptitle(f'{PHASE_NAMES[i]} · 位移诊断报告',
                    fontsize=14, fontweight='bold')
    gs_sd = GridSpec(2, 2, figure=fig_sd, hspace=0.35, wspace=0.30,
                     left=0.08, right=0.96, top=0.92, bottom=0.07)

    # ① 时序对比
    ax1 = fig_sd.add_subplot(gs_sd[0, 0])
    ax1.plot(t, sd_true, 'k-', linewidth=1.5, label='实际位移')
    ax1.plot(t, sd_pred, color=line_color, linewidth=1.5, linestyle='--', label='拟合位移')
    ax1.fill_between(t, sd_true, sd_pred, alpha=0.10, color=line_color)
    ax1.set_title('① 表面位移 · 时序对比', fontsize=12, fontweight='bold')
    ax1.set_xlabel('时间序号'); ax1.set_ylabel('表面位移 (mm)')
    ax1.legend(fontsize=9, loc='upper left'); ax1.grid(alpha=0.3)
    ax1.text(0.98, 0.05, f'R^2 = {r2_sd:.6f}', transform=ax1.transAxes,
             fontsize=10, ha='right', va='bottom',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

    # ② 散点
    ax2 = fig_sd.add_subplot(gs_sd[0, 1])
    ax2.scatter(sd_true, sd_pred, c=[line_color], alpha=0.45, s=18, edgecolors='none')
    lim_min = min(sd_true.min(), sd_pred.min())
    lim_max = max(sd_true.max(), sd_pred.max())
    margin = (lim_max - lim_min) * 0.08
    ax2.plot([lim_min - margin, lim_max + margin], [lim_min - margin, lim_max + margin],
             'k--', linewidth=1.0, alpha=0.7, label='y = x')
    ax2.set_xlabel('实际位移 (mm)'); ax2.set_ylabel('拟合位移 (mm)')
    ax2.set_title('② 拟合位移 vs 实际位移', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9); ax2.grid(alpha=0.25)
    ax2.text(0.05, 0.95, f'R^2 = {r2_sd:.6f}\nMAE = {mae_sd:.4f} mm',
             transform=ax2.transAxes, fontsize=9, va='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

    # ③ 残差时序
    ax3 = fig_sd.add_subplot(gs_sd[1, 0])
    ax3.fill_between(t, sd_res, 0, alpha=0.25, color='steelblue')
    ax3.plot(t, sd_res, color='steelblue', linewidth=0.7)
    ax3.axhline(0, color='black', linestyle='--', linewidth=0.8)
    std_res = np.std(sd_res)
    ax3.axhline(+2 * std_res, color='red', linestyle=':', linewidth=1.0, alpha=0.7, label='+2σ')
    ax3.axhline(-2 * std_res, color='red', linestyle=':', linewidth=1.0, alpha=0.7, label='-2σ')
    outlier_ratio = np.mean(np.abs(sd_res) > 2 * std_res) * 100
    ax3.set_title(f'③ 位移残差时序（超出 ±2σ: {outlier_ratio:.1f}%）',
                  fontsize=12, fontweight='bold')
    ax3.set_xlabel('时间序号'); ax3.set_ylabel('位移残差 (mm)')
    ax3.legend(fontsize=8, loc='upper right'); ax3.grid(alpha=0.25)

    # ④ 残差分布
    ax4 = fig_sd.add_subplot(gs_sd[1, 1])
    ax4.hist(sd_res, bins=35, density=True, alpha=0.6,
             color='steelblue', edgecolor='white', linewidth=0.6)
    mu, sigma = stats.norm.fit(sd_res)
    x_norm = np.linspace(sd_res.min(), sd_res.max(), 200)
    ax4.plot(x_norm, stats.norm.pdf(x_norm, mu, sigma), 'r-', linewidth=1.8,
             label=f'N({mu:.4f}, {sigma:.4f}^2)')
    ax4.axvline(0, color='black', linestyle='--', linewidth=1.0)
    ax4.set_title('④ 位移残差分布', fontsize=12, fontweight='bold')
    ax4.set_xlabel('位移残差 (mm)'); ax4.set_ylabel('概率密度')
    ax4.legend(fontsize=9)
    ax4.text(0.98, 0.95, f'均值 = {mu:.4f}\n标准差 = {sigma:.4f}',
             transform=ax4.transAxes, fontsize=8, va='top', ha='right',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

    save_path = os.path.join(seg_dirs[i], "stage_SD_diagnosis.png")
    fig_sd.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig_sd)
    print(f"[OK] seg{i+1}/stage_SD_diagnosis.png")

    # ---------- 图 C: Delta_D 诊断 ----------
    fig_delta = plt.figure(figsize=(14, 10))
    fig_delta.suptitle(f'{PHASE_NAMES[i]} · 速度诊断报告',
                       fontsize=14, fontweight='bold')
    gs_d = GridSpec(2, 2, figure=fig_delta, hspace=0.35, wspace=0.30,
                    left=0.08, right=0.96, top=0.92, bottom=0.07)

    # ① 时序对比
    ax1 = fig_delta.add_subplot(gs_d[0, 0])
    ax1.plot(t, y_true_delta, 'k-', linewidth=0.9, alpha=0.7, label='实际速度')
    ax1.plot(t, y_pred_delta, color=line_color, linewidth=2, linestyle='--', label='预测速度')
    ax1.set_title('① 速度 · 时序对比', fontsize=12, fontweight='bold')
    ax1.set_xlabel('时间序号'); ax1.set_ylabel('速度 (mm/min)')
    ax1.legend(fontsize=9, loc='upper left'); ax1.grid(alpha=0.3)
    ax1.text(0.98, 0.95, f'R^2 = {r2_delta:.4f}', transform=ax1.transAxes,
             fontsize=10, va='top', ha='right',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

    # ② 散点
    ax2 = fig_delta.add_subplot(gs_d[0, 1])
    ax2.scatter(y_true_delta, y_pred_delta, c=[line_color], alpha=0.45, s=18, edgecolors='none')
    lim_min = min(y_true_delta.min(), y_pred_delta.min())
    lim_max = max(y_true_delta.max(), y_pred_delta.max())
    margin = (lim_max - lim_min) * 0.08
    ax2.plot([lim_min - margin, lim_max + margin], [lim_min - margin, lim_max + margin],
             'k--', linewidth=1.0, alpha=0.7, label='y = x')
    ax2.set_xlabel('实际速度 (mm/min)'); ax2.set_ylabel('预测速度 (mm/min)')
    ax2.set_title('② 预测速度 vs 实际速度', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9); ax2.grid(alpha=0.25)
    ax2.text(0.05, 0.95, f'R^2 = {r2_delta:.4f}\nMAE = {mae_delta:.4f} mm/min',
             transform=ax2.transAxes, fontsize=9, va='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

    # ③ 残差时序
    ax3 = fig_delta.add_subplot(gs_d[1, 0])
    ax3.fill_between(t, delta_residual, 0, alpha=0.25, color='steelblue')
    ax3.plot(t, delta_residual, color='steelblue', linewidth=0.7)
    ax3.axhline(0, color='black', linestyle='--', linewidth=0.8)
    std_res_d = np.std(delta_residual)
    ax3.axhline(+2 * std_res_d, color='red', linestyle=':', linewidth=1.0, alpha=0.7, label='+2σ')
    ax3.axhline(-2 * std_res_d, color='red', linestyle=':', linewidth=1.0, alpha=0.7, label='-2σ')
    outlier_ratio_d = np.mean(np.abs(delta_residual) > 2 * std_res_d) * 100
    ax3.set_title(f'③ 速度残差时序（超出 ±2σ: {outlier_ratio_d:.1f}%）',
                  fontsize=12, fontweight='bold')
    ax3.set_xlabel('时间序号'); ax3.set_ylabel('速度残差 (mm/min)')
    ax3.legend(fontsize=8, loc='upper right'); ax3.grid(alpha=0.25)

    # ④ 残差分布
    ax4 = fig_delta.add_subplot(gs_d[1, 1])
    ax4.hist(delta_residual, bins=35, density=True, alpha=0.6,
             color='steelblue', edgecolor='white', linewidth=0.6)
    mu_d, sigma_d = stats.norm.fit(delta_residual)
    x_norm_d = np.linspace(delta_residual.min(), delta_residual.max(), 200)
    ax4.plot(x_norm_d, stats.norm.pdf(x_norm_d, mu_d, sigma_d), 'r-', linewidth=1.8,
             label=f'N({mu_d:.5f}, {sigma_d:.5f}^2)')
    ax4.axvline(0, color='black', linestyle='--', linewidth=1.0)
    ax4.set_title('④ 速度残差分布', fontsize=12, fontweight='bold')
    ax4.set_xlabel('速度残差 (mm/min)'); ax4.set_ylabel('概率密度')
    ax4.legend(fontsize=9)
    ax4.text(0.98, 0.95, f'均值 = {mu_d:.5f}\n标准差 = {sigma_d:.5f}',
             transform=ax4.transAxes, fontsize=8, va='top', ha='right',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

    save_path = os.path.join(seg_dirs[i], "stage_delta_diagnosis.png")
    fig_delta.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig_delta)
    print(f"[OK] seg{i+1}/stage_delta_diagnosis.png")

    # ---------- 图 D: 特征重要性 ----------
    if len(top_feat) > 0:
        fig_imp = plt.figure(figsize=(8, 6))
        fig_imp.suptitle(f'{PHASE_NAMES[i]} · 特征重要性 (Top {top_n})',
                         fontsize=13, fontweight='bold')
        ax_imp = fig_imp.add_subplot(111)
        colors_bar = [line_color] * len(top_feat)
        bars = ax_imp.barh(range(len(top_feat)), top_imp, color=colors_bar,
                           edgecolor='white', height=0.6)
        ax_imp.set_yticks(range(len(top_feat)))
        ax_imp.set_yticklabels(top_feat, fontsize=9)
        ax_imp.set_xlabel('相对重要性', fontsize=11)
        ax_imp.invert_yaxis()
        ax_imp.grid(alpha=0.25, axis='x')

        for bar, val in zip(bars, top_imp):
            ax_imp.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                        f'{val:.3f}', va='center', fontsize=8)

        save_path = os.path.join(seg_dirs[i], "stage_coefficients.png")
        fig_imp.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig_imp)
        print(f"[OK] seg{i+1}/stage_coefficients.png")
    else:
        print(f"[跳过] seg{i+1}/stage_coefficients.png（无特征重要性数据）")

# ==================== 5. 实验集预测 ====================
print("\n" + "=" * 60)
print("步骤 5/7: 实验集预测")
print("=" * 60)

test_delta_pred = np.zeros(len(test_feat))
for ph in [0, 1, 2]:
    mask = test_feat['Phase'] == ph
    if mask.sum() == 0:
        continue
    X_test = test_feat.loc[mask, feature_cols].values
    if ph not in models or len(models[ph]) == 0:
        test_delta_pred[mask] = 0
        continue
    preds = [m.predict(X_test).flatten() for m in models[ph]]
    test_delta_pred[mask] = np.mean(preds, axis=0)

# 链式累积
train_last_disp = train_df_raw['Displacement'].iloc[-1]
test_disp_pred = np.cumsum(test_delta_pred)
test_disp_pred = test_disp_pred + train_last_disp - test_disp_pred[0]

# 各阶段实验集 DataFrame
test_predicted = []
for ph in [0, 1, 2]:
    mask = test_feat['Phase'] == ph
    df_te = test_feat[mask].copy()
    df_te['Delta_D_pred'] = test_delta_pred[mask]
    df_te['Displacement_pred'] = test_disp_pred[mask]
    test_predicted.append(df_te)

# 实验集连续坐标
seg_ends = np.cumsum(test_lengths)
seg_starts = np.concatenate([[0], seg_ends[:-1]])
all_exp_pred = np.concatenate([df['Displacement_pred'].values for df in test_predicted])
total_test = len(test_feat)

# ==================== 图表 E: prediction_overview.png ====================
print("生成图表 E: 实验集预测总览 (prediction_overview.png)")
setup_zh()

fig, ax = plt.subplots(figsize=(16, 5))
fig.suptitle('实验集预测表面位移总览（三段链式拼接）', fontsize=15, fontweight='bold')

y_min = all_exp_pred.min() - 5
y_max = all_exp_pred.max() + 5

# 背景色块
for i in range(3):
    x_start = seg_starts[i] - 0.5
    x_end = seg_ends[i] - 0.5
    ax.fill_between([x_start, x_end], y_min, y_max,
                    color=PHASE_COLORS['bg'][i], alpha=0.3, linewidth=0)

# 分隔虚线
for x_div in seg_ends[:-1]:
    ax.axvline(x=x_div - 0.5, color='black', linestyle='--', linewidth=1.0)

# 预测曲线
for i in range(3):
    t_start = seg_starts[i]
    t_end = seg_ends[i] - 1
    t_range = np.arange(t_start, t_end + 1)
    ax.plot(t_range, test_predicted[i]['Displacement_pred'].values,
            color=PHASE_COLORS['line'][i], linewidth=1.8,
            marker='.', markersize=2,
            label=f'预测位移·{PHASE_NAMES_SHORT[i]}')

# 阶段名标注
for i in range(3):
    x_mid = (seg_starts[i] + seg_ends[i] - 1) / 2
    ax.text(x_mid, y_max - 0.04 * (y_max - y_min),
            PHASE_NAMES_SHORT[i], ha='center', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.75))

ax.set_xlabel('实验集时间序号', fontsize=12)
ax.set_ylabel('预测表面位移 (mm)', fontsize=12)
ax.set_title('实验集预测表面位移（三段拼接，链式累积）', fontsize=13)
ax.legend(fontsize=9, loc='upper left')
ax.grid(alpha=0.3)
ax.set_xlim(-20, total_test + 20)

plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.93])
plt.savefig(os.path.join(ROOT, 'prediction_overview.png'), dpi=200, bbox_inches='tight')
plt.close(fig)
print("[OK] prediction_overview.png")

# ==================== 图表 F: 各阶段实验集预测 ====================
print("生成图表 F: 各阶段实验集预测")
for i in range(3):
    df_te = test_predicted[i]
    if len(df_te) == 0:
        print(f"[跳过] seg{i+1}/stage_prediction.png（空）")
        continue

    t_te = np.arange(len(df_te))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t_te, df_te['Displacement_pred'].values,
            color=PHASE_COLORS['line'][i], linewidth=1.8,
            marker='.', markersize=3, label='预测表面位移')

    ax.set_title(f'{PHASE_NAMES[i]} · 实验集预测', fontsize=14, fontweight='bold')
    ax.set_xlabel('时间序号', fontsize=12)
    ax.set_ylabel('表面位移 (mm)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    save_path = os.path.join(seg_dirs[i], "stage_prediction.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] seg{i+1}/stage_prediction.png")

# ==================== 6. 汇总 ====================
print("\n" + "=" * 60)
print("Q4 LightGBM 可视化完成！")
print("=" * 60)
print(f"\n输出文件:")
print(f"  {ROOT}\\training_fit_overview.png")
print(f"  {ROOT}\\prediction_overview.png")
for i in range(3):
    print(f"  {seg_dirs[i]}\\")
    print(f"    ├── stage_SD_diagnosis.png")
    print(f"    ├── stage_delta_diagnosis.png")
    print(f"    ├── stage_coefficients.png")
    print(f"    └── stage_prediction.png")

print(f"\n整体拟合指标:")
print(f"  R^2  = {R2_overall:.6f}")
print(f"  RMSE = {RMSE_overall:.4f} mm")
print(f"  MAE  = {MAE_overall:.4f} mm")
