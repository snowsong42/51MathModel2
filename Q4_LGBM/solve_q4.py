"""
Q4 LightGBM 求解脚本
流程:
  1. 加载 Q4 数据 (训练集 + 实验集)
  2. 训练集分段检测 (Pelt)
  3. 特征工程
  4. 分阶段 LightGBM 训练
  5. 实验集预测
  6. 提取指定时间点预测值
  7. 输出结果

用法: python Q4_LGBM/solve_q4.py
"""

import os, sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 0. 路径设置 & 导入
# ============================================================
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(ROOT))  # 项目根目录

from Q4_LGBM.common.data_utils import load_data, label_phase, load_segment
from Q4_LGBM.feature.feature_engineering import build_features, exp_decay_series, effective_rainfall
from Q4_LGBM.common.plot_utils import setup_zh, save_dir, phase_name

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import lightgbm as lgb

# ============================================================
# 1. 加载数据
# ============================================================
print("=" * 60)
print("步骤 1/7: 加载数据")
print("=" * 60)

train_df, test_df = load_data()

print(f"训练集: {train_df.shape}, 时间范围: {train_df['Time'].min()} ~ {train_df['Time'].max()}")
print(f"实验集: {test_df.shape}, 时间范围: {test_df['Time'].min()} ~ {test_df['Time'].max()}")

# ============================================================
# 2. 训练集分段检测（读取 MATLAB segment.m 生成的 segment.csv）
# ============================================================
print("\n" + "=" * 60)
print("步骤 2/7: 训练集分段检测（读取 segment.csv）")
print("=" * 60)

# 创建输出目录
os.makedirs(os.path.join(ROOT, 'segment'), exist_ok=True)
os.makedirs(os.path.join(ROOT, 'feature'), exist_ok=True)

segment_csv = os.path.join(ROOT, 'segment', 'segment.csv')
from Q4_LGBM.common.data_utils import load_segment
b1, b2 = load_segment(segment_csv)
print(f"  从 {segment_csv} 读取断点: b1={b1}, b2={b2}")

train_df = label_phase(train_df, b1, b2)

print(f"阶段边界: b1={b1}, b2={b2}")
print(f"阶段分布: Phase 0={train_df[train_df['Phase']==0].shape[0]}, "
      f"Phase 1={train_df[train_df['Phase']==1].shape[0]}, "
      f"Phase 2={train_df[train_df['Phase']==2].shape[0]}")

# 绘制分段图
setup_zh()
vel_series = np.gradient(train_df['Displacement'].values)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

colors = {0: 'green', 1: 'orange', 2: 'red'}
phase_labels = {0: '缓慢变形', 1: '加速变形', 2: '快速变形'}
# ============================================================
# 3. 特征工程
# ============================================================
print("\n" + "=" * 60)
print("步骤 3/7: 特征工程")
print("=" * 60)

# 训练集特征
train_feat = build_features(train_df.copy(), is_train=True)
# 清除 Delta_D 列（已经从 Displacement.diff() 计算得到）
if 'Delta_D' not in train_feat.columns and 'Displacement' in train_feat.columns:
    train_feat['Delta_D'] = train_feat['Displacement'].diff().fillna(0)

# 实验集特征
test_feat = build_features(test_df.copy(), is_train=False)
# 实验集没有 Displacement，所有 Delta_D 为 NaN
if 'Displacement' in test_feat.columns:
    test_feat.drop(columns=['Displacement'], inplace=True)

# 打印特征列
feature_cols = [c for c in train_feat.columns
                if c not in ['Time', 'Phase', 'Displacement', 'Delta_D',
                             'BlastDist', 'BlastCharge']]
print(f"总特征数: {len(feature_cols)}")
print("特征列表:", feature_cols)

# 保存特征
train_out = os.path.join(ROOT, 'feature', 'train_features.xlsx')
test_out = os.path.join(ROOT, 'feature', 'test_features.xlsx')
train_feat.to_excel(train_out, index=False)
test_feat.to_excel(test_out, index=False)
print(f"训练集特征保存: {train_out}")
print(f"实验集特征保存: {test_out}")

# ============================================================
# 4. 分阶段 LightGBM 训练
# ============================================================
print("\n" + "=" * 60)
print("步骤 4/7: 分阶段 LightGBM 训练")
print("=" * 60)

seeds = [42, 43, 44]
n_repeats = len(seeds)
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
    'random_state': 42,
    'verbosity': -1,
}

models = {}  # phase -> list of models (one per seed)
train_preds = {}  # phase -> list of prediction arrays

for ph in [0, 1, 2]:
    mask = train_feat['Phase'] == ph
    X = train_feat.loc[mask, feature_cols].values
    y = train_feat.loc[mask, 'Delta_D'].values
    print(f"\nPhase {ph} ({phase_labels[ph]}): {len(y)} 样本")

    if len(y) < 50:
        print(f"    跳过阶段 {ph}: 样本不足")
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
        print(f"    种子 {seed}: 训练 R² = {np.corrcoef(y, y_pred)[0, 1] ** 2:.4f}")

    models[ph] = ph_models
    train_preds[ph] = ph_preds

# 绘制训练集拟合效果
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig.suptitle('Q4 训练集位移拟合效果（LightGBM 分阶段）', fontsize=14)

disp_full = train_feat['Displacement'].values
disp_pred_full = np.zeros(len(train_feat))

for ph in [0, 1, 2]:
    ax = axes[ph]
    mask = train_feat['Phase'] == ph
    idx = np.where(mask)[0]

    # 真实位移
    real_disp = disp_full[mask]
    # 预测位移（累加 Delta_D）
    if ph in train_preds and len(train_preds[ph]) > 0:
        pred_delta = np.mean(train_preds[ph], axis=0)  # 3次平均
        pred_disp = np.cumsum(pred_delta)
        # 对齐起始值
        if len(idx) > 0:
            start_disp = disp_full[idx[0]]
            pred_disp = pred_disp + start_disp - pred_disp[0]
        disp_pred_full[mask] = pred_disp
    else:
        pred_disp = real_disp

    ax.plot(idx, real_disp, 'b-', alpha=0.6, linewidth=1, label='真实位移')
    if ph in train_preds and len(train_preds[ph]) > 0:
        ax.plot(idx, pred_disp, 'r--', alpha=0.8, linewidth=1.5, label='预测位移')

    ax.set_ylabel('位移 (mm)')
    ax.set_title(f'Phase {ph}: {phase_labels[ph]}')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # R²
    if ph in train_preds and len(train_preds[ph]) > 0:
        r2 = np.corrcoef(real_disp, pred_disp)[0, 1] ** 2
        rmse = np.sqrt(np.mean((real_disp - pred_disp) ** 2))
        ax.text(0.05, 0.85, f'R²={r2:.4f}, RMSE={rmse:.3f}',
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

axes[-1].set_xlabel('时间步 (每步10min)')
plt.tight_layout()
fit_path = os.path.join(ROOT, 'training_fit.png')
plt.savefig(fit_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n训练集拟合图保存: {fit_path}")

# ============================================================
# 5. 实验集预测
# ============================================================
print("\n" + "=" * 60)
print("步骤 5/7: 实验集预测")
print("=" * 60)

test_feat = pd.read_excel(test_out)  # 重新载入确保一致性
# 确保 Phase 列存在
if 'Phase' not in test_feat.columns:
    # 从测试集数据中获取
    test_df_reload, _ = load_data()
    test_feat['Phase'] = test_df_reload['Phase'].values

# 打印实验集阶段分布
print("实验集阶段分布:")
print(test_feat['Phase'].value_counts().sort_index())

# 逐阶段预测
test_delta_pred = np.zeros(len(test_feat))
for ph in [0, 1, 2]:
    mask = test_feat['Phase'] == ph
    if mask.sum() == 0:
        print(f"Phase {ph}: 无样本")
        continue
    X_test = test_feat.loc[mask, feature_cols].values
    if ph not in models or len(models[ph]) == 0:
        print(f"Phase {ph}: 模型不存在（训练阶段被跳过）")
        test_delta_pred[mask] = 0
        continue
    # 3次平均
    preds = []
    for model in models[ph]:
        preds.append(model.predict(X_test).flatten())
    test_delta_pred[mask] = np.mean(preds, axis=0)
    print(f"Phase {ph}: {mask.sum()} 样本, "
          f"Delta_D 范围 [{test_delta_pred[mask].min():.4f}, {test_delta_pred[mask].max():.4f}]")

# 累积求位移
test_disp_pred = np.cumsum(test_delta_pred)
# 对齐起始位移：从训练集最后一个位移值开始
train_last_disp = train_df['Displacement'].iloc[-1]
test_disp_pred = test_disp_pred + train_last_disp - test_disp_pred[0]
print(f"预测位移范围: [{test_disp_pred.min():.2f}, {test_disp_pred.max():.2f}] mm")

# 保存预测结果（增量+位移）
test_feat['Delta_D_pred'] = test_delta_pred
test_feat['Displacement_pred'] = test_disp_pred

# ============================================================
# 6. 提取指定时间点预测值
# ============================================================
print("\n" + "=" * 60)
print("步骤 6/7: 提取指定时间点预测值")
print("=" * 60)

target_times = [
    '2025-05-09 12:00',
    '2025-05-27 08:00',
    '2025-06-01 12:00',
    '2025-06-03 22:00',
    '2025-06-04 01:40',
]

results = []
for t in target_times:
    dt = pd.to_datetime(t)
    # 找最近的时间点
    if 'Time' in test_feat.columns:
        time_diff = (pd.to_datetime(test_feat['Time']) - dt).abs()
    else:
        # 按行号推算（实验集起始 2025-05-01 16:40）
        base_time = pd.to_datetime('2025-05-01 16:40')
        time_diff = (pd.to_datetime(base_time + pd.to_timedelta(test_feat.index * 10, unit='m')) - dt).abs()
    idx = time_diff.idxmin()
    match_time = dt
    pred_val = test_disp_pred[idx]
    results.append({
        '目标时间': t,
        '索引': idx,
        '预测位移 (mm)': round(float(pred_val), 3)
    })
    print(f"  {t} → 索引 {idx}, 预测位移 = {pred_val:.3f} mm")

result_df = pd.DataFrame(results)
result_path = os.path.join(ROOT, 'result_table.csv')
result_df.to_csv(result_path, index=False, encoding='utf-8-sig')
print(f"结果表保存: {result_path}")

# ============================================================
# 6.5 绘制实验集预测曲线
# ============================================================
print("\n" + "=" * 60)
print("步骤 6.5/7: 绘制实验集预测曲线")
print("=" * 60)

setup_zh()
fig, ax = plt.subplots(figsize=(14, 6))

# 分阶段着色
test_time = pd.to_datetime(test_feat['Time']) if 'Time' in test_feat.columns else test_feat.index

for ph in [2, 1, 0]:
    mask = test_feat['Phase'] == ph
    if mask.sum() == 0:
        continue
    ax.plot(test_time[mask], test_disp_pred[mask], color=colors[ph],
            linewidth=1.5, alpha=0.8, label=f'Phase {ph}: {phase_labels[ph]}')

# 标记目标时间点
for r in results:
    t = pd.to_datetime(r['目标时间'])
    v = r['预测位移 (mm)']
    ax.scatter(t, v, color='blue', s=50, zorder=5)
    ax.annotate(f"({t.strftime('%m-%d %H:%M')}, {v:.1f}mm)",
                xy=(t, v), xytext=(10, 10), textcoords='offset points',
                fontsize=8, arrowprops=dict(arrowstyle='->', color='gray'))

ax.set_xlabel('时间')
ax.set_ylabel('表面位移 (mm)')
ax.set_title('Q4 实验集 LightGBM 预测位移曲线')
ax.legend()
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
pred_path = os.path.join(ROOT, 'test_prediction.png')
plt.savefig(pred_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"预测结果图保存: {pred_path}")

# ============================================================
# 7. 特征重要性
# ============================================================
print("\n" + "=" * 60)
print("步骤 7/7: 特征重要性分析")
print("=" * 60)

# 汇总各阶段特征重要性
importance_df = pd.DataFrame({'feature': feature_cols})
for ph in [0, 1, 2]:
    col_name = f'Phase{ph}_importance'
    if ph in models and len(models[ph]) > 0:
        # 所有种子平均
        imp = np.mean([m.feature_importances_ for m in models[ph]], axis=0)
        importance_df[col_name] = imp
    else:
        importance_df[col_name] = 0

# 归一化
for ph in [0, 1, 2]:
    col = f'Phase{ph}_importance'
    if importance_df[col].sum() > 0:
        importance_df[col] = importance_df[col] / importance_df[col].sum()

importance_df['avg_importance'] = importance_df[[c for c in importance_df.columns if 'importance' in c]].mean(axis=1)
importance_df = importance_df.sort_values('avg_importance', ascending=False)

imp_path = os.path.join(ROOT, 'feature_importance.csv')
importance_df.to_csv(imp_path, index=False, encoding='utf-8-sig')
print(f"特征重要性表保存: {imp_path}")

# 绘制特征重要性（top 20）
setup_zh()
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, ph in enumerate([0, 1, 2]):
    ax = axes[i]
    col = f'Phase{ph}_importance'
    top = importance_df.nlargest(20, col)
    ax.barh(range(len(top)), top[col].values, color=list(colors.values())[ph], alpha=0.7)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top['feature'].values, fontsize=8)
    ax.set_xlabel('相对重要性')
    ax.set_title(f'Phase {ph}: {phase_labels[ph]}')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

fig.suptitle('Q4 各阶段特征重要性 (Top 20)', fontsize=14)
plt.tight_layout()
importance_img_path = os.path.join(ROOT, 'feature_importance.png')
plt.savefig(importance_img_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"特征重要性图保存: {importance_img_path}")

# ============================================================
# 输出汇总
# ============================================================
print("\n" + "=" * 60)
print("Q4 LightGBM 求解完成！")
print("=" * 60)
print(f"\n输出文件清单:")
print(f"  1. Q4_LGBM/segment/segment.csv - 训练集分段结果")
print(f"  2. Q4_LGBM/segment/phase_segmentation.png - 分段可视化")
print(f"  3. Q4_LGBM/feature/train_features.xlsx - 训练集特征表")
print(f"  4. Q4_LGBM/feature/test_features.xlsx - 实验集特征表")
print(f"  5. Q4_LGBM/training_fit.png - 训练集拟合效果")
print(f"  6. Q4_LGBM/test_prediction.png - 实验集预测曲线")
print(f"  7. Q4_LGBM/result_table.csv - 表4.1 指定时间点预测位移")
print(f"  8. Q4_LGBM/feature_importance.csv - 特征重要性表")
print(f"  9. Q4_LGBM/feature_importance.png - 特征重要性图")

print(f"\n表4.1 预测结果:")
for r in results:
    print(f"  {r['目标时间']}: {r['预测位移 (mm)']:.3f} mm")
