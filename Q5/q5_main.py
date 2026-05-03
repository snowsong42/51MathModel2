"""
Q5 一体化建模脚本（Q5.1 最优变量组合 + Q5.2 消融实验）
======================================================
用法: python q5_main.py
输出: Q5/结果与使用指南/图表/ 下的全部汉化图片
"""

import warnings, os, sys
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common.data_utils import load_pipeline, get_all_features
from common.plot_utils import setup_zh, save_dir, phase_name

setup_zh()
OUT_DIR = save_dir()
print('='*60)

# ──────────────── 1. 加载数据 ────────────────
df, base_vars, (b1, b2) = load_pipeline()
all_feats = get_all_features(df, base_vars)
df = df.dropna(subset=all_feats + ['Delta_D']).reset_index(drop=True)
n_phase = [sum(df['Phase']==i) for i in range(3)]
print(f'数据: {len(df)} 行 | 特征: {len(all_feats)} | '
      f'缓慢={n_phase[0]} 加速={n_phase[1]} 快速={n_phase[2]}')
print(f'阶段节点: b1={b1} b2={b2}')

# ──────────────── 2. 模型函数 ────────────────
def train_lgb(df, feats, target='Delta_D'):
    """分阶段 LightGBM，返回全量预测值和阶段指标"""
    y_pred = np.zeros(len(df))
    phase_metrics = {}
    for pid in range(3):
        mask = df['Phase'] == pid
        if mask.sum() < 30: continue
        X, y = df.loc[mask, feats].values, df.loc[mask, target].values
        model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.01,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1)
        model.fit(X, y)
        yp = model.predict(X)
        phase_metrics[pid] = {
            'rmse': np.sqrt(mean_squared_error(y, yp)),
            'mae': mean_absolute_error(y, yp),
            'r2': r2_score(y, yp), 'n': int(mask.sum())}
        y_pred[mask] = yp
    return y_pred, phase_metrics

# ──────────────── 3. Q5.1 全模型 ────────────────
print('\n──── Q5.1 全模型 ────')
y_full, pm_full = train_lgb(df, all_feats)
rmse_f = np.sqrt(mean_squared_error(df['Delta_D'], y_full))
mae_f  = mean_absolute_error(df['Delta_D'], y_full)
r2_f   = r2_score(df['Delta_D'], y_full)
print(f'全特征模型: RMSE={rmse_f:.4f} MAE={mae_f:.4f} R²={r2_f:.4f}')

# ──────────────── 4. Q5.1 逐变量剔除 ────────────────
base_vars_cn = {'Rainfall':'降雨量','PorePressure':'孔隙水压力','Microseismic':'微震事件数',
                'Infiltration':'干湿入渗系数','BlastDist':'爆破距离','BlastCharge':'单段药量'}
ablation_families = {
    '降雨量':        ['Rainfall', 'Rain_', 'rain', 'Time_since_rain'],
    '孔隙水压力':    ['PorePressure', 'Pore', 'pore'],
    '微震事件数':    ['Microseismic', 'microseismic', 'Microseismic_roll6'],
    '干湿入渗系数':  ['Infiltration', 'infiltration', 'Infilt'],
    '爆破':          ['Blast', 'blast', 'Time_since_blast'],
}

def exclude_features(feats, keywords):
    return [f for f in feats if not any(kw in f for kw in keywords)]

results = []
for cn_name, kws in ablation_families.items():
    excl = exclude_features(all_feats, kws)
    y_abl, pm_abl = train_lgb(df, excl)
    rmse_a = np.sqrt(mean_squared_error(df['Delta_D'], y_abl))
    r2_a   = r2_score(df['Delta_D'], y_abl)
    delta_r2 = r2_f - r2_a
    results.append({
        '变量': cn_name, '特征数': len(excl),
        'RMSE': rmse_a, 'ΔRMSE': rmse_a - rmse_f,
        'R²': r2_a, 'ΔR²': delta_r2,
        'RMSE增比%': (rmse_a/rmse_f - 1)*100,
        'pm': pm_abl})
    print(f'  剔除 {cn_name:6s}: RMSE={rmse_a:.4f} (Δ+{rmse_a-rmse_f:.4f}) R²={r2_a:.4f} (Δ{delta_r2:+.4f})')

# 最优/最差
best = min(results, key=lambda r: r['ΔR²'])
worst = max(results, key=lambda r: r['ΔR²'])
print(f'\n最优组合: 剔除 {best["变量"]} (ΔR²={best["ΔR²"]:+.4f}) → 该变量贡献最小')
print(f'最差组合: 剔除 {worst["变量"]} (ΔR²={worst["ΔR²"]:+.4f}) → 该变量贡献最大')

# ──────────────── 5. 绘图 ────────────────
colors = ['#3498db','#e74c3c','#2ecc71','#9b59b6','#f39c12','#1abc9c']
labels = [r['变量'] for r in results]

# 图1: 总体 RMSE 柱状 + R² 排序
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

ax = axes[0]
deltas = [r['ΔRMSE'] for r in results]
bars = ax.bar(range(len(labels)), deltas, color=colors[:len(labels)],
              edgecolor='white', linewidth=1.2)
for i, (b, d) in enumerate(zip(bars, deltas)):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.001 if d>=0 else -0.003,
            f'{d:+.4f}', ha='center', fontsize=9, fontweight='bold')
ax.axhline(0, color='gray', ls='--')
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('ΔRMSE (mm)')
ax.set_title(f'Q5.1 剔除各变量后 RMSE 变化\n(全模型 RMSE={rmse_f:.4f})', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

ax = axes[1]
sorted_idx = np.argsort([r['ΔR²'] for r in results])
for i, idx in enumerate(sorted_idx):
    r = results[idx]
    c = '#e74c3c' if r['ΔR²'] > np.median([x['ΔR²'] for x in results]) else '#2ecc71'
    ax.barh(i, r['ΔR²'], color=c, edgecolor='white')
    ax.text(r['ΔR²']+0.001, i, f'{r["ΔR²"]:.4f}', va='center', fontsize=9, fontweight='bold')
ax.axvline(0, color='gray', ls='--')
ax.set_yticks(range(len(labels)))
ax.set_yticklabels([results[i]['变量'] for i in sorted_idx], fontsize=9)
ax.set_xlabel('ΔR² (下降越大越重要)')
ax.set_title(f'Q5.1 变量重要度排序\n(全模型 R²={r2_f:.4f})', fontweight='bold')
ax.grid(axis='x', alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'Q5_1_变量组合对比.png'), dpi=150, bbox_inches='tight')
plt.close(fig); print(f'[图] Q5_1_变量组合对比.png')

# 图2: 分阶段热力图
fig, ax = plt.subplots(figsize=(10, 5))
phase_deltas = np.zeros((len(results), 3))
for i, r in enumerate(results):
    for pid in range(3):
        if pid in r['pm'] and pid in pm_full:
            phase_deltas[i, pid] = r['pm'][pid]['rmse'] - pm_full[pid]['rmse']
im = ax.imshow(phase_deltas, cmap='RdYlGn_r', aspect='auto')
ax.set_xticks(range(3)); ax.set_xticklabels(['缓慢变形','加速变形','快速变形'])
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
for i in range(len(labels)):
    for j in range(3):
        v = phase_deltas[i, j]
        ax.text(j, i, f'{v:+.4f}', ha='center', va='center',
                fontsize=9, fontweight='bold',
                color='white' if abs(v)>np.percentile(abs(phase_deltas), 70) else 'black')
plt.colorbar(im, ax=ax, label='ΔRMSE (mm)')
ax.set_title('Q5.2 分阶段 RMSE 变化热力图\n(红色=影响大 绿色=影响小)', fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'Q5_2_分阶段热力图.png'), dpi=150, bbox_inches='tight')
plt.close(fig); print(f'[图] Q5_2_分阶段热力图.png')

# 图3: 分阶段时序对比（最重要的变量单独绘制）
worst_var_excl = exclude_features(all_feats, ablation_families[worst['变量']])
y_worst, _ = train_lgb(df, worst_var_excl)

fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
phase_ranges = [(0, b1), (b1, b2), (b2, len(df))]
phase_names_list = ['缓慢变形阶段','加速变形阶段','快速变形阶段']
for pid, (s, e) in enumerate(phase_ranges):
    ax = axes[pid]
    idx = np.arange(s, e)
    ax.plot(idx, df['Delta_D'].iloc[s:e].values, 'k-', lw=1, alpha=0.5, label='真实值')
    ax.plot(idx, y_full[s:e], 'b-', lw=1.5, alpha=0.7, label='全模型预测')
    ax.plot(idx, y_worst[s:e], 'r--', lw=1.5, alpha=0.7, label=f'剔除{worst["变量"]}')
    ax.set_ylabel('位移增量 (mm)')
    ax.set_title(f'{phase_names_list[pid]}', fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
axes[-1].set_xlabel('时间步')
fig.suptitle(f'Q5.2 剔除{worst["变量"]}的预测对比（该变量影响最大）', fontweight='bold', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(OUT_DIR, 'Q5_2_分阶段时序对比.png'), dpi=150, bbox_inches='tight')
plt.close(fig); print(f'[图] Q5_2_分阶段时序对比.png')

# 图4: 汇总表格图
fig, ax = plt.subplots(figsize=(12, 3.5))
ax.axis('off')
col_l = ['变量','特征数','RMSE','ΔRMSE','RMSE增比%','R²','ΔR²']
tbl = [[r['变量'], r['特征数'], f'{r["RMSE"]:.4f}', f'{r["ΔRMSE"]:+.4f}',
        f'{r["RMSE增比%"]:+.1f}%', f'{r["R²"]:.4f}', f'{r["ΔR²"]:+.4f}'] for r in results]
tbl.insert(0, ['全模型', len(all_feats), f'{rmse_f:.4f}', '-', '-', f'{r2_f:.4f}', '-'])
table = ax.table(cellText=tbl, colLabels=col_l, cellLoc='center', loc='center')
table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 1.6)
for j in range(len(col_l)):
    table[0, j].set_text_props(color='white', weight='bold')
    table[0, j].set_facecolor('#2c3e50')
ax.set_title('Q5.1 消融实验汇总表', fontweight='bold', fontsize=12, pad=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'Q5_汇总表.png'), dpi=150, bbox_inches='tight')
plt.close(fig); print(f'[图] Q5_汇总表.png')

# ──────────────── 6. 终端结论输出 ────────────────
print('\n'+'='*60)
print('  Q5 建模结论')
print('='*60)
print(f'\n【Q5.1 最优变量组合】')
print(f'全特征模型 (共{len(all_feats)}项): RMSE={rmse_f:.4f}mm, R²={r2_f:.4f}')
print(f'影响最小变量: {best["变量"]} (剔除后 R² 仅下降 {best["ΔR²"]:.4f})')
print(f'影响最大变量: {worst["变量"]} (剔除后 R² 下降 {worst["ΔR²"]:.4f})')
print(f'最优组合 = 全特征 或 剔除{best["变量"]}（可根据实际监测条件取舍）')

print(f'\n【Q5.2 变量重要度排序】')
for r in sorted(results, key=lambda x: -x['ΔR²']):
    flag = '★重要' if r['ΔR²'] > np.median([x['ΔR²'] for x in results]) else ''
    print(f'  {r["变量"]:8s}: ΔR²={r["ΔR²"]:+.4f}  {flag}')

vel_phases = {}
for pid, nm in [(0,'缓慢变形'),(1,'加速变形'),(2,'快速变形')]:
    v = n_phase[pid]
    if v > 0: vel_phases[nm] = (df.loc[df['Phase']==pid, 'Displacement'].max() -
                                 df.loc[df['Phase']==pid, 'Displacement'].min()) / (v * 10/60) if v>1 else 0
print(f'\n【阶段平均速度 (mm/h)】')
for nm, v in vel_phases.items():
    print(f'  {nm}: {v:.4f} mm/h')

print(f'\n【预警阈值建议】')
for nm, v in vel_phases.items():
    if nm == '缓慢变形':
        print(f'  • {nm}: v≈{v:.4f} mm/h → 正常监测，阈值 v > {v*2:.4f}')
    elif nm == '加速变形':
        print(f'  • {nm}: v≈{v:.4f} mm/h → 黄色预警，阈值 v > {v*0.8:.4f}')
    elif nm == '快速变形':
        print(f'  • {nm}: v≈{v:.4f} mm/h → 红色预警，阈值 v > {v*0.5:.4f}')
print(f'  原理: 边坡失稳前位移速度呈指数增长，阈值设为各阶段平均速度的 0.5~2 倍')

print(f'\n图表已保存至: {os.path.abspath(OUT_DIR)}')
print('='*60)
