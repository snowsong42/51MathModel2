"""
Q5.1 消融实验与变量组合优化
===================================
基于 feature_56.xlsx + segment.csv，使用 LightGBM 分阶段训练
评估 6 类变量（降雨量、孔隙水压力、微震事件数、干湿入渗系数、
爆破点距离、单段最大药量）对位移预测的贡献度。

输出：
  - ablation_results.csv   : 各组合指标（ΔR^2、RMSE、分阶段 RMSE）
  - ablation_bar.png       : ΔR^2 柱状图
  - ablation_heatmap.png   : 分阶段 RMSE 变化热力图
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# ─── 设置中文 ───
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common.plot_utils import setup_zh
setup_zh()

# ─── 目录 ───
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, '..', '5.1'))
DATA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'feature'))
SEG_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'segment'))
os.makedirs(OUT_DIR, exist_ok=True)

print('=' * 60)
print('Q5.1 消融实验与变量组合优化')
print('=' * 60)

# ════════════════════════════════════════════════════════════════
# 1. 加载数据
# ════════════════════════════════════════════════════════════════
print('\n[1/6] 加载数据...')
df = pd.read_excel(os.path.join(DATA_DIR, 'feature_56.xlsx'))
seg = pd.read_csv(os.path.join(SEG_DIR, 'segment.csv'))
print(f'  feature_56.xlsx: {df.shape[0]} 行 × {df.shape[1]} 列')
print(f'  segment.csv: {len(seg)} 个阶段')

# ── 根据 segment.csv 重新生成 Phase 列 ──
df['Phase'] = -1
for _, row in seg.iterrows():
    start = int(row['起始索引'])
    end = int(row['结束索引'])
    phase_id = int(row['阶段编号']) - 1  # 转为 0,1,2
    df.loc[start:end, 'Phase'] = phase_id

# 检查 Phase 是否完整
if (df['Phase'] == -1).any():
    print('  警告: 部分样本未被分配到任何阶段！')
df['Phase'] = df['Phase'].astype(int)
print(f'  Phase 分布: {df["Phase"].value_counts().sort_index().to_dict()}')

# ════════════════════════════════════════════════════════════════
# 2. 定义变量分类
# ════════════════════════════════════════════════════════════════
print('\n[2/6] 定义变量分类...')

target = 'Delta_D'

# 始终保留的特征（不参与剔除）
always_keep = ['Day_sin', 'Day_cos', 'Disp_cum24', 'Time_since_rain', 'Time_since_blast']

# 需排除的列（目标、原始变量、阶段标记）
exclude_cols = ['Phase', 'Delta_D', 'Displacement', 'BlastDist', 'BlastCharge']

# 6 个变量家族（关键字匹配）
families = {
    '降雨量':       ['Rain'],
    '孔隙水压力':   ['Pore'],
    '微震事件数':   ['Micro'],
    '干湿入渗系数': ['Infilt'],
    '爆破点距离':   ['BlastDist'],
    '单段最大药量': ['BlastCharge'],
}

# 爆破共享依赖列（不含 BlastDist/BlastCharge 关键字，但属爆破衍生）
blast_shared = ['Blast_PPV', 'Blast_Energy', 'BlastEnergy_decay_impact', 'Blast_interval']

# 全部可用特征（排除无关列后）
all_features = [c for c in df.columns if c not in exclude_cols]
print(f'  可用特征数: {len(all_features)}')
print(f'  始终保留: {always_keep}')

# 打印各家族匹配情况
for fam_name, keywords in families.items():
    matched = [f for f in all_features
               if any(kw.lower() in f.lower() for kw in keywords)
               and not any(ak in f for ak in always_keep)]
    print(f'  {fam_name:>8s} ({keywords}): {len(matched)} 列 → {matched}')


def get_features_for_family(all_features, exclude_keywords):
    """
    剔除指定家族关键词后，返回保留的特征列表。

    规则：
      1. 包含 always_keep 中任一关键字 → 保留
      2. 包含 exclude_keywords 中任一关键字（不区分大小写）→ 剔除
      3. 其余保留
    """
    kept = []
    for feat in all_features:
        feat_lower = feat.lower()

        # 始终保留
        if any(ak.lower() in feat_lower for ak in always_keep):
            kept.append(feat)
            continue

        # 需要剔除
        if any(kw.lower() in feat_lower for kw in exclude_keywords):
            continue

        kept.append(feat)
    return kept


# ════════════════════════════════════════════════════════════════
# 3. 训练函数
# ════════════════════════════════════════════════════════════════
def train_and_predict(df, features, target='Delta_D', seed=42):
    """
    分阶段训练 LightGBM，返回全序列预测值。

    对 Phase 0/1/2 分别训练独立的 LGBMRegressor，
    然后预测对应阶段的全部样本，拼接为完整序列。
    """
    preds = np.full(len(df), np.nan)
    models = {}
    phase_rmses = {}

    for p in [0, 1, 2]:
        mask = df['Phase'] == p
        n_samples = mask.sum()
        if n_samples < 30:
            print(f'    警告: Phase {p} 仅 {n_samples} 个样本，跳过')
            preds[mask] = 0.0
            phase_rmses[p] = np.nan
            continue

        X = df.loc[mask, features].values
        y = df.loc[mask, target].values

        model = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.01,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            verbose=-1,
            deterministic=True,
        )
        model.fit(X, y)
        models[p] = model

        preds[mask] = model.predict(X)
        phase_rmses[p] = np.sqrt(mean_squared_error(y, preds[mask]))

    return preds, models, phase_rmses


# ════════════════════════════════════════════════════════════════
# 4. 全特征基线
# ════════════════════════════════════════════════════════════════
print('\n[3/6] 计算全特征基线...')

N_REPEATS = 3
baseline_r2_list, baseline_rmse_list = [], []
baseline_phase_rmse_list = []

for rep in range(N_REPEATS):
    seed = 42 + rep
    preds, _, phase_rmse = train_and_predict(df, all_features, target, seed)
    rmse = np.sqrt(mean_squared_error(df[target], preds))
    r2 = r2_score(df[target], preds)

    baseline_r2_list.append(r2)
    baseline_rmse_list.append(rmse)
    baseline_phase_rmse_list.append(phase_rmse)

baseline_r2 = float(np.mean(baseline_r2_list))
baseline_rmse = float(np.mean(baseline_rmse_list))
baseline_phase_rmse = {
    p: float(np.mean([d[p] for d in baseline_phase_rmse_list]))
    for p in [0, 1, 2]
}

print(f'  基线 R^2  = {baseline_r2:.6f} (±{np.std(baseline_r2_list):.6f})')
print(f'  基线 RMSE = {baseline_rmse:.4f}')
for p in [0, 1, 2]:
    print(f'  Phase {p} RMSE = {baseline_phase_rmse[p]:.4f}')


# ════════════════════════════════════════════════════════════════
# 5. 消融实验
# ════════════════════════════════════════════════════════════════
print('\n[4/6] 执行消融实验（逐类剔除）...')

results = []

for fam_name, keywords in families.items():
    print(f'\n  >>> 剔除 [{fam_name}] (关键词: {keywords})')

    r2_list, rmse_list = [], []
    phase_delta_list = []

    for rep in range(N_REPEATS):
        seed = 42 + rep
        kept = get_features_for_family(all_features, keywords)

        # 剔除爆破家族时，同时移除共享的爆破衍生列
        if fam_name in ('爆破点距离', '单段最大药量'):
            kept = [f for f in kept if f not in blast_shared]

        preds, _, phase_rmse = train_and_predict(df, kept, target, seed)
        rmse = float(np.sqrt(mean_squared_error(df[target], preds)))
        r2 = float(r2_score(df[target], preds))

        r2_list.append(r2)
        rmse_list.append(rmse)
        phase_delta_list.append({
            p: phase_rmse[p] - baseline_phase_rmse[p]
            for p in [0, 1, 2]
        })

    avg_r2 = float(np.mean(r2_list))
    avg_rmse = float(np.mean(rmse_list))
    delta_r2 = baseline_r2 - avg_r2
    avg_phase_delta = {
        p: float(np.mean([d[p] for d in phase_delta_list]))
        for p in [0, 1, 2]
    }

    print(f'    剔除后 R^2 = {avg_r2:.6f}, ΔR^2 = {delta_r2:.6f}')
    print(f'    分阶段 RMSE 变化: '
          f'Phase0={avg_phase_delta[0]:+.4f}, '
          f'Phase1={avg_phase_delta[1]:+.4f}, '
          f'Phase2={avg_phase_delta[2]:+.4f}')

    results.append({
        '剔除家族': fam_name,
        'R^2': f'{avg_r2:.6f}',
        'ΔR^2': f'{delta_r2:.6f}',
        'RMSE': f'{avg_rmse:.4f}',
        'Phase0_RMSE_差值': f'{avg_phase_delta[0]:+.4f}',
        'Phase1_RMSE_差值': f'{avg_phase_delta[1]:+.4f}',
        'Phase2_RMSE_差值': f'{avg_phase_delta[2]:+.4f}',
    })

# ── 保存结果表 ──
res_df = pd.DataFrame(results)
res_df.to_csv(
    os.path.join(OUT_DIR, 'ablation_results.csv'),
    index=False,
    encoding='utf-8-sig'
)
print(f'\n  已保存: ablation_results.csv')


# ════════════════════════════════════════════════════════════════
# 6. 图表绘制
# ════════════════════════════════════════════════════════════════
print('\n[5/6] 绘制图表...')

# ── 图 1: ΔR^2 柱状图 ──
fam_labels = [r['剔除家族'] for r in results]
delta_vals = np.array([float(r['ΔR^2']) for r in results])

# 排序（降序）
sort_idx = np.argsort(delta_vals)[::-1]
fam_labels_sorted = [fam_labels[i] for i in sort_idx]
delta_vals_sorted = delta_vals[sort_idx]

# 颜色：重要 > 0.005 红色，中等 > 0.001 橙色，不重要绿色
colors = []
for v in delta_vals_sorted:
    if v >= 0.005:
        colors.append('#e74c3c')   # 红
    elif v >= 0.001:
        colors.append('#f39c12')   # 橙
    else:
        colors.append('#27ae60')   # 绿

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(fam_labels_sorted, delta_vals_sorted, color=colors,
              edgecolor='gray', linewidth=0.5, width=0.6)

# 标注数值
for bar, val in zip(bars, delta_vals_sorted):
    y_pos = bar.get_height() + max(delta_vals) * 0.02
    ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
            f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
ax.set_xlabel('剔除的变量类别', fontsize=12)
ax.set_ylabel('ΔR^2（下降幅度）', fontsize=12)
ax.set_title('剔除各类变量后 R^2 下降幅度', fontsize=14, fontweight='bold')
ax.set_ylim(min(delta_vals) - 0.002, max(delta_vals) * 1.25)

# 图注
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#e74c3c', label='重要 (ΔR^2 ≥ 0.005)'),
    Patch(facecolor='#f39c12', label='中等 (0.001 ≤ ΔR^2 < 0.005)'),
    Patch(facecolor='#27ae60', label='可忽略 (ΔR^2 < 0.001)'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'ablation_bar.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  已保存: ablation_bar.png')

# ── 图 2: 分阶段 RMSE 变化热力图 ──
heat_data = np.array([
    [float(r[f'Phase{p}_RMSE_差值']) for p in range(3)]
    for r in results
])

fig, ax = plt.subplots(figsize=(8, 5))
vlim = max(abs(heat_data.min()), abs(heat_data.max()))
im = ax.imshow(heat_data, cmap='RdBu_r', aspect='auto',
               vmin=-vlim, vmax=vlim)

# 轴标签
ax.set_xticks(range(3))
ax.set_xticklabels(['缓慢变形\n(Phase 0)', '加速变形\n(Phase 1)', '快速变形\n(Phase 2)'],
                   fontsize=11)
ax.set_yticks(range(len(fam_labels)))
ax.set_yticklabels(fam_labels, fontsize=11)

# 标注数值
for i in range(len(fam_labels)):
    for j in range(3):
        val = heat_data[i, j]
        text_color = 'white' if abs(val) > vlim * 0.6 else 'black'
        ax.text(j, i, f'{val:+.4f}', ha='center', va='center',
                fontsize=9, color=text_color, fontweight='bold')

ax.set_title('分阶段 RMSE 变化热力图\n（剔除后 全特征基线，正值表示变差）',
             fontsize=13, fontweight='bold')
fig.colorbar(im, ax=ax, label='RMSE 变化量', shrink=0.8)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'ablation_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  已保存: ablation_heatmap.png')


# ════════════════════════════════════════════════════════════════
# 7. 结论输出
# ════════════════════════════════════════════════════════════════
print('\n[6/6] 结论输出')
print('=' * 60)
print(f'全特征基线:')
print(f'  R^2  = {baseline_r2:.6f}')
print(f'  RMSE = {baseline_rmse:.4f}')
print(f'  分阶段 RMSE: Phase0={baseline_phase_rmse[0]:.4f}, '
      f'Phase1={baseline_phase_rmse[1]:.4f}, '
      f'Phase2={baseline_phase_rmse[2]:.4f}')

print(f'\n各变量剔除后的 ΔR^2（降序，越大越重要）：')
for i, (fam, val) in enumerate(zip(fam_labels_sorted, delta_vals_sorted)):
    if val >= 0.005:
        flag = ' ★★★ 重要'
    elif val >= 0.001:
        flag = ' ★★ 中等'
    else:
        flag = '   可忽略'
    print(f'  {i+1}. [{fam:>8s}]  ΔR^2 = {val:+.6f}{flag}')

# 最优剔除建议
least_idx = np.argmin(delta_vals)
least_fam = fam_labels[least_idx]
least_delta = delta_vals[least_idx]
least_r2 = float(res_df.loc[least_idx, 'R^2'])
least_rmse = float(res_df.loc[least_idx, 'RMSE'])

print(f'\n▶ 最优剔除建议：')
print(f'   剔除 [{least_fam}]（ΔR^2 = {least_delta:+.6f}，贡献最小）')
print(f'   剔除后全校正 R^2 = {least_r2:.6f}, RMSE = {least_rmse:.4f}')

# 推荐保留
important = [(fam, val) for fam, val in zip(fam_labels, delta_vals)
             if val >= 0.001]
removable = [(fam, val) for fam, val in zip(fam_labels, delta_vals)
             if val < 0.001]

print(f'\n▶ 推荐保留（ΔR^2 ≥ 0.001）：')
for fam, val in important:
    print(f'   ✓ {fam} (ΔR^2 = {val:+.6f})')
print(f'\n▶ 可考虑剔除（ΔR^2 < 0.001）：')
for fam, val in removable:
    print(f'   ✗ {fam} (ΔR^2 = {val:+.4f})')

print(f'\n{"=" * 60}')
print(f'全部结果已保存至: {OUT_DIR}')
print(f'  - ablation_results.csv')
print(f'  - ablation_bar.png')
print(f'  - ablation_heatmap.png')
print('=' * 60)
print('[完成] Q5.1 消融实验与变量优化')
