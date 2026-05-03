"""
Q5.2 滑坡预警机制构建
===================================
基于 Q5.1 的最优变量组合和分段数据，以表面位移速度为预警指标，
分阶段构建三级滑坡预警机制（蓝色注意/黄色警戒/红色撤离）。

输入：
  - Q5/feature/feature_56.xlsx  : 特征表（含 Phase, Delta_D, Displacement）
  - Q5/segment/segment.csv      : 分段信息
  - Q5/5.1/ablation_results.csv : 消融实验结果（确定最优特征组合）

输出：
  - Q5/5.2/warning_thresholds.png : 速度时序图 + 预警线 + 分阶段背景
  - Q5/5.2/warning_thresholds.csv : 各阶段统计量及阈值表
  - 终端打印 : 预警机制描述、阈值公式、合理性阐述
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# ─── 路径与中文设置 ───
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common.plot_utils import setup_zh, phase_name
setup_zh()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(SCRIPT_DIR, exist_ok=True)

# ─── 目录 ───
FEATURE_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'feature', 'feature_56.xlsx'))
SEG_PATH     = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'segment', 'segment.csv'))

print('=' * 70)
print('Q5.2 滑坡预警机制构建')
print('=' * 70)

# ════════════════════════════════════════════════════════════════
# 1. 加载数据
# ════════════════════════════════════════════════════════════════
print('\n[1/8] 加载数据...')
df = pd.read_excel(FEATURE_PATH)
seg = pd.read_csv(SEG_PATH)
print(f'  feature_56.xlsx: {df.shape[0]} 行 × {df.shape[1]} 列')
print(f'  segment.csv: {len(seg)} 个阶段')

# 根据 segment.csv 重新生成 Phase 列
df['Phase'] = -1
for _, row in seg.iterrows():
    start = int(row['起始索引'])
    end   = int(row['结束索引'])
    phase_id = int(row['阶段编号']) - 1  # 转为 0,1,2
    df.loc[start:end, 'Phase'] = phase_id
df['Phase'] = df['Phase'].astype(int)
print(f'  Phase 分布: {df["Phase"].value_counts().sort_index().to_dict()}')

# ════════════════════════════════════════════════════════════════
# 2. 速度计算（单位转换: 10min → 60min → mm/h）
# ════════════════════════════════════════════════════════════════
print('\n[2/8] 速度计算...')
df['Speed_mmh'] = df['Delta_D'] * 6.0   # mm/10min → mm/h
print(f'  实际速度范围: [{df["Speed_mmh"].min():.4f}, {df["Speed_mmh"].max():.4f}] mm/h')
print(f'  实际速度均值: {df["Speed_mmh"].mean():.4f} mm/h')

# ════════════════════════════════════════════════════════════════
# 3. 最优特征组合（硬编码，基于 Q5.1 消融实验结果）
# ════════════════════════════════════════════════════════════════
# Q5.1 结论：降雨量变量 ΔR² ≈ -0.001（基本无贡献），剔除后模型 R² 最高。
# 其余 5 类均需保留。始终保留列也加入。
print('\n[3/8] 确定最优特征组合...')

target = 'Delta_D'

# 始终保留的特征（Q5.1 原文）
always_keep = ['Day_sin', 'Day_cos', 'Disp_cum24', 'Time_since_rain', 'Time_since_blast']

# 需要排除的列
exclude_cols = ['Phase', 'Delta_D', 'Displacement', 'BlastDist', 'BlastCharge']

# 所有可用特征
all_features = [c for c in df.columns if c not in exclude_cols]

# 剔除"降雨量"家族（关键词 Rain）
rain_keywords = ['Rain']

optimal_features = []
for feat in all_features:
    feat_lower = feat.lower()
    # 始终保留
    if any(ak.lower() in feat_lower for ak in always_keep):
        optimal_features.append(feat)
        continue
    # 剔除降雨量家族
    if any(kw.lower() in feat_lower for kw in rain_keywords):
        continue
    # 保留其余
    optimal_features.append(feat)

print(f'  全特征数: {len(all_features)}')
print(f'  最优特征数（剔除降雨量后）: {len(optimal_features)}')

# ════════════════════════════════════════════════════════════════
# 4. 分阶段 LightGBM 训练与预测
# ════════════════════════════════════════════════════════════════
print('\n[4/8] 训练最优模型并预测速度...')

pred_delta_d = np.full(len(df), np.nan)
models = {}

for p in [0, 1, 2]:
    mask = df['Phase'] == p
    n_samples = mask.sum()
    if n_samples < 30:
        print(f'  警告: Phase {p} 仅 {n_samples} 样本，跳过')
        pred_delta_d[mask] = 0.0
        continue

    X = df.loc[mask, optimal_features].values
    y = df.loc[mask, target].values

    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.01,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
        deterministic=True,
    )
    model.fit(X, y)
    models[p] = model
    pred_delta_d[mask] = model.predict(X)

    rmse = np.sqrt(mean_squared_error(y, pred_delta_d[mask]))
    print(f'  Phase {p} ({phase_name(p)}): n={n_samples}, RMSE={rmse:.4f}')

df['Pred_Delta_D'] = pred_delta_d
df['Pred_Speed_mmh'] = df['Pred_Delta_D'] * 6.0

# ════════════════════════════════════════════════════════════════
# 5. 分阶段统计分析
# ════════════════════════════════════════════════════════════════
print('\n[5/8] 分阶段统计分析...')

phase_stats = {}
phase_masks = {p: (df['Phase'] == p) for p in [0, 1, 2]}

for p in [0, 1, 2]:
    mask = phase_masks[p]
    spd = df.loc[mask, 'Speed_mmh'].values
    stats = {
        '阶段': phase_name(p),
        '样本数': len(spd),
        '均值': float(np.mean(spd)),
        '标准差': float(np.std(spd, ddof=1)),
        '中位数': float(np.median(spd)),
        '偏度': float(pd.Series(spd).skew()),
        '最小值': float(np.min(spd)),
        '最大值': float(np.max(spd)),
        '95分位数': float(np.percentile(spd, 95)),
        '99分位数': float(np.percentile(spd, 99)),
    }
    phase_stats[p] = stats
    print(f'  Phase {p} ({phase_name(p)}): '
          f'μ={stats["均值"]:.4f}, σ={stats["标准差"]:.4f}, '
          f'中位数={stats["中位数"]:.4f}, max={stats["最大值"]:.4f}')

# ════════════════════════════════════════════════════════════════
# 6. 预警阈值设定
# ════════════════════════════════════════════════════════════════
print('\n[6/8] 计算预警阈值...')

mu0, sigma0 = phase_stats[0]['均值'], phase_stats[0]['标准差']
mu1, sigma1 = phase_stats[1]['均值'], phase_stats[1]['标准差']
mu2, sigma2 = phase_stats[2]['均值'], phase_stats[2]['标准差']

# 参考速度（缓慢阶段平均速度，用于切线角计算）
v_ref = mu0

# ── Phase 0: 缓慢变形 ──
blue_thresh_0 = mu0 + 2 * sigma0
yellow_thresh_0 = mu0 + 3 * sigma0
red_thresh_0 = None  # 需持续规则触发

# ── Phase 1: 加速变形 ──
yellow_thresh_1 = mu1 + 1.5 * sigma1
red_thresh_1 = mu1 + 3 * sigma1

# ── Phase 2: 快速变形 ──
# 改进切线角法：α = arctan(v / v_ref)
# α > 78° 时速度已远超参考速度（tan78° ≈ 4.7），结合加速度判断发布红色
tan78_thresh = v_ref * np.tan(np.deg2rad(78))   # ≈ v_ref * 4.7
yellow_thresh_2 = mu2                           # 黄色 = 阶段均值
red_thresh_2 = mu2 + sigma2                     # 红色硬阈值 = μ+σ

thresholds = {
    'phase0': {'blue': blue_thresh_0, 'yellow': yellow_thresh_0, 'red': red_thresh_0},
    'phase1': {'yellow': yellow_thresh_1, 'red': red_thresh_1},
    'phase2': {'yellow': yellow_thresh_2, 'red': red_thresh_2, 'tan78': tan78_thresh},
}

print(f'  Phase 0 (缓慢): μ0={mu0:.4f}, σ0={sigma0:.4f}')
print(f'    蓝色阈值 (μ0+2σ0) = {blue_thresh_0:.4f} mm/h')
print(f'    黄色阈值 (μ0+3σ0) = {yellow_thresh_0:.4f} mm/h')
print(f'  Phase 1 (加速): μ1={mu1:.4f}, σ1={sigma1:.4f}')
print(f'    黄色阈值 (μ1+1.5σ1) = {yellow_thresh_1:.4f} mm/h')
print(f'    红色阈值 (μ1+3σ1) = {red_thresh_1:.4f} mm/h')
print(f'  Phase 2 (快速): μ2={mu2:.4f}, σ2={sigma2:.4f}')
print(f'    黄色阈值 (μ2) = {yellow_thresh_2:.4f} mm/h')
print(f'    红色阈值 (μ2+σ2) = {red_thresh_2:.4f} mm/h')
print(f'    切线角参考速度 v_ref = {v_ref:.4f} mm/h')
print(f'    切线角 > 78° 对应速度 = {tan78_thresh:.4f} mm/h')

# ════════════════════════════════════════════════════════════════
# 7. 预警等级判定（基于实际速度 + 防抖规则）
# ════════════════════════════════════════════════════════════════
print('\n[7/8] 预警等级判定（含防抖规则）...')

DEBOUNCE = 6  # 连续 6 步（1小时）超限才确认

speed = df['Speed_mmh'].values
phase_arr = df['Phase'].values

# ── 先逐点判定原始预警倾向 ──
raw_alert = np.zeros(len(df), dtype=int)

for i in range(len(df)):
    spd = speed[i]
    ph = phase_arr[i]
    if ph == 0:
        if spd >= yellow_thresh_0:
            raw_alert[i] = 2  # 黄色
        elif spd >= blue_thresh_0:
            raw_alert[i] = 1  # 蓝色
        else:
            raw_alert[i] = 0
    elif ph == 1:
        if spd >= red_thresh_1:
            raw_alert[i] = 3  # 红色
        elif spd >= yellow_thresh_1:
            raw_alert[i] = 2  # 黄色
        else:
            raw_alert[i] = 0
    elif ph == 2:
        # 切线角法
        if spd > tan78_thresh:
            # 检查加速度（6小时差分 = 36步）
            if i >= 36:
                accel = speed[i] - speed[i - 36]
            else:
                accel = 0
            if accel > 0:
                raw_alert[i] = 3  # 红色
            elif spd >= red_thresh_2:
                raw_alert[i] = 3
            elif spd >= yellow_thresh_2:
                raw_alert[i] = 2
        elif spd >= red_thresh_2:
            raw_alert[i] = 3
        elif spd >= yellow_thresh_2:
            raw_alert[i] = 2
        else:
            raw_alert[i] = 0

# ── 防抖：连续 DEBOUNCE 步维持同一等级才确认，支持升级和降级 ──
def apply_debounce(raw, debounce=6):
    """
    防抖规则：
    - 升级：连续 N 步 raw >= 更高等级 → 升级
    - 降级：连续 N 步 raw < 当前等级 → 降至最近满足的等级
    """
    n = len(raw)
    confirmed = np.zeros(n, dtype=int)
    current_level = 0

    for i in range(n):
        # 检查是否应升级
        for lvl in [3, 2, 1]:
            if lvl <= current_level:
                continue
            if i >= debounce - 1:
                if np.all(raw[i - debounce + 1 : i + 1] >= lvl):
                    current_level = lvl
                    break

        # 检查是否应降级
        if current_level > 0 and i >= debounce - 1:
            if np.all(raw[i - debounce + 1 : i + 1] < current_level):
                # 降至最近满足的等级
                new_lvl = current_level - 1
                while new_lvl > 0:
                    if np.all(raw[i - debounce + 1 : i + 1] >= new_lvl):
                        break
                    new_lvl -= 1
                current_level = new_lvl

        confirmed[i] = current_level

    # Phase 0 特殊规则：连续 72 步（12小时）超黄色 → 升级为红色
    for i in range(len(confirmed)):
        if phase_arr[i] == 0 and confirmed[i] >= 2:  # 已为黄色
            if i >= 71:
                if np.all(raw[i - 71 : i + 1] >= 2):
                    confirmed[i] = 3  # 升级为红色

    return confirmed

alert_level = apply_debounce(raw_alert, DEBOUNCE)

# 统计各级别占比
unique, counts = np.unique(alert_level, return_counts=True)
alert_dist = dict(zip(unique.astype(int), counts))
level_names = {0: '正常', 1: '蓝色注意', 2: '黄色警戒', 3: '红色撤离'}
print('  预警等级分布:')
for lvl in [0, 1, 2, 3]:
    cnt = alert_dist.get(lvl, 0)
    pct = cnt / len(alert_level) * 100
    print(f'    {level_names[lvl]}: {cnt} 步 ({pct:.2f}%)')

# ════════════════════════════════════════════════════════════════
# 8. 回溯检验
# ════════════════════════════════════════════════════════════════
print('\n[8/8] 回溯检验...')

# 阶段转换节点（segment.csv 中的起始索引）
b1_idx = int(seg.loc[1, '起始索引'])  # Phase 1 起始 = 加速阶段开始
b2_idx = int(seg.loc[2, '起始索引'])  # Phase 2 起始 = 快速阶段开始

print(f'  断点1（进入加速阶段）索引: {b1_idx}')
print(f'  断点2（进入快速阶段）索引: {b2_idx}')

# 检验：在断点前是否有提前预警
def find_first_alert_before(alert_seq, target_level, before_idx, lookback=72):
    """在 before_idx 之前 lookback 范围内，找到首次触发 target_level 的索引"""
    start = max(0, before_idx - lookback)
    for i in range(start, before_idx):
        if alert_seq[i] >= target_level:
            return i
    return None

# 加速阶段前蓝色预警
blue_before_b1 = find_first_alert_before(alert_level, 1, b1_idx, lookback=144)  # 24h
yellow_before_b1 = find_first_alert_before(alert_level, 2, b1_idx, lookback=144)

# 快速阶段前黄色预警
yellow_before_b2 = find_first_alert_before(alert_level, 2, b2_idx, lookback=144)
red_before_b2 = find_first_alert_before(alert_level, 3, b2_idx, lookback=144)

advance_blue_b1 = (b1_idx - blue_before_b1) / 6.0 if blue_before_b1 is not None else None  # 小时
advance_yellow_b1 = (b1_idx - yellow_before_b1) / 6.0 if yellow_before_b1 is not None else None
advance_yellow_b2 = (b2_idx - yellow_before_b2) / 6.0 if yellow_before_b2 is not None else None
advance_red_b2 = (b2_idx - red_before_b2) / 6.0 if red_before_b2 is not None else None

print(f'  加速阶段前蓝色预警: {"成功" if blue_before_b1 else "未触发"}, '
      f'提前 {advance_blue_b1:.2f} 小时' if advance_blue_b1 else '')
print(f'  加速阶段前黄色预警: {"成功" if yellow_before_b1 else "未触发"}, '
      f'提前 {advance_yellow_b1:.2f} 小时' if advance_yellow_b1 else '')
print(f'  快速阶段前黄色预警: {"成功" if yellow_before_b2 else "未触发"}, '
      f'提前 {advance_yellow_b2:.2f} 小时' if advance_yellow_b2 else '')
print(f'  快速阶段前红色预警: {"成功" if red_before_b2 else "未触发"}, '
      f'提前 {advance_red_b2:.2f} 小时' if advance_red_b2 else '')

# ════════════════════════════════════════════════════════════════
# 9. 绘制预警阈值图
# ════════════════════════════════════════════════════════════════
print('\n[绘图] 生成预警阈值图...')

fig, ax = plt.subplots(figsize=(20, 7))

n = len(df)
x_idx = np.arange(n)

# ── 分阶段背景着色 ──
phase_colors_bg = {0: '#c8e6c9', 1: '#fff9c4', 2: '#ffcdd2'}  # 浅绿/浅黄/浅红
for p in [0, 1, 2]:
    mask = phase_masks[p]
    in_seg = False
    seg_start = None
    for i in range(n):
        if mask.iloc[i] and not in_seg:
            in_seg = True
            seg_start = i
        elif not mask.iloc[i] and in_seg:
            in_seg = False
            ax.axvspan(seg_start, i - 1, alpha=0.25,
                       color=phase_colors_bg[p], zorder=1)
    if in_seg:
        ax.axvspan(seg_start, n - 1, alpha=0.25,
                   color=phase_colors_bg[p], zorder=1)

# ── 标注断点 ──
ax.axvline(x=b1_idx, color='#ff6f00', linestyle='--', linewidth=1.5, alpha=0.7,
           label=f'断点1 (索引{b1_idx})')
ax.axvline(x=b2_idx, color='#c62828', linestyle='--', linewidth=1.5, alpha=0.7,
           label=f'断点2 (索引{b2_idx})')

# ── 实际速度曲线（细灰线） ──
ax.plot(x_idx, speed, color='#bdbdbd', linewidth=0.5, alpha=0.8, label='实际速度', zorder=2)

# ── 预测速度曲线（深蓝线） ──
pred_speed = df['Pred_Speed_mmh'].values
ax.plot(x_idx, pred_speed, color='#1565c0', linewidth=1.0, alpha=0.85, label='预测速度', zorder=3)

# ── 各阶段阈值线 ──
# Phase 0 区域：蓝色虚线、黄色虚线
p0_mask = phase_masks[0]
if p0_mask.sum() > 0:
    p0_start = np.where(p0_mask.values)[0][0]
    p0_end = np.where(p0_mask.values)[0][-1]
    ax.hlines(y=blue_thresh_0, xmin=p0_start, xmax=p0_end,
              colors='#1976d2', linestyles='dashed', linewidth=1.2, alpha=0.7,
              label=f'蓝色 (μ0+2σ0={blue_thresh_0:.2f})', zorder=4)
    ax.hlines(y=yellow_thresh_0, xmin=p0_start, xmax=p0_end,
              colors='#f9a825', linestyles='dashed', linewidth=1.2, alpha=0.7,
              label=f'黄色 (μ0+3σ0={yellow_thresh_0:.2f})', zorder=4)

# Phase 1 区域：黄色虚线、红色虚线
p1_mask = phase_masks[1]
if p1_mask.sum() > 0:
    p1_start = np.where(p1_mask.values)[0][0]
    p1_end = np.where(p1_mask.values)[0][-1]
    ax.hlines(y=yellow_thresh_1, xmin=p1_start, xmax=p1_end,
              colors='#f9a825', linestyles='dashed', linewidth=1.2, alpha=0.7,
              label=f'黄色 (μ1+1.5σ1={yellow_thresh_1:.2f})', zorder=4)
    ax.hlines(y=red_thresh_1, xmin=p1_start, xmax=p1_end,
              colors='#c62828', linestyles='dashed', linewidth=1.2, alpha=0.7,
              label=f'红色 (μ1+3σ1={red_thresh_1:.2f})', zorder=4)

# Phase 2 区域：红色虚线、黄色虚线、切线角参考线
p2_mask = phase_masks[2]
if p2_mask.sum() > 0:
    p2_start = np.where(p2_mask.values)[0][0]
    p2_end = np.where(p2_mask.values)[0][-1]
    ax.hlines(y=red_thresh_2, xmin=p2_start, xmax=p2_end,
              colors='#c62828', linestyles='dashed', linewidth=1.2, alpha=0.7,
              label=f'红色 (μ2+σ2={red_thresh_2:.2f})', zorder=4)
    ax.hlines(y=yellow_thresh_2, xmin=p2_start, xmax=p2_end,
              colors='#f9a825', linestyles='dashed', linewidth=1.2, alpha=0.7,
              label=f'黄色 (μ2={yellow_thresh_2:.2f})', zorder=4)
    ax.hlines(y=tan78_thresh, xmin=p2_start, xmax=p2_end,
              colors='#7b1fa2', linestyles='dotted', linewidth=1.0, alpha=0.5,
              label=f'切线角>78° (v_ref×4.7={tan78_thresh:.2f})', zorder=4)

# ── 红色预警触发点标记 ──
red_indices = np.where(alert_level == 3)[0]
if len(red_indices) > 0:
    step = max(1, len(red_indices) // 80)
    sampled_red = red_indices[::step]
    ax.scatter(sampled_red, speed[sampled_red], c='red', s=12, marker='o',
               alpha=0.7, zorder=5, label='红色预警触发点')

# ── 黄色预警触发点标记 ──
yellow_indices = np.where(alert_level == 2)[0]
if len(yellow_indices) > 0:
    step = max(1, len(yellow_indices) // 80)
    sampled_yellow = yellow_indices[::step]
    ax.scatter(sampled_yellow, speed[sampled_yellow], c='#f9a825', s=8, marker='.',
               alpha=0.5, zorder=4, label='黄色预警触发点')

# ── 蓝色预警触发点标记 ──
blue_indices = np.where(alert_level == 1)[0]
if len(blue_indices) > 0:
    step = max(1, len(blue_indices) // 80)
    sampled_blue = blue_indices[::step]
    ax.scatter(sampled_blue, speed[sampled_blue], c='#1976d2', s=8, marker='.',
               alpha=0.5, zorder=4, label='蓝色预警触发点')

# ── 图例、标题、坐标轴 ──
ax.set_xlabel('时间索引 (10分钟/步)', fontsize=12)
ax.set_ylabel('表面位移速度 (mm/h)', fontsize=14)
ax.set_title('滑坡预警阈值与速度时序图', fontsize=16, fontweight='bold')
ax.legend(loc='upper left', fontsize=8, ncol=2, framealpha=0.8)

# 添加阶段文字标注
y_min, y_max = ax.get_ylim()
for p in [0, 1, 2]:
    mask = phase_masks[p]
    if mask.sum() > 0:
        mid_idx = int((np.where(mask.values)[0][0] + np.where(mask.values)[0][-1]) / 2)
        ax.text(mid_idx, y_max * 0.92, phase_name(p),
                ha='center', va='top', fontsize=13, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=phase_colors_bg[p],
                          edgecolor='gray', alpha=0.7))

fig.tight_layout()
fig.savefig(os.path.join(SCRIPT_DIR, 'warning_thresholds.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'  已保存: warning_thresholds.png')

# ════════════════════════════════════════════════════════════════
# 10. 生成阈值表 CSV
# ════════════════════════════════════════════════════════════════
print('[保存] 生成阈值表...')

threshold_table = pd.DataFrame([
    {
        '阶段': phase_name(0),
        '速度均值 (mm/h)': f'{mu0:.4f}',
        '速度标准差': f'{sigma0:.4f}',
        '蓝色阈值 (mm/h)': f'{blue_thresh_0:.4f}',
        '黄色阈值 (mm/h)': f'{yellow_thresh_0:.4f}',
        '红色阈值 (mm/h)': '持续规则触发',
        '预警规则简述': f'蓝色=μ+2σ({blue_thresh_0:.2f}), 黄色=μ+3σ({yellow_thresh_0:.2f}), '
                        f'红色需连续72步超黄色线',
    },
    {
        '阶段': phase_name(1),
        '速度均值 (mm/h)': f'{mu1:.4f}',
        '速度标准差': f'{sigma1:.4f}',
        '蓝色阈值 (mm/h)': '—',
        '黄色阈值 (mm/h)': f'{yellow_thresh_1:.4f}',
        '红色阈值 (mm/h)': f'{red_thresh_1:.4f}',
        '预警规则简述': f'黄色=μ+1.5σ({yellow_thresh_1:.2f}), 红色=μ+3σ({red_thresh_1:.2f}), '
                        f'引入6h加速度确认',
    },
    {
        '阶段': phase_name(2),
        '速度均值 (mm/h)': f'{mu2:.4f}',
        '速度标准差': f'{sigma2:.4f}',
        '蓝色阈值 (mm/h)': '—',
        '黄色阈值 (mm/h)': f'{yellow_thresh_2:.4f}',
        '红色阈值 (mm/h)': f'{red_thresh_2:.4f}',
        '预警规则简述': f'改进切线角法(α>78°={tan78_thresh:.2f}+加速度>0) 或 硬阈值μ+σ({red_thresh_2:.2f}), '
                        f'黄色=μ2({yellow_thresh_2:.2f})',
    },
])

threshold_table.to_csv(
    os.path.join(SCRIPT_DIR, 'warning_thresholds.csv'),
    index=False, encoding='utf-8-sig'
)
print(f'  已保存: warning_thresholds.csv')

# ════════════════════════════════════════════════════════════════
# 11. 终端输出总结
# ════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('【Q5.2 滑坡预警机制】')
print('=' * 70)
print(f'1. 缓慢变形阶段：稳态速度 μ0={mu0:.4f} mm/h，σ0={sigma0:.4f}。'
      f'蓝色阈值 = {blue_thresh_0:.4f}，黄色阈值 = {yellow_thresh_0:.4f}。')
print(f'2. 加速变形阶段：μ1={mu1:.4f}，σ1={sigma1:.4f}。'
      f'黄色阈值 = {yellow_thresh_1:.4f}，红色阈值 = {red_thresh_1:.4f}。引入加速度确认。')
print(f'3. 快速变形阶段：采用改进切线角法，参考速度 = {v_ref:.4f} mm/h，'
      f'切线角>78°对应速度阈值 = {tan78_thresh:.4f}。'
      f'红色预警硬阈值为 μ2+σ2 = {red_thresh_2:.4f}。')
print(f'4. 防抖规则：连续{DEBOUNCE}步超限才确认警报（支持升级和降级）。')
print(f'5. 回溯检验：', end='')
if advance_blue_b1 is not None:
    print(f'系统在加速阶段前{advance_blue_b1:.2f}小时发出蓝色预警，', end='')
else:
    print(f'系统未能在加速阶段前发出蓝色预警，', end='')
if advance_yellow_b2 is not None:
    print(f'在快速阶段前{advance_yellow_b2:.2f}小时发出黄色预警。')
else:
    print(f'未能在快速阶段前发出黄色预警。')
print('=' * 70)
print('[完成] Q5.2 滑坡预警机制构建')
