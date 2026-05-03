"""
问题 4.5：分阶段多元线性回归 —— 表面位移增量建模（仅训练集）
================================================================
读入 4.4 生成的 ap4_features.xlsx 的 train sheet，
按 Stage 列分组，对每个阶段用训练集拟合 ΔSD ~ R_eff + P_drive + M_cum + BlastMem，
输出：
  1) model_params.csv —— 三阶段模型参数与评价指标（含链式预测所需信息）
  2) fit_evaluation.png —— 总览：三阶段 SD 对比（对齐 MATLAB 样式）
  3) segN/stage_report.png —— 各阶段 6 面板详细诊断图
  4) 控制台 —— 全中文格式化报告

配色对齐 Q4_0/4.1/Assumption.m 的 MATLAB 脚本：
  阶段1（匀速）：背景浅绿 [0.7 1 0.7]，线条深绿 [0 0.6 0]
  阶段2（加速）：背景浅黄 [1 1 0.2]，线条深黄 [1 0.8 0.2]
  阶段3（快速）：背景浅红 [1 0.7 0.7]，线条深红 [0.8 0 0]
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ==================== 全局 matplotlib 设置 ====================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 路径设置 ====================
script_dir = os.path.dirname(os.path.abspath(__file__))
features_file = os.path.join(script_dir, "../4.4/ap4_features.xlsx")

seg_dirs = [
    os.path.join(script_dir, "seg1"),
    os.path.join(script_dir, "seg2"),
    os.path.join(script_dir, "seg3"),
]
for d in seg_dirs:
    os.makedirs(d, exist_ok=True)

output_params = os.path.join(script_dir, "model_params.csv")
output_plot   = os.path.join(script_dir, "fit_evaluation.png")

# ==================== 阶段信息（对齐 MATLAB 配色） ====================
stage_names    = ['第1段 · 匀速蠕变',  '第2段 · 加速变形',  '第3段 · 快速破坏']
stage_bg_rgb   = [(0.7, 1.0, 0.7), (1.0, 1.0, 0.2), (1.0, 0.7, 0.7)]
stage_line_rgb = [(0.0, 0.6, 0.0), (1.0, 0.8, 0.2), (0.8, 0.0, 0.0)]

feature_cols  = ['R_eff_norm', 'P_drive_norm', 'M_cum_norm', 'BlastMem_norm']
feature_names = ['有效入渗 R_eff', '孔压驱动 P_drive', '累积微震 M_cum', '爆破记忆 BlastMem']
feature_note  = [' (z-score)', ' (z-score)', ' (z-score)', ' (z-score)']

# ==================== 读取训练集 ====================
df_train_all = pd.read_excel(features_file, sheet_name='train')
df_test_all  = pd.read_excel(features_file, sheet_name='test')  # 仅获取各阶段 test 长度

# ==================== 拟合与评估 ====================
results       = []
train_fitted  = []
test_lengths  = []

for stage_id in [1, 2, 3]:
    i = stage_id - 1
    df_train = df_train_all[df_train_all['Stage'] == stage_id].copy()
    df_test  = df_test_all[df_test_all['Stage'] == stage_id].copy()

    test_lengths.append(len(df_test))

    # --- 训练集拟合 ---
    X_train = df_train[feature_cols].values
    y_train = df_train['Delta_SD'].values

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)

    # 累积位移预测（训练集）
    sd0_train = df_train['SD'].iloc[0] if pd.notna(df_train['SD'].iloc[0]) else 0.0
    df_train['Delta_SD_pred'] = y_pred_train
    df_train['SD_pred']       = sd0_train + df_train['Delta_SD_pred'].cumsum()
    df_train['Residual']      = y_train - y_pred_train
    train_fitted.append(df_train)

    # 最后训练点的预测 SD（供 predict.py 链式累积）
    last_train_sd_pred = df_train['SD_pred'].iloc[-1]

    # --- 评估指标 ---
    r2  = r2_score(y_train, y_pred_train)
    mae = mean_absolute_error(y_train, y_pred_train)
    mse = mean_squared_error(y_train, y_pred_train)
    rmse = np.sqrt(mse)

    params = {
        'Stage': stage_id,
        'Stage Name': stage_names[i],
        'Intercept': model.intercept_,
        'coef_R_eff_norm': model.coef_[0],
        'coef_P_drive_norm': model.coef_[1],
        'coef_M_cum_norm': model.coef_[2],
        'coef_BlastMem_norm': model.coef_[3],
        'R2': r2,
        'MAE': mae,
        'RMSE': rmse,
        'MSE': mse,
        'Training Samples': len(y_train),
        'Test Samples': len(df_test),
        'Last_Train_SD_pred': last_train_sd_pred,
        'Train_Length': len(df_train),
        'Test_Length': len(df_test),
    }
    results.append(params)

    # --- 控制台报告 ---
    print()
    print("═" * 64)
    print(f"    {stage_names[i]}")
    print("═" * 64)
    print()
    print("  【模型参数（基于 z-score 归一化特征）】")
    print(f"    截距                  : {model.intercept_:+.6f}")
    for j, name in enumerate(feature_names):
        sign = '+' if model.coef_[j] >= 0 else ''
        print(f"    {name:<20s} : {sign}{model.coef_[j]:.6f}  (每 +1σ 对 ΔSD 的贡献)")

    # 主导因子（系数已可比，直接比较）
    abs_coef = np.abs(model.coef_)
    dominant_idx = np.argmax(abs_coef)
    print()
    print("  【因子贡献排序（按 |β_std| → 每增加 1 个标准差的影响）】")
    sorted_idx = np.argsort(abs_coef)[::-1]
    for rank, idx in enumerate(sorted_idx):
        bar = '█' * int(10 * abs_coef[idx] / abs_coef[sorted_idx[0]])
        print(f"    {rank+1}. {feature_names[idx]:<20s} |β_std| = {abs_coef[idx]:.6f}  {bar}")

    print()
    print("  【模型性能】")
    print(f"    决定系数 R²           : {r2:.4f}  ({r2*100:.1f}%)")
    print(f"    平均绝对误差 MAE       : {mae:.6f} mm/步")
    print(f"    均方根误差 RMSE        : {rmse:.6f} mm/步")
    print(f"    训练集样本数            : {len(y_train)}")
    print(f"    实验集样本数            : {len(df_test)}")

    # 物理含义
    print()
    print("  【物理含义解读】")
    for j, name in enumerate(feature_names):
        coef = model.coef_[j]
        if coef > 0:
            print(f"    • {name}：系数为正（{coef:+.6f}），每 +1σ 使 ΔSD 增大约 {abs(coef):.4f} mm/步，"
                  f"是滑坡的\"促进因子\"。")
        else:
            print(f"    • {name}：系数为负（{coef:+.6f}），每 +1σ 使 ΔSD 减小约 {abs(coef):.4f} mm/步，"
                  f"可能起\"抑制/滞后\"作用。")

    dominant_name = feature_names[dominant_idx]
    print(f"\n    ★ 本阶段主导因子：{dominant_name}，|β_std| = {abs_coef[dominant_idx]:.6f}")

    if r2 > 0.8:
        print(f"    ★ 模型拟合优度 R² = {r2:.3f}，线性假设对本阶段数据解释力较强。")
    elif r2 > 0.5:
        print(f"    ★ 模型拟合优度 R² = {r2:.3f}，线性假设基本可用，但存在一定非线性因素。")
    else:
        print(f"    ★ 模型拟合优度 R² = {r2:.3f}，线性假设解释力不足，本阶段可能存在较强的非线性机制。")

    print()

# ==================== 保存参数 CSV ====================
params_df = pd.DataFrame(results)
cols_order = ['Stage', 'Stage Name', 'Intercept',
              'coef_R_eff_norm', 'coef_P_drive_norm', 'coef_M_cum_norm', 'coef_BlastMem_norm',
              'R2', 'MAE', 'RMSE', 'MSE',
              'Training Samples', 'Test Samples',
              'Last_Train_SD_pred', 'Train_Length', 'Test_Length']
params_df = params_df[cols_order]
params_df.to_csv(output_params, index=False, float_format='%.6f')
print(f"[OK] 模型参数已保存至 {output_params}")

# ======================================================================
# 拼接连续数据（仅训练集）
# ======================================================================
all_sd_actual = []
all_sd_pred   = []
all_residual  = []
seg_ranges    = []
train_ranges  = []

total_len_global = 0
train_offset     = 0
for i in range(3):
    df_tr = train_fitted[i]
    n_tr = len(df_tr)
    n_te = test_lengths[i]

    seg_ranges.append({
        'train': (total_len_global, total_len_global + n_tr - 1),
        'test':  (total_len_global + n_tr, total_len_global + n_tr + n_te - 1),
    })
    train_ranges.append((train_offset, train_offset + n_tr - 1))

    all_sd_actual.append(df_tr['SD'].values)
    all_sd_pred.append(df_tr['SD_pred'].values)
    all_residual.append(df_tr['Residual'].values)

    total_len_global += n_tr + n_te
    train_offset    += n_tr

all_sd_actual = np.concatenate(all_sd_actual)
all_sd_pred   = np.concatenate(all_sd_pred)
all_residual  = np.concatenate(all_residual)

# 训练集全局时间轴
t_train_global_list = []
for i in range(3):
    t_start, t_end = seg_ranges[i]['train']
    t_train_global_list.append(np.arange(t_start, t_end + 1))
t_all = np.concatenate(t_train_global_list)

# 计算整体评价指标（训练集）
SS_res = np.sum(all_residual ** 2)
SS_tot = np.sum((all_sd_actual - np.mean(all_sd_actual)) ** 2)
R_sq_overall = 1 - SS_res / SS_tot if SS_tot > 0 else 0
RMSE_overall  = np.sqrt(np.mean(all_residual ** 2))
MAE_overall   = np.mean(np.abs(all_residual))
max_res_overall = np.max(np.abs(all_residual))

# 阶段分界线（用于虚线）
dividers = [seg_ranges[1]['train'][0], seg_ranges[2]['train'][0]]

# ======================================================================
# 总览图：训练集拟合 + 残差（fit_evaluation.png）
# ======================================================================
fig = plt.figure(figsize=(16, 9))
fig.suptitle('分阶段表面位移拟合总览', fontsize=15, fontweight='bold')

gs_total = GridSpec(2, 1, figure=fig, hspace=0.15,
                    left=0.07, right=0.97, top=0.93, bottom=0.06,
                    height_ratios=[2.5, 1])

# ------- 上图：位移 -------
ax1 = fig.add_subplot(gs_total[0])
yl_default = (all_sd_actual.min() - 5, all_sd_actual.max() + 5)

for i in range(3):
    x_start = seg_ranges[i]['train'][0] - 0.5
    x_end   = seg_ranges[i]['test'][1] + 0.5
    ax1.fill_between([x_start, x_end], yl_default[0], yl_default[1],
                     color=stage_bg_rgb[i], alpha=0.3, linewidth=0)
for di in dividers:
    ax1.axvline(x=di - 0.5, color='black', linestyle='--', linewidth=1.0)

ax1.plot(t_all, all_sd_actual, 'k-', linewidth=1.5, label='实际位移')
for i in range(3):
    idx_start, idx_end = train_ranges[i]
    t_tr = t_all[idx_start:idx_end + 1]
    sd_tr = all_sd_pred[idx_start:idx_end + 1]
    ax1.plot(t_tr, sd_tr, color=stage_line_rgb[i], linewidth=1.5,
             label=f'拟合位移·{stage_names[i]}')

for i in range(3):
    x_mid = (seg_ranges[i]['train'][0] + seg_ranges[i]['test'][1]) / 2
    ax1.text(x_mid, yl_default[1] - 0.04 * (yl_default[1] - yl_default[0]),
             stage_names[i], ha='center', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.75))

ax1.set_ylabel('表面位移 (mm)', fontsize=12)
ax1.set_title(f'位移：实际位移 — 模型拟合位移（R² = {R_sq_overall:.6f}，RMSE = {RMSE_overall:.4f} mm）',
              fontsize=13)
ax1.legend(fontsize=8, loc='upper left', ncol=2)
ax1.grid(alpha=0.3)
ax1.set_xlim(-20, total_len_global + 20)

# ------- 下图：残差 -------
ax2 = fig.add_subplot(gs_total[1])
for i in range(3):
    x_start = seg_ranges[i]['train'][0] - 0.5
    x_end   = seg_ranges[i]['test'][1] + 0.5
    ax2.fill_between([x_start, x_end], -10, 10,
                     color=stage_bg_rgb[i], alpha=0.3, linewidth=0)
for di in dividers:
    ax2.axvline(x=di - 0.5, color='black', linestyle='--', linewidth=1.0)

for i in range(3):
    idx_start, idx_end = train_ranges[i]
    t_tr = t_all[idx_start:idx_end + 1]
    res_tr = all_residual[idx_start:idx_end + 1]
    ax2.plot(t_tr, res_tr, color=stage_line_rgb[i], linewidth=0.8)

ax2.axhline(0, color='black', linestyle='-', linewidth=0.8)
ax2.set_xlabel('时间序号', fontsize=12)
ax2.set_ylabel('位移残差 (mm)', fontsize=12)
ax2.set_title(f'位移拟合残差（MAE = {MAE_overall:.4f} mm，最大 |残差| = {max_res_overall:.4f} mm）',
              fontsize=13)
ax2.grid(alpha=0.3)
ax2.set_xlim(-20, total_len_global + 20)

plt.savefig(output_plot, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"[OK] 总览图已保存至 {output_plot}")

# ------- 控制台输出整体统计 -------
print()
print("═══════════════════════════════════════════")
print("  整体位移拟合评价")
print("═══════════════════════════════════════════")
print(f"  决定系数 R²        : {R_sq_overall:.9f}")
print(f"  均方根误差 RMSE    : {RMSE_overall:.4f} mm")
print(f"  平均绝对误差 MAE   : {MAE_overall:.4f} mm")
print(f"  最大绝对残差       : {max_res_overall:.4f} mm")
print()

# ======================================================================
# 每阶段 6 面板详细诊断图 → segN/stage_report.png
# ======================================================================
for i in range(3):
    df = train_fitted[i]
    y_true   = df['Delta_SD'].values
    y_pred   = df['Delta_SD_pred'].values
    residual = df['Residual'].values
    sd_true  = df['SD'].values
    sd_pred  = df['SD_pred'].values
    t        = np.arange(len(df))

    r2   = results[i]['R2']
    mae  = results[i]['MAE']
    rmse = results[i]['RMSE']
    coefs = [results[i]['coef_R_eff_norm'], results[i]['coef_P_drive_norm'],
             results[i]['coef_M_cum_norm'], results[i]['coef_BlastMem_norm']]

    line_color = stage_line_rgb[i]

    fig = plt.figure(figsize=(18, 11))
    fig.suptitle(f'{stage_names[i]} · 回归诊断报告', fontsize=15, fontweight='bold')

    gs = GridSpec(3, 3, figure=fig, hspace=0.38, wspace=0.38,
                  left=0.06, right=0.98, top=0.92, bottom=0.06)

    # ----- ① 表面位移 SD 时序对比 -----
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(t, sd_true, 'k-', linewidth=1.5, label='实际位移')
    ax1.plot(t, sd_pred, color=line_color, linewidth=1.5, linestyle='--', label='拟合位移')
    ax1.fill_between(t, sd_true, sd_pred, alpha=0.10, color=line_color)
    ax1.set_title('① 表面位移 SD · 时序对比', fontsize=12, fontweight='bold')
    ax1.set_ylabel('表面位移 (mm)')
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(alpha=0.3)
    ax1.text(0.98, 0.95, f'R² = {r2:.4f}', transform=ax1.transAxes,
             fontsize=11, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

    # ----- ② 位移增量 ΔSD 对比 -----
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(t, y_true, 'k-', linewidth=0.9, alpha=0.7, label='实际 ΔSD')
    ax2.plot(t, y_pred, color=line_color, linewidth=0.9, linestyle='--', label='预测 ΔSD')
    ax2.set_title('② 位移增量 ΔSD 对比', fontsize=12, fontweight='bold')
    ax2.set_ylabel('ΔSD (mm/步)')
    ax2.legend(fontsize=7.5, loc='upper left')
    ax2.grid(alpha=0.3)
    ax2.text(0.98, 0.95, f'MAE={mae:.4f}\nRMSE={rmse:.4f}',
             transform=ax2.transAxes, fontsize=8, verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

    # ----- ③ 预测值 vs 实际值散点 -----
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(y_true, y_pred, c=[line_color], alpha=0.45, s=18, edgecolors='none')
    lim_min = min(y_true.min(), y_pred.min())
    lim_max = max(y_true.max(), y_pred.max())
    margin = (lim_max - lim_min) * 0.08
    ax3.plot([lim_min - margin, lim_max + margin], [lim_min - margin, lim_max + margin],
             'k--', linewidth=1.0, alpha=0.7, label='y = x')
    ax3.set_xlabel('实际 ΔSD')
    ax3.set_ylabel('预测 ΔSD')
    ax3.set_title('③ 预测值 vs 实际值', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.25)
    ax3.set_aspect('equal')
    ax3.text(0.05, 0.95, f'R² = {r2:.4f}\nMAE = {mae:.4f}',
             transform=ax3.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

    # ----- ④ 残差分布直方图 -----
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(residual, bins=35, density=True, alpha=0.6,
             color='steelblue', edgecolor='white', linewidth=0.6)
    mu, sigma = stats.norm.fit(residual)
    x_norm = np.linspace(residual.min(), residual.max(), 200)
    ax4.plot(x_norm, stats.norm.pdf(x_norm, mu, sigma), 'r-', linewidth=1.8,
             label=f'正态拟合 N({mu:.4f}, {sigma:.4f}²)')
    ax4.axvline(0, color='black', linestyle='--', linewidth=1.0)
    ax4.axvline(mu, color='red', linestyle=':', linewidth=1.0, alpha=0.6)
    ax4.set_title('④ 残差分布', fontsize=12, fontweight='bold')
    ax4.set_xlabel('残差 (mm/步)')
    ax4.set_ylabel('概率密度')
    ax4.legend(fontsize=8)
    ax4.text(0.98, 0.95, f'均值 = {mu:.5f}\n标准差 = {sigma:.5f}',
             transform=ax4.transAxes, fontsize=8, verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

    # ----- ⑤ 特征系数柱状图 -----
    ax5 = fig.add_subplot(gs[1, 2])
    colors_bar = ['#28a745' if c >= 0 else '#dc3545' for c in coefs]
    bars = ax5.barh(feature_names, coefs, color=colors_bar, edgecolor='white', height=0.6)
    ax5.axvline(0, color='black', linewidth=0.8)
    ax5.set_title('⑤ 标准化特征系数 (β_std)', fontsize=12, fontweight='bold')
    ax5.set_xlabel('β_std (mm·步⁻¹·σ⁻¹)')
    for bar, val in zip(bars, coefs):
        offset = 0.001 if abs(val) < 0.0005 else val * 0.05
        ax5.text(val + offset, bar.get_y() + bar.get_height()/2,
                 f'{val:+.4f}', va='center', fontsize=9,
                 color='#155724' if val >= 0 else '#721c24')
    ax5.grid(alpha=0.25, axis='x')

    # ----- ⑥ 残差时序图 -----
    ax6 = fig.add_subplot(gs[2, :])
    ax6.fill_between(t, residual, 0, alpha=0.25, color='steelblue')
    ax6.plot(t, residual, color='steelblue', linewidth=0.7)
    ax6.axhline(0, color='black', linestyle='--', linewidth=0.8)
    std_res = np.std(residual)
    ax6.axhline(+2 * std_res, color='red', linestyle=':', linewidth=1.0, alpha=0.7, label='+2σ')
    ax6.axhline(-2 * std_res, color='red', linestyle=':', linewidth=1.0, alpha=0.7, label='-2σ')
    outlier_ratio = np.mean(np.abs(residual) > 2 * std_res) * 100
    ax6.set_title(f'⑥ 残差时序图（超出 ±2σ 比例: {outlier_ratio:.1f}%）',
                  fontsize=12, fontweight='bold')
    ax6.set_xlabel('时间序号')
    ax6.set_ylabel('残差 (mm/步)')
    ax6.legend(fontsize=8, loc='upper right')
    ax6.grid(alpha=0.25)

    save_path = os.path.join(seg_dirs[i], "stage_report.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] {stage_names[i]} 诊断报告已保存至 {save_path}")

print()
print("═" * 64)
print("  回归分析完成")
print("═" * 64)
print(f"  参数表     : {output_params}")
print(f"  总览图     : {output_plot}")
for i in range(3):
    print(f"  {stage_names[i]:<30s} → {seg_dirs[i]}/stage_report.png")
print()
print("  请运行 predict.py 完成实验集预测与输出。")
