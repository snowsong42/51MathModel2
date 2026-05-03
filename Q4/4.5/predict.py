"""
问题 4.5：实验集预测（使用 regression.py 得出的模型参数）
================================================================
读取：
  model_params.csv —— 三阶段模型参数（截距、系数、链式累积信息）
  ap4_features.xlsx 的 test sheet —— 实验集特征

输出：
  1) prediction_overview.png —— 实验集预测位移总览（三段首尾相接）
  2) predict.xlsx —— 序号、预测位移、阶段号（实验集内部连续）
  3) segN/stage_prediction.png —— 各阶段实验集预测时序图（独立基线）
  4) 控制台 —— 全中文格式化报告

重要更正（2026-05-03）：
  - 去掉"全局序号"概念：训练集与实验集时间独立，不应混排
  - 总览图 X 轴 = 实验集内部连续时间轴 [0, total_test-1]
  - 三段预测链式累积：阶段 1 末尾 → 阶段 2 起始 → 阶段 3 起始

配色对齐 Q4_0/4.1/Assumption.m：
  阶段1：背景浅绿 [0.7 1 0.7]，线条深绿 [0 0.6 0]
  阶段2：背景浅黄 [1 1 0.2]，线条深黄 [1 0.8 0.2]
  阶段3：背景浅红 [1 0.7 0.7]，线条深红 [0.8 0 0]
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==================== 全局 matplotlib 设置 ====================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 路径设置 ====================
script_dir = os.path.dirname(os.path.abspath(__file__))
params_file   = os.path.join(script_dir, "model_params.csv")
features_file = os.path.join(script_dir, "../4.4/ap4_features.xlsx")

seg_dirs = [
    os.path.join(script_dir, "seg1"),
    os.path.join(script_dir, "seg2"),
    os.path.join(script_dir, "seg3"),
]
for d in seg_dirs:
    os.makedirs(d, exist_ok=True)

output_predict_xlsx    = os.path.join(script_dir, "predict.xlsx")
output_prediction_plot = os.path.join(script_dir, "prediction_overview.png")

# ==================== 阶段信息 ====================
stage_names    = ['第1段 · 匀速蠕变',  '第2段 · 加速变形',  '第3段 · 快速破坏']
stage_bg_rgb   = [(0.7, 1.0, 0.7), (1.0, 1.0, 0.2), (1.0, 0.7, 0.7)]
stage_line_rgb = [(0.0, 0.6, 0.0), (1.0, 0.8, 0.2), (0.8, 0.0, 0.0)]

feature_cols  = ['R_eff_norm', 'P_drive_norm', 'C_event_norm', 'BlastMem_norm']
feature_names = ['有效入渗 R_eff', '孔压驱动 P_drive', '微震事件数 C_event', '爆破记忆 BlastMem']

# ==================== 读取模型参数 ====================
params_df = pd.read_csv(params_file)
if len(params_df) != 3:
    raise ValueError(f"model_params.csv 应有 3 行，实际 {len(params_df)} 行")

# 提取参数
intercepts = params_df['Intercept'].values              # [β0, β0, β0]
coefs = params_df[['coef_R_eff_norm', 'coef_P_drive_norm',
                    'coef_C_event_norm', 'coef_BlastMem_norm']].values  # 3×4
last_train_sd_pred = params_df['Last_Train_SD_pred'].values
train_lengths = params_df['Train_Length'].values
test_lengths  = params_df['Test_Length'].values

# ==================== 读取实验集 ====================
df_test_all = pd.read_excel(features_file, sheet_name='test')

# ======================================================================
# 预测：三段链式累积
# ======================================================================
test_predicted = []       # 各阶段的 DataFrame（独立基线，用于子图）
continuous_sd  = []       # 连续拼接的 SD_pred 数组（总览图用）

print()
print("═" * 64)
print("  实验集预测（链式累积）")
print("═" * 64)

for stage_id in [1, 2, 3]:
    i = stage_id - 1
    df_test = df_test_all[df_test_all['Stage'] == stage_id].copy()

    if len(df_test) != test_lengths[i]:
        print(f"  [警告] Stage {stage_id} 实验集预期 {int(test_lengths[i])} 行，实际 {len(df_test)} 行")

    X_test = df_test[feature_cols].values

    # ΔSD_pred = intercept + Σ(coef_j × X_j)
    y_pred_test = intercepts[i] + X_test @ coefs[i]

    # 链式累积起始点：
    #   阶段 1 → 训练集最后 SD_pred
    #   阶段 2 → 阶段 1 实验集最后预测 SD
    #   阶段 3 → 阶段 2 实验集最后预测 SD
    if stage_id == 1:
        sd0 = last_train_sd_pred[i]
    else:
        sd0 = continuous_sd[-1][-1]   # 上一段最后一个预测值

    df_test['Delta_SD_pred'] = y_pred_test
    sd_vals = sd0 + df_test['Delta_SD_pred'].cumsum()
    df_test['SD_pred'] = sd_vals
    test_predicted.append(df_test)
    continuous_sd.append(sd_vals.values)

    print(f"  {stage_names[i]:<30s} "
          f"样本数: {len(df_test):<6d} "
          f"起始 SD: {sd0:.4f} mm  "
          f"结束 SD: {sd_vals.iloc[-1]:.4f} mm")

print()

# ======================================================================
# 实验集内部连续坐标
# ======================================================================
test_len_total = int(np.sum(test_lengths))           # 实验集总样本数
seg_ends = np.cumsum([int(tl) for tl in test_lengths])  # [n_te1, n_te1+n_te2, total]
seg_starts = np.concatenate([[0], seg_ends[:-1]])       # [0, n_te1, n_te1+n_te2]

all_exp_pred = np.concatenate(continuous_sd)

# ======================================================================
# 总览图：实验集预测位移总览（prediction_overview.png）
#   X 轴 = 实验集内部连续时间轴 [0, total_test-1]
#   三段首尾相接，背景色块 + 分隔虚线
# ======================================================================
fig, ax_pred = plt.subplots(figsize=(16, 5))
fig.suptitle('实验集预测位移总览（三段链式拼接）', fontsize=15, fontweight='bold')

y_min = all_exp_pred.min() - 5
y_max = all_exp_pred.max() + 5

# 背景色块
for i in range(3):
    x_start = seg_starts[i] - 0.5
    x_end   = seg_ends[i] - 0.5
    ax_pred.fill_between([x_start, x_end], y_min, y_max,
                         color=stage_bg_rgb[i], alpha=0.3, linewidth=0)

# 虚线分隔
for x_div in seg_ends[:-1]:
    ax_pred.axvline(x=x_div - 0.5, color='black', linestyle='--', linewidth=1.0)

# 三段预测曲线
for i in range(3):
    t_start = seg_starts[i]
    t_end   = seg_ends[i] - 1
    t_range = np.arange(t_start, t_end + 1)
    ax_pred.plot(t_range, continuous_sd[i], color=stage_line_rgb[i],
                 linewidth=1.8, marker='.', markersize=2,
                 label=f'预测位移·{stage_names[i]}')

# 阶段名标注
for i in range(3):
    x_mid = (seg_starts[i] + seg_ends[i] - 1) / 2
    ax_pred.text(x_mid, y_max - 0.04 * (y_max - y_min),
                 stage_names[i], ha='center', fontsize=10,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.75))

ax_pred.set_xlabel('实验集时间序号', fontsize=12)
ax_pred.set_ylabel('预测表面位移 (mm)', fontsize=12)
ax_pred.set_title('实验集预测表面位移（三段拼接，链式累积）', fontsize=13)
ax_pred.legend(fontsize=9, loc='upper left')
ax_pred.grid(alpha=0.3)
ax_pred.set_xlim(-20, test_len_total + 20)

plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.93])
plt.savefig(output_prediction_plot, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"[OK] 预测总览图已保存至 {output_prediction_plot}")

# ======================================================================
# 输出 predict.xlsx：实验集内部序号、预测位移、阶段号
# ======================================================================
predict_rows = []
for i in range(3):
    sd_pred_vals = continuous_sd[i]
    for k, sd_val in enumerate(sd_pred_vals):
        predict_rows.append({
            '序号': k,
            '预测位移_SD': sd_val,
            '阶段号': i + 1,
        })
df_predict = pd.DataFrame(predict_rows)

# 重新编号为全局实验集序号
df_predict['序号'] = np.arange(len(df_predict))

df_predict.to_excel(output_predict_xlsx, index=False, float_format='%.6f')
print(f"[OK] 预测表格已保存至 {output_predict_xlsx}")

# ======================================================================
# 实验集预测时序图 → segN/stage_prediction.png
# （各阶段独立基线 last_train_sd_pred[i]，X 轴 = 阶段内部序号）
# ======================================================================
for i in range(3):
    df_te = test_predicted[i]
    t_te = np.arange(len(df_te))

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(t_te, df_te['SD_pred'].values, color=stage_line_rgb[i],
            linewidth=1.8, marker='.', markersize=3, label='预测表面位移')

    if 'SD' in df_te.columns and df_te['SD'].notna().any():
        ax.plot(t_te, df_te['SD'].values, 'k-', linewidth=1.2,
                alpha=0.6, label='实际 SD（如有）')

    ax.set_title(f'{stage_names[i]} · 实验集预测', fontsize=14, fontweight='bold')
    ax.set_xlabel('时间序号', fontsize=12)
    ax.set_ylabel('表面位移 (mm)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    save_path = os.path.join(seg_dirs[i], "stage_prediction.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] {stage_names[i]} 实验集预测图已保存至 {save_path}")

# ======================================================================
# 控制台总结
# ======================================================================
print()
print("═" * 64)
print("  实验集预测完成")
print("═" * 64)
print(f"  预测总览   : {output_prediction_plot}")
print(f"  预测表格   : {output_predict_xlsx}")
for i in range(3):
    print(f"  {stage_names[i]:<30s} → {seg_dirs[i]}/stage_prediction.png")
print()
