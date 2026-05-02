import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error
import os
# ===== matplotlib 显示设置 =====
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 读取3.1的去噪结果 ====================
script_dir = os.path.dirname(os.path.abspath(__file__))
denoised_dir = os.path.join(script_dir, "../3.1")

df_train_clean = pd.read_excel(os.path.join(denoised_dir, "train_denoised.xlsx"))
df_exp_clean_raw = pd.read_excel(os.path.join(denoised_dir, "exp_denoised.xlsx"))

# 统一列名映射
train_col_map = {
    'a': 'a: Rainfall (mm)',
    'b': 'b: Pore Water Pressure (kPa)',
    'c': 'c: Microseismic Event Count',
    'd': 'd: Deep Displacement (mm)',
    'e': 'e: Surface Displacement (mm)'
}
exp_col_map = {
    'a': 'Rainfall (mm)',
    'b': 'Pore Water Pressure (kPa)',
    'c': 'Microseismic Event Count',
    'd': 'Deep Displacement (mm)',
    'e': 'Surface Displacement (mm)'
}

df_train_clean = df_train_clean.rename(columns=lambda c: c.strip())
df_exp_clean_raw = df_exp_clean_raw.rename(columns=lambda c: c.strip())

# 提取需要的列
train_keys = ['a','b','c','d','e']
df_train_clean = pd.DataFrame({k: df_train_clean[train_col_map[k].strip()].values for k in train_keys})

exp_keys = ['a','b','c','d']
df_exp_clean = pd.DataFrame({k: df_exp_clean_raw[exp_col_map[k].strip()].values for k in exp_keys})

# 保留实验集的 Serial No. 用于输出
exp_serial = df_exp_clean_raw['Serial No.'].values if 'Serial No.' in df_exp_clean_raw.columns else df_exp_clean_raw['Serial No. '].values

print("训练集（已去噪）形状:", df_train_clean.shape)
print("实验集（已去噪）形状:", df_exp_clean.shape)

# ==================== 3. 建立模型 (随机森林) ====================
X_train = df_train_clean[['a','b','c','d']].values
y_train = df_train_clean['e'].values

rf = RandomForestRegressor(n_estimators=500, max_depth=12, min_samples_split=10,
                           random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred_train = rf.predict(X_train)
r2 = r2_score(y_train, y_pred_train)
rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
print(f"\n随机森林模型在训练集上的性能:")
print(f"R^2 = {r2:.4f}, RMSE = {rmse:.2f} mm")

# ---- 特征重要性: 使用排列重要性 (Permutation Importance) ----
perm_result = permutation_importance(
    rf, X_train, y_train,
    n_repeats=5,
    random_state=42,
    n_jobs=-1,
    scoring='r2'
)
perm_importance = perm_result.importances_mean
perm_std = perm_result.importances_std

features = ['降雨量', '孔隙水压力', '微震事件数', '深部位移']
print("\n特征贡献度 (排列重要性, 能更好区分相关特征):")
for f, imp, std in zip(features, perm_importance, perm_std):
    print(f"  {f}: {imp:.4f} ± {std:.4f}")

# 同时输出传统重要性作对比
tree_importance = rf.feature_importances_
print("\n特征贡献度 (传统Gini重要性):")
for f, imp in zip(features, tree_importance):
    print(f"  {f}: {imp:.4f}")

# ==================== 4. 实验集预测 ====================
X_exp = df_exp_clean[['a','b','c','d']].values
y_pred_exp = rf.predict(X_exp)

df_exp_result = pd.DataFrame({'Serial No. ': exp_serial})
df_exp_result['预测表面位移 (mm)'] = y_pred_exp
exp_output = os.path.join(script_dir, "实验集表面位移预测.xlsx")
df_exp_result.to_excel(exp_output, index=False)
print(f"\n实验集预测结果已保存至 {exp_output}")

# ==================== 5. 整合成一张图 ====================
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# ---- 子图1: 特征重要性 (排列重要性) ----
ax1 = fig.add_subplot(gs[0, 0])
colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
bars = ax1.bar(features, perm_importance, color=colors, width=0.5, edgecolor='black', linewidth=0.8)
ax1.errorbar(features, perm_importance, yerr=perm_std, fmt='none', ecolor='black', capsize=5, capthick=1.5, elinewidth=1.5)
for i, (imp, std) in enumerate(zip(perm_importance, perm_std)):
    ax1.text(i, imp + std + 0.005, f'{imp:.4f}', ha='center', fontsize=10, fontweight='bold')
ax1.set_title('特征贡献度（排列重要性）\n能更好区分相关特征的真实影响', fontsize=12)
ax1.set_ylabel('打乱后 R² 下降值')
ax1.grid(axis='y', alpha=0.3)

# ---- 子图2: 训练集真实值 vs 预测值散点图 ----
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(y_train, y_pred_train, s=1, alpha=0.5, c='steelblue')
ax2.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', linewidth=1.5)
ax2.set_xlabel('真实表面位移 (mm)')
ax2.set_ylabel('预测表面位移 (mm)')
ax2.set_title(f'训练集：预测值 vs 真实值 (R²={r2:.3f})', fontsize=13)
ax2.grid(alpha=0.3)
ax2.set_aspect('equal')

# ---- 子图3: 实验集预测结果时序图 ----
ax3 = fig.add_subplot(gs[1, :])
ax3.plot(df_exp_result['Serial No. '], y_pred_exp, 'b.', markersize=2, alpha=0.6)
ax3.set_xlabel('时间序号')
ax3.set_ylabel('预测表面位移 (mm)')
ax3.set_title('实验集：预测表面位移', fontsize=13)
ax3.grid(alpha=0.3)

plt.suptitle('问题 3.3：随机森林表面位移预测', fontsize=15, y=0.98)
plt.savefig(os.path.join(script_dir, 'Q3.3_回归预测综合图.png'), dpi=300, bbox_inches='tight')
print(f"\n整合结果图已保存至 {os.path.join(script_dir, 'Q3.3_回归预测综合图.png')}")
plt.close()

# 同时保存单独的图
plt.figure(figsize=(8,5))
bars = plt.bar(features, perm_importance, color=colors, width=0.5, edgecolor='black', linewidth=0.8)
plt.errorbar(features, perm_importance, yerr=perm_std, fmt='none', ecolor='black', capsize=5, capthick=1.5, elinewidth=1.5)
for i, (imp, std) in enumerate(zip(perm_importance, perm_std)):
    plt.text(i, imp + std + 0.005, f'{imp:.4f}', ha='center', fontsize=10, fontweight='bold')
plt.title('特征贡献度（排列重要性）', fontsize=13)
plt.ylabel('打乱后 R² 下降值')
plt.grid(axis='y', alpha=0.3)
plt.savefig(os.path.join(script_dir, '特征重要性图.png'), dpi=300)
plt.close()

plt.figure(figsize=(6,6))
plt.scatter(y_train, y_pred_train, s=1, alpha=0.5)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
plt.xlabel('真实表面位移 (mm)')
plt.ylabel('预测表面位移 (mm)')
plt.title(f'训练集：预测值 vs 真实值 (R²={r2:.3f})')
plt.tight_layout()
plt.savefig(os.path.join(script_dir, '训练集预测对比图.png'), dpi=300)
plt.close()

plt.figure(figsize=(12,5))
plt.plot(df_exp_result['Serial No. '], y_pred_exp, 'b.', markersize=2, alpha=0.6)
plt.xlabel('时间序号')
plt.ylabel('预测表面位移 (mm)')
plt.title('实验集：预测表面位移')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, '实验集预测时序图.png'), dpi=300)
plt.close()

print("任务3.3全部完成。")
