import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import factorized
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import os

# ==================== 1. TV去噪函数 (与3.1相同) ====================
def tv_denoise_admm(y, lam=1.0, rho=1.0, max_iter=100, tol=1e-4):
    y = np.asarray(y, dtype=float)
    N = len(y)
    e = np.ones(N)
    D = sparse.diags([-e, e], [0, 1], shape=(N-1, N)).tocsc()
    DTD = D.T @ D
    I = sparse.eye(N)
    A = I + rho * DTD
    solve_A = factorized(A.tocsc())
    x = y.copy()
    z = np.zeros(N-1)
    u = np.zeros(N-1)
    for k in range(max_iter):
        rhs = y + rho * (D.T @ (z - u))
        x_new = solve_A(rhs)
        d = D @ x_new + u
        z_new = np.maximum(0, d - lam/rho) - np.maximum(0, -d - lam/rho)
        u_new = u + d - z_new
        if np.linalg.norm(x_new - x) < tol * np.linalg.norm(x):
            break
        x, z, u = x_new, z_new, u_new
    return x

# ==================== 2. 数据读取与预处理 ====================
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "ap3.xlsx")

df_train = pd.read_excel(file_path, sheet_name="训练集")
df_exp = pd.read_excel(file_path, sheet_name="实验集")

train_cols = {
    'a': 'a: Rainfall (mm)',
    'b': 'b: Pore Water Pressure (kPa)',
    'c': 'c: Microseismic Event Count',
    'd': 'd: Deep Displacement (mm)',
    'e': 'e: Surface Displacement (mm)'
}

exp_cols = {
    'a': 'Rainfall (mm)',
    'b': 'Pore Water Pressure (kPa)',
    'c': 'Microseismic Event Count',
    'd': 'Deep Displacement (mm)',
    'e': 'Surface Displacement (mm)'
}

lambda_dict = {'a':0.3, 'b':0.8, 'c':0.2, 'd':0.5, 'e':0.5}

def preprocess_single(df, key, col_name, lam):
    series = pd.to_numeric(df[col_name], errors='coerce')
    filled = series.interpolate(method='linear', limit_direction='both')
    filled = filled.bfill().ffill().fillna(0)
    denoised = tv_denoise_admm(filled.values, lam=lam)
    if key == 'c':
        denoised = np.round(denoised).astype(int)
    return denoised

def preprocess_dataset(df, col_map, keys, lambda_dict):
    processed = {}
    for key in keys:
        processed[key] = preprocess_single(df, key, col_map[key], lambda_dict[key])
    return pd.DataFrame(processed)

train_keys = ['a','b','c','d','e']
df_train_clean = preprocess_dataset(df_train, train_cols, train_keys, lambda_dict)
print("训练集预处理完成，形状:", df_train_clean.shape)

exp_keys = ['a','b','c','d']
df_exp_clean = preprocess_dataset(df_exp, exp_cols, exp_keys, lambda_dict)
print("实验集预处理完成，形状:", df_exp_clean.shape)

# ==================== 3. 建立模型 (随机森林) ====================
X_train = df_train_clean[['a','b','c','d']].values
y_train = df_train_clean['e'].values

rf = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=5,
                           random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred_train = rf.predict(X_train)
r2 = r2_score(y_train, y_pred_train)
rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
print(f"\n随机森林模型在训练集上的性能:")
print(f"R^2 = {r2:.4f}, RMSE = {rmse:.2f} mm")

importance = rf.feature_importances_
features = ['Rainfall', 'Pore pressure', 'Microseismic', 'Deep displacement']
print("\n特征贡献度 (重要性):")
for f, imp in zip(features, importance):
    print(f"  {f}: {imp:.4f}")

# ==================== 4. 实验集预测 ====================
X_exp = df_exp_clean[['a','b','c','d']].values
y_pred_exp = rf.predict(X_exp)

# 输出预测结果
df_exp_result = df_exp[['Serial No. ']].copy()
df_exp_result['Predicted Surface Displacement (mm)'] = y_pred_exp
exp_output = os.path.join(script_dir, "experimental_set_predictions.xlsx")
df_exp_result.to_excel(exp_output, index=False)
print(f"\n实验集预测结果已保存至 {exp_output}")

# ==================== 5. 整合成一张图 ====================
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# ---- 子图1: 特征重要性散点图 ----
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(features, importance, color='steelblue', s=120, zorder=5)
ax1.plot(features, importance, 'gray', linestyle='--', alpha=0.5)
for i, (f, imp) in enumerate(zip(features, importance)):
    ax1.text(i, imp + 0.02, f'{imp:.3f}', ha='center', fontsize=10)
ax1.set_title('Feature Importance', fontsize=13)
ax1.set_ylabel('Importance')
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim(0, max(importance) * 1.3)

# ---- 子图2: 训练集真实值 vs 预测值散点图 ----
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(y_train, y_pred_train, s=1, alpha=0.5, c='steelblue')
ax2.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', linewidth=1.5)
ax2.set_xlabel('True Surface Displacement (mm)')
ax2.set_ylabel('Predicted Surface Displacement (mm)')
ax2.set_title(f'Training Set: Predicted vs True (R^2={r2:.3f})', fontsize=13)
ax2.grid(alpha=0.3)
ax2.set_aspect('equal')

# ---- 子图3: 实验集预测结果时序图 ----
ax3 = fig.add_subplot(gs[1, :])
ax3.plot(df_exp_result['Serial No. '], y_pred_exp, 'b.', markersize=2, alpha=0.6)
ax3.set_xlabel('Serial No. (time order)')
ax3.set_ylabel('Predicted Surface Displacement (mm)')
ax3.set_title('Experimental Set: Predicted Surface Displacement', fontsize=13)
ax3.grid(alpha=0.3)

plt.suptitle('Question 3.3: Random Forest Prediction for Surface Displacement', fontsize=15, y=0.98)
plt.savefig(os.path.join(script_dir, 'Q3.3_combined_results.png'), dpi=300, bbox_inches='tight')
print(f"\n整合结果图已保存至 {os.path.join(script_dir, 'Q3.3_combined_results.png')}")
plt.show()

# 同时保存单独的图（方便单独查看）
plt.figure(figsize=(8,5))
plt.scatter(features, importance, color='steelblue', s=100, zorder=5)
plt.plot(features, importance, 'gray', linestyle='--', alpha=0.5)
for i, (f, imp) in enumerate(zip(features, importance)):
    plt.text(i, imp + 0.02, f'{imp:.3f}', ha='center', fontsize=10)
plt.title('Feature Importance for Surface Displacement (Random Forest)')
plt.ylabel('Importance')
plt.grid(axis='y', alpha=0.3)
plt.ylim(0, max(importance) * 1.3)
plt.savefig(os.path.join(script_dir, 'feature_importance.png'), dpi=300)
plt.close()

plt.figure(figsize=(6,6))
plt.scatter(y_train, y_pred_train, s=1, alpha=0.5)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
plt.xlabel('True Surface Displacement (mm)')
plt.ylabel('Predicted Surface Displacement (mm)')
plt.title(f'Training Set: Predicted vs True (R^2={r2:.3f})')
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'train_pred_vs_true.png'), dpi=300)
plt.close()

plt.figure(figsize=(12,5))
plt.plot(df_exp_result['Serial No. '], y_pred_exp, 'b.', markersize=2, alpha=0.6)
plt.xlabel('Serial No. (time order)')
plt.ylabel('Predicted Surface Displacement (mm)')
plt.title('Experimental Set: Predicted Surface Displacement')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'exp_predicted_scatter.png'), dpi=300)
plt.close()

print("任务3.3全部完成。")
