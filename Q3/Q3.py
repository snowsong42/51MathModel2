import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, interp1d
from scipy.signal import savgol_filter
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error
import os, warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']; plt.rcParams['axes.unicode_minus'] = False

# 路径设置
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'ap3.xlsx')
df_train = pd.read_excel(file_path, sheet_name='训练集')
df_exp   = pd.read_excel(file_path, sheet_name='实验集')

# 列名映射 (实验集不读取表面位移，仅用前4列)
train_map = {'a':'a: Rainfall (mm)','b':'b: Pore Water Pressure (kPa)','c':'c: Microseismic Event Count',
             'd':'d: Deep Displacement (mm)','e':'e: Surface Displacement (mm)'}
exp_map   = {'a':'Rainfall (mm)','b':'Pore Water Pressure (kPa)','c':'Microseismic Event Count',
             'd':'Deep Displacement (mm)'}
keys = ['a','b','c','d']
e_key = 'e'

# 缺失值填充与平滑
def fill_series(s):
    arr = s.values.astype(float)
    idx = np.arange(len(arr))
    mask = ~np.isnan(arr)
    if mask.sum() == 0: return np.zeros_like(arr)
    if mask.sum() == 1: return np.full_like(arr, arr[mask][0])
    if mask.sum() < 4:
        return interp1d(idx[mask], arr[mask], kind='linear', fill_value='extrapolate')(idx)
    cs = CubicSpline(idx[mask], arr[mask], bc_type='natural')
    return cs(idx)

print('>>> 去噪与缺失值补齐...')
train_data, exp_data = {}, {}
for key in keys:
    ts = pd.to_numeric(df_train[train_map[key]], errors='coerce')
    es = pd.to_numeric(df_exp[exp_map[key]], errors='coerce')
    train_data[key] = savgol_filter(fill_series(ts), window_length=21, polyorder=3)
    exp_data[key]   = savgol_filter(fill_series(es), window_length=21, polyorder=3)
# 表面位移（训练集）
ts_e = pd.to_numeric(df_train[train_map[e_key]], errors='coerce')
train_data[e_key] = savgol_filter(fill_series(ts_e), window_length=21, polyorder=3)
print('完成。\n')

# 异常值检测 (Robust MAD, k=4.5)
k = 4.5
outlier_counts = {}
total = len(train_data['a'])
outlier_flags = {}
for key in keys + [e_key]:
    y = train_data[key]
    med = np.median(y); mad = np.median(np.abs(y - med))
    if mad == 0: mad = 1.0
    z = (y - med) / mad
    med_z = np.median(z); mad_z = np.median(np.abs(z - med_z))
    flags = np.abs(z - med_z) > k * mad_z if mad_z != 0 else np.zeros(total, dtype=bool)
    outlier_flags[key] = flags
    outlier_counts[key] = int(np.sum(flags))

print('===== 表3.1 单变量异常点检出结果 =====')
for key in ['a','b','c','d','e']:
    print(f'  {key}: {outlier_counts[key]} 个')
print(f'  总数: {sum(outlier_counts.values())}')
common = [(i+1, ''.join(sorted(k for k in keys+[e_key] if outlier_flags[k][i]))) 
          for i in range(total) if sum(outlier_flags[k][i] for k in keys+[e_key]) >= 2]
print(f'共同异常点个数: {len(common)}')
print('前10个共同异常点 (序号, 变量组合):')
for idx, combo in common[:10]:
    print(f'  {idx}: {combo}')
print()

# 随机森林回归与特征重要性
print('>>> 训练随机森林模型...')
X_train = np.column_stack([train_data[k] for k in keys])
y_train = train_data[e_key]
rf = RandomForestRegressor(n_estimators=500, max_depth=12, min_samples_split=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_train)
r2 = r2_score(y_train, y_pred)
rmse = np.sqrt(mean_squared_error(y_train, y_pred))
print(f'训练集 R² = {r2:.4f}, RMSE = {rmse:.2f} mm')

perm = permutation_importance(rf, X_train, y_train, n_repeats=5, random_state=42, scoring='r2')
feat_names = ['降雨量', '孔隙水压力', '微震事件数', '深部位移']
print('排列重要性 (R²下降):')
for name, imp, std in zip(feat_names, perm.importances_mean, perm.importances_std):
    print(f'  {name}: {imp:.4f} ± {std:.4f}')

# 实验集预测
X_exp = np.column_stack([exp_data[k] for k in keys])
y_exp_pred = rf.predict(X_exp)
plt.figure(figsize=(10,4))
# 绘制散点图
plt.figure(figsize=(10,4))
plt.scatter(range(len(y_exp_pred)), y_exp_pred, s=2, c='steelblue', alpha=0.7)
plt.xlabel('Time'); plt.ylabel('Displacement (mm)')
plt.title('Experimental Set Surface Displacement Prediction Scatter Plot'); plt.grid(alpha=0.3)
out_path = os.path.join(script_dir, 'Prediction.png')
plt.tight_layout(); plt.savefig(out_path, dpi=300); plt.close()
print(f'Scatter Plot saved to {out_path}')