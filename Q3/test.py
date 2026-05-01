import os
import pandas as pd
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

# ==================== 1. TV去噪算法 (使用FISTA或Chambolle-Pock，这里采用ADMM更快) ====================
def tv_denoise_admm(y, lam=1.0, rho=1.0, max_iter=100, tol=1e-4):
    """
    总变分去噪 (ADMM求解器)
    min 0.5*||y - x||^2 + lam * ||Dx||_1
    其中 D 为一阶差分矩阵 (N-1 x N)
    参数:
        y: 一维数组，含缺失值已补齐的原始信号
        lam: 正则化强度 (越大越平滑)
        rho: ADMM 惩罚参数，通常取1
        max_iter: 最大迭代次数
        tol: 收敛容差
    返回:
        x: 去噪后的信号
    """
    y = np.asarray(y, dtype=float)
    N = len(y)
    # 构造差分矩阵 D (N-1, N)
    e = np.ones(N)
    D = sparse.diags([-e, e], [0, 1], shape=(N-1, N)).tocsc()
    
    # 初始化变量
    x = y.copy()
    z = np.zeros(N-1)
    u = np.zeros(N-1)
    
    # 预计算 D^T D 和 (I + rho * D^T D) 的 Cholesky 分解 (用于x更新)
    DTD = D.T @ D
    I = sparse.eye(N)
    A = I + rho * DTD
    # 使用稀疏求解器
    from scipy.sparse.linalg import factorized
    solve_A = factorized(A.tocsc())
    
    for k in range(max_iter):
        # 更新 x: 求解 (I + rho D^T D) x = y + rho D^T (z - u)
        rhs = y + rho * (D.T @ (z - u))
        x_new = solve_A(rhs)
        # 更新 z: soft thresholding of (D x + u)
        d = D @ x_new + u
        z_new = np.maximum(0, d - lam/rho) - np.maximum(0, -d - lam/rho)  # 软阈值
        # 更新 u
        u_new = u + d - z_new
        
        # 收敛检查
        if np.linalg.norm(x_new - x) < tol * np.linalg.norm(x):
            break
        x, z, u = x_new, z_new, u_new
    return x

# ==================== 2. 读取数据 ====================
# 自动获取脚本所在目录，支持相对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "ap3.xlsx")
# 读取训练集 sheet
df_train = pd.read_excel(file_path, sheet_name="训练集")

# 五个目标变量 (原题中字母对应) —— 列名匹配实际 Excel
cols = {
    'a': 'a: Rainfall (mm)',
    'b': 'b: Pore Water Pressure (kPa)',
    'c': 'c: Microseismic Event Count',
    'd': 'd: Deep Displacement (mm)',
    'e': 'e: Surface Displacement (mm)'
}
# 提取数据 (跳过 Serial No. 列)
data_raw = {}
for key, col in cols.items():
    series = df_train[col].copy()
    # 将空白等转换为 NaN
    series = pd.to_numeric(series, errors='coerce')
    data_raw[key] = series

# ==================== 3. 缺失值补齐 (线性插值 + 边缘填充) ====================
data_filled = {}
for key, series in data_raw.items():
    # 使用 pandas 线性插值 (仅插值内部缺失，两端可能仍为 NaN)
    filled = series.interpolate(method='linear', limit_direction='both', limit_area='inside')
    # 首尾缺失用最近有效值填充 (前向/后向填充)
    filled = filled.bfill().ffill()
    # 如果仍有 NaN (全为缺失)，则填0 (实际不会发生)
    filled = filled.fillna(0)
    data_filled[key] = filled.values

# ==================== 4. TV去噪 (不同变量采用不同的 lambda) ====================
# 根据变量特性选择正则化参数 (可根据噪声水平微调)
lambda_dict = {
    'a': 0.3,    # 降雨量: 保留有效脉冲，适度平滑
    'b': 0.8,    # 孔隙水压力: 有一定噪声但趋势明显
    'c': 0.2,    # 微震事件数: 计数数据，避免过度平滑
    'd': 0.5,    # 深部位移: 趋势为主
    'e': 0.5     # 表面位移: 同深部位移
}

data_denoised = {}
for key, y in data_filled.items():
    lam = lambda_dict[key]
    y_den = tv_denoise_admm(y, lam=lam, max_iter=200, tol=1e-5)
    # 对于计数变量，四舍五入为整数
    if key == 'c':
        y_den = np.round(y_den).astype(int)
    data_denoised[key] = y_den

# ==================== 5. 结果输出与可视化 (可选) ====================
# 构造清洗后的DataFrame，保持与原训练集相同行数
df_cleaned = df_train[['Serial No. ']].copy()
for key, col in cols.items():
    df_cleaned[col] = data_denoised[key]

# 保存结果 (可选)
output_path = os.path.join(script_dir, "training_set_cleaned.xlsx")
df_cleaned.to_excel(output_path, index=False)
print(f"清洗后的训练集已保存至 {output_path}")

# 可选：绘制第一个变量去噪前后对比
plt.figure(figsize=(14, 6))
for i, key in enumerate(['a', 'e']):  # 展示降雨量和表面位移
    plt.subplot(2, 1, i+1)
    orig = data_raw[key].values
    filled = data_filled[key]
    denoised = data_denoised[key]
    t = np.arange(len(orig))
    plt.plot(t, orig, 'gray', alpha=0.4, label='Raw (with gaps)')
    plt.plot(t, filled, 'b--', alpha=0.7, label='Filled')
    plt.plot(t, denoised, 'r-', linewidth=1.5, label='TV Denoised')
    plt.title(f"{cols[key]} - Denoising result")
    plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "denoising_example.png"), dpi=300)
plt.show()
