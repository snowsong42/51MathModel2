import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 读取数据 ====================
script_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(script_dir, "../4.2/ap4_denoise.xlsx")
output_path = os.path.join(script_dir, "ap4_features.xlsx")

xls = pd.read_excel(input_path, sheet_name=None)
# 按常见命名匹配 sheet（训练集/实验集 或 第一个/第二个）
if '训练集' in xls and '实验集' in xls:
    train = xls['训练集']
    test  = xls['实验集']
else:
    sheets = list(xls.keys())
    train = xls[sheets[0]]
    test  = xls[sheets[1]]

# ==================== 爆破特征（脉冲型） ====================
train['d'] = train['d'].fillna(0)
train['e'] = train['e'].fillna(0)
test['d'] = test['d'].fillna(0)
test['e'] = test['e'].fillna(0)

# （旧: IsBlast/PropDistInv 已被 BlastMem 替代，见 add_features 内）

# ==================== 固定超参数（初始默认值） ====================
TAU_R = 80          # 降雨衰减时间常数（点数）
L_R   = 100         # 降雨窗口长度
L_M   = 150         # 微震累积窗口长度
BLAST_TAU = 100     # 爆破影响衰减时间常数（步数）

# 孔压参数：基于训练集全局统计（确保实验集使用相同常数）
P_CRIT = train['b'].quantile(0.75)   # 临界孔压（75%分位数）
P0     = train['b'].mean()           # 参考孔压（均值）

print(f"孔压参数：P_crit = {P_CRIT:.2f}, P0 = {P0:.2f}")

# ==================== 特征工程函数 ====================
def exp_weighted_mean(series, tau):
    """对窗口数组计算指数加权平均，最近数据权重最大"""
    n = len(series)
    if n == 0:
        return np.nan
    weights = np.exp(-np.arange(n)[::-1] / tau)   # 最新权重=1，最旧=exp(-(n-1)/tau)
    return np.average(series, weights=weights)

def add_features(df, tau_r=TAU_R, L_r=L_R, L_m=L_M, P_crit=P_CRIT, P0_val=P0):
    df = df.copy()
    
    # 1. 有效入渗降雨量 R_eff（指数衰减滑动窗）
    df['R_eff'] = (df['a']
                    .rolling(window=L_r, min_periods=1)
                    .apply(exp_weighted_mean, args=(tau_r,), raw=True))
    
    # 2. 累积微震事件数 M_cum（简单滑动求和）
    df['M_cum'] = df['c'].rolling(window=L_m, min_periods=1).sum()
    
    # 3. 孔压驱动项 P_drive
    df['P_drive'] = np.maximum(0, df['b'] - P_crit) * (df['b'] / P0_val)
    
    # 4. 位移增量 ΔSD（训练集有 SD，实验集自动填 NaN）
    if 'SD' in df.columns and df['SD'].notna().any():
        df['Delta_SD'] = df['SD'].diff()
        # 第一条差分结果为 NaN，填 0（位移序列起始瞬间无变化）
        df.loc[df.index[0], 'Delta_SD'] = 0.0
    else:
        df['Delta_SD'] = np.nan
    
    # 5. 爆破记忆量 BlastMem（指数衰减递推）
    I = np.where((df['e'] > 0) & (df['d'] > 0), df['e']**(1/3) / df['d'], 0.0)
    gamma = np.exp(-1.0 / BLAST_TAU)
    blast_mem = np.empty_like(I)
    blast_mem[0] = I[0]
    for t in range(1, len(I)):
        blast_mem[t] = gamma * blast_mem[t-1] + I[t]
    df['BlastMem'] = blast_mem
    
    return df

# ==================== 应用特征构造 ====================
train_feat = add_features(train)
test_feat  = add_features(test)

# ==================== 归一化（训练集 fit，实验集 transform） ====================
norm_cols = ['R_eff', 'P_drive', 'M_cum', 'BlastMem']
scaler = StandardScaler()
train_feat_norm = train_feat.copy()
test_feat_norm = test_feat.copy()

train_feat_norm[[c + '_norm' for c in norm_cols]] = scaler.fit_transform(train_feat[norm_cols].values)
test_feat_norm[[c + '_norm' for c in norm_cols]] = scaler.transform(test_feat[norm_cols].values)

print(f"\n归一化完成（训练集 fit，实验集使用相同参数）")
for j, col in enumerate(norm_cols):
    print(f"  {col}: mean = {scaler.mean_[j]:.6f}, std = {scaler.scale_[j]:.6f}")

# ==================== 整理输出列 ====================
cols_out = [
    'Time', 'Stage',          # 时间、阶段
    'a', 'b', 'c', 'd', 'e',  # 原始五维特征
    'BlastMem',               # 爆破连续记忆量（指数衰减递推）
    'R_eff', 'M_cum', 'P_drive',  # 降雨滞后、微震累积、孔压驱动
    'R_eff_norm', 'P_drive_norm', 'M_cum_norm', 'BlastMem_norm',  # 归一化（z-score）
    'SD', 'Delta_SD'          # 位移（原始+增量目标）
]

train_out = train_feat_norm[cols_out]
test_out  = test_feat_norm[cols_out]

# ==================== 保存总表 ====================
with pd.ExcelWriter(output_path) as writer:
    train_out.to_excel(writer, sheet_name='train', index=False)
    test_out.to_excel(writer, sheet_name='test', index=False)
    # 保存归一化参数
    scaler_params = pd.DataFrame({
        'Feature': norm_cols,
        'Mean': scaler.mean_,
        'Std': scaler.scale_,
    })
    scaler_params.to_excel(writer, sheet_name='scaler_params', index=False)
print("特征构造完成，已保存至 ./ap4_features.xlsx（含 scaler_params sheet）")

# ==================== 分段输出 3 个 xlsx ====================
for seg_id in [1, 2, 3]:
    seg_train = train_out[train_out['Stage'] == seg_id].copy()
    seg_test  = test_out[test_out['Stage'] == seg_id].copy()
    seg_path = os.path.join(script_dir, f"seg{seg_id}_features.xlsx")
    with pd.ExcelWriter(seg_path) as writer:
        seg_train.to_excel(writer, sheet_name='train', index=False)
        seg_test.to_excel(writer, sheet_name='test', index=False)
    print(f"分段保存: seg{seg_id}_features.xlsx (训练集{len(seg_train)}行, 实验集{len(seg_test)}行)")

# ==================== 对比图绘制 ====================

def draw_compare(name, raw_col, proc_col, raw_label, proc_label, unit_raw, unit_proc):
    """绘制 2×1 子图（train / test）双纵轴对比原始特征和处理后特征"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle(f'{raw_label} vs {proc_label}', fontsize=13, fontweight='bold')
    
    for i, (label, df) in enumerate([("训练集", train_out), ("实验集", test_out)]):
        t = np.arange(len(df))
        raw_vals = pd.to_numeric(df[raw_col], errors='coerce').fillna(0).values
        proc_vals = pd.to_numeric(df[proc_col], errors='coerce').fillna(0).values
        
        ax = axes[i]
        # 左纵轴：原始特征（蓝色）
        color_raw = 'royalblue'
        ln1 = ax.plot(t, raw_vals, color=color_raw, alpha=0.7, linewidth=0.8, label=raw_label)
        ax.set_ylabel(f'{raw_label} [{unit_raw}]', fontsize=10, color=color_raw)
        ax.tick_params(axis='y', labelcolor=color_raw)
        
        # 右纵轴：归一化后构造特征（红色）
        ax2 = ax.twinx()
        color_proc = 'tomato'
        proc_norm_vals = pd.to_numeric(df[proc_col + '_norm'], errors='coerce').fillna(0).values
        ln2 = ax2.plot(t, proc_norm_vals, color=color_proc, alpha=0.8, linewidth=1.0, label=f'{proc_label} (归一化)')
        ax2.set_ylabel(f'{proc_label} [z-score]', fontsize=10, color=color_proc)
        ax2.tick_params(axis='y', labelcolor=color_proc)
        
        # 合并图例
        ln_all = ln1 + ln2
        labs = [l.get_label() for l in ln_all]
        ax.legend(ln_all, labs, fontsize=9)
        
        ax.set_title(f'{label}', fontsize=11)
        ax.grid(alpha=0.2)
    
    axes[-1].set_xlabel('时间序号', fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(script_dir, f"compare_{name}.png")
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"对比图已保存: compare_{name}.png")

# 图①：a ↔ R_eff
draw_compare('R_eff', 'a', 'R_eff',
             '原始降雨量', '有效入渗 R_eff', 'mm', 'mm')

# 图②：b ↔ P_drive
draw_compare('P_drive', 'b', 'P_drive',
             '原始孔压', '孔压驱动 P_drive', 'kPa', 'kPa')

# 图③：c ↔ M_cum
draw_compare('M_cum', 'c', 'M_cum',
             '原始微震事件数', '累积微震 M_cum', 'count', 'count')

# 图④：爆破对比图（BlastMem + 爆破事件标注距离）
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
fig.suptitle('BlastMem 时序图（爆破事件标注距离）', fontsize=13, fontweight='bold')

for ax, (label, df_raw, df_out) in zip(axes,
    [("训练集", train, train_out), ("实验集", test, test_out)]):
    t = np.arange(len(df_out))
    blast_vals = df_out['BlastMem'].values
    d_vals = pd.to_numeric(df_raw['d'], errors='coerce').fillna(0).values
    e_vals = pd.to_numeric(df_raw['e'], errors='coerce').fillna(0).values
    blast_flags = (e_vals > 0).astype(int)
    
    # 基线
    ax.plot(t, blast_vals, color='seagreen', alpha=0.7, linewidth=0.8, label='BlastMem')
    
    # 爆破散点
    blast_idx = np.where(blast_flags == 1)[0]
    if len(blast_idx) > 0:
        ax.scatter(blast_idx, blast_vals[blast_idx], color='darkorange', s=30,
                   edgecolors='k', linewidths=0.5, zorder=5, label='爆破事件')
        # 每隔若干点标注一次 d 值，避免文字拥堵
        step = max(1, len(blast_idx) // 15)
        for ii in blast_idx[::step]:
            ax.annotate(f'd={d_vals[ii]:.1f}', (ii, blast_vals[ii]),
                        textcoords="offset points", xytext=(5, 8),
                        fontsize=7, color='darkred', alpha=0.8,
                        arrowprops=dict(arrowstyle='->', color='gray', alpha=0.4, lw=0.5))
    
    ax.set_ylabel('BlastMem', fontsize=10)
    ax.set_title(f'{label}', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

axes[-1].set_xlabel('时间序号', fontsize=10)
plt.tight_layout(rect=[0, 0, 1, 0.95])
save_path = os.path.join(script_dir, "compare_blast.png")
plt.savefig(save_path, dpi=200)
plt.close()
print("对比图已保存: compare_blast.png")

print("全部完成")
