"""
====================================================================
Q4 速率叠加模型 (Rate Superposition Model)
=============================================

基于问题四建模文档的核心思想：
  将"位移加法"转换为"速率叠加"，对速率积分获得位移。

  v(t) = v_basic + v_rain(t) + v_pore(t) + v_micro(t) + v_blast(t)

  其中各项物理含义:
    v_basic   : 基础蠕变速率 (常数, 各阶段不同)
    v_rain(t) : 降雨有效入渗 → 滞后窗口 + 衰减记忆
    v_pore(t) : 孔压 → 超过阈值后即时贡献
    v_micro(t): 微震累积损伤 → 累积窗口驱动
    v_blast(t): 爆破 → 萨道夫斯基衰减 (距离R + 药量Q)

  三阶段划分:
    阶段一 (缓慢匀速): 低速恒定, 降雨/微震响应弱
    阶段二 (加速形变): 中速增长, 水弱化累积
    阶段三 (快速形变): 高速增长, 多因素耦合

数据源: Q4_1/ap4.xlsx → 训练集(10000点, 含位移) + 实验集(5000点, 不含位移)
====================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import medfilt, savgol_filter
from scipy.optimize import nnls

# ── 编码 & 字体 ──
if sys.stdout.encoding == 'gbk':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ── 全局常量 ──
DT = 10.0  # 采样间隔 (分钟)
DT_HOUR = DT / 60.0  # 小时


# ================================================================
#  1. 数据加载与预处理
# ================================================================
def load_data(data_dir=None):
    """加载 ap4.xlsx 的训练集和实验集"""
    if data_dir is None:
        data_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(data_dir, "../ap4.xlsx")

    train_df = pd.read_excel(data_path, sheet_name='训练集')
    test_df = pd.read_excel(data_path, sheet_name='实验集')

    print("=" * 72)
    print("  Q4 速率叠加模型 — 数据加载")
    print(f"  训练集: {len(train_df)} 点")
    print(f"  实验集: {len(test_df)} 点")
    print(f"  列: {list(train_df.columns)}")
    print("=" * 72)

    return train_df, test_df


def preprocess(df, is_train=True):
    """
    预处理: 时间标准化, 缺失插值, 中值滤波, 特征工程
    """
    data = df.copy()

    # ── 时间 ──
    time_col = data.columns[0]
    data['Time'] = pd.to_datetime(data[time_col])
    data = data.sort_values('Time').reset_index(drop=True)
    N = len(data)
    data['Serial'] = np.arange(N)
    data['t_hour'] = np.arange(N) * DT_HOUR

    # ── 列名映射 ──
    col_map = {
        'Surface Displacement (mm)': 'displacement',
        'Rainfall (mm)': 'rainfall',
        'Pore Water Pressure (kPa)': 'pore',
        'Microseismic Event Count': 'micro',
        'Blasting Point Distance (m)': 'blast_dist',
        'Maximum Charge per Segment (kg)': 'blast_charge',
    }
    for old, new in col_map.items():
        if old in data.columns:
            data[new] = data[old].values

    if 'Stage Label' in data.columns:
        stage_vals = data['Stage Label'].values
        stage_vals = pd.Series(stage_vals).ffill().bfill().fillna(0).astype(int).values
        data['stage'] = stage_vals

    # ── 位移预处理 ──
    if 'displacement' in data.columns:
        disp = data['displacement'].values.astype(float)
        # NaN/零值插值
        bad = np.isnan(disp) | (disp == 0)
        if np.sum(bad) > 0 and not is_train:
            # 实验集: 全部为NaN, 用0占位
            disp[:] = 0.0
        elif np.sum(bad) > 0:
            valid = np.where(~bad)[0]
            if len(valid) >= 2:
                disp[bad] = np.interp(np.where(bad)[0], valid, disp[valid])
            elif len(valid) == 1:
                disp[bad] = disp[valid[0]]
            print(f"  [预处理] 位移插值: {int(np.sum(bad))} 点")
        # 中值滤波 (仅训练集)
        if is_train:
            disp_filt = medfilt(disp, kernel_size=9)
        else:
            disp_filt = disp.copy()
        data['displacement_raw'] = disp.copy()
        data['displacement'] = disp_filt
    else:
        data['displacement'] = np.zeros(N)
        data['displacement_raw'] = np.zeros(N)

    # ── 孔压: 前向填充 ──
    if 'pore' in data.columns:
        data['pore'] = pd.Series(data['pore'].values).ffill().bfill().interpolate(
            method='linear').values
    else:
        data['pore'] = np.zeros(N)

    # ── 降雨/微震: NaN → 0 ──
    for col in ['rainfall', 'micro']:
        if col in data.columns:
            data[col] = data[col].fillna(0).values.astype(float)
        else:
            data[col] = np.zeros(N)

    # ── 爆破: NaN → 0 ──
    for col in ['blast_dist', 'blast_charge']:
        if col in data.columns:
            data[col] = data[col].fillna(0).values.astype(float)
        else:
            data[col] = np.zeros(N)

    # ── 爆破指示变量 ──
    data['blast_flag'] = ((data['blast_dist'] > 0) & (data['blast_charge'] > 0)).astype(int)

    return data


# ================================================================
#  2. 速率叠加各分量计算
# ================================================================
def compute_rain_effective(rainfall, window=72, decay=0.92):
    """
    有效入渗降雨量: 滞后窗口 + 指数衰减记忆
    R_eff(t) = Σ w_k · R(t-k),  w_k = decay^k
    """
    N = len(rainfall)
    rain_eff = np.zeros(N)
    for i in range(N):
        start = max(0, i - window + 1)
        w = np.array([decay ** (i - k) for k in range(start, i + 1)])
        rain_eff[i] = np.sum(rainfall[start:i + 1] * w[::-1])
    return rain_eff


def compute_micro_cumulative(micro, window=72):
    """微震累积: 滑动平均"""
    N = len(micro)
    micro_cum = np.convolve(micro, np.ones(window) / window, mode='same')
    half = window // 2
    for i in range(half):
        micro_cum[i] = np.mean(micro[:i + half + 1])
    for i in range(N - half, N):
        micro_cum[i] = np.mean(micro[i - half:])
    return micro_cum


def compute_blast_feature(blast_dist, blast_charge, blast_flag, alpha=1.5, decay_factor=0.7):
    """
    爆破特征: 萨道夫斯基公式 (Q^(1/3)/R)^α + 指数衰减
    返回: 爆破特征序列 (单位: 振动能量代理变量)
    """
    N = len(blast_flag)
    raw = np.zeros(N)
    for i in range(N):
        if blast_flag[i] > 0 and blast_dist[i] > 0 and blast_charge[i] > 0:
            R = max(blast_dist[i], 0.1)
            Q = max(blast_charge[i], 0.01)
            raw[i] = (Q ** (1.0 / 3.0) / R) ** alpha

    # 指数衰减
    feat = np.zeros(N)
    cum = 0.0
    for i in range(N):
        cum = cum * decay_factor + raw[i]
        feat[i] = cum
    return feat


# ================================================================
#  3. 速率拟合与位移重构
# ================================================================
def fit_rate_model(data, stage_labels):
    """
    分阶段拟合速率叠加模型: v = v_basic + γ·R_eff + δ·(P - P_th)+ + η·M_cum + β·B

    使用非负最小二乘 (NNLS) 保证各贡献非负
    返回: 各阶段参数列表
    """
    N = len(data)
    rainfall = data['rainfall'].values
    micro = data['micro'].values
    pore = data['pore'].values
    blast_dist = data['blast_dist'].values
    blast_charge = data['blast_charge'].values
    blast_flag = data['blast_flag'].values
    disp = data['displacement'].values

    # 计算共享特征
    rain_eff = compute_rain_effective(rainfall)
    micro_cum = compute_micro_cumulative(micro)
    blast_feat = compute_blast_feature(blast_dist, blast_charge, blast_flag)

    # 速率: 使用S-G滤波平滑位移后计算梯度, 再取非负
    disp_smooth = savgol_filter(disp, window_length=21, polyorder=3)
    v_obs = np.maximum(np.gradient(disp_smooth), 0)  # 速率非负

    # 找出各自的最佳孔压阈值 (搜索)
    n_stages = len(np.unique(stage_labels))
    stage_params = []

    for s in sorted(np.unique(stage_labels)):
        mask = (stage_labels == s)
        if np.sum(mask) < 20:
            stage_params.append({
                'v_basic': 0.0, 'gamma': 0.0, 'delta': 0.0,
                'eta': 0.0, 'beta': 0.0, 'pore_threshold': 0.0
            })
            continue

        v_seg = v_obs[mask]
        rain_seg = rain_eff[mask]
        micro_seg = micro_cum[mask]
        blast_seg = blast_feat[mask]
        pore_seg = pore[mask]

        # 搜索最佳孔压阈值 (使拟合残差最小)
        best_rss = np.inf
        best_pth = 0.0
        best_coeff = None

        for pth_pct in range(0, 101, 5):
            pth = np.percentile(pore_seg, pth_pct)
            pore_excess = np.maximum(0, pore_seg - pth)

            # 设计矩阵 [1, R_eff, max(0,P-P_th), M_cum, B]
            A = np.column_stack([np.ones(np.sum(mask)), rain_seg, pore_excess, micro_seg, blast_seg])
            coeff, rss = nnls(A, v_seg)
            if rss < best_rss:
                best_rss = rss
                best_pth = pth
                best_coeff = coeff

        stage_params.append({
            'v_basic': float(best_coeff[0]),
            'gamma': float(best_coeff[1]),
            'delta': float(best_coeff[2]),
            'eta': float(best_coeff[3]),
            'beta': float(best_coeff[4]),
            'pore_threshold': float(best_pth),
        })

        print(f"  阶段{s}: v_b={best_coeff[0]:.6f}, γ={best_coeff[1]:.6f}, "
              f"δ={best_coeff[2]:.6f}, η={best_coeff[3]:.6f}, β={best_coeff[4]:.6f}, "
              f"P_th={best_pth:.2f}")

    return stage_params, v_obs


def reconstruct_displacement(data, stage_params, stage_labels, init_disp=0.0):
    """
    用拟合的速率模型重构/预测位移
    """
    N = len(data)
    rainfall = data['rainfall'].values
    micro = data['micro'].values
    pore = data['pore'].values
    blast_dist = data['blast_dist'].values
    blast_charge = data['blast_charge'].values
    blast_flag = data['blast_flag'].values

    rain_eff = compute_rain_effective(rainfall)
    micro_cum = compute_micro_cumulative(micro)
    blast_feat = compute_blast_feature(blast_dist, blast_charge, blast_flag)

    v = np.zeros(N)
    v_basic_a = np.zeros(N)
    v_rain_a = np.zeros(N)
    v_pore_a = np.zeros(N)
    v_micro_a = np.zeros(N)
    v_blast_a = np.zeros(N)

    for i in range(N):
        s_idx = int(stage_labels[i]) - 1
        if s_idx < 0 or s_idx >= len(stage_params):
            sp = stage_params[1] if len(stage_params) > 1 else stage_params[0]
        else:
            sp = stage_params[s_idx]

        v_basic_i = sp['v_basic']
        gamma_i = sp['gamma']
        delta_i = sp['delta']
        eta_i = sp['eta']
        beta_i = sp['beta']
        pth_i = sp['pore_threshold']

        v_rain_i = gamma_i * rain_eff[i]
        v_pore_i = delta_i * max(0, pore[i] - pth_i)
        v_micro_i = eta_i * micro_cum[i]
        v_blast_i = beta_i * blast_feat[i]

        v[i] = v_basic_i + v_rain_i + v_pore_i + v_micro_i + v_blast_i

        v_basic_a[i] = v_basic_i
        v_rain_a[i] = v_rain_i
        v_pore_a[i] = v_pore_i
        v_micro_a[i] = v_micro_i
        v_blast_a[i] = v_blast_i

    # 积分: D(t) = D(0) + Σ v(τ)
    D = np.zeros(N)
    D[0] = init_disp
    for i in range(1, N):
        D[i] = D[i - 1] + v[i]

    components = {
        'v_basic': v_basic_a, 'v_rain': v_rain_a, 'v_pore': v_pore_a,
        'v_micro': v_micro_a, 'v_blast': v_blast_a, 'v_total': v
    }

    return D, v, components


def auto_stage_labels(disp, N):
    """根据位移水平自动划分三阶段"""
    disp_min = np.min(disp)
    disp_max = np.max(disp)
    th1 = disp_min + (disp_max - disp_min) * 0.05  # 5%
    th2 = disp_min + (disp_max - disp_min) * 0.30  # 30%
    labels = np.ones(N, dtype=int)
    labels[disp >= th2] = 3
    labels[(disp >= th1) & (disp < th2)] = 2
    return labels


# ================================================================
#  4. 主流程
# ================================================================
def run_model(train_df, test_df):
    """完整运行速率叠加模型"""
    print("\n" + "=" * 72)
    print("  Q4 速率叠加模型 — 主流程")
    print("=" * 72)

    # ── 预处理 ──
    train_data = preprocess(train_df, is_train=True)
    test_data = preprocess(test_df, is_train=False)

    N_train = len(train_data)
    N_test = len(test_data)

    # ── 三阶段划分 (训练集) ──
    disp_train = train_data['displacement'].values
    stage_train = auto_stage_labels(disp_train, N_train)
    train_data['stage'] = stage_train

    n1 = int(np.sum(stage_train == 1))
    n2 = int(np.sum(stage_train == 2))
    n3 = int(np.sum(stage_train == 3))
    print(f"\n  三阶段划分 (训练集):")
    print(f"    阶段一 (缓慢): {n1} 点 ({n1 / N_train * 100:.1f}%)")
    print(f"    阶段二 (加速): {n2} 点 ({n2 / N_train * 100:.1f}%)")
    print(f"    阶段三 (快速): {n3} 点 ({n3 / N_train * 100:.1f}%)")

    # ── 拟合 ──
    print(f"\n  参数拟合 (NNLS):")
    stage_params, v_obs = fit_rate_model(train_data, stage_train)

    # ── 重构训练集位移 ──
    D_pred_train, v_pred_train, comp_train = reconstruct_displacement(
        train_data, stage_params, stage_train, init_disp=disp_train[0])

    # ── 评估 ──
    res_train = disp_train - D_pred_train
    rmse = float(np.sqrt(np.mean(res_train ** 2)))
    mae = float(np.mean(np.abs(res_train)))
    ss_res = np.sum(res_train ** 2)
    ss_tot = np.sum((disp_train - np.mean(disp_train)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    nrmse = rmse / (np.max(disp_train) - np.min(disp_train)) * 100

    print(f"\n  训练集评估:")
    print(f"    RMSE  = {rmse:.4f} mm")
    print(f"    MAE   = {mae:.4f} mm")
    print(f"    NRMSE = {nrmse:.2f} %")
    print(f"    R²    = {r2:.4f}")

    # ── 实验集预测 ──
    if 'stage' in test_data.columns:
        stage_test = test_data['stage'].values.astype(int)
    else:
        # 实验集没有位移, 用训练集阈值
        stage_test = auto_stage_labels(np.zeros(N_test), N_test)  # 退化
        # 更合理的: 使用时间比例划分
        stage_test[:N_test // 2] = 2
        stage_test[N_test // 2:] = 3

    test_data['stage'] = stage_test
    D_pred_test, v_pred_test, comp_test = reconstruct_displacement(
        test_data, stage_params, stage_test, init_disp=0.0)

    # ── 整理结果 ──
    results = {
        'train': {
            'data': train_data,
            'displacement_pred': D_pred_train,
            'v_obs': v_obs,
            'v_pred': v_pred_train,
            'components': comp_train,
            'residual': res_train,
            'rmse': rmse, 'mae': mae, 'r2': r2, 'nrmse': nrmse,
            'params': stage_params,
        },
        'test': {
            'data': test_data,
            'displacement_pred': D_pred_test,
            'v_pred': v_pred_test,
            'components': comp_test,
        }
    }

    return results


# ================================================================
#  5. 可视化
# ================================================================
def plot_results(results, save_path=None):
    """绘制4×2综合结果图"""
    train = results['train']
    test = results['test']
    train_data = train['data']
    test_data = test['data']
    t_train = train_data['t_hour'].values / 24.0  # 天
    t_test = test_data['t_hour'].values / 24.0

    comp = train['components']

    fig, axes = plt.subplots(4, 2, figsize=(20, 18))
    fig.suptitle('Q4 速率叠加模型\n'
                 'v(t) = v_basic + γ·R_eff + δ·(P−P_th)⁺ + η·M_cum + β·B',
                 fontsize=14, fontweight='bold', y=0.98)

    ax1, ax2 = axes[0, 0], axes[0, 1]
    ax3, ax4 = axes[1, 0], axes[1, 1]
    ax5, ax6 = axes[2, 0], axes[2, 1]
    ax7, ax8 = axes[3, 0], axes[3, 1]

    # (a) 训练集位移
    ax1.plot(t_train, train_data['displacement'].values, 'b-', lw=0.8, alpha=0.5, label='实测')
    ax1.plot(t_train, train['displacement_pred'], 'r-', lw=2.0,
             label=f'速率叠加重构 (R²={train["r2"]:.4f})')
    ax1.fill_between(t_train, train_data['displacement'].values, train['displacement_pred'],
                     color='gray', alpha=0.1, label=f'残差 RMSE={train["rmse"]:.2f}mm')
    for s, c in [(1, 'green'), (2, 'orange'), (3, 'red')]:
        mask = train_data['stage'].values == s
        if np.sum(mask) > 0:
            idx = np.where(mask)[0]
            ax1.axvspan(t_train[idx[0]], t_train[idx[-1]], alpha=0.05, color=c)
            ax1.text(t_train[(idx[0] + idx[-1]) // 2], ax1.get_ylim()[1] * 0.9,
                     f'S{s}', ha='center', fontweight='bold', fontsize=10, color=c)
    ax1.set_ylabel('位移 (mm)', fontsize=12)
    ax1.set_title('(a) 训练集: 实测 vs 速率叠加模型重构', fontsize=13)
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(alpha=0.2)

    # (b) 速率对比
    ax2.plot(t_train, train['v_obs'], 'gray', lw=0.4, alpha=0.3, label='观测速率')
    ax2.plot(t_train, comp['v_total'], 'r-', lw=1.5, label='模型速率')
    for s, c in [(1, 'green'), (2, 'orange'), (3, 'red')]:
        mask = train_data['stage'].values == s
        if np.sum(mask) > 0:
            idx = np.where(mask)[0]
            ax2.axvspan(t_train[idx[0]], t_train[idx[-1]], alpha=0.05, color=c)
    ax2.set_ylabel('速率 (mm/10min)', fontsize=12)
    ax2.set_title('(b) 速率: 观测 vs 模型 (平滑后)', fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.2)

    # (c) 速率分量堆叠
    base = np.zeros(len(t_train))
    layers = []
    labels = []
    colors = []
    for k, lb, cl in [('v_basic', '基础蠕变', '#888888'), ('v_rain', '降雨', '#1f77b4'),
                       ('v_pore', '孔压', '#2ca02c'), ('v_micro', '微震', '#9467bd'),
                       ('v_blast', '爆破', '#d62728')]:
        layers.append(base + comp[k])
        labels.append(lb)
        colors.append(cl)
        base = base + comp[k]

    ax3.stackplot(t_train, *layers, labels=labels, colors=colors, alpha=0.6)
    ax3.set_ylabel('速率 (mm/10min)', fontsize=12)
    ax3.set_title('(c) 速率分量堆叠图', fontsize=13)
    ax3.legend(fontsize=8, loc='upper left')
    ax3.grid(alpha=0.2)

    # (d) 各阶段平均贡献柱状图
    stage_names = ['阶段一 (缓慢)', '阶段二 (加速)', '阶段三 (快速)']
    mean_comp = {k: [] for k in ['v_basic', 'v_rain', 'v_pore', 'v_micro', 'v_blast']}
    for s in [1, 2, 3]:
        mask = train_data['stage'].values == s
        for k in mean_comp:
            mean_comp[k].append(np.mean(comp[k][mask]) if np.sum(mask) > 0 else 0.0)

    x = np.arange(3)
    width = 0.15
    labels_dict = {'v_basic': '基础', 'v_rain': '降雨', 'v_pore': '孔压',
                   'v_micro': '微震', 'v_blast': '爆破'}
    colors_list = ['#888888', '#1f77b4', '#2ca02c', '#9467bd', '#d62728']
    for i, (k, c) in enumerate(zip(mean_comp.keys(), colors_list)):
        ax4.bar(x + i * width, mean_comp[k], width, label=labels_dict[k], color=c, alpha=0.7)
    ax4.set_xticks(x + width * 2)
    ax4.set_xticklabels(stage_names, fontsize=9)
    ax4.set_ylabel('平均速率贡献 (mm/10min)', fontsize=12)
    ax4.set_title('(d) 各阶段平均速率贡献分解', fontsize=13)
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.2, axis='y')

    # (e) 实验集预测位移
    ax5.plot(t_test, test['displacement_pred'], 'r-', lw=2.0, label='模型预测')
    for s, c in [(1, 'green'), (2, 'orange'), (3, 'red')]:
        mask = test_data['stage'].values == s
        if np.sum(mask) > 0:
            idx = np.where(mask)[0]
            ax5.axvspan(t_test[idx[0]], t_test[idx[-1]], alpha=0.05, color=c)
    ax5.set_xlabel('时间 (天)', fontsize=12)
    ax5.set_ylabel('位移 (mm)', fontsize=12)
    ax5.set_title('(e) 实验集: 位移预测', fontsize=13)
    ax5.legend(fontsize=9)
    ax5.grid(alpha=0.2)

    # (f) 实验集速率
    ax6.plot(t_test, test['v_pred'], 'r-', lw=1.5, label='预测速率')
    for s, c in [(1, 'green'), (2, 'orange'), (3, 'red')]:
        mask = test_data['stage'].values == s
        if np.sum(mask) > 0:
            idx = np.where(mask)[0]
            ax6.axvspan(t_test[idx[0]], t_test[idx[-1]], alpha=0.05, color=c)
    ax6.set_xlabel('时间 (天)', fontsize=12)
    ax6.set_ylabel('速率 (mm/10min)', fontsize=12)
    ax6.set_title('(f) 实验集: 速率预测', fontsize=13)
    ax6.legend(fontsize=9)
    ax6.grid(alpha=0.2)

    # (g) 残差分布
    res = train['residual']
    ax7.hist(res, bins=80, color='steelblue', edgecolor='white', alpha=0.7, density=True)
    mu_r, std_r = np.mean(res), np.std(res)
    x_n = np.linspace(mu_r - 4 * std_r, mu_r + 4 * std_r, 200)
    ax7.plot(x_n, 1 / (std_r * np.sqrt(2 * np.pi)) *
             np.exp(-(x_n - mu_r) ** 2 / (2 * std_r ** 2)),
             'r-', lw=2, label=f'N({mu_r:.1f},{std_r:.1f})')
    ax7.axvline(0, color='gray', ls='--', lw=1)
    ax7.set_xlabel('残差 (mm)', fontsize=12)
    ax7.set_ylabel('密度', fontsize=12)
    ax7.set_title(f'(g) 残差分布 (MAE={train["mae"]:.1f}mm)', fontsize=13)
    ax7.legend(fontsize=9)
    ax7.grid(alpha=0.2)

    # (h) 残差时序
    ax8.plot(t_train, res, 'k-', lw=0.5, alpha=0.5, label='残差')
    ax8.axhline(0, color='gray', lw=1)
    ax8.axhline(2 * std_r, color='red', ls='--', lw=1, alpha=0.5, label=f'±2σ={2 * std_r:.1f}mm')
    ax8.axhline(-2 * std_r, color='red', ls='--', lw=1, alpha=0.5)
    ax8.fill_between(t_train, -2 * std_r, 2 * std_r, color='red', alpha=0.04)
    ax8.set_xlabel('时间 (天)', fontsize=12)
    ax8.set_ylabel('残差 (mm)', fontsize=12)
    ax8.set_title(f'(h) 残差时序 (NRMSE={train["nrmse"]:.1f}%)', fontsize=13)
    ax8.legend(fontsize=9)
    ax8.grid(alpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"\n  图像保存: {save_path}")
    plt.close()
    return fig


def plot_components_detail(results, save_path=None):
    """绘制各分量详细时序"""
    train = results['train']
    train_data = train['data']
    t_train = train_data['t_hour'].values / 24.0
    comp = train['components']

    fig, axes = plt.subplots(5, 1, figsize=(18, 14), sharex=True)
    fig.suptitle('速率叠加分量详细时序 (训练集)', fontsize=14, fontweight='bold', y=0.97)

    keys = ['v_basic', 'v_rain', 'v_pore', 'v_micro', 'v_blast']
    y_labels = ['基础蠕变\n(mm/10min)', '降雨贡献\n(mm/10min)', '孔压贡献\n(mm/10min)',
                '微震贡献\n(mm/10min)', '爆破贡献\n(mm/10min)']
    colors = ['#888888', '#1f77b4', '#2ca02c', '#9467bd', '#d62728']

    for ax, key, yl, c in zip(axes, keys, y_labels, colors):
        ax.plot(t_train, comp[key], '-', color=c, lw=1.0, alpha=0.7)
        ax.fill_between(t_train, 0, comp[key], color=c, alpha=0.1)
        ax.set_ylabel(yl, fontsize=10)
        ax.grid(alpha=0.2)
        for s, cs in [(1, 'green'), (2, 'orange'), (3, 'red')]:
            mask = train_data['stage'].values == s
            if np.sum(mask) > 0:
                idx = np.where(mask)[0]
                ax.axvspan(t_train[idx[0]], t_train[idx[-1]], alpha=0.03, color=cs)

    axes[-1].set_xlabel('时间 (天)', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"  分量图保存: {save_path}")
    plt.close()
    return fig


# ================================================================
#  6. 主入口
# ================================================================
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    print("\n" + "░" * 72)
    print("  Q4 速率叠加模型 (Rate Superposition Model)")
    print("  v(t) = v_basic + γ·R_eff + δ·(P−P_th)⁺ + η·M_cum + β·B")
    print("  D(t) = D(0) + ∫₀ᵗ v(τ) dτ")
    print("░░" + "=" * 68 + "░")

    # ── 加载 ──
    train_df, test_df = load_data(script_dir)

    # ── 运行 ──
    results = run_model(train_df, test_df)

    # ── 图像 ──
    fig_path = os.path.join(script_dir, "rate_superposition_results.png")
    plot_results(results, save_path=fig_path)

    detail_path = os.path.join(script_dir, "rate_components_detail.png")
    plot_components_detail(results, save_path=detail_path)

    # ── 保存CSV ──
    train_df_out = results['train']['data'].copy()
    train_df_out['Predicted_Displacement'] = results['train']['displacement_pred']
    train_df_out['Rate_Observed'] = results['train']['v_obs']
    train_df_out['Rate_Predicted'] = results['train']['v_pred']
    train_df_out['Residual'] = results['train']['residual']

    test_df_out = results['test']['data'].copy()
    test_df_out['Predicted_Displacement'] = results['test']['displacement_pred']
    test_df_out['Rate_Predicted'] = results['test']['v_pred']

    train_csv = os.path.join(script_dir, "rate_superposition_train.csv")
    train_df_out.to_csv(train_csv, index=False, encoding='utf-8-sig')
    print(f"\n  训练集结果: {train_csv}")

    test_csv = os.path.join(script_dir, "rate_superposition_test.csv")
    test_df_out.to_csv(test_csv, index=False, encoding='utf-8-sig')
    print(f"  实验集预测: {test_csv}")

    # ── 参数总结 ──
    print("\n" + "=" * 72)
    print("  模型参数总结")
    print("=" * 72)
    for i, sp in enumerate(results['train']['params']):
        print(f"  阶段{i + 1}:")
        print(f"    v_basic  = {sp['v_basic']:.6f}  (基础蠕变速率)")
        print(f"    γ (gamma) = {sp['gamma']:.6f}  (降雨效率)")
        print(f"    δ (delta) = {sp['delta']:.6f}  (孔压效率)")
        print(f"    η (eta)   = {sp['eta']:.6f}  (微震效率)")
        print(f"    β (beta)  = {sp['beta']:.6f}  (爆破响应)")
        print(f"    P_th      = {sp['pore_threshold']:.2f} kPa")

    print(f"\n  最终指标:")
    print(f"    训练集: R²={results['train']['r2']:.4f}, "
          f"RMSE={results['train']['rmse']:.3f}mm")
    print("=" * 72)
    print("  Q4 速率叠加模型完成!")
    print("=" * 72)
