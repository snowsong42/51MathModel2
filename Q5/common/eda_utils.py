"""Q5 EDA 通用工具函数"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal

# 输出目录
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'EDA')

def ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)
    return OUT_DIR

def save_and_show(fig, path, dpi=150, tight=True):
    """保存图片（Agg）后，用 TkAgg 弹窗展示"""
    # 保存
    kw = dict(dpi=dpi, bbox_inches='tight') if tight else dict(dpi=dpi)
    fig.savefig(path, **kw)
    plt.close(fig)
    print(f"[保存] {path}")
    # 弹窗展示
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt2
    img = plt2.imread(path)
    fig2, ax2 = plt2.subplots(figsize=(10, 6))
    ax2.imshow(img)
    ax2.axis('off')
    ax2.set_title(os.path.basename(path))
    plt2.tight_layout()
    plt2.show(block=False)
    # 切回 Agg 避免后续绘图冲突
    matplotlib.use('Agg')

def classify_vars(all_cols, base_vars):
    """
    根据列名自动分类变量
    参数:
        all_cols: DataFrame 所有列名列表
        base_vars: load_pipeline() 返回的 base_vars（原始基础变量）
    返回:
        dict: {类别: [列名列表]}
    """
    # 从 base_vars 和 all_cols 中智能分类
    s = set(all_cols)
    b = set(base_vars) if base_vars else set()

    continuous = [c for c in s if any(k in c for k in ['PorePressure', 'Infiltration'])
                  and 'lag' not in c and 'cum' not in c]
    rainfall = [c for c in s if any(k in c for k in ['Rainfall', 'Time_since_rain'])
                and 'Displacement' not in c]
    microseismic = [c for c in s if 'Microseismic' in c]
    blast = [c for c in s if any(k in c for k in ['Blast', 'Time_since_blast'])]
    target = [c for c in s if c in ['Delta_D', 'Displacement', 'Disp_cum24']]

    return {
        'continuous': sorted(set(continuous)),
        'rainfall': sorted(set(rainfall)),
        'microseismic': sorted(set(microseismic)),
        'blast': sorted(set(blast)),
        'target': sorted(set(target)),
    }

def compute_variable_stats(df, cols):
    """
    计算单变量统计描述
    返回 DataFrame: 列为变量名，行为统计量
    """
    rows = []
    for c in cols:
        if c not in df.columns:
            continue
        s = df[c].dropna()
        if len(s) == 0:
            rows.append({'变量': c, '缺失比例': 1.0, '均值': np.nan, '标准差': np.nan,
                         '偏度': np.nan, '峰度': np.nan,
                         'min': np.nan, '25%': np.nan, '50%': np.nan, '75%': np.nan, 'max': np.nan})
            continue
        from scipy.stats import skew, kurtosis
        rows.append({
            '变量': c,
            '缺失比例': df[c].isna().mean(),
            '均值': s.mean(),
            '标准差': s.std(),
            '偏度': skew(s),
            '峰度': kurtosis(s),
            'min': s.min(),
            '25%': s.quantile(0.25),
            '50%': s.median(),
            '75%': s.quantile(0.75),
            'max': s.max(),
            '非空数量': len(s),
        })
    return pd.DataFrame(rows).set_index('变量')

def plot_rolling_mean(ax, timestamps, series, windows, colors=None, labels=None):
    """
    在给定 ax 上绘制原始序列 + 多窗口滑动平均
    参数:
        windows: 窗口大小列表（步数）
        colors: 颜色列表（可选）
        labels: 标签列表（可选）
    """
    ax.plot(timestamps, series, alpha=0.4, linewidth=0.6, label='原始', color='gray')
    if colors is None:
        colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
    if labels is None:
        labels = [f'{w}步' for w in windows]

    for w, c, lab in zip(windows, colors, labels):
        roll = pd.Series(series).rolling(w, center=True, min_periods=1).mean()
        ax.plot(timestamps, roll, color=c, linewidth=1.2, label=lab)
    ax.legend(fontsize=8)
    ax.set_xlabel('时间')
    ax.grid(True, alpha=0.3)

def ccf_compute(s1, s2, max_lag):
    """
    计算互相关函数（无偏估计）
    返回: lags, corrs
    """
    n = len(s1)
    lags = np.arange(-max_lag, max_lag + 1)
    corrs = np.zeros(len(lags))
    # 去均值
    x = s1 - np.nanmean(s1)
    y = s2 - np.nanmean(s2)
    # 填充 nan
    mask = np.isnan(x) | np.isnan(y)
    x = np.where(mask, 0, x)
    y = np.where(mask, 0, y)
    for i, lag in enumerate(lags):
        if lag < 0:
            c = np.corrcoef(x[-lag:], y[:lag])[0, 1] if len(x[-lag:]) > 1 else 0
        elif lag > 0:
            c = np.corrcoef(x[:-lag], y[lag:])[0, 1] if len(x[:-lag]) > 1 else 0
        else:
            c = np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0
        corrs[i] = c if not np.isnan(c) else 0
    return lags, corrs

def blast_response_extract(df, blast_mask, before=144, after=144):
    """
    提取爆破事件前后目标变量 Delta_D 的序列
    参数:
        blast_mask: 布尔序列，True 表示该时刻有爆破
        before: 向前提取的步数
        after: 向后提取的步数
    返回:
        list of arrays, 每个为 [before+after+1] 的序列
    """
    if 'Delta_D' not in df.columns:
        return []
    idx = np.where(blast_mask)[0]
    vals = df['Delta_D'].values
    sequences = []
    for i in idx:
        start = max(0, i - before)
        end = min(len(vals), i + after + 1)
        seq = vals[start:end]
        # 补齐到统一长度
        if len(seq) < before + after + 1:
            pad_before = before - (i - start)
            pad_after = after - (end - 1 - i)
            seq = np.pad(seq, (pad_before, pad_after), mode='constant', constant_values=np.nan)
        sequences.append(seq)
    return sequences

def effective_rainfall(rain_series, decay=0.85):
    """计算有效降雨: eff_rain(t) = rain(t) + decay * eff_rain(t-1)"""
    eff = np.zeros(len(rain_series))
    for i in range(len(rain_series)):
        if i == 0:
            eff[i] = rain_series[i]
        else:
            eff[i] = rain_series[i] + decay * eff[i-1]
    return eff
