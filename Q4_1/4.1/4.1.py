"""
====================================================================
Q4.1 边坡变形阶段识别与分段匀加速分析
数据源: Q3/ap3.xlsx
方法: 中值滤波 → 自适应分段二次拟合（隐含匀加速）→ BIC选段
====================================================================

【使用说明】
1. 所有12个可调参数集中在【用户调参区】（程序开头）
2. 修改参数后直接运行即可
3. 输出：表面位移曲线图 + 分量图 + 详细诊断结果
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import medfilt

if sys.stdout.encoding == 'gbk':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
#  【用户调参区】12个可调参数
#  ============================================================
#  修改以下参数即可调整模型行为。调整后直接运行。
# ============================================================

# --- P1: 数据预处理（2个） ---
MEDFILT_K = 13           # 中值滤波窗口（奇数），13=约2h窗口
INTERP_NAN = True        # 是否线性插值NaN值

# --- P2: 分段搜索（3个） ---
MIN_SEG_LEN = 80         # 最小段长（点数），~13h，避免过短段
MAX_BREAKPOINTS = 18     # 最大断点数
BIC_STRATEGY = 1         # 选段策略: 0=BIC最小, 1=肘部, 2=手动

# --- P3: 模型选择（3个） ---
BIC_DISPLAY = 15         # BIC对比表显示的段数上限
MANUAL_NSEG = 8          # BIC_STRATEGY=2时使用
SIGMA_BAND = 2           # 预测带倍数（2=95%区间）

# --- P4: 绘图控制（4个） ---
FIG_W = 18               # 图宽（inches）
FIG_H = 14               # 图高（inches）
SAVE_PNG = True          # 是否保存图片
DPI_VAL = 200            # 图像分辨率


# ============================================================
#  数据读取
# ============================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "../../Q3/ap3.xlsx")
df = pd.read_excel(data_path)

t_idx = df["Serial No. "].values.astype(float)
D = df["e: Surface Displacement (mm)"].values.astype(float)
rain = df["a: Rainfall (mm)"].fillna(0).values.astype(float)
pore = df["b: Pore Water Pressure (kPa)"].values.astype(float)
micro = df["c: Microseismic Event Count"].fillna(0).values.astype(float)
deep = df["d: Deep Displacement (mm)"].values.astype(float)

N = len(D)
dt = 10 / 60
t_h = (t_idx - t_idx[0]) * dt
t_d = t_h / 24

# NaN插值
for name, arr in [("位移", D), ("孔压", pore), ("深部位移", deep)]:
    mask = np.isnan(arr)
    if np.any(mask):
        filled = np.interp(np.arange(N), np.arange(N)[~mask], arr[~mask])
        if name == "位移":   D = filled
        if name == "孔压":   pore = filled
        if name == "深部位移": deep = filled
        print(f"  [插值] {name}: {int(np.sum(mask))} 点")

y = medfilt(D, kernel_size=MEDFILT_K)

print("=" * 72)
print("  Q4.1 边坡变形阶段识别与分段匀加速分析")
print(f"  数据: {N} 点, Q3/ap3.xlsx")
print(f"  滤波: 中值滤波 k={MEDFILT_K}")
print("=" * 72)


# ============================================================
#  核心算法
# ============================================================

def quad_fit(t, y):
    """二次拟合 y = a*t^2 + b*t + c → (系数, RSS)"""
    A = np.vstack([t**2, t, np.ones_like(t)]).T
    coeff, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    rss = float(np.sum((y - A @ coeff)**2))
    return coeff, rss


def bic(N, rss, nseg):
    """BIC = N*log(RSS/N) + k*log(N), k=3*nseg"""
    return N * np.log(max(rss, 1e-20) / N) + 3 * nseg * np.log(N)


def greedy_search(t, y, max_bp, min_len):
    """贪心断点搜索"""
    N = len(y)
    bps, bic_hist = [], []
    segs = [(0, N - 1)]
    _, base_rss = quad_fit(t, y)
    bic_hist.append((0, bic(N, base_rss, 1), base_rss))
    for _ in range(max_bp):
        best = (-1, -1, -1)
        for si, (ss, se) in enumerate(segs):
            if se - ss + 1 < 2 * min_len:
                continue
            _, old_rss = quad_fit(t[ss:se+1], y[ss:se+1])
            for cp in range(ss + min_len, se - min_len + 1):
                _, rl = quad_fit(t[ss:cp+1], y[ss:cp+1])
                _, rr = quad_fit(t[cp+1:se+1], y[cp+1:se+1])
                imp = old_rss - (rl + rr)
                if imp > best[0]:
                    best = (imp, cp, si)
        if best[0] <= 0:
            break
        _, cp, si = best
        ss, se = segs.pop(si)
        segs += [(ss, cp), (cp + 1, se)]
        segs.sort(key=lambda x: x[0])
        bps.append(cp)
        cur_rss = sum(quad_fit(t[s:e+1], y[s:e+1])[1] for s, e in segs)
        bic_hist.append((len(bps), bic(N, cur_rss, len(bps) + 1), cur_rss))
    return np.array(sorted(bps)), bic_hist


# 搜索
bps_all, bic_hist = greedy_search(t_h, y, MAX_BREAKPOINTS, MIN_SEG_LEN)
print(f"\n  [搜索] 贪心搜索: {len(bps_all)} 个断点")

# 选段
bic_ns = np.array([h[0] + 1 for h in bic_hist])
bic_v = np.array([h[1] for h in bic_hist])
if BIC_STRATEGY == 0:
    n_sel = int(bic_ns[np.argmin(bic_v)])
elif BIC_STRATEGY == 1:
    d2 = np.diff(bic_v, 2)
    elbow = np.argmin(d2[:min(12, len(d2))]) + 2 if len(d2) >= 2 else len(bic_v) - 1
    n_sel = int(bic_ns[min(elbow, len(bic_ns) - 1)])
else:
    n_sel = MANUAL_NSEG

n_bp = max(0, n_sel - 1)
bp = bps_all[:n_bp] if n_bp > 0 else np.array([], dtype=int)

# 分段列表
seg = []
p = 0
for b in bp:
    seg.append((p, b))
    p = b + 1
seg.append((p, N - 1))

# 拟合
y_fit = np.zeros(N)
coeffs = []
for s, e in seg:
    A = np.vstack([t_h[s:e+1]**2, t_h[s:e+1], np.ones(e - s + 1)]).T
    c, _, _, _ = np.linalg.lstsq(A, y[s:e+1], rcond=None)
    y_fit[s:e+1] = A @ c
    coeffs.append(c)

# 评价
res = D - y_fit
rmse = float(np.sqrt(np.mean(res**2)))
mae = float(np.mean(np.abs(res)))
nrmse = rmse / (np.max(D) - np.min(D)) * 100
r2 = 1 - np.sum(res**2) / np.sum((D - np.mean(D))**2)

# BIC表
print(f"  [选段] 策略={BIC_STRATEGY}, 选择 {n_sel} 段 ({n_bp} 断点, RMSE={rmse:.2f}mm)")
print(f"\n  BIC对比表:")
print(f"  {'段数':>4} {'BIC':>12} {'RSS':>14}")
for n, b, r in bic_hist:
    ns = n + 1
    if ns > BIC_DISPLAY:
        break
    mark = " <--" if ns == n_sel else ""
    print(f"  {ns:>4} {b:>12.1f} {r:>14.2f}{mark}")

# 详细分段结果
print(f"\n{'='*72}")
print(f"  分段匀加速详细结果 ({n_sel} 段)")
print(f"{'='*72}")
for i, (s, e) in enumerate(seg):
    c = coeffs[i]
    a_2 = 2 * c[0]  # 加速度 = 2*a
    seg_res = D[s:e+1] - y_fit[s:e+1]
    seg_rmse = float(np.sqrt(np.mean(seg_res**2)))
    t_start, t_end = t_h[s], t_h[e]
    motion = "加速" if a_2 > 1e-6 else ("减速" if a_2 < -1e-6 else "匀速")
    print(f"  S{i+1}: [{t_start:.0f}h~{t_end:.0f}h] ({t_start/24:.1f}~{t_end/24:.1f}天)")
    print(f"        点数={e-s+1}, 加速度={a_2:+.6f} mm/h² ({motion})")
    print(f"        拟合: d={c[0]:.6f}t²+{c[1]:.6f}t+{c[2]:.2f}, RMSE={seg_rmse:.3f}mm")

print(f"\n{'='*44}")
print(f"  {'指标':>16}   {'数值':>12}")
print(f"  {'='*44}")
print(f"  {'RMSE':>16}   {rmse:>10.3f} mm")
print(f"  {'MAE':>16}   {mae:>10.3f} mm")
print(f"  {'NRMSE':>16}   {nrmse:>9.2f} %")
print(f"  {'R²':>16}   {r2:>10.4f}")
print(f"  {'='*44}")
print(f"  [说明] 数据呈振荡特征,R²参考价值有限,NRMSE更可靠")


# ============================================================
#  绘图
# ============================================================
fig, axes = plt.subplots(3, 2, figsize=(FIG_W, FIG_H))
ax1, ax2, ax3 = axes[0, 0], axes[0, 1], axes[1, 0]
ax4, ax5, ax6 = axes[1, 1], axes[2, 0], axes[2, 1]

# 颜色表
C = plt.cm.Set2(np.linspace(0, 1, max(n_sel, 3)))

# -- 图1: 主图 --
ax1.plot(t_d, D, '-', color='#1f77b4', lw=0.7, alpha=0.5, label='实测位移')
ax1.plot(t_d, y_fit, '-', color='#d62728', lw=2.2, label=f'分段匀加速拟合 ({n_sel}段)')
std_r = float(np.std(res))
ax1.fill_between(t_d, y_fit - SIGMA_BAND * std_r, y_fit + SIGMA_BAND * std_r,
                 color='#d62728', alpha=0.06,
                 label=f'±{SIGMA_BAND}σ = {SIGMA_BAND * std_r:.1f}mm')
for b in bp:
    ax1.axvline(t_d[b], color='gray', ls='--', lw=0.8, alpha=0.5)
for i, (s, e) in enumerate(seg):
    mid = (s + e) // 2
    ax1.annotate(f'S{i+1}', (t_d[mid], np.max(D) * 0.95),
                 fontsize=11, fontweight='bold', ha='center', color=C[i])
ax1.set_ylabel('表面位移 (mm)', fontsize=12)
ax1.set_title(f'Q3 边坡表面位移: 实测 vs 分段匀加速 (RMSE={rmse:.2f}mm)',
              fontsize=13, fontweight='bold')
ax1.legend(fontsize=9, loc='upper left')
ax1.grid(alpha=0.2)

# -- 图2: 残差 --
ax2.plot(t_d, res, '-', color='black', lw=0.5, alpha=0.6)
ax2.axhline(0, color='gray', lw=0.5)
ax2.axhline(SIGMA_BAND * std_r, color='red', ls='--', lw=1, alpha=0.5,
            label=f'±{SIGMA_BAND}σ={SIGMA_BAND * std_r:.2f}mm')
ax2.axhline(-SIGMA_BAND * std_r, color='red', ls='--', lw=1, alpha=0.5)
ax2.fill_between(t_d, -SIGMA_BAND * std_r, SIGMA_BAND * std_r, color='red', alpha=0.04)
for b in bp:
    ax2.axvline(t_d[b], color='gray', ls='--', lw=0.6, alpha=0.3)
ax2.set_ylabel('残差 (mm)', fontsize=12)
ax2.set_title(f'残差 (MAE={mae:.2f}mm, NRMSE={nrmse:.1f}%)', fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.2)

# -- 图3: 分段趋势 + 降雨 --
seg_t = np.zeros(N)
for i, (s, e) in enumerate(seg):
    A = np.vstack([t_h[s:e+1]**2, t_h[s:e+1], np.ones(e - s + 1)]).T
    seg_t[s:e+1] = A @ coeffs[i]
ax3.plot(t_d, seg_t, '-', color='#ff7f0e', lw=2.0, label='分段二次趋势')
ax3_t = ax3.twinx()
ax3_t.bar(t_d, rain, width=0.02, color='#1f77b4', alpha=0.3, label='降雨')
ax3_t.set_ylabel('降雨量 (mm)', fontsize=11, color='#1f77b4')
for b in bp:
    ax3.axvline(t_d[b], color='gray', ls='--', lw=0.6, alpha=0.3)
ax3.set_ylabel('位移趋势 (mm)', fontsize=12)
ax3.set_title(f'分段二次趋势 + 降雨量 ({n_sel}段)', fontsize=12)
ax3.legend(fontsize=9, loc='upper left')
ax3.grid(alpha=0.2)

# -- 图4: 加速度柱状图 + 孔压/微震/深度 --
a_arr = np.array([2 * c[0] for c in coeffs])
labels = [f'S{i+1}' for i in range(n_sel)]
colors_b = ['gray' if abs(a) < 1e-6 else ('green' if a > 0 else 'red') for a in a_arr]
ax4.bar(labels, a_arr, color=colors_b, alpha=0.6, edgecolor='black', lw=0.5, width=0.6)
ax4.axhline(0, color='black', lw=0.8)
ax4.set_ylabel('加速度 (mm/h²)', fontsize=12, color='darkgreen')
ax4.set_title('各段加速度 (负=减速, 正=加速)', fontsize=12)
ax4.grid(alpha=0.2, axis='y')

# -- 图5: 孔压+微震+深部 --
ax5.plot(t_d, pore, '-', color='#9467bd', lw=0.8, label='孔压')
ax5_t = ax5.twinx()
ax5_t.plot(t_d, micro, 'o', markersize=1, color='#8c564b', alpha=0.4, label='微震')
ax5_t.plot(t_d, deep, '-', color='#7f7f7f', lw=0.8, alpha=0.7, label='深部位移')
ax5_t.set_ylabel('微震/深部', fontsize=11, color='#8c564b')
lines1, labels1 = ax5.get_legend_handles_labels()
lines2, labels2 = ax5_t.get_legend_handles_labels()
ax5.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper left')
for b in bp:
    ax5.axvline(t_d[b], color='gray', ls='--', lw=0.6, alpha=0.3)
ax5.set_xlabel('时间 (天)', fontsize=12)
ax5.set_ylabel('孔压 (kPa)', fontsize=12)
ax5.set_title('孔压 / 微震 / 深部位移', fontsize=12)
ax5.grid(alpha=0.2)

# -- 图6: 拟合效果对比 --
ax6.plot(t_d, y, '-', color='#1f77b4', lw=0.7, alpha=0.5, label='滤波后')
ax6.plot(t_d, y_fit, '-', color='#d62728', lw=2.0, label=f'拟合 (RMSE={rmse:.2f}mm)')
for b in bp:
    ax6.axvline(t_d[b], color='gray', ls='--', lw=0.6, alpha=0.3)
ax6.set_xlabel('时间 (天)', fontsize=12)
ax6.set_ylabel('位移 (mm)', fontsize=12)
ax6.set_title(f'滤波后 vs 拟合 (R²={r2:.4f})', fontsize=12)
ax6.legend(fontsize=9)
ax6.grid(alpha=0.2)

plt.suptitle('Q4.1 边坡变形阶段识别 · 分段匀加速模型 (Q3/ap3.xlsx)',
             fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])

if SAVE_PNG:
    fig_path = os.path.join(script_dir, "4.1_multi_factor_model.png")
    plt.savefig(fig_path, dpi=DPI_VAL, bbox_inches='tight')
    print(f"\n  图像保存: {fig_path}")
plt.close()
print(f"  [完成] 12个可调参数 | 策略={['BIC最小','肘部','手动'][BIC_STRATEGY]}")
print("=" * 72)
