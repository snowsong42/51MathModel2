"""
三段式形变建模与检验
根据识别出的阶段节点，将位移时序分为三阶段，分别建模并计算指标
对标 Q1 correct_and_test.m 的建模+检验部分
输出：
  - 图6 三段拟合曲线.png  原始位移 + 分段拟合曲线 + 节点标记
  - 图7 各阶段残差.png    三阶段残差分布
  - 命令窗口：各阶段模型参数与评估指标
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()
from sklearn.metrics import r2_score, mean_squared_error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# ===== matplotlib 显示设置 =====
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']  # 中文字体 + 英文字体fallback
plt.rcParams['axes.unicode_minus'] = False           # 修复负号显示为方块

def _load_clean_data():
    """读取清洗后数据"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, config.CLEAN_DATA)
    df = pd.read_excel(path, sheet_name=0)
    d = df["Surface Displacement (mm)"].values
    dt = config.DT
    t = np.arange(len(d)) * dt
    return d, t, dt


def fit_linear(t, d):
    """线性回归: d = a*t + b"""
    A = np.vstack([t, np.ones_like(t)]).T
    a, b = np.linalg.lstsq(A, d, rcond=None)[0]
    d_pred = a * t + b
    r2 = r2_score(d, d_pred)
    rmse = np.sqrt(mean_squared_error(d, d_pred))
    return a, b, d_pred, r2, rmse


def fit_quadratic(t, d):
    """二次回归: d = a*t^2 + b*t + c"""
    A = np.vstack([t**2, t, np.ones_like(t)]).T
    a, b, c = np.linalg.lstsq(A, d, rcond=None)[0]
    d_pred = a * t**2 + b * t + c
    r2 = r2_score(d, d_pred)
    rmse = np.sqrt(mean_squared_error(d, d_pred))
    return a, b, c, d_pred, r2, rmse


def modeling_with_nodes(idx1, idx2):
    """
    根据节点序号进行三段式建模
    参数：
      idx1: 节点1的位移点序号 (1-index)
      idx2: 节点2的位移点序号 (1-index)
    返回：
      stages: dict，包含各阶段的模型参数和指标
    """
    d, t, dt = _load_clean_data()

    # 阶段划分（注意：节点序号是1-index，转换为数组索引0-index）
    # 阶段I: 0 ~ idx1-1
    tI = t[:idx1]
    dI = d[:idx1]
    # 阶段II: idx1 ~ idx2-1
    tII = t[idx1:idx2]
    dII = d[idx1:idx2]
    # 阶段III: idx2 ~ end
    tIII = t[idx2:]
    dIII = d[idx2:]

    stages = {}

    # ---- 阶段I：线性 ----
    a1, b1, dI_pred, r2_1, rmse_1 = fit_linear(tI, dI)
    dur_I = tI[-1] - tI[0] if len(tI) > 1 else 0
    delta_d_I = dI[-1] - dI[0]
    v_mean_I = delta_d_I / dur_I if dur_I > 0 else 0
    stages["I"] = {
        "name": "Slow Constant", "model": "linear",
        "params": {"a": a1, "b": b1},
        "model_str": f"d = {a1:.6f} t + {b1:.4f}",
        "r2": r2_1, "rmse": rmse_1,
        "duration_h": dur_I, "delta_d_mm": delta_d_I,
        "v_mean": v_mean_I,
        "t": tI, "d": dI, "d_pred": dI_pred
    }
    _print_stage("I", stages["I"])

    # ---- 阶段II：二次 ----
    a2, b2, c2, dII_pred, r2_2, rmse_2 = fit_quadratic(tII, dII)
    dur_II = tII[-1] - tII[0]
    delta_d_II = dII[-1] - dII[0]
    v_mean_II = delta_d_II / dur_II
    stages["II"] = {
        "name": "Accelerating", "model": "quadratic",
        "params": {"a": a2, "b": b2, "c": c2},
        "model_str": f"d = {a2:.6e} t^2 + {b2:.6f} t + {c2:.4f}",
        "r2": r2_2, "rmse": rmse_2,
        "duration_h": dur_II, "delta_d_mm": delta_d_II,
        "v_mean": v_mean_II,
        "t": tII, "d": dII, "d_pred": dII_pred
    }
    _print_stage("II", stages["II"])

    # ---- 阶段III：线性 ----
    a3, b3, dIII_pred, r2_3, rmse_3 = fit_linear(tIII, dIII)
    dur_III = tIII[-1] - tIII[0]
    delta_d_III = dIII[-1] - dIII[0]
    v_mean_III = delta_d_III / dur_III
    stages["III"] = {
        "name": "Rapid", "model": "linear",
        "params": {"a": a3, "b": b3},
        "model_str": f"d = {a3:.6f} t + {b3:.4f}",
        "r2": r2_3, "rmse": rmse_3,
        "duration_h": dur_III, "delta_d_mm": delta_d_III,
        "v_mean": v_mean_III,
        "t": tIII, "d": dIII, "d_pred": dIII_pred
    }
    _print_stage("III", stages["III"])

    return stages


def _print_stage(label, s):
    """打印单个阶段的建模结果"""
    print(f"\n  阶段 {label} ({s['name']})")
    print(f"    模型: {s['model_str']}")
    print(f"    R² = {s['r2']:.4f}, RMSE = {s['rmse']:.3f} mm")
    print(f"    持续时间: {s['duration_h']:.1f} h, 位移增量: {s['delta_d_mm']:.2f} mm")
    print(f"    平均速度: {s['v_mean']:.4f} mm/h")


def plot_fitting_curve(stages, t_full, d_full, idx1, t1, idx2, t2, save_dir):
    """图6：原始位移 + 分段拟合曲线 + 节点标记 + 三阶段色块"""
    plt.figure(figsize=(12, 6))
    ax = plt.gca()

    t_end = t_full[-1] / 24
    t1_day = t1 / 24
    t2_day = t2 / 24

    # 三阶段色块背景（与图5统一）
    ax.axvspan(0, t1_day, alpha=0.06, color='blue', label='阶段 I (匀速)')
    ax.axvspan(t1_day, t2_day, alpha=0.06, color='green', label='阶段 II (加速)')
    ax.axvspan(t2_day, t_end, alpha=0.06, color='red', label='阶段 III (快速)')

    # 原始数据调暗，拟合曲线加粗
    plt.plot(t_full / 24, d_full, 'k-', linewidth=0.5, alpha=0.5,
             label='滤波后位移')

    colors = {'I': 'b', 'II': 'g', 'III': 'r'}
    stage_names = {'I': '匀速', 'II': '加速', 'III': '快速'}
    for label in ['I', 'II', 'III']:
        s = stages[label]
        plt.plot(s['t'] / 24, s['d_pred'], f'{colors[label]}--',
                 linewidth=2.5, label=f'阶段{label} ({stage_names[label]}, R²={s["r2"]:.4f})')

    # 节点线：黑色加粗虚线
    if idx1:
        plt.axvline(x=t1_day, color='black', linestyle='--', linewidth=2, alpha=0.7,
                    label=f'节点1 t={t1_day:.1f}d')
    if idx2:
        plt.axvline(x=t2_day, color='black', linestyle='--', linewidth=2, alpha=0.7,
                    label=f'节点2 t={t2_day:.1f}d')

    plt.xlabel('时间 (天)', fontsize=12)
    plt.ylabel('位移 (mm)', fontsize=12)
    plt.title('三段式形变建模', fontsize=14)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, '图6 三段拟合曲线.png')
    plt.savefig(path, dpi=300)
    plt.show()
    print(f"[图6] 三段拟合曲线 → {path}")


def plot_residuals(stages, save_dir):
    """图7：各阶段残差 + 整体残差（2×2布局）"""
    colors = {'I': 'b', 'II': 'g', 'III': 'r'}
    labels = {'I': '匀速阶段', 'II': '加速阶段', 'III': '快速阶段'}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for i, label in enumerate(['I', 'II', 'III']):
        ax = axes[i // 2][i % 2]
        s = stages[label]
        res = s['d'] - s['d_pred']
        res_std = np.std(res)

        # 误差带
        ax.axhline(0, color='k', linestyle='-', linewidth=1)
        ax.axhline(2 * res_std, color='dimgray', linestyle='--', linewidth=1.0, alpha=0.7)
        ax.axhline(-2 * res_std, color='dimgray', linestyle='--', linewidth=1.0, alpha=0.7)
        ax.axhline(3 * res_std, color='gray', linestyle=':', linewidth=1.0, alpha=0.5)
        ax.axhline(-3 * res_std, color='gray', linestyle=':', linewidth=1.0, alpha=0.5)
        ax.fill_between(s['t'] / 24, -2 * res_std, 2 * res_std,
                        color='gray', alpha=0.15, label='±2σ')
        ax.fill_between(s['t'] / 24, -3 * res_std, 3 * res_std,
                        color='gray', alpha=0.10, label='±3σ')

        # 残差点
        ax.plot(s['t'] / 24, res, f'{colors[label]}.', markersize=3, alpha=0.3)
        ax.set_ylabel('残差 (mm)', fontsize=10)

        # R²和RMSE文字框-左上角统一位置
        ax.text(0.02, 0.95, f'R²={s["r2"]:.4f}\nRMSE={s["rmse"]:.3f} mm',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
        ax.set_title(f'阶段{label} ({labels[label]}) 残差', fontsize=12)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8, loc='upper right')

    # 右下：整体残差
    ax = axes[1][1]
    t_all = np.concatenate([stages[l]['t'] for l in ['I', 'II', 'III']])
    d_all = np.concatenate([stages[l]['d'] for l in ['I', 'II', 'III']])
    d_pred_all = np.concatenate([stages[l]['d_pred'] for l in ['I', 'II', 'III']])
    res_all = d_all - d_pred_all
    res_std_all = np.std(res_all)
    r2_all = r2_score(d_all, d_pred_all)
    rmse_all = np.sqrt(mean_squared_error(d_all, d_pred_all))

    ax.axhline(0, color='k', linestyle='-', linewidth=1)
    ax.axhline(2 * res_std_all, color='dimgray', linestyle='--', linewidth=1.0, alpha=0.7)
    ax.axhline(-2 * res_std_all, color='dimgray', linestyle='--', linewidth=1.0, alpha=0.7)
    ax.axhline(3 * res_std_all, color='gray', linestyle=':', linewidth=1.0, alpha=0.5)
    ax.axhline(-3 * res_std_all, color='gray', linestyle=':', linewidth=1.0, alpha=0.5)
    ax.fill_between(t_all / 24, -2 * res_std_all, 2 * res_std_all,
                    color='gray', alpha=0.15, label='±2σ')
    ax.fill_between(t_all / 24, -3 * res_std_all, 3 * res_std_all,
                    color='gray', alpha=0.10, label='±3σ')

    ax.plot(t_all / 24, res_all, '.', color='purple', markersize=3, alpha=0.3)
    ax.set_ylabel('残差 (mm)', fontsize=10)
    ax.text(0.02, 0.95, f'R²={r2_all:.4f}\nRMSE={rmse_all:.3f} mm',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
    ax.set_title('整体残差', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='upper right')

    fig.text(0.5, 0.02, '时间 (天)', ha='center', fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    path = os.path.join(save_dir, '图7 各阶段残差.png')
    plt.savefig(path, dpi=300)
    plt.show()
    print(f"[图7] 各阶段残差 → {path}")


def main():
    """独立运行：检测节点 → 三段建模 → 画图6+图7"""
    print("=" * 55)
    print("三段式形变建模")
    print("=" * 55)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 读取清洗后数据
    d, t, dt = _load_clean_data()

    # 获取节点
    from detect_nodes import sliding_window_detect
    idx1, t1, idx2, t2 = sliding_window_detect()

    if idx1 is None or idx2 is None:
        print("\n[警告] 滑动窗口法失败，尝试MAD+持久性法...")
        from detect_nodes import mad_persistence_detect
        idx1, t1, idx2, t2 = mad_persistence_detect()

    if idx1 is None or idx2 is None:
        print("\n[错误] 两种检测方法均未能识别节点")
        return

    print(f"\n使用节点: 节点1=#{idx1} (t={t1/24:.2f}d), 节点2=#{idx2} (t={t2/24:.2f}d)")
    stages = modeling_with_nodes(idx1, idx2)

    # 汇总表
    print("\n" + "=" * 55)
    print("阶段汇总")
    print("-" * 55)
    print(f"{'阶段':<8} {'模型':<12} {'R²':>8} {'RMSE(mm)':>10} {'平均速度(mm/h)':>16}")
    print("-" * 55)
    for label in ["I", "II", "III"]:
        s = stages[label]
        print(f"{label:<8} {s['model']:<12} {s['r2']:>8.4f} {s['rmse']:>10.3f} {s['v_mean']:>16.4f}")
    print("=" * 55)

    # 画图6和图7
    print("\n>>> 生成图6：三段拟合曲线...")
    plot_fitting_curve(stages, t, d, idx1, t1, idx2, t2, script_dir)
    print("\n>>> 生成图7：各阶段残差...")
    plot_residuals(stages, script_dir)

    plt.show(block=True)


if __name__ == "__main__":
    main()
