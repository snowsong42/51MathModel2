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
    print(f"\n  Stage {label} ({s['name']})")
    print(f"    Model: {s['model_str']}")
    print(f"    R^2 = {s['r2']:.4f}, RMSE = {s['rmse']:.3f} mm")
    print(f"    Duration: {s['duration_h']:.1f} h, Delta d: {s['delta_d_mm']:.2f} mm")
    print(f"    Mean velocity: {s['v_mean']:.4f} mm/h")


def plot_fitting_curve(stages, t_full, d_full, idx1, t1, idx2, t2, save_dir):
    """图6：原始位移 + 分段拟合曲线 + 节点标记"""
    plt.figure(figsize=(12, 6))
    plt.plot(t_full / 24, d_full, 'k-', linewidth=0.6, alpha=0.7,
             label='Filtered displacement')

    colors = {'I': 'b', 'II': 'g', 'III': 'r'}
    for label in ['I', 'II', 'III']:
        s = stages[label]
        plt.plot(s['t'] / 24, s['d_pred'], f'{colors[label]}--',
                 linewidth=2, label=f'Stage {label} ({s["model"]} fit)')

    # 标记节点
    if idx1:
        plt.axvline(x=t1 / 24, color='blue', linestyle=':', linewidth=1.5,
                    label=f'Node1 t={t1/24:.1f}d')
    if idx2:
        plt.axvline(x=t2 / 24, color='red', linestyle=':', linewidth=1.5,
                    label=f'Node2 t={t2/24:.1f}d')

    plt.xlabel('Time (days)', fontsize=12)
    plt.ylabel('Displacement (mm)', fontsize=12)
    plt.title('Three-stage Modeling of Surface Displacement', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, '图6 三段拟合曲线.png')
    plt.savefig(path, dpi=300)
    plt.show()
    print(f"[图6] 三段拟合曲线 → {path}")


def plot_residuals(stages, save_dir):
    """图7：各阶段残差"""
    plt.figure(figsize=(12, 10))
    colors = {'I': 'b', 'II': 'g', 'III': 'r'}
    labels = {'I': 'Slow Constant', 'II': 'Accelerating', 'III': 'Rapid'}

    for i, label in enumerate(['I', 'II', 'III'], 1):
        plt.subplot(3, 1, i)
        s = stages[label]
        res = s['d'] - s['d_pred']
        plt.plot(s['t'] / 24, res, f'{colors[label]}.', markersize=2, alpha=0.5)
        plt.axhline(0, color='k', linestyle='-')
        plt.ylabel('Residual (mm)', fontsize=10)
        # 使用 text 显示 R² 和 RMSE，避免字体警告
        plt.text(0.02, 0.95, f'$R^2$={s["r2"]:.4f}, RMSE={s["rmse"]:.3f}mm',
                 transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.title(f'Stage {label} ({labels[label]}) Residuals')
        plt.grid(True, alpha=0.3)

    plt.xlabel('Time (days)', fontsize=12)
    plt.tight_layout()
    path = os.path.join(save_dir, '图7 各阶段残差.png')
    plt.savefig(path, dpi=300)
    plt.show()
    print(f"[图7] 各阶段残差 → {path}")


def main():
    """独立运行：检测节点 → 三段建模 → 画图6+图7"""
    print("=" * 55)
    print("Three-stage Deformation Modeling")
    print("=" * 55)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 读取清洗后数据
    d, t, dt = _load_clean_data()

    # 获取节点
    from detect_nodes import sliding_window_detect
    idx1, t1, idx2, t2 = sliding_window_detect()

    if idx1 is None or idx2 is None:
        print("\n[WARN] Sliding window failed, trying MAD+persistence...")
        from detect_nodes import mad_persistence_detect
        idx1, t1, idx2, t2 = mad_persistence_detect()

    if idx1 is None or idx2 is None:
        print("\n[ERROR] Both methods failed to identify nodes")
        return

    print(f"\nUsing nodes: Node1=#{idx1} (t={t1/24:.2f}d), Node2=#{idx2} (t={t2/24:.2f}d)")
    stages = modeling_with_nodes(idx1, idx2)

    # 汇总表
    print("\n" + "=" * 55)
    print("Stage Summary")
    print("-" * 55)
    print(f"{'Stage':<8} {'Model':<12} {'R²':>8} {'RMSE(mm)':>10} {'Mean Vel(mm/h)':>16}")
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
