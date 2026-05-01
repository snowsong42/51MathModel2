"""
Q2 全流程主入口（对标 Q1 的 filter.m + correct_and_test.m 顺序执行）
一键运行：预处理 -> 节点检测 -> 三段建模 -> 结果可视化

使用：
  uv run python Q2/run_pipeline.py              # 全流程
  uv run python Q2/run_pipeline.py --skip-eda   # 跳过EDA
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def run_preprocess():
    """第一步：数据预处理，输出 Filtered 2.xlsx"""
    print("\n" + "=" * 55)
    print("Step 1: Data Preprocessing")
    print("=" * 55)
    import preprocess
    preprocess.main()


def run_explore():
    """可选：原始数据EDA"""
    print("\n" + "=" * 55)
    print("Optional: Exploratory Data Analysis")
    print("=" * 55)
    import explore_raw
    explore_raw.main()


def run_detect():
    """第二步：阶段节点检测"""
    print("\n" + "=" * 55)
    print("Step 2: Stage Transition Detection")
    print("=" * 55)
    from detect_nodes import sliding_window_detect, mad_persistence_detect

    idx1, t1, idx2, t2 = sliding_window_detect()
    if idx1 is None or idx2 is None:
        print("Sliding window failed, trying MAD+persistence...")
        idx1, t1, idx2, t2 = mad_persistence_detect()
    return idx1, t1, idx2, t2


def run_modeling(idx1, t1, idx2, t2):
    """第三步：三段式建模"""
    print("\n" + "=" * 55)
    print("Step 3: Three-stage Deformation Modeling")
    print("=" * 55)
    from modeling import modeling_with_nodes

    stages = modeling_with_nodes(idx1, idx2)

    # 汇总表
    print("\n" + "=" * 55)
    print("Stage Summary")
    print("-" * 55)
    print(f"{'Stage':<8} {'Model':<12} {'R^2':>8} {'RMSE(mm)':>10} {'Mean Vel(mm/h)':>16}")
    print("-" * 55)
    for label in ["I", "II", "III"]:
        s = stages[label]
        print(f"{label:<8} {s['model']:<12} {s['r2']:>8.4f} {s['rmse']:>10.3f} {s['v_mean']:>16.4f}")
    print("=" * 55)

    return stages


def run_plotting():
    """第四步：结果可视化"""
    print("\n" + "=" * 55)
    print("Step 4: Result Visualization")
    print("=" * 55)
    import plotting
    plotting.plot_all()


def main():
    print("=" * 55)
    print("     Q2 Three-stage Deformation Analysis")
    print("=" * 55)

    # 检查参数
    skip_eda = "--skip-eda" in sys.argv

    # 第一步：预处理
    run_preprocess()

    # 可选：EDA
    if not skip_eda:
        run_explore()
    else:
        print("\n(Skip EDA)")

    # 第二步：检测节点
    idx1, t1, idx2, t2 = run_detect()
    if idx1 is None or idx2 is None:
        print("\n[ERROR] Both methods failed to identify nodes, aborting")
        return

    # 第三步：建模
    stages = run_modeling(idx1, t1, idx2, t2)

    # 第四步：画图
    run_plotting()

    print("\n" + "=" * 55)
    print("[OK] Q2 pipeline completed!")
    print(f"  Clean data: {config.CLEAN_DATA}")
    print(f"  Node1: #{idx1} (t={t1/24:.2f}d)")
    print(f"  Node2: #{idx2} (t={t2/24:.2f}d)")
    print("  Figures: stage_fitting_curve.png, stage_residuals.png, velocity_transition.png")
    print("=" * 55)


if __name__ == "__main__":
    main()
