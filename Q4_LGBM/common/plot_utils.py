"""绘图工具（Q4_LGBM 版）"""

import matplotlib.pyplot as plt

# 阶段颜色和标签（全局共享）
COLORS = {0: 'green', 1: 'orange', 2: 'red'}
PHASE_LABELS = {0: '缓慢变形', 1: '加速变形', 2: '快速变形'}


def setup_zh():
    """设置中文字体"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False


def save_dir(path):
    """兼容旧接口（空函数）"""
    pass


def phase_name(ph):
    """获取阶段中文名"""
    return PHASE_LABELS.get(ph, f'Phase {ph}')
