"""中文绘图工具"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def setup_zh():
    """配置中文字体"""
    plt.rcParams['axes.unicode_minus'] = False
    for f in ['SimHei','Microsoft YaHei','WenQuanYi Micro Hei','DejaVu Sans']:
        try:
            plt.rcParams['font.sans-serif'] = [f]
            fig = plt.figure(); fig.text(.5,.5,'测试'); plt.close(fig)
            return
        except: pass

def save_dir(base='Q5/结果与使用指南'):
    """创建并返回图表保存目录"""
    d = os.path.join(base, '图表')
    os.makedirs(d, exist_ok=True)
    return d

def phase_name(pid):
    return {0:'缓慢变形',1:'加速变形',2:'快速变形'}.get(pid, f'Phase{pid}')

__all__ = ['setup_zh','save_dir','phase_name']
