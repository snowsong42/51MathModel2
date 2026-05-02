"""
参数配置文件 —— 所有可调参数集中管理
对标 Q1 filter.m 的 "参数设置" 块
"""

# ===== 文件路径 =====
SCRIPT_DIR = None  # 运行时由各脚本动态设置
RAW_DATA = "Attachment 2.xlsx"
CLEAN_DATA = "Filtered 2.xlsx"

# ===== 时间参数 =====
DT = 10 / 60          # 采样间隔 10分钟 = 1/6 小时

# ===== 预处理参数 =====
MEDFILT_KERNEL = 31   # 中值滤波窗口（奇数）

# ===== 滑动窗口检测参数（velocity.py 简化方案） =====
SLIDING_WINDOW = 144      # 窗口长度 144点 (24小时)
MIN_GAP = 500             # 两节点最小间隔点数
THRESH_SLOW = 0.15        # 匀速阶段速度上界 (mm/h)
THRESH_FAST = 2.0         # 快速阶段速度下界 (mm/h)
PERSIST_FAST = 50         # 快速阈值持续检验点数
PERSIST_SLOW = 100        # 低速阈值持续检验点数

# ===== MAD+持久性检测参数（stage_recognition.py 文档算法） =====
MAD_WINDOW = 72           # 窗口 72点 (12小时)
MAD_HOLD = 36             # 持久性检验窗口 (6小时)
C_JUMP = 5.0              # 跳变指标阈值
C_TRANS = 0.6             # 阶跃量阈值
THETA_BACK = 0.5          # 回退阈值
SPEED_THRESH_LOW = 0.6    # 低速阈值 (mm/h)
SPEED_THRESH_HIGH = 5.0   # 高速阈值 (mm/h)

# ===== 速度平滑参数 =====
SAVGOL_WINDOW = 151      # Savgol滤波窗口
SAVGOL_POLY = 2           # 多项式阶数
