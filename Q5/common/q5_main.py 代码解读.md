# `q5_main.py` 代码解读

## 概述

`q5_main.py` 是 Q5 问题的一体化建模脚本，整合了 **Q5.1（最优变量组合搜索）** 与 **Q5.2（消融实验/分阶段分析）** 两大任务。该脚本通过逐变量族剔除的方式，评估降雨量、孔隙水压力、微震事件数、干湿入渗系数、爆破参数对边坡位移预测的贡献，并输出四张高质量汉化图表。

**运行方式**：
```bash
python q5_main.py
```

**输出位置**：`Q5/结果与使用指南/图表/` 目录下生成 `.png` 图表。

---

## 模块结构

脚本依赖两个 `common` 工具模块：

| 模块 | 功能 |
|------|------|
| `common/data_utils.py` | 数据加载、清洗、特征工程、三阶段划分 |
| `common/plot_utils.py` | 中文字体配置、图表输出目录管理 |

---

## 逐段解析

### 1. 环境准备与导入

```python
import warnings, os, sys
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
```

- 忽略所有警告，避免 LightGBM 的冗余告警干扰终端输出。
- 将 `q5_main.py` 所在目录（即 `Q5/`）加入系统路径，使 `from common.data_utils import ...` 能正确解析。

```python
matplotlib.use('Agg')
```
设置 Matplotlib 为非交互式后端（不弹窗），专用于服务器或脚本批量出图。

```python
from common.data_utils import load_pipeline, get_all_features
from common.plot_utils import setup_zh, save_dir, phase_name
setup_zh()
OUT_DIR = save_dir()
```

- `load_pipeline()`：一站式数据管线。
- `get_all_features()`：从 DataFrame 中提取全部特征列名。
- `setup_zh()`：自动探测系统中可用的中文字体并配置。
- `save_dir()`：创建 `Q5/结果与使用指南/图表/` 目录。

---

### 2. 数据加载

```python
df, base_vars, (b1, b2) = load_pipeline()
all_feats = get_all_features(df, base_vars)
df = df.dropna(subset=all_feats + ['Delta_D']).reset_index(drop=True)
n_phase = [sum(df['Phase']==i) for i in range(3)]
```

`load_pipeline()` 内部依次完成：
1. **文件查找**（`find_data`）：在上级目录中搜索 `.xlsx` 文件。
2. **列名映射**（`map_columns`）：中文列名 → 英文标准化名称（`降雨`→`Rainfall`，`微震`→`Microseismic` 等）。
3. **数据清洗**（`clean`）：时间排序、缺失值填充（爆破类填 0，其他前向填充）。
4. **特征工程**（`feat_engineer`）：
   - 计算位移增量 `Delta_D`。
   - 为每个基础变量搜索最优滞后步长（0~288 步，步长 6），生成 `{feat}_lag` 滞后特征。
   - 衍生特征：`Pore_Diff`（孔压差分）、`Rainfall_cum24` / `Infiltration_cum24`（24 小时累积）、`Microseismic_roll6`（6 步滑动和）、交叉特征 `Pore_Rain` / `Pore_Infilt`、爆破 PPV 与能量、距上次爆破/降雨时间、时间周期特征（`Day_sin`, `Day_cos`）。
5. **三阶段划分**（`divide_phases` + `label_phase`）：
   - 基于位移速度的 50 步滑动平均。
   - 连续 10 步平均速度 > 0.02 → 进入**加速变形阶段**（b1）。
   - 连续 10 步平均速度 > 0.10 → 进入**快速变形阶段**（b2）。
   - 生成 `Phase` 列：`0`=缓慢，`1`=加速，`2`=快速。

最终 `dropna` 确保所有特征和目标值均无缺失，`reset_index` 使索引连续。

---

### 3. LightGBM 分阶段训练函数

```python
def train_lgb(df, feats, target='Delta_D'):
    y_pred = np.zeros(len(df))
    phase_metrics = {}
    for pid in range(3):
        mask = df['Phase'] == pid
        if mask.sum() < 30: continue
        X, y = df.loc[mask, feats].values, df.loc[mask, target].values
        model = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.01,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1)
        model.fit(X, y)
        yp = model.predict(X)
        phase_metrics[pid] = {
            'rmse': np.sqrt(mean_squared_error(y, yp)),
            'mae': mean_absolute_error(y, yp),
            'r2': r2_score(y, yp), 'n': int(mask.sum())}
        y_pred[mask] = yp
    return y_pred, phase_metrics
```

**核心设计理念**：
- 对三个变形阶段**分别独立训练** LightGBM 回归模型（同一套超参数）。
- 每个阶段仅使用属于该阶段的数据（`mask` 过滤）。
- 容错机制：若某阶段数据量 < 30 行则跳过。
- 返回值包含：全量预测值 `y_pred` 和各阶段的性能指标字典 `phase_metrics`。

**超参数说明**：
| 参数 | 值 | 含义 |
|------|----|------|
| `n_estimators` | 500 | 500 轮迭代（学习率低，需足够轮数） |
| `learning_rate` | 0.01 | 低学习率，防止过拟合 |
| `num_leaves` | 31 | 每棵树最多 31 个叶节点 |
| `subsample` | 0.8 | 每轮随机采样 80% 数据训练 |
| `colsample_bytree` | 0.8 | 每棵树随机选择 80% 特征 |
| `random_state` | 42 | 固定随机种子（可复现） |

---

### 4. Q5.1：全模型基线

```python
y_full, pm_full = train_lgb(df, all_feats)
rmse_f = np.sqrt(mean_squared_error(df['Delta_D'], y_full))
r2_f   = r2_score(df['Delta_D'], y_full)
```

使用**全部特征**训练模型，得到全模型基线指标 `rmse_f` 和 `r2_f`，作为后续消融实验的参考标准。

---

### 5. Q5.1：逐变量族剔除消融实验

```python
ablation_families = {
    '降雨量':        ['Rainfall', 'Rain_', 'rain', 'Time_since_rain'],
    '孔隙水压力':    ['PorePressure', 'Pore', 'pore'],
    '微震事件数':    ['Microseismic', 'microseismic', 'Microseismic_roll6'],
    '干湿入渗系数':  ['Infiltration', 'infiltration', 'Infilt'],
    '爆破':          ['Blast', 'blast', 'Time_since_blast'],
}
```

**关键设计——变量族**：由于特征工程会从原始变量衍生大量新特征（滞后特征、交叉特征、累积特征等），直接按原始列名剔除会漏掉衍生特征。此处定义了五个"变量族"，每个族包含一组关键词，可彻底剔除该原始变量的所有衍生特征。

例如，剔除"爆破"变量族会同时移除：
- `BlastDist`、`BlastDist_lag`
- `BlastCharge`、`BlastCharge_lag`
- `Blast_PPV`、`Blast_Energy`、`Time_since_blast`

```python
def exclude_features(feats, keywords):
    return [f for f in feats if not any(kw in f for kw in keywords)]
```

`exclude_features` 过滤掉特征名中包含任一关键词的列，实现干净彻底的变量族剔除。

**消融循环**：
```python
for cn_name, kws in ablation_families.items():
    excl = exclude_features(all_feats, kws)
    y_abl, pm_abl = train_lgb(df, excl)
    rmse_a = np.sqrt(mean_squared_error(df['Delta_D'], y_abl))
    r2_a   = r2_score(df['Delta_D'], y_abl)
    delta_r2 = r2_f - r2_a
```

对每个变量族：
1. 从特征列表中剔除该族所有特征。
2. 用剩余特征重新训练模型。
3. 计算 RMSE、R² 相对于全模型的差值（ΔRMSE、ΔR²）。

**变量贡献度判断逻辑**：
- **ΔR² 越大** → 剔除后 R² 下降越多 → **该变量越重要**。
- **最优组合** = 剔除 **ΔR² 最小** 的变量族（剔除后模型几乎不受影响，该变量可省略）。
- **最差组合** = 剔除 **ΔR² 最大** 的变量族（该变量对预测最为关键）。

---

### 6. Q5.2：可视化输出（4 张图表）

#### 图 1：`Q5_1_变量组合对比.png`（双面板）

- **左面板**：柱状图展示各变量族剔除后的 **ΔRMSE**（正值表示误差增加，即模型变差）。
- **右面板**：水平条形图展示变量重要度排序（**ΔR²**），红色 = 重要（ΔR² 高于中位数），绿色 = 次要。

#### 图 2：`Q5_2_分阶段热力图.png`

- 热力图矩阵：行 = 变量族，列 = 三个阶段（缓慢/加速/快速）。
- 每个单元格显示该变量族在该阶段的 **ΔRMSE**。
- 颜色映射：红色 → 该变量在该阶段影响大，绿色 → 影响小。

#### 图 3：`Q5_2_分阶段时序对比.png`

- 三行子图（缓慢/加速/快速阶段）。
- 每行绘制三条曲线：真实值（黑色）、全模型预测（蓝色）、剔除最重要变量后的预测（红色虚线）。
- 直观对比**最重要变量是否存在**对预测效果的影响。

#### 图 4：`Q5_汇总表.png`

- 表格形式汇总消融实验全部数据：变量名、特征数、RMSE、ΔRMSE、RMSE 增幅百分比、R²、ΔR²。
- 包含全模型基线行，便于对比。

---

### 7. 终端结论输出

脚本在终端打印完整的分析结论：

1. **Q5.1 最优变量组合**：全模型指标 + 最优/最差变量名。
2. **Q5.2 变量重要度排序**：按 ΔR² 降序排列。
3. **阶段平均速度**：基于三阶段位移数据计算实际形变速率（mm/h）。
4. **预警阈值建议**：
   - 缓慢变形：阈值 > 2 倍平均速度（正常监测）。
   - 加速变形：阈值 > 0.8 倍平均速度（黄色预警）。
   - 快速变形：阈值 > 0.5 倍平均速度（红色预警）。
   - 原理：边坡失稳前位移呈指数加速，阈值设为递减倍数以平衡灵敏度与误报率。

---

## 数据流示意图

```
附件5.xlsx
    │
    ▼
load_pipeline()
    ├── find_data()           # 自动定位 xlsx 文件
    ├── map_columns()         # 中文列名 → 英文标准名
    ├── clean()               # 排序 + 缺失值填充
    ├── feat_engineer()       # 衍生特征 + 最优滞后搜索
    ├── divide_phases()       # 基于速度的三阶段划分
    └── label_phase()         # 写入 Phase 标签
    │
    ▼
df (含全部特征 + Phase + Delta_D)
    │
    ├── get_all_features()    # 提取特征列名清单
    │
    ▼
Q5.1 全模型 → train_lgb(df, all_feats) → 基线 RMSE/R²
    │
    ▼
逐变量族剔除循环
    ├── exclude_features()    # 移除该族所有衍生特征
    ├── train_lgb(df, excl)   # 重新训练
    └── 记录 ΔRMSE, ΔR²
    │
    ▼
Q5.2 绘图输出
    ├── 图1: 变量组合对比（柱状 + 排序）
    ├── 图2: 分阶段热力图
    ├── 图3: 分阶段时序对比
    └── 图4: 汇总表
    │
    ▼
终端打印结论
```

---

## 与 common 模块的关系

| 函数/变量 | 来源模块 | 说明 |
|-----------|----------|------|
| `load_pipeline()` | `data_utils` | 端到端数据管线 |
| `get_all_features()` | `data_utils` | 动态特征列表提取 |
| `setup_zh()` | `plot_utils` | 中文字体配置 |
| `save_dir()` | `plot_utils` | 图表输出目录 |
| `phase_name()` | `plot_utils` | 阶段编号 → 中文名 |

`common/` 目录对 `q5_main.py` 提供了数据加载、特征工程、绘图配置的封装，使主脚本聚焦于建模逻辑与消融实验流程。

---

## 关键设计要点总结

1. **变量族剔除**：不是简单按原始列名剔除，而是通过关键词匹配移除所有衍生特征，确保消融实验的干净性。
2. **分阶段独立建模**：三个变形阶段分别训练三个 LightGBM，适配不同阶段的非线性模式差异。
3. **自动化程度高**：`load_pipeline()` 自动查找数据文件、自动特征工程、自动阶段划分，脚本只需一行调用。
4. **可复现性**：固定随机种子 `random_state=42` 确保每次运行结果一致。
5. **终端输出与图片双通道**：既在终端打印结构化的文本结论，又生成四张高质量矢量图供报告使用。
