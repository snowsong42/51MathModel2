# Q5 探索性数据分析 (EDA) 摘要报告

生成时间: 2026-05-03 18:44

---

## 1. 数据概览

- 记录总数: 9449
- 时间范围: 2024-01-01 16:50:00 ~ 2024-03-07 07:30:00
- 采样间隔: 10分钟/步
- 原始变量数: 6 (基础传感器变量)
- 特征工程后变量数: 36
- 阶段边界: 缓慢→加速 = 130步, 加速→快速 = 588步

## 2. 缺失值分析

### 各变量缺失比例

| 变量 | 缺失比例 | 非空数量 |
|------|---------|---------|
| PorePressure | 0.0000 | 9450 |
| Infiltration | 0.0000 | 9450 |
| Rainfall | 0.0000 | 9450 |
| Microseismic | 0.0000 | 9450 |
| BlastDist | 0.9950 | 47 |
| BlastCharge | 0.9950 | 47 |
| Displacement | 0.0000 | 9450 |

**关键发现：**
- 爆破相关变量（BlastDist, BlastCharge）缺失比例很高，属于偶发事件变量，仅在爆破时刻非空
- 连续传感器变量（PorePressure, Infiltration 等）缺失较少，可前向填充
- 缺失矩阵图见 `missing_pattern.png`

## 3. 变量分类

```
continuous: Infiltration, PorePressure
rainfall: Rainfall, Rainfall_cum24, Rainfall_lag, Rainfall_lag78, Time_since_rain
microseismic: Microseismic, Microseismic_lag, Microseismic_lag24, Microseismic_roll6
blast: BlastCharge, BlastCharge_lag, BlastCharge_lag60, BlastDist, BlastDist_lag, BlastDist_lag60, Blast_Energy, Blast_PPV, Time_since_blast
target: Delta_D, Disp_cum24, Displacement

```

### 按物理意义分类

- **continuous**: Infiltration, PorePressure
- **rainfall**: Rainfall, Rainfall_cum24, Rainfall_lag, Rainfall_lag78, Time_since_rain
- **microseismic**: Microseismic, Microseismic_lag, Microseismic_lag24, Microseismic_roll6
- **blast**: BlastCharge, BlastCharge_lag, BlastCharge_lag60, BlastDist, BlastDist_lag, BlastDist_lag60, Blast_Energy, Blast_PPV, Time_since_blast
- **target**: Delta_D, Disp_cum24, Displacement

## 4. 基本统计表

详细统计表见 `variable_statistics.csv`

（请查看 variable_statistics.csv 获取完整统计表）

## 5. 各变量类型推断与主要发现

### 5.1 连续传感器变量

- **Infiltration**: 连续型传感器读数
  - 均值=0.5850, 标准差=0.0696
  - 缺失比例=0.0000
  - 分布形态: 偏度=0.001，近似对称
  - 存在异常值: False
- **PorePressure**: 连续型传感器读数
  - 均值=38.1722, 标准差=10.4800
  - 缺失比例=0.0000
  - 分布形态: 偏度=-0.136，近似对称
  - 存在异常值: False

### 5.2 降雨变量

- **Rainfall**: 事件型/连续混合变量，降雨时刻占比 1.76%
  - 最大值=20.1000, 均值=0.0723
  - 多窗口累积分析见 `rainfall_cumulative.png`
  - 有效降雨采用衰减常数 0.85 建模
- **CCF 分析**: 降雨滞后 **1步（约0.2小时）**时与位移增量相关性最大 (r=0.065)

### 5.3 微震变量

- **Microseismic**: 计数型变量，事件时刻占比 80.77%
  - 单步最大事件数=27
  - 累积事件数与位移呈正相关趋势，见 `cum_microseismic_vs_displacement.png`

### 5.4 爆破变量

- **BlastDist/BlastCharge**: 偶发事件变量，共 47 次爆破
  - 爆破距离范围: 0.6 ~ 5.8
  - 单段药量范围: 1.0 ~ 11.9
  - 爆破影响持续时间约 **4-8小时（24-48步）**，建议取 τ=50 步作为衰减常数
  - 响应曲线分析见 `blast_response_curves.png`

### 5.5 目标变量（位移/Delta_D）

- **Delta_D**（位移增量）: 均值=0.376480, 标准差=1.000593
  - 三阶段速度均值: 阶段1(缓慢)=0.006454, 阶段2(加速)=0.029487, 阶段3(快速)=0.399843
  - 位移速度时序图见 `displacement_velocity_phases.png`

## 6. 时滞参数建议

基于 CCF 图和爆破响应叠加图：

- **降雨→位移**: 降雨滞后 **1步（约0.2小时）**时与位移增量相关性最大 (r=0.065)
- **降雨特征工程**: 建议采用 12h (72步) 和 24h (144步) 滑动累积量作为特征
- **有效降雨衰减常数**: τ=0.85 较合适
- **爆破特征**: 建议使用 `Blast_Energy = q/d²` 作为爆破强度指标
- **爆破衰减**: 建议 τ=50 步（约8.3小时）的指数衰减，或使用 Time_since_blast 作为特征
- **微震特征**: 建议使用 6h (36步) 和 12h (72步) 滑动窗口计数

## 7. 初步特征工程方向

| 变量类别 | 建议特征 | 滑动窗口 | 备注 |
|---------|---------|---------|------|
| 孔隙水压力 | 原始值 + 差分 + 24h滑动平均 | 18/36/72/144步 | 孔压变化(Pore_Diff)已证明有效 |
| 入渗系数 | 原始值 + 24h累积滑动和 | 144步 | 与孔压的交叉特征 (Pore_Infilt) |
| 降雨 | 多窗口累积 + 有效降雨 + 距上次降雨时间 | 18/36/72/144步 | CCF显示最佳滞后约 xx 步 |
| 微震 | 滑动窗口计数 + 累积事件数 | 36/72/144步 | 累积事件数作为整体趋势指标 |
| 爆破 | Blast_Energy + 指数衰减 + 距上次爆破时间 | - | Time_since_blast 已实现 |
| 交互特征 | Pore_Rain, Pore_Infilt | - | 已实现 |
| 时间特征 | 时间编码 (Hour, Day_sin/cos) | - | 已实现 |

## 8. 数据质量问题与潜在传感器故障

- 爆破变量缺失比例极高（~99%+），属于正常现象（偶发事件），但需要特殊处理
- 部分传感器可能在某些时段存在漂移或异常尖峰（见分布图的长尾现象）
- 降雨序列存在大量零值，可能导致 CCF 计算中相关性偏低
- 微震事件集中在特定时段，事件间隔分布呈长尾特征
- 位移在第三阶段（快速变形）波动增大，可能反映传感器在高速变形时的测量噪声
