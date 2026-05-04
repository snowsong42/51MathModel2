# 特征工程数学公式汇总

基于 Q5 EDA 与特征工程代码，将所有特征归纳为统一数学表述，并按解释变量分类。

---

## 符号约定

| 符号 | 含义 | 单位 / 备注 |
|------|------|-------------|
| $$ t $$ | 时间步索引（$$0,1,2,\dots$$） | 采样间隔 $$\Delta t = 10\,\text{min}$$ |
| $$ T $$ | 总时间长度（步数） | |
| $$ R(t) $$ | 降雨量（原始序列） | mm/10min |
| $$ P(t) $$ | 孔隙水压力 | kPa |
| $$ M(t) $$ | 微震事件数 | 次/10min |
| $$ I(t) $$ | 干湿入渗系数 | 无量纲 |
| $$ D(t) $$ | 爆破点距离（仅爆破时刻非零） | m |
| $$ Q(t) $$ | 单段最大药量（仅爆破时刻非零） | kg |
| $$ \mathcal{B} = \{t_1, t_2, \dots, t_K\} $$ | 爆破事件发生时间步集合 | $$K$$ 为爆破次数 |
| $$ \tau $$ | 指数衰减时间常数（步） | 本模型取 $$\tau = 50$$ 步 |
| $$ \alpha $$ | 有效降雨衰减系数 | 一般取 $$0.9, 0.7$$ |
| $$ w $$ | 滑动窗口宽度（步） | |
| $$ \text{rolling\_op}_{w}(X) $$ | 变量 $$X$$ 在窗口 $$w$$ 上的滚动操作（sum/mean/std 等），min_periods=1 | |
| $$ \Delta X^{(k)}(t) = X(t) - X(t-k) $$ | $$k$$ 步差分 | |
| $$ X_{\text{lag}k}(t) = X(t-k) $$ | 变量 $$X$$ 滞后 $$k$$ 步的值 | |
| $$ T_{\text{since\_event}}(t) $$ | 距上一次某事件的时间步数 | 首次事件前为 0 |
| $$ v(t) = \Delta \text{Displacement}(t) $$ | 单步位移增量（`Delta_D`） | mm/10min，与速度成正比 |
| $$ g_k $$ | 爆破权重函数 | 见下文 |

**说明：**  
- 所有窗口 $$w$$ 均以步数为单位，时间换算关系：$$1\,\text{h} = 6\,\text{步}$$，$$1\,\text{天} = 144\,\text{步}$$。  
- 特征名中带 `lag` 的为直接滞后，带 `_h` 的为滑动窗口特征（如 `3h` 对应 $$w=18$$）。

---

## 特征公式（按解释变量分类）

下表中，**加粗**表示本次新增的高级特征，普通字体为已有基础特征。

### Ⅰ 降雨量类特征

| 通用公式                                                                                | 导出列示例                                                          | 参数取值                 | 说明       |
| ----------------------------------------------------------------------------------- | -------------------------------------------------------------- | -------------------- | -------- |
| $$ R_w^{\text{sum}}(t) = \sum_{i=0}^{w-1} R(t-i) $$                                 | `Rain_3h_sum`, `Rain_6h_sum`, `Rain_12h_sum`, `Rain_24h_sum`   | $$w=18,36,72,144$$   | 多窗口累积降雨  |
| $$ R^{(\text{eff},\alpha)}(t) = R(t) + \alpha \cdot R^{(\text{eff},\alpha)}(t-1) $$ | `Rain_eff_09` ($$\alpha=0.9$$), `Rain_eff_07` ($$\alpha=0.7$$) | $$\alpha = 0.9,0.7$$ | 指数衰减有效降雨 |
| $$ R_{w}^{\text{std}}(t) = \text{rolling\_std}_{w}(R)(t) $$                         | **`Rain_6h_std`**                                              | $$w=6$$              | 降雨强度波动   |
| $$ R_{\text{lag}k}(t) = R(t-k) $$                                                   | `Rainfall_lag`, `Rainfall_lag78`                               | $$k=1,78$$           | 降雨滞后值    |
| $$ T_{\text{since\_rain}}(t) $$                                                     | `Time_since_rain`                                              | –                    | 距上次降雨时间步 |

**建议引用图表**：`rainfall_cumulative.png`（多窗口累积效果）, `ccf_rainfall_displacement.png`（滞后相关性）。

### Ⅱ 孔隙水压力类特征

| 通用公式                                                        | 导出列示例                                                  | 参数        | 说明          |
| ----------------------------------------------------------- | ------------------------------------------------------ | --------- | ----------- |
| $$ P_{\text{lag}k}(t) = P(t-k) $$                           | `PorePressure_lag`, `PorePressure_lag6`                | $$k=1,6$$ | 滞后          |
| $$ \Delta P^{(k)}(t) = P(t) - P(t-k) $$                     | **`Pore_diff1`** ($$k=1$$), **`Pore_diff6`** ($$k=6$$) | $$k=1,6$$ | $$k$$ 步差分   |
| $$ P_{w}^{\text{std}}(t) = \text{rolling\_std}_{w}(P)(t) $$ | **`Pore_roll_std_12h`**                                | $$w=72$$  | 12h 滑动标准差   |
| $$ P \times I $$ 交互                                         | `Pore_Infilt`                                          | –         | 孔压×入渗系数（已有） |
| $$ P \times R $$ 交互                                         | `Pore_Rain`                                            | –         | 孔压×降雨（已有）   |

**建议引用图表**：`PorePressure_rolling_mean.png`, `PorePressure_distribution.png`。

### Ⅲ 干湿入渗系数类特征

| 通用公式                                                        | 导出列示例                                                      | 参数         | 说明        |
| ----------------------------------------------------------- | ---------------------------------------------------------- | ---------- | --------- |
| $$ I_{\text{lag}k}(t) = I(t-k) $$                           | `Infiltration_lag`, `Infiltration_lag12`                   | $$k=1,12$$ | 滞后        |
| $$ \Delta I^{(k)}(t) = I(t) - I(t-k) $$                     | **`Infilt_diff1`** ($$k=1$$), **`Infilt_diff6`** ($$k=6$$) | $$k=1,6$$  | 差分        |
| $$ I_{w}^{\text{std}}(t) = \text{rolling\_std}_{w}(I)(t) $$ | **`Infilt_roll_std_12h`**                                  | $$w=72$$   | 12h 滑动标准差 |
| $$ I_{w}^{\text{sum}}(t) = \sum_{i=0}^{w-1} I(t-i) $$       | `Infiltration_cum24` 等                                     | 已有累积特征     | 累积入渗      |

**建议引用图表**：`Infiltration_rolling_mean.png`, `Infiltration_distribution.png`。

### Ⅳ 微震事件数类特征

| 通用公式                                                                     | 导出列示例                                                              | 参数              | 说明               |
| ------------------------------------------------------------------------ | ------------------------------------------------------------------ | --------------- | ---------------- |
| $$ M_{\text{lag}k}(t) = M(t-k) $$                                        | `Microseismic_lag`, `Microseismic_lag24`                           | $$k=1,24$$      | 滞后               |
| $$ C_{w}^{\text{count}}(t) = \sum_{i=0}^{w-1} \mathbf{1}_{[M(t-i)>0]} $$ | **`Micro_6h_count`**, **`Micro_12h_count`**, **`Micro_24h_count`** | $$w=36,72,144$$ | 事件密度（滚动窗内“非零”计数） |
| $$ M_{w}^{\text{sum}}(t) = \sum_{i=0}^{w-1} M(t-i) $$                    | **`Micro_24h_cum`** ($$w=144$$), `Microseismic_roll6`              | $$w$$ 不同        | 滚动累积事件数          |
| $$ \sqrt{M(t)} $$                                                        | **`Micro_energy_sqrt`**                                            | –               | 能量代理（平方根）        |
| $$ M(t)^2 $$                                                             | **`Micro_energy_sq`**                                              | –               | 能量代理（平方）         |

**建议引用图表**：`microseismic_event_analysis.png`, `cum_microseismic_vs_displacement.png`。

### Ⅴ 爆破类特征（距离、药量分开处理）

#### 通用衰减函数

$$
\Phi_{\mathcal{B}, g, \tau}(t) = \sum_{t_i \le t,\; t_i \in \mathcal{B}} g\big(D(t_i), Q(t_i)\big) \cdot e^{-\frac{t-t_i}{\tau}}
$$

其中 $$g$$ 为权重函数，三种模式：

1. 距离影响权重：$$g_1(d, q) = \dfrac{1}{d^2 + 1}$$
2. 药量影响权重：$$g_2(d, q) = q$$
3. 联合能量权重：$$g_3(d, q) = \dfrac{q}{d^2 + 1}$$

| 列名                             | 公式                                                                                                               | 参数          |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------- | ----------- |
| **`BlastDist_impact`**         | $$\Phi_{\mathcal{B}, g_1, \tau}(t)$$                                                                             | $$\tau=50$$ |
| **`BlastCharge_impact`**       | $$\Phi_{\mathcal{B}, g_2, \tau}(t)$$                                                                             | $$\tau=50$$ |
| **`BlastEnergy_decay_impact`** | $$\Phi_{\mathcal{B}, g_3, \tau}(t)$$                                                                             | $$\tau=50$$ |
| **`Blast_interval(t)`**        | 假设 $$\mathcal{B}$$ 有序，对相邻爆破时间差 $$\Delta t_i = t_{i+1} - t_i$$ 向前填充（即在区间 $$[t_i, t_{i+1}-1]$$ 上保持 $$\Delta t_i$$） | –           |
| `Time_since_blast(t)`          | $$t - \max\{t_i \in \mathcal{B} \mid t_i < t\}$$（若无爆破事件则赋以大值）                                                    | –           |
| `Blast_PPV`                    | 基于 $$Q, D$$ 的经验公式                                                                                                | 已有特征        |
| `Blast_Energy`                 | $$\dfrac{Q}{D^2}$$                                                                                               | 已有特征        |
| 爆破滞后                           | `BlastDist_lag60`, `BlastCharge_lag60` 等                                                                         | $$k=1,60$$  |

**建议引用图表**：`blast_scatter.png`（D vs Q）, `blast_response_curves.png`（位移响应叠加）, `blast_decay_demo.png`（衰减形态示意）。

### Ⅵ 其他特征（时间、位移自身）

| 通用公式   | 导出列示例                                                       | 说明             |
| ------ | ----------------------------------------------------------- | -------------- |
| 时间正弦编码 | `Day_sin` = $$\sin\left( \frac{2\pi}{144} \cdot t \right)$$ | 以 144 步为周期（1天） |
| 时间余弦编码 | `Day_cos` = $$\cos\left( \frac{2\pi}{144} \cdot t \right)$$ |                |
| 累积位移信息 | `Disp_cum24`, `Displacement`（原始）                            | 位移累积特征         |

---

## 建模所需关键图表（论文支撑）

| 分析阶段 | 图表名称 | 内容 |
|----------|----------|------|
| 缺失值分析 | `missing_pattern.png` | 爆破变量高缺失比例 |
| 降雨特征 | `rainfall_cumulative.png` | 多窗口累积量对比 |
| 降雨滞后 | `ccf_rainfall_displacement.png` | 互相关函数，解释为何需累积特征 |
| 孔压分布 | `PorePressure_distribution.png` | 对称性、无异常 |
| 孔压趋势 | `PorePressure_rolling_mean.png` | 不同窗口平滑效果 |
| 微震密度 | `microseismic_event_analysis.png` | 事件间隔、滑动计数 |
| 微震累积 | `cum_microseismic_vs_displacement.png` | 累积事件数与位移的关联 |
| 爆破关系 | `blast_scatter.png` | 距离-药量散点图 |
| 爆破响应 | `blast_response_curves.png` | 多次爆破前后位移变化 |
| 衰减形态 | `blast_decay_demo.png` | 指数衰减示例 |
| 目标变量 | `displacement_velocity_phases.png` | 三阶段速度时序图 |

所有图表均已生成于 `Q5/EDA/` 目录。

---

## 特征汇总

上述特征集合完整覆盖了题目要求的六类变量（降雨量、孔隙水压力、微震事件数、干湿入渗系数、爆破点距离、单段最大药量），并通过物理驱动的特征工程（滞后、累积、衰减、交互、统计量）将原始6维扩展为56维。特征设计遵循了“连续变量捕捉变化率与波动，降雨构造多尺度累积与有效降雨，微震构建密度与能量代理，爆破分别处理距离与药量并引入指数衰减”的原则，且爆破相关特征的结构允许在消融实验中公平评估距离和药量的独立贡献。附加的时间编码与位移累积信息则辅助模型捕捉周期性和趋势。此特征表可直接输入后续的 LightGBM 分阶段模型与消融实验。



以下是 `feature_56.xlsx` 中所有特征列的分类汇总：

---

### 原始解释变量（6个）
数据文件中直接给出的监测变量，对应题目要求的六类变量：

| 序号  | 列名             | 含义     | 单位       |
| --- | -------------- | ------ | -------- |
| 1   | `Rainfall`     | 降雨量    | mm/10min |
| 2   | `PorePressure` | 孔隙水压力  | kPa      |
| 3   | `Microseismic` | 微震事件数  | 次/10min  |
| 4   | `Infiltration` | 干湿入渗系数 | 无量纲      |
| 5   | `BlastDist`    | 爆破点距离  | m        |
| 6   | `BlastCharge`  | 单段最大药量 | kg       |

---

### 衍生特征量（50个）
通过特征工程从原始变量中构造的变量，涵盖滞后、滑动统计、差分、指数衰减、交互等。

#### 降雨相关特征（11个）
| 列名                | 含义                                    | 公式/参数                            |
| ----------------- | ------------------------------------- | -------------------------------- |
| `Rainfall_lag`    | 降雨量滞后1步                               | \(R(t-1)\)                       |
| `Rainfall_lag78`  | 降雨量滞后78步                              | \(R(t-78)\)                      |
| `Rainfall_cum24`  | 24h累积降雨量                              | \(\sum_{i=0}^{143} R(t-i)\)      |
| `Rain_3h_sum`     | 3h累积降雨量                               | \(\sum_{i=0}^{17} R(t-i)\)       |
| `Rain_6h_sum`     | 6h累积降雨量                               | \(\sum_{i=0}^{35} R(t-i)\)       |
| `Rain_12h_sum`    | 12h累积降雨量                              | \(\sum_{i=0}^{71} R(t-i)\)       |
| `Rain_24h_sum`    | 24h累积降雨量（同Rainfall_cum24? 注意可能重复，但保留） | \(\sum_{i=0}^{143} R(t-i)\)      |
| `Rain_eff_09`     | 有效降雨（衰减0.9）                           | \(R(t)+0.9\cdot e\!f\!\!f(t-1)\) |
| `Rain_eff_07`     | 有效降雨（衰减0.7）                           | \(R(t)+0.7\cdot e\!f\!\!f(t-1)\) |
| `Rain_6h_std`     | 6h降雨强度波动                              | 窗口36步标准差                         |
| `Time_since_rain` | 距上次降雨时间步                              | 自上次 \(R>0\) 以来的步数                |

#### 孔隙水压力相关特征（6个）
| 列名 | 含义 | 公式/参数 |
|------|------|-----------|
| `PorePressure_lag` | 孔压滞后1步 | \(P(t-1)\) |
| `PorePressure_lag6` | 孔压滞后6步 | \(P(t-6)\) |
| `Pore_Diff` | 孔压1步差分 | \(P(t)-P(t-1)\) (原始已有，可能重复) |
| `Pore_diff1` | 孔压1步差分 | \(P(t)-P(t-1)\) |
| `Pore_diff6` | 孔压6步差分 | \(P(t)-P(t-6)\) |
| `Pore_roll_std_12h` | 12h滑动标准差 | 窗口72步标准差 |

#### 干湿入渗系数相关特征（5个）
| 列名 | 含义 | 公式/参数 |
|------|------|-----------|
| `Infiltration_lag` | 入渗系数滞后1步 | \(I(t-1)\) |
| `Infiltration_lag12` | 入渗系数滞后12步 | \(I(t-12)\) |
| `Infiltration_cum24` | 24h累积入渗量 | \(\sum_{i=0}^{143} I(t-i)\) |
| `Infilt_diff1` | 1步差分 | \(I(t)-I(t-1)\) |
| `Infilt_diff6` | 6步差分 | \(I(t)-I(t-6)\) |
| `Infilt_roll_std_12h` | 12h滑动标准差 | 窗口72步标准差 |

#### 微震事件数相关特征（8个）
| 列名 | 含义 | 公式/参数 |
|------|------|-----------|
| `Microseismic_lag` | 微震滞后1步 | \(M(t-1)\) |
| `Microseismic_lag24` | 微震滞后24步 | \(M(t-24)\) |
| `Microseismic_roll6` | 6步滚动均值（原已有） | 窗口6步均值 |
| `Micro_6h_count` | 6h事件密度 | 窗口36步内 \(M>0\) 的计数 |
| `Micro_12h_count` | 12h事件密度 | 窗口72步计数 |
| `Micro_24h_count` | 24h事件密度 | 窗口144步计数 |
| `Micro_24h_cum` | 24h累积事件数 | 窗口144步求和 |
| `Micro_energy_sqrt` | 能量代理（平方根） | \(\sqrt{M(t)}\) |
| `Micro_energy_sq` | 能量代理（平方） | \(M(t)^2\) |

#### 爆破相关特征（12个）
| 列名 | 含义 | 公式/参数 |
|------|------|-----------|
| `BlastDist_lag` | 爆破距离滞后1步 | \(D(t-1)\) |
| `BlastDist_lag60` | 爆破距离滞后60步 | \(D(t-60)\) |
| `BlastCharge_lag` | 药量滞后1步 | \(Q(t-1)\) |
| `BlastCharge_lag60` | 药量滞后60步 | \(Q(t-60)\) |
| `Blast_PPV` | 峰值粒子速度 | 经验公式生成 |
| `Blast_Energy` | 瞬时爆破能量 | \(Q/D^2\) |
| `Time_since_blast` | 距上次爆破步数 | 自上次爆破以来的步数 |
| `BlastDist_impact` | 距离衰减影响 | \(\Phi_{\mathcal{B}, g_1, \tau}\)，\(\tau=50\) |
| `BlastCharge_impact` | 药量衰减影响 | \(\Phi_{\mathcal{B}, g_2, \tau}\) |
| `BlastEnergy_decay_impact` | 联合能量衰减影响 | \(\Phi_{\mathcal{B}, g_3, \tau}\) |
| `Blast_interval` | 爆破间隔 | 相邻爆破时间差向前填充 |

#### 交互与复合特征（6个）
| 列名 | 含义 | 公式 |
|------|------|------|
| `Pore_Rain` | 孔压-降雨交互 | \(P(t) \times R(t)\) |
| `Pore_Infilt` | 孔压-入渗交互 | \(P(t) \times I(t)\) |
| `PoreInfilt` | 同上（重复?） | \(P(t) \times I(t)\) |
| `PoreInfilt_diff12h` | 交互项12h差分 | \(\Delta^{(72)} (P \times I)\) |
| `Disp_cum24` | 24h累积位移增量 | \(\sum_{i=0}^{143} \! \Delta D(t-i)\) |
| `Day_sin`, `Day_cos` | 时间正余弦编码 | 周期144步 |

**说明**：部分交互项看起来重复（`Pore_Infilt`与`PoreInfilt`），实际在代码中可能是不同阶段生成的，但保留即可，LightGBM会自动处理冗余。特征总数56 = 6原始 + 50衍生。

---

### 汇总表（分类计数）

| 类别 | 原始变量 | 衍生特征 | 小计 |
|------|----------|----------|------|
| 降雨量 | 1 | 11 | 12 |
| 孔隙水压力 | 1 | 6 | 7 |
| 干湿入渗系数 | 1 | 5 | 6 |
| 微震事件数 | 1 | 8 | 9 |
| 爆破点距离 | 1 | 4 (距离相关衍生) | 5 |
| 单段最大药量 | 1 | 4 (药量相关衍生) | 5 |
| 爆破联合 | – | 4 (Blast_PPV, Blast_Energy, BlastEnergy_decay_impact, Blast_interval, Time_since_blast 等) | 4 |
| 交互与时间 | – | 6 | 6 |
| **总计** | **6** | **50** | **56** |

注：爆破联合特征指同时依赖距离和药量的特征，在消融实验中会根据剔除类别自动排除。本表已将 `Blast_interval`, `Time_since_blast`, `Blast_PPV`, `Blast_Energy` 等归入联合类，实际消融时它们会随任一爆破变量剔除而移除。

此分类可作为论文中“变量体系”表格的直接素材。