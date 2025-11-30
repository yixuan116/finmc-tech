# 所有 Monte Carlo Scenarios 的 Shock 完整总结

## 概述

你的项目中有**两套场景系统**，共定义了 **9 个不同的场景**：

1. **旧系统** (`build_scenarios`): 5 个简单宏观 shock 场景
2. **新系统** (`simulate_paths`): 4 个 driver-aware 多类别 shock 场景

---

## 系统 1: 简单宏观 Shock 场景（5个）

### 1. BASELINE（基准场景）

**Shock 定义**:
```python
scenarios["baseline"] = {}
```

**Shock 值**:
- **无 shock** - 使用当前特征值

**经济含义**: 
- 保持当前宏观环境不变
- 作为其他场景的对比基准

---

### 2. RATE_CUT（降息场景）

**Shock 定义**:
```python
scenarios["rate_cut"] = {
    "tnx_yield": -0.50,  # 10年期国债收益率下降 50 基点
    "tnx_change_3m": None,  # 自动计算
}
```

**Shock 值**:
- `tnx_yield`: **-0.50%** (绝对变化)
- `tnx_change_3m`: 自动从 `tnx_yield` 变化计算

**经济含义**:
- 美联储降息，10年期国债收益率下降 50 基点
- 对成长股（如 NVDA）通常利好
- 降低贴现率，提升估值

**实际影响**（从你的结果看）:
- 预期收益: **+1593.81%** (极端利好)
- 说明 NVDA 对利率非常敏感

---

### 3. RATE_SPIKE（加息场景）

**Shock 定义**:
```python
scenarios["rate_spike"] = {
    "tnx_yield": +1.00,  # 10年期国债收益率上升 100 基点
    "tnx_change_3m": None,
}
```

**Shock 值**:
- `tnx_yield`: **+1.00%** (绝对变化)
- `tnx_change_3m`: 自动计算

**经济含义**:
- 美联储加息，10年期国债收益率上升 100 基点
- 对成长股通常利空
- 提高贴现率，压缩估值

**实际影响**（从你的结果看）:
- 预期收益: **+107.53%** (仍然为正，但远低于降息场景)
- 说明即使加息，NVDA 仍有增长动力

---

### 4. VIX_CRASH（波动率暴跌场景）

**Shock 定义**:
```python
current_vix = X_last["vix_level"].iloc[0]  # 当前 VIX 值
vix_p12 = history["vix_level"].quantile(0.12)  # 历史 12 分位数

scenarios["vix_crash"] = {
    "vix_level": vix_p12 - current_vix,  # 绝对变化
    "vix_change_3m": None,
}
```

**Shock 值**（实际运行）:
- `vix_level`: **约 -1.92** (从当前值降至历史 12 分位数)
- 目标 VIX 水平: **约 12-15** (历史低波动率)

**经济含义**:
- 市场恐慌情绪消退
- 波动率降至历史低位
- 通常伴随市场稳定和上涨

**实际影响**:
- 预期收益: **+56.08%**
- 上涨概率: **72.60%**

---

### 5. VIX_SPIKE（波动率飙升场景）

**Shock 定义**:
```python
vix_p90 = history["vix_level"].quantile(0.90)  # 历史 90 分位数

scenarios["vix_spike"] = {
    "vix_level": vix_p90 - current_vix,  # 绝对变化
    "vix_change_3m": None,
}
```

**Shock 值**（实际运行）:
- `vix_level`: **约 +14.96** (从当前值升至历史 90 分位数)
- 目标 VIX 水平: **约 30-35** (历史高波动率)

**经济含义**:
- 市场恐慌情绪上升
- 波动率升至历史高位
- 通常伴随市场下跌和不确定性

**实际影响**:
- 预期收益: **+42.47%** (仍然为正，但低于低波动场景)
- 上涨概率: **66.20%**

---

## 系统 2: Driver-Aware 多类别 Shock 场景（4个）

### 全局 Scale Factors

```python
MACRO_SCALE_STRESS = 1.5      # 宏观压力放大系数
FIRM_SCALE_STRESS = 1.5      # 基本面压力放大系数
INTERACTION_SCALE_BULL = 1.5 # AI 牛市交互项放大系数
MACRO_SCALE_BULL_CUT = -1.0  # AI 牛市宏观削减（负值=降息）
```

### Scale Factors 背后的 Features 和 Drivers

#### 1. MACRO_SCALE_STRESS = 1.5

**影响的 Features（Macro 类别）**:
- `tnx_yield` - 10年期国债收益率（利率水平）
- `tnx_change_3m` - 10年期国债收益率3个月变化
- `vix_level` - VIX 波动率指数水平
- `vix_change_3m` - VIX 波动率指数3个月变化

**经济 Drivers**:
- **利率上升**: 美联储加息，10年期国债收益率上升
- **波动率上升**: 市场恐慌情绪增加，VIX 上升
- **通胀压力**: 通胀预期上升
- **流动性收紧**: 货币政策收紧

**在 MACRO_STRESS 场景中的应用**:
```python
eps_macro = eps_macro_base + 1.5 * sigma_macro_step
```
- 所有宏观 features 的 shock 平均增加 1.5 个标准差
- 相当于利率上升、VIX 上升等宏观不利因素的组合

---

#### 2. FIRM_SCALE_STRESS = 1.5

**影响的 Features（Firm 类别，共 27 个）**:

**收入相关**:
- `revenue` - 季度收入
- `rev_yoy` - 收入同比增长率
- `rev_qoq` - 收入环比增长率
- `rev_accel` - 收入增长加速度

**价格/收益相关**:
- `price_returns_1m`, `price_returns_3m`, `price_returns_6m`, `price_returns_12m` - 不同时间窗口的价格收益率
- `price_momentum` - 价格动量
- `price_volatility` - 价格波动率
- `price_ma_4q` - 4季度移动平均价格
- `price_to_ma_4q` - 价格相对移动平均的比率

**其他 Firm Features**:
- `adj_close` - 调整后收盘价
- 以及其他公司特定的基本面指标

**经济 Drivers**:
- **收入下降**: 季度收入同比下降
- **增长放缓**: 收入增长率下降，增长加速度为负
- **盈利能力下降**: 利润率压缩
- **竞争加剧**: 市场份额下降
- **技术落后**: 产品竞争力下降

**在 FUNDAMENTAL_STRESS 场景中的应用**:
```python
eps_firm = eps_firm_base + 1.5 * sigma_firm_step
```
- 所有基本面 features 的 shock 平均增加 1.5 个标准差
- 相当于收入下降、增长放缓等基本面不利因素

---

#### 3. INTERACTION_SCALE_BULL = 1.5

**影响的 Features（Interaction 类别，共 40 个）**:

**TNX 交互项** (20 个):
- `ix_tnx_yield__revenue` - 利率 × 收入
- `ix_tnx_yield__rev_yoy` - 利率 × 收入同比增长
- `ix_tnx_yield__rev_qoq` - 利率 × 收入环比增长
- `ix_tnx_yield__rev_accel` - 利率 × 收入加速度
- `ix_tnx_yield__price_returns_1m/3m/6m/12m` - 利率 × 不同时间窗口收益
- `ix_tnx_yield__price_momentum` - 利率 × 价格动量
- `ix_tnx_yield__price_volatility` - 利率 × 价格波动率
- `ix_tnx_change_3m__*` - 利率变化 × 各种基本面指标（同上结构）

**VIX 交互项** (20 个):
- `ix_vix_level__revenue` - VIX × 收入
- `ix_vix_level__rev_yoy` - VIX × 收入同比增长
- `ix_vix_level__rev_qoq` - VIX × 收入环比增长
- `ix_vix_level__rev_accel` - VIX × 收入加速度
- `ix_vix_level__price_returns_1m/3m/6m/12m` - VIX × 不同时间窗口收益
- `ix_vix_level__price_momentum` - VIX × 价格动量
- `ix_vix_level__price_volatility` - VIX × 价格波动率
- `ix_vix_change_3m__*` - VIX 变化 × 各种基本面指标（同上结构）

**经济 Drivers**:
- **AI 投资增加**: 利率下降时，AI 相关投资增加（利率 × AI 收入）
- **AI 收入增长**: AI 相关业务收入快速增长（利率 × AI 收入增长率）
- **市场情绪 + 基本面**: VIX 下降时，基本面改善的放大效应
- **宏观 × 微观协同**: 宏观环境（利率/VIX）与公司基本面（收入/增长）的交互作用

**在 AI_BULL 场景中的应用**:
```python
eps_interaction = eps_interaction_base + 1.5 * sigma_interaction_step
```
- 所有交互项 features 的 shock 平均增加 1.5 个标准差
- 相当于"利率下降时 AI 收入增长"、"VIX 下降时增长加速"等交互效应的增强

---

#### 4. MACRO_SCALE_BULL_CUT = -1.0

**影响的 Features（Macro 类别，同 MACRO_SCALE_STRESS）**:
- `tnx_yield` - 10年期国债收益率
- `tnx_change_3m` - 10年期国债收益率3个月变化
- `vix_level` - VIX 波动率指数水平
- `vix_change_3m` - VIX 波动率指数3个月变化

**经济 Drivers**:
- **降息**: 美联储降息，10年期国债收益率下降
- **波动率下降**: 市场恐慌情绪消退，VIX 下降
- **流动性宽松**: 货币政策宽松
- **风险偏好上升**: 投资者风险偏好增加

**在 AI_BULL 场景中的应用**:
```python
eps_macro = eps_macro_base + (-1.0) * sigma_macro_step
```
- 所有宏观 features 的 shock 平均减少 1.0 个标准差
- 负值表示利好（降息、VIX 下降）
- 与 `INTERACTION_SCALE_BULL` 组合，形成"降息 + AI 主题增强"的双重利好

---

## Features 分类总结

### Macro Features (4 个)
1. `tnx_yield` - 10年期国债收益率
2. `tnx_change_3m` - 10年期国债收益率3个月变化
3. `vix_level` - VIX 波动率指数水平
4. `vix_change_3m` - VIX 波动率指数3个月变化

### Firm Features (27 个)
**收入类**:
- `revenue`, `rev_yoy`, `rev_qoq`, `rev_accel`

**价格/收益类**:
- `price_returns_1m/3m/6m/12m`, `price_momentum`, `price_volatility`
- `price_ma_4q`, `price_to_ma_4q`

**其他**:
- `adj_close`, `days_since_start`, `form`, `fp`, `fy`, `month`, `quarter`, `year`
- `period_end`, `px_date`, `ticker`, `tag_used`
- `future_12m_logprice`, `future_12m_price`, `future_12m_return`

### Interaction Features (40 个)
**TNX 交互项** (20 个):
- `ix_tnx_yield__*` (10 个): revenue, rev_yoy, rev_qoq, rev_accel, price_returns_1m/3m/6m/12m, price_momentum, price_volatility
- `ix_tnx_change_3m__*` (10 个): 同上结构

**VIX 交互项** (20 个):
- `ix_vix_level__*` (10 个): 同上结构
- `ix_vix_change_3m__*` (10 个): 同上结构

---

## Scale Factors 如何应用到 Features

### 权重计算（基于 Feature Importance）

```python
# 1. 从 feature importance 聚合到类别
category_importance = {
    "Macro": sum(importance of all macro features),
    "Firm": sum(importance of all firm features),
    "Interaction": sum(importance of all interaction features),
}

# 2. 归一化
weights = category_importance / sum(category_importance)

# 3. 在 Monte Carlo 中应用
S = w_macro * eps_macro + w_firm * eps_firm + w_interaction * eps_interaction
```

### 实际应用示例

假设 feature importance 显示：
- Macro features 总重要性: 40%
- Firm features 总重要性: 35%
- Interaction features 总重要性: 25%

则权重为：
- `w_macro = 0.40`
- `w_firm = 0.35`
- `w_interaction = 0.25`

在 **MACRO_STRESS** 场景中：
```python
eps_macro = eps_macro_base + 1.5 * 0.1386 = eps_macro_base + 0.2079
S = 0.40 * (eps_macro_base + 0.2079) + 0.35 * eps_firm_base + 0.25 * eps_interaction_base
```

这意味着：
- 宏观 shock 的影响权重为 40%
- 宏观 shock 平均增加 0.2079，通过 40% 权重影响最终 return
- 总影响 ≈ 0.40 × 0.2079 = 0.083 (约 8.3% 的月度 return 影响)

### Volatility 定义

```python
# 基础年化波动率（从历史数据估计）
base_sigma = returns.std() * sqrt(12)  # 默认约 40%

# 按类别分配波动率
sigmas = {
    "Macro": base_sigma * 1.2,      # 宏观：基础 × 1.2
    "Firm": base_sigma * 0.8,        # 基本面：基础 × 0.8
    "Interaction": base_sigma * 1.0, # 交互项：基础 × 1.0
}

# 转换为月度波动率
sigma_macro_step = sigmas["Macro"] / sqrt(12)
sigma_firm_step = sigmas["Firm"] / sqrt(12)
sigma_interaction_step = sigmas["Interaction"] / sqrt(12)
```

**假设 base_sigma = 0.40 (40%)**:
- `sigma_macro_step` ≈ **0.1386** (13.86% 月度)
- `sigma_firm_step` ≈ **0.0924** (9.24% 月度)
- `sigma_interaction_step` ≈ **0.1155** (11.55% 月度)

---

### 6. BASE（基准场景 - Driver-Aware）

**Shock 定义**:
```python
if scenario_label == "base":
    eps_macro = eps_macro_base          # N(0, σ_macro_step)
    eps_firm = eps_firm_base            # N(0, σ_firm_step)
    eps_interaction = eps_interaction_base  # N(0, σ_interaction_step)
```

**Shock 值**:
- 宏观 shock: `N(0, 0.1386)` - 正常随机波动
- 基本面 shock: `N(0, 0.0924)` - 正常随机波动
- 交互项 shock: `N(0, 0.1155)` - 正常随机波动

**经济含义**:
- 所有类别使用正常随机 shock
- 作为 driver-aware 系统的基准

---

### 7. MACRO_STRESS（宏观压力场景）

**Shock 定义**:
```python
elif scenario_label == "macro_stress":
    eps_macro = eps_macro_base + 1.5 * sigma_macro_step
    eps_firm = eps_firm_base
    eps_interaction = eps_interaction_base
```

**Shock 值**:
- 宏观 shock: `N(0, 0.1386) + 1.5 × 0.1386 = N(0.2079, 0.1386)`
  - **平均增加**: +0.2079 (约 +20.8% 月度波动率)
  - **含义**: 宏观压力平均增加约 20.8%
- 基本面 shock: `N(0, 0.0924)` - 正常
- 交互项 shock: `N(0, 0.1155)` - 正常

**经济含义**:
- 利率上升、VIX 上升、通胀压力等宏观不利因素
- 类似 `rate_spike` 但更全面（包含所有宏观特征）

**数值示例**:
- 如果随机 shock = 0，则宏观 shock = +0.2079
- 如果随机 shock = +1σ，则宏观 shock = +0.2079 + 0.1386 = +0.3465

---

### 8. FUNDAMENTAL_STRESS（基本面压力场景）

**Shock 定义**:
```python
elif scenario_label == "fundamental_stress":
    eps_macro = eps_macro_base
    eps_firm = eps_firm_base + 1.5 * sigma_firm_step
    eps_interaction = eps_interaction_base
```

**Shock 值**:
- 宏观 shock: `N(0, 0.1386)` - 正常
- 基本面 shock: `N(0, 0.0924) + 1.5 × 0.0924 = N(0.1386, 0.0924)`
  - **平均增加**: +0.1386 (约 +13.9% 月度波动率)
  - **含义**: 基本面压力平均增加约 13.9%
- 交互项 shock: `N(0, 0.1155)` - 正常

**经济含义**:
- 公司收入下降、利润率压缩、竞争加剧等基本面不利因素
- 直接影响公司盈利能力

**数值示例**:
- 如果随机 shock = 0，则基本面 shock = +0.1386
- 如果随机 shock = +1σ，则基本面 shock = +0.1386 + 0.0924 = +0.2310

---

### 9. AI_BULL（AI 牛市场景）

**Shock 定义**:
```python
elif scenario_label == "ai_bull":
    eps_macro = eps_macro_base + (-1.0) * sigma_macro_step  # 降息
    eps_firm = eps_firm_base
    eps_interaction = eps_interaction_base + 1.5 * sigma_interaction_step  # AI 增强
```

**Shock 值**:
- 宏观 shock: `N(0, 0.1386) - 1.0 × 0.1386 = N(-0.1386, 0.1386)`
  - **平均减少**: -0.1386 (约 -13.9% 月度波动率)
  - **含义**: 降息效应，宏观环境改善
- 基本面 shock: `N(0, 0.0924)` - 正常
- 交互项 shock: `N(0, 0.1155) + 1.5 × 0.1155 = N(0.1733, 0.1155)`
  - **平均增加**: +0.1733 (约 +17.3% 月度波动率)
  - **含义**: AI 相关特征（如 AI 投资、AI 收入等）增强

**经济含义**:
- **宏观**: 降息环境（类似 `rate_cut`）
- **交互项**: AI 相关特征大幅增强（AI 投资增加、AI 收入增长等）
- 双重利好：宽松货币 + AI 主题

**数值示例**:
- 宏观: 如果随机 shock = 0，则宏观 shock = -0.1386 (降息)
- 交互项: 如果随机 shock = 0，则交互项 shock = +0.1733 (AI 增强)

---

## 完整对比表

### 系统 1: 简单宏观 Shock

| 场景 | Shock 变量 | Shock 值 | 经济含义 |
|------|-----------|----------|----------|
| **BASELINE** | - | 0 | 无变化 |
| **RATE_CUT** | `tnx_yield` | **-0.50%** | 降息 50 基点 |
| **RATE_SPIKE** | `tnx_yield` | **+1.00%** | 加息 100 基点 |
| **VIX_CRASH** | `vix_level` | **-1.92** | VIX 降至 12 分位数 |
| **VIX_SPIKE** | `vix_level` | **+14.96** | VIX 升至 90 分位数 |

### 系统 2: Driver-Aware 多类别 Shock

| 场景 | 宏观 Shock | 基本面 Shock | 交互项 Shock | 经济含义 |
|------|-----------|-------------|-------------|----------|
| **BASE** | 正常 `N(0, σ)` | 正常 `N(0, σ)` | 正常 `N(0, σ)` | 基准情况 |
| **MACRO_STRESS** | **+1.5σ** | 正常 | 正常 | 宏观压力（利率↑、VIX↑） |
| **FUNDAMENTAL_STRESS** | 正常 | **+1.5σ** | 正常 | 基本面恶化（收入↓、利润率↓） |
| **AI_BULL** | **-1.0σ** | 正常 | **+1.5σ** | 降息 + AI 主题增强 |

### 数值换算（假设 base_sigma = 0.40）

| 场景 | 宏观调整 | 基本面调整 | 交互项调整 |
|------|---------|-----------|-----------|
| **MACRO_STRESS** | **+0.2079** | 0 | 0 |
| **FUNDAMENTAL_STRESS** | 0 | **+0.1386** | 0 |
| **AI_BULL** | **-0.1386** | 0 | **+0.1733** |

---

## Shock 应用方式

### 系统 1: 直接修改特征值

```python
# 在 apply_shock 函数中
X_shocked[feat] = current_val + shock_value
```

### 系统 2: 在 Monte Carlo 路径中应用

```python
# 在 simulate_paths 函数中
# 1. 生成基础随机 shock
eps_macro_base ~ N(0, σ_macro_step)
eps_firm_base ~ N(0, σ_firm_step)
eps_interaction_base ~ N(0, σ_interaction_step)

# 2. 根据场景调整
eps_macro = eps_macro_base + scenario_adjustment

# 3. 加权聚合（基于 feature importance）
S = w_macro * eps_macro + w_firm * eps_firm + w_interaction * eps_interaction

# 4. 计算 return
return_t = μ_horizon + S + ε_idiosyncratic
```

---

## 关键发现

1. **RATE_CUT 影响最大**: -0.50% 的利率下降导致 +1593.81% 的预期收益
   - 说明 NVDA 对利率极其敏感

2. **VIX 影响相对温和**: VIX 变化对收益影响较小（56% vs 42%）
   - 说明波动率变化不如利率变化重要

3. **Driver-Aware 系统更精细**: 
   - 区分宏观、基本面、交互项三类 shock
   - 可以组合不同类别的压力（如 AI_BULL 同时有宏观利好和交互项利好）

4. **Scale Factors 设计**:
   - 1.5σ 的调整相当于"一个半标准差"的压力
   - -1.0σ 的宏观调整相当于"一个标准差"的利好（降息）

---

## 使用建议

- **系统 1** (`build_scenarios`): 适合快速测试单一宏观变量的影响
- **系统 2** (`simulate_paths`): 适合深入分析多类别驱动的综合影响

两个系统可以互补使用，提供不同层次的场景分析。

