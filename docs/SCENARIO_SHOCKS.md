# Monte Carlo Scenario Shocks 定义和数值

## 概述

你的项目中有**两套场景系统**：

1. **旧系统** (`build_scenarios`): 简单的宏观变量 shock
2. **新系统** (`build_shock_table` + `simulate_paths`): Driver-aware 多类别 shock

---

## 1. 旧系统：简单宏观 Shock (`build_scenarios`)

### 场景定义

```python
def build_scenarios(X_last, history):
    scenarios = {}
    
    # 1. BASELINE - 无 shock
    scenarios["baseline"] = {}
    
    # 2. RATE_CUT - 利率下降
    scenarios["rate_cut"] = {
        "tnx_yield": -0.50,  # TNX 下降 50 个基点 (0.5%)
        "tnx_change_3m": None,  # 自动计算
    }
    
    # 3. RATE_SPIKE - 利率上升
    scenarios["rate_spike"] = {
        "tnx_yield": +1.00,  # TNX 上升 100 个基点 (1.0%)
        "tnx_change_3m": None,
    }
    
    # 4. VIX_CRASH - 波动率暴跌
    scenarios["vix_crash"] = {
        "vix_level": vix_p12 - current_vix,  # 降至历史 12 分位数
        "vix_change_3m": None,
    }
    
    # 5. VIX_SPIKE - 波动率飙升
    scenarios["vix_spike"] = {
        "vix_level": vix_p90 - current_vix,  # 升至历史 90 分位数
        "vix_change_3m": None,
    }
```

### Shock 数值总结

| 场景 | 变量 | Shock 值 | 说明 |
|------|------|----------|------|
| **baseline** | - | 0 | 无 shock，使用当前值 |
| **rate_cut** | `tnx_yield` | **-0.50%** | 10 年期国债收益率下降 50 基点 |
| **rate_spike** | `tnx_yield` | **+1.00%** | 10 年期国债收益率上升 100 基点 |
| **vix_crash** | `vix_level` | **降至 12 分位数** | VIX 降至历史低点（约 12-15） |
| **vix_spike** | `vix_level` | **升至 90 分位数** | VIX 升至历史高点（约 30-35） |

---

## 2. 新系统：Driver-Aware 多类别 Shock

### 场景 Scale Factors（全局常量）

```python
MACRO_SCALE_STRESS = 1.5      # 宏观压力场景的放大系数
FIRM_SCALE_STRESS = 1.5      # 基本面压力场景的放大系数
INTERACTION_SCALE_BULL = 1.5 # AI 牛市场景的交互项放大系数
MACRO_SCALE_BULL_CUT = -1.0  # AI 牛市场景的宏观削减（负值表示降息）
```

### 场景定义（在 `simulate_paths` 中）

#### BASE 场景
```python
if scenario_label == "base":
    eps_macro = eps_macro_base          # 正常宏观 shock
    eps_firm = eps_firm_base            # 正常基本面 shock
    eps_interaction = eps_interaction_base  # 正常交互项 shock
```

#### MACRO_STRESS 场景
```python
elif scenario_label == "macro_stress":
    eps_macro = eps_macro_base + 1.5 * sigma_macro_step  # 宏观 shock + 1.5σ
    eps_firm = eps_firm_base
    eps_interaction = eps_interaction_base
```

#### FUNDAMENTAL_STRESS 场景
```python
elif scenario_label == "fundamental_stress":
    eps_macro = eps_macro_base
    eps_firm = eps_firm_base + 1.5 * sigma_firm_step  # 基本面 shock + 1.5σ
    eps_interaction = eps_interaction_base
```

#### AI_BULL 场景
```python
elif scenario_label == "ai_bull":
    eps_macro = eps_macro_base - 1.0 * sigma_macro_step  # 宏观 shock - 1.0σ (降息)
    eps_firm = eps_firm_base
    eps_interaction = eps_interaction_base + 1.5 * sigma_interaction_step  # 交互项 + 1.5σ
```

### Shock 计算公式

每个时间步的 return 计算：

```python
# 1. 基础随机 shock（正态分布）
eps_macro_base ~ N(0, σ_macro_step)
eps_firm_base ~ N(0, σ_firm_step)
eps_interaction_base ~ N(0, σ_interaction_step)

# 2. 根据场景调整
eps_macro = eps_macro_base + scenario_adjustment_macro
eps_firm = eps_firm_base + scenario_adjustment_firm
eps_interaction = eps_interaction_base + scenario_adjustment_interaction

# 3. 加权聚合（基于 feature importance）
S = w_macro * eps_macro + w_firm * eps_firm + w_interaction * eps_interaction

# 4. 最终 return
return_t = μ_horizon + S + ε_idiosyncratic
```

### Volatility 定义

```python
# 从历史数据估计基础波动率
base_sigma = returns.std() * sqrt(12)  # 年化波动率（默认约 40%）

# 按类别分配波动率
sigmas = {
    "Macro": base_sigma * 1.2,      # 宏观：基础波动率 × 1.2
    "Firm": base_sigma * 0.8,        # 基本面：基础波动率 × 0.8
    "Interaction": base_sigma * 1.0, # 交互项：基础波动率 × 1.0
}

# 转换为月度波动率
sigma_macro_step = sigmas["Macro"] / sqrt(12)
sigma_firm_step = sigmas["Firm"] / sqrt(12)
sigma_interaction_step = sigmas["Interaction"] / sqrt(12)
```

### 权重计算（基于 Feature Importance）

```python
# 从 feature importance 聚合到类别
category_importance = {
    "Firm": sum(importance of all firm features),
    "Macro": sum(importance of all macro features),
    "Interaction": sum(importance of all interaction features),
}

# 归一化
weights = category_importance / sum(category_importance)
```

---

## 3. 实际数值示例

假设：
- 基础年化波动率：40% (`base_sigma = 0.40`)
- 月度波动率：`0.40 / sqrt(12) ≈ 0.1155`

### MACRO_STRESS 场景的 Shock

```python
# 基础宏观 shock
eps_macro_base ~ N(0, 0.1155 * 1.2) = N(0, 0.1386)

# MACRO_STRESS 调整
eps_macro = eps_macro_base + 1.5 * 0.1386
          = eps_macro_base + 0.2079

# 这意味着宏观 shock 平均增加 0.2079（约 20.8% 的月度波动率）
```

### AI_BULL 场景的 Shock

```python
# 宏观：降息效应
eps_macro = eps_macro_base - 1.0 * 0.1386
          = eps_macro_base - 0.1386  # 平均减少 13.9%

# 交互项：AI 相关特征增强
eps_interaction = eps_interaction_base + 1.5 * 0.1155
                = eps_interaction_base + 0.1733  # 平均增加 17.3%
```

---

## 4. 场景对比表

| 场景 | 宏观 Shock | 基本面 Shock | 交互项 Shock | 经济含义 |
|------|-----------|-------------|-------------|----------|
| **BASE** | 正常 | 正常 | 正常 | 基准情况 |
| **MACRO_STRESS** | **+1.5σ** | 正常 | 正常 | 宏观压力（利率上升、VIX 上升） |
| **FUNDAMENTAL_STRESS** | 正常 | **+1.5σ** | 正常 | 基本面恶化（收入下降、利润率压缩） |
| **AI_BULL** | **-1.0σ** | 正常 | **+1.5σ** | AI 牛市（降息 + AI 相关特征增强） |

---

## 5. 关键参数总结

### 全局 Scale Factors
- `MACRO_SCALE_STRESS = 1.5`
- `FIRM_SCALE_STRESS = 1.5`
- `INTERACTION_SCALE_BULL = 1.5`
- `MACRO_SCALE_BULL_CUT = -1.0`

### Volatility 倍数
- Macro: `1.2 × base_sigma`
- Firm: `0.8 × base_sigma`
- Interaction: `1.0 × base_sigma`

### 旧系统 Shock 值
- Rate Cut: `-0.50%` (TNX)
- Rate Spike: `+1.00%` (TNX)
- VIX Crash: 降至 12 分位数
- VIX Spike: 升至 90 分位数

---

## 6. 如何查看实际运行的 Shock 值

运行场景后，可以查看：

```bash
# 查看场景预测表
cat results/step7/scenario_forecast_table.csv

# 查看 shock 定义（代码中）
grep -A 10 "scenarios\[" finmc_tech/simulation/scenario_mc.py
```

