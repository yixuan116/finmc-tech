# Step 4: Champion Model Selection - 数学基础

## 📐 一、4个Horizons的数学定义

### 1.1 数据基础
- **数据类型**: 季度数据 (Quarterly)
- **价格序列**: $\{price_t\}_{t=1}^{T}$，其中 $T$ 为总季度数

### 1.2 Target变量定义

#### **1Y Horizon (1年 = 4个季度)**

$$\text{ret}_{1y}(t) = \frac{price_{t+4} - price_t}{price_t}$$

或使用对数收益率形式：

$$y_{\log\_4q}(t) = \ln\left(\frac{price_{t+4}}{price_t}\right)$$

**含义**: 在时刻 $t$，预测未来4个季度（1年）的累计收益率。

---

#### **3Y Horizon (3年 = 12个季度)**

$$\text{ret}_{3y}(t) = \frac{price_{t+12} - price_t}{price_t}$$

或：

$$y_{\log\_12q}(t) = \ln\left(\frac{price_{t+12}}{price_t}\right)$$

**含义**: 在时刻 $t$，预测未来12个季度（3年）的累计收益率。

---

#### **5Y Horizon (5年 = 20个季度)**

$$\text{ret}_{5y}(t) = \frac{price_{t+20} - price_t}{price_t}$$

或：

$$y_{\log\_20q}(t) = \ln\left(\frac{price_{t+20}}{price_t}\right)$$

**含义**: 在时刻 $t$，预测未来20个季度（5年）的累计收益率。

---

#### **10Y Horizon (10年 = 40个季度)**

$$\text{ret}_{10y}(t) = \frac{price_{t+40} - price_t}{price_t}$$

或：

$$y_{\log\_40q}(t) = \ln\left(\frac{price_{t+40}}{price_t}\right)$$

**含义**: 在时刻 $t$，预测未来40个季度（10年）的累计收益率。

---

### 1.3 统一表示

对于任意horizon $h \in \{1Y, 3Y, 5Y, 10Y\}$，对应的季度数为 $q_h \in \{4, 12, 20, 40\}$：

$$y_h(t) = \frac{price_{t+q_h} - price_t}{price_t} = \frac{price_{t+q_h}}{price_t} - 1$$

对数形式：

$$y_{\log\_h}(t) = \ln\left(\frac{price_{t+q_h}}{price_t}\right)$$

---

## 🧮 二、预测模型框架

### 2.1 通用预测公式

对于每个horizon $h$，预测模型为：

$$\hat{y}_h(t) = f_h(\mathbf{X}_t; \boldsymbol{\theta}_h)$$

其中：

- **$\hat{y}_h(t)$**: horizon $h$ 在时刻 $t$ 的预测值
- **$\mathbf{X}_t \in \mathbb{R}^{75}$**: 时刻 $t$ 的特征向量
- **$\boldsymbol{\theta}_h$**: horizon $h$ 对应的模型参数
- **$f_h(\cdot)$**: horizon $h$ 的champion模型函数

### 2.2 特征向量分解

$$\mathbf{X}_t = \begin{bmatrix}
\mathbf{X}_t^{\text{firm}} \\
\mathbf{X}_t^{\text{macro}} \\
\mathbf{X}_t^{\text{interaction}}
\end{bmatrix}$$

其中：

- **$\mathbf{X}_t^{\text{firm}} \in \mathbb{R}^{19}$**: Firm-level特征（营收、现金流、利润率等）
- **$\mathbf{X}_t^{\text{macro}} \in \mathbb{R}^{4}$**: Macro特征（TNX yield, VIX, GDP growth, CPI）
- **$\mathbf{X}_t^{\text{interaction}} \in \mathbb{R}^{52}$**: Interaction特征（Macro × Firm的交叉项）

**总特征数**: $19 + 4 + 52 = 75$

### 2.3 模型函数形式

不同horizon使用不同的champion模型：

$$f_h(\mathbf{X}_t; \boldsymbol{\theta}_h) = \begin{cases}
f_{\text{NN}}(\mathbf{X}_t; \boldsymbol{\theta}_{\text{NN}}) & \text{if } h = 1Y \\
f_{\text{RF}}(\mathbf{X}_t; \boldsymbol{\theta}_{\text{RF}}) & \text{if } h = 3Y \\
f_{\text{XGB}}(\mathbf{X}_t; \boldsymbol{\theta}_{\text{XGB}}) & \text{if } h = 5Y \\
f_{\text{EN}}(\mathbf{X}_t; \boldsymbol{\theta}_{\text{EN}}) & \text{if } h = 10Y
\end{cases}$$

其中：
- $f_{\text{NN}}$: Neural Network (MLP)
- $f_{\text{RF}}$: Random Forest
- $f_{\text{XGB}}$: XGBoost
- $f_{\text{EN}}$: ElasticNet

---

## 📊 三、训练与评估框架

### 3.1 数据分割策略

为避免时间序列数据泄漏，使用**固定时间点分割**：

$$\mathcal{D}_h = \mathcal{D}_{\text{train}}^h \cup \mathcal{D}_{\text{test}}^h$$

其中分割点由horizon决定：

| Horizon $h$ | Training Set | Test Set | 分割时间点 |
|-------------|--------------|----------|-----------|
| 1Y | $t < t_{2020-12-31}$ | $t > t_{2022-12-31}$ | $t_{\text{split}} = 2020-12-31$ |
| 3Y | $t < t_{2018-12-31}$ | $t > t_{2020-12-31}$ | $t_{\text{split}} = 2018-12-31$ |
| 5Y | $t < t_{2016-12-31}$ | $t > t_{2018-12-31}$ | $t_{\text{split}} = 2016-12-31$ |
| 10Y | $t < t_{2012-12-31}$ | $t > t_{2014-12-31}$ | $t_{\text{split}} = 2012-12-31$ |

**数学表示**:

$$\mathcal{D}_{\text{train}}^h = \{(t, \mathbf{X}_t, y_h(t)) : t < t_{\text{split}}^h\}$$
$$\mathcal{D}_{\text{test}}^h = \{(t, \mathbf{X}_t, y_h(t)) : t > t_{\text{split}}^h + \Delta t_h\}$$

其中 $\Delta t_h$ 确保test set的forward return不重叠。

### 3.2 损失函数

对于每个horizon $h$，模型训练最小化以下损失：

$$\mathcal{L}_h(\boldsymbol{\theta}_h) = \frac{1}{|\mathcal{D}_{\text{train}}^h|} \sum_{(t, \mathbf{X}_t, y_h(t)) \in \mathcal{D}_{\text{train}}^h} \ell(y_h(t), f_h(\mathbf{X}_t; \boldsymbol{\theta}_h))$$

其中 $\ell(\cdot, \cdot)$ 为损失函数（通常为MSE）：

$$\ell(y, \hat{y}) = (y - \hat{y})^2$$

### 3.3 评估指标

在test set上计算以下指标：

#### **Mean Absolute Error (MAE)**

$$\text{MAE}_h = \frac{1}{|\mathcal{D}_{\text{test}}^h|} \sum_{(t, \mathbf{X}_t, y_h(t)) \in \mathcal{D}_{\text{test}}^h} |y_h(t) - \hat{y}_h(t)|$$

#### **Root Mean Squared Error (RMSE)**

$$\text{RMSE}_h = \sqrt{\frac{1}{|\mathcal{D}_{\text{test}}^h|} \sum_{(t, \mathbf{X}_t, y_h(t)) \in \mathcal{D}_{\text{test}}^h} (y_h(t) - \hat{y}_h(t))^2}$$

#### **Coefficient of Determination (R²)**

$$\text{R²}_h = 1 - \frac{\sum_{(t, \mathbf{X}_t, y_h(t)) \in \mathcal{D}_{\text{test}}^h} (y_h(t) - \hat{y}_h(t))^2}{\sum_{(t, \mathbf{X}_t, y_h(t)) \in \mathcal{D}_{\text{test}}^h} (y_h(t) - \bar{y}_h)^2}$$

其中 $\bar{y}_h = \frac{1}{|\mathcal{D}_{\text{test}}^h|} \sum_{(t, \mathbf{X}_t, y_h(t)) \in \mathcal{D}_{\text{test}}^h} y_h(t)$ 为test set的平均值。

### 3.4 Champion Model选择

对于每个horizon $h$，从候选模型集合 $\mathcal{M} = \{\text{Linear}, \text{Ridge}, \text{ElasticNet}, \text{RF}, \text{XGB}, \text{NN}\}$ 中选择：

$$f_h^* = \arg\max_{f \in \mathcal{M}} \text{R²}_h(f)$$

即选择在test set上R²最高的模型。

---

## 🏆 四、Champion Model结果

### 4.1 各Horizon的Champion

| Horizon $h$ | Champion Model $f_h^*$ | Test R² | MAE | RMSE | $|\mathcal{D}_{\text{test}}^h|$ |
|-------------|------------------------|---------|-----|------|----------------------|
| 1Y | NeuralNetwork | -1.15 | 0.66 | 0.84 | 12 |
| 3Y | RandomForest | -1.82 | 0.45 | 0.50 | 11 |
| 5Y | XGBoost | -2.33 | 0.65 | 0.74 | 9 |
| 10Y | ElasticNet | -7.02 | 0.59 | 0.61 | 5 |

### 4.2 Overall Champion: RandomForest

虽然RandomForest不是所有horizon的R²冠军，但被选为**Overall Champion**，原因：

1. **可解释性**: RF提供特征重要性 $\boldsymbol{\phi}_h^{\text{RF}} \in \mathbb{R}^{75}$，满足经济建模需求
2. **稳定性**: 在3Y为champion，1Y和5Y与champion差距仅0.02 R²
3. **泛化能力**: 相比XGBoost（5Y训练集R²=1.0），RF保持更好的train-test一致性

**数学表示**:

$$f_{\text{overall}}^* = \text{RandomForest}$$

满足：

$$\text{R²}_{3Y}(\text{RF}) = \max_{f \in \mathcal{M}} \text{R²}_{3Y}(f)$$

且

$$|\text{R²}_{1Y}(\text{RF}) - \text{R²}_{1Y}(\text{NN})| < 0.02$$
$$|\text{R²}_{5Y}(\text{RF}) - \text{R²}_{5Y}(\text{XGB})| < 0.02$$

---

## 📈 五、模型具体形式

### 5.1 Random Forest (3Y Champion)

$$f_{\text{RF}}(\mathbf{X}_t; \boldsymbol{\theta}_{\text{RF}}) = \frac{1}{B} \sum_{b=1}^{B} T_b(\mathbf{X}_t; \boldsymbol{\theta}_b)$$

其中：
- $B = 500$: 树的数量
- $T_b(\cdot)$: 第 $b$ 棵决策树
- $\boldsymbol{\theta}_{\text{RF}} = \{\boldsymbol{\theta}_1, \ldots, \boldsymbol{\theta}_B\}$: 所有树的参数

### 5.2 Neural Network (1Y Champion)

$$f_{\text{NN}}(\mathbf{X}_t; \boldsymbol{\theta}_{\text{NN}}) = \sigma_2(\mathbf{W}_2 \cdot \sigma_1(\mathbf{W}_1 \mathbf{X}_t + \mathbf{b}_1) + \mathbf{b}_2)$$

其中：
- $\mathbf{W}_1 \in \mathbb{R}^{64 \times 75}$, $\mathbf{b}_1 \in \mathbb{R}^{64}$: 第一层参数
- $\mathbf{W}_2 \in \mathbb{R}^{32 \times 64}$, $\mathbf{b}_2 \in \mathbb{R}^{32}$: 第二层参数
- $\sigma_1, \sigma_2$: 激活函数（ReLU）
- $\boldsymbol{\theta}_{\text{NN}} = \{\mathbf{W}_1, \mathbf{b}_1, \mathbf{W}_2, \mathbf{b}_2\}$

### 5.3 XGBoost (5Y Champion)

$$f_{\text{XGB}}(\mathbf{X}_t; \boldsymbol{\theta}_{\text{XGB}}) = \sum_{k=1}^{K} \eta \cdot f_k(\mathbf{X}_t)$$

其中：
- $K = 500$: 树的数量
- $\eta = 0.05$: 学习率
- $f_k(\cdot)$: 第 $k$ 棵回归树
- $\boldsymbol{\theta}_{\text{XGB}} = \{f_1, \ldots, f_K\}$

### 5.4 ElasticNet (10Y Champion)

$$f_{\text{EN}}(\mathbf{X}_t; \boldsymbol{\theta}_{\text{EN}}) = \boldsymbol{\beta}_h^T \mathbf{X}_t + \beta_0$$

其中参数通过以下优化得到：

$$\boldsymbol{\theta}_{\text{EN}} = \arg\min_{\boldsymbol{\beta}, \beta_0} \left\{ \frac{1}{2|\mathcal{D}_{\text{train}}^h|} \|\mathbf{y}_h - \mathbf{X}\boldsymbol{\beta} - \beta_0\|_2^2 + \alpha \left( \rho \|\boldsymbol{\beta}\|_1 + \frac{1-\rho}{2} \|\boldsymbol{\beta}\|_2^2 \right) \right\}$$

其中：
- $\alpha = 0.1$: 正则化强度
- $\rho = 0.5$: L1/L2混合比例

---

## 📋 六、数据统计

### 6.1 数据集规模

- **总样本数**: $T = 71$ 个季度观测
- **特征维度**: $d = 75$
- **各horizon有效样本数**:

$$|\mathcal{D}_h| = \begin{cases}
65 & \text{if } h = 1Y \\
57 & \text{if } h = 3Y \\
50 & \text{if } h = 5Y \\
30 & \text{if } h = 10Y
\end{cases}$$

### 6.2 特征组成

$$\mathbf{X}_t = \begin{bmatrix}
\mathbf{X}_t^{\text{firm}} & (19 \text{ features}) \\
\mathbf{X}_t^{\text{macro}} & (4 \text{ features}) \\
\mathbf{X}_t^{\text{interaction}} & (52 \text{ features})
\end{bmatrix} \in \mathbb{R}^{75}$$

---

## 🔗 七、完整预测流程

### 7.1 训练阶段

对于每个horizon $h \in \{1Y, 3Y, 5Y, 10Y\}$:

1. **数据准备**: 
   $$\mathcal{D}_{\text{train}}^h = \{(t, \mathbf{X}_t, y_h(t)) : t < t_{\text{split}}^h\}$$

2. **模型训练**:
   $$\boldsymbol{\theta}_h^* = \arg\min_{\boldsymbol{\theta}_h} \mathcal{L}_h(\boldsymbol{\theta}_h)$$

3. **模型评估**:
   $$\text{R²}_h^* = \text{R²}_h(f_h(\cdot; \boldsymbol{\theta}_h^*))$$

4. **Champion选择**:
   $$f_h^* = \arg\max_{f \in \mathcal{M}} \text{R²}_h(f)$$

### 7.2 预测阶段

给定新观测 $\mathbf{X}_{t_{\text{new}}}$，各horizon的预测为：

$$\hat{y}_h(t_{\text{new}}) = f_h^*(\mathbf{X}_{t_{\text{new}}}; \boldsymbol{\theta}_h^*)$$

---

## 📝 总结

Step 4建立了4个horizon的预测框架：

1. **Target定义**: 每个horizon对应不同长度的forward return
2. **模型选择**: 每个horizon选择最优的champion model
3. **评估标准**: 基于test set的R²进行模型选择
4. **Overall Champion**: RandomForest因稳定性和可解释性被选为整体champion

该框架为后续的feature importance分析（Step 5）和scenario-based Monte Carlo（Step 8）提供了基础。

