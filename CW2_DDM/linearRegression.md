# Linear Regression: SGD & MAE

## Key Concepts

| Term | Symbol | Description |
|------|--------|-------------|
| **Features** | $X$ | Input variables (dT in our case) |
| **Target** | $y$ | What we predict (Qdot) |
| **Weights** | $w$ | Parameters the model learns |
| **Bias** | $w_0$ | Intercept term (prediction when features = 0) |
| **Prediction** | $\hat{y}$ | Model output: $\hat{y} = Xw$ |
| **Loss** | $L$ | Measures prediction error |
| **Gradient** | $\nabla L$ | Direction of steepest loss increase |
| **Learning Rate** | $\alpha$ | Step size for weight updates |
| **Epoch** | - | One complete pass through all samples |

---

## The Model

$$\hat{y} = w_0 + w_1 x_1 = \begin{bmatrix} 1 & x_1 \end{bmatrix} \begin{bmatrix} w_0 \\ w_1 \end{bmatrix} = Xw$$

For window heat loss: $\hat{Q}_{dot} = w_0 + w_1 \cdot \Delta T$

---

## Loss Functions

| Loss | Formula | Gradient | Properties |
|------|---------|----------|------------|
| **MSE** | $\frac{1}{N}\sum(y - \hat{y})^2$ | $\frac{2}{N}X^T(\hat{y} - y)$ | Penalizes large errors heavily |
| **MAE** | $\frac{1}{N}\sum\|y - \hat{y}\|$ | $\frac{1}{N}X^T \cdot \text{sign}(\hat{y} - y)$ | Robust to outliers |

---

## Optimizers

| Method | Update Rule | Samples per Update |
|--------|-------------|-------------------|
| **Batch GD** | $w = w - \alpha \nabla L$ | All N samples |
| **SGD** | $w = w - \alpha \nabla L_i$ | 1 sample |

---

## Task 1: Stochastic Gradient Descent (MSE Loss)

### Workflow

```
1. Load data → X (with bias column), y
2. Normalize features
3. Initialize weights randomly
4. For each EPOCH:
   ├── Shuffle sample indices
   ├── For each SAMPLE i:
   │   ├── Predict: ŷ = xᵢ · w
   │   ├── Error: e = ŷ - yᵢ
   │   ├── Gradient: g = 2 · xᵢᵀ · e
   │   └── Update: w = w - α · g
   └── Record loss (end of epoch)
```

### Key Formulas

**Prediction (single sample):**
$$\hat{y}_i = x_i \cdot w = [1, x_{i1}] \cdot [w_0, w_1]^T$$

**MSE Gradient (single sample):**
$$\nabla L_i = 2 \cdot x_i^T \cdot (\hat{y}_i - y_i)$$

**Weight Update:**
$$w_{new} = w_{old} - \alpha \cdot \nabla L_i$$

### Epoch Definition (Important!)

- **Batch GD:** 1 epoch = 1 weight update (uses all samples)
- **SGD:** 1 epoch = N weight updates (one per sample)

---

## Task 2: Gradient Descent with MAE Loss

### Workflow

```
1. Load data → X (with bias column), y
2. Normalize features
3. Initialize: w₀ = mean(y), w₁ = 0
4. For each ITERATION:
   ├── Predict all: ŷ = X · w
   ├── Error: e = ŷ - y
   ├── Gradient: g = (1/N) · Xᵀ · sign(e)
   ├── Update: w = w - α · g
   ├── Record loss
   └── Check early stopping
```

### Key Formulas

**MAE Loss:**
$$L_{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|$$

**Sign Function:**
$$\text{sign}(x) = \begin{cases} +1 & x > 0 \\ 0 & x = 0 \\ -1 & x < 0 \end{cases}$$

**MAE Gradient:**
$$\nabla L_{MAE} = \frac{1}{N} X^T \cdot \text{sign}(\hat{y} - y)$$

### Why MAE Needs Larger Learning Rate

- MSE gradient ∝ error magnitude (large errors → large gradients)
- MAE gradient ∝ sign only (±1 regardless of error size)
- Therefore MAE needs larger $\alpha$ to make meaningful steps

---

## Data Preprocessing

### Adding Bias Column

```
Before:          After:
[[21.18],        [[1, 21.18],
 [14.29],    →    [1, 14.29],
 [4.46]]          [1, 4.46]]
```

### Normalization

$$x_{norm} = \frac{x - \mu}{\sigma}$$

**Why?** Features on different scales cause unstable training.

---

## Summary Comparison

| Aspect | Task 1 (SGD) | Task 2 (MAE) |
|--------|--------------|--------------|
| **Loss** | MSE | MAE |
| **Optimization** | Stochastic (1 sample) | Batch (all samples) |
| **Gradient** | $2 x_i^T (ŷ_i - y_i)$ | $\frac{1}{N} X^T \text{sign}(ŷ - y)$ |
| **Learning Rate** | Small (~0.01) | Large (~50) |
| **Loop Structure** | Epochs → Samples | Iterations |
| **Outlier Sensitivity** | High | Low |

---

## Physical Interpretation

For window heat loss: $\dot{Q} = w_0 + w_1 \cdot \Delta T$

- **$w_0$ (bias):** Base heat loss when ΔT = 0
- **$w_1$ (slope):** Heat loss per degree temperature difference (∝ h·A)
- **Expected:** $w_1 > 0$ (higher ΔT → more heat loss)