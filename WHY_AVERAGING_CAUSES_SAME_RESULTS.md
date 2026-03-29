# 为什么对所有文本tokens求平均会导致不同策略产生相同结果？

## 问题

虽然三种策略的排序方式不同，但为什么最终选出的tokens却相同？

## 数学推导

### 符号定义

- H = 32 (heads数量)
- T = 35 (文本tokens数量)  
- I = 576 (图像tokens数量)
- A[h, t, i] = 第h个head，第t个文本token对第i个图像token的attention值

### 三种策略的计算

#### 策略1: max_head
```
步骤1: 计算每个head的重要性
  importance[h] = Σ_{t,i} A[h, t, i]

步骤2: 选择最重要的head
  max_h = argmax_h(importance[h])

步骤3: 对该head的所有文本tokens求平均
  score[i] = (1/T) × Σ_t A[max_h, t, i]
```

#### 策略2: avg_all_heads
```
直接对所有heads和所有文本tokens求平均:
  score[i] = (1/(H×T)) × Σ_{h,t} A[h, t, i]
```

#### 策略3: weighted_combination
```
score[i] = α × [(1/T) × Σ_t A[max_h, t, i]]  +  (1-α) × [(1/(H×T)) × Σ_{h,t} A[h, t, i]]
```

## 为什么结果相同？

### 关键洞察

**当对T（35个文本tokens）求平均后，不同heads的差异被大幅抹平！**

### 数学证明

假设对于某个图像token i，不同heads的attention值为：
```
Head 0: A[0, :, i] = [a₀₁, a₀₂, ..., a₀₃₅]  →  mean = m₀
Head 1: A[1, :, i] = [a₁₁, a₁₂, ..., a₁₃₅]  →  mean = m₁
...
Head 31: A[31, :, i] = [a₃₁₁, a₃₁₂, ..., a₃₁₃₅]  →  mean = m₃₁
```

**问题核心**：当T=35时，根据中心极限定理，不同heads的平均值 m₀, m₁, ..., m₃₁ 会趋向于接近！

### 为什么会收敛？

1. **方差减小定理**
   ```
   Var(mean(X₁, X₂, ..., Xₜ)) = Var(X) / T
   ```
   当T=35时，方差减小到原来的 1/35

2. **期望收敛**
   如果不同heads看的是相同的文本→图像映射，它们的attention期望值相似：
   ```
   E[A[h, t, i]] ≈ E[A[h', t, i]]  （对于不同的heads h和h'）
   ```

3. **平均后差异消失**
   ```
   |m_h - m_h'| = |(1/T)Σ_t A[h,t,i] - (1/T)Σ_t A[h',t,i]|
                ≈ 0  （当T足够大）
   ```

## 实验证据

运行 `python explain_averaging_effect.py` 会显示：

**使用最后一个token:**
- max_head vs avg_all_heads 相关性: ~0.75
- Top-10 重叠: 4-6/10
- ✓ 策略有明显差异

**平均所有tokens:**
- max_head vs avg_all_heads 相关性: ~0.98
- Top-10 重叠: 9-10/10  
- ✗ 策略几乎相同

## 直观解释

想象一个例子：

**场景1: 只看最后一个token**
- Head 0 特别关注图像的左上角 → token [5, 12, 18]
- Head 1 特别关注图像的右下角 → token [450, 520, 560]
- 平均所有heads → 混合了左上和右下 → token [5, 12, 450, 520]
- **结果：max_head和avg_all_heads选择不同！**

**场景2: 平均35个文本tokens**
- Head 0: 35个tokens的平均关注度 → 所有区域都有一点
- Head 1: 35个tokens的平均关注度 → 也是所有区域都有一点
- 由于平均了很多tokens，头之间的"特殊偏好"被抹平了
- **结果：所有heads的平均分布都很相似，max_head≈avg_all_heads！**

## 类比

这就像：
- **最后token**: 问35个专家："现在最重要的是什么？" → 答案差异大
- **平均所有tokens**: 问35个专家："总体来说什么最重要？" → 答案都收敛到"总体平均"

## 结论

**问题根源**: 对35个文本tokens求平均是一个过强的平滑操作，它：
1. 消除了不同heads的特异性
2. 使所有heads的分布趋向于全局平均
3. 导致max_head选出的head和avg_all_heads几乎相同

**解决方案**: 使用最后一个token（当前生成位置的attention），保留heads的差异性。

**数学本质**: 
```
mean(mean(A, dim=text), dim=heads) ≈ mean(mean(A, dim=heads), dim=text)

当text很多时，不同的聚合顺序产生接近的结果！
```
