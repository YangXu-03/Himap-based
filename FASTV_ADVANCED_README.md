# FastV Advanced - 升级版 Token 选择策略

## 概述

`fastv_advanced.py` 是 FastV 的升级版实现，提供了三种可选的 token 剪枝排序策略，以更灵活地选择保留哪些视觉 tokens。

## 三种 Token 选择策略

### 1. **max_head** - 最大注意力头策略
选取当前层中文本到视觉注意力值最大的注意力头，然后从该头的注意力分布中选择 top-k 个 tokens。

**适用场景**：
- 当某个特定注意力头特别关注视觉信息时
- 希望保留最显著的视觉特征
- 单一头的决策能力强

**实现细节**：
```python
# 1. 计算每个注意力头对图像 tokens 的总注意力
head_importance = image_attention.sum(dim=-1)  # [batch, num_heads]

# 2. 找到注意力值最大的头
max_head_idx = head_importance.argmax(dim=1)

# 3. 从该头中选择 top-k tokens
top_indices = max_head_attention.topk(attention_rank).indices
```

### 2. **avg_all_heads** - 全头平均策略（原始 FastV）
对所有注意力头的注意力值取平均，然后选择平均注意力最高的 top-k 个 tokens。

**适用场景**：
- 需要综合考虑所有注意力头的信息
- 避免单一头的偏差
- 原始 FastV 方法，经过验证的基线

**实现细节**：
```python
# 1. 对所有头求平均
avg_attention = torch.mean(attention_weights, dim=1)

# 2. 获取最后一个 token 对图像 tokens 的注意力
last_token_image_attention = avg_attention[0, -1, sys_length:sys_length+image_token_length]

# 3. 选择 top-k tokens
top_indices = last_token_image_attention.topk(attention_rank).indices
```

### 3. **weighted_combination** - 加权组合策略
使用加权组合的方式：`综合得分 = max_head_attention × α + avg_other_heads_attention × (1 - α)`

**适用场景**：
- 既想利用最强注意力头的信息，又想考虑其他头的贡献
- 需要平衡单头决策和多头共识
- 可以通过调整 α 参数控制两者的权重

**实现细节**：
```python
# 1. 找到注意力最大的头
max_head_idx = head_importance.argmax(dim=1)

# 2. 计算其他头的平均注意力
other_heads_attention = image_attention[:, other_heads_mask, :].mean(dim=1)

# 3. 加权组合
combined_attention = alpha * max_head_attention + (1 - alpha) * other_heads_attention

# 4. 选择 top-k tokens
top_indices = combined_attention.topk(attention_rank).indices
```

## 配置参数

在 `himap_configuration_llama.py` 中新增了两个配置参数：

```python
# Token 选择方法：'max_head', 'avg_all_heads', 'weighted_combination'
fast_v_token_selection_method = 'avg_all_heads'

# 加权组合策略的 alpha 参数（仅在 weighted_combination 时使用）
fast_v_weighted_alpha = 0.5
```

## 使用方法

### 方法 1: 在配置文件中设置

```python
from llava.model.language_model.himap_configuration_llama import LlamaConfig

config = LlamaConfig(
    # 基础 FastV 参数
    use_fast_v=True,
    fast_v_sys_length=36,
    fast_v_image_token_length=576,
    fast_v_attention_rank=288,
    fast_v_agg_layer=2,
    
    # 新增参数
    fast_v_token_selection_method='weighted_combination',  # 选择策略
    fast_v_weighted_alpha=0.7,  # alpha 权重
)
```

### 方法 2: 动态修改

```python
from llava.model.language_model.fastv_advanced import FastvAdvanced_LlamaModel

# 创建模型
model = FastvAdvanced_LlamaModel(config)

# 动态切换策略
model.token_selection_method = 'max_head'
model.weighted_alpha = 0.6
```

### 方法 3: 使用示例脚本

```python
import torch
from transformers import AutoTokenizer
from llava.model.language_model.fastv_advanced import FastvAdvanced_LlamaModel
from llava.model.language_model.himap_configuration_llama import LlamaConfig

# 1. 创建配置
config = LlamaConfig(
    use_fast_v=True,
    fast_v_sys_length=36,
    fast_v_image_token_length=576,
    fast_v_attention_rank=288,
    fast_v_agg_layer=2,
    fast_v_token_selection_method='max_head',
)

# 2. 创建模型
model = FastvAdvanced_LlamaModel(config)

# 3. 进行推理
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    output_attentions=True,  # 需要输出注意力
)

# 4. 查看选择的 tokens
print(f"选择方法: {model.last_selection_metadata['method']}")
print(f"保留的 token 索引: {model.last_gen_kept_indices}")
if 'max_head_idx' in model.last_selection_metadata:
    print(f"最大注意力头索引: {model.last_selection_metadata['max_head_idx']}")
```

## 三种策略的比较

| 策略 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **max_head** | 保留最显著特征；决策清晰 | 可能忽略其他头的信息；对噪声敏感 | 单头决策能力强的模型 |
| **avg_all_heads** | 稳定；综合所有头的信息 | 可能平滑掉重要特征 | 通用场景；基线方法 |
| **weighted_combination** | 平衡单头和多头信息；可调节 | 需要调参；计算复杂度略高 | 需要精细控制的场景 |

## 实验建议

### 1. 探索最佳策略
建议在您的数据集上测试三种策略：

```bash
# 测试 max_head
python eval.py --fast_v_token_selection_method max_head

# 测试 avg_all_heads
python eval.py --fast_v_token_selection_method avg_all_heads

# 测试 weighted_combination
python eval.py --fast_v_token_selection_method weighted_combination --fast_v_weighted_alpha 0.7
```

### 2. 调整 alpha 参数
对于 `weighted_combination` 策略，建议尝试不同的 alpha 值：

```python
alpha_values = [0.3, 0.5, 0.7, 0.9]
for alpha in alpha_values:
    model.weighted_alpha = alpha
    # 评估性能...
```

### 3. 分析选择的 tokens
使用内置的元数据跟踪功能：

```python
# 获取选择元数据
metadata = model.last_selection_metadata

print(f"选择方法: {metadata['method']}")
if 'max_head_idx' in metadata:
    print(f"最大注意力头: {metadata['max_head_idx']}")
    print(f"各头重要性: {metadata['head_importance']}")
```

## 性能考虑

- **max_head**: 最快（只需要找到最大头）
- **avg_all_heads**: 中等（需要对所有头求平均）
- **weighted_combination**: 稍慢（需要计算最大头和其他头的平均）

但三种方法的计算开销差异很小，主要差异在于 token 选择的质量。

## 调试和可视化

模型保存了以下信息供外部访问：

```python
# 生成的注意力 mask（bool tensor）
model.last_gen_attention_mask  # [batch_size, seq_length]

# 保留的 token 索引（numpy array）
model.last_gen_kept_indices  # [num_kept_tokens]

# 选择元数据（dict）
model.last_selection_metadata  # {'method': ..., 'max_head_idx': ..., ...}
```

## 与原始 FastV 的兼容性

`FastvAdvanced_LlamaModel` 完全向后兼容原始 FastV：
- 默认使用 `avg_all_heads` 策略，与原始 FastV 行为一致
- 支持所有原始 FastV 的参数和功能
- 可以无缝替换原始 `Fastv_LlamaModel`

## 后续改进方向

1. **自适应策略选择**：根据任务类型自动选择最佳策略
2. **层级策略**：不同层使用不同的选择策略
3. **动态 alpha**：根据注意力分布动态调整 alpha 值
4. **多样性约束**：在选择 tokens 时考虑多样性

## 参考文献

- 原始 FastV 方法
- 注意力机制在视觉语言模型中的应用
- Token pruning 技术

## 联系和贡献

如有问题或建议，欢迎提 issue 或 PR！
