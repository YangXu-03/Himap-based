# Token Condensation for Vision-Language Models

## 概述

这是一个针对多模态大模型（LMM）的 Token 浓缩（Token Condensation）优化方案，用于替代传统的硬剪枝（Hard Pruning）方法。

### 核心思想

不再直接丢弃视觉 Token，而是：
1. **筛选代表性 Token**：选择 M 个最具代表性的 Token 作为"质心"
2. **信息聚合**：通过交叉注意力机制让质心聚合全局信息
3. **保留上下文**：避免丢失重要的背景信息

### 算法流程

```
输入: N 个 Token 的特征 [B, N, D]
输出: M 个浓缩后的 Token [B, M, D]

1. 相似度计算
   - L2 归一化特征
   - 计算余弦相似度矩阵 [B, N, N]

2. 代表性 Token 筛选（支持两种策略）
   策略 A - 最远点采样 (FPS):
     - 迭代选择距离已选点最远的 Token
     - 保证选中的 Token 在特征空间中分布均匀
   
   策略 B - 连通性密度筛选 (Connectivity):
     - 计算每个 Token 的"度"（与其他 Token 相似度之和）
     - 选择度最大的 M 个 Token
     - 速度更快，适合实时推理

3. 交叉注意力聚合
   - 质心 Token 作为 Query
   - 所有原始 Token 作为 Key 和 Value
   - 通过多头注意力聚合全局信息
```

## 文件结构

```
src/LLaVA/llava/model/language_model/
├── token_condensation.py          # Token Condensation 核心模块
│   ├── TokenCondensation          # 基础浓缩模块
│   └── TokenCondensationIntegrated # 集成包装器
├── himap_condensation.py          # 集成到 HiMAP 的完整模型
└── himap_configuration_llama.py   # 配置文件（需添加新参数）

src/HiMAP/analysis/
├── test_token_condensation.py     # 单元测试和性能基准
└── MME/
    ├── mme_condensation_experiment.py  # MME 对比实验
    └── mme_cutoff_experiment.py        # 原始实验（参考）
```

## 快速开始

### 1. 运行单元测试

测试 Token Condensation 模块的功能和性能：

```bash
cd /root/nfs/code/HiMAP
python src/HiMAP/analysis/test_token_condensation.py
```

测试内容包括：
- 相似度矩阵计算
- FPS 和 Connectivity 选择策略
- 交叉注意力聚合
- 完整流程测试
- 性能基准测试
- CUDA 支持测试

### 2. 运行 MME 对比实验

对比 Token Condensation 和 Hard Pruning 在 MME 基准上的表现：

```bash
# 运行所有方法（Hard Pruning + FPS + Connectivity）
python src/HiMAP/analysis/MME/mme_condensation_experiment.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --image-folder data/MME/images/test \
    --question-file data/MME/MME_test.json \
    --run-all \
    --target-tokens 100 \
    --start-layer 5 \
    --end-layer 15 \
    --output-file mme_condensation_results.json \
    --output-plot mme_condensation_comparison.png

# 或者只运行 Connectivity 策略（推荐，速度快）
python src/HiMAP/analysis/MME/mme_condensation_experiment.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --image-folder data/MME/images/test \
    --question-file data/MME/MME_test.json \
    --run-condensation-connectivity \
    --target-tokens 100 \
    --num-samples 100  # 快速测试，使用部分样本
```

### 3. 集成到现有模型

#### 步骤 1: 更新配置文件

在 `himap_configuration_llama.py` 中添加以下参数：

```python
# Token Condensation 参数
use_token_condensation = False,
condensation_strategy = 'connectivity',  # 'fps' or 'connectivity'
condensation_num_heads = 8,
```

#### 步骤 2: 使用新模型

```python
from llava.model.language_model.himap_condensation import HimapCondensation_LlamaModel

# 在模型初始化时启用 Token Condensation
config.use_token_condensation = True
config.condensation_strategy = 'connectivity'  # 或 'fps'
config.condensation_num_heads = 8

model = HimapCondensation_LlamaModel(config)
```

#### 步骤 3: 设置浓缩参数

```python
# 设置在哪些层进行浓缩
config.use_hmap_v = True
config.hmap_v_attn_txt_layer = 5    # 第一次浓缩层
config.hmap_v_attn_txt_rank = 100   # 浓缩到 100 个 Token
config.hmap_v_attn_img_layer = 10   # 第二次浓缩层
config.hmap_v_attn_img_rank = 50    # 进一步浓缩到 50 个
```

## 参数说明

### Token Condensation 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_token_condensation` | bool | False | 是否启用 Token Condensation |
| `condensation_strategy` | str | 'connectivity' | 选择策略：'fps' 或 'connectivity' |
| `condensation_num_heads` | int | 8 | 交叉注意力的头数 |

### HiMAP 参数（兼容）

| 参数 | 类型 | 说明 |
|------|------|------|
| `hmap_v_sys_length` | int | 系统提示的 Token 数量 |
| `hmap_v_img_length` | int | 原始视觉 Token 数量（如 576） |
| `hmap_v_attn_txt_layer` | int | 第一次浓缩/剪枝的层索引 |
| `hmap_v_attn_txt_rank` | int | 第一次保留的 Token 数量 |
| `hmap_v_attn_img_layer` | int | 第二次浓缩/剪枝的层索引 |
| `hmap_v_attn_img_rank` | int | 第二次保留的 Token 数量 |

## 性能对比

### 策略对比

| 策略 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **Hard Pruning** | 实现简单，速度最快 | 丢失信息，性能下降明显 | 极端压缩需求 |
| **Connectivity** | 速度快，效果好，易于并行 | 可能偏向局部中心 | 推荐用于实时推理 |
| **FPS** | 覆盖更均匀，理论上更优 | 计算开销较大（迭代选择） | 离线分析，追求极致性能 |

### 预期性能提升

基于理论分析，Token Condensation 相比 Hard Pruning：

1. **保留上下文**：通过交叉注意力聚合全局信息
2. **更好的语义表示**：质心选择保证覆盖主要视觉区域
3. **更高的准确率**：预期在 MME 等基准上提升 2-5%

## 技术细节

### 交叉注意力机制

```python
# 质心作为 Query，吸收全局信息
Q = W_q @ centroids          # [B, M, D]
K = W_k @ all_tokens         # [B, N, D]
V = W_v @ all_tokens         # [B, N, D]

# 多头注意力
Attention(Q, K, V) = Softmax(QK^T / √d) @ V
```

### 相似度矩阵计算

```python
# L2 归一化保证数值稳定性
features_norm = F.normalize(features, p=2, dim=-1)

# 余弦相似度
similarity = features_norm @ features_norm.T
```

### FPS 算法伪代码

```
selected = [random_initial_token]
min_distances = infinity

for i in range(M - 1):
    # 更新每个点到已选集合的最小距离
    for each unselected token:
        dist = distance(token, selected)
        min_distances[token] = min(min_distances[token], dist)
    
    # 选择距离最大的点
    farthest = argmax(min_distances)
    selected.append(farthest)
```

## 实验结果

运行实验后会生成：

1. **JSON 结果文件** (`mme_condensation_results.json`)
   - 包含所有方法在不同层的详细分数
   - Perception/Cognition 分数分解
   - 子任务分数

2. **对比可视化图** (`mme_condensation_comparison.png`)
   - 总分对比曲线
   - Perception/Cognition 分数对比
   - 性能提升百分比

## 故障排除

### 问题 1: CUDA Out of Memory

**解决方案**：
- 减少 `condensation_num_heads`（如从 32 降到 8）
- 使用 `connectivity` 策略而不是 `fps`
- 减少 batch size

### 问题 2: 导入错误

**解决方案**：
```bash
# 确保路径正确
export PYTHONPATH=/root/nfs/code/HiMAP/src/LLaVA:$PYTHONPATH
```

### 问题 3: 模型配置不兼容

**解决方案**：
- 检查 `himap_configuration_llama.py` 是否添加了新参数
- 使用 `reset_hmapv()` 方法重置配置

## 下一步工作

1. **性能优化**
   - [ ] 使用 FlashAttention 加速交叉注意力
   - [ ] GPU 内核优化 FPS 算法
   - [ ] 量化支持（FP16/INT8）

2. **算法改进**
   - [ ] 自适应选择质心数量
   - [ ] 多层级浓缩策略
   - [ ] 与知识蒸馏结合

3. **更多基准测试**
   - [ ] ScienceQA
   - [ ] TextVQA
   - [ ] COCO Caption

## 参考文献

- HiMAP: 原始的多模态注意力剪枝方法
- FastV: 视觉 Token 剪枝加速
- Token Merging: Token 合并策略
- Perceiver: 跨模态注意力机制

## 联系方式

如有问题或建议，请提交 Issue 或 Pull Request。
