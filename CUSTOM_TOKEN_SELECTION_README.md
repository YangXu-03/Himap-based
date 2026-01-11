# 自定义Token选择模型使用说明

## 概述

本模块实现了一个自定义的视觉token选择和聚合机制，用于替代FastV的注意力mask方法。主要特点：

1. **相似度计算**：计算N个视觉token之间的余弦相似度矩阵
2. **质心选取**：支持两种算法
   - **FPS (Farthest Point Sampling)**：最远点采样，确保选中的token具有最大多样性
   - **ToMe (Token Merging)**：基于连通性的聚类选择
3. **Cross-Attention聚合**：将M个质心token作为Query，原始N个token作为Key和Value，通过加权聚合获取信息

## 文件结构

- `custom_token_selection.py`: 核心实现，包含CustomTokenSelection_LlamaModel类
- `llava_llama.py`: 添加了LlavaLlamaModel_CustomSelection和LlavaLlamaForCausalLM_CustomSelection类
- `builder.py`: 修改了load_pretrained_model函数，支持use_custom_selection参数
- `custom_selection_inference.py`: 示例推理脚本

## 配置参数

在模型配置中添加以下参数：

```python
config.use_custom_selection = True          # 启用自定义选择
config.custom_sys_length = 35               # 系统提示长度
config.custom_image_token_length = 576      # 图像token数量
config.custom_kept_tokens = 8               # 保留的token数量
config.custom_agg_layer = 2                 # 应用选择的层索引
config.custom_selection_method = 'fps'      # 选择算法: 'fps' 或 'tome'
config.custom_temperature = 0.1             # Cross-Attention温度参数
```

## 使用方法

### 方法1：直接使用推理脚本

```bash
python src/HiMAP/analysis/custom_selection_inference.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --image-folder data/scienceqa/images/test \
    --question-file data/scienceqa/himap-inference-MCQ.json \
    --custom-kept-tokens 8 \
    --custom-selection-method fps \
    --custom-agg-layer 2 \
    --custom-temperature 0.1 \
    --num-samples 50 \
    --output-file custom_fps_results.json
```

### 方法2：在代码中使用

```python
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

disable_torch_init()

# 加载模型（启用自定义选择）
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path="liuhaotian/llava-v1.5-7b",
    model_base=None,
    model_name="llava-v1.5-7b",
    use_custom_selection=True  # 关键参数
)

# 配置参数
model.config.use_custom_selection = True
model.config.custom_kept_tokens = 8
model.config.custom_selection_method = 'fps'  # 或 'tome'
model.config.custom_agg_layer = 2
model.config.custom_temperature = 0.1

# 重置模型以应用配置
model.model.reset_custom_selection()

# 正常使用模型进行推理
output = model.generate(input_ids, images=images, ...)
```

## 算法对比

### FPS (Farthest Point Sampling)
- **优点**：确保选中的token具有最大的多样性和覆盖范围
- **适用场景**：当图像内容多样、需要保留不同区域信息时
- **计算复杂度**：O(N*M)

### ToMe (Token Merging 简化版)
- **优点**：基于token的"重要性"（度）选择，能找到局部代表性token
- **适用场景**：当图像有明显的聚类结构时
- **计算复杂度**：O(N*M*K)，其中K是迭代次数（默认3）

## 实现细节

### 序列长度保持
为避免position_ids不匹配的问题，本实现采用了**保持序列长度不变**的策略：
- 将聚合后的M个token放在图像token区域的前M个位置
- 其余位置填充为零向量
- 这样保证了position_ids和attention_mask的兼容性

### Cross-Attention聚合
```python
attention_weights = softmax(cosine_similarity(Q, K) / temperature)
aggregated = attention_weights @ V
```
- 使用余弦相似度作为attention score
- temperature控制attention分布的锐度（越小越集中）
- 每个质心token会自动从相似的token中聚合信息

## 与FastV的区别

| 特性 | FastV | Custom Token Selection |
|------|-------|----------------------|
| 选择方式 | 基于注意力权重 | 基于相似度（FPS/ToMe） |
| 信息聚合 | 通过attention mask屏蔽 | 通过Cross-Attention显式聚合 |
| token保留 | 保留原始token，只mask | 替换为聚合后的新token |
| 参数依赖 | 依赖上一层的attention | 无参数，非学习式 |

## 实验建议

1. **kept_tokens数量**：建议从8, 16, 32, 64开始测试
2. **agg_layer选择**：建议测试层2-4，太早可能丢失信息，太晚效果有限
3. **temperature调整**：0.05-0.2范围内调整，观察聚合效果
4. **算法选择**：
   - 图像内容多样：使用FPS
   - 图像有明显结构：使用ToMe

## 调试和监控

模型会保存最近一次的token选择信息：
```python
# 获取选中的token位置（相对于图像token区域）
selected_positions = model.model.last_selected_positions

# 获取全局索引
global_indices = model.model.last_gen_kept_indices
```

## 性能优化建议

1. 使用FP16推理以提高速度
2. 对于大batch，考虑并行处理每个样本的token选择
3. 如果内存受限，可以减少kept_tokens数量

## 未来改进方向

1. 支持可学习的Cross-Attention参数
2. 添加基于语义的token选择策略
3. 支持动态调整kept_tokens数量
4. 实现多尺度token选择（不同层保留不同数量）
