# ScienceQA 注意力头分析

本目录包含用于分析 LLaVA 模型在 ScienceQA 数据集上注意力头行为的脚本。

## 功能

1. **统计每个样本每一层的最大注意力头序号**
2. **绘制所有样本中每一层最大注意力头的分布直方图**
3. **生成详细的可视化和统计报告**

## 文件说明

- `analyze_scienceqa_attention_heads.py` - 主分析脚本，处理 ScienceQA 数据集并提取注意力头信息
- `visualize_scienceqa_attention_heads.py` - 可视化脚本，生成详细的图表和报告
- `run_scienceqa_attention_head_analysis.sh` - 便捷的运行脚本

## 使用方法

### 1. 运行主分析脚本

使用提供的 shell 脚本：

```bash
bash run_scienceqa_attention_head_analysis.sh
```

或者直接运行 Python 脚本：

```bash
python -m src.HiMAP.analysis.ScienceQA.analyze_scienceqa_attention_heads \
    --model-path "liuhaotian/llava-v1.5-7b" \
    --image-folder "./data/scienceqa/images/test" \
    --question-file "./data/scienceqa/llava_test_QCM-LEA.json" \
    --output-dir "./scienceqa_attention_head_analysis" \
    --num-samples 100 \
    --conv-mode "vicuna_v1"
```

### 参数说明

- `--model-path`: 模型路径（本地或 HuggingFace）
- `--image-folder`: 图像文件夹路径
- `--question-file`: 问题 JSON 文件路径
- `--output-dir`: 输出目录
- `--num-samples`: 处理的样本数量（-1 表示全部）
- `--conv-mode`: 对话模式（默认 "vicuna_v1"）

### 2. 生成额外的可视化

运行主分析脚本后，可以使用可视化脚本生成更详细的图表：

```bash
python -m src.HiMAP.analysis.ScienceQA.visualize_scienceqa_attention_heads \
    --input-dir "./scienceqa_attention_head_analysis"
```

## 输出文件

运行后会在输出目录生成以下文件：

### 数据文件

- `attention_head_records.pt` - PyTorch 格式的样本级记录
  - 包含每个样本的 ID、图像文件名和每一层的最大注意力头序号
  
- `statistics.json` - JSON 格式的统计数据
  - 包含总样本数、层数、头数以及每层每个头的计数

### 可视化文件

1. **max_attention_head_distribution.png** - 每一层的注意力头分布直方图
   - 子图形式展示每一层的头序号分布
   - 红色高亮显示最频繁的头

2. **max_attention_head_heatmap.png** - 跨层的注意力头分布热图
   - Y 轴：层索引
   - X 轴：注意力头索引
   - 颜色：该头在该层成为最大注意力头的次数

3. **top_k_heads_per_layer.png** - 每层 Top-K 最频繁头的可视化
   - 显示每层中最常出现的前 K 个注意力头及其计数

4. **head_frequency_total.png** - 注意力头在所有层中的总频率
   - 横跨所有层统计每个头成为最大注意力头的总次数
   - 红色标记 Top-5 最频繁的头

5. **layer_diversity.png** - 层级多样性分析
   - 左图：每层有多少不同的头成为过最大值
   - 右图：每层的熵（分布均匀程度）

### 报告文件

- `analysis_report.txt` - 文本格式的详细分析报告
  - 每层的统计信息
  - 最频繁的头
  - 多样性指标

## 数据格式

### attention_head_records.pt 格式

```python
[
    {
        'sample_id': '1',
        'image_file': 'test/1.png',
        'max_head_per_layer': [2, 5, 1, 3, ...]  # 长度为 num_layers
    },
    ...
]
```

### statistics.json 格式

```json
{
    "num_samples": 100,
    "num_layers": 32,
    "num_heads": 32,
    "layer_head_counts": {
        "0": {
            "0": 5,
            "1": 10,
            "2": 15,
            ...
        },
        ...
    }
}
```

## 示例输出

```
Model has 32 layers and 32 attention heads per layer
Processing 100 samples...
Processing samples: 100%|████████| 100/100 [02:30<00:00,  1.50s/it]

Saved sample records to ./scienceqa_attention_head_analysis/attention_head_records.pt
Saved statistics to ./scienceqa_attention_head_analysis/statistics.json

Generating histograms...
Saved histogram to ./scienceqa_attention_head_analysis/max_attention_head_distribution.png
Saved heatmap to ./scienceqa_attention_head_analysis/max_attention_head_heatmap.png

Analysis complete!
Results saved to: ./scienceqa_attention_head_analysis
```

## 注意事项

1. 需要足够的 GPU 内存来加载模型和处理样本
2. 处理大量样本可能需要较长时间
3. 可以通过 `--num-samples` 参数先用少量样本测试
4. 确保已安装所有依赖：`torch`, `matplotlib`, `numpy`, `PIL`, `tqdm`

## 后续分析

可以使用保存的 `attention_head_records.pt` 文件进行进一步分析：

```python
import torch

# 加载记录
records = torch.load('scienceqa_attention_head_analysis/attention_head_records.pt')

# 分析特定样本
sample = records[0]
print(f"Sample ID: {sample['sample_id']}")
print(f"Max heads per layer: {sample['max_head_per_layer']}")

# 分析特定层
layer_5_heads = [r['max_head_per_layer'][5] for r in records]
print(f"Layer 5 head distribution: {layer_5_heads}")
```
