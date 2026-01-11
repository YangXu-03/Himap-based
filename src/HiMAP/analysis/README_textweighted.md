# 文本加权的图像Token剪枝实验

## 实验目的
测试一种新的图像token剪枝策略：使用文本token的重要性来加权文本到图像的注意力，以此来选择最重要的图像token。

## 核心思想

传统的FastV剪枝策略：
- 直接使用最后一个token对图像token的注意力分数进行排序
- 选择top-k个注意力分数最高的图像token

新的文本加权剪枝策略：
1. **计算文本token的重要性**：分析文本token之间的注意力，计算每个文本token从其他文本token收到的总注意力（列求和），作为该文本token的重要性权重
2. **加权文本到图像的注意力**：使用文本token的重要性权重，对文本到图像的注意力进行加权求和
3. **基于加权分数剪枝**：根据加权后的图像token分数进行排序，选择top-k个图像token

## 数学公式

设：
- $A \in \mathbb{R}^{S \times S}$ 为注意力矩阵（已在heads上平均）
- $T_s, T_e$ 为文本token的起始和结束位置
- $I_s, I_e$ 为图像token的起始和结束位置

1. **文本重要性权重**：
   $$w_i = \frac{\sum_{j=T_s}^{T_e} A_{j,i}}{\sum_{i=T_s}^{T_e}\sum_{j=T_s}^{T_e} A_{j,i}}, \quad i \in [T_s, T_e]$$

2. **加权图像分数**：
   $$s_k = \sum_{i=T_s}^{T_e} w_i \cdot A_{i,k}, \quad k \in [I_s, I_e]$$

3. **选择top-K图像token**：
   $$\text{selected} = \text{topk}(\{s_k | k \in [I_s, I_e]\}, K)$$

## 优势

- **更全面的上下文考虑**：不仅考虑最后一个token，而是综合所有文本token的视角
- **文本重要性感知**：更重要的文本token（被其他文本token关注更多的）在选择图像token时有更大的话语权
- **更稳健的选择**：避免单一token视角可能带来的偏差

## 文件说明

1. **fastv_textweighted_experiment.py** - 主实验脚本
   - 实现文本加权的剪枝策略
   - 在指定层应用该策略
   - 分析不同保留token数下的准确率和秩

2. **fastv_textweighted_experiment.sh** - 运行脚本
   - 配置实验参数
   - 运行实验

## 参数说明

- `--pruning-layer`: 在哪一层应用文本加权剪枝策略（默认：2）
- `--num-samples`: 每个rank配置测试的样本数（默认：50）
- `--fast-v-sys-length`: 系统提示长度（默认：35）
- `--fast-v-image-token-length`: 图像token总数（默认：576）

## 实验配置

- **测试范围**：从 576 个token递减到 16 个token
- **测试步长**：每次减少 10 个token
- **秩计算**：只计算第 31 层（最后一层）的数值秩和有效秩
- **总测试点**：约 56 个不同的token保留配置

## 使用方法

```bash
# 运行实验
bash /root/nfs/code/HiMAP/src/HiMAP/analysis/fastv_textweighted_experiment.sh

# 或直接运行Python脚本
python /root/nfs/code/HiMAP/src/HiMAP/analysis/fastv_textweighted_experiment.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --image-folder /path/to/images \
    --question-file /path/to/questions.json \
    --num-samples 50 \
    --pruning-layer 2 \
    --single-pred-prompt
```

## 输出文件

- `fastv_textweighted_layer{N}_experiment.json` - 实验结果数据
- `fastv_textweighted_layer{N}_experiment.png` - 可视化图表

## 实验结果包含

- `tokens`: 保留的图像token数量
- `accuracies`: 对应的准确率
- `layer31_numeric_ranks`: Layer 31的数值秩
- `layer31_effective_ranks`: Layer 31的有效秩

## 对比实验

可以与原始FastV方法（`fastv_rank_experiment.py`）的结果进行对比，评估文本加权策略的效果。
