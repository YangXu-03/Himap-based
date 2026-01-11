# FastV Layer Cutoff Experiment

## 实验目的

测试在ScienceQA数据集上，使用FastV方法时，从模型的第2层到最后一层分别完全剪除图像token对准确率的影响。

## 实验设计

- **方法**：FastV（Fast Vision Token Pruning）
- **数据集**：ScienceQA
- **实验变量**：在不同层（第2层到第32层）开始完全剪除图像token
- **固定参数**：
  - `fast_v_attention_rank = 0`（完全剪除所有图像token）
  - `fast_v_sys_length = 35`（系统提示长度）
  - `fast_v_image_token_length = 576`（图像token总数）

## 文件说明

1. **fastv_layer_cutoff_experiment.py** - 主实验脚本
   - 遍历从第2层到最后一层的每一层
   - 在每一层设置完全剪除图像token（rank=0）
   - 记录每个配置下的准确率
   - 生成结果JSON和可视化图表

2. **fastv_layer_cutoff_experiment.sh** - 运行脚本
   - 配置模型路径、数据路径等参数
   - 调用Python实验脚本

3. **修改的文件**：
   - `src/LLaVA/llava/model/language_model/fastv.py` - 添加了对rank=0的边界情况处理

## 使用方法

### 1. 快速测试（100个样本）

```bash
cd /root/nfs/code/HiMAP/src/HiMAP/analysis
chmod +x fastv_layer_cutoff_experiment.sh
./fastv_layer_cutoff_experiment.sh
```

### 2. 完整测试（所有样本）

修改 `fastv_layer_cutoff_experiment.sh` 中的参数：

```bash
NUM_SAMPLES=-1  # 使用所有样本
```

### 3. 自定义运行

```bash
python fastv_layer_cutoff_experiment.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --image-folder /path/to/scienceqa/images/test \
    --question-file /path/to/llava_test_CQM-A.json \
    --conv-mode vicuna_v1 \
    --single-pred-prompt \
    --num-samples 100 \
    --fast-v-sys-length 35 \
    --fast-v-image-token-length 576 \
    --output-file results.json \
    --output-plot plot.png
```

## 输出结果

### 1. JSON结果文件（fastv_layer_cutoff_results.json）

```json
{
  "layers": [2, 3, 4, ..., 32],
  "accuracies": [0.75, 0.73, 0.70, ...],
  "correct_counts": [75, 73, 70, ...],
  "total_samples": 100
}
```

### 2. 可视化图表

- **fastv_layer_cutoff_plot.png** - 准确率随层数变化曲线
- **fastv_layer_cutoff_plot_relative_drop.png** - 相对性能下降曲线（以第2层为基准）

## 预期结果

根据FastV的设计原理，预期观察到：

1. **早期层（2-10层）**：完全剪除图像token会显著降低准确率，因为这些层正在处理视觉信息
2. **中间层（11-20层）**：准确率下降趋势可能放缓，表明视觉信息逐渐融合到文本表示中
3. **后期层（21-32层）**：准确率下降可能进一步放缓或趋于稳定，表明高层更依赖抽象语义而非原始视觉token

## 代码修改说明

为了支持完全剪除图像token（rank=0）的实验，修改了`fastv.py`：

### 修改1：处理idx != 0时的rank=0情况

```python
# 在生成attention mask时，添加边界检查
if ATTENTION_RANK > 0:
    top_attention_rank_index = last_layer_attention_avg_last_tok_image.topk(ATTENTION_RANK).indices + SYS_LENGTH
    gen_attention_mask[:, top_attention_rank_index] = True
# 如果ATTENTION_RANK=0，则保持所有图像token为False（完全剪除）
```

### 修改2：处理idx==0时的rank=0情况

```python
if ATTENTION_RANK > 0:
    rand_image_attention_mask = [1]*ATTENTION_RANK + [0]*(IMAGE_TOKEN_LENGTH-ATTENTION_RANK)
    random.shuffle(rand_image_attention_mask)
    gen_attention_mask[:, SYS_LENGTH:SYS_LENGTH+IMAGE_TOKEN_LENGTH] = torch.tensor(rand_image_attention_mask, ...)
else:
    # Complete removal
    gen_attention_mask[:, SYS_LENGTH:SYS_LENGTH+IMAGE_TOKEN_LENGTH] = False
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model-path` | liuhaotian/llava-v1.5-7b | 模型路径 |
| `--image-folder` | 必需 | ScienceQA图像文件夹路径 |
| `--question-file` | 必需 | 问题JSON文件路径 |
| `--conv-mode` | vicuna_v1 | 对话模板 |
| `--single-pred-prompt` | False | 是否添加单答案提示 |
| `--num-samples` | -1 | 测试样本数（-1表示全部） |
| `--fast-v-sys-length` | 35 | 系统提示长度 |
| `--fast-v-image-token-length` | 576 | 图像token数量 |
| `--output-file` | fastv_layer_cutoff_results.json | 输出JSON文件 |
| `--output-plot` | fastv_layer_cutoff_plot.png | 输出图表文件 |

## 常见问题

### Q1: 为什么从第2层开始测试而不是第1层？
A: 第1层通常是embedding层，直接处理输入token，从第2层开始测试更能反映transformer层的视觉信息处理过程。

### Q2: 实验需要多长时间？
A: 对于7B模型，100个样本大约需要30-60分钟（取决于GPU性能）。全部样本（~2000个）可能需要10-20小时。

### Q3: 可以在多GPU上运行吗？
A: 当前版本是单GPU实现。如需多GPU，可以修改代码添加`torch.nn.DataParallel`或使用分布式推理。

## 相关文献

- FastV: FastV: Fast Vision Token Pruning for Vision-Language Models
- LLaVA: Visual Instruction Tuning
- ScienceQA: Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering
