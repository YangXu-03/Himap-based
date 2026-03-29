# FastV Advanced MME 测试说明

本文档说明如何测试 FastV Advanced 的多种 token 选择策略在 MME 数据集上的表现。

## 文件说明

### 1. 评估脚本
- **eval_mme_fastv_advanced.sh**: 主测试脚本，测试所有 FastV Advanced 方法
- **eval_mme_fastv_advanced.py**: Python 评估脚本，支持多种 token 选择方法

### 2. 可视化脚本
- **visualize_mme_fastv_results.py**: 结果可视化脚本，生成对比图表

## Token 选择策略

FastV Advanced 实现了以下 5 种 token 选择策略：

1. **baseline**: 不使用任何剪枝（参考基准）
2. **max_head**: 使用具有最大 text-to-vision attention 的 head 进行 token 选择
3. **avg_all_heads**: 对所有 head 的 attention 取平均（原始 FastV 方法）
4. **text_weighted**: 使用 text-to-text attention 加权 text-to-vision attention
5. **text_weighted_max_head**: 结合 text weighting 和 max head 选择
6. **weighted_combination**: max head 和其他 head 平均的加权组合（α=0.7）

## 使用方法

### 步骤 1: 运行评估

```bash
cd /root/nfs/code/HiMAP
chmod +x src/HiMAP/inference/eval_mme_fastv_advanced.sh
./src/HiMAP/inference/eval_mme_fastv_advanced.sh
```

**注意**: 
- 脚本默认使用 GPU 2 (`CUDA_VISIBLE_DEVICES=2`)
- 模型路径: `/root/nfs/model/llava-v1.5-7b`
- 数据路径: `/root/nfs/code/HiMAP/data/MME/`

如需修改配置，请编辑 `eval_mme_fastv_advanced.sh` 中的变量。

### 步骤 2: 生成可视化

评估完成后，运行可视化脚本：

```bash
python ./src/HiMAP/inference/visualize_mme_fastv_results.py
```

## 输出文件

### 评估结果
- `mme_results_baseline.json`
- `mme_results_fastv_max_head.json`
- `mme_results_fastv_avg_all_heads.json`
- `mme_results_fastv_text_weighted.json`
- `mme_results_fastv_text_weighted_max_head.json`
- `mme_results_fastv_weighted_combination_alpha0.7.json`

### 可视化图表（保存在 `mme_fastv_visualizations/` 目录）
1. **total_scores_comparison.png**: 总分对比柱状图
   - 左图: 各方法的总分对比
   - 右图: Perception vs Cognition 分数分解

2. **category_scores_comparison.png**: 各个子任务详细分数对比
   - 14个子任务的分组柱状图
   - 包含: existence, count, position, color, posters, celebrity, scene, landmark, artwork, OCR, commonsense_reasoning, numerical_calculation, text_translation, code_reasoning

3. **performance_heatmap.png**: 性能热力图
   - 展示各方法在不同类别上的分数分布
   - 颜色深浅表示分数高低

4. **relative_performance.png**: 相对于 baseline 的性能变化
   - 显示各方法与 baseline 的分数差异
   - 绿色表示提升，红色表示下降

5. **summary_table.png**: 汇总表格
   - 包含所有方法的总分、Perception、Cognition 和平均延迟

## 结果文件格式

每个结果 JSON 文件包含：

```json
{
  "total_score": 总分,
  "perception_score": 感知任务总分,
  "cognition_score": 认知任务总分,
  "category_scores": {
    "category_name": {
      "accuracy": 准确率,
      "accuracy_plus": 配对准确率,
      "score": 分数 (accuracy + accuracy_plus),
      "num_samples": 样本数,
      "num_pairs": 配对数
    },
    ...
  },
  "total_samples": 总样本数,
  "avg_latency": 平均推理延迟,
  "model_config": 模型配置信息,
  "predictions": 所有预测结果
}
```

## 单独运行某个方法

如果只想测试某个特定方法，可以直接运行 Python 脚本：

```bash
# 测试 baseline
python ./src/HiMAP/inference/eval_mme_fastv_advanced.py \
    --model-path /root/nfs/model/llava-v1.5-7b \
    --question-file /root/nfs/code/HiMAP/data/MME/llava_mme.json \
    --image-folder /root/nfs/code/HiMAP/data/MME/MME_Benchmark_release_version \
    --single-pred-prompt \
    --output-file mme_results_baseline.json

# 测试 max_head 策略
python ./src/HiMAP/inference/eval_mme_fastv_advanced.py \
    --model-path /root/nfs/model/llava-v1.5-7b \
    --question-file /root/nfs/code/HiMAP/data/MME/llava_mme.json \
    --image-folder /root/nfs/code/HiMAP/data/MME/MME_Benchmark_release_version \
    --single-pred-prompt \
    --use-fast-v \
    --fast-v-sys-length 35 \
    --fast-v-image-token-length 576 \
    --fast-v-attention-rank 288 \
    --fast-v-agg-layer 12 \
    --fast-v-token-selection-method max_head \
    --output-file mme_results_fastv_max_head.json
```

## 参数说明

### FastV Advanced 参数
- `--use-fast-v`: 启用 FastV
- `--fast-v-sys-length`: 系统提示词长度（默认: 35）
- `--fast-v-image-token-length`: 图像 token 总数（默认: 576）
- `--fast-v-attention-rank`: 保留的 token 数量（默认: 288，即保留50%）
- `--fast-v-agg-layer`: 开始剪枝的层数（默认: 12）
- `--fast-v-token-selection-method`: token 选择策略
  - `max_head`: 最大注意力头策略
  - `avg_all_heads`: 平均所有头（原始 FastV）
  - `text_weighted`: 文本加权策略
  - `text_weighted_max_head`: 文本加权 + 最大头
  - `weighted_combination`: 加权组合
- `--fast-v-weighted-alpha`: weighted_combination 的 alpha 参数（默认: 0.5）

### 其他参数
- `--model-path`: 模型路径
- `--question-file`: 问题文件路径
- `--image-folder`: 图像文件夹路径
- `--output-file`: 输出结果文件路径
- `--temperature`: 采样温度（默认: 0.0，即贪婪解码）

## 故障排除

### 问题 1: CUDA 内存不足
如果遇到 CUDA OOM 错误，可以：
1. 减少 `--fast-v-attention-rank` 的值（保留更少的 token）
2. 使用更大显存的 GPU
3. 减少 batch size（当前为 1，已经是最小）

### 问题 2: 图像文件找不到
检查以下路径是否正确：
- 问题文件: `/root/nfs/code/HiMAP/data/MME/llava_mme.json`
- 图像文件夹: `/root/nfs/code/HiMAP/data/MME/MME_Benchmark_release_version`

### 问题 3: 可视化中文显示乱码
如果中文显示为方框，尝试：
1. 安装中文字体: `sudo apt-get install fonts-wqy-zenhei`
2. 或者编辑 `visualize_mme_fastv_results.py`，将中文改为英文

## 预期运行时间

根据 MME 数据集大小（约 2,374 个样本）和 GPU 性能：
- 单个方法评估时间: 约 30-60 分钟
- 所有方法（6个）: 约 3-6 小时
- 可视化生成: < 1 分钟

## 性能评估指标

MME 评估指标说明：
1. **Accuracy**: 简单准确率（正确预测数 / 总样本数）
2. **Accuracy+**: 配对准确率（一个图像的所有问题都正确才算正确）
3. **Score**: Accuracy + Accuracy+（最终分数）
4. **Perception Score**: 所有感知任务分数之和
5. **Cognition Score**: 所有认知任务分数之和
6. **Total Score**: Perception Score + Cognition Score

## 参考

- FastV 论文: [FastV: An Image is Worth 1/2 Tokens After Layer 2](https://arxiv.org/abs/2403.06764)
- MME 数据集: [MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation)
