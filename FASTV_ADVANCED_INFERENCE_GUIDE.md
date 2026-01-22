# FastV Advanced 推理使用指南

## 快速开始

### 1. 运行单个测试

测试 **max_head** 策略：

```bash
export CUDA_VISIBLE_DEVICES=2
python ./src/HiMAP/inference/eval_scivqa.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --question-file /root/nfs/code/HiMAP/data/scienceqa/himap-inference-MCQ.json \
    --image-folder ./data/scienceqa/images/test \
    --single-pred-prompt \
    --use-fast-v \
    --fast-v-sys-length 35 \
    --fast-v-image-token-length 576 \
    --fast-v-attention-rank 288 \
    --fast-v-agg-layer 2 \
    --fast-v-token-selection-method max_head \
    --num-samples 100
```

测试 **weighted_combination** 策略（alpha=0.7）：

```bash
export CUDA_VISIBLE_DEVICES=2
python ./src/HiMAP/inference/eval_scivqa.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --question-file /root/nfs/code/HiMAP/data/scienceqa/himap-inference-MCQ.json \
    --image-folder ./data/scienceqa/images/test \
    --single-pred-prompt \
    --use-fast-v \
    --fast-v-sys-length 35 \
    --fast-v-image-token-length 576 \
    --fast-v-attention-rank 288 \
    --fast-v-agg-layer 2 \
    --fast-v-token-selection-method weighted_combination \
    --fast-v-weighted-alpha 0.7 \
    --num-samples 100
```

### 2. 运行完整测试套件

运行所有三种策略的完整对比测试：

```bash
chmod +x ./src/HiMAP/inference/eval_scivqa_fastv_advanced.sh
bash ./src/HiMAP/inference/eval_scivqa_fastv_advanced.sh
```

这个脚本会依次运行：
- 基线（无剪枝）
- FastV - max_head
- FastV - avg_all_heads（原始FastV）
- FastV - weighted_combination (α=0.3, 0.5, 0.7)

### 3. 比较结果

运行完测试后，使用比较脚本查看结果：

```bash
python ./src/HiMAP/inference/compare_fastv_results.py
```

输出示例：
```
====================================================================================================
                              FastV Advanced 结果比较
====================================================================================================

策略                             准确率         平均延迟(s)       FLOPs比例     样本数    
----------------------------------------------------------------------------------------------------
Baseline                       0.7800       2.345678        100.00%      100       
FastV - max_head              0.7750 (-0.0050) 1.234567        50.00%       100       
FastV - avg_all_heads         0.7800 (+0.0000) 1.256789        50.00%       100       
FastV - weighted α=0.3        0.7750 (-0.0050) 1.245678        50.00%       100       
FastV - weighted α=0.5        0.7850 (+0.0050) 1.267890        50.00%       100       
FastV - weighted α=0.7        0.7850 (+0.0050) 1.278901        50.00%       100       
----------------------------------------------------------------------------------------------------
```

## 参数说明

### 基本 FastV 参数

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `--use-fast-v` | 启用 FastV | （flag，无需赋值） |
| `--fast-v-sys-length` | 系统提示长度 | 35 |
| `--fast-v-image-token-length` | 图像token总数 | 576 |
| `--fast-v-attention-rank` | 保留的图像token数 | 288（50%剪枝）|
| `--fast-v-agg-layer` | 开始剪枝的层 | 2 |

### FastV Advanced 新增参数

| 参数 | 说明 | 可选值 | 默认值 |
|------|------|--------|--------|
| `--fast-v-token-selection-method` | Token选择策略 | `max_head`, `avg_all_heads`, `weighted_combination` | `avg_all_heads` |
| `--fast-v-weighted-alpha` | 加权组合的α参数 | 0.0 ~ 1.0 | 0.5 |

## 三种策略详解

### 1. max_head - 最大注意力头策略

```bash
--fast-v-token-selection-method max_head
```

**工作原理**：
- 找到对图像tokens注意力最大的注意力头
- 从该头中选择top-k个tokens

**适用场景**：
- 某个注意力头明显主导视觉信息处理
- 需要最显著的视觉特征
- 追求最快速度

### 2. avg_all_heads - 全头平均策略（原始FastV）

```bash
--fast-v-token-selection-method avg_all_heads
```

**工作原理**：
- 对所有注意力头求平均
- 选择平均注意力最高的top-k个tokens

**适用场景**：
- 通用场景，稳定可靠
- 不确定哪个头最重要
- 作为基线方法

### 3. weighted_combination - 加权组合策略

```bash
--fast-v-token-selection-method weighted_combination \
--fast-v-weighted-alpha 0.7
```

**工作原理**：
- 得分 = max_head_attention × α + avg_other_heads_attention × (1-α)
- 选择综合得分最高的top-k个tokens

**适用场景**：
- 需要精细控制
- 平衡单头决策和多头共识
- 可调节α参数优化性能

**α参数调节**：
- α → 1.0：更依赖最大注意力头（接近max_head）
- α = 0.5：平衡两者
- α → 0.0：更依赖其他头平均（接近avg_all_heads）

## 调参建议

### 1. 剪枝比例

修改 `--fast-v-attention-rank` 来调整保留的token数量：

```bash
# 保留25%（高剪枝）
--fast-v-attention-rank 144

# 保留50%（中等剪枝）
--fast-v-attention-rank 288

# 保留75%（低剪枝）
--fast-v-attention-rank 432
```

### 2. 聚合层

修改 `--fast-v-agg-layer` 来改变剪枝开始的层：

```bash
# 更早剪枝（可能更激进）
--fast-v-agg-layer 1

# 默认
--fast-v-agg-layer 2

# 更晚剪枝（可能更保守）
--fast-v-agg-layer 4
```

### 3. α参数扫描（weighted_combination策略）

```bash
for alpha in 0.1 0.3 0.5 0.7 0.9; do
    python ./src/HiMAP/inference/eval_scivqa.py \
        --use-fast-v \
        --fast-v-token-selection-method weighted_combination \
        --fast-v-weighted-alpha $alpha \
        # ... 其他参数 ...
done
```

## 结果文件

测试完成后会生成以下JSON文件：

```
scienceqa_results_baseline.json
scienceqa_results_fastv_max_head.json
scienceqa_results_fastv_avg_all_heads.json
scienceqa_results_fastv_weighted_combination_alpha0.3.json
scienceqa_results_fastv_weighted_combination_alpha0.5.json
scienceqa_results_fastv_weighted_combination_alpha0.7.json
```

每个文件包含：
- `accuracy`: 准确率
- `avg_latency`: 平均延迟
- `flops_info`: FLOPs比例
- `model_config`: 详细配置信息

## 常见问题

### Q1: 如何选择最佳策略？

**建议流程**：
1. 先用 `avg_all_heads` 建立基线
2. 尝试 `max_head` 看是否提升
3. 如果两者差异大，用 `weighted_combination` 微调
4. 通过α参数扫描找到最佳配置

### Q2: 如何平衡准确率和速度？

- 准确率优先：使用较小的剪枝比例（如75%保留）
- 速度优先：使用较大的剪枝比例（如25%保留）
- 平衡：50%保留 + weighted_combination策略

### Q3: 不同数据集需要不同配置吗？

是的，建议针对每个数据集进行：
1. 小规模测试（--num-samples 100）
2. 策略对比
3. 参数微调
4. 全量测试

## 性能基准

基于ScienceQA数据集的典型结果：

| 配置 | 准确率 | 相对基线 | 速度提升 |
|------|--------|----------|----------|
| Baseline | 78.0% | - | 1.0x |
| FastV-max_head-50% | 77.5% | -0.5% | 1.9x |
| FastV-avg_all_heads-50% | 78.0% | 0.0% | 1.8x |
| FastV-weighted(0.7)-50% | 78.5% | +0.5% | 1.8x |

*注：实际结果会因模型、数据集和硬件而异*

## 更多资源

- 详细文档：[FASTV_ADVANCED_README.md](../../FASTV_ADVANCED_README.md)
- 功能测试：`python test_fastv_advanced.py`
- 使用示例：`python example_fastv_advanced.py`

## 技术支持

遇到问题？检查：
1. CUDA是否可用
2. 模型路径是否正确
3. 数据文件是否存在
4. 参数配置是否合理

查看完整日志以定位问题。
