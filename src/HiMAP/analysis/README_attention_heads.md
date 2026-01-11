# LLaVA 注意力头可视化分析工具

## 功能概述

本工具用于观测和可视化 LLaVA 多模态大模型在 MME 数据集上推理时，各个注意力头（Attention Heads）的"视觉-文本"交互模式。

## 核心指标说明

### 1. Text-to-Visual (T2V)
- **定义**: 文本token对图像token的注意力强度
- **计算步骤**:
  1. 提取Query为文本、Key为图像的注意力子矩阵
  2. 沿Key(图像)维度求和 → 每个文本token对所有图像的总关注度（范围0~1）
  3. 沿Query(文本)维度取平均 → 得到该头的T2V值

### 2. Visual-to-Text (V2T)
- **定义**: 图像token对文本token的注意力强度
- **计算步骤**:
  1. 提取Query为图像、Key为文本的注意力子矩阵
  2. 沿Key(文本)维度求和 → 每个图像token对所有文本的总关注度（范围0~1）
  3. 沿Query(图像)维度取平均 → 得到该头的V2T值

## 可视化输出

### 1. 总体堆叠柱状图 (Overall Stacked Histogram)
- **文件**: `mme_overall_stacked_hist.png`
- **内容**: 所有样本、所有层、所有头的注意力值分布
- **X轴**: 注意力值 (0.0 ~ 1.0, bins步长0.05)
- **Y轴**: 占所有头总数的比例
- **颜色**: 蓝色(T2V) 和 橙色(V2T) 堆叠

### 2. 分层堆叠柱状图 (Per-layer Stacked Histograms)
- **文件**: `mme_layer{i}_stacked_hist.png` (i = 0, 1, 2, ...)
- **内容**: 每一层的注意力头分布
- **用途**: 观察不同层的视觉-文本交互模式差异

### 3. 分类堆叠柱状图 (Category-specific Histograms)
- **目录**: `category_{category_name}/`
- **文件**: `mme_{category_name}_stacked_hist.png`
- **内容**: 按MME数据集的任务类别（如code_reasoning、OCR等）分别统计
- **用途**: 分析不同任务类型下的注意力模式差异

## 使用方法

### 基本用法
```bash
python observe_mme_attention_heads.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --image-folder /path/to/MME/images \
    --question-file /path/to/MME_test.json \
    --output-dir mme_attention_observation
```

### 完整参数说明
```bash
python observe_mme_attention_heads.py \
    --model-path <模型路径> \            # LLaVA模型路径或HuggingFace ID
    --model-base <基础模型路径> \        # 可选，用于LoRA模型
    --image-folder <图片目录> \          # MME数据集图片目录
    --question-file <问题文件> \         # MME_test.json文件路径
    --output-dir <输出目录> \            # 结果保存目录（默认: mme_attention_observation）
    --conv-mode <对话模式> \             # 对话模板（默认: vicuna_v1）
    --num-samples <样本数量>             # 处理的样本数量（-1表示全部）
```

### 实际使用示例

#### 示例1: 使用本地模型处理所有样本
```bash
cd /root/nfs/code/HiMAP
python src/HiMAP/analysis/observe_mme_attention_heads.py \
    --model-path /path/to/llava-v1.5-7b \
    --image-folder data/MME/images/test \
    --question-file data/MME/MME_test.json \
    --output-dir mme_attention_observation
```

#### 示例2: 快速测试（仅处理100个样本）
```bash
python src/HiMAP/analysis/observe_mme_attention_heads.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --image-folder data/MME/images/test \
    --question-file data/MME/MME_test.json \
    --output-dir mme_attention_test \
    --num-samples 100
```

#### 示例3: 使用Shell脚本批量运行
```bash
bash src/HiMAP/analysis/observe_mme_attention_head.sh
```

## 输出文件结构

```
mme_attention_observation/
├── mme_attention_head_records.pt          # 原始注意力数据（PyTorch格式）
├── mme_overall_stacked_hist.png           # 总体分布图
├── mme_layer0_stacked_hist.png            # 第0层分布图
├── mme_layer1_stacked_hist.png            # 第1层分布图
├── ...
├── category_code_reasoning/               # 代码推理类别
│   └── mme_code_reasoning_stacked_hist.png
├── category_OCR/                          # OCR类别
│   └── mme_OCR_stacked_hist.png
└── ...
```

## 技术细节

### 依赖库
- PyTorch
- Transformers
- Matplotlib
- NumPy
- PIL (Pillow)
- tqdm

### 性能考虑
- 使用PyTorch矩阵操作加速计算
- 支持CUDA加速（自动检测）
- 支持半精度推理（FP16）以节省显存
- 使用try-except处理图片加载错误，提高鲁棒性

### 内存优化
- 逐样本处理，避免一次性加载所有数据
- 使用torch.no_grad()减少显存占用
- 注意力矩阵仅保存统计值，不保存完整矩阵

## 结果解读

### 理想的注意力模式
1. **T2V高**: 表示文本在"查询"图像信息，适合需要理解图像内容的任务
2. **V2T高**: 表示图像在"反馈"到文本，适合生成描述或回答问题
3. **分层差异**: 浅层倾向局部特征，深层倾向全局语义

### 异常模式识别
- **T2V和V2T都很低**: 可能表示该头在处理其他类型的交互（如text-to-text）
- **分布极端**: 所有值都接近0或1，可能表示模型过拟合或注意力机制失效
- **类别间差异大**: 不同任务需要不同的视觉-文本交互模式

## 扩展应用

### 1. 模型比较
比较不同模型（如LLaVA-7B vs LLaVA-13B）的注意力模式差异

### 2. 层剪枝
识别贡献较小的层，用于模型压缩

### 3. 故障诊断
定位模型在特定任务上表现不佳的原因

### 4. 提示工程
根据注意力模式优化输入提示词

## 常见问题

### Q1: 为什么有些样本被跳过？
A: 可能原因：
- 图片文件损坏或无法加载
- 没有检测到IMAGE_TOKEN
- 文本或视觉token数量为0

### Q2: 如何减少运行时间？
A: 
- 使用`--num-samples`限制样本数量
- 确保使用GPU（CUDA）
- 使用较小的模型（如7B而非13B）

### Q3: 输出图片分辨率如何调整？
A: 修改代码中的`dpi=300`参数，或调整`figsize`

### Q4: 如何自定义bins的范围和步长？
A: 修改代码中的`bins = np.arange(0.0, 1.05, 0.05)`

## 引用

如果您在研究中使用本工具，请引用相关论文和代码库。

## 联系方式

如有问题或建议，请提交Issue或Pull Request。
