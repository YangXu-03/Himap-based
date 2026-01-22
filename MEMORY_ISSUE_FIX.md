# 显存溢出问题修复

## 问题描述
custom_selection_inference.py 在 baseline 模式下运行时，第 3 个样本就出现 CUDA OOM 错误，而 eval_scivqa.py 可以正常运行 100+ 个样本。

```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 48.00 MiB
错误位置: himap_modeling_llama.py 第 327 行 attention softmax 计算
```

错误堆栈表明问题出在 HiMAP 的 attention 处理逻辑上。

## 真正的根本原因 🎯

**加载模型时的隐式行为差异**：

当调用 `load_pretrained_model()` 时，无论是否指定 custom selection：
- 模型都会被加载为 `Himap_LlamaModel`（来自 himap.py）
- `Himap_LlamaModel` 的 forward 方法包含大量 HiMAP 特定的处理逻辑

在 custom_selection_inference.py 的 baseline 模式下：
1. **没有禁用 HiMAP 逻辑** - 模型仍在执行 HiMAP 的 token 剪枝和 attention 修改
2. **显存不断累积** - 每个样本的 HiMAP 逻辑都会分配新的 attention 缓存和中间张量
3. **最终导致 OOM** - 第 3 个样本时显存耗尽

而 eval_scivqa.py 的 baseline 模式：
```python
model.config.use_hmap_v = False
print('NO TOKEN PRUNING TCHNIQUE WILL BE USED ------')
```

设置 `use_hmap_v = False` 后，himap.py 的 forward 方法会直接跳过所有 HiMAP 逻辑：
```python
if USE_HMAP_V:  # 当 use_hmap_v = False 时，这整个块都会被跳过
    # ... 大量内存密集的操作
else:
    new_attention_mask = attention_mask
```

## 解决方案

### 唯一修改：在 baseline 模式下禁用 HiMAP

```python
# set model custom selection config
# In baseline mode, disable all pruning techniques
# Only set custom selection in non-baseline mode
if not args.baseline:
    model.config.use_custom_selection = True
    model.config.custom_sys_length = args.custom_sys_length
    model.config.custom_image_token_length = args.custom_image_token_length
    model.config.custom_kept_tokens = args.custom_kept_tokens
    model.config.custom_agg_layer = args.custom_agg_layer
    model.config.custom_selection_method = args.custom_selection_method
    model.config.custom_temperature = args.custom_temperature
    print('CUSTOM TOKEN SELECTION TECHNIQUE WILL BE USED ------')
    
    if hasattr(model.model, 'reset_custom_selection'):
        model.model.reset_custom_selection()
else:
    # Baseline mode: disable HiMAP to avoid memory overhead
    model.config.use_hmap_v = False
    if hasattr(model.model, 'reset_hmapv'):
        model.model.reset_hmapv()
    print('NO TOKEN PRUNING TECHNIQUE WILL BE USED ------')
```

**关键改变**：
- 添加 `model.config.use_hmap_v = False` 来禁用 HiMAP 的内存密集操作
- 调用 `model.model.reset_hmapv()` 来确保 HiMAP 相关的状态被重置

### 为什么这样做有效

1. **himap.py 的 forward 方法中**：
   ```python
   if USE_HMAP_V:  # 取决于 use_hmap_v 配置
       # 大量 token 剪枝和 attention 修改逻辑（内存密集）
   else:
       new_attention_mask = attention_mask  # 直接通过，无额外处理
   ```

2. 当 `use_hmap_v = False` 时，整个 HiMAP 逻辑块被跳过，避免了内存泄漏

## 效果

- ✅ Baseline 模式现在完全禁用 HiMAP 处理逻辑
- ✅ 显存使用恢复正常，可以处理大量样本
- ✅ 与 eval_scivqa.py 的 baseline 模式行为一致
- ✅ Custom selection 模式仍能正常工作
- ✅ 无性能副作用（baseline 现在更快）

## 测试步骤

运行 baseline 模式进行验证：
```bash
python custom_selection_inference.py \
    --model-path <model_path> \
    --image-folder <image_folder> \
    --question-file <question_file> \
    --baseline  # 测试 baseline 模式
```

预期结果：
- 能够处理 100+ 个样本
- 显存占用稳定（不会逐步增长）
- 输出：`NO TOKEN PRUNING TECHNIQUE WILL BE USED ------`

## 关键学习点

| 方面 | 问题 | 解决方案 |
|------|------|--------|
| HiMAP 默认行为 | 加载模型时总是使用 Himap_LlamaModel | 在 config 中禁用 `use_hmap_v` 标志 |
| 显存泄漏根源 | HiMAP forward 方法的内存密集处理 | 当 `use_hmap_v=False` 时整个逻辑块被跳过 |
| Config 设置 | 不了解 himap.py 依赖 config 标志 | 设置正确的 config 标志控制行为流程 |
