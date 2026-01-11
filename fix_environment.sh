#!/bin/bash
# HiMAP环境修复脚本

echo "=== HiMAP 环境修复脚本 ==="

# 1. 修复NumPy版本兼容性
echo "1. 修复NumPy版本兼容性..."
pip install "numpy<2.0" --force-reinstall

# 2. 修复PyTorch版本
echo "2. 修复PyTorch版本..."
pip install torch==2.0.1 torchvision==0.15.2 --force-reinstall

# 3. 修复transformers版本
echo "3. 修复transformers版本..."
pip install transformers==4.31.0 --force-reinstall

# 4. 修复tokenizers版本
echo "4. 修复tokenizers版本..."
pip install "tokenizers<0.14,>=0.12.1" --force-reinstall

# 5. 修复bitsandbytes版本
echo "5. 修复bitsandbytes版本..."
pip install bitsandbytes==0.41.0 --force-reinstall

# 6. 清理缓存
echo "6. 清理缓存..."
pip cache purge

echo "=== 修复完成 ==="
echo "请重新运行测试脚本"
