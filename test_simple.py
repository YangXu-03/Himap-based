#!/usr/bin/env python3
"""
简单的HiMAP测试脚本 - 避免网络问题
"""
import torch
import os
import sys

# 添加项目路径
sys.path.append('/root/nfs/code/HiMAP/src')

def test_basic_imports():
    """测试基本导入"""
    try:
        print("测试基本导入...")
        import numpy as np
        print(f"✓ NumPy版本: {np.__version__}")
        
        import torch
        print(f"✓ PyTorch版本: {torch.__version__}")
        print(f"✓ CUDA可用: {torch.cuda.is_available()}")
        
        return True
    except Exception as e:
        print(f"✗ 基本导入失败: {e}")
        return False

def test_llava_imports():
    """测试LLaVA导入"""
    try:
        print("\n测试LLaVA导入...")
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        print("✓ LLaVA常量导入成功")
        
        from llava.model.builder import load_pretrained_model
        print("✓ LLaVA模型构建器导入成功")
        
        return True
    except Exception as e:
        print(f"✗ LLaVA导入失败: {e}")
        return False

def test_device_detection():
    """测试设备检测"""
    print("\n测试设备检测...")
    if torch.cuda.is_available():
        print("✓ CUDA可用，将使用GPU")
        device = "cuda"
    else:
        print("✓ CUDA不可用，将使用CPU")
        device = "cpu"
    
    print(f"当前设备: {device}")
    return device

def test_tensor_creation():
    """测试张量创建"""
    try:
        print("\n测试张量创建...")
        import numpy as np
        
        # 测试NumPy到PyTorch的转换
        np_array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        torch_tensor = torch.from_numpy(np_array)
        print("✓ NumPy到PyTorch转换成功")
        
        # 测试设备转换
        if torch.cuda.is_available():
            torch_tensor = torch_tensor.cuda()
            print("✓ CUDA张量创建成功")
        else:
            print("✓ CPU张量创建成功")
        
        return True
    except Exception as e:
        print(f"✗ 张量创建失败: {e}")
        return False

if __name__ == "__main__":
    print("=== HiMAP 简单测试 ===")
    
    # 测试基本导入
    if not test_basic_imports():
        print("基本导入失败，请检查环境")
        sys.exit(1)
    
    # 测试设备检测
    device = test_device_detection()
    
    # 测试张量创建
    if not test_tensor_creation():
        print("张量创建失败，请检查NumPy/PyTorch兼容性")
        sys.exit(1)
    
    # 测试LLaVA导入
    if not test_llava_imports():
        print("LLaVA导入失败，请检查依赖版本")
        sys.exit(1)
    
    print("\n✓ 所有测试通过！")
    print(f"当前设备: {device}")
    print("环境修复成功，可以尝试运行评估脚本")
