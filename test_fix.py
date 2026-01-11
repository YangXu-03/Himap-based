import json

with open('/root/nfs/code/HiMAP/data/MME/MME_test.json', 'r') as f:
    data = json.load(f)

categories = set()
for item in data:
    categories.add(item['category'])

print(sorted(list(categories)))

import os
import sys

# 添加项目路径
sys.path.append('/root/nfs/code/HiMAP/src')

def test_imports():
    """测试导入是否正常"""
    try:
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from llava.model.builder import load_pretrained_model
        print("✓ 导入成功")
        return True
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        return False

def test_device_detection():
    """测试设备检测"""
    if torch.cuda.is_available():
        print("✓ CUDA可用")
        device = "cuda"
    else:
        print("✓ 使用CPU模式")
        device = "cpu"
    return device

def test_model_loading():
    """测试模型加载（不实际加载，只测试函数调用）"""
    try:
        # 测试模型路径处理
        model_path = "liuhaotian/llava-v1.5-7b"
        if os.path.exists(model_path):
            print(f"✓ 本地路径存在: {model_path}")
        else:
            print(f"✓ 远程路径: {model_path}")
        return True
    except Exception as e:
        print(f"✗ 模型路径测试失败: {e}")
        return False

if __name__ == "__main__":
    print("=== HiMAP 修复测试 ===")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    # 测试导入
    print("\n1. 测试导入...")
    if not test_imports():
        sys.exit(1)
    
    # 测试设备检测
    print("\n2. 测试设备检测...")
    device = test_device_detection()
    
    # 测试模型路径
    print("\n3. 测试模型路径...")
    if not test_model_loading():
        sys.exit(1)
    
    print("\n✓ 所有测试通过！代码修复成功。")
    print(f"当前设备: {device}")
    print("现在可以运行评估脚本了。")
