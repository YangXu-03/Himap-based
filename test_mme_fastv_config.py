#!/usr/bin/env python3
"""
FastV Advanced MME 配置测试脚本

快速测试评估脚本的配置和数据路径是否正确
"""

import os
import json
import sys
from pathlib import Path

def check_file(path, description):
    """检查文件是否存在"""
    if os.path.exists(path):
        print(f"✓ {description}: {path}")
        return True
    else:
        print(f"✗ {description} 不存在: {path}")
        return False

def check_directory(path, description):
    """检查目录是否存在"""
    if os.path.isdir(path):
        print(f"✓ {description}: {path}")
        return True
    else:
        print(f"✗ {description} 不存在: {path}")
        return False

def check_json_file(path):
    """检查 JSON 文件是否有效"""
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        print(f"  - 包含 {len(data)} 条数据")
        if len(data) > 0:
            print(f"  - 示例键: {list(data[0].keys())}")
        return True
    except Exception as e:
        print(f"  - 错误: {e}")
        return False

def main():
    print("=" * 70)
    print("FastV Advanced MME 配置测试")
    print("=" * 70)
    print()
    
    all_ok = True
    
    # 检查模型路径
    print("1. 检查模型路径...")
    model_path = "/root/nfs/model/llava-v1.5-7b"
    if check_directory(model_path, "模型目录"):
        # 检查关键文件
        config_file = os.path.join(model_path, "config.json")
        check_file(config_file, "  配置文件")
    else:
        all_ok = False
    print()
    
    # 检查数据路径
    print("2. 检查 MME 数据集...")
    question_file = "/root/nfs/code/HiMAP/data/MME/MME_test.json"
    image_folder = "/root/nfs/code/HiMAP/data/MME/images/test"
    
    if check_file(question_file, "问题文件"):
        check_json_file(question_file)
    else:
        all_ok = False
        
    if not check_directory(image_folder, "图像目录"):
        all_ok = False
    else:
        # 统计子目录数量
        subdirs = [d for d in os.listdir(image_folder) 
                  if os.path.isdir(os.path.join(image_folder, d))]
        print(f"  - 包含 {len(subdirs)} 个子目录")
        if len(subdirs) > 0:
            print(f"  - 示例子目录: {subdirs[:5]}")
    print()
    
    # 检查脚本文件
    print("3. 检查脚本文件...")
    scripts = [
        ("评估脚本", "src/HiMAP/inference/eval_mme_fastv_advanced.py"),
        ("Bash 脚本", "src/HiMAP/inference/eval_mme_fastv_advanced.sh"),
        ("可视化脚本", "src/HiMAP/inference/visualize_mme_fastv_results.py"),
    ]
    
    for desc, script_path in scripts:
        if not check_file(script_path, desc):
            all_ok = False
    print()
    
    # 检查 FastV Advanced 模型文件
    print("4. 检查 FastV Advanced 模型文件...")
    fastv_file = "src/LLaVA/llava/model/language_model/fastv_advanced.py"
    if check_file(fastv_file, "FastV Advanced 实现"):
        # 检查关键类是否存在
        try:
            with open(fastv_file, 'r') as f:
                content = f.read()
            if 'FastvAdvanced_LlamaModel' in content:
                print("  - ✓ FastvAdvanced_LlamaModel 类已定义")
            else:
                print("  - ✗ FastvAdvanced_LlamaModel 类未找到")
                all_ok = False
                
            methods = [
                '_select_tokens_max_head',
                '_select_tokens_avg_all_heads', 
                '_select_tokens_weighted_combination',
                '_select_tokens_text_weighted',
                '_select_tokens_text_weighted_max_head'
            ]
            
            found_methods = sum(1 for m in methods if m in content)
            print(f"  - 找到 {found_methods}/{len(methods)} 个 token 选择方法")
            
        except Exception as e:
            print(f"  - 错误: {e}")
            all_ok = False
    else:
        all_ok = False
    print()
    
    # 检查 GPU 可用性
    print("5. 检查 CUDA 环境...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA 可用")
            print(f"  - GPU 数量: {torch.cuda.device_count()}")
            print(f"  - 当前 GPU: {torch.cuda.current_device()}")
            print(f"  - GPU 名称: {torch.cuda.get_device_name(0)}")
        else:
            print("✗ CUDA 不可用")
            all_ok = False
    except ImportError:
        print("✗ PyTorch 未安装")
        all_ok = False
    print()
    
    # 检查必要的 Python 包
    print("6. 检查 Python 依赖...")
    packages = [
        'torch',
        'transformers',
        'PIL',
        'matplotlib',
        'numpy',
        'seaborn',
        'tqdm'
    ]
    
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"✓ {pkg} 已安装")
        except ImportError:
            print(f"✗ {pkg} 未安装")
            all_ok = False
    print()
    
    # 总结
    print("=" * 70)
    if all_ok:
        print("✓ 所有检查通过！可以开始运行评估。")
        print()
        print("运行评估:")
        print("  bash src/HiMAP/inference/eval_mme_fastv_advanced.sh")
        print()
        print("或运行单个方法:")
        print("  python src/HiMAP/inference/eval_mme_fastv_advanced.py \\")
        print("    --model-path /root/nfs/model/llava-v1.5-7b \\")
        print("    --question-file /root/nfs/code/HiMAP/data/MME/llava_mme.json \\")
        print("    --image-folder /root/nfs/code/HiMAP/data/MME/MME_Benchmark_release_version \\")
        print("    --use-fast-v \\")
        print("    --fast-v-token-selection-method max_head \\")
        print("    --output-file test_results.json")
    else:
        print("✗ 存在配置问题，请修复后再运行评估。")
        sys.exit(1)
    print("=" * 70)

if __name__ == "__main__":
    main()
