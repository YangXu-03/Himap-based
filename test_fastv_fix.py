#!/usr/bin/env python3
"""快速测试 FastV Advanced 修复是否有效"""

import subprocess
import sys

print("=" * 70)
print("测试 FastV Advanced 修复")
print("=" * 70)
print()

# 测试单个样本的 weighted_combination 方法
print("测试 weighted_combination 方法（1个样本）...")
cmd = [
    "python", "./src/HiMAP/inference/eval_mme_fastv_advanced.py",
    "--model-path", "/root/nfs/model/llava-v1.5-7b",
    "--question-file", "/root/nfs/code/HiMAP/data/MME/MME_test.json",
    "--image-folder", "/root/nfs/code/HiMAP/data/MME/images/test",
    "--single-pred-prompt",
    "--use-fast-v",
    "--fast-v-sys-length", "35",
    "--fast-v-image-token-length", "576",
    "--fast-v-attention-rank", "288",
    "--fast-v-agg-layer", "12",
    "--fast-v-token-selection-method", "weighted_combination",
    "--fast-v-weighted-alpha", "0.7",
    "--num-samples", "1",
    "--output-file", "test_fix_result.json"
]

result = subprocess.run(cmd, capture_output=False)

if result.returncode == 0:
    print()
    print("✓ 测试成功！修复有效。")
    print()
    print("现在可以运行完整测试:")
    print("  bash src/HiMAP/inference/eval_mme_fastv_advanced.sh")
    sys.exit(0)
else:
    print()
    print("✗ 测试失败，仍有问题。")
    sys.exit(1)
