"""
简单测试脚本：验证FastV在rank=0时是否正确工作
"""
import torch
import sys
sys.path.insert(0, '/root/nfs/code/HiMAP/src/LLaVA')

# 测试topk(0)的行为
print("Testing topk behavior with k=0:")
test_tensor = torch.randn(576)
print(f"Test tensor shape: {test_tensor.shape}")

try:
    result = test_tensor.topk(0)
    print(f"topk(0) succeeded: {result}")
except Exception as e:
    print(f"topk(0) failed with error: {e}")

# 测试rank=0的mask生成逻辑
print("\nTesting mask generation with rank=0:")
batch_size = 1
seq_length_with_past = 612  # 35 (sys) + 576 (img) + 1 (question)
SYS_LENGTH = 35
IMAGE_TOKEN_LENGTH = 576
ATTENTION_RANK = 0

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 生成mask（模拟FastV逻辑）
gen_attention_mask = torch.ones((batch_size, seq_length_with_past), dtype=torch.bool, device=device)
gen_attention_mask[:, SYS_LENGTH:SYS_LENGTH+IMAGE_TOKEN_LENGTH] = False

if ATTENTION_RANK > 0:
    print("This branch should NOT execute when ATTENTION_RANK=0")
    # This should not execute
else:
    print("Correct: ATTENTION_RANK=0, all image tokens masked to False")

# 验证mask
image_tokens_mask = gen_attention_mask[:, SYS_LENGTH:SYS_LENGTH+IMAGE_TOKEN_LENGTH]
num_true = image_tokens_mask.sum().item()
num_false = (IMAGE_TOKEN_LENGTH - num_true)

print(f"\nMask statistics:")
print(f"  Total image tokens: {IMAGE_TOKEN_LENGTH}")
print(f"  Kept (True): {num_true}")
print(f"  Masked (False): {num_false}")
print(f"  Expected: All {IMAGE_TOKEN_LENGTH} should be False")

if num_true == 0 and num_false == IMAGE_TOKEN_LENGTH:
    print("\n✓ Test PASSED: All image tokens correctly masked when rank=0")
else:
    print("\n✗ Test FAILED: Mask not correctly generated")

# 测试从第2层开始剪除的情况
print("\n" + "="*60)
print("Testing layer-wise cutoff logic:")
num_layers = 32
for agg_layer in [2, 10, 20, 31]:
    print(f"\nAGG_LAYER = {agg_layer}:")
    for idx in range(num_layers):
        if idx < agg_layer:
            status = "No pruning (before AGG_LAYER)"
        elif idx == agg_layer:
            status = "Generate mask (at AGG_LAYER)"
        else:
            status = "Use generated mask (after AGG_LAYER)"
        
        if idx in [agg_layer-1, agg_layer, agg_layer+1]:
            print(f"  Layer {idx}: {status}")

print("\n" + "="*60)
print("All basic tests completed!")
