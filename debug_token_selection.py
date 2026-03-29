#!/usr/bin/env python3
"""
测试实际运行时的token选择
"""
import argparse
import torch
import os
import json
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path

from PIL import Image

def test_single_inference(model_path, strategy, alpha=0.5):
    """测试单次推理"""
    
    # Load model
    disable_torch_init()
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, get_model_name_from_path(model_path)
    )
    
    # Configure FastV
    model.config.use_fast_v = True
    model.config.fast_v_sys_length = 35
    model.config.fast_v_image_token_length = 576
    model.config.fast_v_attention_rank = 128
    model.config.fast_v_agg_layer = 12
    model.config.fast_v_token_selection_method = strategy
    model.config.fast_v_weighted_alpha = alpha
    
    model.model.reset_fastv()
    
    print(f"\n配置:")
    print(f"  Strategy: {model.model.token_selection_method}")
    print(f"  Alpha: {model.model.weighted_alpha}")
    print(f"  Attention Rank: {model.model.fast_v_attention_rank}")
    
    # Create a dummy image
    image = Image.new('RGB', (336, 336), color='white')
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
    images = image_tensor.unsqueeze(0).half().cuda()
    
    # Create prompt
    qs = "What do you see in this image?"
    qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
    qs = qs + '\n' + "Answer the question using a single word or phrase."
    
    conv = conv_templates['vicuna_v1'].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    
    # Run inference
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images,
            do_sample=False,
            temperature=0.0,
            max_new_tokens=50,
            use_cache=False,
        )
    
    # Get selected indices
    if hasattr(model.model, 'last_gen_kept_indices') and model.model.last_gen_kept_indices is not None:
        kept_indices = model.model.last_gen_kept_indices
        print(f"  保留的token数量: {len(kept_indices)}")
        print(f"  前10个索引: {sorted(kept_indices)[:10]}")
        return set(kept_indices)
    else:
        print(f"  WARNING: No kept indices found!")
        return None

def main():
    model_path = "/root/nfs/model/llava-v1.5-7b"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
    
    print("="*60)
    print("测试不同策略的token选择")
    print("="*60)
    
    strategies = [
        ('max_head', 0.5),
        ('avg_all_heads', 0.5),
        ('weighted_combination', 0.7),
    ]
    
    results = {}
    for strategy, alpha in strategies:
        print(f"\n{'='*60}")
        print(f"测试策略: {strategy} (alpha={alpha})")
        print('='*60)
        results[strategy] = test_single_inference(model_path, strategy, alpha)
    
    # Compare
    print(f"\n{'='*60}")
    print("策略对比")
    print('='*60)
    
    if len(results) >= 2:
        keys = list(results.keys())
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                if results[keys[i]] and results[keys[j]]:
                    overlap = len(results[keys[i]] & results[keys[j]]) / len(results[keys[i]] | results[keys[j]]) * 100
                    print(f"{keys[i]} vs {keys[j]}: {overlap:.1f}% 重叠")

if __name__ == "__main__":
    main()
