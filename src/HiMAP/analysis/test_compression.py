"""
快速测试token压缩是否正常工作
"""

import sys
import os
sys.path.insert(0, '/root/nfs/code/HiMAP/src/LLaVA')

import argparse
import torch
import json

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path

from PIL import Image


def test_compression():
    """测试不同token数量下的压缩效果"""
    
    # 加载模型
    disable_torch_init()
    model_path = "liuhaotian/llava-v1.5-7b"
    model_name = get_model_name_from_path(model_path)
    
    print("加载模型...")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name
    )
    
    # 加载一个测试样本
    question_file = "/root/nfs/code/HiMAP/data/scienceqa/himap-inference-MCQ.json"
    image_folder = "./data/scienceqa/images/test"
    
    questions = json.load(open(question_file, "r"))
    line = questions[0]  # 使用第一个样本
    
    # 准备输入
    question = line['conversations'][0]
    qs = question['value'].replace('<image>', '').strip()
    qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
    qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
    
    image_file = line["image"]
    image = Image.open(os.path.join(image_folder, image_file))
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
    images = image_tensor.unsqueeze(0).half().cuda()
    
    conv = conv_templates["vicuna_v1"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    
    print(f"输入序列长度: {input_ids.shape[1]}")
    print(f"IMAGE_TOKEN数量: {(input_ids == IMAGE_TOKEN_INDEX).sum().item()}")
    
    # 测试不同的token数量
    test_configs = [
        {"name": "基线(无压缩)", "num_tokens": 576, "use_hmap": False},
        {"name": "压缩到288", "num_tokens": 288, "use_hmap": True},
        {"name": "压缩到144", "num_tokens": 144, "use_hmap": True},
        {"name": "压缩到72", "num_tokens": 72, "use_hmap": True},
        {"name": "压缩到36", "num_tokens": 36, "use_hmap": True},
    ]
    
    # Hook to capture sequence length at each layer
    layer_seq_lengths = []
    
    def capture_seq_length(module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        layer_seq_lengths.append(hidden_states.shape[1])
    
    for config in test_configs:
        print(f"\n{'='*60}")
        print(f"测试配置: {config['name']}")
        print(f"{'='*60}")
        
        # 配置模型
        if config['use_hmap']:
            model.config.use_hmap_v = True
            model.config.hmap_v_sys_length = 35
            model.config.hmap_v_img_length = 576
            model.config.hmap_v_attn_txt_layer = 8
            model.config.hmap_v_attn_txt_rank = config['num_tokens']
            model.config.hmap_v_attn_img_layer = 9
            model.config.hmap_v_attn_img_rank = config['num_tokens']
            model.config.cut_off_layer = None
        else:
            model.config.use_hmap_v = False
        
        model.model.reset_hmapv()
        
        # 注册hooks
        layer_seq_lengths = []
        hooks = []
        for i, layer in enumerate(model.model.layers):
            if i in [0, 7, 8, 9, 10, 31]:  # 关键层
                hook = layer.register_forward_hook(capture_seq_length)
                hooks.append(hook)
        
        # 运行推理
        with torch.inference_mode():
            output = model.generate(
                input_ids,
                images=images,
                max_new_tokens=10,
                use_cache=False,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
        
        # 移除hooks
        for hook in hooks:
            hook.remove()
        
        # 打印序列长度变化
        print(f"关键层的序列长度:")
        layer_indices = [0, 7, 8, 9, 10, 31]
        for i, seq_len in enumerate(layer_seq_lengths):
            layer_idx = layer_indices[i]
            print(f"  Layer {layer_idx}: {seq_len}")
        
        # 解析输出
        input_token_len = input_ids.shape[1]
        outputs = tokenizer.batch_decode(output['sequences'][:, input_token_len:], skip_special_tokens=True)[0]
        print(f"模型输出: {outputs.strip()}")
    
    print(f"\n{'='*60}")
    print("测试完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    test_compression()
