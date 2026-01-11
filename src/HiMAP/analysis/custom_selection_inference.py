"""
使用自定义Token选择模型的示例脚本

展示如何使用FPS/ToMe + Cross-Attention进行视觉token选择和聚合
"""
import argparse
import torch
import os
import json
from tqdm import tqdm
from PIL import Image

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


def run_inference_with_custom_selection(args):
    """
    使用自定义Token选择模型进行推理
    """
    # 初始化
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    
    # 加载模型（启用自定义选择）
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, 
        args.model_base, 
        model_name,
        use_custom_selection=True  # 关键参数：启用自定义Token选择
    )
    
    # 设置自定义选择参数
    model.config.use_custom_selection = True
    model.config.custom_sys_length = args.custom_sys_length
    model.config.custom_image_token_length = args.custom_image_token_length
    model.config.custom_kept_tokens = args.custom_kept_tokens
    model.config.custom_agg_layer = args.custom_agg_layer
    model.config.custom_selection_method = args.custom_selection_method
    model.config.custom_temperature = args.custom_temperature
    
    # 重置模型参数
    model.model.reset_custom_selection()
    
    print(f"Custom Token Selection Settings:")
    print(f"  - Selection Method: {args.custom_selection_method}")
    print(f"  - Kept Tokens: {args.custom_kept_tokens}")
    print(f"  - Aggregation Layer: {args.custom_agg_layer}")
    print(f"  - Temperature: {args.custom_temperature}")
    
    # 加载问题
    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    
    if args.num_samples > 0:
        questions = questions[:args.num_samples]
    
    print(f"\nProcessing {len(questions)} samples...")
    
    correct = 0
    results = []
    
    for line in tqdm(questions):
        idx = line["id"]
        question = line['conversations'][0]
        qs = question['value'].replace('<image>', '').strip()
        label = line['conversations'][1]['value']
        
        image_file = line["image"]
        image = Image.open(os.path.join(args.image_folder, image_file))
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        
        if torch.cuda.is_available():
            images = image_tensor.unsqueeze(0).to(device='cuda', dtype=model.dtype)
        else:
            images = image_tensor.unsqueeze(0).float()
        
        # 构建prompt
        if getattr(model.config, 'mm_use_im_start_end', False):
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        
        if args.single_pred_prompt:
            qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
        
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
        
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = [KeywordsStoppingCriteria(keywords, tokenizer, input_ids)] if conv.version == "v0" else None
        
        # 推理
        with torch.inference_mode():
            output = model.generate(
                input_ids,
                images=images,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=stopping_criteria,
            )
        
        # 解析输出
        input_token_len = input_ids.shape[1]
        outputs = tokenizer.batch_decode(output[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        
        # 检查准确率
        is_correct = (outputs == label)
        if is_correct:
            correct += 1
        
        # 获取选择的token位置
        selected_positions = None
        if hasattr(model.model, 'last_selected_positions'):
            selected_positions = model.model.last_selected_positions
        
        results.append({
            'id': idx,
            'question': qs,
            'label': label,
            'prediction': outputs,
            'correct': is_correct,
            'selected_positions': selected_positions.tolist() if selected_positions is not None else None
        })
    
    # 计算准确率
    accuracy = correct / len(questions)
    print(f"\nAccuracy: {accuracy:.4f} ({correct}/{len(questions)})")
    
    # 保存结果
    if args.output_file:
        output_data = {
            'config': {
                'selection_method': args.custom_selection_method,
                'kept_tokens': args.custom_kept_tokens,
                'agg_layer': args.custom_agg_layer,
                'temperature': args.custom_temperature,
            },
            'accuracy': accuracy,
            'correct': correct,
            'total': len(questions),
            'results': results
        }
        with open(args.output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Custom Token Selection Inference")
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--num-samples", type=int, default=-1)
    
    # Custom selection parameters
    parser.add_argument('--custom-sys-length', type=int, default=35,
                        help="System prompt length (tokens before image)")
    parser.add_argument('--custom-image-token-length', type=int, default=576,
                        help="Number of image tokens")
    parser.add_argument('--custom-kept-tokens', type=int, default=8,
                        help="Number of image tokens to keep (default: 8)")
    parser.add_argument('--custom-agg-layer', type=int, default=12,
                        help="Layer to apply token selection (default: 12)")
    parser.add_argument('--custom-selection-method', type=str, default='fps', 
                        choices=['fps', 'tome'],
                        help="Token selection method: 'fps' or 'tome'")
    parser.add_argument('--custom-temperature', type=float, default=0.1,
                        help="Temperature for cross-attention softmax")
    
    # Output
    parser.add_argument('--output-file', type=str, default=None,
                        help="Output JSON file for results")
    
    args = parser.parse_args()
    run_inference_with_custom_selection(args)
