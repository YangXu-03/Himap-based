import argparse
import torch
import os
import json
from tqdm import tqdm
import time

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


if __name__ == "__main__":
        

    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="llava-7B")
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str)
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--cutoff-start", type=int, default=0, help="the starting cut-off layer")
    parser.add_argument("--cutoff-end", type=int, default=12, help="the ending cut-off")

    # HiMAP hyperparameter
    parser.add_argument('--use-hmap-v', default=False, action='store_true', help='whether to use hmap-v')
    parser.add_argument('--sys-length', type=int, required=False, help='the length of system prompt')
    parser.add_argument('--img-length', type=int, required=False, help='the length of image token')
    parser.add_argument('--hmap-v-attn-txt-layer', type=int, required=False, help='the layer of pruning accorading to img2txt information')
    parser.add_argument('--hmap-v-attn-img-layer', type=int, required=False, help='the layer of pruning accorading to img2img information')
    parser.add_argument('--hmap-v-attn-txt-rank', type=int, required=False, help='the rank of attn accorading to img2txt information')
    parser.add_argument('--hmap-v-attn-img-rank', type=int, required=False, help='the rank of attn accorading to img2img information')
    parser.add_argument('--cut-off-layer', type=int, required=False, help='the layer index after which all image tokens are removed')
    # fastv config
    parser.add_argument('--use-fast-v', default=False, action='store_true', help='whether to use fast-v')
    parser.add_argument('--fast-v-sys-length', type=int, required=False, help='the length of system prompt for fast-v')
    parser.add_argument('--fast-v-image-token-length', type=int, required=False, help='the length of image token for fast-v')
    parser.add_argument('--fast-v-attention-rank', type=int, required=False, help='the rank of attention for fast-v')
    parser.add_argument('--fast-v-agg-layer', type=int, required=False, help='the aggregation layer for fast-v')
    args = parser.parse_args()
    

    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    
    # 检查是否为本地路径
    if os.path.exists(model_path):
        # 本地路径，直接使用
        model_name = get_model_name_from_path(model_path)
    else:
        # 远程路径，使用原始路径
        model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    # 小样本调试
    questions = questions[:10]  # 只用前10个样本
    
    if args.use_hmap_v:
        print('---BASELINE HIMAP---')
        model.config.use_hmap_v = True
        model.config.hmap_v_sys_length = args.sys_length
        model.config.hmap_v_img_length = args.img_length
        model.config.hmap_v_attn_txt_layer = args.hmap_v_attn_txt_layer
        model.config.hmap_v_attn_img_layer = args.hmap_v_attn_img_layer
        model.config.hmap_v_attn_txt_rank = args.hmap_v_attn_txt_rank
        model.config.hmap_v_attn_img_rank = args.hmap_v_attn_img_rank
        # cut-off layer to drop all image tokens after a specific layer
        model.config.cut_off_layer = 0
        model.model.reset_hmapv() 



        num_sample = len(questions)
        corr_sample = 0
        total_latency = 0.0
        total_flops_ratio_attn_ffn = 0.0

        for i, line in enumerate(tqdm(questions)):
            
            idx = line["id"]
            question = line['conversations'][0]
            qs = question['value'].replace('<image>', '').strip()
            cur_prompt = qs
            label = line['conversations'][1]['value']

            image_file = line["image"]
            image = Image.open(os.path.join(args.image_folder, image_file))
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            if torch.cuda.is_available():
                images = image_tensor.unsqueeze(0).half().cuda()
            else:
                images = image_tensor.unsqueeze(0).float()
            if getattr(model.config, 'mm_use_im_start_end', False):
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            cur_prompt = '<image>' + '\n' + cur_prompt


            if args.single_pred_prompt:
                qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
                cur_prompt = cur_prompt + '\n' + "Answer with the option's letter from the given choices directly."

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

            with torch.inference_mode():
                t0 = time.time()
                output_ids = model.generate(
                    input_ids,
                    images=images,
                    max_new_tokens=1024,
                    use_cache=False,
                    stopping_criteria=stopping_criteria,
                    output_attentions=True,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
            torch.cuda.synchronize()  # 确保GPU操作完成
            end_time = time.time()
            inference_latency = end_time - t0
            total_latency += inference_latency

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids['sequences'][:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids['sequences'][:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()

            if outputs == label:
                corr_sample += 1
            

        
        # Report metrics
        avg_latency = total_latency / max(num_sample, 1)
        avg_flops_ratio = total_flops_ratio_attn_ffn / max(num_sample, 1)
        baseline_acc = corr_sample/num_sample
        print(f'Accuracy: {baseline_acc:.4f}')
        print(f'Avg Latency/Example: {avg_latency:.6f}s')
        print(f'FLOPs Ratio (attn+ffn approx): {avg_flops_ratio*100:.2f}%')


    sweep_results = {
        'layers': [],
        'accuracies': [],
        'correct_counts': [],
        'total_samples': len(questions)
    }

    for cutoff_layer in range(args.cutoff_start, args.cutoff_end + 1):
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir, exist_ok=True)

        # set model hmapv config
        if args.use_hmap_v == True:
            model.config.use_hmap_v = True
            model.config.hmap_v_sys_length = args.sys_length
            model.config.hmap_v_img_length = args.img_length
            model.config.hmap_v_attn_txt_layer = args.hmap_v_attn_txt_layer
            model.config.hmap_v_attn_img_layer = args.hmap_v_attn_img_layer
            model.config.hmap_v_attn_txt_rank = args.hmap_v_attn_txt_rank
            model.config.hmap_v_attn_img_rank = args.hmap_v_attn_img_rank
            # cut-off layer to drop all image tokens after a specific layer
            model.config.cut_off_layer = cutoff_layer
            model.model.reset_hmapv()   
            
        elif args.use_fast_v == True:
            model.config.use_fast_v = True
            model.config.fast_v_sys_length = args.fast_v_sys_length
            model.config.fast_v_image_token_length = args.fast_v_image_token_length
            model.config.fast_v_attention_rank = args.fast_v_attention_rank
            model.config.fast_v_agg_layer = args.fast_v_agg_layer
            model.model.reset_fastv()
        else:
            model.config.use_hmap_v = False
            print('NO TOKEN PRUNING TCHNIQUE WILL BE USED ------')

        


        print(f"\n=== Evaluating cut_off_layer = {cutoff_layer} ===")

        # Evaluate

        # questions = json.load(open(os.path.expanduser(args.question_file), "r"))
        # questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
        # questions = questions[:10]  # 只用前5个样本

        num_sample = len(questions)
        corr_sample = 0
        total_latency = 0.0
        total_flops_ratio_attn_ffn = 0.0

        for i, line in enumerate(tqdm(questions)):
            
            idx = line["id"]
            question = line['conversations'][0]
            qs = question['value'].replace('<image>', '').strip()
            cur_prompt = qs
            label = line['conversations'][1]['value']

            image_file = line["image"]
            image = Image.open(os.path.join(args.image_folder, image_file))
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            if torch.cuda.is_available():
                images = image_tensor.unsqueeze(0).half().cuda()
            else:
                images = image_tensor.unsqueeze(0).float()
            if getattr(model.config, 'mm_use_im_start_end', False):
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            cur_prompt = '<image>' + '\n' + cur_prompt


            if args.single_pred_prompt:
                qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
                cur_prompt = cur_prompt + '\n' + "Answer with the option's letter from the given choices directly."

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

            with torch.inference_mode():
                t0 = time.time()
                output_ids = model.generate(
                    input_ids,
                    images=images,
                    max_new_tokens=1024,
                    use_cache=False,
                    stopping_criteria=stopping_criteria,
                    output_attentions=True,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
            torch.cuda.synchronize()  # 确保GPU操作完成
            end_time = time.time()
            inference_latency = end_time - t0
            total_latency += inference_latency

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids['sequences'][:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids['sequences'][:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()

            if outputs == label:
                corr_sample += 1
            
        
        # Report metrics
        avg_latency = total_latency / max(num_sample, 1)
        accuracy = corr_sample/num_sample
        print(corr_sample/num_sample)
        print(f'Avg Latency/Example: {avg_latency:.6f}s')
  

        # 保存结果到文件
        sweep_results['layers'].append(cutoff_layer)
        sweep_results['accuracies'].append(accuracy)
        sweep_results['correct_counts'].append(corr_sample)
    

    # Save json
    out_json = os.path.join(args.output_dir, f"himap_cutoff_sweep_{args.tag}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(sweep_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_json}")

    # Plot

    import matplotlib.pyplot as plt

    layers = sweep_results['layers']
    accs = sweep_results['accuracies']

    plt.figure(figsize=(10, 6))
    plt.plot(layers, accs, marker="o")
    plt.axhline(y=baseline_acc, color='gray', linestyle='--', linewidth=1.5, label='baseline')
    plt.xlabel("cut_off_layer")
    plt.ylabel("accuracy")
    plt.title("HiMAP cut_off_layer sweep")
    plt.grid(True)
    # 添加数值标注
    for i, (layer, acc) in enumerate(zip(layers, accs)):
        if i % 3 == 0:  # 每隔3个点标注一次，避免拥挤
            plt.annotate(f'{acc:.3f}', 
                        xy=(layer, acc), 
                        xytext=(0, 8),
                        textcoords='offset points',
                        ha='center',
                        fontsize=8,
                        alpha=0.7)
    
    plt.tight_layout()
    out_png = os.path.join(args.output_dir, f"himap_cutoff_sweep_{args.tag}.png")
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_png}")






