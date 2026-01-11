import argparse
import torch
import os
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
import torch.nn.functional as F

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path

from HiMAP.analysis.atten_adapter import AttnAdapter, get_props

def calculate_mme_scores(results):
    # Define categories
    perception_cats = ["existence", "count", "position", "color", "posters", "celebrity", "scene", "landmark", "artwork", "OCR"]
    cognition_cats = ["commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"]
    
    cat_results = {}
    for r in results:
        cat = r['category']
        if cat not in cat_results:
            cat_results[cat] = []
        cat_results[cat].append(r)

    scores = {}
    perception_score = 0
    cognition_score = 0
    
    print(f"\n{'Category':<25} {'Acc':<10} {'Acc+':<10} {'Score':<10}")
    print("-" * 60)

    for cat, items in cat_results.items():
        # 1. Accuracy
        correct = sum(1 for x in items if x['pred'].lower() == x['gt'].lower())
        acc = correct / len(items) * 100
        
        # 2. Accuracy+ (Group by question_id/image pair)
        img_groups = {}
        for x in items:
            qid = x['question_id']
            if qid not in img_groups: img_groups[qid] = []
            img_groups[qid].append(x['pred'].lower() == x['gt'].lower())
        
        correct_pairs = sum(1 for v in img_groups.values() if all(v))
        acc_plus = correct_pairs / len(img_groups) * 100
        
        score = acc + acc_plus
        scores[cat] = score
        
        print(f"{cat:<25} {acc:<10.2f} {acc_plus:<10.2f} {score:<10.2f}")

        if cat in perception_cats:
            perception_score += score
        elif cat in cognition_cats:
            cognition_score += score
            
    print("-" * 60)
    print(f"Perception Score: {perception_score:.2f}")
    print(f"Cognition Score: {cognition_score:.2f}")
    print(f"Total MME Score: {perception_score + cognition_score:.2f}")
    
    return scores, perception_score, cognition_score

def plot_props_by_category(results, output_dir):
    # Group props by category
    cat_props = {}
    for r in results:
        cat = r['category']
        if cat not in cat_props:
            cat_props[cat] = {'props_all': [], 'props_img': []}
        cat_props[cat]['props_all'].append(r['props_all'])
        cat_props[cat]['props_img'].append(r['props_img'])

    os.makedirs(output_dir, exist_ok=True)

    for cat, props in cat_props.items():
        # Average props
        props_all = np.array(props['props_all']) # (N, Layers, 3)
        props_img = np.array(props['props_img']) # (N, Layers, 2)
        
        props_all_mean = props_all.mean(axis=0).transpose(1, 0) # (3, Layers)
        props_img_mean = props_img.mean(axis=0).transpose(1, 0) # (2, Layers)
        
        # Plot props_all
        plot_props4all(props_all_mean, os.path.join(output_dir, f"{cat}_props4all.png"), title=f"{cat} - All Tokens")
        # Plot props_img
        plot_props4img(props_img_mean, os.path.join(output_dir, f"{cat}_props4img.png"), title=f"{cat} - Image Tokens")

def plot_props4img(data, path, title=""):
    data_min = np.min(data)
    data_max = np.max(data)
    if data_max - data_min > 1e-6:
        data = (data - data_min) / (data_max - data_min)

    x = np.arange(data.shape[1])
    colors = ['#007c9a', '#965e9b']
    custom_legend_labels = ['Intra-Visual Flow', 'Visual-Textual Flow']

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.bar(x, data[0], label=custom_legend_labels[0], alpha=0.3, color=colors[0], width=0.7)
    ax.bar(x, data[1], label=custom_legend_labels[1], alpha=0.3, color=colors[1], width=0.7)

    ax.set_xlabel('Transformer Layer', fontsize=24, labelpad=12)
    ax.set_ylabel('Importance Metric', fontsize=24, labelpad=12)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    ax.legend(fontsize=24, fancybox=True, loc='upper right',
                   prop={'size':24, 'style': 'italic'})
    if title:
        plt.title(title, fontsize=24)

    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_props4all(data, path, title=""):
    data_min = np.min(data)
    data_max = np.max(data)
    if data_max - data_min > 1e-6:
        data = (data - data_min) / (data_max - data_min)

    x = np.arange(data.shape[1])
    colors = ['#C6210D',  '#007fbc', '#3f7430']
    custom_legend_labels = ['System Prompts', 'Image Tokens', 'User Instructions']

    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.bar(x, data[2], label=custom_legend_labels[2], alpha=0.5, color=colors[2], width=0.8)
    ax.bar(x, data[0], label=custom_legend_labels[0], alpha=0.3, color=colors[0], width=0.8)
    ax.bar(x, data[1], label=custom_legend_labels[1], alpha=0.3, color=colors[1], width=0.8)

    ax.set_xlabel('Transformer Layer', fontsize=24, labelpad=12)
    ax.set_ylabel('Importance Metric', fontsize=24, labelpad=12)
    plt.xticks(fontsize=24)  
    plt.yticks(fontsize=24)  
    ax.legend(fontsize=4, fancybox=True, loc='upper right',
                   prop={'size':24, 'style': 'italic'})
    
    if title:
        plt.title(title, fontsize=24)

    plt.tight_layout()
    plt.savefig(path, dpi=600)
    plt.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--result-file", type=str, default="mme_saliency_results.pt")
    parser.add_argument("--output-dir", type=str, default="mme_saliency_plots")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-samples", type=int, default=-1)
    args = parser.parse_args()

    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    model.config.use_hmap_v = False
    if hasattr(model.model, 'reset_hmapv'):
        model.model.reset_hmapv()

    # Change the attention module 
    for layer in model.model.layers:
        attn_adap = AttnAdapter(layer.self_attn.config)
        attn_adap.load_state_dict(layer.self_attn.state_dict())
        attn_adap = attn_adap.half().cuda()
        layer.self_attn = attn_adap

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    if args.num_samples > 0:
        questions = questions[:args.num_samples]

    results = []

    for i, line in enumerate(tqdm(questions)):

        idx = line["question_id"]
        qs = line['question']
        answer = line['answer']
        category = line['category']
        image_file = line["image_file"]

        # Prepare Label
        # MME answers are typically "Yes" or "No".
        # We use the first token of the answer as label.
        label_ids = tokenizer(answer).input_ids
        # Remove start token if present (LLaMA tokenizer adds it by default)
        if label_ids[0] == tokenizer.bos_token_id:
            label_ids = label_ids[1:]
        
        if len(label_ids) == 0:
            print(f"Warning: Empty label for answer '{answer}'. Skipping.")
            continue
            
        label_id = label_ids[0]
        label = torch.tensor([label_id], dtype=torch.int64).cuda()

        # Prepare Image
        image_path = os.path.join(args.image_folder, image_file)
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            continue

        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        images = image_tensor.unsqueeze(0).half().cuda()

        # Prepare Prompt
        cur_prompt = qs
        if getattr(model.config, 'mm_use_im_start_end', False):
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        
        # Append instruction for short answer
        qs = qs + '\n' + "Answer the question using a single word or phrase."

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        model.zero_grad()
        output_ids = model.forward(
            input_ids,
            images=images,
            use_cache=True,
            output_attentions=True,
            return_dict=True,
        )

        # Prediction
        pred_logit = output_ids['logits'][:,-1,:].squeeze(0) # (Vocab,)
        pred_id = torch.argmax(pred_logit).item()
        pred_text = tokenizer.decode([pred_id]).strip()
        
        # Normalize prediction
        if "yes" in pred_text.lower():
            pred_text = "Yes"
        elif "no" in pred_text.lower():
            pred_text = "No"
        
        # Saliency
        loss = F.cross_entropy(pred_logit.unsqueeze(0), label)
        loss.backward()

        # compute the saliency score
        props_all, props_img = [], []
        for idx_layer, layer in enumerate(model.model.layers):

            attn_grad = layer.self_attn.attn_map.grad.detach().clone().cpu()
            attn_map = output_ids['attentions'][idx_layer].detach().clone().cpu()
            saliency = torch.abs(attn_grad * attn_map)

            props4all, props4img = get_props(saliency)
            props_all.append(props4all)
            props_img.append(props4img)

        results.append({
            "question_id": idx,
            "category": category,
            "pred": pred_text,
            "gt": answer,
            "props_all": props_all,
            "props_img": props_img
        })
    
    # Save results
    torch.save(results, args.result_file)
    
    # Calculate Scores
    calculate_mme_scores(results)
    
    # Plot
    plot_props_by_category(results, args.output_dir)
