import argparse
import torch
import os
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

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
        # MME usually has 2 questions per image. 
        # We group by question_id (which seems to be the original filename)
        img_groups = {}
        for x in items:
            qid = x['question_id']
            if qid not in img_groups: img_groups[qid] = []
            # Check if prediction matches ground truth
            img_groups[qid].append(x['pred'].lower() == x['gt'].lower())
        
        # A pair is correct only if ALL questions for that image are correct
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

def plot_layer_cutoff_results(results, output_plot):
    """Plot MME scores across different cutoff layers"""
    layers = results['layers']
    
    # Plot 1: Overall Scores
    plt.figure(figsize=(12, 6))
    plt.plot(layers, results['total_scores'], marker='o', label='Total MME Score', linewidth=2)
    plt.plot(layers, results['perception_scores'], marker='s', label='Perception Score', linewidth=2)
    plt.plot(layers, results['cognition_scores'], marker='^', label='Cognition Score', linewidth=2)
    
    plt.xlabel('Layer Index (Image Token Cutoff Layer)', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('MME Scores vs Layer Cutoff', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"Overall scores plot saved to {output_plot}")
    
    # Plot 2: Subtask Scores
    plt.figure(figsize=(14, 8))
    # Get all subtasks from the first layer result
    subtasks = list(results['subtask_scores'][0].keys())
    
    # Use a colormap to handle many lines
    colors = plt.cm.tab20(np.linspace(0, 1, len(subtasks)))
    
    for i, task in enumerate(subtasks):
        task_scores = [layer_scores.get(task, 0) for layer_scores in results['subtask_scores']]
        plt.plot(layers, task_scores, marker='.', label=task, color=colors[i], linewidth=1.5, alpha=0.8)
        
    plt.xlabel('Layer Index (Image Token Cutoff Layer)', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('MME Subtask Scores vs Layer Cutoff', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.tight_layout()
    
    subtask_plot = output_plot.replace('.png', '_subtasks.png')
    plt.savefig(subtask_plot, dpi=300, bbox_inches='tight')
    print(f"Subtask scores plot saved to {subtask_plot}")

def run_experiment(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # Load MME data
    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    
    if args.num_samples > 0:
        questions = questions[:args.num_samples]
    
    print(f"Total samples: {len(questions)}")
    
    # Get model layers
    num_layers = len(model.model.layers)
    print(f"Model has {num_layers} layers")
    
    # Experiment parameters: Test from layer 2 to last layer (similar to reference script)
    # Reference script uses range(2, num_layers + 1)
    layers_to_test = list(range(2, num_layers + 1))
    
    experiment_results = {
        'layers': [],
        'total_scores': [],
        'perception_scores': [],
        'cognition_scores': [],
        'subtask_scores': []
    }

    # Enable FastV
    model.config.use_fast_v = True
    model.config.fast_v_sys_length = args.fast_v_sys_length
    model.config.fast_v_image_token_length = args.fast_v_image_token_length
    # Complete pruning: set rank to 0
    model.config.fast_v_attention_rank = 0
    
    print(f"Starting layer cutoff experiment (completely removing image tokens)")
    
    for layer_idx in tqdm(layers_to_test, desc="Testing Layers"):
        # Set cutoff layer
        model.config.fast_v_agg_layer = layer_idx
        if hasattr(model.model, 'reset_fastv'):
            model.model.reset_fastv()
            
        # Run inference for this layer
        layer_results = []
        
        for line in questions:
            idx = line.get("question_id")
            qs = line["question"]
            label = line["answer"]
            category = line["category"]
            image_file = line["image_file"]
            
            # Load image
            image_path = os.path.join(args.image_folder, image_file)
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                # print(f"Error loading image {image_path}: {e}")
                continue

            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            
            if torch.cuda.is_available():
                images = image_tensor.unsqueeze(0).half().cuda()
            else:
                images = image_tensor.unsqueeze(0).float()

            # Prepare prompt
            if getattr(model.config, 'mm_use_im_start_end', False):
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            qs = qs + '\n' + "Answer the question using a single word or phrase."

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
                output = model.generate(
                    input_ids,
                    images=images,
                    max_new_tokens=10,
                    use_cache=True,
                    stopping_criteria=stopping_criteria,
                )

            input_token_len = input_ids.shape[1]
            outputs = tokenizer.batch_decode(output[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
            
            pred = outputs
            if "yes" in pred.lower():
                pred = "Yes"
            elif "no" in pred.lower():
                pred = "No"
            
            layer_results.append({
                "question_id": idx,
                "category": category,
                "pred": pred,
                "gt": label
            })

        # Calculate scores for this layer
        scores, perception_score, cognition_score = calculate_mme_scores(layer_results)
        
        experiment_results['layers'].append(layer_idx)
        experiment_results['total_scores'].append(perception_score + cognition_score)
        experiment_results['perception_scores'].append(perception_score)
        experiment_results['cognition_scores'].append(cognition_score)
        experiment_results['subtask_scores'].append(scores)
        
        print(f"Layer {layer_idx}: Total Score = {perception_score + cognition_score:.2f}")

    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(experiment_results, f, indent=2)
    print(f"Results saved to {args.output_file}")
    
    # Plot
    plot_layer_cutoff_results(experiment_results, args.output_plot)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-samples", type=int, default=-1)
    parser.add_argument("--output-file", type=str, default="mme_layer_cutoff_results.json")
    parser.add_argument("--output-plot", type=str, default="mme_layer_cutoff_plot.png")
    
    # FastV parameters
    parser.add_argument('--fast-v-sys-length', type=int, default=35)
    parser.add_argument('--fast-v-image-token-length', type=int, default=576)
    
    args = parser.parse_args()
    run_experiment(args)
