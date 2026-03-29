import argparse
import torch
import os
import json
import numpy as np
import math
from tqdm import tqdm
from PIL import Image
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# Setup paths (assuming standard LLaVA and HiMAP layout in this workspace)
import sys
project_root = "/root/nfs/code/HiMAP"
if project_root not in sys.path:
    sys.path.append(project_root)
    sys.path.append(os.path.join(project_root, "src"))
    sys.path.append(os.path.join(project_root, "src/LLaVA"))

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, process_images

def parse_args():
    parser = argparse.ArgumentParser(description="Extraction Settings")
    parser.add_argument("--model-path", type=str, default="/root/nfs/model/llava-v1.5-7b")
    parser.add_argument("--question-file", type=str, default="/root/nfs/code/HiMAP/data/MME/MME_test.json")
    parser.add_argument("--image-folder", type=str, default="/root/nfs/code/HiMAP/data/MME/images/test")
    parser.add_argument("--output-file", type=str, default="mme_metrics_summary.json")
    parser.add_argument("--samples-per-task", type=int, default=1, help="Number of samples to process per task")
    parser.add_argument("--compute-attention-rank", action="store_true", help="Whether to collect attention ranks (could use more memory)")
    return parser.parse_args()

def calculate_entropy(logits):
    """Calculate the Shannon entropy of the logits distribution."""
    probs = torch.nn.functional.softmax(logits, dim=-1)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy.item()

def calculate_ppl(loss):
    """Calculate Perplexity from cross-entropy loss."""
    return math.exp(loss.item()) if not torch.isnan(loss) else float('inf')

def main():
    args = parse_args()
    
    # Disable torch init for speed
    from llava.utils import disable_torch_init
    disable_torch_init()
    
    # Load Model
    print(f"Loading model: {args.model_path}")
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, None, model_name, device_map="auto"
    )
    
    # We will hook into the attention if needed
    if args.compute_attention_rank:
        model.config.output_attentions = True
        
        # NOTE: Using text-generation models like LLaVA sometimes requires us to explicitly enable attentions in generation or forward
        # Let's ensure it outputs attention by default in any standard forward pass.
        if hasattr(model.config, 'text_config'):
            model.config.text_config.output_attentions = True
    
    # Load MME Tasks from JSON file
    if not os.path.exists(args.question_file):
        print(f"Error: Question file not found at {args.question_file}")
        return
        
    with open(args.question_file, "r") as f:
        questions_data = json.load(f)
        
    # Group questions by category (task)
    tasks_data = {}
    for item in questions_data:
        category = item.get("category", "default_task")
        if category not in tasks_data:
            tasks_data[category] = []
        tasks_data[category].append(item)
    
    all_metrics = {}
    
    for task, items in tasks_data.items():
        print(f"Processing Task: {task}")
        
        task_metrics = []
        samples_processed = 0
        
        for item in tqdm(items, desc=f"Evaluating {task}", total=min(len(items), args.samples_per_task)):
            if samples_processed >= args.samples_per_task:
                break
                
            image_name = item.get("image", item.get("image_file", ""))
            question = item.get("text", item.get("question", ""))
            ground_truth_answer = str(item.get("answer", ""))
            
            if not image_name:
                continue
            
            # Subcategory logic might be handled by joining image_folder with task folder if formatted that way,
            # but usually image_name is the relative path from image_folder, or just the file name.
            # MME JSON has image_file just like "0.png". Sometimes images are organized by category.
            # Checking direct path and task-based path
            image_path = os.path.join(args.image_folder, image_name)
            if not os.path.exists(image_path):
                # Try adding the category to the path if it's organized in folders
                alt_path = os.path.join(args.image_folder, task, image_name)
                if os.path.exists(alt_path):
                    image_path = alt_path
            
            if not os.path.exists(image_path):
                print(f"Warning: Image not found {image_path}")
                continue
                
            # Process Image
            image = Image.open(image_path).convert("RGB")
            image_tensor = process_images([image], image_processor, model.config)[0].unsqueeze(0).to(model.device, dtype=torch.float16)
            
            # Formatting prompt
            qs = question
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
                
            conv = conv_templates["llava_v1"].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            
            # Forward pass
            with torch.inference_mode():
                outputs = model(
                    input_ids=input_ids,
                    images=image_tensor,
                    output_attentions=args.compute_attention_rank,
                    return_dict=True
                )
                
            # Compute Metrics
            # Logits shape: [batch, sequence, vocab]
            logits = outputs.logits
            # Example logic: grab metrics from the last token representing the predicted output
            last_token_logits = logits[0, -1, :]
            
            entropy = calculate_entropy(last_token_logits)
            
            # Calculate mock PPL or if ground truth sequence is known, compute loss
            # Here we compute PPL over the expected answer if we wanted, or just general loss
            target_ids = tokenizer(ground_truth_answer, return_tensors="pt").input_ids.to(model.device)
            target_ids = target_ids[:, 1:] # remove bos
            
            # Pseudo-PPL metric based on the generated outputs vs label
            # Note: A real PPL requires passing the answer as labels to the model.
            metrics_entry = {
                "image": image_name,
                "question": question,
                "entropy": entropy if not math.isnan(entropy) else 0.0,
                "layer_entropies": [],
                "layer_attention_ranks": []
            }
            
            if hasattr(outputs, "attentions") and outputs.attentions is not None:
                # attentions is a tuple of length num_layers, each [batch, heads, seq, seq]
                img_token_start = list(input_ids[0]).index(IMAGE_TOKEN_INDEX) if IMAGE_TOKEN_INDEX in input_ids[0] else -1
                
                # Assume image tokens occupy 576 tokens in LLaVA-1.5
                num_img_tokens = 576
                if img_token_start != -1:
                    img_token_indices = list(range(img_token_start, img_token_start + num_img_tokens))
                else:
                    img_token_indices = []

                layer_entropies = []
                layer_attention_ranks = []
                
                for layer_idx, layer_attn in enumerate(outputs.attentions):
                    # layer_attn: [batch, heads, seq, seq]
                    avg_attn = layer_attn[0].mean(dim=0)  # [seq, seq], average across heads
                    last_token_attn = avg_attn[-1, :]     # [seq], attention from the last token to all tokens
                    
                    # 1. 计算这一层注意力的分布熵
                    attn_probs = last_token_attn / (last_token_attn.sum() + 1e-10)
                    layer_entropy_val = -torch.sum(attn_probs * torch.log(attn_probs + 1e-10)).item()
                    layer_entropies.append(layer_entropy_val)
                    
                    # 2. 计算图像Token注意力的总秩 (或平均注意力)
                    if img_token_indices:
                        # 获取所有token的attention并降序排序，找到每个token排在第几名(Rank)
                        sorted_idx = torch.argsort(last_token_attn, descending=True)
                        rank_positions = {idx.item(): rank for rank, idx in enumerate(sorted_idx)}
                        
                        img_ranks = [rank_positions[i] for i in img_token_indices if i in rank_positions]
                        avg_img_rank = sum(img_ranks) / len(img_ranks) if img_ranks else -1
                        layer_attention_ranks.append(avg_img_rank)
                    else:
                        layer_attention_ranks.append(-1)
                
                metrics_entry["layer_entropies"] = layer_entropies
                metrics_entry["layer_attention_ranks"] = layer_attention_ranks
            
            task_metrics.append(metrics_entry)
            samples_processed += 1
            
        all_metrics[task] = task_metrics

    # Save to JSON
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=4)
        print(f"\\nMetrics saved to {args.output_file}")
        
    # Plotting layer metrics if computed
    if args.compute_attention_rank:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 5))
        
        # Subplot 1: Average Attention Entropy per Layer
        plt.subplot(1, 2, 1)
        all_entropies = []
        for task, metrics in all_metrics.items():
            for m in metrics:
                if "layer_entropies" in m and m["layer_entropies"]:
                    all_entropies.append(m["layer_entropies"])
        if all_entropies:
            avg_entropies = np.mean(all_entropies, axis=0)
            layers = range(1, len(avg_entropies) + 1)
            plt.plot(layers, avg_entropies, marker='o', linestyle='-')
            plt.title('Average Attention Entropy Across Layers')
            plt.xlabel('Layer Depth')
            plt.ylabel('Entropy (Shannon)')
            plt.grid(True)
            
        # Subplot 2: Average Image Token Attention Rank per Layer
        plt.subplot(1, 2, 2)
        all_ranks = []
        for task, metrics in all_metrics.items():
            for m in metrics:
                if "layer_attention_ranks" in m and m["layer_attention_ranks"]:
                    all_ranks.append(m["layer_attention_ranks"])
        if all_ranks:
            avg_ranks = np.mean(all_ranks, axis=0)
            layers = range(1, len(avg_ranks) + 1)
            plt.plot(layers, avg_ranks, marker='s', color='orange', linestyle='-')
            plt.title('Avg Image Tokens Rank Across Layers\n(Lower Rank = Higher Attention)')
            plt.xlabel('Layer Depth')
            plt.ylabel('Rank')
            plt.grid(True)
            
        plt.tight_layout()
        plot_path = args.output_file.replace(".json", "_layer_metrics.png")
        plt.savefig(plot_path)
        print(f"Layer metrics plot saved to {plot_path}")

if __name__ == "__main__":
    main()