
import argparse
import torch
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Add src to path
import sys
project_root = "/root/nfs/code/HiMAP"
sys.path.append(os.path.join(project_root, "src"))
sys.path.append(os.path.join(project_root, "src/LLaVA"))

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, process_images, KeywordsStoppingCriteria


def tensor_nan_stats(tensor):
    """
    Return lightweight NaN/Inf diagnostics for a tensor.
    """
    t = tensor.detach().float()
    total = t.numel()
    nan_mask = torch.isnan(t)
    inf_mask = torch.isinf(t)
    finite_mask = torch.isfinite(t)

    nan_count = int(nan_mask.sum().item())
    inf_count = int(inf_mask.sum().item())
    finite_count = int(finite_mask.sum().item())

    stats = {
        "shape": list(t.shape),
        "total": int(total),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "finite_count": finite_count,
        "nan_ratio": float(nan_count / total) if total > 0 else 0.0,
        "is_all_nan": bool(nan_count == total and total > 0),
        "has_any_nan": bool(nan_count > 0),
        "has_any_inf": bool(inf_count > 0),
    }

    if finite_count > 0:
        finite_vals = t[finite_mask]
        stats["finite_min"] = float(finite_vals.min().item())
        stats["finite_max"] = float(finite_vals.max().item())
        stats["finite_mean"] = float(finite_vals.mean().item())
    else:
        stats["finite_min"] = None
        stats["finite_max"] = None
        stats["finite_mean"] = None

    return stats

def load_mme_samples(mme_json_path):
    """
    Load MME data and select one sample per category (subtask).
    Returns a dictionary: {category: sample_item}
    """
    print(f"Loading MME data from {mme_json_path}...")
    try:
        with open(mme_json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: MME file not found at {mme_json_path}")
        return {}
    
    samples_by_category = {}
    for item in data:
        cat = item.get('category', 'unknown')
        # Select the first sample encountered for this category
        if cat not in samples_by_category:
            samples_by_category[cat] = item
    
    print(f"Selected {len(samples_by_category)} samples (one per category).")
    return samples_by_category

def get_overlap_ratio(set_a, set_b):
    """
    Compute overlap ratio: size(intersection) / size(set_a)
    Assumes size(set_a) == size(set_b) roughly, or normalized by set_a size.
    """
    if len(set_a) == 0:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    return intersection / len(set_a)

def is_mme_prediction_correct(pred, gt):
    """
    Keep the same correctness rule as eval_mme.py:
    lowercase string exact match after prediction cleanup.
    """
    return str(pred).lower() == str(gt).lower()

def visualize_heatmap(matrix, labels, title, output_path):
    """
    Generate and save a heatmap.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=False, cmap="Blues", xticklabels=labels, yticklabels=labels, vmin=0, vmax=1)
    
    plt.title(title)
    plt.xlabel("Layer Index")
    plt.ylabel("Layer Index")
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved heatmap to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", type=str, default="1", help="GPU ID to use")
    parser.add_argument("--model-dtype", type=str, default="float32", choices=["float32", "bfloat16", "float16"],
                        help="Compute dtype used for model and image tensor")
    parser.add_argument("--max-categories", type=int, default=0,
                        help="If > 0, only process the first N categories (for quick debugging)")
    parser.add_argument("--save-nan-diagnostics", action="store_true",
                        help="Save per-layer NaN diagnostics JSON for each category")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    # Configuration
    model_path = "/root/nfs/model/llava-v1.5-7b"
    mme_json_path = os.path.join(project_root, "data/MME/MME_test.json")
    image_folder = os.path.join(project_root, "data/MME/images/test")
    output_dir = os.path.join(project_root, "mme_layer_overlap_results")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print("Loading LLaVA model...")
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    try:
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, None, model_name
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    compute_dtype = dtype_map[args.model_dtype]

    # Use to(dtype=...) for explicit control.
    model = model.to(dtype=compute_dtype)
    if hasattr(model.get_model(), 'mm_projector'):
        model.get_model().mm_projector.to(dtype=compute_dtype)
    if hasattr(model.get_model(), 'vision_tower'):
        model.get_model().vision_tower.to(dtype=compute_dtype)
    print(f"Model compute dtype: {compute_dtype}")

    # Disable FastV during inference to get full attention maps (we want to analyze unmodified behavior)
    if hasattr(model.config, "use_fast_v"):
        print("Disabling FastV to capture full attention maps...")
        model.config.use_fast_v = False
        if hasattr(model, "model") and hasattr(model.model, "reset_fastv"):
            model.model.reset_fastv()
        # 【新增】必须直接修改模型实例内部的硬编码属性！
    if hasattr(model, "model"):
        model.model.use_fast_v = False
        model.model.use_himap = False # 如果有这个开关一并关掉
        if hasattr(model.model, "reset_fastv"):
            model.model.reset_fastv()
            model.model.use_fast_v = False # reset之后再强制赋False一次
    
    # Parameters for image token extraction
    # Standard LLaVA-1.5 uses 576 tokens.
    # System prompt length is typically 35.
    sys_length = 35 
    image_token_len = 576
    
    # Get one sample per subtask
    samples = load_mme_samples(mme_json_path)
    
    # Iterate over samples
    sample_items = list(samples.items())
    if args.max_categories > 0:
        sample_items = sample_items[:args.max_categories]

    for category, sample in tqdm(sample_items, desc="Processing categories"):
        image_file = sample.get('image_file')
        question = sample.get('question')
        gt_answer = sample.get('answer', '')
        
        if not image_file or not question:
            continue
            
        # Construct image path
        image_path = os.path.join(image_folder, image_file)
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found. Skipping {category}.")
            continue
            
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            continue
            
        # Prepare inputs
        image_tensor = process_images([image], image_processor, model.config)
        if type(image_tensor) is list:
            image_tensor = [img.to(model.device, dtype=compute_dtype) for img in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=compute_dtype)

        # Prepare prompt (aligned with eval_mme.py)
        qs = question
        if getattr(model.config, 'mm_use_im_start_end', False):
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        qs = qs + '\n' + "Answer the question using a single word or phrase."

        conv = conv_templates['vicuna_v1'].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        
        # Run inference
        # We only need the forward pass to get attentions, we don't need to generate text generally.
        # But let's run a single forward pass.
        with torch.inference_mode():
            outputs = model(
                input_ids,
                images=image_tensor,
                output_hidden_states=True, # 开启隐藏状态输出
                output_attentions=True,
                return_dict=True
            )


            stopping_criteria = [KeywordsStoppingCriteria(keywords, tokenizer, input_ids)] if conv.version == "v0" else None

            # # Extra generation for per-sample terminal logging.
            # output_ids = model.generate(
            #     input_ids,
            #     images=image_tensor,
            #     do_sample=False,
            #     temperature=0.0,
            #     max_new_tokens=1024,
            #     use_cache=False,
            #     stopping_criteria=stopping_criteria,
            # )
            
            print("Image tensor has NaN:", torch.isnan(image_tensor).any().item())
            print("Image tensor has Inf:", torch.isinf(image_tensor).any().item())
            
                # 逐层检查 hidden_states 到底在哪一层变成了 NaN
            for i, hidden_state in enumerate(outputs.hidden_states):
                if torch.isnan(hidden_state).any():
                    print(f"🚨 NaN 首次出现在第 {i} 层的 Hidden States!")
                    break

            # 逐层检查 Attention Map
            for i, attn_map in enumerate(outputs.attentions):
                if torch.isnan(attn_map).any():
                    print(f"🚨 NaN 首次出现在第 {i} 层的 Attention Map!")
                    break

        # input_token_len = input_ids.shape[1]
        # pred_text = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0].strip()
        # if pred_text.endswith(stop_str):
        #     pred_text = pred_text[:-len(stop_str)]
        # pred_text = pred_text.strip()

        # pred_for_eval = pred_text[:-1] if pred_text.endswith('.') else pred_text

        # if isinstance(gt_answer, str) and gt_answer.strip() != "":
        #     is_correct = is_mme_prediction_correct(pred_for_eval, gt_answer)
        #     print(
        #         f"[Sample Result] category={category} | pred={pred_for_eval} | gt={gt_answer} | correct={is_correct}"
        #     )
        # else:
        #     print(f"[Sample Result] category={category} | pred={pred_for_eval} | gt=N/A | correct=N/A")
            
        attentions = outputs.attentions # Tuple of (batch, head, seq, seq)
        num_layers = len(attentions)

        nan_diagnostics = {
            "category": category,
            "image_file": image_file,
            "num_layers": int(num_layers),
            "layers": {}
        }



        if isinstance(image_tensor, list):
            nan_diagnostics["image_tensor"] = [tensor_nan_stats(t) for t in image_tensor]
        else:
            nan_diagnostics["image_tensor"] = tensor_nan_stats(image_tensor)
        
        # Collect top indices per layer
        top1_indices = {}
        top10_indices = {}
        top30_attn_data = {}
        
        # Calculate K for 1% and 10%
        k_1pct = max(1, int(image_token_len * 0.30))
        k_10pct = max(1, int(image_token_len * 0.50))

        for layer_idx, layer_attn in enumerate(attentions):
            # layer_attn: (batch=1, heads, seq_len, seq_len)
            # User request: Sum over heads, then sum over queries (dim=1 of the remaining)
            # This calculates total attention received by each token from all other tokens
            layer_diag = {}
            layer_diag["layer_attn"] = tensor_nan_stats(layer_attn)


            sum_heads = layer_attn.sum(dim=1)
            layer_diag["after_sum_heads"] = tensor_nan_stats(sum_heads)

            sum_heads_queries = sum_heads.sum(dim=1)
            layer_diag["after_sum_queries"] = tensor_nan_stats(sum_heads_queries)

            token_importance = sum_heads_queries[0] # (seq_len,)
            layer_diag["token_importance"] = tensor_nan_stats(token_importance)
            
            # Extract image part
            # Check length to avoid index errors
            if token_importance.shape[0] < sys_length + image_token_len:
                print(f"Warning: Sequence length {token_importance.shape[0]} too short for category {category}. Skipping.")
                break

            image_attn_scores = token_importance[sys_length : sys_length + image_token_len]
            layer_diag["image_attn_scores"] = tensor_nan_stats(image_attn_scores)
            
            # Get indices with highest attention scores using topk
            top1_vals_t, top1_indices_t = torch.topk(image_attn_scores, k_1pct)
            layer_diag["top1_values"] = tensor_nan_stats(top1_vals_t)
            top1_indices[layer_idx] = set(top1_indices_t.cpu().numpy())
            
            # top30_attn_data[str(layer_idx)] = {
            #     "indices": top1_indices_t.cpu().tolist(),
            #     "values": top1_vals_t.cpu().tolist()
            # }
            
            _, top10_indices_t = torch.topk(image_attn_scores, k_10pct)
            top10_indices[layer_idx] = set(top10_indices_t.cpu().numpy())

            nan_diagnostics["layers"][str(layer_idx)] = layer_diag

        # Compute overlap matrices
        matrix_1 = np.zeros((num_layers, num_layers))
        matrix_10 = np.zeros((num_layers, num_layers))
        
        for i in range(num_layers):
            for j in range(num_layers):
                if i in top1_indices and j in top1_indices:
                     # Jaccard or simple Intersection/Size?
                     # Request says "Compare pairwise overlap". Let's use Intersection / Size (which is constant).
                     # This represents "What fraction of Layer I's top tokens are also in Layer J's top tokens?"
                    matrix_1[i, j] = get_overlap_ratio(top1_indices[i], top1_indices[j])
                    matrix_10[i, j] = get_overlap_ratio(top10_indices[i], top10_indices[j])
        
        # Generate clean filename from category
        safe_cat = category.replace("/", "_").replace(" ", "_")
        
        # Save top 30% attention data to JSON
        # json_path = os.path.join(output_dir, f"{safe_cat}_top30_attn_values.json")
        # with open(json_path, 'w') as f:
        #     json.dump(top30_attn_data, f, indent=4)

        if args.save_nan_diagnostics:
            diag_path = os.path.join(output_dir, f"{safe_cat}_nan_diagnostics.json")
            with open(diag_path, 'w') as f:
                json.dump(nan_diagnostics, f, indent=4)
            print(f"Saved NaN diagnostics to {diag_path}")
        
        # Save heatmaps
        visualize_heatmap(matrix_1, list(range(num_layers)), 
                         f"{category} - Top30% Token Overlap", 
                         os.path.join(output_dir, f"{safe_cat}_top30_heatmap.png"))
        
        visualize_heatmap(matrix_10, list(range(num_layers)), 
                         f"{category} - Top 50% Token Overlap", 
                         os.path.join(output_dir, f"{safe_cat}_top50_heatmap.png"))
        
    print(f"\nProcessing complete. All results saved to {output_dir}")

if __name__ == "__main__":
    main()
