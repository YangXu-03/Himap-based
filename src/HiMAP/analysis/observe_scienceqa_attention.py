import argparse
import os
import json
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path

# 注意力类型定义
ATTN_TYPES = [
    "text_to_text",
    "text_to_visual",
    "visual_to_text",
    "visual_to_visual",
    "system_to_text",
    "system_to_visual",
    "text_to_system",
    "visual_to_system",
]

PLOT_TYPES = [
    "text_to_text",
    "text_to_visual",
    "visual_to_text",
    "visual_to_visual",
    "system_to_text",
    "system_to_visual",
]

TYPE_LABELS = {
    "text_to_text": "Text → Text",
    "text_to_visual": "Text → Visual",
    "visual_to_text": "Visual → Text",
    "visual_to_visual": "Visual → Visual",
    "system_to_text": "System → Text",
    "system_to_visual": "System → Visual",
    "text_to_system": "Text → System",
    "visual_to_system": "Visual → System",
}

def attention_share(attn: torch.Tensor, query_idx: List[int], key_idx: List[int]) -> float:
    if len(query_idx) == 0 or len(key_idx) == 0:
        return 0.0
    sub = attn[:, query_idx, :]
    sub = sub[:, :, key_idx]
    weights = sub.sum(dim=-1)
    return weights.mean().item()

def compute_token_groups(conv, prompt: str, tokenizer, input_ids: torch.Tensor, attentions: List[torch.Tensor]):
    image_token_indices = torch.where(input_ids == IMAGE_TOKEN_INDEX)[1]
    if len(image_token_indices) == 0:
        return [], [], []
    img_start_idx_input = image_token_indices[0].item()
    seq_len_output = attentions[0].shape[-1]
    num_patches = seq_len_output - input_ids.shape[1] + 1
    vis_indices = list(range(img_start_idx_input, img_start_idx_input + num_patches))

    system_tokens = tokenizer(conv.system + conv.sep, add_special_tokens=False).input_ids
    system_start = 1 if input_ids[0, 0].item() == tokenizer.bos_token_id else 0
    system_end = min(system_start + len(system_tokens), seq_len_output)
    system_indices = list(range(system_start, system_end))

    text_indices = [i for i in range(seq_len_output) if i not in vis_indices and i not in system_indices]
    return system_indices, text_indices, vis_indices

def process_sample(model, tokenizer, image_processor, line: Dict, args, device: torch.device):
    # ScienceQA格式：image, conversations
    image_file = line["image"]
    question = line["conversations"][0]["value"].replace("<image>", "").strip()
    image_path = os.path.join(args.image_folder, image_file)
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception:
        return None

    image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
    images = image_tensor.unsqueeze(0)
    if device.type == "cuda":
        images = images.half()
    images = images.to(device)

    # 构造 prompt
    if getattr(model.config, "mm_use_im_start_end", False):
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + question
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + question
    qs = qs + "\n" + "Answer the question using a single word or phrase."

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
    input_ids = input_ids.to(device)

    with torch.no_grad():
        outputs = model(
            input_ids,
            images=images,
            output_attentions=True,
            use_cache=True,
            return_dict=True,
        )

    attentions = outputs.attentions
    if attentions is None or len(attentions) == 0:
        return None

    system_idx, text_idx, vis_idx = compute_token_groups(conv, prompt, tokenizer, input_ids, attentions)
    if len(text_idx) == 0 or len(vis_idx) == 0:
        return None

    layer_attn = {k: [] for k in ATTN_TYPES}
    for attn in attentions:
        attn = attn.squeeze(0)
        layer_attn["text_to_text"].append(attention_share(attn, text_idx, text_idx))
        layer_attn["text_to_visual"].append(attention_share(attn, text_idx, vis_idx))
        layer_attn["visual_to_text"].append(attention_share(attn, vis_idx, text_idx))
        layer_attn["visual_to_visual"].append(attention_share(attn, vis_idx, vis_idx))
        layer_attn["system_to_text"].append(attention_share(attn, system_idx, text_idx))
        layer_attn["system_to_visual"].append(attention_share(attn, system_idx, vis_idx))
        layer_attn["text_to_system"].append(attention_share(attn, text_idx, system_idx))
        layer_attn["visual_to_system"].append(attention_share(attn, vis_idx, system_idx))

    return {
        "layers": list(range(len(attentions))),
        "attn": layer_attn,
    }

def average_attentions(records: List[Dict]):
    if not records:
        return {}, []
    num_layers = len(records[0]["layers"])
    sums = {k: np.zeros(num_layers, dtype=np.float64) for k in ATTN_TYPES}
    for rec in records:
        for k in ATTN_TYPES:
            sums[k] += np.array(rec["attn"][k])
    count = len(records)
    means = {k: (sums[k] / count).tolist() for k in ATTN_TYPES}
    layers = records[0]["layers"]
    return means, layers

def plot_overall(means: Dict[str, List[float]], layers: List[int], output_dir: str):
    plt.figure(figsize=(12, 6))
    for k in PLOT_TYPES:
        plt.plot(layers, means.get(k, []), label=TYPE_LABELS[k], linewidth=2)
    plt.xlabel("Layer")
    plt.ylabel("Attention mass")
    plt.title("ScienceQA Attention Breakdown (overall)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, "scienceqa_attention_overall.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="scienceqa_attention_observation")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-samples", type=int, default=-1)
    args = parser.parse_args()

    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, args.model_base, model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    if args.num_samples > 0:
        questions = questions[: args.num_samples]

    records = []
    for line in tqdm(questions, desc="Observing attention"):
        rec = process_sample(model, tokenizer, image_processor, line, args, device)
        if rec is not None:
            records.append(rec)

    torch.save(records, os.path.join(args.output_dir, "scienceqa_attention_records.pt"))

    means, layers = average_attentions(records)
    overall_plot = plot_overall(means, layers, args.output_dir)

    print(f"Saved {len(records)} samples of attention stats to {args.output_dir}")
    print(f"Overall plot: {overall_plot}")

if __name__ == "__main__":
    main()
