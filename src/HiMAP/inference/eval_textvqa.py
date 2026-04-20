import argparse
import json
import math
import os
import time
import uuid

import torch
from PIL import Image
from tqdm import tqdm

from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates
from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator
from llava.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENDPOINT", "https://hf-mirror.com")


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks."""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def parse_stage_ranges(range_text):
    if range_text is None:
        return [(2, 8), (9, 20), (21, 31)]
    ranges = []
    for part in str(range_text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" not in part:
            raise ValueError(f"Invalid stage range format: {part}")
        lo_s, hi_s = part.split("-", 1)
        ranges.append((int(lo_s), int(hi_s)))
    if not ranges:
        raise ValueError("No valid stage ranges parsed from --jsd-entropy-stage-ranges")
    return ranges


def parse_stage_prune_ratios(ratio_text):
    if ratio_text is None:
        return [0.2, 0.3, 0.5]
    ratios = [float(x.strip()) for x in str(ratio_text).split(",") if x.strip()]
    if len(ratios) == 0:
        raise ValueError("No valid prune ratios parsed from --jsd-entropy-stage-prune-ratios")
    return ratios


def find_textvqa_image_path(image_folder, image_id):
    base = str(image_id)
    candidates = [
        base,
        base + ".jpg",
        base + ".png",
        base + ".jpeg",
        os.path.join("train_images", base + ".jpg"),
        os.path.join("train_images", base + ".png"),
        os.path.join("train_images", base + ".jpeg"),
    ]
    for cand in candidates:
        path = os.path.join(image_folder, cand)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Cannot resolve TextVQA image for image_id={image_id} under {image_folder}")


def build_multimodal_question(question, model_config, add_short_answer_prompt):
    qs = question.strip()
    if getattr(model_config, "mm_use_im_start_end", False):
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if add_short_answer_prompt:
        qs = qs + "\n" + "Answer the question using a single word or phrase."
    return qs


def decode_generate_output(sequences, input_ids, tokenizer, stop_str):
    input_token_len = input_ids.shape[1]
    seq_token_len = sequences.shape[1]

    if seq_token_len >= input_token_len:
        n_diff_input_output = (input_ids != sequences[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids")
        decode_ids = sequences[:, input_token_len:]
    else:
        decode_ids = sequences

    outputs = tokenizer.batch_decode(decode_ids, skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    return outputs.strip()


def configure_pruning(model, args):
    if args.use_hmap_v:
        model.config.use_hmap_v = True
        model.config.use_fast_v = False
        model.config.use_jsd_entropy_pruning = False
        model.config.hmap_v_sys_length = args.sys_length
        model.config.hmap_v_img_length = args.img_length
        model.config.hmap_v_attn_txt_layer = args.hmap_v_attn_txt_layer
        model.config.hmap_v_attn_img_layer = args.hmap_v_attn_img_layer
        model.config.hmap_v_attn_txt_rank = args.hmap_v_attn_txt_rank
        model.config.hmap_v_attn_img_rank = args.hmap_v_attn_img_rank
        model.config.cut_off_layer = args.cut_off_layer
        print("HiMAP TECHNIQUE WILL BE USED ------")
        if hasattr(model.model, "reset_hmapv"):
            model.model.reset_hmapv()
        return

    if args.use_fast_v:
        model.config.use_hmap_v = False
        model.config.use_fast_v = True
        model.config.use_jsd_entropy_pruning = False
        model.config.fast_v_sys_length = args.fast_v_sys_length
        model.config.fast_v_image_token_length = args.fast_v_image_token_length
        model.config.fast_v_attention_rank = args.fast_v_attention_rank
        model.config.fast_v_agg_layer = args.fast_v_agg_layer
        model.config.fast_v_token_selection_method = args.fast_v_token_selection_method
        model.config.fast_v_weighted_alpha = args.fast_v_weighted_alpha
        print("FASTV TECHNIQUE WILL BE USED ------")
        print(f"  Token Selection Method: {args.fast_v_token_selection_method}")
        if args.fast_v_token_selection_method == "weighted_combination":
            print(f"  Weighted Alpha: {args.fast_v_weighted_alpha}")
        if hasattr(model.model, "reset_fastv"):
            model.model.reset_fastv()
        return

    if args.use_jsd_entropy_prune:
        adaptive_sys_len = args.jsd_entropy_sys_length if args.jsd_entropy_sys_length is not None else args.sys_length
        adaptive_img_len = args.jsd_entropy_img_length if args.jsd_entropy_img_length is not None else args.img_length
        if adaptive_sys_len is None or adaptive_img_len is None:
            raise ValueError(
                "Adaptive pruning requires --jsd-entropy-sys-length/--jsd-entropy-img-length (or fallback --sys-length/--img-length)."
            )

        model.config.use_hmap_v = False
        model.config.use_fast_v = False
        model.config.use_jsd_entropy_pruning = True
        model.config.jsd_entropy_sys_length = adaptive_sys_len
        model.config.jsd_entropy_image_token_length = adaptive_img_len
        model.config.jsd_entropy_topk_percent = args.jsd_entropy_topk_percent
        model.config.jsd_entropy_topk_attention_mode = args.jsd_entropy_topk_attention_mode
        model.config.jsd_entropy_stage_ranges = parse_stage_ranges(args.jsd_entropy_stage_ranges)
        model.config.jsd_entropy_stage_prune_ratios = parse_stage_prune_ratios(args.jsd_entropy_stage_prune_ratios)

        if args.jsd_entropy_target_tokens is not None:
            model.config.jsd_entropy_target_tokens = args.jsd_entropy_target_tokens
        if args.jsd_entropy_n0 is not None:
            model.config.jsd_entropy_n0 = args.jsd_entropy_n0
        if args.jsd_entropy_phase1_prune_layer is not None:
            model.config.jsd_entropy_phase1_prune_layer = args.jsd_entropy_phase1_prune_layer
        if args.jsd_entropy_phase2_prune_layer is not None:
            model.config.jsd_entropy_phase2_prune_layer = args.jsd_entropy_phase2_prune_layer
        if args.jsd_entropy_phase3_prune_layer is not None:
            model.config.jsd_entropy_phase3_prune_layer = args.jsd_entropy_phase3_prune_layer
        if args.jsd_entropy_mu_h is not None:
            model.config.jsd_entropy_mu_h = args.jsd_entropy_mu_h
        if args.jsd_entropy_sigma_h is not None:
            model.config.jsd_entropy_sigma_h = args.jsd_entropy_sigma_h
        if args.jsd_entropy_mu_w is not None:
            model.config.jsd_entropy_mu_w = args.jsd_entropy_mu_w
        if args.jsd_entropy_sigma_w is not None:
            model.config.jsd_entropy_sigma_w = args.jsd_entropy_sigma_w
        if args.jsd_entropy_alpha is not None:
            model.config.jsd_entropy_alpha = args.jsd_entropy_alpha
        if args.jsd_entropy_beta is not None:
            model.config.jsd_entropy_beta = args.jsd_entropy_beta

        print("ADAPTIVE JSD+ENTROPY 3-STAGE PRUNING WILL BE USED ------")
        if hasattr(model.model, "reset_jsd_entropy_pruning"):
            model.model.reset_jsd_entropy_pruning()
        return

    model.config.use_hmap_v = False
    model.config.use_fast_v = False
    model.config.use_jsd_entropy_pruning = False
    print("NO TOKEN PRUNING TECHNIQUE WILL BE USED ------")


def main(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, args.model_base, model_name)

    configure_pruning(model, args)

    annotations = json.load(open(os.path.expanduser(args.annotation_file), "r", encoding="utf-8"))
    if "data" not in annotations:
        raise ValueError(f"Invalid annotation file: missing 'data' key in {args.annotation_file}")

    questions = annotations["data"]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    if args.num_samples > 0:
        questions = questions[: args.num_samples]

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    evaluator = TextVQAAccuracyEvaluator()
    pred_list = []
    total_latency = 0.0

    with open(answers_file, "w", encoding="utf-8") as f_out:
        for line in tqdm(questions, desc="TextVQA"):
            question = line["question"].strip()
            image_id = line["image_id"]
            gt_answers = line.get("answers", [])

            prompt_suffix = "Answer the question using a single word or phrase." if args.single_pred_prompt else ""
            eval_prompt = question if not prompt_suffix else question + "\n" + prompt_suffix
            mm_prompt = build_multimodal_question(question, model.config, args.single_pred_prompt)

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], mm_prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()

            image_path = find_textvqa_image_path(args.image_folder, image_id)
            image = Image.open(image_path).convert("RGB")
            image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            if torch.cuda.is_available():
                images = image_tensor.unsqueeze(0).half().cuda()
            else:
                images = image_tensor.unsqueeze(0).float()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            stopping_criteria = [KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)] if conv.version == "v0" else None

            if args.use_jsd_entropy_prune and hasattr(model.model, "reset_jsd_entropy_pruning"):
                model.model.reset_jsd_entropy_pruning()

            with torch.inference_mode():
                t0 = time.time()
                output = model.generate(
                    input_ids,
                    images=images,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=False,
                    stopping_criteria=stopping_criteria,
                    return_dict_in_generate=True,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                total_latency += time.time() - t0

            pred = decode_generate_output(output["sequences"], input_ids, tokenizer, stop_str)
            pred_list.append({"pred_answer": pred, "gt_answers": gt_answers})

            record = {
                "question_id": image_id,
                "prompt": eval_prompt,
                "text": pred,
                "answer_id": str(uuid.uuid4()),
                "model_id": model_name,
                "metadata": {
                    "question": question,
                    "image": os.path.basename(image_path),
                },
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    acc = evaluator.eval_pred_list(pred_list) if pred_list else 0.0
    avg_latency = total_latency / max(len(pred_list), 1)

    print(f"Samples: {len(pred_list)}")
    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"Avg Latency/Example: {avg_latency:.6f}s")
    print(f"Answers saved to: {answers_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--annotation-file", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=-1)
    parser.add_argument("--single-pred-prompt", action="store_true")

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=64)

    parser.add_argument("--use-hmap-v", default=False, action="store_true", help="whether to use hmap-v")
    parser.add_argument("--sys-length", type=int, required=False, help="the length of system prompt")
    parser.add_argument("--img-length", type=int, required=False, help="the length of image token")
    parser.add_argument("--hmap-v-attn-txt-layer", type=int, required=False, help="the layer of pruning according to img2txt information")
    parser.add_argument("--hmap-v-attn-img-layer", type=int, required=False, help="the layer of pruning according to img2img information")
    parser.add_argument("--hmap-v-attn-txt-rank", type=int, required=False, help="the rank of attention according to img2txt information")
    parser.add_argument("--hmap-v-attn-img-rank", type=int, required=False, help="the rank of attention according to img2img information")
    parser.add_argument("--cut-off-layer", type=int, required=False, help="the layer index after which all image tokens are removed")

    parser.add_argument("--use-fast-v", default=False, action="store_true", help="whether to use fast-v")
    parser.add_argument("--fast-v-sys-length", type=int, required=False, help="the length of system prompt for fast-v")
    parser.add_argument("--fast-v-image-token-length", type=int, required=False, help="the length of image token for fast-v")
    parser.add_argument("--fast-v-attention-rank", type=int, required=False, help="the rank of attention for fast-v")
    parser.add_argument("--fast-v-agg-layer", type=int, required=False, help="the aggregation layer for fast-v")
    parser.add_argument(
        "--fast-v-token-selection-method",
        type=str,
        default="avg_all_heads",
        choices=["max_head", "avg_all_heads", "weighted_combination", "text_weighted", "text_weighted_max_head"],
        help="token selection strategy: max_head, avg_all_heads, weighted_combination, text_weighted, or text_weighted_max_head",
    )
    parser.add_argument("--fast-v-weighted-alpha", type=float, default=0.5, help="alpha weight for weighted_combination method (0.0 to 1.0)")

    parser.add_argument("--use-jsd-entropy-prune", default=False, action="store_true", help="whether to use adaptive top-k JSD+entropy 3-stage pruning")
    parser.add_argument("--jsd-entropy-sys-length", type=int, required=False, help="system token length for adaptive pruning")
    parser.add_argument("--jsd-entropy-img-length", type=int, required=False, help="image token length for adaptive pruning")
    parser.add_argument("--jsd-entropy-topk-percent", type=float, default=10.0, help="top-k percent used for JSD+entropy stage detection")
    parser.add_argument(
        "--jsd-entropy-topk-attention-mode",
        "--jsd_entropy_topk_attention_mode",
        dest="jsd_entropy_topk_attention_mode",
        type=str,
        default="prompt_image",
        choices=["prompt_image", "global"],
        help="whether to calculate JSD+entropy based top-k attention scores using only prompt+image tokens or all tokens",
    )
    parser.add_argument("--jsd-entropy-stage-ranges", type=str, default="2-8,9-20,21-31", help="layer ranges to pick 3 stage nodes, e.g. 2-8,9-20,21-31")
    parser.add_argument("--jsd-entropy-stage-prune-ratios", type=str, default="0.2,0.3,0.5", help="incremental prune ratios per stage, default 20%,30%,50%")
    parser.add_argument("--jsd-entropy-target-tokens", type=int, required=False, help="target number of image tokens for JSD entropy pruning")
    parser.add_argument("--jsd-entropy-n0", type=int, required=False, help="initial image token length n0 used in stage budgeting")
    parser.add_argument("--jsd-entropy-phase1-prune-layer", type=int, required=False, help="1-based layer index to apply phase-1 pruning")
    parser.add_argument("--jsd-entropy-phase2-prune-layer", type=int, required=False, help="1-based layer index to apply phase-2 pruning")
    parser.add_argument("--jsd-entropy-phase3-prune-layer", type=int, required=False, help="1-based layer index to drop all remaining image tokens")
    parser.add_argument("--jsd-entropy-mu-h", type=float, required=False, help="mean of H metric for z-score normalization")
    parser.add_argument("--jsd-entropy-sigma-h", type=float, required=False, help="std of H metric for z-score normalization")
    parser.add_argument("--jsd-entropy-mu-w", type=float, required=False, help="mean of W metric for z-score normalization")
    parser.add_argument("--jsd-entropy-sigma-w", type=float, required=False, help="std of W metric for z-score normalization")
    parser.add_argument("--jsd-entropy-alpha", type=float, required=False, help="alpha weight for H z-score term in phase-1 keep budget")
    parser.add_argument("--jsd-entropy-beta", type=float, required=False, help="beta weight for W z-score term in phase-1 keep budget")

    args = parser.parse_args()
    main(args)
