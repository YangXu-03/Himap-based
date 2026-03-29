import torch

from llava.constants import IMAGE_TOKEN_INDEX


def resolve_attention_spans(input_ids, seq_len, image_token_len, fallback_sys_length=36):
    """
    Resolve the image-token span from the actual prompt rather than a hard-coded
    system length.

    `input_ids` still contains a single `IMAGE_TOKEN_INDEX` placeholder, while
    the model attention expands it into `image_token_len` visual tokens. The
    placeholder position therefore matches the start index of the expanded image
    token block in attention matrices.
    """
    if input_ids.dim() == 2:
        token_ids = input_ids[0].tolist()
    else:
        token_ids = input_ids.tolist()

    try:
        image_start = token_ids.index(IMAGE_TOKEN_INDEX)
    except ValueError:
        image_start = fallback_sys_length

    image_end = min(image_start + image_token_len, seq_len)
    prompt_start = image_end
    prompt_end = seq_len
    return image_start, image_end, prompt_start, prompt_end


def sanitize_scores(scores):
    """Replace NaN/Inf before top-k selection."""
    return torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)


def get_received_attention_scores(layer_attn, image_start, image_end):
    """
    Baseline score: total attention received by each image token from all heads
    and all query positions.
    """
    token_importance = layer_attn.sum(dim=1).sum(dim=1)[0]
    return sanitize_scores(token_importance[image_start:image_end])


def get_prompt_image_received_scores(layer_attn, image_start, image_end, prompt_start, prompt_end):
    """
    Attention received by each image token from image tokens and prompt tokens.

    To avoid the residual all-ones heatmaps caused by the causal mask, this
    function uses a visibility-corrected aggregation:
    - prompt -> image: mean over prompt queries
    - image -> image: sum over valid image queries, then divide each key token
      by the number of image queries that are causally allowed to see it

    The final score is the sum of the prompt contribution and the normalized
    image contribution for each image token.
    """
    attn = layer_attn[0]
    image_len = image_end - image_start

    # 【新增：抓取真实的底层异常】
    if torch.isnan(attn).any():
        print(f"\n[诊断] 警告！注意力矩阵中存在 NaN！")
    if (attn == 0.0).all():
        print(f"\n[诊断] 警告！注意力矩阵原生全是 0.0！")

    # 转为 float32 运算，防止求和时精度溢出
    attn = attn.to(torch.float32)
    # 提前填充 NaN/Inf，防止由于 causal mask 产生的 NaN 通过 .mean()/.sum() 传播
    attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)

    if image_len <= 0:
        return sanitize_scores(attn.new_zeros(0))

    image_to_image = attn[:, image_start:image_end, image_start:image_end]
    image_scores = image_to_image.mean(dim=0).sum(dim=0)
    visible_query_counts = torch.arange(
        image_len,
        0,
        -1,
        device=image_scores.device,
        dtype=image_scores.dtype,
    )
    # 加入 eps=1e-9 防止除零
    image_scores = image_scores / (visible_query_counts + 1e-9)

    if prompt_start < prompt_end:
        prompt_to_image = attn[:, prompt_start:prompt_end, image_start:image_end]
        prompt_scores = prompt_to_image.mean(dim=0).mean(dim=0)
    else:
        prompt_scores = torch.zeros_like(image_scores)

    return sanitize_scores(prompt_scores + image_scores)


def get_avg_all_heads_scores(layer_attn, image_start, image_end, prompt_start, prompt_end):
    """Match `FastV` avg-all-heads selection logic."""
    avg_attention = layer_attn.mean(dim=1)
    if prompt_start < prompt_end:
        image_scores = avg_attention[0, prompt_start:prompt_end, image_start:image_end].mean(dim=0)
    else:
        image_scores = avg_attention[0, :, image_start:image_end].mean(dim=0)
    return sanitize_scores(image_scores)


def get_text_to_image_scores(layer_attn, image_start, image_end, prompt_start, prompt_end):
    """
    Unweighted text-to-image attention.
    Average over heads and prompt tokens to avoid query-count scale effects.
    """
    if prompt_start < prompt_end:
        text_to_image = layer_attn[0, :, prompt_start:prompt_end, image_start:image_end]
    else:
        text_to_image = layer_attn[0, :, :, image_start:image_end]
    image_scores = text_to_image.mean(dim=0).mean(dim=0)
    return sanitize_scores(image_scores)


def get_text_weighted_scores(layer_attn, image_start, image_end, prompt_start, prompt_end):
    """Match `FastV` text-weighted selection logic."""
    if prompt_start < prompt_end:
        prompt_attention_matrix = layer_attn[:, :, prompt_start:prompt_end, prompt_start:prompt_end]
        text_importance = prompt_attention_matrix.sum(dim=2).mean(dim=1)
        text_importance = torch.softmax(text_importance, dim=-1)

        text_to_image = layer_attn[:, :, prompt_start:prompt_end, image_start:image_end]
        weighted_text_to_image = text_to_image * text_importance.unsqueeze(1).unsqueeze(-1)
        image_scores = weighted_text_to_image.sum(dim=2).mean(dim=1)[0]
    else:
        image_scores = layer_attn[:, :, :, image_start:image_end].mean(dim=1).mean(dim=1)[0]
    return sanitize_scores(image_scores)


def get_topk_index_set(scores, k, return_details=False):
    topk_vals, topk_indices = torch.topk(scores, k)
    idx_set = set(topk_indices.detach().cpu().tolist())
    if return_details:
        return idx_set, topk_indices.detach().cpu().tolist(), topk_vals.detach().cpu().tolist()
    return idx_set
