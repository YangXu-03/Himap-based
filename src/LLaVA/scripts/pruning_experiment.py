"""
Visual Token Pruning Experiment for LLaVA

This script implements an attention-based visual token pruning experiment on ScienceQA.
It performs:
1. Attention-based visual token pruning at the prefilling stage
2. Accuracy evaluation across different token retention counts (576 to 20, interval 10)
3. Matrix rank calculation for visual tokens from first and last LLM layers
4. Min-max normalization and visualization of all curves
"""

import argparse
import torch
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
from PIL import Image
from typing import Optional, Tuple

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path


def compute_matrix_rank(matrix, threshold=1e-5):
    """
    Compute the numerical rank of a matrix using SVD.
    
    Args:
        matrix: Input tensor
        threshold: Relative threshold for singular values
        
    Returns:
        Numerical rank of the matrix
    """
    if matrix is None or matrix.numel() == 0:
        return 0
    
    # Flatten if needed and convert to 2D
    if matrix.dim() > 2:
        matrix = matrix.reshape(-1, matrix.shape[-1])
    elif matrix.dim() == 1:
        matrix = matrix.unsqueeze(0)
    
    # Ensure matrix is on CPU and float for numerical stability
    matrix = matrix.detach().cpu().float()
    
    # Compute SVD
    try:
        U, S, V = torch.linalg.svd(matrix, full_matrices=False)
        # Compute rank based on relative threshold
        max_sv = S.max()
        if max_sv > 0:
            rank = (S > threshold * max_sv).sum().item()
        else:
            rank = 0
    except Exception:
        # Fallback to matrix_rank if SVD fails
        try:
            rank = torch.linalg.matrix_rank(matrix, tol=threshold).item()
        except:
            rank = 0
    
    return rank


def get_pred_idx(prediction, choices, options=["A", "B", "C", "D", "E"]):
    """Get the index from the prediction."""
    if prediction in options[:len(choices)]:
        return options.index(prediction)
    else:
        return -1


def parse_answer(pred_text, options=["A", "B", "C", "D", "E"]):
    """Parse answer from model output text."""
    if pred_text in options:
        return pred_text
    elif len(pred_text) >= 3 and pred_text[0] in options and pred_text[1:3] == ". ":
        return pred_text[0]
    else:
        pattern = re.compile(r'The answer is ([A-Z]).')
        res = pattern.findall(pred_text)
        if len(res) == 1:
            return res[0]
        else:
            return "FAILED"


class PruningModelWrapper:
    """
    A wrapper class that handles visual token pruning during inference.
    
    This wrapper intercepts the image encoding process and prunes visual tokens
    based on attention score (approximated by feature magnitude).
    """
    
    def __init__(self, model, tokenizer, num_tokens_to_keep=576):
        self.model = model
        self.tokenizer = tokenizer
        self.num_tokens_to_keep = num_tokens_to_keep
        self.last_first_layer_visual = None
        self.last_last_layer_visual = None
        self.visual_token_start = None
        self.visual_token_end = None
    
    def set_num_tokens(self, num_tokens):
        """Set the number of visual tokens to keep."""
        self.num_tokens_to_keep = num_tokens
    
    def encode_images_with_pruning(self, images):
        """
        Encode images and prune visual tokens based on importance scores.
        
        Uses L2 norm of visual features as a proxy for attention-based importance.
        """
        # Get raw image features from vision tower
        image_features = self.model.get_model().get_vision_tower()(images)
        # Project through mm_projector
        image_features = self.model.get_model().mm_projector(image_features)
        
        original_num_tokens = image_features.shape[1]
        num_to_keep = min(self.num_tokens_to_keep, original_num_tokens)
        
        if num_to_keep >= original_num_tokens:
            return image_features
        
        # Compute importance scores (L2 norm as attention proxy)
        with torch.no_grad():
            importance_scores = torch.norm(image_features, dim=-1)  # [batch, num_patches]
            
            # Get indices of top-k important tokens
            _, top_indices = torch.topk(importance_scores, k=num_to_keep, dim=1)
            # Sort to maintain spatial order
            top_indices, _ = torch.sort(top_indices, dim=1)
            
            # Gather the most important visual tokens
            pruned_features = torch.gather(
                image_features,
                dim=1,
                index=top_indices.unsqueeze(-1).expand(-1, -1, image_features.shape[-1])
            )
        
        return pruned_features
    
    def generate_with_pruning(
        self,
        input_ids: torch.Tensor,
        images: Optional[torch.Tensor],
        image_sizes: Optional[list],
        max_new_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> Tuple[str, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Generate output with pruned visual tokens.
        
        Returns:
            output_text: Generated text
            first_layer_visual: Hidden states from first LLM layer for visual tokens
            last_layer_visual: Hidden states from last LLM layer for visual tokens
        """
        self.last_first_layer_visual = None
        self.last_last_layer_visual = None
        
        if images is None:
            # No image, regular generation
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=None,
                    do_sample=temperature > 0,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                )
            output_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            return output_text, None, None
        
        # Encode and prune images
        pruned_image_features = self.encode_images_with_pruning(images)
        num_visual_tokens = pruned_image_features.shape[1]
        
        # Build input embeddings manually
        batch_size = input_ids.shape[0]
        embed_tokens = self.model.get_model().embed_tokens
        
        new_input_embeds_list = []
        visual_positions = []
        
        for batch_idx in range(batch_size):
            cur_input_ids = input_ids[batch_idx]
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            
            if len(image_token_indices) == 0:
                cur_embeds = embed_tokens(cur_input_ids)
                new_input_embeds_list.append(cur_embeds)
                visual_positions.append(None)
                continue
            
            # Build sequence with pruned visual tokens
            cur_new_embeds = []
            prev_idx = 0
            visual_start = None
            
            for img_token_idx in image_token_indices:
                # Text before image token
                if img_token_idx > prev_idx:
                    text_ids = cur_input_ids[prev_idx:img_token_idx]
                    cur_new_embeds.append(embed_tokens(text_ids))
                
                # Record visual token start position
                if visual_start is None:
                    visual_start = sum(e.shape[0] for e in cur_new_embeds)
                
                # Add pruned visual features
                cur_new_embeds.append(pruned_image_features[batch_idx])
                prev_idx = img_token_idx + 1
            
            # Text after last image token
            if prev_idx < len(cur_input_ids):
                text_ids = cur_input_ids[prev_idx:]
                cur_new_embeds.append(embed_tokens(text_ids))
            
            cur_new_embeds = [x.to(self.model.device) for x in cur_new_embeds]
            cur_new_embeds = torch.cat(cur_new_embeds, dim=0)
            new_input_embeds_list.append(cur_new_embeds)
            visual_positions.append((visual_start, visual_start + num_visual_tokens))
        
        # Pad sequences
        max_len = max(x.shape[0] for x in new_input_embeds_list)
        new_input_embeds = torch.zeros(
            batch_size, max_len, new_input_embeds_list[0].shape[-1],
            dtype=new_input_embeds_list[0].dtype,
            device=new_input_embeds_list[0].device
        )
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long, device=input_ids.device)
        
        for i, embeds in enumerate(new_input_embeds_list):
            new_input_embeds[i, :embeds.shape[0]] = embeds
            attention_mask[i, :embeds.shape[0]] = 1
        
        # Forward pass to get hidden states for rank calculation
        with torch.no_grad():
            forward_outputs = self.model.model(
                input_ids=None,
                attention_mask=attention_mask,
                inputs_embeds=new_input_embeds,
                output_hidden_states=True,
                return_dict=True,
            )
        
        hidden_states = forward_outputs.hidden_states
        # hidden_states[0] is embedding output, hidden_states[1] is first layer, hidden_states[-1] is last layer
        first_layer_output = hidden_states[1] if len(hidden_states) > 1 else hidden_states[0]
        last_layer_output = hidden_states[-1]
        
        # Extract visual token portions
        first_layer_visual = None
        last_layer_visual = None
        
        if visual_positions[0] is not None:
            vs, ve = visual_positions[0]
            first_layer_visual = first_layer_output[0, vs:ve, :].clone()
            last_layer_visual = last_layer_output[0, vs:ve, :].clone()
        
        self.last_first_layer_visual = first_layer_visual
        self.last_last_layer_visual = last_layer_visual
        
        # Generate output text
        # Since LlavaLlamaForCausalLM.generate doesn't support inputs_embeds directly,
        # we use the underlying model's generate capability through the grandparent class
        with torch.inference_mode():
            # Access the generate method from PreTrainedModel which does support inputs_embeds
            output_ids = super(type(self.model), self.model).generate(
                inputs_embeds=new_input_embeds,
                attention_mask=attention_mask,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
        
        output_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        return output_text, first_layer_visual, last_layer_visual


def run_experiment(args):
    """Run the pruning experiment."""
    
    # Setup
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    
    print(f"Loading model from {model_path}...")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name
    )
    model.eval()
    
    # Create pruning wrapper
    wrapper = PruningModelWrapper(model, tokenizer)
    
    # Load questions and problems
    print(f"Loading questions from {args.question_file}...")
    with open(args.question_file, 'r') as f:
        questions = json.load(f)
    
    # Load problems for evaluation
    base_dir = args.base_dir
    problems = args.image_folder
    
    # Limit samples if specified
    if args.num_samples > 0:
        questions = questions[:args.num_samples]
    
    print(f"Total samples to evaluate: {len(questions)}")
    
    # Token counts to experiment with (576 to 20, interval 10)
    # Generate: 576, 570, 560, 550, ..., 30, 20
    # First step is 576 -> 570, then regular 10-step intervals
    token_counts = [576] + list(range(570, 19, -10))
    
    print(f"Token counts to evaluate: {token_counts}")
    
    # Results storage
    results = {
        'token_counts': token_counts,
        'accuracies': [],
        'first_layer_ranks': [],
        'last_layer_ranks': [],
    }
    
    # Run experiments for each token count
    for num_tokens in token_counts:
        print(f"\n{'='*60}")
        print(f"Evaluating with {num_tokens} visual tokens...")
        print(f"{'='*60}")
        
        wrapper.set_num_tokens(num_tokens)
        
        correct = 0
        total = 0
        first_layer_ranks = []
        last_layer_ranks = []
        
        for i, line in enumerate(tqdm(questions, desc=f"Tokens={num_tokens}")):
            idx = line["id"]
            question = line['conversations'][0]
            qs = question['value'].replace('<image>', '').strip()
            
            # Process image if present
            if 'image' in line:
                image_file = line["image"]
                image_path = os.path.join(args.image_folder, image_file)
                
                if not os.path.exists(image_path):
                    continue
                    
                image = Image.open(image_path)
                image_tensor = process_images([image], image_processor, model.config)[0]
                images = image_tensor.unsqueeze(0).half().to(model.device)
                image_sizes = [image.size]
                
                if getattr(model.config, 'mm_use_im_start_end', False):
                    qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
                else:
                    qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            else:
                images = None
                image_sizes = None
            
            if args.single_pred_prompt:
                qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
            
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            input_ids = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
            ).unsqueeze(0).to(model.device)
            
            # Generate with pruning
            try:
                output_text, first_layer_visual, last_layer_visual = wrapper.generate_with_pruning(
                    input_ids,
                    images,
                    image_sizes,
                    max_new_tokens=1024,
                    temperature=args.temperature,
                )
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue
            
            # Compute ranks if we have visual token outputs
            if first_layer_visual is not None:
                first_rank = compute_matrix_rank(first_layer_visual)
                first_layer_ranks.append(first_rank)
            
            if last_layer_visual is not None:
                last_rank = compute_matrix_rank(last_layer_visual)
                last_layer_ranks.append(last_rank)
            
            # Evaluate answer
            if idx in problems:
                prob = problems[idx]
                answer = parse_answer(output_text)
                pred_idx = get_pred_idx(answer, prob['choices'])
                
                if pred_idx == prob['answer']:
                    correct += 1
                total += 1
        
        # Compute metrics for this token count
        accuracy = correct / total * 100 if total > 0 else 0
        avg_first_rank = np.mean(first_layer_ranks) if first_layer_ranks else 0
        avg_last_rank = np.mean(last_layer_ranks) if last_layer_ranks else 0
        
        results['accuracies'].append(accuracy)
        results['first_layer_ranks'].append(avg_first_rank)
        results['last_layer_ranks'].append(avg_last_rank)
        
        print(f"Tokens: {num_tokens}, Accuracy: {accuracy:.2f}%, "
              f"First Layer Rank: {avg_first_rank:.2f}, Last Layer Rank: {avg_last_rank:.2f}")
    
    # Save raw results
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'raw_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Apply min-max normalization
    def min_max_normalize(values):
        values = np.array(values)
        min_val = values.min()
        max_val = values.max()
        if max_val - min_val == 0:
            return np.zeros_like(values)
        return (values - min_val) / (max_val - min_val)
    
    norm_accuracies = min_max_normalize(results['accuracies'])
    norm_first_ranks = min_max_normalize(results['first_layer_ranks'])
    norm_last_ranks = min_max_normalize(results['last_layer_ranks'])
    
    # Save normalized results
    normalized_results = {
        'token_counts': results['token_counts'],
        'normalized_accuracies': norm_accuracies.tolist(),
        'normalized_first_layer_ranks': norm_first_ranks.tolist(),
        'normalized_last_layer_ranks': norm_last_ranks.tolist(),
        'raw_accuracies': results['accuracies'],
        'raw_first_layer_ranks': results['first_layer_ranks'],
        'raw_last_layer_ranks': results['last_layer_ranks'],
    }
    
    with open(os.path.join(args.output_dir, 'normalized_results.json'), 'w') as f:
        json.dump(normalized_results, f, indent=2)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    x = results['token_counts']
    
    plt.plot(x, norm_accuracies, 'b-o', label='Normalized Accuracy', linewidth=2, markersize=6)
    plt.plot(x, norm_first_ranks, 'r-s', label='Normalized First Layer Rank', linewidth=2, markersize=6)
    plt.plot(x, norm_last_ranks, 'g-^', label='Normalized Last Layer Rank', linewidth=2, markersize=6)
    
    plt.xlabel('Number of Visual Tokens', fontsize=14)
    plt.ylabel('Normalized Value (Min-Max)', fontsize=14)
    plt.title('Visual Token Pruning Experiment on ScienceQA', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.gca().invert_xaxis()  # Higher token counts on the left
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'pruning_experiment_results.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(args.output_dir, 'pruning_experiment_results.pdf'), bbox_inches='tight')
    
    print(f"\nResults saved to {args.output_dir}")
    print("Files generated:")
    print(f"  - raw_results.json")
    print(f"  - normalized_results.json")
    print(f"  - pruning_experiment_results.png")
    print(f"  - pruning_experiment_results.pdf")


def main():
    parser = argparse.ArgumentParser(description="Visual Token Pruning Experiment")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the LLaVA model")
    parser.add_argument("--model-base", type=str, default=None,
                        help="Base model path for LoRA models")
    parser.add_argument("--question-file", type=str, required=True,
                        help="Path to the question JSON file")
    parser.add_argument("--image-folder", type=str, required=True,
                        help="Path to the image folder")
    parser.add_argument("--base-dir", type=str, required=True,
                        help="Base directory containing problems.json and pid_splits.json")
    parser.add_argument("--output-dir", type=str, default="./pruning_results",
                        help="Output directory for results")
    parser.add_argument("--conv-mode", type=str, default="llava_v1",
                        help="Conversation mode")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Temperature for generation")
    parser.add_argument("--single-pred-prompt", action="store_true",
                        help="Use single prediction prompt")
    parser.add_argument("--num-samples", type=int, default=-1,
                        help="Number of samples to evaluate (-1 for all)")
    
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
