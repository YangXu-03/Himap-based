# Visual Token Pruning Experiment for LLaVA

This directory contains scripts for running visual token pruning experiments on the ScienceQA dataset using the LLaVA model.

## Overview

The experiment investigates the effect of visual token pruning on LLaVA's performance by:

1. **Attention-based Visual Token Pruning**: At the beginning of the prefilling phase, we compute importance scores for visual tokens (using L2 norm as a proxy for attention scores) and retain only the top-k most important tokens.

2. **Accuracy Evaluation**: We evaluate the model's accuracy on ScienceQA test set across different token retention counts (from 576 to 20, with intervals of 10).

3. **Matrix Rank Analysis**: For each configuration, we extract the attention output matrices from the first and last LLM layers, isolate the visual token portions, and compute their numerical rank.

4. **Normalization and Visualization**: All metrics (accuracy, first layer rank, last layer rank) are normalized using min-max normalization and plotted together.

## Requirements

- LLaVA model (e.g., `liuhaotian/llava-v1.5-7b`)
- ScienceQA dataset with the following structure:
  ```
  scienceqa/
  ├── problems.json
  ├── pid_splits.json
  ├── llava_test_QCM-LEA.json
  └── images/
      └── test/
  ```

## Usage

### Quick Start

```bash
# Set environment variables
export MODEL_PATH="liuhaotian/llava-v1.5-7b"
export SCIENCEQA_DIR="./data/scienceqa"
export OUTPUT_DIR="./pruning_results"

# Run the experiment
bash scripts/run_pruning_experiment.sh
```

### Manual Execution

```bash
python -m scripts.pruning_experiment \
    --model-path liuhaotian/llava-v1.5-7b \
    --question-file ./data/scienceqa/llava_test_QCM-LEA.json \
    --image-folder ./data/scienceqa/images/test \
    --base-dir ./data/scienceqa \
    --output-dir ./pruning_results \
    --conv-mode llava_v1 \
    --single-pred-prompt \
    --num-samples 100  # Use -1 for all samples
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model-path` | Path to the LLaVA model | Required |
| `--model-base` | Base model path for LoRA models | None |
| `--question-file` | Path to the question JSON file | Required |
| `--image-folder` | Path to the image folder | Required |
| `--base-dir` | Directory containing problems.json and pid_splits.json | Required |
| `--output-dir` | Output directory for results | `./pruning_results` |
| `--conv-mode` | Conversation mode | `llava_v1` |
| `--temperature` | Temperature for generation | `0.2` |
| `--single-pred-prompt` | Use single prediction prompt | False |
| `--num-samples` | Number of samples to evaluate (-1 for all) | `-1` |

## Output

The experiment generates the following files in the output directory:

1. **raw_results.json**: Contains raw experiment results
   - `token_counts`: List of visual token counts tested
   - `accuracies`: Accuracy for each configuration
   - `first_layer_ranks`: Average matrix rank from first LLM layer
   - `last_layer_ranks`: Average matrix rank from last LLM layer

2. **normalized_results.json**: Contains normalized results
   - All raw values plus min-max normalized versions

3. **pruning_experiment_results.png/pdf**: Visualization showing:
   - Normalized accuracy curve
   - Normalized first layer rank curve  
   - Normalized last layer rank curve

## Technical Details

### Pruning Method

The pruning is performed based on feature magnitude (L2 norm) as a proxy for attention importance:

```python
importance_scores = torch.norm(image_features, dim=-1)
_, top_indices = torch.topk(importance_scores, k=num_tokens_to_keep, dim=1)
top_indices, _ = torch.sort(top_indices, dim=1)  # Maintain spatial order
pruned_features = torch.gather(image_features, dim=1, 
                               index=top_indices.unsqueeze(-1).expand(...))
```

### Rank Calculation

Matrix rank is computed using SVD with a relative threshold:

```python
U, S, V = torch.linalg.svd(matrix, full_matrices=False)
rank = (S > threshold * S.max()).sum().item()
```

### Min-Max Normalization

```python
normalized = (value - min_value) / (max_value - min_value)
```

## Example Results

Expected output plot shows three curves:
- **Blue line (Normalized Accuracy)**: Generally decreases as fewer tokens are retained
- **Red line (First Layer Rank)**: Shows how representation rank changes at early layers
- **Green line (Last Layer Rank)**: Shows how representation rank changes at final layers

The x-axis shows the number of visual tokens (from 576 to 20), and the y-axis shows normalized values (0 to 1).

## Citation

If you use this experiment in your research, please cite the original LLaVA paper:

```bibtex
@inproceedings{liu2023llava,
    author = {Liu, Haotian and Li, Chunyuan and Wu, Qingyang and Lee, Yong Jae},
    title = {Visual Instruction Tuning},
    booktitle = {NeurIPS},
    year = {2023}
}
```
