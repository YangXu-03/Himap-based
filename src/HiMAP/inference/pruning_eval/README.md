# Pruning Evaluation Scripts

Each script is now dataset-specific.

## MME Scripts

- Single run: `./src/HiMAP/inference/pruning_eval/run_pruning_eval.sh`
- All modes: `./src/HiMAP/inference/pruning_eval/run_all_modes.sh`

### MME Flow (same idea as mme.sh)

1. `python -m llava.eval.model_vqa_loader` to generate `answers/*.jsonl`
2. `convert_answer_to_mme.py --experiment <EXP_NAME>` to build MME eval files
3. `python ./src/HiMAP/eval_tool/calculation.py --results_dir ...` for final score

### MME Examples

```bash
MODE=baseline EXP_NAME=llava-v1.5-7b-baseline bash ./src/HiMAP/inference/pruning_eval/run_pruning_eval.sh
```

```bash
MODE=himap EXP_NAME=llava-v1.5-7b-himap GPU_ID=0 bash ./src/HiMAP/inference/pruning_eval/run_pruning_eval.sh
```

```bash
MODE=fastv EXP_NAME=llava-v1.5-7b-fastv GPU_ID=0 bash ./src/HiMAP/inference/pruning_eval/run_pruning_eval.sh
```

```bash
MODE=jsd_entropy EXP_NAME=llava-v1.5-7b-jsd GPU_ID=0 bash ./src/HiMAP/inference/pruning_eval/run_pruning_eval.sh
```

## ScienceQA Scripts

- Single run: `./src/HiMAP/inference/pruning_eval/run_scienceqa_eval.sh`
- All modes: `./src/HiMAP/inference/pruning_eval/run_scienceqa_all_modes.sh`

### ScienceQA Examples

```bash
MODE=baseline bash ./src/HiMAP/inference/pruning_eval/run_scienceqa_eval.sh
```

```bash
MODE=himap GPU_ID=0 NUM_SAMPLES=-1 bash ./src/HiMAP/inference/pruning_eval/run_scienceqa_eval.sh
```

## Common Variables

- `MODEL_PATH` default: `/root/nfs/model/llava-v1.5-7b`
- `GPU_ID` default: `0`
- `MODE`: `baseline|himap|fastv|jsd_entropy`
- `SYS_LENGTH` default: `35`
- `IMG_LENGTH` default: `576`

### HiMAP

- `HMAP_TXT_LAYER` default: `2`
- `HMAP_IMG_LAYER` default: `8`
- `HMAP_TXT_RANK` default: `128`
- `HMAP_IMG_RANK` default: `72`

### FastV

- `FASTV_RANK` default: `128`
- `FASTV_AGG_LAYER` default: `2`

### JSD+Entropy

- `JSD_TOPK_PERCENT` default: `10`
- `JSD_STAGE_RANGES` default: `2-8,9-20,21-31`
- `JSD_STAGE_PRUNE_RATIOS` default: `0.1,0.4,0.5`

## MME Path Variables

- `MME_ROOT` default: `./playground/data/eval/MME`
- `QUESTION_FILE` default: `${MME_ROOT}/llava_mme.jsonl`
- `IMAGE_FOLDER` default: `${MME_ROOT}/MME_Benchmark_release_version`
- `ANSWERS_DIR` default: `${MME_ROOT}/answers`
- `EXP_NAME` default: `llava-v1.5-7b-<MODE>`
- `CONVERT_SCRIPT` default: `${MME_ROOT}/convert_answer_to_mme.py`

## ScienceQA Path Variables

- `QUESTION_FILE` default: `/root/nfs/code/HiMAP/data/scienceqa/himap-inference-MCQ.json`
- `IMAGE_FOLDER` default: `/root/nfs/code/HiMAP/data/scienceqa/images/test`
- `NUM_SAMPLES` default: `-1`
