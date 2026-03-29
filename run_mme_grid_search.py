import json
import os
import subprocess
import math
import csv
import time
import gc
from PIL import Image, ImageDraw, ImageFont


LAYERS = [8, 10, 12, 14]
RATIOS = [0.5, 0.7, 0.9]
IMG_TOKENS = 576  # Default LLaVA image tokens
MODEL_PATH = "/root/nfs/model/llava-v1.5-7b"
QUESTION_FILE_ORIG = "/root/nfs/code/HiMAP/data/MME/MME_test.json"
QUESTION_FILE_FILTERED = "/root/nfs/code/HiMAP/data/MME/MME_test_filtered.json"
IMAGE_FOLDER = "/root/nfs/code/HiMAP/data/MME/images/test"
RESULT_FILE_TEMP = "mme_results_fastv.json"

TARGET_CATEGORIES = ["existence", "color", "count", "OCR"]

def filter_questions():
    print(f"Reading {QUESTION_FILE_ORIG}...")
    if not os.path.exists(QUESTION_FILE_ORIG):
        print(f"Error: {QUESTION_FILE_ORIG} does not exist.")
        return False

    with open(QUESTION_FILE_ORIG, 'r') as f:
        data = json.load(f)
    
    # Filter categories if needed
    filtered_data = [
        item for item in data 
        if item.get('category') in TARGET_CATEGORIES
    ]

    print(f"Filtered {len(filtered_data)} questions out of {len(data)}")
    
    with open(QUESTION_FILE_FILTERED, 'w') as f:
        json.dump(filtered_data, f, indent=2)
    print(f"Saved filtered questions to {QUESTION_FILE_FILTERED}")
    return True

def run_evaluation(layer, ratio):
    rank = int(IMG_TOKENS * ratio)
    print(f"\nWork for Layer={layer}, Ratio={ratio:.1f}, Rank={rank}")
    
    # Construct command
    cmd = [
        "python", "./src/HiMAP/inference/eval_mme.py",
        "--model-path", MODEL_PATH,
        "--question-file", QUESTION_FILE_FILTERED,
        "--image-folder", IMAGE_FOLDER,
        "--use-fast-v",
        "--fast-v-sys-length", "35",
        "--fast-v-image-token-length", str(IMG_TOKENS),
        "--fast-v-attention-rank", str(rank),
        "--fast-v-agg-layer", str(layer),
        "--temperature", "0"
    ]

    print("Run finished, waiting for GPU memory release...")
    time.sleep(10)  # 等待 10 秒，让 OS 回收显存
    
    return execute_command(cmd)

def run_baseline():
    print(f"\nWork for Baseline (No Pruning)")
    
    # Construct command (no fast-v args)
    cmd = [
        "python", "./src/HiMAP/inference/eval_mme.py",
        "--model-path", MODEL_PATH,
        "--question-file", QUESTION_FILE_FILTERED,
        "--image-folder", IMAGE_FOLDER,
        "--temperature", "0"
    ]
    
    # The output filename in eval_mme.py changes based on args.
    # We need to make sure we read the correct file or rely on eval_mme behavior.
    # mme_results_baseline.json will be created by eval_mme.py if no fastv/himap args.
    
    return execute_command(cmd, result_filename="mme_results_baseline.json")

def execute_command(cmd, result_filename=RESULT_FILE_TEMP):
    # Environment variables
    env = os.environ.copy()
    pp = env.get("PYTHONPATH", "")
    src_llava = os.path.abspath("src/LLaVA")
    env["PYTHONPATH"] = f"{src_llava}:{pp}" if pp else src_llava
    
    # Force CUDA_VISIBLE_DEVICES if not set, to avoid device_map='auto' spanning all GPUs
    if "CUDA_VISIBLE_DEVICES" not in env:
        env["CUDA_VISIBLE_DEVICES"] = "2"
        print("Note: Setting CUDA_VISIBLE_DEVICES=2 by default.")
    
    try:
        if not os.path.exists(MODEL_PATH):
             pass

        print(f"Executing: {' '.join(cmd)}")
        # Check=True ensures we catch errors. Use run to wait for completion.
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running evaluation: {e}")
        return None

    # Load result
    if os.path.exists(result_filename):
        try:
            with open(result_filename, 'r') as f:
                res = json.load(f)
            return res
        except json.JSONDecodeError:
            print(f"Error deciding JSON from {result_filename}")
            return None
    else:
        print(f"Result file {result_filename} not found!")
        return None

def create_table_image(headers, rows, output_file="mme_grid_search_table.png"):
    """
    Generate a simple table image using PIL.
    """
    # Configuration
    padding = 10
    font_size = 15
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    
    # Calculate column widths
    col_widths = [0] * len(headers)
    all_data = [headers] + rows
    
    # Temporary image for measuring text
    temp_img = Image.new('RGB', (1, 1))
    draw = ImageDraw.Draw(temp_img)
    
    for row in all_data:
        for i, cell in enumerate(row):
            bbox = draw.textbbox((0, 0), str(cell), font=font)
            width = bbox[2] - bbox[0]
            col_widths[i] = max(col_widths[i], width + 2 * padding)
            
    total_width = sum(col_widths)
    row_height = int(font_size * 2.5)  # More spacing
    total_height = row_height * len(all_data)
    
    # Create image
    img = Image.new('RGB', (total_width, total_height), color='white')
    draw = ImageDraw.Draw(img)
    
    y = 0
    for row_idx, row in enumerate(all_data):
        x = 0
        bg_color = '#dddddd' if row_idx == 0 else ('#f9f9f9' if row_idx % 2 == 0 else 'white')
        
        # Draw row background
        draw.rectangle([0, y, total_width, y + row_height], fill=bg_color)
        
        for i, cell in enumerate(row):
            # If the cell is the header or matches max logic (simple string matching here won't work perfectly for bolding, 
            # but we can check if it looks like a bold markdown string "**val**")
            text = str(cell)
            is_bold = text.startswith("**") and text.endswith("**")
            display_text = text.replace("**", "")
            
            # Simple bold simulation if we don't have a bold font loaded
            text_color = 'black'
            if is_bold and row_idx > 0:
                text_color = 'red' # Highlight max values in red
                
            draw.text((x + padding, y + padding), display_text, fill=text_color, font=font)
            
            # Draw vertical line
            draw.line([(x + col_widths[i], y), (x + col_widths[i], y + row_height)], fill='#cccccc', width=1)
            
            x += col_widths[i]
            
        # Draw horizontal line
        draw.line([(0, y + row_height), (total_width, y + row_height)], fill='#cccccc', width=1)
        y += row_height
        
    # Draw border
    draw.rectangle([0, 0, total_width - 1, total_height - 1], outline='black', width=1)
    
    img.save(output_file)
    print(f"Table image saved to {output_file}")


def main():
    if not filter_questions():
        return
    
    all_results = []
    
    # 1. Run Baseline
    base_res = run_baseline()
    if base_res:
        scores = base_res.get('scores', {})
        entry = {
            'Layer': 'Baseline',
            'Ratio': '1.0',
            'Rank': IMG_TOKENS,
        }
        total_score = 0
        for cat in TARGET_CATEGORIES:
            val = scores.get(cat, 0.0)
            entry[cat] = val
            total_score += val
        entry['Total'] = total_score
        all_results.append(entry)

    # 2. Run Grid Search
    for layer in LAYERS:
        for ratio in RATIOS:
            res = run_evaluation(layer, ratio)
            if res:
                scores = res.get('scores', {})
                entry = {
                    'Layer': layer,
                    'Ratio': ratio,
                    'Rank': int(IMG_TOKENS * ratio),
                }
                total_score = 0
                for cat in TARGET_CATEGORIES:
                    val = scores.get(cat, 0.0)
                    entry[cat] = val
                    total_score += val
                entry['Total'] = total_score
                
                # Append to list
                all_results.append(entry)

    if not all_results:
        print("No results collected.")
        return

    # 3. Find Max Values for Bolding
    max_vals = {}
    check_cols = TARGET_CATEGORIES + ['Total']
    for col in check_cols:
        max_vals[col] = max(r[col] for r in all_results)

    # Format table
    headers = ["Layer", "Ratio", "Rank"] + TARGET_CATEGORIES + ["Total"]
    
    # Create rows and format strings
    formatted_rows = []
    raw_rows = [] # For CSV
    
    for r in all_results:
        # CSV Row (raw values)
        csv_row = [str(r['Layer']), str(r['Ratio']), str(r['Rank'])]
        for cat in TARGET_CATEGORIES:
            csv_row.append(f"{r[cat]:.2f}")
        csv_row.append(f"{r['Total']:.2f}")
        raw_rows.append(csv_row)
        
        # Markdown/Display Row (with bolding)
        md_row = [str(r['Layer']), str(r['Ratio']), str(r['Rank'])]
        for cat in TARGET_CATEGORIES:
            val = r[cat]
            s_val = f"{val:.2f}"
            if val == max_vals[cat]:
                s_val = f"**{s_val}**"
            md_row.append(s_val)
            
        # Total column
        val_total = r['Total']
        s_total = f"{val_total:.2f}"
        if val_total == max_vals['Total']:
            s_total = f"**{s_total}**"
        md_row.append(s_total)
        
        formatted_rows.append(md_row)
    
    # Print table to console
    print("\nFinal Results Table:")
    print("-" * 120)
    print(" | ".join(f"{h:<12}" for h in headers))
    print("-" * 120)
    for row in formatted_rows:
        print(" | ".join(f"{val:<12}" for val in row))
    print("-" * 120)
    
    # Save CSV
    try:
        with open("mme_grid_search_results.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(raw_rows)
        print("Results saved to mme_grid_search_results.csv")
    except Exception as e:
        print(f"Failed to save CSV: {e}")

    # Save Markdown
    try:
        with open("MME_GRID_SEARCH_RESULTS.md", "w") as f:
            f.write("# MME Grid Search Results (FastV)\n\n")
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
            for r in formatted_rows:
                f.write("| " + " | ".join(r) + " |\n")
        print("Results saved to MME_GRID_SEARCH_RESULTS.md")
    except Exception as e:
        print(f"Failed to save Markdown: {e}")

    # Create Image
    try:
        create_table_image(headers, formatted_rows)
    except Exception as e:
        print(f"Failed to create table image: {e}")

if __name__ == "__main__":
    main()
