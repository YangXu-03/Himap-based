# filepath: download_mme.py
from datasets import load_dataset
import os
import json
from PIL import Image

dataset = load_dataset("lmms-lab/MME", cache_dir="./data/MME")
data = dataset['test']

# 创建图片保存目录
image_dir = "/root/nfs/code/HiMAP/data/MME/images/test"
os.makedirs(image_dir, exist_ok=True)

export_data = []
for idx, item in enumerate(data):
    item = dict(item)
    # 保存图片
    if "image" in item:
        img = item["image"]
        img_file = f"{idx}.png"
        img_path = os.path.join(image_dir, img_file)
        img.save(img_path)
        # 用图片文件名替换原图片字段
        item["image_file"] = img_file
        item.pop("image")
    export_data.append(item)

# 导出为 JSON
json_path = "/root/nfs/code/HiMAP/data/MME/MME_test.json"
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(export_data, f, ensure_ascii=False, indent=2)

print(f"图片已保存到: {image_dir}")
print(f"JSON已保存到: {json_path}")