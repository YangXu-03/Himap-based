#!/usr/bin/env python3
"""
下载ScienceQA数据集并设置正确的目录结构
"""

import os
import json
from datasets import load_dataset
from PIL import Image
import requests
from tqdm import tqdm
import io

def download_scienceqa():
    """下载ScienceQA数据集"""
    print("正在下载ScienceQA数据集...")
    
    # 加载数据集
    dataset = load_dataset("xbench/ScienceQA")
    
    # 创建图片目录
    image_dir = "/root/nfs/code/HiMAP/data/scienceqa/images/test"
    os.makedirs(image_dir, exist_ok=True)
    
    # 处理测试集
    test_data = dataset['test']
    
    print(f"测试集包含 {len(test_data)} 个样本")
    
    # 下载图片并保存
    for i, item in enumerate(tqdm(test_data, desc="下载图片")):
        try:
            # 获取图片ID
            image_id = str(item['id'])
            
            # 创建子目录
            item_dir = os.path.join(image_dir, image_id)
            os.makedirs(item_dir, exist_ok=True)
            
            # 保存图片
            image_path = os.path.join(item_dir, "image.png")
            if not os.path.exists(image_path):
                # 如果数据集中有图片数据，直接保存
                if 'image' in item and item['image'] is not None:
                    image = item['image']
                    if hasattr(image, 'save'):
                        image.save(image_path)
                    else:
                        # 如果是字节数据
                        with open(image_path, 'wb') as f:
                            f.write(image)
                else:
                    print(f"警告: 样本 {image_id} 没有图片数据")
            
        except Exception as e:
            print(f"处理样本 {item.get('id', i)} 时出错: {e}")
            continue
    
    print("数据集下载完成！")
    print(f"图片保存在: {image_dir}")

if __name__ == "__main__":
    download_scienceqa()
