#!/usr/bin/env python3
"""
创建测试图片用于验证ScienceQA脚本
"""

import os
import json
from PIL import Image, ImageDraw, ImageFont
import random

def create_test_images():
    """创建测试图片"""
    print("正在创建测试图片...")
    
    # 创建图片目录
    image_dir = "/root/nfs/code/HiMAP/data/scienceqa/images/test"
    os.makedirs(image_dir, exist_ok=True)
    
    # 读取JSON文件获取需要创建的图片列表
    json_file = "/root/nfs/code/HiMAP/data/scienceqa/himap-inference-MCQ.json"
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print(f"需要创建 {len(data)} 个测试图片")
    
    # 创建测试图片
    for i, item in enumerate(data[:10]):  # 只创建前10个作为测试
        try:
            image_id = str(item['id'])
            
            # 创建子目录
            item_dir = os.path.join(image_dir, image_id)
            os.makedirs(item_dir, exist_ok=True)
            
            # 创建测试图片
            image_path = os.path.join(item_dir, "image.png")
            
            # 创建一个简单的测试图片
            img = Image.new('RGB', (224, 224), color='white')
            draw = ImageDraw.Draw(img)
            
            # 添加一些文本
            try:
                # 尝试使用默认字体
                font = ImageFont.load_default()
            except:
                font = None
            
            # 绘制一些内容
            text = f"Test Image {image_id}"
            if font:
                draw.text((10, 10), text, fill='black', font=font)
            else:
                draw.text((10, 10), text, fill='black')
            
            # 绘制一些几何图形
            draw.rectangle([50, 50, 150, 150], outline='blue', width=2)
            draw.ellipse([60, 60, 140, 140], outline='red', width=2)
            
            # 保存图片
            img.save(image_path)
            
            if i % 100 == 0:
                print(f"已创建 {i+1} 个图片...")
                
        except Exception as e:
            print(f"创建图片 {item.get('id', i)} 时出错: {e}")
            continue
    
    print("测试图片创建完成！")
    print(f"图片保存在: {image_dir}")

if __name__ == "__main__":
    create_test_images()
