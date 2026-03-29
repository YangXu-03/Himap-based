#!/bin/bash

# FastV Advanced MME 快速开始脚本

echo "=========================================="
echo "FastV Advanced MME 评估 - 快速开始"
echo "=========================================="
echo ""

# 1. 首先测试配置
echo "步骤 1: 测试配置..."
python test_mme_fastv_config.py

if [ $? -ne 0 ]; then
    echo ""
    echo "配置测试失败，请检查上述错误信息。"
    exit 1
fi

echo ""
echo "配置测试通过！"
echo ""

# 2. 询问用户是否继续
read -p "是否开始运行完整评估？(这将需要 3-6 小时) [y/N]: " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消。"
    echo ""
    echo "你可以稍后运行以下命令开始评估:"
    echo "  bash src/HiMAP/inference/eval_mme_fastv_advanced.sh"
    exit 0
fi

echo ""
echo "步骤 2: 开始运行完整评估..."
echo "注意: 这将测试 6 种方法，预计需要 3-6 小时"
echo ""

# 3. 运行评估
cd /root/nfs/code/HiMAP
bash src/HiMAP/inference/eval_mme_fastv_advanced.sh

if [ $? -ne 0 ]; then
    echo ""
    echo "评估过程中出现错误。"
    exit 1
fi

echo ""
echo "步骤 3: 生成可视化..."
python src/HiMAP/inference/visualize_mme_fastv_results.py

if [ $? -ne 0 ]; then
    echo ""
    echo "可视化生成失败，但评估结果已保存。"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ 完成！"
echo "=========================================="
echo ""
echo "结果文件:"
echo "  - JSON 结果: mme_results_*.json"
echo "  - 可视化图表: mme_fastv_visualizations/"
echo ""
