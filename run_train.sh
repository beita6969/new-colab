#!/bin/bash
# 快速启动GRPO训练
# 用法: ./run_train.sh

set -e

echo "=== AFlow + ROLL GRPO训练 ==="
echo ""

# 设置环境变量
export LD_LIBRARY_PATH=/usr/lib64-nvidia:/usr/local/cuda-12.5/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.5
export PATH=/usr/local/cuda-12.5/bin:$PATH

# OpenAI API (LLM Judge)
export OPENAI_API_KEY="${OPENAI_API_KEY:-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}"

# HuggingFace缓存
export HF_HOME=/home/claude-user/.cache/huggingface
export TRANSFORMERS_CACHE=/home/claude-user/.cache/huggingface/transformers

echo "环境检查:"
echo "- GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
echo "- CUDA: $(python3 -c 'import torch; print(torch.version.cuda)')"
echo "- OpenAI API: 已配置"
echo ""

# 切换到项目目录
cd /home/claude-user/ntu

# 启动训练
echo "开始训练..."
python3 train.py --config config/training.yaml
