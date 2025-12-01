#!/bin/bash
# 环境配置脚本 - A100/Colab
# 设置CUDA环境变量

export LD_LIBRARY_PATH=/usr/lib64-nvidia:/usr/local/cuda-12.5/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.5
export PATH=/usr/local/cuda-12.5/bin:$PATH

# OpenAI API配置 (用于LLM Judge)
export OPENAI_API_KEY="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"

# 禁用代理
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
export no_proxy="localhost,127.0.0.1"

# HuggingFace配置
export HF_HOME=/home/claude-user/.cache/huggingface
export TRANSFORMERS_CACHE=/home/claude-user/.cache/huggingface/transformers

# 打印配置信息
echo "=== 环境配置完成 ==="
echo "CUDA_HOME: $CUDA_HOME"
echo "GPU可用: $(python3 -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'PyTorch未安装')"
echo "OpenAI API: 已配置"
echo "====================="
