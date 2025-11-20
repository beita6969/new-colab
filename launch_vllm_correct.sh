#!/bin/bash

echo "🚀 使用正确的环境启动 vLLM..."

export CUDA_VISIBLE_DEVICES=3
export VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1

# 使用之前工作的环境
/home/yijia/lhy/.venv/bin/python3 -m vllm.entrypoints.openai.api_server \
  --model /home/yijia/lhy/openai/gpt-oss-120b \
  --port 8002 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 4096 \
  --dtype bfloat16 \
  --disable-log-requests \
  > logs/vllm_correct.log 2>&1 &

VLLM_PID=$!
echo "✅ vLLM 已启动 (PID: $VLLM_PID)"
echo "🔄 等待模型加载... (约60秒)"

sleep 70

# 验证
echo "🔍 验证健康状态..."
curl -s http://localhost:8002/v1/models 2>&1 | grep -q "data" && echo "✅ vLLM 已就绪" || echo "⏳ 仍在初始化"

