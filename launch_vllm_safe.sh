#!/bin/bash
echo "🔧 启动 vLLM (安全模式)..."

# 清理环境
export VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1
export CUDA_VISIBLE_DEVICES=3
export TOKENIZERS_PARALLELISM=false

# 用标准配置启动
python3 -m vllm.entrypoints.openai.api_server \
  --model /home/yijia/lhy/openai/gpt-oss-120b \
  --port 8002 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.7 \
  --max-model-len 2048 \
  --dtype bfloat16 \
  --disable-log-requests \
  2>&1 | tee logs/vllm_safe.log &

echo "✅ vLLM 已启动 (PID: $!)"
sleep 60
echo "🔄 验证 vLLM 健康状态..."
curl -s http://localhost:8002/v1/models 2>&1 | grep -q "data" && echo "✅ vLLM 健康" || echo "❌ vLLM 异常"
