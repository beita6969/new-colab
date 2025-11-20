#!/bin/bash
# 启动vLLM本地服务器 - 替代OpenAI API加速operator执行
# GPU: A100 #2 (与训练共享，使用50%内存 ~20GB)

set -e

echo "============================================================"
echo "🚀 启动vLLM本地服务器 - Operator执行加速"
echo "============================================================"
echo ""

# 配置
MODEL_PATH="/home/yijia/verl-agent/models/qwen/Qwen2___5-7B-Instruct"
GPU_ID=2
PORT=8000
GPU_MEM_UTIL=0.5  # 使用50%的40GB空闲内存（约20GB）

echo "📋 配置信息:"
echo "  模型: Qwen2.5-7B-Instruct"
echo "  GPU: #${GPU_ID} (A100 80GB)"
echo "  内存占用: ~20GB (gpu_memory_utilization=${GPU_MEM_UTIL})"
echo "  端口: ${PORT}"
echo "  用途: 替代gpt-4o-mini OpenAI API调用"
echo ""

# 检查端口
if lsof -Pi :${PORT} -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠️  警告: 端口 ${PORT} 已被占用"
    echo "正在终止旧进程..."
    lsof -ti:${PORT} | xargs kill -9 2>/dev/null || true
    sleep 2
fi

# 检查GPU
echo "📊 当前GPU状态:"
nvidia-smi --id=${GPU_ID} --query-gpu=name,memory.used,memory.total --format=csv,noheader

echo ""
echo "============================================================"
echo "🚀 启动vLLM服务器..."
echo "============================================================"
echo ""

# 启动vLLM（后台运行）
CUDA_VISIBLE_DEVICES=${GPU_ID} nohup python3 -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --host 127.0.0.1 \
  --port ${PORT} \
  --gpu-memory-utilization ${GPU_MEM_UTIL} \
  --max-model-len 4096 \
  --dtype bfloat16 \
  --trust-remote-code \
  --disable-log-requests \
  > logs/vllm_server_$(date +%Y%m%d_%H%M%S).log 2>&1 &

VLLM_PID=$!
echo "✅ vLLM服务器已启动"
echo "  PID: ${VLLM_PID}"
echo "  日志: logs/vllm_server_*.log"
echo ""

# 等待服务器就绪
echo "⏳ 等待服务器就绪（约30-60秒）..."
sleep 5

# 检查进程
if ps -p ${VLLM_PID} > /dev/null 2>&1; then
    echo "✅ 进程运行正常"
else
    echo "❌ 进程启动失败，请检查日志"
    exit 1
fi

# 等待API就绪
MAX_WAIT=120
ELAPSED=0
while [ $ELAPSED -lt $MAX_WAIT ]; do
    if curl -s http://127.0.0.1:${PORT}/v1/models >/dev/null 2>&1; then
        echo "✅ API服务就绪！"
        break
    fi
    sleep 2
    ELAPSED=$((ELAPSED + 2))
    echo -n "."
done

echo ""
echo ""

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "⚠️  警告: API在${MAX_WAIT}秒内未就绪，请检查日志"
    echo "  tail -f logs/vllm_server_*.log"
    exit 1
fi

echo "============================================================"
echo "✅ vLLM服务器启动成功"
echo "============================================================"
echo ""
echo "📊 服务信息:"
echo "  Base URL: http://127.0.0.1:${PORT}/v1"
echo "  Models endpoint: http://127.0.0.1:${PORT}/v1/models"
echo "  PID: ${VLLM_PID}"
echo ""
echo "📝 下一步:"
echo "  1. 测试API: curl http://127.0.0.1:${PORT}/v1/models"
echo "  2. 修改 config/aflow_llm.yaml:"
echo "     base_url: \"http://127.0.0.1:${PORT}/v1\""
echo "  3. 重启训练以使用本地模型"
echo ""
echo "🛑 停止服务器: kill ${VLLM_PID}"
echo "============================================================"
