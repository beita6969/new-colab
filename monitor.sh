#!/bin/bash
# 训练实时监控脚本

LOG_FILE="logs/train_direct.log"
PID=1785150

echo "=========================================="
echo "训练监控 - $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# 1. 进程状态
echo -e "\n【进程状态】"
if ps -p $PID > /dev/null 2>&1; then
    ps -p $PID -o pid,pcpu,pmem,etime,args --no-headers | awk '{printf "PID: %s | CPU: %s%% | 内存: %s%% | 运行时间: %s\n", $1, $2, $3, $4}'
    echo "✅ 进程运行正常"
else
    echo "❌ 训练进程已停止！"
    exit 1
fi

# 2. GPU状态
echo -e "\n【GPU状态】"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader | grep '^2,' | awk -F, '{printf "GPU 2 | 显存: %s/%s | 利用率: %s | 温度: %s\n", $2, $3, $4, $5}'

# 3. 最新训练日志 (提取关键信息)
echo -e "\n【最新训练进度】"
tail -200 "$LOG_FILE" | grep -E '(Step [0-9]+/|📦 Batch|准确率|Accuracy|Loss|reward|完成时间|GPU显存)' | tail -15

# 4. 错误检查
echo -e "\n【错误检测】"
ERROR_COUNT=$(tail -100 "$LOG_FILE" | grep -i -E '(error|exception|traceback|failed)' | wc -l)
if [ $ERROR_COUNT -gt 0 ]; then
    echo "⚠️  检测到 $ERROR_COUNT 个错误，最新错误:"
    tail -100 "$LOG_FILE" | grep -i -E '(error|exception|traceback)' | tail -3
else
    echo "✅ 无错误"
fi

# 5. WandB链接
echo -e "\n【监控链接】"
grep -o 'https://wandb.ai/[^[:space:]]*' "$LOG_FILE" | tail -1

echo -e "\n=========================================="
echo "提示: 使用 watch -n 10 bash monitor.sh 每10秒刷新"
echo "=========================================="
