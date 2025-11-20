#!/bin/bash
# 持续监督脚本 - 每5分钟分析一次训练状态

LOG_FILE="logs/monitor_history.log"
INTERVAL=300  # 5分钟

echo "🔍 启动持续训练监督系统"
echo "监控间隔: ${INTERVAL}秒 (5分钟)"
echo "历史日志: ${LOG_FILE}"
echo "按 Ctrl+C 停止监督"
echo ""

# 创建历史日志
mkdir -p logs
> "$LOG_FILE"

iteration=1

while true; do
    clear
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║       持续训练监督 #${iteration} - $(date '+%Y-%m-%d %H:%M:%S')       ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""

    # 运行分析
    python3 analyze_training.py | tee -a "$LOG_FILE"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    # 检查进程状态
    echo "📊 系统状态:"
    if ps aux | grep -q "[2]255398.*python3.*train.py"; then
        echo "  ✅ 训练进程运行中 (PID: 2255398)"

        # GPU状态
        gpu_info=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | grep -E "^2,|^3," | awk -F, '{printf "  GPU %s: 内存=%s, 利用率=%s\n", $1, $2, $3}')
        echo "$gpu_info"

        # 最近日志
        echo ""
        echo "📝 最近活动 (最后3条):"
        tail -200 logs/training_output.log | grep -E "Step \d+/\d+|🔄 更新策略|⚠️|❌" | tail -3 | sed 's/^/  /'
    else
        echo "  ❌ 训练进程未运行！"
        echo "  建议检查: ps aux | grep train.py"
        break
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "下次分析: $(date -d "+${INTERVAL} seconds" '+%H:%M:%S')"
    echo "历史记录: ${LOG_FILE}"
    echo ""

    # 保存时间戳
    echo "=== 分析 #${iteration} @ $(date '+%Y-%m-%d %H:%M:%S') ===" >> "$LOG_FILE"

    iteration=$((iteration + 1))

    # 等待下次分析
    sleep "$INTERVAL"
done
