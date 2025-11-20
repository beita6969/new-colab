#!/bin/bash
# 训练监控脚本 - 每30秒检查一次进度

echo "🔍 GRPO训练监控"
echo "============================================================"
echo ""

while true; do
    clear
    echo "🔍 GRPO训练监控 - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"
    echo ""

    # 检查进程状态
    TRAIN_PID=$(ps aux | grep "python3 train.py" | grep -v grep | awk '{print $2}' | head -1)
    if [ -n "$TRAIN_PID" ]; then
        echo "✅ 训练进程: 运行中 (PID: $TRAIN_PID)"

        # 获取CPU和内存使用
        CPU_MEM=$(ps -p $TRAIN_PID -o %cpu,%mem,etime --no-headers)
        echo "📊 资源使用: CPU/MEM/TIME = $CPU_MEM"
    else
        echo "❌ 训练进程: 已停止"
        break
    fi

    echo ""
    echo "📈 最新进度:"
    echo "------------------------------------------------------------"

    # 显示最新步骤
    tail -100 logs/training_output.log | grep -E "Step [0-9]+/500" | tail -1

    # 显示最新准确率
    echo ""
    tail -100 logs/training_output.log | grep "准确率统计" | tail -3

    echo ""
    echo "📁 日志文件大小: $(wc -l < logs/training_output.log) 行"
    echo ""
    echo "------------------------------------------------------------"
    echo "⏱️  下次更新: 30秒后... (Ctrl+C 停止监控)"

    sleep 30
done
