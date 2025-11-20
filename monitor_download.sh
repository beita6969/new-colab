#!/bin/bash
# 监控模型下载进度

for i in {1..15}; do
    echo "=== 检查点 $i ($(date +%H:%M:%S)) ==="

    # 计算总大小
    total_size=$(ls -l ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/blobs/*.incomplete 2>/dev/null | awk '{sum+=$5} END {printf "%.2f", sum/1024/1024/1024}')

    # 计算剩余文件数
    incomplete_count=$(ls ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/blobs/*.incomplete 2>/dev/null | wc -l)

    # 显示进度
    echo "已下载: ${total_size} GB / 15.2 GB"
    echo "剩余文件: ${incomplete_count}"
    echo ""

    # 检查是否完成
    if [ "$incomplete_count" -eq "0" ]; then
        echo "======================================="
        echo "✅ 模型下载完成！"
        echo "======================================="
        break
    fi

    # 等待1分钟
    sleep 60
done
