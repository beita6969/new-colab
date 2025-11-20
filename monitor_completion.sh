#!/bin/bash
# 监控模型文件下载完成

for i in {1..20}; do
    echo "=== 检查 $i ($(date +%H:%M:%S)) ==="

    # 统计已完成的文件
    completed=$(ls ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/*/model*.safetensors 2>/dev/null | wc -l)

    # 统计未完成的文件
    incomplete=$(ls ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/blobs/*.incomplete 2>/dev/null | wc -l)

    echo "已完成: $completed/4 | 下载中: $incomplete"
    echo ""

    # 如果所有文件都完成了
    if [ "$incomplete" -eq "0" ]; then
        echo "======================================="
        echo "🎉 所有模型文件下载完成！"
        echo "======================================="
        break
    fi

    sleep 60
done
