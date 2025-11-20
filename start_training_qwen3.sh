#!/bin/bash
# Qwen3-8B训练启动脚本 (GPU 3专用)

# Set environment variables
export WANDB_MODE=online
export WANDB_API_KEY=b42ca0000cf06f97b05eba34f58823ad5f3122a4
export CUDA_VISIBLE_DEVICES=3  # 只使用GPU 3
export PYTHONPATH=/home/yijia/.claude/11/AFlow:$PYTHONPATH

# Clear previous log
> logs_qwen3/training_output.log

# Start training
nohup python3 train.py --config config/training_qwen3.yaml > logs_qwen3/training_output.log 2>&1 &
NEW_PID=$!

echo ""
echo "🚀 Qwen3-8B训练已启动 (GPU 3)"
echo ""
echo "📊 模型配置:"
echo "  • 模型: Qwen3-8B (16GB)"
echo "  • GPU: 3 (单卡)"
echo "  • LoRA: rank=32, alpha=32"
echo "  • 训练参数: 与Qwen2.5-7B相同"
echo ""
echo "✨ 改进特性:"
echo "  ✅ 文字数字识别 (v2.5)"
echo "  ✅ wandb在线模式 (实时同步)"
echo "  ✅ 5维奖励函数 (ROLL风格)"
echo ""
echo "📁 输出路径:"
echo "  • 日志: logs_qwen3/training_output.log"
echo "  • Checkpoint: checkpoints_qwen3/"
echo "  • Wandb项目: aflow-roll-integration"
echo ""
echo "🔢 训练PID: $NEW_PID"
echo ""
echo "🔍 监控命令:"
echo "  tail -f logs_qwen3/training_output.log"
echo ""
echo "⚖️ 对比训练:"
echo "  • GPU 2: Qwen2.5-7B (PID 2705043)"
echo "  • GPU 3: Qwen3-8B (PID $NEW_PID) ← 新启动"
echo ""
