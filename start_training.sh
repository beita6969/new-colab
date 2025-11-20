#!/bin/bash
# 启动GRPO训练 - 双层动态提示词优化版本
# 日期: 2025-11-18
# 特性: 7个operator + Few-shot学习 + 验证集评估

set -e  # 遇到错误立即退出

echo "============================================================"
echo "🚀 启动GRPO训练 - 双层动态提示词优化"
echo "============================================================"
echo ""

# 1. 环境检查
echo "📋 Step 1: 环境检查"
echo "------------------------------------------------------------"

# 检查Python
if ! command -v python &> /dev/null; then
    echo "❌ 错误: 未找到Python"
    exit 1
fi
echo "✅ Python版本: $(python --version)"

# 检查CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  警告: 未找到nvidia-smi，无法检查GPU"
else
    echo "✅ CUDA可用"
    echo "GPU状态:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader | head -n 4
fi

echo ""

# 2. 文件检查
echo "📋 Step 2: 关键文件检查"
echo "------------------------------------------------------------"

REQUIRED_FILES=(
    "src/train.py"
    "src/grpo_trainer.py"
    "src/experience_buffer.py"
    "src/prompt_optimizer.py"
    "src/operator_prompt_enhancer.py"
    "config/training.yaml"
    "config/aflow_llm.yaml"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ 缺失: $file"
        exit 1
    fi
done

echo ""

# 3. 数据集检查
echo "📋 Step 3: 数据集检查"
echo "------------------------------------------------------------"

if [ -f "data/train/mixed_dataset.jsonl" ]; then
    TRAIN_COUNT=$(wc -l < data/train/mixed_dataset.jsonl)
    echo "✅ 训练集: $TRAIN_COUNT 样本"
else
    echo "❌ 缺失训练集: data/train/mixed_dataset.jsonl"
    exit 1
fi

if [ -f "data/val/mixed_dataset.jsonl" ]; then
    VAL_COUNT=$(wc -l < data/val/mixed_dataset.jsonl)
    echo "✅ 验证集: $VAL_COUNT 样本"
else
    echo "⚠️  警告: 缺失验证集，验证功能将无法使用"
fi

echo ""

# 4. GPU环境设置
echo "📋 Step 4: GPU环境设置"
echo "------------------------------------------------------------"
export CUDA_VISIBLE_DEVICES=2
echo "✅ 设置 CUDA_VISIBLE_DEVICES=2 (使用物理GPU 2)"
echo ""

# 5. 创建必要目录
echo "📋 Step 5: 创建输出目录"
echo "------------------------------------------------------------"
mkdir -p checkpoints
mkdir -p logs
mkdir -p data/experience_buffer
echo "✅ 目录已创建"
echo ""

# 6. 配置总结
echo "📋 Step 6: 训练配置总结"
echo "------------------------------------------------------------"
echo "实验名称: aflow_grpo_dynamic_prompts_v1"
echo "GPU: 物理GPU 2 (单GPU模式)"
echo "Batch Size: 4"
echo "Max Steps: 500"
echo "验证频率: 每10步"
echo "验证样本: 50个"
echo ""
echo "🆕 新特性:"
echo "  ✅ 完整7个operator (vs 原来3个)"
echo "  ✅ 动态Few-shot学习 (top-3相似样本)"
echo "  ✅ 问题类型自适应 (math/code/qa)"
echo "  ✅ ExperienceBuffer (reward>=8.0)"
echo "  ✅ 验证集评估 (每10步)"
echo ""
echo "📊 WandB监控:"
echo "  Project: agent-prompt"
echo "  Entity: yao110002-sdfsdfsdfsdf-com"
echo "  实时URL: 启动后查看日志"
echo ""

# 7. 确认启动
echo "============================================================"
echo "准备启动训练..."
echo "============================================================"
echo ""
read -p "按Enter继续，或Ctrl+C取消: "
echo ""

# 8. 启动训练
echo "🚀 启动训练..."
echo "============================================================"
echo ""

# 创建日志文件名（带时间戳）
LOG_FILE="logs/training_$(date +%Y%m%d_%H%M%S).log"

# 启动训练（后台运行，输出到日志文件和终端）
python src/train.py --config config/training.yaml 2>&1 | tee "$LOG_FILE"

# 注意: 上面的命令会阻塞，直到训练完成或被中断

echo ""
echo "============================================================"
echo "训练日志已保存到: $LOG_FILE"
echo "============================================================"
