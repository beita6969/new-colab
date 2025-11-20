#!/usr/bin/env python3
"""
改进的GRPO训练脚本 - 应用所有修复
"""
import sys
import os
sys.path.append('/home/yijia/.claude/11/integrated_aflow_roll')

import asyncio
import torch
import wandb
from datetime import datetime
import json
from pathlib import Path

# 导入改进的组件
from src.grpo_trainer import GRPOTrainer
from src.workflow_validator import WorkflowValidator
from src.aflow_executor import AFlowExecutor
from src.code_executor import CodeExecutor
from src.data_manager import DataManager
from src.reward_computer import RewardComputer
from src.rl_workflow_generator import RLWorkflowGenerator
from src.prompt_optimizer import PromptOptimizer
from src.operator_prompt_enhancer import OperatorPromptEnhancer


def setup_gpu_environment():
    """设置GPU环境"""
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
    print(f"✅ 设置 CUDA_VISIBLE_DEVICES=2,3")
    print(f"  可用GPU数: {torch.cuda.device_count()}")


def load_config():
    """加载训练配置"""
    config_path = Path('/home/yijia/.claude/11/integrated_aflow_roll/config/training.yaml')

    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return None

    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # 配置已经在training.yaml中设置好，无需额外修改
    # 仅添加improvements标志
    config['improvements'] = {
        'enable_validator': True,
        'enable_fallback': True,
        'enable_code_executor': True,
        'enable_double_layer_prompts': True,
        'enable_experience_buffer': True,
        'enable_prompt_optimizer': True,
        'enable_operator_enhancer': True,
        'use_10_point_reward': True  # 使用10分制奖励
    }

    print(f"✅ 加载配置完成")
    print(f"  训练步数: {config['max_steps']}")
    print(f"  GRPO组大小: {config['num_return_sequences_in_group']}")
    print(f"  温度: {config.get('generation_config', {}).get('temperature', 0.3)}")

    return config


async def main():
    """
    主训练流程
    """
    print("\n" + "="*60)
    print("🚀 启动改进的GRPO训练 - Phase 1 修复已应用")
    print("="*60)

    # 1. 设置GPU环境
    setup_gpu_environment()

    # 2. 加载配置
    config = load_config()
    if not config:
        return

    # 3. 初始化W&B
    run_name = f"grpo_improved_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        project=config.get('wandb', {}).get('project', 'aflow-grpo'),
        name=run_name,
        config=config
    )
    print(f"✅ W&B初始化: {run_name}")

    # 4. 初始化数据管理器
    print("\n📂 初始化数据管理器...")
    data_manager = DataManager(
        data_dir=config['data_dir'],
        domain_ratios=config['domain_ratios']
    )
    # 加载数据
    data_manager.train_data = data_manager.load_data('train')
    data_manager.val_data = data_manager.load_data('val')
    print("✅ 数据管理器初始化完成")

    # 5. 初始化改进的组件
    print("\n🔧 初始化改进的组件...")

    # 工作流验证器
    validator = WorkflowValidator()
    print("  ✅ 工作流验证器")

    # Code执行器（修复Sympy）
    code_executor = CodeExecutor(timeout=10)
    print("  ✅ Code执行器（Sympy修复）")

    # 提示词优化器
    prompt_optimizer = PromptOptimizer()
    print("  ✅ 提示词优化器")

    # 算子增强器
    operator_enhancer = OperatorPromptEnhancer(
        llm_model='gpt-oss-120b',
        max_enhancement_length=100
    )
    print("  ✅ 算子提示词增强器")

    # AFlow执行器（带Fallback）
    aflow_executor = AFlowExecutor(
        llm_config_path='/home/yijia/.claude/11/integrated_aflow_roll/config/aflow_llm.yaml',
        enable_fallback=True,
        operator_enhancer=operator_enhancer
    )
    print("  ✅ AFlow执行器（Fallback已启用）")

    # 奖励计算器（10分制）
    reward_computer = RewardComputer(
        use_10_point_scale=True
    )
    print("  ✅ 奖励计算器（10分制）")

    # 6. 初始化GRPO训练器
    print("\n🤖 初始化GRPO训练器...")
    trainer = GRPOTrainer(
        config=config,
        data_manager=data_manager,
        executor=aflow_executor,
        reward_computer=reward_computer,
        prompt_optimizer=prompt_optimizer,
        operator_enhancer=operator_enhancer
    )

    print("\n" + "="*60)
    print("🎓 开始训练")
    print("="*60)

    # 7. 运行训练
    try:
        await trainer.train()
        print("\n✅ 训练完成！")

    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断")

    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 8. 清理
        print("\n🧹 清理资源...")
        wandb.finish()
        torch.cuda.empty_cache()
        print("✅ 清理完成")


if __name__ == "__main__":
    # 设置事件循环
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n❌ 程序异常退出: {e}")
        import traceback
        traceback.print_exc()
