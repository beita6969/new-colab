#!/usr/bin/env python3
"""
分组训练入口 - 启动分组GRPO训练
"""
import sys
import os
import asyncio
import argparse

# 禁用代理
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ['no_proxy'] = 'localhost,127.0.0.1'

# 添加src到路径
sys.path.insert(0, 'src')

from grouped_grpo_trainer import GroupedGRPOTrainer


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="分组GRPO训练")
    parser.add_argument(
        '--config',
        type=str,
        default='config/grouped_training.yaml',
        help='训练配置文件路径'
    )
    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     分组GRPO训练 - 多问题加权评分 + 多样性打破平局          ║
║                                                              ║
║     每 step: 3 组 (math + qa + code) × 4 问题 = 12 问题     ║
║     权重: Easy=30% Hard=70%                                  ║
║     多样性tiebreak: 分数差 < 0.05 时启用                    ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # 创建训练器
    trainer = GroupedGRPOTrainer(config_path=args.config)

    # 开始训练
    await trainer.train()


if __name__ == "__main__":
    asyncio.run(main())
