#!/usr/bin/env python3
"""
训练入口 - 启动GRPO训练
"""
import sys
import os
import asyncio
import argparse

# P12修复: 禁用代理，确保LLM Judge可以直连localhost:8002
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ['no_proxy'] = 'localhost,127.0.0.1'

# 添加src到路径
sys.path.insert(0, 'src')

from grpo_trainer import GRPOTrainer


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="AFlow + ROLL GRPO训练")
    parser.add_argument(
        '--config',
        type=str,
        default='config/training.yaml',
        help='训练配置文件路径'
    )
    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     AFlow + ROLL 深度融合 - GRPO在线学习                    ║
║                                                              ║
║     基于Qwen2.5-7B的工作流优化                               ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # 创建训练器
    trainer = GRPOTrainer(config_path=args.config)

    # 开始训练
    await trainer.train()


if __name__ == "__main__":
    asyncio.run(main())
