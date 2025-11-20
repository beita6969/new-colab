#!/usr/bin/env python3
"""
推理脚本 - 测试训练好的模型
"""
import sys
import os
import asyncio
import argparse

sys.path.insert(0, 'src')

from rl_workflow_generator import RLWorkflowGenerator
from aflow_executor import AFlowExecutor
from reward_computer import RewardComputer


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="推理测试")
    parser.add_argument('--checkpoint', type=str, required=True, help='LoRA检查点路径')
    parser.add_argument('--problem', type=str, required=True, help='问题文本')
    parser.add_argument('--problem-type', type=str, default='math', choices=['math', 'code', 'qa'], help='问题类型')
    parser.add_argument('--ground-truth', type=str, default=None, help='正确答案（可选）')
    args = parser.parse_args()

    print("=" * 60)
    print("🔍 推理测试")
    print("=" * 60)

    # 1. 加载RL模型
    print(f"\n📥 加载模型检查点: {args.checkpoint}")
    generator = RLWorkflowGenerator(
        base_model="Qwen/Qwen2.5-7B-Instruct",
        lora_checkpoint=args.checkpoint,
        device_ids=[2, 3]
    )

    # 2. 生成工作流
    print(f"\n📝 问题: {args.problem}")
    print(f"   类型: {args.problem_type}")

    result = generator.generate_workflow(
        problem=args.problem,
        problem_type=args.problem_type,
        temperature=0.7
    )

    print(f"\n✅ 工作流生成:")
    print(f"   有效性: {result['valid']}")
    if result['error']:
        print(f"   错误: {result['error']}")

    print(f"\n📄 生成的工作流代码:")
    print(result['workflow_code'])

    # 3. 执行工作流
    print(f"\n⚙️  执行工作流...")
    executor = AFlowExecutor(
        llm_config_path="config/aflow_llm.yaml",
        llm_model_name="gpt-4o-mini"
    )

    answer, cost, metadata = await executor.execute_workflow(
        workflow_code=result['workflow_code'],
        problem=args.problem,
        problem_type=args.problem_type
    )

    print(f"\n✅ 执行结果:")
    print(f"   成功: {metadata['success']}")
    print(f"   答案: {answer}")
    print(f"   成本: ${cost:.6f}")
    print(f"   时间: {metadata.get('execution_time', 0):.2f}秒")

    # 4. 计算奖励（如果提供了ground truth）
    if args.ground_truth:
        print(f"\n🎯 评估:")
        print(f"   正确答案: {args.ground_truth}")

        reward_computer = RewardComputer()
        reward = reward_computer.compute_reward(
            problem=args.problem,
            prediction=answer,
            ground_truth=args.ground_truth,
            problem_type=args.problem_type,
            metadata=metadata
        )

        print(f"   奖励分数: {reward:.2f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
