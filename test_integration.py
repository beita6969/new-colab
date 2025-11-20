#!/usr/bin/env python3
"""
集成测试脚本 - 验证整个系统
"""
import sys
import os
import asyncio

sys.path.insert(0, 'src')

from data_manager import DataManager
from gpu_manager import GPUManager
from reward_computer import RewardComputer


async def test_system():
    """测试系统各个组件"""

    print("\n" + "=" * 70)
    print(" 🧪 AFlow + ROLL 集成系统测试")
    print("=" * 70)

    # 1. GPU管理器测试
    print("\n[1/4] 🖥️  测试GPU管理器...")
    gpu_manager = GPUManager(
        target_gpus=[2, 3],
        protected_pids=[3819483],
        auto_clean=False  # 测试时不自动清理
    )

    if not gpu_manager.check_gpu_available():
        print("  ❌ GPU不可用")
        return False

    print("  ✅ GPU管理器正常")

    # 2. 数据管理器测试
    print("\n[2/4] 📂 测试数据管理器...")
    data_manager = DataManager(
        data_dir="data",
        domain_ratios={"math": 0.4, "code": 0.3, "qa": 0.3}
    )

    data_manager.initialize()

    # 采样测试
    batch = data_manager.sample_batch(batch_size=4, split="train")
    if len(batch) == 0:
        print("  ❌ 数据采样失败")
        return False

    print(f"  ✅ 数据管理器正常（采样了{len(batch)}个样本）")

    # 3. 奖励计算器测试
    print("\n[3/4] 🎯 测试奖励计算器...")
    reward_computer = RewardComputer()

    # 测试案例
    reward = reward_computer.compute_reward(
        problem="What is 2+2?",
        prediction="The answer is 4.",
        ground_truth="4",
        problem_type="math",
        metadata={"cost": 0.001, "execution_time": 2.0}
    )

    if reward > 0:
        print(f"  ✅ 奖励计算器正常（奖励={reward:.2f}）")
    else:
        print(f"  ⚠️  奖励计算结果异常（奖励={reward:.2f}）")

    # 4. 配置文件测试
    print("\n[4/4] ⚙️  测试配置文件...")
    import yaml
    from pathlib import Path

    config_files = [
        "config/training.yaml",
        "config/aflow_llm.yaml"
    ]

    all_configs_ok = True
    for config_file in config_files:
        if not Path(config_file).exists():
            print(f"  ❌ 配置文件不存在: {config_file}")
            all_configs_ok = False
        else:
            try:
                with open(config_file) as f:
                    yaml.safe_load(f)
                print(f"  ✅ {config_file}")
            except Exception as e:
                print(f"  ❌ {config_file} 加载失败: {e}")
                all_configs_ok = False

    if not all_configs_ok:
        return False

    # 总结
    print("\n" + "=" * 70)
    print(" ✅ 所有测试通过！系统准备就绪")
    print("=" * 70)

    print("\n📋 下一步:")
    print("  1. 确保Qwen2.5-7B模型已下载")
    print("  2. 运行: CUDA_VISIBLE_DEVICES=2,3 python3 train.py")
    print("  3. 监控日志: tail -f logs/training.log")
    print("  4. 测试模型: python3 inference.py --checkpoint checkpoints/step_50 --problem '2+2=?'")

    return True


if __name__ == "__main__":
    success = asyncio.run(test_system())
    sys.exit(0 if success else 1)
