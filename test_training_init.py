#!/usr/bin/env python3
"""
快速测试训练流程 - 仅运行1步验证配置
"""
import sys
import os
import yaml
from pathlib import Path

# 设置环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'
os.environ['no_proxy'] = 'localhost,127.0.0.1'

sys.path.insert(0, '/home/yijia/.claude/11/integrated_aflow_roll/src')
sys.path.insert(0, '/home/yijia/.claude/11/AFlow')

def test_training_step():
    print("=" * 60)
    print("快速训练测试 - 1步验证")
    print("=" * 60)

    # 1. 导入必要模块
    print("\n1. 导入模块...")
    try:
        from data_manager import DataManager
        from aflow_executor import AFlowExecutor
        from reward_computer import RewardComputer
        print("   ✅ 模块导入成功")
    except Exception as e:
        print(f"   ❌ 模块导入失败: {e}")
        return False

    # 2. 加载配置
    print("\n2. 加载配置...")
    try:
        config_path = '/home/yijia/.claude/11/integrated_aflow_roll/config/training.yaml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"   ✅ 配置加载成功")
        print(f"   executor_model: {config.get('aflow_executor_model')}")
        print(f"   train_dataset: {config.get('train_dataset')}")
    except Exception as e:
        print(f"   ❌ 配置加载失败: {e}")
        return False

    # 3. 初始化数据管理器
    print("\n3. 初始化数据管理器...")
    try:
        data_manager = DataManager(
            data_dir='data',
            domain_ratios=config['domain_ratios']
        )
        # 加载训练数据
        data_manager.train_data = data_manager.load_data('train')
        data_manager.val_data = data_manager.load_data('val')

        print(f"   ✅ 数据管理器初始化成功")
        total_train = sum(len(samples) for samples in data_manager.train_data.values())
        total_val = sum(len(samples) for samples in data_manager.val_data.values())
        print(f"   训练集大小: {total_train}")
        print(f"   验证集大小: {total_val}")
        for ptype, samples in data_manager.train_data.items():
            print(f"     - {ptype}: {len(samples)}")
    except Exception as e:
        print(f"   ❌ 数据管理器初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 4. 初始化AFlow执行器
    print("\n4. 初始化AFlow执行器...")
    try:
        aflow_executor = AFlowExecutor(
            llm_config_path=config['aflow_config_path'],
            llm_model_name=config['aflow_executor_model'],
            timeout=60,
            enable_fallback=True
        )
        print(f"   ✅ AFlow执行器初始化成功")
    except Exception as e:
        print(f"   ❌ AFlow执行器初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 5. 初始化奖励计算器
    print("\n5. 初始化奖励计算器...")
    try:
        llm_config = {
            "base_url": "http://localhost:8002/v1",
            "api_key": "sk-dummy",
            "model_name": "/home/yijia/lhy/openai/gpt-oss-120b"
        }
        reward_computer = RewardComputer(
            reward_weights=config['reward_weights'],
            use_answer_extractor=True,
            use_llm_judge=True,
            llm_config=llm_config
        )
        print(f"   ✅ 奖励计算器初始化成功")
    except Exception as e:
        print(f"   ❌ 奖励计算器初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 6. 测试采样一个问题
    print("\n6. 测试数据采样...")
    try:
        batch = data_manager.sample_batch(batch_size=1)
        print(f"   ✅ 采样成功")
        sample = batch[0]
        print(f"   问题类型: {sample['problem_type']}")
        print(f"   问题: {sample['problem'][:100]}...")
    except Exception as e:
        print(f"   ❌ 数据采样失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("✅ 训练组件初始化测试通过")
    print("系统已准备好进行完整训练")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_training_step()
    sys.exit(0 if success else 1)
