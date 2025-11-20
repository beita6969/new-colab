#!/usr/bin/env python3
"""
测试配置是否正确加载
"""
import sys
import yaml
from pathlib import Path

sys.path.insert(0, '/home/yijia/.claude/11/AFlow')

def test_config():
    print("=" * 60)
    print("测试配置加载")
    print("=" * 60)

    # 1. 测试aflow_llm.yaml
    print("\n1. 测试 config/aflow_llm.yaml")
    config_path = Path('/home/yijia/.claude/11/integrated_aflow_roll/config/aflow_llm.yaml')
    with open(config_path, 'r') as f:
        aflow_config = yaml.safe_load(f)

    models = aflow_config.get('models', {})
    print(f"   模型配置数量: {len(models)}")

    if 'gpt-oss-120b' in models:
        print("   ✅ 找到 gpt-oss-120b 配置")
        model_cfg = models['gpt-oss-120b']
        print(f"      base_url: {model_cfg.get('base_url')}")
        print(f"      model_name: {model_cfg.get('model_name')}")
    else:
        print("   ❌ 未找到 gpt-oss-120b 配置")
        print(f"   可用的模型: {list(models.keys())}")
        return False

    if 'gpt-4o-mini' in models:
        print("   ⚠️  仍然存在 gpt-4o-mini 配置（应该已移除）")

    # 2. 测试training.yaml
    print("\n2. 测试 config/training.yaml")
    training_config_path = Path('/home/yijia/.claude/11/integrated_aflow_roll/config/training.yaml')
    with open(training_config_path, 'r') as f:
        training_config = yaml.safe_load(f)

    executor_model = training_config.get('aflow_executor_model')
    print(f"   aflow_executor_model: {executor_model}")
    if executor_model == 'gpt-oss-120b':
        print("   ✅ 正确配置为 gpt-oss-120b")
    else:
        print(f"   ❌ 错误: 应为 gpt-oss-120b，实际为 {executor_model}")
        return False

    train_dataset = training_config.get('train_dataset')
    val_dataset = training_config.get('val_dataset')
    print(f"   train_dataset: {train_dataset}")
    print(f"   val_dataset: {val_dataset}")

    # 检查数据集文件
    train_path = Path('/home/yijia/.claude/11/integrated_aflow_roll') / train_dataset
    val_path = Path('/home/yijia/.claude/11/integrated_aflow_roll') / val_dataset

    if train_path.exists():
        print(f"   ✅ 训练集文件存在: {train_path}")
    else:
        print(f"   ❌ 训练集文件不存在: {train_path}")
        return False

    if val_path.exists():
        print(f"   ✅ 验证集文件存在: {val_path}")
    else:
        print(f"   ❌ 验证集文件不存在: {val_path}")
        return False

    # 3. 测试LLM配置加载
    print("\n3. 测试 LLM 配置加载")
    try:
        from scripts.async_llm import LLMsConfig
        llm_configs = LLMsConfig(models)
        llm_instance = llm_configs.get('gpt-oss-120b')
        print(f"   ✅ LLMsConfig 加载成功")
        print(f"   LLM实例类型: {type(llm_instance).__name__}")
    except Exception as e:
        print(f"   ❌ LLM配置加载失败: {e}")
        return False

    # 4. 测试端口连接
    print("\n4. 测试 8002 端口连接")
    import requests
    try:
        response = requests.get('http://localhost:8002/v1/models', timeout=5)
        if response.status_code == 200:
            data = response.json()
            models_list = data.get('data', [])
            if models_list:
                model_id = models_list[0].get('id')
                print(f"   ✅ 端口 8002 服务正常")
                print(f"   模型ID: {model_id}")
            else:
                print("   ⚠️  端口 8002 响应无模型数据")
        else:
            print(f"   ❌ 端口 8002 响应错误: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ 无法连接到端口 8002: {e}")
        return False

    print("\n" + "=" * 60)
    print("✅ 所有配置测试通过")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_config()
    sys.exit(0 if success else 1)
