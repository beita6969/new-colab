#!/usr/bin/env python3
"""MBPP数据集快速使用示例"""

import json
import random

# 数据文件路径
DATA_DIR = "/home/yijia/.claude/11/integrated_aflow_roll/data/raw"

def load_mbpp_data():
    """加载MBPP数据集"""
    with open(f"{DATA_DIR}/mbpp_train.json", 'r') as f:
        train = json.load(f)
    with open(f"{DATA_DIR}/mbpp_test.json", 'r') as f:
        test = json.load(f)
    with open(f"{DATA_DIR}/mbpp_validation.json", 'r') as f:
        val = json.load(f)

    return train, test, val

def load_mbpp_plus_data():
    """加载MBPP+数据集"""
    with open(f"{DATA_DIR}/mbpp_plus_test.json", 'r') as f:
        test = json.load(f)

    return test

def show_sample(sample, dataset_name="MBPP"):
    """显示一个样本"""
    print(f"\n{'='*60}")
    print(f"{dataset_name} 样本示例 (Task ID: {sample['task_id']})")
    print(f"{'='*60}")

    # MBPP使用'text'字段，MBPP+使用'prompt'字段
    description = sample.get('text') or sample.get('prompt', 'N/A')
    print(f"\n📝 问题描述:\n{description}")
    print(f"\n✅ 测试用例数量: {len(sample['test_list'])}")
    print(f"\n🧪 测试用例示例:")
    for i, test in enumerate(sample['test_list'][:3], 1):
        print(f"  {i}. {test}")
    if len(sample['test_list']) > 3:
        print(f"  ... 还有 {len(sample['test_list']) - 3} 个测试用例")
    print(f"\n💻 参考代码:")
    print("```python")
    code = sample['code'][:300] + "..." if len(sample['code']) > 300 else sample['code']
    print(code)
    print("```")
    print(f"\n{'='*60}\n")

def main():
    print("\n🚀 MBPP数据集快速使用示例\n")

    # 加载MBPP原版
    print("[1/2] 加载MBPP原版数据集...")
    train, test, val = load_mbpp_data()
    print(f"  ✅ 训练集: {len(train)} 题")
    print(f"  ✅ 测试集: {len(test)} 题")
    print(f"  ✅ 验证集: {len(val)} 题")
    print(f"  总计: {len(train) + len(test) + len(val)} 题")

    # 加载MBPP+
    print("\n[2/2] 加载MBPP+增强版数据集...")
    mbpp_plus_test = load_mbpp_plus_data()
    print(f"  ✅ 测试集: {len(mbpp_plus_test)} 题")
    print(f"  ✅ 平均测试用例数: {sum(len(s['test_list']) for s in mbpp_plus_test) / len(mbpp_plus_test):.1f} 个/题")

    # 显示MBPP样本
    print("\n" + "="*60)
    print("MBPP原版样本示例")
    print("="*60)
    mbpp_sample = random.choice(train)
    show_sample(mbpp_sample, "MBPP")

    # 显示MBPP+样本
    print("\n" + "="*60)
    print("MBPP+样本示例（注意测试用例数量）")
    print("="*60)
    mbpp_plus_sample = random.choice(mbpp_plus_test)
    show_sample(mbpp_plus_sample, "MBPP+")

    # 统计信息
    print("\n" + "="*60)
    print("📊 数据集统计对比")
    print("="*60)
    print(f"\n{'指标':<20} {'MBPP原版':<15} {'MBPP+'}")
    print("-" * 60)
    print(f"{'样本总数':<20} {len(train) + len(test) + len(val):<15} {len(mbpp_plus_test)}")
    print(f"{'训练集':<20} {len(train):<15} N/A")
    print(f"{'测试集':<20} {len(test):<15} {len(mbpp_plus_test)}")
    print(f"{'验证集':<20} {len(val):<15} N/A")

    avg_tests_mbpp = sum(len(s['test_list']) for s in test) / len(test)
    avg_tests_plus = sum(len(s['test_list']) for s in mbpp_plus_test) / len(mbpp_plus_test)
    print(f"{'平均测试用例':<20} {avg_tests_mbpp:<15.1f} {avg_tests_plus:.1f}")
    print(f"{'测试用例增强倍数':<20} {'1x':<15} {avg_tests_plus/avg_tests_mbpp:.1f}x")

    print("\n" + "="*60)
    print("✅ 数据集加载和验证完成！")
    print("="*60)
    print("\n💡 使用建议:")
    print("  - 训练阶段: 使用MBPP原版 (974题) 进行大规模训练")
    print("  - 评估阶段: 使用MBPP+ (378题) 进行严格评估")
    print("  - GRPO训练: 两个数据集都非常适合，测试用例可作为奖励信号")
    print("\n")

if __name__ == "__main__":
    main()
