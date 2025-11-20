#!/usr/bin/env python3
"""
创建示例数据集 - 用于快速测试系统
"""
import json
from pathlib import Path

def create_sample_data():
    """创建示例数据集"""

    # 数学问题示例
    math_samples = [
        {
            "problem": "What is 15 + 27?",
            "problem_type": "math",
            "tag": "gsm8k",
            "ground_truth": "42"
        },
        {
            "problem": "Solve for x: 2x + 5 = 15",
            "problem_type": "math",
            "tag": "math",
            "ground_truth": "5"
        },
        {
            "problem": "Calculate the area of a rectangle with length 8 and width 5.",
            "problem_type": "math",
            "tag": "gsm8k",
            "ground_truth": "40"
        },
        {
            "problem": "If a = 3 and b = 4, what is a^2 + b^2?",
            "problem_type": "math",
            "tag": "math",
            "ground_truth": "25"
        },
    ]

    # 代码问题示例
    code_samples = [
        {
            "problem": "Write a function that returns the sum of two numbers.",
            "problem_type": "code",
            "tag": "humaneval",
            "entry_point": "add_numbers",
            "test_code": "assert add_numbers(2, 3) == 5\nassert add_numbers(0, 0) == 0",
            "ground_truth": "def add_numbers(a, b):\n    return a + b"
        },
        {
            "problem": "Write a function that checks if a number is even.",
            "problem_type": "code",
            "tag": "mbpp",
            "entry_point": "is_even",
            "test_code": "assert is_even(4) == True\nassert is_even(5) == False",
            "ground_truth": "def is_even(n):\n    return n % 2 == 0"
        },
        {
            "problem": "Write a function that returns the maximum of two numbers.",
            "problem_type": "code",
            "tag": "humaneval",
            "entry_point": "max_num",
            "test_code": "assert max_num(5, 3) == 5\nassert max_num(1, 10) == 10",
            "ground_truth": "def max_num(a, b):\n    return a if a > b else b"
        },
    ]

    # QA问题示例
    qa_samples = [
        {
            "problem": "What is the capital of France?",
            "problem_type": "qa",
            "tag": "hotpotqa",
            "ground_truth": "Paris"
        },
        {
            "problem": "Who wrote 'Romeo and Juliet'?",
            "problem_type": "qa",
            "tag": "drop",
            "ground_truth": "William Shakespeare"
        },
        {
            "problem": "What is the largest ocean on Earth?",
            "problem_type": "qa",
            "tag": "hotpotqa",
            "ground_truth": "Pacific Ocean"
        },
    ]

    # 组合所有样本
    all_samples = math_samples * 10 + code_samples * 10 + qa_samples * 10  # 复制以增加数量

    # 创建目录
    data_dir = Path("data")
    (data_dir / "train").mkdir(parents=True, exist_ok=True)
    (data_dir / "val").mkdir(parents=True, exist_ok=True)
    (data_dir / "test").mkdir(parents=True, exist_ok=True)

    # 分割数据
    import random
    random.shuffle(all_samples)

    train_size = int(len(all_samples) * 0.8)
    val_size = int(len(all_samples) * 0.1)

    train_data = all_samples[:train_size]
    val_data = all_samples[train_size:train_size+val_size]
    test_data = all_samples[train_size+val_size:]

    # 保存
    for split, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        output_file = data_dir / f"{split}/mixed_dataset.jsonl"
        with open(output_file, 'w') as f:
            for sample in data:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        print(f"✅ {split.upper()}: {len(data)} 样本 -> {output_file}")

    print(f"\n📊 总计: {len(all_samples)} 样本")
    print(f"  训练集: {len(train_data)}")
    print(f"  验证集: {len(val_data)}")
    print(f"  测试集: {len(test_data)}")

if __name__ == "__main__":
    create_sample_data()
