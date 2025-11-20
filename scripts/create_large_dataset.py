#!/usr/bin/env python3
"""
创建大规模混合数据集
直接生成合成数据,不依赖AFlow benchmark
"""
import json
import random
from pathlib import Path


def generate_math_samples(num_samples: int) -> list:
    """生成数学问题样本"""
    samples = []

    operations = [
        ("addition", lambda a, b: (f"What is {a} + {b}?", a + b)),
        ("subtraction", lambda a, b: (f"What is {a} - {b}?", a - b)),
        ("multiplication", lambda a, b: (f"What is {a} × {b}?", a * b)),
        ("division", lambda a, b: (f"What is {a} ÷ {b}?", a // b if b != 0 else a)),
    ]

    for i in range(num_samples):
        op_name, op_func = random.choice(operations)
        a = random.randint(1, 100)
        b = random.randint(1, 50)

        problem, answer = op_func(a, b)

        samples.append({
            "problem": problem,
            "problem_type": "math",
            "tag": "gsm8k",
            "ground_truth": str(answer)
        })

    # 添加一些更复杂的数学问题
    complex_problems = [
        {
            "problem": "A train travels 60 miles per hour for 3 hours. How far does it travel?",
            "problem_type": "math",
            "tag": "math",
            "ground_truth": "180"
        },
        {
            "problem": "If a book costs $15 and you have $50, how many books can you buy?",
            "problem_type": "math",
            "tag": "gsm8k",
            "ground_truth": "3"
        },
        {
            "problem": "What is the area of a rectangle with length 12 and width 8?",
            "problem_type": "math",
            "tag": "math",
            "ground_truth": "96"
        },
        {
            "problem": "If 5 apples cost $10, how much does 1 apple cost?",
            "problem_type": "math",
            "tag": "gsm8k",
            "ground_truth": "2"
        }
    ]

    # 重复复杂问题以填充数据集
    while len(samples) < num_samples:
        samples.append(random.choice(complex_problems))

    return samples[:num_samples]


def generate_code_samples(num_samples: int) -> list:
    """生成代码问题样本"""
    samples = []

    code_problems = [
        {
            "problem": "Write a function that returns the sum of two numbers.",
            "problem_type": "code",
            "tag": "humaneval",
            "ground_truth": "def add(a, b):\n    return a + b"
        },
        {
            "problem": "Write a function that checks if a number is even.",
            "problem_type": "code",
            "tag": "mbpp",
            "ground_truth": "def is_even(n):\n    return n % 2 == 0"
        },
        {
            "problem": "Write a function that returns the maximum of two numbers.",
            "problem_type": "code",
            "tag": "humaneval",
            "ground_truth": "def max_of_two(a, b):\n    return a if a > b else b"
        },
        {
            "problem": "Write a function that returns the length of a list.",
            "problem_type": "code",
            "tag": "mbpp",
            "ground_truth": "def list_length(lst):\n    return len(lst)"
        },
        {
            "problem": "Write a function that reverses a string.",
            "problem_type": "code",
            "tag": "humaneval",
            "ground_truth": "def reverse_string(s):\n    return s[::-1]"
        },
        {
            "problem": "Write a function that squares a number.",
            "problem_type": "code",
            "tag": "mbpp",
            "ground_truth": "def square(x):\n    return x * x"
        },
        {
            "problem": "Write a function that checks if a string is empty.",
            "problem_type": "code",
            "tag": "humaneval",
            "ground_truth": "def is_empty(s):\n    return len(s) == 0"
        },
        {
            "problem": "Write a function that returns the first element of a list.",
            "problem_type": "code",
            "tag": "mbpp",
            "ground_truth": "def first_element(lst):\n    return lst[0] if lst else None"
        },
        {
            "problem": "Write a function that counts vowels in a string.",
            "problem_type": "code",
            "tag": "humaneval",
            "ground_truth": "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')"
        },
        {
            "problem": "Write a function that multiplies all numbers in a list.",
            "problem_type": "code",
            "tag": "mbpp",
            "ground_truth": "def multiply_list(lst):\n    result = 1\n    for num in lst:\n        result *= num\n    return result"
        }
    ]

    # 重复问题以填充数据集
    for i in range(num_samples):
        samples.append(random.choice(code_problems))

    return samples


def generate_qa_samples(num_samples: int) -> list:
    """生成QA问题样本"""
    samples = []

    qa_problems = [
        {
            "problem": "What is the capital of France?",
            "problem_type": "qa",
            "tag": "hotpotqa",
            "ground_truth": "Paris"
        },
        {
            "problem": "What is the largest ocean on Earth?",
            "problem_type": "qa",
            "tag": "drop",
            "ground_truth": "Pacific Ocean"
        },
        {
            "problem": "Who wrote 'Romeo and Juliet'?",
            "problem_type": "qa",
            "tag": "hotpotqa",
            "ground_truth": "William Shakespeare"
        },
        {
            "problem": "What is the speed of light?",
            "problem_type": "qa",
            "tag": "drop",
            "ground_truth": "299792458 meters per second"
        },
        {
            "problem": "What is the capital of Japan?",
            "problem_type": "qa",
            "tag": "hotpotqa",
            "ground_truth": "Tokyo"
        },
        {
            "problem": "How many continents are there?",
            "problem_type": "qa",
            "tag": "drop",
            "ground_truth": "7"
        },
        {
            "problem": "What is the chemical symbol for water?",
            "problem_type": "qa",
            "tag": "hotpotqa",
            "ground_truth": "H2O"
        },
        {
            "problem": "Who painted the Mona Lisa?",
            "problem_type": "qa",
            "tag": "drop",
            "ground_truth": "Leonardo da Vinci"
        },
        {
            "problem": "What is the largest planet in our solar system?",
            "problem_type": "qa",
            "tag": "hotpotqa",
            "ground_truth": "Jupiter"
        },
        {
            "problem": "What year did World War II end?",
            "problem_type": "qa",
            "tag": "drop",
            "ground_truth": "1945"
        }
    ]

    # 重复问题以填充数据集
    for i in range(num_samples):
        samples.append(random.choice(qa_problems))

    return samples


def create_dataset(
    train_size: int = 1000,
    val_size: int = 100,
    test_size: int = 100,
    math_ratio: float = 0.4,
    code_ratio: float = 0.3,
    qa_ratio: float = 0.3
):
    """创建混合数据集"""

    print("\n" + "=" * 60)
    print("📦 创建大规模混合数据集")
    print("=" * 60)

    # 创建目录
    Path("data/train").mkdir(parents=True, exist_ok=True)
    Path("data/val").mkdir(parents=True, exist_ok=True)
    Path("data/test").mkdir(parents=True, exist_ok=True)

    # 生成训练集
    print(f"\n📝 生成训练集 ({train_size} 样本)...")
    train_math = generate_math_samples(int(train_size * math_ratio))
    train_code = generate_code_samples(int(train_size * code_ratio))
    train_qa = generate_qa_samples(int(train_size * qa_ratio))

    train_data = train_math + train_code + train_qa
    random.shuffle(train_data)

    with open("data/train/mixed_dataset.jsonl", "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")

    print(f"  ✅ 数学: {len(train_math)} 样本")
    print(f"  ✅ 代码: {len(train_code)} 样本")
    print(f"  ✅ QA: {len(train_qa)} 样本")
    print(f"  ✅ 总计: {len(train_data)} 样本")

    # 生成验证集
    print(f"\n📝 生成验证集 ({val_size} 样本)...")
    val_math = generate_math_samples(int(val_size * math_ratio))
    val_code = generate_code_samples(int(val_size * code_ratio))
    val_qa = generate_qa_samples(int(val_size * qa_ratio))

    val_data = val_math + val_code + val_qa
    random.shuffle(val_data)

    with open("data/val/mixed_dataset.jsonl", "w") as f:
        for item in val_data:
            f.write(json.dumps(item) + "\n")

    print(f"  ✅ 数学: {len(val_math)} 样本")
    print(f"  ✅ 代码: {len(val_code)} 样本")
    print(f"  ✅ QA: {len(val_qa)} 样本")
    print(f"  ✅ 总计: {len(val_data)} 样本")

    # 生成测试集
    print(f"\n📝 生成测试集 ({test_size} 样本)...")
    test_math = generate_math_samples(int(test_size * math_ratio))
    test_code = generate_code_samples(int(test_size * code_ratio))
    test_qa = generate_qa_samples(int(test_size * qa_ratio))

    test_data = test_math + test_code + test_qa
    random.shuffle(test_data)

    with open("data/test/mixed_dataset.jsonl", "w") as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")

    print(f"  ✅ 数学: {len(test_math)} 样本")
    print(f"  ✅ 代码: {len(test_code)} 样本")
    print(f"  ✅ QA: {len(test_qa)} 样本")
    print(f"  ✅ 总计: {len(test_data)} 样本")

    print("\n" + "=" * 60)
    print("✅ 大规模混合数据集创建完成")
    print("=" * 60)

    print(f"\n📊 数据集总览:")
    print(f"  训练集: {len(train_data)} 样本")
    print(f"  验证集: {len(val_data)} 样本")
    print(f"  测试集: {len(test_data)} 样本")
    print(f"  总计: {len(train_data) + len(val_data) + len(test_data)} 样本")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="创建大规模混合数据集")
    parser.add_argument("--train-size", type=int, default=1000, help="训练集大小")
    parser.add_argument("--val-size", type=int, default=100, help="验证集大小")
    parser.add_argument("--test-size", type=int, default=100, help="测试集大小")
    parser.add_argument("--math-ratio", type=float, default=0.4, help="数学比例")
    parser.add_argument("--code-ratio", type=float, default=0.3, help="代码比例")
    parser.add_argument("--qa-ratio", type=float, default=0.3, help="QA比例")

    args = parser.parse_args()

    create_dataset(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        math_ratio=args.math_ratio,
        code_ratio=args.code_ratio,
        qa_ratio=args.qa_ratio
    )
