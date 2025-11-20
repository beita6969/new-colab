#!/usr/bin/env python3
"""
将merged_aflow_dataset.jsonl分割为训练/验证/测试集
分割比例：80% train / 10% val / 10% test
"""

import json
import random
from pathlib import Path

def split_dataset(input_file: str, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42):
    """Split dataset into train/val/test"""
    random.seed(seed)

    # Load all samples
    samples = []
    print(f"正在读取 {input_file}...")
    with open(input_file, 'r') as f:
        for i, line in enumerate(f):
            try:
                sample = json.loads(line)
                samples.append(sample)
            except json.JSONDecodeError:
                continue

    total_samples = len(samples)
    print(f"✓ 读取了 {total_samples} 个样本")

    # Shuffle
    random.shuffle(samples)

    # Split indices
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)

    train_samples = samples[:train_size]
    val_samples = samples[train_size:train_size + val_size]
    test_samples = samples[train_size + val_size:]

    # Create directories
    Path("data/train").mkdir(parents=True, exist_ok=True)
    Path("data/val").mkdir(parents=True, exist_ok=True)
    Path("data/test").mkdir(parents=True, exist_ok=True)

    # Write train set
    with open("data/train/mixed_dataset.jsonl", 'w') as f:
        for sample in train_samples:
            f.write(json.dumps(sample) + '\n')
    print(f"✅ 训练集: {len(train_samples)} 样本 → data/train/mixed_dataset.jsonl")

    # Write val set
    with open("data/val/mixed_dataset.jsonl", 'w') as f:
        for sample in val_samples:
            f.write(json.dumps(sample) + '\n')
    print(f"✅ 验证集: {len(val_samples)} 样本 → data/val/mixed_dataset.jsonl")

    # Write test set
    with open("data/test/mixed_dataset.jsonl", 'w') as f:
        for sample in test_samples:
            f.write(json.dumps(sample) + '\n')
    print(f"✅ 测试集: {len(test_samples)} 样本 → data/test/mixed_dataset.jsonl")

    # Statistics
    print("\n" + "="*70)
    print("📊 数据集分割统计:")
    print("="*70)
    print(f"总样本数: {total_samples}")
    print(f"训练集: {len(train_samples)} ({len(train_samples)/total_samples*100:.1f}%)")
    print(f"验证集: {len(val_samples)} ({len(val_samples)/total_samples*100:.1f}%)")
    print(f"测试集: {len(test_samples)} ({len(test_samples)/total_samples*100:.1f}%)")

    # Count by problem type
    print("\n" + "="*70)
    print("📈 按问题类型统计:")
    print("="*70)

    type_stats = {}
    for samples_list, split_name in [(train_samples, "训练"), (val_samples, "验证"), (test_samples, "测试")]:
        type_count = {}
        for sample in samples_list:
            ptype = sample.get('problem_type', 'unknown')
            type_count[ptype] = type_count.get(ptype, 0) + 1

        print(f"\n{split_name}集:")
        for ptype, count in sorted(type_count.items()):
            percentage = count / len(samples_list) * 100
            print(f"  {ptype}: {count} ({percentage:.1f}%)")


if __name__ == "__main__":
    split_dataset("merged_aflow_dataset.jsonl")
    print("\n✨ 数据分割完成!")
