#!/usr/bin/env python3
"""
标准化数据集字段名称
将所有 "question" 转换为 "problem"
确保与trainer兼容
"""

import json
from pathlib import Path

def normalize_sample(sample):
    """Normalize field names in a sample"""
    # 将 question 转换为 problem
    if 'question' in sample and 'problem' not in sample:
        sample['problem'] = sample.pop('question')

    # 将 answer 转换为 ground_truth
    if 'answer' in sample and 'ground_truth' not in sample:
        sample['ground_truth'] = sample.pop('answer')

    # 确保有问题类型
    if 'problem_type' not in sample:
        sample['problem_type'] = 'unknown'

    return sample

def normalize_file(input_file: str, output_file: str):
    """Normalize a JSONL file"""
    count = 0
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            try:
                sample = json.loads(line)
                sample = normalize_sample(sample)
                outfile.write(json.dumps(sample) + '\n')
                count += 1
            except json.JSONDecodeError:
                continue

    return count

def main():
    print("=" * 70)
    print("📊 标准化数据集字段")
    print("=" * 70)

    files = [
        ("data/train/mixed_dataset.jsonl", "data/train/mixed_dataset_normalized.jsonl"),
        ("data/val/mixed_dataset.jsonl", "data/val/mixed_dataset_normalized.jsonl"),
        ("data/test/mixed_dataset.jsonl", "data/test/mixed_dataset_normalized.jsonl"),
    ]

    for input_file, output_file in files:
        if Path(input_file).exists():
            count = normalize_file(input_file, output_file)
            print(f"✅ {input_file}")
            print(f"   → {output_file} ({count} 样本)")

            # Replace original with normalized
            Path(input_file).unlink()
            Path(output_file).rename(input_file)
            print(f"   ✓ 已替换原文件")
        else:
            print(f"⚠️  {input_file} 不存在")

    print("\n✨ 标准化完成!")

if __name__ == "__main__":
    main()
