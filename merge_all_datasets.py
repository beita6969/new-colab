#!/usr/bin/env python3
"""
合并所有数据集为混合训练集

支持的数据集：
- GSM8K (7,473 train + 1,319 test)
- MATH (多个配置)
- HotpotQA (90,447 train samples)
- DROP (77,400 train samples)
- MBPP (120 train samples)
- HumanEval (from data/humaneval)
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
import random

class DatasetMerger:
    def __init__(self, output_file: str = "merged_aflow_dataset.jsonl"):
        self.output_file = output_file
        self.total_samples = 0
        self.samples_by_type = {}

    def add_samples(self, samples: List[Dict[str, Any]], problem_type: str):
        """Add samples with problem type label"""
        if problem_type not in self.samples_by_type:
            self.samples_by_type[problem_type] = []

        self.samples_by_type[problem_type].extend(samples)
        self.total_samples += len(samples)
        print(f"  Added {len(samples)} {problem_type} samples (total: {self.total_samples})")

    def load_jsonl_file(self, filepath: str, problem_type: str, limit: int = None) -> List[Dict]:
        """Load samples from JSONL file"""
        samples = []
        try:
            with open(filepath, 'r') as f:
                for i, line in enumerate(f):
                    if limit and i >= limit:
                        break
                    try:
                        sample = json.loads(line)
                        # Add problem type if not present
                        sample['problem_type'] = problem_type
                        samples.append(sample)
                    except json.JSONDecodeError:
                        continue
            print(f"✓ Loaded {len(samples)} samples from {filepath}")
            return samples
        except FileNotFoundError:
            print(f"⚠️  File not found: {filepath}")
            return []

    def load_gsm8k(self):
        """Load GSM8K dataset"""
        print("\n【1/6】Loading GSM8K...")
        samples = []

        # Load train
        train_samples = self.load_jsonl_file("data/gsm8k/train.jsonl", "math")
        samples.extend(train_samples)

        # Load test
        test_samples = self.load_jsonl_file("data/gsm8k/test.jsonl", "math")
        samples.extend(test_samples)

        self.add_samples(samples, "math")
        return len(samples)

    def load_math(self):
        """Load MATH dataset"""
        print("\n【2/6】Loading MATH...")
        samples = self.load_jsonl_file("math_dataset.jsonl", "math")
        self.add_samples(samples, "math")
        return len(samples)

    def load_hotpotqa(self):
        """Load HotpotQA dataset"""
        print("\n【3/6】Loading HotpotQA...")
        samples = self.load_jsonl_file("hotpotqa_dataset.jsonl", "qa")
        self.add_samples(samples, "qa")
        return len(samples)

    def load_drop(self):
        """Load DROP dataset"""
        print("\n【4/6】Loading DROP...")
        samples = self.load_jsonl_file("drop_dataset.jsonl", "qa")
        self.add_samples(samples, "qa")
        return len(samples)

    def load_mbpp(self):
        """Load MBPP dataset"""
        print("\n【5/6】Loading MBPP...")
        samples = self.load_jsonl_file("mbpp_dataset.jsonl", "code")
        self.add_samples(samples, "code")
        return len(samples)

    def load_humaneval(self):
        """Load HumanEval dataset"""
        print("\n【6/6】Loading HumanEval...")
        samples = []

        humaneval_dir = "data/humaneval"
        for filename in ["humaneval_train.jsonl", "humaneval_test.jsonl", "humaneval_validation.jsonl"]:
            filepath = os.path.join(humaneval_dir, filename)
            if os.path.exists(filepath):
                file_samples = self.load_jsonl_file(filepath, "code")
                samples.extend(file_samples)

        self.add_samples(samples, "code")
        return len(samples)

    def merge_all(self):
        """Merge all datasets"""
        print("=" * 70)
        print("合并所有AFlow数据集")
        print("=" * 70)

        # Load all datasets
        self.load_gsm8k()
        self.load_math()
        self.load_hotpotqa()
        self.load_drop()
        self.load_mbpp()
        self.load_humaneval()

        # Flatten and write
        print("\n" + "=" * 70)
        print(f"写入混合数据集: {self.output_file}")
        print("=" * 70)

        with open(self.output_file, 'w') as f:
            for problem_type, samples in self.samples_by_type.items():
                for sample in samples:
                    f.write(json.dumps(sample) + '\n')

        # Print statistics
        print("\n📊 最终统计:")
        print(f"  总样本数: {self.total_samples}")
        for problem_type, samples in sorted(self.samples_by_type.items()):
            percentage = (len(samples) / self.total_samples * 100) if self.total_samples > 0 else 0
            print(f"  {problem_type}: {len(samples)} ({percentage:.1f}%)")

        print(f"\n✅ 合并完成! 输出文件: {self.output_file}")
        return self.total_samples


def main():
    merger = DatasetMerger(output_file="merged_aflow_dataset.jsonl")
    total = merger.merge_all()

    if total > 0:
        print(f"\n🎉 成功合并 {total} 个样本!")
    else:
        print("\n⚠️  没有找到任何数据集文件")


if __name__ == "__main__":
    main()
