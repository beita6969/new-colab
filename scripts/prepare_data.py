#!/usr/bin/env python3
"""
数据准备脚本 - 从ROLL和AFlow提取和整合数据集
"""
import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Optional
import random

# 添加路径
sys.path.append('/home/yijia/.claude/11/ROLL')
sys.path.append('/home/yijia/.claude/11/AFlow')

class DataPreparer:
    """数据准备器：整合ROLL和AFlow的数据集"""

    def __init__(self, output_dir: str = "data"):
        self.output_dir = Path(output_dir)
        self.aflow_path = Path('/home/yijia/.claude/11/AFlow')
        self.roll_path = Path('/home/yijia/.claude/11/ROLL')

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "train").mkdir(exist_ok=True)
        (self.output_dir / "val").mkdir(exist_ok=True)
        (self.output_dir / "test").mkdir(exist_ok=True)

    def extract_aflow_benchmark_data(self, benchmark_name: str) -> List[Dict]:
        """从AFlow提取benchmark数据"""
        data = []

        try:
            # 动态导入AFlow的benchmark
            sys.path.insert(0, str(self.aflow_path))

            if benchmark_name == "gsm8k":
                from benchmarks.gsm8k import GSM8KBenchmark
                benchmark = GSM8KBenchmark()
                dataset_type = "math"
            elif benchmark_name == "math":
                from benchmarks.math import MATHBenchmark
                benchmark = MATHBenchmark()
                dataset_type = "math"
            elif benchmark_name == "humaneval":
                from benchmarks.humaneval import HumanEvalBenchmark
                benchmark = HumanEvalBenchmark()
                dataset_type = "code"
            elif benchmark_name == "mbpp":
                from benchmarks.mbpp import MBPPBenchmark
                benchmark = MBPPBenchmark()
                dataset_type = "code"
            elif benchmark_name == "hotpotqa":
                from benchmarks.hotpotqa import HotpotQABenchmark
                benchmark = HotpotQABenchmark()
                dataset_type = "qa"
            elif benchmark_name == "drop":
                from benchmarks.drop import DROPBenchmark
                benchmark = DROPBenchmark()
                dataset_type = "qa"
            else:
                print(f"⚠️  未知的benchmark: {benchmark_name}")
                return []

            # 加载数据
            import asyncio
            raw_data = asyncio.run(benchmark.load_data(split="all"))

            # 转换格式
            for item in raw_data:
                sample = {
                    "problem": self._extract_problem(item, dataset_type),
                    "problem_type": dataset_type,
                    "tag": benchmark_name,
                    "ground_truth": self._extract_ground_truth(item, dataset_type),
                    "metadata": item
                }

                # 代码类型需要额外字段
                if dataset_type == "code":
                    sample["entry_point"] = item.get("entry_point", "")
                    sample["test_code"] = item.get("test", "")

                data.append(sample)

            print(f"✅ 从 {benchmark_name} 提取 {len(data)} 个样本")

        except Exception as e:
            print(f"❌ 提取 {benchmark_name} 失败: {e}")
            import traceback
            traceback.print_exc()

        return data

    def _extract_problem(self, item: Dict, dataset_type: str) -> str:
        """提取问题文本"""
        if "prompt" in item:
            return item["prompt"]
        elif "question" in item:
            return item["question"]
        elif "problem" in item:
            return item["problem"]
        else:
            return str(item.get("input", ""))

    def _extract_ground_truth(self, item: Dict, dataset_type: str) -> str:
        """提取正确答案"""
        if "answer" in item:
            return str(item["answer"])
        elif "code" in item:
            return item["code"]
        elif "canonical_solution" in item:
            return item["canonical_solution"]
        else:
            return str(item.get("output", ""))

    def create_mixed_dataset(
        self,
        math_ratio: float = 0.4,
        code_ratio: float = 0.3,
        qa_ratio: float = 0.3,
        total_train: int = 10000,
        total_val: int = 1000,
        total_test: int = 500
    ):
        """创建混合数据集"""
        print("=" * 60)
        print("📦 创建混合数据集")
        print("=" * 60)

        # 从AFlow提取数据
        print("\n🔍 从AFlow提取数据...")

        math_data = []
        math_data.extend(self.extract_aflow_benchmark_data("gsm8k"))
        math_data.extend(self.extract_aflow_benchmark_data("math"))

        code_data = []
        code_data.extend(self.extract_aflow_benchmark_data("humaneval"))
        code_data.extend(self.extract_aflow_benchmark_data("mbpp"))

        qa_data = []
        qa_data.extend(self.extract_aflow_benchmark_data("hotpotqa"))
        qa_data.extend(self.extract_aflow_benchmark_data("drop"))

        print(f"\n📊 数据统计:")
        print(f"  数学: {len(math_data)} 样本")
        print(f"  代码: {len(code_data)} 样本")
        print(f"  QA: {len(qa_data)} 样本")

        # 洗牌
        random.shuffle(math_data)
        random.shuffle(code_data)
        random.shuffle(qa_data)

        # 按比例分配
        def create_split(total: int, split_name: str):
            """创建数据集分割"""
            math_count = int(total * math_ratio)
            code_count = int(total * code_ratio)
            qa_count = total - math_count - code_count

            split_data = []

            # 采样数学数据
            if len(math_data) >= math_count:
                split_data.extend(random.sample(math_data, math_count))
            else:
                print(f"⚠️  数学数据不足，需要{math_count}，只有{len(math_data)}")
                split_data.extend(math_data)

            # 采样代码数据
            if len(code_data) >= code_count:
                split_data.extend(random.sample(code_data, code_count))
            else:
                print(f"⚠️  代码数据不足，需要{code_count}，只有{len(code_data)}")
                split_data.extend(code_data)

            # 采样QA数据
            if len(qa_data) >= qa_count:
                split_data.extend(random.sample(qa_data, qa_count))
            else:
                print(f"⚠️  QA数据不足，需要{qa_count}，只有{len(qa_data)}")
                split_data.extend(qa_data)

            # 洗牌
            random.shuffle(split_data)

            # 保存
            output_file = self.output_dir / f"{split_name}/mixed_dataset.jsonl"
            with open(output_file, 'w') as f:
                for sample in split_data:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')

            print(f"✅ {split_name.upper()} 数据集: {len(split_data)} 样本 -> {output_file}")

            return split_data

        # 创建训练、验证、测试集
        train_data = create_split(total_train, "train")
        val_data = create_split(total_val, "val")
        test_data = create_split(total_test, "test")

        print("\n" + "=" * 60)
        print("✅ 混合数据集创建完成")
        print("=" * 60)

        return {
            "train": train_data,
            "val": val_data,
            "test": test_data
        }

    def verify_dataset(self, split: str = "train"):
        """验证数据集格式"""
        print(f"\n🔍 验证 {split.upper()} 数据集...")

        dataset_file = self.output_dir / f"{split}/mixed_dataset.jsonl"

        if not dataset_file.exists():
            print(f"❌ 数据集文件不存在: {dataset_file}")
            return False

        samples = []
        with open(dataset_file, 'r') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))

        print(f"✅ 共加载 {len(samples)} 个样本")

        # 验证格式
        required_fields = ["problem", "problem_type", "tag", "ground_truth"]

        for i, sample in enumerate(samples[:5]):  # 检查前5个
            print(f"\n样本 {i+1}:")
            print(f"  问题类型: {sample.get('problem_type')}")
            print(f"  标签: {sample.get('tag')}")
            print(f"  问题: {sample.get('problem', '')[:100]}...")
            print(f"  答案: {sample.get('ground_truth', '')[:100]}...")

            # 检查必需字段
            for field in required_fields:
                if field not in sample:
                    print(f"  ❌ 缺少字段: {field}")

        # 统计
        type_counts = {}
        for sample in samples:
            ptype = sample.get('problem_type', 'unknown')
            type_counts[ptype] = type_counts.get(ptype, 0) + 1

        print(f"\n📊 类型分布:")
        for ptype, count in type_counts.items():
            ratio = count / len(samples) * 100
            print(f"  {ptype}: {count} ({ratio:.1f}%)")

        return True


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="准备混合数据集")
    parser.add_argument('--output-dir', type=str, default='data', help='输出目录')
    parser.add_argument('--train-size', type=int, default=10000, help='训练集大小')
    parser.add_argument('--val-size', type=int, default=1000, help='验证集大小')
    parser.add_argument('--test-size', type=int, default=500, help='测试集大小')
    parser.add_argument('--math-ratio', type=float, default=0.4, help='数学数据比例')
    parser.add_argument('--code-ratio', type=float, default=0.3, help='代码数据比例')
    parser.add_argument('--qa-ratio', type=float, default=0.3, help='QA数据比例')
    parser.add_argument('--verify-only', action='store_true', help='仅验证现有数据集')

    args = parser.parse_args()

    preparer = DataPreparer(output_dir=args.output_dir)

    if args.verify_only:
        preparer.verify_dataset("train")
        preparer.verify_dataset("val")
        preparer.verify_dataset("test")
    else:
        preparer.create_mixed_dataset(
            math_ratio=args.math_ratio,
            code_ratio=args.code_ratio,
            qa_ratio=args.qa_ratio,
            total_train=args.train_size,
            total_val=args.val_size,
            total_test=args.test_size
        )

        # 验证
        preparer.verify_dataset("train")


if __name__ == "__main__":
    main()
