#!/usr/bin/env python3
"""
数据管理器 - 混合数据集加载和采样
"""
import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

class DataManager:
    """混合数据集管理器"""

    def __init__(
        self,
        data_dir: str = "data",
        domain_ratios: Optional[Dict[str, float]] = None,
        shuffle: bool = True
    ):
        """
        Args:
            data_dir: 数据目录
            domain_ratios: 领域采样比例，如 {"math": 0.4, "code": 0.3, "qa": 0.3}
            shuffle: 是否打乱数据
        """
        self.data_dir = Path(data_dir)
        self.domain_ratios = domain_ratios or {"math": 0.4, "code": 0.3, "qa": 0.3}
        self.shuffle = shuffle

        # 加载的数据
        self.train_data = {"math": [], "code": [], "qa": []}
        self.val_data = {"math": [], "code": [], "qa": []}
        self.test_data = {"math": [], "code": [], "qa": []}

        # 当前迭代位置
        self.current_indices = {"math": 0, "code": 0, "qa": 0}

    def load_data(self, split: str = "train") -> Dict[str, List[Dict]]:
        """加载指定分割的数据"""
        data_by_type = defaultdict(list)

        # 1. Load HumanEval code data (NEW - proper format)
        humaneval_file = self._get_humaneval_file(split)
        if humaneval_file and humaneval_file.exists():
            with open(humaneval_file, 'r') as f:
                for line in f:
                    if line.strip():
                        sample = json.loads(line)
                        # Convert HumanEval format to our format
                        converted = self._convert_humaneval_sample(sample)
                        data_by_type["code"].append(converted)
            print(f"✅ 加载 HumanEval code: {len(data_by_type['code'])} 样本")

        # 2. Load mixed dataset (math/code/qa)
        # 优先使用新路径格式: mixed/{split}_mixed.jsonl
        dataset_file = self.data_dir / f"mixed/{split}_mixed.jsonl"
        if not dataset_file.exists():
            # 后备旧路径: {split}/mixed_dataset.jsonl
            dataset_file = self.data_dir / f"{split}/mixed_dataset.jsonl"

        if dataset_file.exists():
            with open(dataset_file, 'r') as f:
                for line in f:
                    if line.strip():
                        sample = json.loads(line)
                        problem_type = sample.get("problem_type", "math")
                        data_by_type[problem_type].append(sample)

        # 打乱
        if self.shuffle:
            for ptype in data_by_type:
                random.shuffle(data_by_type[ptype])

        print(f"✅ 加载 {split.upper()} 数据:")
        for ptype, samples in data_by_type.items():
            print(f"  {ptype}: {len(samples)} 样本")

        return dict(data_by_type)

    def _get_humaneval_file(self, split: str) -> Optional[Path]:
        """获取HumanEval文件路径"""
        if split == "train":
            file_path = self.data_dir / "humaneval/humaneval_full.jsonl"
        elif split == "val":
            file_path = self.data_dir / "humaneval/humaneval_validate.jsonl"
        elif split == "test":
            file_path = self.data_dir / "humaneval/humaneval_test.jsonl"
        else:
            return None

        return file_path if file_path.exists() else None

    def _convert_humaneval_sample(self, sample: Dict) -> Dict:
        """
        转换HumanEval格式到我们的格式

        HumanEval format:
        {
            "task_id": "HumanEval/0",
            "prompt": "def has_close_elements(...):\n    ...",
            "entry_point": "has_close_elements",
            "canonical_solution": "    for idx...",
            "test": "def check(candidate):\n    assert..."
        }

        Our format (extended):
        {
            "problem": str,
            "problem_type": "code",
            "tag": "humaneval",
            "ground_truth": str,  # canonical_solution
            "entry_point": str,  # NEW
            "prompt": str,       # NEW - full function signature
            "test": str          # NEW - test cases
        }
        """
        return {
            "problem": sample["prompt"],  # Full function signature + docstring
            "problem_type": "code",
            "tag": "humaneval",
            "task_id": sample.get("task_id", ""),
            "ground_truth": sample["canonical_solution"],
            "entry_point": sample["entry_point"],
            "prompt": sample["prompt"],
            "test": sample["test"]
        }

    def initialize(self):
        """初始化所有数据"""
        print("=" * 60)
        print("📂 初始化数据管理器")
        print("=" * 60)

        self.train_data = self.load_data("train")
        self.val_data = self.load_data("val")
        self.test_data = self.load_data("test")

        # 验证数据
        total_train = sum(len(samples) for samples in self.train_data.values())
        total_val = sum(len(samples) for samples in self.val_data.values())
        total_test = sum(len(samples) for samples in self.test_data.values())

        print(f"\n📊 数据集统计:")
        print(f"  训练集: {total_train} 样本")
        print(f"  验证集: {total_val} 样本")
        print(f"  测试集: {total_test} 样本")

        print("\n🎯 采样比例:")
        for domain, ratio in self.domain_ratios.items():
            print(f"  {domain}: {ratio*100:.1f}%")

        print("=" * 60)

    def sample_batch(
        self,
        batch_size: int,
        split: str = "train",
        domain_ratios: Optional[Dict[str, float]] = None
    ) -> List[Dict]:
        """
        按比例采样一个batch

        Args:
            batch_size: batch大小
            split: 数据分割 (train/val/test)
            domain_ratios: 自定义比例（None则使用默认）

        Returns:
            混合batch样本列表
        """
        ratios = domain_ratios or self.domain_ratios

        # 选择数据源
        if split == "train":
            data_source = self.train_data
        elif split == "val":
            data_source = self.val_data
        else:
            data_source = self.test_data

        # 计算每个领域的样本数
        domain_counts = {}
        remaining = batch_size

        for domain in ["math", "code", "qa"]:
            if domain in ratios:
                count = int(batch_size * ratios[domain])
                domain_counts[domain] = count
                remaining -= count

        # 将余数分配给第一个领域
        if remaining > 0:
            first_domain = list(domain_counts.keys())[0]
            domain_counts[first_domain] += remaining

        # 采样
        batch = []

        for domain, count in domain_counts.items():
            if domain not in data_source or len(data_source[domain]) == 0:
                print(f"⚠️  {domain} 数据不足，跳过")
                continue

            domain_data = data_source[domain]

            # 在线学习模式：循环采样
            samples = []
            for _ in range(count):
                # 获取当前样本
                idx = self.current_indices[domain] % len(domain_data)
                samples.append(domain_data[idx])

                # 更新索引
                self.current_indices[domain] += 1

                # 如果一轮结束，重新打乱
                if self.current_indices[domain] % len(domain_data) == 0:
                    if self.shuffle:
                        random.shuffle(domain_data)

            batch.extend(samples)

        # 打乱batch
        if self.shuffle:
            random.shuffle(batch)

        return batch

    def get_batch_stats(self, batch: List[Dict]) -> Dict[str, int]:
        """获取batch的统计信息"""
        stats = defaultdict(int)

        for sample in batch:
            ptype = sample.get("problem_type", "unknown")
            stats[ptype] += 1

        return dict(stats)

    def reset_indices(self):
        """重置采样索引"""
        self.current_indices = {"math": 0, "code": 0, "qa": 0}
        print("✅ 采样索引已重置")


def test_data_manager():
    """测试数据管理器"""
    print("\n" + "=" * 60)
    print("🧪 测试数据管理器")
    print("=" * 60)

    # 创建管理器
    manager = DataManager(
        data_dir="data",
        domain_ratios={"math": 0.4, "code": 0.3, "qa": 0.3}
    )

    # 初始化
    manager.initialize()

    # 采样测试
    print("\n🔬 采样测试:")
    for i in range(3):
        batch = manager.sample_batch(batch_size=10, split="train")
        stats = manager.get_batch_stats(batch)

        print(f"\nBatch {i+1}:")
        print(f"  总数: {len(batch)}")
        print(f"  分布: {stats}")

        # 显示前2个样本
        for j, sample in enumerate(batch[:2]):
            print(f"  样本{j+1}: {sample['problem_type']} - {sample['problem'][:50]}...")


if __name__ == "__main__":
    test_data_manager()
