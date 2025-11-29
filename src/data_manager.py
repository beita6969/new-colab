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
        shuffle: bool = True,
        train_dataset: Optional[str] = None,  # P6: 支持配置文件指定训练集路径
        val_dataset: Optional[str] = None,
        test_dataset: Optional[str] = None
    ):
        """
        Args:
            data_dir: 数据目录
            domain_ratios: 领域采样比例，如 {"math": 0.4, "code": 0.3, "qa": 0.3}
            shuffle: 是否打乱数据
            train_dataset: 训练集路径 (可选，优先于默认路径)
            val_dataset: 验证集路径 (可选)
            test_dataset: 测试集路径 (可选)
        """
        self.data_dir = Path(data_dir)
        self.domain_ratios = domain_ratios or {"math": 0.4, "code": 0.3, "qa": 0.3}
        self.shuffle = shuffle

        # P6: 存储配置的数据集路径
        self.train_dataset_path = train_dataset
        self.val_dataset_path = val_dataset
        self.test_dataset_path = test_dataset

        # 加载的数据
        self.train_data = {"math": [], "code": [], "qa": []}
        self.val_data = {"math": [], "code": [], "qa": []}
        self.test_data = {"math": [], "code": [], "qa": []}

        # 当前迭代位置
        self.current_indices = {"math": 0, "code": 0, "qa": 0}

    def load_data(self, split: str = "train") -> Dict[str, List[Dict]]:
        """加载指定分割的数据"""
        data_by_type = defaultdict(list)
        source_stats = defaultdict(int)  # 统计source字段

        # 1. Load HumanEval code data (NEW - proper format)
        # 🔴 测试集不单独加载HumanEval，因为ready_to_train/test.jsonl已包含
        if split != "test":
            humaneval_file = self._get_humaneval_file(split)
            if humaneval_file and humaneval_file.exists():
                with open(humaneval_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            sample = json.loads(line)
                            # Convert HumanEval format to our format
                            converted = self._convert_humaneval_sample(sample)
                            data_by_type["code"].append(converted)
                            source_stats["humaneval"] += 1
                print(f"✅ 加载 HumanEval code: {len(data_by_type['code'])} 样本")

        # 2. Load from processed directory for test/val (contains source field)
        if split in ["val", "test"]:
            # 验证集使用balanced_val.jsonl（均匀分配）
            if split == "val":
                dataset_file = self.data_dir / "balanced_val.jsonl"
                if not dataset_file.exists():
                    # 后备：使用processed目录
                    dataset_file = self.data_dir / "processed/val_mixed.jsonl"
                    print(f"⚠️  balanced_val.jsonl不存在，使用processed/val_mixed.jsonl")
            else:
                # 🚀 测试集：优先使用ready_to_train/test.jsonl（100个样本的混合测试集）
                ready_test_file = self.data_dir / "ready_to_train/test.jsonl"
                if ready_test_file.exists():
                    dataset_file = ready_test_file
                    print(f"✅ 使用ready_to_train测试集: {ready_test_file}")
                else:
                    # 后备：使用processed目录
                    dataset_file = self.data_dir / f"processed/{split}_mixed.jsonl"
                    print(f"⚠️  ready_to_train/test.jsonl不存在，使用processed/{split}_mixed.jsonl")
        else:
            # 训练数据：P6修复 - 优先使用配置的数据集路径
            if split == "train":
                if self.train_dataset_path:
                    # 使用配置指定的路径
                    config_path = Path(self.train_dataset_path)
                    if not config_path.is_absolute():
                        config_path = Path.home() / ".claude" / self.train_dataset_path
                    if config_path.exists():
                        dataset_file = config_path
                        print(f"✅ 使用配置训练集: {config_path}")
                    else:
                        print(f"⚠️  配置训练集不存在: {config_path}，尝试默认路径")
                        self.train_dataset_path = None  # 清除，使用默认

                if not self.train_dataset_path:
                    # 后备：默认路径
                    ready_train_file = self.data_dir / "ready_to_train/train_10k_final.jsonl"  # P6: 更新默认路径
                    if ready_train_file.exists():
                        dataset_file = ready_train_file
                        print(f"✅ 使用ready_to_train训练集: {ready_train_file.name}")
                    else:
                        # 进一步后备
                        ready_train_file = self.data_dir / "ready_to_train/train_llm_cleaned.jsonl"
                        if ready_train_file.exists():
                            dataset_file = ready_train_file
                            print(f"✅ 使用ready_to_train训练集: {ready_train_file.name}")
                        else:
                            # 后备：使用mixed目录的大数据集
                            fixed_file = self.data_dir / "mixed/train_mixed_with_math_fixed.jsonl"
                            math_file = self.data_dir / "mixed/train_mixed_with_math.jsonl"
                            balanced_file = self.data_dir / "mixed/train_mixed_balanced.jsonl"

                            if fixed_file.exists():
                                dataset_file = fixed_file
                                print(f"⚠️  使用后备: {fixed_file.name}")
                            elif math_file.exists():
                                dataset_file = math_file
                                print(f"⚠️  使用后备: {math_file.name}")
                            elif balanced_file.exists():
                                dataset_file = balanced_file
                            else:
                                dataset_file = self.data_dir / f"mixed/{split}_mixed.jsonl"
            else:
                dataset_file = self.data_dir / f"mixed/{split}_mixed.jsonl"
            if not dataset_file.exists():
                dataset_file = self.data_dir / f"{split}/mixed_dataset.jsonl"

        if dataset_file.exists():
            with open(dataset_file, 'r') as f:
                for line in f:
                    if line.strip():
                        sample = json.loads(line)

                        # 🔴 过滤MBPP数据集 - 数据质量问题
                        if sample.get("source") == "mbpp":
                            continue

                        problem_type = sample.get("problem_type", "math")
                        # 保留source字段（如果存在）
                        if "source" in sample:
                            source_stats[sample["source"]] += 1
                        data_by_type[problem_type].append(sample)

        # 打乱
        if self.shuffle:
            for ptype in data_by_type:
                random.shuffle(data_by_type[ptype])

        print(f"✅ 加载 {split.upper()} 数据:")
        for ptype, samples in data_by_type.items():
            print(f"  {ptype}: {len(samples)} 样本")

        # 打印source统计
        if source_stats:
            print(f"\n📊 数据源分布:")
            for source, count in sorted(source_stats.items()):
                print(f"  {source}: {count} 样本")

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
            "source": "humaneval",  # NEW - 添加source字段
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
            "source": "humaneval",  # 添加source字段用于追踪
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
