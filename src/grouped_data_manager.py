#!/usr/bin/env python3
"""
分组数据管理器 - 确保每个 step 包含 math/qa/code 三种类型

设计:
1. 每个 step 采样 3 个问题组（每种类型 1 个）
2. 每个问题组包含 2 easy + 2 hard 问题
3. 总计每 step: 3 组 × 4 问题 = 12 问题
4. 每个 workflow 在 4 个问题上评分，加权计算最终得分
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict


class GroupedDataManager:
    """分组数据管理器 - 确保每 step 包含三种类型"""

    def __init__(
        self,
        data_dir: str = "data/grouped",
        groups_per_domain: int = 1,  # 每种领域每 step 采样几组
        shuffle: bool = True
    ):
        """
        Args:
            data_dir: 分组数据目录
            groups_per_domain: 每种领域每 step 采样的组数
            shuffle: 是否打乱数据
        """
        self.data_dir = Path(data_dir)
        self.groups_per_domain = groups_per_domain
        self.shuffle = shuffle

        # 按领域存储的数据
        self.train_data = {"math": [], "qa": [], "code": []}
        self.val_data = {"math": [], "qa": [], "code": []}
        self.test_data = {"math": [], "qa": [], "code": []}

        # 当前迭代位置
        self.current_indices = {"math": 0, "qa": 0, "code": 0}

    def load_grouped_data(self, filepath: Path) -> Dict[str, List[Dict]]:
        """加载分组数据，按领域分类"""
        data_by_domain = defaultdict(list)

        if not filepath.exists():
            print(f"⚠️  文件不存在: {filepath}")
            return dict(data_by_domain)

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        group = json.loads(line)
                        domain = group.get('domain', 'math')
                        data_by_domain[domain].append(group)
                    except json.JSONDecodeError as e:
                        print(f"JSON解析错误: {e}")

        return dict(data_by_domain)

    def initialize(self):
        """初始化数据"""
        print("=" * 60)
        print("📂 初始化分组数据管理器")
        print("=" * 60)

        # 加载训练数据
        train_file = self.data_dir / "grouped_train.jsonl"
        self.train_data = self.load_grouped_data(train_file)

        # 加载验证数据
        val_file = self.data_dir / "grouped_val.jsonl"
        self.val_data = self.load_grouped_data(val_file)

        # 加载测试数据
        test_file = self.data_dir / "grouped_test.jsonl"
        self.test_data = self.load_grouped_data(test_file)

        # 打乱数据
        if self.shuffle:
            for domain in ["math", "qa", "code"]:
                random.shuffle(self.train_data.get(domain, []))
                random.shuffle(self.val_data.get(domain, []))
                random.shuffle(self.test_data.get(domain, []))

        # 统计
        print(f"\n📊 训练集分组统计:")
        total_train_groups = 0
        total_train_problems = 0
        for domain in ["math", "qa", "code"]:
            count = len(self.train_data.get(domain, []))
            total_train_groups += count
            total_train_problems += count * 4
            print(f"  {domain}: {count} 组 ({count * 4} 问题)")
        print(f"  总计: {total_train_groups} 组 ({total_train_problems} 问题)")

        print(f"\n📊 验证集分组统计:")
        for domain in ["math", "qa", "code"]:
            count = len(self.val_data.get(domain, []))
            print(f"  {domain}: {count} 组")

        print(f"\n📊 测试集分组统计:")
        for domain in ["math", "qa", "code"]:
            count = len(self.test_data.get(domain, []))
            print(f"  {domain}: {count} 组")

        print(f"\n🎯 每 step 采样配置:")
        print(f"  每种领域: {self.groups_per_domain} 组")
        print(f"  总组数: {self.groups_per_domain * 3} 组")
        print(f"  总问题数: {self.groups_per_domain * 3 * 4} 问题")

        print("=" * 60)

    def sample_step_groups(
        self,
        split: str = "train",
        groups_per_domain: Optional[int] = None
    ) -> List[Dict]:
        """
        采样一个 step 的问题组（确保包含 math/qa/code）

        Args:
            split: 数据分割 (train/val/test)
            groups_per_domain: 每种领域采样组数（None 使用默认值）

        Returns:
            问题组列表（每种领域各 N 组）
        """
        n_groups = groups_per_domain or self.groups_per_domain

        # 选择数据源
        if split == "train":
            data_source = self.train_data
        elif split == "val":
            data_source = self.val_data
        else:
            data_source = self.test_data

        step_groups = []

        for domain in ["math", "qa", "code"]:
            domain_data = data_source.get(domain, [])

            if len(domain_data) == 0:
                print(f"⚠️  {domain} 数据为空，跳过")
                continue

            # 采样 n_groups 个组
            for _ in range(n_groups):
                idx = self.current_indices[domain] % len(domain_data)
                step_groups.append(domain_data[idx])

                # 更新索引
                self.current_indices[domain] += 1

                # 如果一轮结束，重新打乱
                if self.current_indices[domain] % len(domain_data) == 0:
                    if self.shuffle:
                        random.shuffle(domain_data)

        # 打乱组顺序（但保证每种类型都有）
        if self.shuffle:
            random.shuffle(step_groups)

        return step_groups

    def get_step_stats(self, groups: List[Dict]) -> Dict[str, int]:
        """获取一个 step 的统计信息"""
        stats = defaultdict(int)
        problem_count = defaultdict(int)

        for group in groups:
            domain = group.get('domain', 'unknown')
            stats[domain] += 1
            problem_count[domain] += len(group.get('problems', []))

        return {
            'groups': dict(stats),
            'problems': dict(problem_count),
            'total_groups': sum(stats.values()),
            'total_problems': sum(problem_count.values())
        }

    def reset_indices(self):
        """重置采样索引"""
        self.current_indices = {"math": 0, "qa": 0, "code": 0}
        print("✅ 分组采样索引已重置")

    def flatten_groups_to_problems(
        self,
        groups: List[Dict]
    ) -> List[Dict]:
        """
        将问题组展平为问题列表（兼容旧接口）

        Args:
            groups: 问题组列表

        Returns:
            展平的问题列表，每个问题包含 group_id 和完整元数据
        """
        problems = []

        for group in groups:
            group_id = group['group_id']
            domain = group['domain']

            for problem in group['problems']:
                flat_problem = {
                    # 基本信息
                    'problem': problem['question'],
                    'problem_type': domain,
                    'ground_truth': problem['answer'],
                    'source': problem.get('source', domain),

                    # 分组信息
                    'group_id': group_id,
                    'difficulty': problem['difficulty'],
                    'weight': problem['weight'],

                    # 代码任务特殊字段
                    'entry_point': problem.get('entry_point', ''),
                    'test': problem.get('test_cases', []),
                    'context': problem.get('context', ''),

                    # 元数据
                    'meta': {
                        'group_id': group_id,
                        'problem_id': problem['id'],
                        'difficulty': problem['difficulty'],
                        'weight': problem['weight'],
                        'entry_point': problem.get('entry_point', ''),
                        'test_cases': problem.get('test_cases', [])
                    }
                }
                problems.append(flat_problem)

        return problems


def test_grouped_data_manager():
    """测试分组数据管理器"""
    print("\n" + "=" * 60)
    print("🧪 测试分组数据管理器")
    print("=" * 60)

    manager = GroupedDataManager(
        data_dir="data/grouped",
        groups_per_domain=1
    )

    manager.initialize()

    # 测试采样
    print("\n🎲 测试采样 3 个 step:")
    for step in range(3):
        groups = manager.sample_step_groups(split="train")
        stats = manager.get_step_stats(groups)
        print(f"\n  Step {step + 1}:")
        print(f"    组分布: {stats['groups']}")
        print(f"    问题分布: {stats['problems']}")
        print(f"    Group IDs: {[g['group_id'] for g in groups]}")

    # 测试展平
    print("\n📋 测试展平问题组:")
    groups = manager.sample_step_groups(split="train")
    flat_problems = manager.flatten_groups_to_problems(groups)
    print(f"  问题组数: {len(groups)}")
    print(f"  展平后问题数: {len(flat_problems)}")

    # 打印第一个问题的结构
    if flat_problems:
        print(f"\n  示例问题结构:")
        p = flat_problems[0]
        print(f"    group_id: {p['group_id']}")
        print(f"    problem_type: {p['problem_type']}")
        print(f"    difficulty: {p['difficulty']}")
        print(f"    weight: {p['weight']}")
        print(f"    问题前100字符: {p['problem'][:100]}...")


if __name__ == "__main__":
    test_grouped_data_manager()
