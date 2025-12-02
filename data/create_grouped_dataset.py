#!/usr/bin/env python3
"""
创建分组数据集 - 每组包含 2 easy + 2 hard 问题（同学科）

设计目标:
1. 让每个 workflow 在多个不同难度问题上运行
2. 加权计算综合得分: score = 0.3 * easy_avg + 0.7 * hard_avg
3. 确保组内有区分度，产生非零梯度

数据结构:
{
    "group_id": "math_001",
    "domain": "math",
    "problems": [
        {"id": "easy_0", "difficulty": "easy", "question": "...", "answer": "...", "source": "gsm8k"},
        {"id": "easy_1", "difficulty": "easy", "question": "...", "answer": "...", "source": "gsm8k"},
        {"id": "hard_0", "difficulty": "hard", "question": "...", "answer": "...", "source": "math"},
        {"id": "hard_1", "difficulty": "hard", "question": "...", "answer": "...", "source": "math"}
    ]
}
"""

import json
import os
import random
from typing import Dict, List, Any
from collections import defaultdict

# 数据源配置
DATA_SOURCES = {
    "math": {
        "easy": "/home/claude-user/AFlow/data/processed/gsm8k_all.jsonl",
        "hard": "/home/claude-user/AFlow/data/processed/math_all.jsonl"
    },
    "qa": {
        "easy": "/home/claude-user/AFlow/data/processed/drop_all.jsonl",
        "hard": "/home/claude-user/AFlow/data/processed/hotpotqa_all.jsonl"
    },
    "code": {
        "easy": "/home/claude-user/AFlow/data/processed/mbpp_all.jsonl",
        "hard": "/home/claude-user/AFlow/data/processed/humaneval_all.jsonl"
    }
}

# 权重配置
WEIGHTS = {
    "easy": 0.3,
    "hard": 0.7
}

def load_jsonl(filepath: str) -> List[Dict]:
    """加载 JSONL 文件"""
    data = []
    if not os.path.exists(filepath):
        print(f"⚠️  文件不存在: {filepath}")
        return data

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"JSON解析错误: {e}")
    return data

def save_jsonl(data: List[Dict], filepath: str):
    """保存为 JSONL 文件"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def create_problem_groups(
    domain: str,
    easy_data: List[Dict],
    hard_data: List[Dict],
    num_groups: int,
    easy_per_group: int = 2,
    hard_per_group: int = 2,
    seed: int = 42
) -> List[Dict]:
    """
    创建问题分组

    Args:
        domain: 学科 (math/qa/code)
        easy_data: 简单题目列表
        hard_data: 困难题目列表
        num_groups: 要创建的组数
        easy_per_group: 每组简单题数量
        hard_per_group: 每组困难题数量
        seed: 随机种子

    Returns:
        问题组列表
    """
    random.seed(seed)

    # 打乱数据
    easy_shuffled = easy_data.copy()
    hard_shuffled = hard_data.copy()
    random.shuffle(easy_shuffled)
    random.shuffle(hard_shuffled)

    # 计算可用组数
    max_groups_easy = len(easy_shuffled) // easy_per_group
    max_groups_hard = len(hard_shuffled) // hard_per_group
    actual_groups = min(num_groups, max_groups_easy, max_groups_hard)

    print(f"  📊 {domain}: easy={len(easy_data)}, hard={len(hard_data)} → {actual_groups} 组")

    groups = []
    for i in range(actual_groups):
        # 取出问题
        easy_problems = easy_shuffled[i*easy_per_group : (i+1)*easy_per_group]
        hard_problems = hard_shuffled[i*hard_per_group : (i+1)*hard_per_group]

        # 构建组
        problems = []
        for j, p in enumerate(easy_problems):
            problems.append({
                "id": f"easy_{j}",
                "difficulty": "easy",
                "weight": WEIGHTS["easy"] / easy_per_group,  # 每题权重
                "question": p["question"],
                "answer": p["answer"],
                "source": p.get("source", domain),
                "domain": domain,
                # 代码任务特殊字段
                "entry_point": p.get("entry_point", ""),
                "test_cases": p.get("test_cases", []),
                "context": p.get("context", "")
            })

        for j, p in enumerate(hard_problems):
            problems.append({
                "id": f"hard_{j}",
                "difficulty": "hard",
                "weight": WEIGHTS["hard"] / hard_per_group,  # 每题权重
                "question": p["question"],
                "answer": p["answer"],
                "source": p.get("source", domain),
                "domain": domain,
                "entry_point": p.get("entry_point", ""),
                "test_cases": p.get("test_cases", []),
                "context": p.get("context", "")
            })

        group = {
            "group_id": f"{domain}_{i:04d}",
            "domain": domain,
            "num_easy": easy_per_group,
            "num_hard": hard_per_group,
            "weight_easy": WEIGHTS["easy"],
            "weight_hard": WEIGHTS["hard"],
            "problems": problems
        }
        groups.append(group)

    return groups

def main():
    print("="*60)
    print("创建分组数据集 (2 easy + 2 hard per group)")
    print("="*60)

    # 配置
    output_dir = "/home/claude-user/colab/data/grouped"
    os.makedirs(output_dir, exist_ok=True)

    # 每个学科的目标组数
    groups_per_domain = {
        "math": 300,   # GSM8K 1319, MATH 605 → max 302 groups
        "qa": 500,     # DROP 1000, HotpotQA 1000 → max 500 groups
        "code": 80     # MBPP 427, HumanEval 164 → max 82 groups
    }

    all_groups = []
    domain_groups = {}

    for domain, sources in DATA_SOURCES.items():
        print(f"\n📂 处理 {domain.upper()} 数据...")

        # 加载数据
        easy_data = load_jsonl(sources["easy"])
        hard_data = load_jsonl(sources["hard"])

        # 创建分组
        groups = create_problem_groups(
            domain=domain,
            easy_data=easy_data,
            hard_data=hard_data,
            num_groups=groups_per_domain[domain],
            easy_per_group=2,
            hard_per_group=2
        )

        domain_groups[domain] = groups
        all_groups.extend(groups)

        # 保存单学科文件
        save_jsonl(groups, os.path.join(output_dir, f"grouped_{domain}.jsonl"))
        print(f"  ✅ 保存: grouped_{domain}.jsonl ({len(groups)} 组)")

    # 打乱并保存总文件
    random.seed(42)
    random.shuffle(all_groups)
    save_jsonl(all_groups, os.path.join(output_dir, "grouped_all.jsonl"))
    print(f"\n✅ 保存: grouped_all.jsonl ({len(all_groups)} 组)")

    # 分割训练/验证/测试
    n_total = len(all_groups)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)

    train_groups = all_groups[:n_train]
    val_groups = all_groups[n_train:n_train+n_val]
    test_groups = all_groups[n_train+n_val:]

    save_jsonl(train_groups, os.path.join(output_dir, "grouped_train.jsonl"))
    save_jsonl(val_groups, os.path.join(output_dir, "grouped_val.jsonl"))
    save_jsonl(test_groups, os.path.join(output_dir, "grouped_test.jsonl"))

    print(f"\n📊 数据集统计:")
    print(f"  训练集: {len(train_groups)} 组 ({len(train_groups)*4} 问题)")
    print(f"  验证集: {len(val_groups)} 组 ({len(val_groups)*4} 问题)")
    print(f"  测试集: {len(test_groups)} 组 ({len(test_groups)*4} 问题)")

    # 统计各学科分布
    print(f"\n📊 学科分布 (总计):")
    for domain in ["math", "qa", "code"]:
        count = len(domain_groups[domain])
        print(f"  {domain}: {count} 组 ({count*4} 问题)")

    # 保存配置信息
    config = {
        "easy_per_group": 2,
        "hard_per_group": 2,
        "weight_easy": WEIGHTS["easy"],
        "weight_hard": WEIGHTS["hard"],
        "total_groups": len(all_groups),
        "train_groups": len(train_groups),
        "val_groups": len(val_groups),
        "test_groups": len(test_groups),
        "domain_groups": {k: len(v) for k, v in domain_groups.items()}
    }
    with open(os.path.join(output_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n✅ 配置已保存到 config.json")
    print("\n" + "="*60)
    print("权重配置:")
    print(f"  Easy: {WEIGHTS['easy']:.1%} (每题 {WEIGHTS['easy']/2:.1%})")
    print(f"  Hard: {WEIGHTS['hard']:.1%} (每题 {WEIGHTS['hard']/2:.1%})")
    print("="*60)

if __name__ == "__main__":
    main()
