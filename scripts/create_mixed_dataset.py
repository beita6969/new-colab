#!/usr/bin/env python3
"""
创建混合数据集：
- 训练集：从各数据集的训练数据中采样
- 验证集：从测试集中按相同比例采样
- 小样本数据集全部保留
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def load_json(file_path: Path) -> List[Dict[str, Any]]:
    """加载JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, dict) and 'data' in data:
            return data['data']
        elif isinstance(data, list):
            return data
        else:
            return [data]

def convert_to_unified_format(sample: Dict[str, Any], source: str, problem_type: str) -> Dict[str, Any]:
    """转换为统一格式"""
    unified = {
        'source': source,
        'problem_type': problem_type,
    }

    # GSM8K
    if source == 'gsm8k':
        unified['problem'] = sample.get('question', '')
        answer = sample.get('answer', '')
        if '####' in answer:
            unified['ground_truth'] = answer.split('####')[-1].strip()
            unified['solution'] = answer
        else:
            unified['ground_truth'] = answer
            unified['solution'] = answer

    # HumanEval
    elif source == 'humaneval':
        unified['problem'] = sample.get('prompt', '')
        unified['ground_truth'] = sample.get('canonical_solution', '')
        unified['entry_point'] = sample.get('entry_point', '')
        unified['test'] = sample.get('test', '')
        unified['task_id'] = sample.get('task_id', '')

    # MBPP
    elif source == 'mbpp':
        unified['problem'] = sample.get('text', sample.get('prompt', ''))
        unified['ground_truth'] = sample.get('code', '')
        unified['test_list'] = sample.get('test_list', [])
        unified['task_id'] = sample.get('task_id', '')

    # CommonsenseQA
    elif source == 'commonsenseqa':
        question = sample.get('question', {})
        if isinstance(question, dict):
            question_text = question.get('stem', '')
        else:
            question_text = str(question)

        # 从question中提取choices
        if isinstance(question, dict) and 'choices' in question:
            choices_list = question.get('choices', [])
            if isinstance(choices_list, list) and len(choices_list) > 0:
                # choices是一个list of dicts
                labels = [c.get('label', '') for c in choices_list]
                texts = [c.get('text', '') for c in choices_list]
                choices_str = ' '.join([f"{l}. {t}" for l, t in zip(labels, texts)])
                choices_dict = {'label': labels, 'text': texts}
            else:
                choices_str = ''
                choices_dict = {}
        else:
            choices_str = ''
            choices_dict = {}

        unified['problem'] = f"{question_text} Choices: {choices_str}"
        unified['ground_truth'] = sample.get('answerKey', '')
        unified['choices'] = choices_dict

    # HotpotQA
    elif source == 'hotpotqa':
        unified['problem'] = sample.get('question', '')
        unified['ground_truth'] = sample.get('answer', '')
        unified['supporting_facts'] = sample.get('supporting_facts', [])

    # MMLU
    elif source == 'mmlu':
        question = sample.get('question', '')
        choices = sample.get('choices', [])
        if choices:
            choices_str = ' '.join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
            unified['problem'] = f"{question} Choices: {choices_str}"
        else:
            unified['problem'] = question

        # MMLU的答案是数字索引，需要转换为字母
        answer = sample.get('answer', '')
        if isinstance(answer, int) and 0 <= answer < 26:
            unified['ground_truth'] = chr(65 + answer)  # 0->A, 1->B, etc.
        else:
            unified['ground_truth'] = str(answer)

        unified['subject'] = sample.get('subject', '')
        unified['choices'] = choices  # 保存原始choices列表

    return unified

def create_mixed_dataset(
    raw_dir: Path,
    output_dir: Path,
    train_size: int = 10000,
    keep_small_threshold: int = 500,
    seed: int = 42
):
    """创建混合数据集"""
    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 类型比例配置
    type_ratios = {
        'math': 0.20,
        'code': 0.20,
        'qa': 0.30,
        'mixed': 0.30
    }

    print("="*60)
    print("创建混合数据集")
    print("="*60)

    # 存储训练和测试数据
    train_samples_by_type = defaultdict(list)
    test_samples_by_type = defaultdict(list)
    dataset_stats = defaultdict(lambda: {'train': 0, 'test': 0})

    print("\n加载数据集...")

    # GSM8K
    gsm8k_train = raw_dir / 'gsm8k' / 'train.jsonl'
    gsm8k_test = raw_dir / 'gsm8k' / 'test.jsonl'
    if gsm8k_train.exists():
        for sample in load_jsonl(gsm8k_train):
            unified = convert_to_unified_format(sample, 'gsm8k', 'math')
            train_samples_by_type['math'].append(unified)
            dataset_stats['gsm8k']['train'] += 1
    if gsm8k_test.exists():
        for sample in load_jsonl(gsm8k_test):
            unified = convert_to_unified_format(sample, 'gsm8k', 'math')
            test_samples_by_type['math'].append(unified)
            dataset_stats['gsm8k']['test'] += 1
    print(f"  GSM8K: train={dataset_stats['gsm8k']['train']}, test={dataset_stats['gsm8k']['test']}")

    # HumanEval (只有测试集)
    humaneval_file = raw_dir / 'humaneval' / 'HumanEval.jsonl'
    if humaneval_file.exists():
        data = load_jsonl(humaneval_file)
        # 小样本，全部作为测试集
        for sample in data:
            unified = convert_to_unified_format(sample, 'humaneval', 'code')
            test_samples_by_type['code'].append(unified)
            dataset_stats['humaneval']['test'] += 1
    print(f"  HumanEval: test={dataset_stats['humaneval']['test']}")

    # MBPP
    mbpp_train = raw_dir / 'mbpp' / 'train.jsonl'
    mbpp_test = raw_dir / 'mbpp' / 'test.jsonl'
    mbpp_val = raw_dir / 'mbpp' / 'validation.jsonl'

    if mbpp_train.exists():
        for sample in load_jsonl(mbpp_train):
            unified = convert_to_unified_format(sample, 'mbpp', 'code')
            train_samples_by_type['code'].append(unified)
            dataset_stats['mbpp']['train'] += 1

    # MBPP的test和validation都算作测试集
    if mbpp_test.exists():
        for sample in load_jsonl(mbpp_test):
            unified = convert_to_unified_format(sample, 'mbpp', 'code')
            test_samples_by_type['code'].append(unified)
            dataset_stats['mbpp']['test'] += 1
    if mbpp_val.exists():
        for sample in load_jsonl(mbpp_val):
            unified = convert_to_unified_format(sample, 'mbpp', 'code')
            test_samples_by_type['code'].append(unified)
            dataset_stats['mbpp']['test'] += 1
    print(f"  MBPP: train={dataset_stats['mbpp']['train']}, test={dataset_stats['mbpp']['test']}")

    # CommonsenseQA
    csqa_train = raw_dir / 'commonsenseqa' / 'train.jsonl'
    csqa_test = raw_dir / 'commonsenseqa' / 'test.jsonl'
    if csqa_train.exists():
        for sample in load_jsonl(csqa_train):
            unified = convert_to_unified_format(sample, 'commonsenseqa', 'qa')
            train_samples_by_type['qa'].append(unified)
            dataset_stats['commonsenseqa']['train'] += 1
    if csqa_test.exists():
        for sample in load_jsonl(csqa_test):
            unified = convert_to_unified_format(sample, 'commonsenseqa', 'qa')
            test_samples_by_type['qa'].append(unified)
            dataset_stats['commonsenseqa']['test'] += 1
    print(f"  CommonsenseQA: train={dataset_stats['commonsenseqa']['train']}, test={dataset_stats['commonsenseqa']['test']}")

    # HotpotQA (只有dev)
    hotpot_file = raw_dir / 'hotpotqa' / 'dev_distractor.json'
    if hotpot_file.exists():
        data = load_json(hotpot_file)
        # 分一半作为训练，一半作为测试
        random.shuffle(data)
        split_point = len(data) // 2
        for sample in data[:split_point]:
            unified = convert_to_unified_format(sample, 'hotpotqa', 'qa')
            train_samples_by_type['qa'].append(unified)
            dataset_stats['hotpotqa']['train'] += 1
        for sample in data[split_point:]:
            unified = convert_to_unified_format(sample, 'hotpotqa', 'qa')
            test_samples_by_type['qa'].append(unified)
            dataset_stats['hotpotqa']['test'] += 1
    print(f"  HotpotQA: train={dataset_stats['hotpotqa']['train']}, test={dataset_stats['hotpotqa']['test']}")

    # MMLU
    mmlu_train = raw_dir / 'mmlu' / 'auxiliary_train.jsonl'
    mmlu_test = raw_dir / 'mmlu' / 'test.jsonl'
    if mmlu_train.exists():
        for sample in load_jsonl(mmlu_train):
            unified = convert_to_unified_format(sample, 'mmlu', 'mixed')
            train_samples_by_type['mixed'].append(unified)
            dataset_stats['mmlu']['train'] += 1
    if mmlu_test.exists():
        for sample in load_jsonl(mmlu_test):
            unified = convert_to_unified_format(sample, 'mmlu', 'mixed')
            test_samples_by_type['mixed'].append(unified)
            dataset_stats['mmlu']['test'] += 1
    print(f"  MMLU: train={dataset_stats['mmlu']['train']}, test={dataset_stats['mmlu']['test']}")

    # 统计信息
    print("\n数据集总计:")
    for ptype in ['math', 'code', 'qa', 'mixed']:
        train_count = len(train_samples_by_type[ptype])
        test_count = len(test_samples_by_type[ptype])
        print(f"  {ptype}: train={train_count}, test={test_count}")

    # 创建训练集
    print(f"\n创建训练集（目标: {train_size}个样本）...")
    train_samples = []
    train_distribution = {}

    for ptype, ratio in type_ratios.items():
        target = int(train_size * ratio)
        available = len(train_samples_by_type[ptype])

        if available <= keep_small_threshold:
            # 小数据集全部保留
            sampled = train_samples_by_type[ptype]
            print(f"  {ptype}: 保留全部 {len(sampled)} 个样本（小数据集）")
        else:
            # 大数据集采样
            sample_count = min(target, available)
            sampled = random.sample(train_samples_by_type[ptype], sample_count)
            print(f"  {ptype}: 采样 {len(sampled)}/{available} 个样本")

        train_samples.extend(sampled)
        train_distribution[ptype] = len(sampled)

    random.shuffle(train_samples)
    print(f"\n训练集总计: {len(train_samples)} 个样本")

    # 创建验证集（从测试集中按相同比例采样）
    print("\n创建验证集（从测试集采样，保持相同比例）...")
    val_samples = []
    val_distribution = {}

    # 计算每个类型在训练集中的实际比例
    actual_ratios = {}
    for ptype in type_ratios.keys():
        actual_ratios[ptype] = train_distribution[ptype] / len(train_samples)

    # 目标验证集大小（训练集的10%）
    val_size = int(len(train_samples) * 0.1)

    for ptype, ratio in actual_ratios.items():
        target = int(val_size * ratio)
        available = len(test_samples_by_type[ptype])

        if available == 0:
            print(f"  {ptype}: 无测试数据可用")
            continue

        if available <= keep_small_threshold:
            # 小数据集全部用作验证
            sampled = test_samples_by_type[ptype]
            print(f"  {ptype}: 保留全部 {len(sampled)} 个样本（小数据集）")
        else:
            # 按比例采样
            sample_count = min(target, available)
            sampled = random.sample(test_samples_by_type[ptype], sample_count)
            print(f"  {ptype}: 采样 {len(sampled)}/{available} 个样本（比例: {ratio:.1%}）")

        val_samples.extend(sampled)
        val_distribution[ptype] = len(sampled)

    random.shuffle(val_samples)
    print(f"\n验证集总计: {len(val_samples)} 个样本")

    # 保存数据集
    print("\n保存数据集...")

    train_file = output_dir / 'train_mixed.jsonl'
    with open(train_file, 'w', encoding='utf-8') as f:
        for sample in train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"  训练集: {train_file}")

    val_file = output_dir / 'val_mixed.jsonl'
    with open(val_file, 'w', encoding='utf-8') as f:
        for sample in val_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"  验证集: {val_file}")

    # 保存统计信息
    stats = {
        'train_size': len(train_samples),
        'val_size': len(val_samples),
        'train_distribution': train_distribution,
        'val_distribution': val_distribution,
        'train_ratios': {k: v/len(train_samples) for k, v in train_distribution.items()},
        'val_ratios': {k: v/len(val_samples) for k, v in val_distribution.items()},
        'dataset_stats': dict(dataset_stats),
        'config': {
            'target_train_size': train_size,
            'keep_small_threshold': keep_small_threshold,
            'seed': seed,
            'type_ratios': type_ratios
        }
    }

    stats_file = output_dir / 'dataset_stats.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  统计信息: {stats_file}")

    # 打印最终分布
    print("\n=" * 60)
    print("最终数据集分布")
    print("=" * 60)
    print(f"\n训练集 ({len(train_samples)} 个样本):")
    for ptype, count in sorted(train_distribution.items()):
        ratio = count / len(train_samples) * 100
        print(f"  {ptype:8s}: {count:5d} ({ratio:5.1f}%)")

    print(f"\n验证集 ({len(val_samples)} 个样本):")
    for ptype, count in sorted(val_distribution.items()):
        ratio = count / len(val_samples) * 100 if len(val_samples) > 0 else 0
        print(f"  {ptype:8s}: {count:5d} ({ratio:5.1f}%)")

    print(f"\n数据集保存位置: {output_dir.absolute()}")
    print("="*60)

if __name__ == '__main__':
    raw_dir = Path('data/raw')
    output_dir = Path('data/mixed')

    create_mixed_dataset(
        raw_dir=raw_dir,
        output_dir=output_dir,
        train_size=10000,
        keep_small_threshold=500,
        seed=42
    )
