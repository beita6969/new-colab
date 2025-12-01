#!/usr/bin/env python3
"""
兼容AFlow的evaluator模块
提供DatasetType枚举和评估相关功能
"""
from enum import Enum
from typing import Optional


class DatasetType(Enum):
    """数据集类型枚举"""
    GSM8K = "gsm8k"
    MATH = "math"
    HOTPOTQA = "hotpotqa"
    HUMANEVAL = "humaneval"
    MBPP = "mbpp"
    COMMONSENSEQA = "commonsenseqa"
    MMLU = "mmlu"
    SQUAD_V2 = "squad_v2"
    DROP = "drop"
    CUSTOM = "custom"

    @classmethod
    def from_string(cls, name: str) -> 'DatasetType':
        """从字符串获取数据集类型"""
        name_lower = name.lower().replace('-', '_').replace(' ', '_')
        for dataset in cls:
            if dataset.value == name_lower:
                return dataset
        return cls.CUSTOM

    @classmethod
    def get_problem_type(cls, dataset_type: 'DatasetType') -> str:
        """获取问题类型"""
        math_datasets = {cls.GSM8K, cls.MATH}
        code_datasets = {cls.HUMANEVAL, cls.MBPP}
        qa_datasets = {cls.HOTPOTQA, cls.COMMONSENSEQA, cls.MMLU, cls.SQUAD_V2, cls.DROP}

        if dataset_type in math_datasets:
            return "math"
        elif dataset_type in code_datasets:
            return "code"
        elif dataset_type in qa_datasets:
            return "qa"
        else:
            return "unknown"


def get_dataset_type_from_source(source: str) -> DatasetType:
    """从source字符串获取DatasetType"""
    return DatasetType.from_string(source)


def evaluate_answer(
    prediction: str,
    ground_truth: str,
    dataset_type: DatasetType = DatasetType.CUSTOM
) -> float:
    """
    评估答案的简单实现

    Args:
        prediction: 预测答案
        ground_truth: 真实答案
        dataset_type: 数据集类型

    Returns:
        分数 (0.0 或 1.0)
    """
    # 标准化
    pred = str(prediction).strip().lower()
    truth = str(ground_truth).strip().lower()

    # 精确匹配
    if pred == truth:
        return 1.0

    # 包含匹配
    if truth in pred or pred in truth:
        return 0.7

    return 0.0
