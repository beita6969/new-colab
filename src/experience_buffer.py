#!/usr/bin/env python3
"""
高质量样本管理器 - 用于提示词优化的经验回放
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from difflib import SequenceMatcher
import numpy as np


class ExperienceBuffer:
    """
    高质量样本缓冲区

    功能：
    1. 收集高奖励样本（reward > threshold）
    2. 按问题类型分类存储（math/code/qa）
    3. Top-K排序和检索
    4. Few-shot示例检索（基于相似度）
    5. Operator级示例检索
    6. 持久化到磁盘
    """

    def __init__(
        self,
        buffer_size: int = 100,
        reward_threshold: float = 8.0,
        persistence_dir: str = "data/experience_buffer",
        problem_types: List[str] = ["math", "code", "qa"]
    ):
        """
        Args:
            buffer_size: 每个问题类型保留的最大样本数
            reward_threshold: 高质量样本的奖励阈值
            persistence_dir: 持久化目录
            problem_types: 支持的问题类型列表
        """
        self.buffer_size = buffer_size
        self.reward_threshold = reward_threshold
        self.persistence_dir = Path(persistence_dir)
        self.problem_types = problem_types

        # 初始化缓冲区 {problem_type: [samples]}
        self.buffers = {pt: [] for pt in problem_types}

        # 创建持久化目录
        self.persistence_dir.mkdir(parents=True, exist_ok=True)

        # 加载已有样本
        self.load()

    def add_sample(self, sample: Dict, problem_type: str) -> bool:
        """
        添加样本到缓冲区

        Args:
            sample: 样本字典，包含:
                - problem: 问题文本
                - workflow_code: 生成的工作流代码
                - answer: 模型输出
                - ground_truth: 正确答案
                - reward: 总奖励
                - correctness_score: 正确性得分
                - metadata: 元数据（cost, execution_time, num_operators等）
                - step: 训练步数
            problem_type: 问题类型 (math/code/qa)

        Returns:
            是否成功添加（reward >= threshold则添加）
        """
        # 检查奖励是否达标
        if sample.get('reward', -10.0) < self.reward_threshold:
            return False

        # 检查问题类型
        if problem_type not in self.buffers:
            print(f"⚠️  未知问题类型: {problem_type}，跳过")
            return False

        buffer = self.buffers[problem_type]

        # 添加样本
        buffer.append(sample)

        # 按奖励降序排序
        buffer.sort(key=lambda x: x.get('reward', -10.0), reverse=True)

        # 保留top-k
        if len(buffer) > self.buffer_size:
            buffer = buffer[:self.buffer_size]

        self.buffers[problem_type] = buffer

        return True

    def retrieve_top_k(
        self,
        problem: str,
        problem_type: str,
        k: int = 3,
        similarity_threshold: float = 0.7
    ) -> List[Dict]:
        """
        检索top-k最相似的高质量样本（用于few-shot）

        Args:
            problem: 当前问题文本
            problem_type: 问题类型
            k: 返回样本数量
            similarity_threshold: 相似度阈值（0-1），低于则不返回

        Returns:
            最相似的k个样本列表
        """
        if problem_type not in self.buffers:
            return []

        buffer = self.buffers[problem_type]

        if len(buffer) == 0:
            return []

        # 计算相似度
        similarities = []
        for sample in buffer:
            sim = self._compute_similarity(problem, sample['problem'])
            if sim >= similarity_threshold:
                similarities.append((sim, sample))

        # 如果没有相似样本，返回空
        if len(similarities) == 0:
            return []

        # 按相似度降序排序
        similarities.sort(key=lambda x: x[0], reverse=True)

        # 返回top-k
        return [sample for _, sample in similarities[:k]]

    def retrieve_by_reward(
        self,
        problem_type: str,
        k: int = 5
    ) -> List[Dict]:
        """
        直接返回该类型的top-k高奖励样本（不考虑相似度）

        Args:
            problem_type: 问题类型
            k: 返回样本数量

        Returns:
            top-k高奖励样本
        """
        if problem_type not in self.buffers:
            return []

        buffer = self.buffers[problem_type]
        return buffer[:min(k, len(buffer))]

    def get_operator_examples(
        self,
        operator_name: str,
        problem_type: str,
        top_k: int = 2
    ) -> List[Dict]:
        """
        获取使用了特定operator的成功案例

        Args:
            operator_name: operator名称（如"Programmer", "ScEnsemble"等）
            problem_type: 问题类型
            top_k: 返回样本数量

        Returns:
            包含该operator的top-k样本
        """
        if problem_type not in self.buffers:
            return []

        buffer = self.buffers[problem_type]

        # 过滤包含该operator的样本
        operator_samples = []
        for sample in buffer:
            workflow_code = sample.get('workflow_code', '')
            # 检查workflow代码中是否使用了该operator
            if f'operator.{operator_name}' in workflow_code or \
               f'self.{operator_name.lower()}' in workflow_code:
                operator_samples.append(sample)

        # 返回top-k
        return operator_samples[:min(top_k, len(operator_samples))]

    def get_stats(self) -> Dict:
        """
        获取缓冲区统计信息

        Returns:
            统计信息字典
        """
        stats = {}
        for pt in self.problem_types:
            buffer = self.buffers[pt]
            if len(buffer) > 0:
                rewards = [s.get('reward', 0) for s in buffer]
                correctness = [s.get('correctness_score', 0) for s in buffer]
                stats[pt] = {
                    'count': len(buffer),
                    'avg_reward': np.mean(rewards),
                    'max_reward': np.max(rewards),
                    'avg_correctness': np.mean(correctness)
                }
            else:
                stats[pt] = {'count': 0}

        return stats

    def save(self, step: Optional[int] = None):
        """
        持久化缓冲区到磁盘

        Args:
            step: 当前训练步数（可选，用于日志）
        """
        for pt in self.problem_types:
            buffer = self.buffers[pt]
            if len(buffer) == 0:
                continue

            filepath = self.persistence_dir / f"{pt}_top_samples.jsonl"

            with open(filepath, 'w', encoding='utf-8') as f:
                for sample in buffer:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')

            if step is not None:
                print(f"💾 Experience buffer saved: {pt} ({len(buffer)} samples) at step {step}")

    def load(self):
        """
        从磁盘加载缓冲区
        """
        loaded_count = 0

        for pt in self.problem_types:
            filepath = self.persistence_dir / f"{pt}_top_samples.jsonl"

            if not filepath.exists():
                continue

            samples = []
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        sample = json.loads(line)
                        samples.append(sample)

            # 按奖励排序并保留top-k
            samples.sort(key=lambda x: x.get('reward', -10.0), reverse=True)
            self.buffers[pt] = samples[:self.buffer_size]

            loaded_count += len(self.buffers[pt])

        if loaded_count > 0:
            print(f"📥 Loaded {loaded_count} samples from experience buffer")

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的相似度（简单快速的方法）

        Args:
            text1: 文本1
            text2: 文本2

        Returns:
            相似度分数 (0-1)
        """
        # 使用SequenceMatcher（快速且无额外依赖）
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def clear(self, problem_type: Optional[str] = None):
        """
        清空缓冲区

        Args:
            problem_type: 指定清空的类型，None则清空所有
        """
        if problem_type is None:
            for pt in self.problem_types:
                self.buffers[pt] = []
        elif problem_type in self.buffers:
            self.buffers[problem_type] = []
