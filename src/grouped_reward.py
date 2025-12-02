#!/usr/bin/env python3
"""
分组奖励计算器 - 支持多问题加权评分 + 多样性打破平局

设计:
1. 每个 workflow 在一组问题 (2 easy + 2 hard) 上运行
2. 计算加权得分: score = Σ(weight_i * correctness_i)
3. 如果组内 K 个 workflow 分数差距 < 阈值，用多样性打破平局

公式:
- easy_weight = 0.3 (每题 0.15)
- hard_weight = 0.7 (每题 0.35)
- final_score = weighted_score + diversity_bonus (当差距 < threshold)
"""

import re
import ast
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
import math


class GroupedRewardCalculator:
    """
    分组奖励计算器

    特性:
    1. 多问题加权评分
    2. 多样性 tie-breaker
    3. 保证组内有非零优势
    """

    def __init__(
        self,
        weight_easy: float = 0.3,
        weight_hard: float = 0.7,
        diversity_threshold: float = 0.05,  # 分数差距阈值
        diversity_weight: float = 0.1,       # 多样性加分权重
        debug: bool = False
    ):
        self.weight_easy = weight_easy
        self.weight_hard = weight_hard
        self.diversity_threshold = diversity_threshold
        self.diversity_weight = diversity_weight
        self.debug = debug

        # 已知的 operator 列表
        self.known_operators = {
            'AnswerGenerate', 'Programmer', 'ScEnsemble',
            'Test', 'Review', 'Revise', 'Custom'
        }

    def calculate_weighted_score(
        self,
        problem_scores: List[Dict[str, Any]]
    ) -> float:
        """
        计算加权得分

        Args:
            problem_scores: 每个问题的评分结果
                [{
                    'difficulty': 'easy'/'hard',
                    'weight': 0.15/0.35,
                    'correctness': 0.0-1.0,
                    'problem_id': 'easy_0'
                }, ...]

        Returns:
            加权总分 (0.0 - 1.0)
        """
        total_score = 0.0
        for p in problem_scores:
            total_score += p['weight'] * p['correctness']
        return total_score

    def calculate_diversity_score(self, workflow_code: str) -> float:
        """
        计算 workflow 的多样性得分

        考虑因素:
        1. 使用的 operator 数量和种类
        2. 控制流复杂度 (if/for/while)
        3. 代码结构多样性

        Returns:
            多样性得分 (0.0 - 1.0)
        """
        if not workflow_code:
            return 0.0

        scores = []

        # 1. Operator 多样性 (0-0.4)
        operators_used = set()
        for op in self.known_operators:
            pattern = rf'\b{op}\b'
            if re.search(pattern, workflow_code):
                operators_used.add(op)

        op_diversity = min(len(operators_used) / 4.0, 1.0) * 0.4
        scores.append(op_diversity)

        # 2. 控制流复杂度 (0-0.3)
        control_patterns = [
            r'\bif\b', r'\bfor\b', r'\bwhile\b',
            r'\btry\b', r'\bawait\b'
        ]
        control_count = sum(1 for p in control_patterns if re.search(p, workflow_code))
        control_score = min(control_count / 4.0, 1.0) * 0.3
        scores.append(control_score)

        # 3. 步骤数量 (0-0.3)
        # 计算 await 调用次数作为步骤数
        await_count = len(re.findall(r'await\s+self\.\w+', workflow_code))
        step_score = min(await_count / 5.0, 1.0) * 0.3
        scores.append(step_score)

        return sum(scores)

    def extract_operators(self, workflow_code: str) -> List[str]:
        """提取 workflow 使用的 operators"""
        operators = []
        for op in self.known_operators:
            if re.search(rf'\b{op}\b', workflow_code):
                operators.append(op)
        return operators

    def calculate_group_rewards(
        self,
        workflows: List[str],
        problem_scores_per_workflow: List[List[Dict[str, Any]]]
    ) -> Tuple[List[float], Dict[str, Any]]:
        """
        计算一组 workflow 的奖励

        Args:
            workflows: K 个 workflow 代码
            problem_scores_per_workflow: 每个 workflow 在每个问题上的得分
                [[{problem_0_score}, {problem_1_score}, ...], ...]

        Returns:
            (rewards, diagnostics)
            - rewards: K 个 workflow 的最终奖励
            - diagnostics: 调试信息
        """
        K = len(workflows)
        if K == 0:
            return [], {}

        # 1. 计算每个 workflow 的加权得分
        weighted_scores = []
        for scores in problem_scores_per_workflow:
            ws = self.calculate_weighted_score(scores)
            weighted_scores.append(ws)

        # 2. 检查是否需要多样性打破平局
        score_range = max(weighted_scores) - min(weighted_scores)
        need_diversity_tiebreak = score_range < self.diversity_threshold

        # 3. 计算多样性得分
        diversity_scores = [self.calculate_diversity_score(w) for w in workflows]

        # 4. 计算最终奖励
        final_rewards = []
        for i in range(K):
            reward = weighted_scores[i]
            if need_diversity_tiebreak:
                # 加入多样性加分
                reward += self.diversity_weight * diversity_scores[i]
            final_rewards.append(reward)

        # 5. 诊断信息
        diagnostics = {
            'weighted_scores': weighted_scores,
            'diversity_scores': diversity_scores,
            'score_range': score_range,
            'need_diversity_tiebreak': need_diversity_tiebreak,
            'final_rewards': final_rewards,
            'operators_per_workflow': [self.extract_operators(w) for w in workflows]
        }

        if self.debug:
            print(f"\n🎯 GroupedReward 诊断:")
            print(f"  加权分数: {[f'{s:.3f}' for s in weighted_scores]}")
            print(f"  多样性分: {[f'{s:.3f}' for s in diversity_scores]}")
            print(f"  分数差距: {score_range:.3f} (阈值: {self.diversity_threshold})")
            print(f"  需要多样性打破平局: {need_diversity_tiebreak}")
            print(f"  最终奖励: {[f'{r:.3f}' for r in final_rewards]}")

        return final_rewards, diagnostics

    def compute_advantages(
        self,
        rewards: List[float],
        min_std: float = 0.01
    ) -> List[float]:
        """
        计算 GRPO 优势值

        Args:
            rewards: K 个 workflow 的奖励
            min_std: 最小标准差（防止除零）

        Returns:
            K 个优势值
        """
        if len(rewards) == 0:
            return []

        mean_reward = sum(rewards) / len(rewards)
        variance = sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)
        std = max(math.sqrt(variance), min_std)

        advantages = [(r - mean_reward) / std for r in rewards]
        return advantages


class GroupedBatchProcessor:
    """
    分组批处理器 - 处理一个 batch 的问题组
    """

    def __init__(
        self,
        reward_calculator: GroupedRewardCalculator,
        base_reward_computer: Any  # 原始的 RewardComputer
    ):
        self.reward_calculator = reward_calculator
        self.base_reward_computer = base_reward_computer

    async def process_group(
        self,
        group: Dict[str, Any],
        workflows: List[str],
        executor: Any  # AFlowExecutor
    ) -> Tuple[List[float], Dict[str, Any]]:
        """
        处理一个问题组

        Args:
            group: 问题组数据
                {
                    'group_id': 'math_001',
                    'domain': 'math',
                    'problems': [{...}, {...}, {...}, {...}]
                }
            workflows: K 个 workflow 代码
            executor: AFlow 执行器

        Returns:
            (rewards, diagnostics)
        """
        problems = group['problems']
        K = len(workflows)

        # 每个 workflow 在每个问题上的得分
        problem_scores_per_workflow = [[] for _ in range(K)]

        # 遍历每个问题
        for problem in problems:
            # 遍历每个 workflow
            for i, workflow_code in enumerate(workflows):
                # 执行 workflow
                result = await executor.execute(
                    workflow_code=workflow_code,
                    problem=problem['question'],
                    ground_truth=problem['answer'],
                    domain=problem['domain'],
                    entry_point=problem.get('entry_point', ''),
                    test_cases=problem.get('test_cases', [])
                )

                # 计算正确性得分
                correctness = result.get('correctness_score', 0.0)

                problem_scores_per_workflow[i].append({
                    'problem_id': problem['id'],
                    'difficulty': problem['difficulty'],
                    'weight': problem['weight'],
                    'correctness': correctness,
                    'execution_time': result.get('execution_time', 0),
                    'success': result.get('success', False)
                })

        # 计算最终奖励
        rewards, diagnostics = self.reward_calculator.calculate_group_rewards(
            workflows=workflows,
            problem_scores_per_workflow=problem_scores_per_workflow
        )

        diagnostics['group_id'] = group['group_id']
        diagnostics['domain'] = group['domain']
        diagnostics['problem_scores'] = problem_scores_per_workflow

        return rewards, diagnostics


# 测试代码
if __name__ == "__main__":
    calc = GroupedRewardCalculator(debug=True)

    # 模拟两个 workflow
    workflows = [
        """class Workflow:
            def __init__(self):
                self.answer_generate = AnswerGenerate()
                self.review = Review()

            async def __call__(self, problem):
                ans = await self.answer_generate(problem)
                if ans:
                    review = await self.review(ans)
                return ans
        """,
        """class Workflow:
            def __init__(self):
                self.answer_generate = AnswerGenerate()
                self.programmer = Programmer()
                self.review = Review()
                self.revise = Revise()

            async def __call__(self, problem):
                ans = await self.answer_generate(problem)
                if not ans:
                    code = await self.programmer(problem)
                    ans = code
                review = await self.review(ans)
                if review.needs_revision:
                    ans = await self.revise(ans, review)
                return ans
        """
    ]

    # 模拟问题得分
    scores_w1 = [
        {'difficulty': 'easy', 'weight': 0.15, 'correctness': 1.0, 'problem_id': 'easy_0'},
        {'difficulty': 'easy', 'weight': 0.15, 'correctness': 1.0, 'problem_id': 'easy_1'},
        {'difficulty': 'hard', 'weight': 0.35, 'correctness': 0.4, 'problem_id': 'hard_0'},
        {'difficulty': 'hard', 'weight': 0.35, 'correctness': 0.0, 'problem_id': 'hard_1'},
    ]

    scores_w2 = [
        {'difficulty': 'easy', 'weight': 0.15, 'correctness': 1.0, 'problem_id': 'easy_0'},
        {'difficulty': 'easy', 'weight': 0.15, 'correctness': 0.7, 'problem_id': 'easy_1'},
        {'difficulty': 'hard', 'weight': 0.35, 'correctness': 0.7, 'problem_id': 'hard_0'},
        {'difficulty': 'hard', 'weight': 0.35, 'correctness': 0.4, 'problem_id': 'hard_1'},
    ]

    print("\n" + "="*60)
    print("测试 GroupedRewardCalculator")
    print("="*60)

    rewards, diag = calc.calculate_group_rewards(
        workflows=workflows,
        problem_scores_per_workflow=[scores_w1, scores_w2]
    )

    print(f"\n最终奖励: {rewards}")

    # 计算优势
    advantages = calc.compute_advantages(rewards)
    print(f"优势值: {advantages}")

    print("\n" + "="*60)
    print("测试平局情况（需要多样性打破平局）")
    print("="*60)

    # 两个 workflow 得分完全相同
    scores_tie = [
        {'difficulty': 'easy', 'weight': 0.15, 'correctness': 1.0, 'problem_id': 'easy_0'},
        {'difficulty': 'easy', 'weight': 0.15, 'correctness': 1.0, 'problem_id': 'easy_1'},
        {'difficulty': 'hard', 'weight': 0.35, 'correctness': 0.5, 'problem_id': 'hard_0'},
        {'difficulty': 'hard', 'weight': 0.35, 'correctness': 0.5, 'problem_id': 'hard_1'},
    ]

    rewards_tie, diag_tie = calc.calculate_group_rewards(
        workflows=workflows,
        problem_scores_per_workflow=[scores_tie, scores_tie]
    )

    print(f"\n最终奖励（有多样性加分）: {rewards_tie}")
    advantages_tie = calc.compute_advantages(rewards_tie)
    print(f"优势值（非零）: {advantages_tie}")
