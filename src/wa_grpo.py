#!/usr/bin/env python3
"""
WA-GRPO: Workflow-Aware Group Relative Policy Optimization

核心创新：利用workflow的结构和过程信号打破组内同分对称性

特征维度：
1. diversity_score - 代码文本/结构多样性
2. process_gain - Revise/Test后的改进幅度
3. exec_success - 执行成功度（分级）
4. efficiency - 运行效率（时间/内存）
5. op_variety - Operator链路覆盖度

作者: Claude + User
日期: 2025-11-25
"""

import numpy as np
import re
import ast
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter
import hashlib


class WAGRPOAdvantageComputer:
    """
    Workflow-Aware GRPO优势计算器

    在组内同分时，使用workflow特征作为tie-breaker打破对称性
    """

    def __init__(
        self,
        alpha: float = 0.12,                    # tie-breaker混合系数
        diversity_weight: float = 0.35,         # 多样性权重
        revise_gain_weight: float = 0.25,       # 改进幅度权重
        exec_success_weight: float = 0.20,      # 执行成功度权重
        efficiency_weight: float = 0.10,        # 效率权重
        op_variety_weight: float = 0.10,        # Operator多样性权重
        min_advantage_std: float = 0.10,        # 最小优势标准差
        same_reward_threshold: float = 1e-6,    # 判定同分的阈值
        use_ast_diversity: bool = True,         # 是否使用AST多样性
        batch_calibration: bool = True,         # 是否进行批内校准
    ):
        self.alpha = alpha
        self.diversity_weight = diversity_weight
        self.revise_gain_weight = revise_gain_weight
        self.exec_success_weight = exec_success_weight
        self.efficiency_weight = efficiency_weight
        self.op_variety_weight = op_variety_weight
        self.min_advantage_std = min_advantage_std
        self.same_reward_threshold = same_reward_threshold
        self.use_ast_diversity = use_ast_diversity
        self.batch_calibration = batch_calibration

        # 已知的Operator集合（用于计算覆盖度）
        self.known_operators = {
            'programmer', 'answer_generate', 'review', 'revise',
            'test', 'sc_ensemble', 'custom'
        }

        # 统计信息
        self.stats = {
            'total_groups': 0,
            'zero_reward_var_groups': 0,
            'alpha_applied_groups': 0,
            'noise_applied_groups': 0,
            'final_zero_adv_groups': 0,
        }

    def reset_stats(self):
        """重置统计信息"""
        for key in self.stats:
            self.stats[key] = 0

    def compute_advantages(
        self,
        rewards: List[float],
        group_size: int,
        workflows: List[str],
        exec_metas: Optional[List[Dict]] = None,
        op_traces: Optional[List[List[str]]] = None,
    ) -> Tuple[List[float], Dict]:
        """
        计算WA-GRPO优势

        Args:
            rewards: 原始奖励 [batch_size * K]
            group_size: K值
            workflows: workflow代码列表
            exec_metas: 执行元信息列表 (可选)
            op_traces: Operator调用链列表 (可选)

        Returns:
            advantages: 计算后的优势值
            info: 诊断信息
        """
        batch_size = len(rewards) // group_size
        all_advantages = []

        self.reset_stats()
        self.stats['total_groups'] = batch_size

        # 诊断信息
        info = {
            'original_zero_var_groups': 0,
            'alpha_applied': 0,
            'noise_applied': 0,
            'final_zero_adv_groups': 0,
            'tie_breaker_stats': {
                'diversity_mean': 0,
                'process_gain_mean': 0,
                'exec_success_mean': 0,
                'efficiency_mean': 0,
                'op_variety_mean': 0,
            },
            'feature_contributions': [],
        }

        # 收集所有组的tie-breaker用于批内校准
        all_tie_breakers = []
        group_data = []

        # 第一遍：计算每组的基础数据和tie-breaker
        for group_idx in range(batch_size):
            start_idx = group_idx * group_size
            end_idx = start_idx + group_size

            group_rewards = np.array(rewards[start_idx:end_idx])
            group_workflows = workflows[start_idx:end_idx]
            group_exec_metas = exec_metas[start_idx:end_idx] if exec_metas else None
            group_op_traces = op_traces[start_idx:end_idx] if op_traces else None

            # 计算tie-breaker分数
            tie_breaker, features = self._compute_tie_breaker(
                group_workflows, group_exec_metas, group_op_traces
            )

            all_tie_breakers.extend(tie_breaker)
            group_data.append({
                'rewards': group_rewards,
                'workflows': group_workflows,
                'tie_breaker': tie_breaker,
                'features': features,
            })

            # 累积特征统计
            for key in features:
                if key in info['tie_breaker_stats']:
                    info['tie_breaker_stats'][key] += np.mean(features[key])

        # 批内校准（可选）
        if self.batch_calibration and len(all_tie_breakers) > 0:
            all_tie_breakers = np.array(all_tie_breakers)
            tb_mean = np.mean(all_tie_breakers)
            tb_std = np.std(all_tie_breakers) + 1e-8
            # Z-score标准化后重新映射到[0,1]
            all_tie_breakers = (all_tie_breakers - tb_mean) / tb_std
            all_tie_breakers = (all_tie_breakers - all_tie_breakers.min()) / \
                              (all_tie_breakers.max() - all_tie_breakers.min() + 1e-8)

            # 更新每组的tie_breaker
            for group_idx in range(batch_size):
                start_idx = group_idx * group_size
                end_idx = start_idx + group_size
                group_data[group_idx]['tie_breaker'] = all_tie_breakers[start_idx:end_idx]

        # 第二遍：计算优势
        for group_idx in range(batch_size):
            data = group_data[group_idx]
            group_rewards = data['rewards'].copy()
            tie_breaker = data['tie_breaker']

            reward_std = np.std(group_rewards)

            # 检测是否需要使用tie-breaker
            if reward_std < self.same_reward_threshold:
                info['original_zero_var_groups'] += 1
                self.stats['zero_reward_var_groups'] += 1

                # 使用tie-breaker微调奖励
                group_rewards = group_rewards + self.alpha * tie_breaker
                info['alpha_applied'] += 1
                self.stats['alpha_applied_groups'] += 1

            # 重新计算标准差
            reward_std = np.std(group_rewards)

            # 最小方差保障（使用顺序对齐噪声）
            if reward_std < self.min_advantage_std:
                # 基于tie-breaker排序添加噪声
                noise = self._generate_aligned_noise(
                    group_rewards, tie_breaker, self.min_advantage_std * 0.5
                )
                group_rewards = group_rewards + noise
                info['noise_applied'] += 1
                self.stats['noise_applied_groups'] += 1

            # 计算归一化优势
            reward_mean = np.mean(group_rewards)
            reward_std = np.std(group_rewards) + 1e-8
            group_advantages = (group_rewards - reward_mean) / reward_std

            # 裁剪优势
            group_advantages = np.clip(group_advantages, -2.0, 2.0)

            # 检查最终是否仍然接近零
            if np.std(group_advantages) < 1e-6:
                info['final_zero_adv_groups'] += 1
                self.stats['final_zero_adv_groups'] += 1

            all_advantages.extend(group_advantages.tolist())

        # 平均特征统计
        for key in info['tie_breaker_stats']:
            info['tie_breaker_stats'][key] /= max(batch_size, 1)

        return all_advantages, info

    def _compute_tie_breaker(
        self,
        workflows: List[str],
        exec_metas: Optional[List[Dict]],
        op_traces: Optional[List[List[str]]],
    ) -> Tuple[np.ndarray, Dict]:
        """
        计算组内每个样本的tie-breaker分数

        Returns:
            tie_breaker: 归一化到[0,1]的分数数组
            features: 各特征的原始值
        """
        n = len(workflows)

        # 1. 多样性分数
        diversity_scores = self._compute_diversity_scores(workflows)

        # 2. 过程改进分数
        if exec_metas:
            process_gains = self._compute_process_gains(exec_metas)
        else:
            process_gains = np.zeros(n)

        # 3. 执行成功度
        if exec_metas:
            exec_success = self._compute_exec_success(exec_metas)
        else:
            exec_success = np.ones(n) * 0.5  # 默认中等

        # 4. 效率分数
        if exec_metas:
            efficiency = self._compute_efficiency(exec_metas)
        else:
            efficiency = np.ones(n) * 0.5  # 默认中等

        # 5. Operator多样性
        if op_traces:
            op_variety = self._compute_op_variety(op_traces)
        else:
            # 从workflow代码中推断
            op_variety = self._infer_op_variety(workflows)

        # 组合分数
        tie_breaker = (
            self.diversity_weight * diversity_scores +
            self.revise_gain_weight * process_gains +
            self.exec_success_weight * exec_success +
            self.efficiency_weight * efficiency +
            self.op_variety_weight * op_variety
        )

        # 组内归一化到[0,1]
        if tie_breaker.max() > tie_breaker.min():
            tie_breaker = (tie_breaker - tie_breaker.min()) / \
                         (tie_breaker.max() - tie_breaker.min())
        else:
            tie_breaker = np.ones(n) * 0.5

        features = {
            'diversity_mean': diversity_scores,
            'process_gain_mean': process_gains,
            'exec_success_mean': exec_success,
            'efficiency_mean': efficiency,
            'op_variety_mean': op_variety,
        }

        return tie_breaker, features

    def _compute_diversity_scores(self, workflows: List[str]) -> np.ndarray:
        """
        计算多样性分数（结合文本和AST）
        """
        n = len(workflows)
        if n <= 1:
            return np.array([0.5] * n)

        diversity_scores = np.zeros(n)

        for i in range(n):
            total_diff = 0
            for j in range(n):
                if i != j:
                    # 文本多样性
                    text_diff = self._text_diversity(workflows[i], workflows[j])

                    # AST多样性（如果启用）
                    if self.use_ast_diversity:
                        ast_diff = self._ast_diversity(workflows[i], workflows[j])
                        diff = 0.6 * text_diff + 0.4 * ast_diff
                    else:
                        diff = text_diff

                    total_diff += diff

            diversity_scores[i] = total_diff / (n - 1)

        # 归一化到[0,1]
        if diversity_scores.max() > diversity_scores.min():
            diversity_scores = (diversity_scores - diversity_scores.min()) / \
                              (diversity_scores.max() - diversity_scores.min())

        return diversity_scores

    def _text_diversity(self, text1: str, text2: str) -> float:
        """
        计算文本多样性（n-gram Jaccard距离）
        """
        def get_ngrams(text, n=3):
            text = text.lower().strip()
            return set(text[i:i+n] for i in range(max(0, len(text) - n + 1)))

        ngrams1 = get_ngrams(text1)
        ngrams2 = get_ngrams(text2)

        if not ngrams1 or not ngrams2:
            return 0.5  # 无法比较时返回中等值

        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)

        jaccard_similarity = intersection / union if union > 0 else 0
        return 1.0 - jaccard_similarity

    def _ast_diversity(self, code1: str, code2: str) -> float:
        """
        计算AST结构多样性
        """
        try:
            features1 = self._extract_ast_features(code1)
            features2 = self._extract_ast_features(code2)

            if not features1 or not features2:
                return 0.5

            # 计算特征向量的余弦距离
            all_keys = set(features1.keys()) | set(features2.keys())
            vec1 = np.array([features1.get(k, 0) for k in all_keys])
            vec2 = np.array([features2.get(k, 0) for k in all_keys])

            # 归一化
            norm1 = np.linalg.norm(vec1) + 1e-8
            norm2 = np.linalg.norm(vec2) + 1e-8

            cosine_sim = np.dot(vec1, vec2) / (norm1 * norm2)
            return 1.0 - cosine_sim

        except Exception:
            return 0.5

    def _extract_ast_features(self, code: str) -> Dict[str, int]:
        """
        提取AST特征
        """
        features = {
            'num_functions': 0,
            'num_classes': 0,
            'num_loops': 0,
            'num_conditions': 0,
            'num_imports': 0,
            'num_calls': 0,
            'num_assigns': 0,
            'max_depth': 0,
        }

        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    features['num_functions'] += 1
                elif isinstance(node, ast.ClassDef):
                    features['num_classes'] += 1
                elif isinstance(node, (ast.For, ast.While)):
                    features['num_loops'] += 1
                elif isinstance(node, ast.If):
                    features['num_conditions'] += 1
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    features['num_imports'] += 1
                elif isinstance(node, ast.Call):
                    features['num_calls'] += 1
                elif isinstance(node, ast.Assign):
                    features['num_assigns'] += 1

            # 计算最大深度
            features['max_depth'] = self._get_ast_depth(tree)

        except SyntaxError:
            pass

        return features

    def _get_ast_depth(self, node, current_depth=0) -> int:
        """
        计算AST最大深度
        """
        max_depth = current_depth
        for child in ast.iter_child_nodes(node):
            child_depth = self._get_ast_depth(child, current_depth + 1)
            max_depth = max(max_depth, child_depth)
        return max_depth

    def _compute_process_gains(self, exec_metas: List[Dict]) -> np.ndarray:
        """
        计算过程改进分数（Revise后的提升）
        """
        n = len(exec_metas)
        gains = np.zeros(n)

        for i, meta in enumerate(exec_metas):
            if meta is None:
                gains[i] = 0.5
                continue

            # 检查是否有revision
            has_revision = meta.get('has_revision', False)
            revision_improved = meta.get('revision_improved', False)
            revision_delta = meta.get('revision_delta', 0.0)

            if has_revision:
                if revision_improved:
                    # 改进了，根据delta给分
                    gains[i] = 0.5 + min(revision_delta * 0.5, 0.5)
                else:
                    # 尝试改进但没成功
                    gains[i] = 0.3
            else:
                # 没有revision（可能一次就对了或者没走revision流程）
                gains[i] = 0.5

        return gains

    def _compute_exec_success(self, exec_metas: List[Dict]) -> np.ndarray:
        """
        计算执行成功度（分级）
        """
        n = len(exec_metas)
        success = np.zeros(n)

        for i, meta in enumerate(exec_metas):
            if meta is None:
                success[i] = 0.5
                continue

            is_success = meta.get('success', False)
            error_type = meta.get('error_type', None)

            if is_success:
                success[i] = 1.0
            elif error_type:
                # 根据错误类型分级
                if error_type in ['timeout', 'TimeoutError']:
                    success[i] = 0.3  # 超时比崩溃好一点
                elif error_type in ['SyntaxError', 'IndentationError']:
                    success[i] = 0.1  # 语法错误最差
                elif error_type in ['RuntimeError', 'ValueError', 'TypeError']:
                    success[i] = 0.2  # 运行时错误
                elif error_type in ['AttributeError', 'KeyError', 'IndexError']:
                    success[i] = 0.25  # 属性/索引错误
                else:
                    success[i] = 0.2  # 其他错误
            else:
                success[i] = 0.0

        return success

    def _compute_efficiency(self, exec_metas: List[Dict]) -> np.ndarray:
        """
        计算效率分数（时间/内存）
        """
        n = len(exec_metas)
        efficiency = np.zeros(n)

        times = []
        for meta in exec_metas:
            if meta and 'execution_time' in meta:
                times.append(meta['execution_time'])
            else:
                times.append(None)

        # 计算有效时间的统计
        valid_times = [t for t in times if t is not None and t > 0]
        if valid_times:
            max_time = max(valid_times)
            min_time = min(valid_times)
            time_range = max_time - min_time + 1e-8
        else:
            max_time = min_time = time_range = 1.0

        for i, time in enumerate(times):
            if time is None:
                efficiency[i] = 0.5
            elif time <= 0:
                efficiency[i] = 1.0  # 极快
            else:
                # 时间越短效率越高
                efficiency[i] = 1.0 - (time - min_time) / time_range

        return efficiency

    def _compute_op_variety(self, op_traces: List[List[str]]) -> np.ndarray:
        """
        计算Operator链路覆盖度
        """
        n = len(op_traces)
        variety = np.zeros(n)
        total_ops = len(self.known_operators)

        for i, trace in enumerate(op_traces):
            if trace:
                unique_ops = set(op.lower() for op in trace)
                coverage = len(unique_ops & self.known_operators) / total_ops
                variety[i] = coverage
            else:
                variety[i] = 0.5

        return variety

    def _infer_op_variety(self, workflows: List[str]) -> np.ndarray:
        """
        从workflow代码中推断Operator使用情况
        """
        n = len(workflows)
        variety = np.zeros(n)

        # Operator关键词模式
        op_patterns = {
            'programmer': r'\bprogrammer\b|\bcode\b|def solve',
            'answer_generate': r'\banswer_generate\b|\banswer\b',
            'review': r'\breview\b|\bcheck\b|\bvalidate\b',
            'revise': r'\brevise\b|\bfix\b|\bcorrect\b',
            'test': r'\btest\b|\bassert\b',
            'sc_ensemble': r'\bensemble\b|\bvote\b|\bconsensus\b',
        }

        total_ops = len(op_patterns)

        for i, workflow in enumerate(workflows):
            workflow_lower = workflow.lower()
            found_ops = 0
            for op_name, pattern in op_patterns.items():
                if re.search(pattern, workflow_lower):
                    found_ops += 1
            variety[i] = found_ops / total_ops

        return variety

    def _generate_aligned_noise(
        self,
        rewards: np.ndarray,
        tie_breaker: np.ndarray,
        noise_scale: float
    ) -> np.ndarray:
        """
        生成与tie-breaker顺序对齐的噪声
        """
        n = len(rewards)

        # 生成随机噪声
        noise = np.random.randn(n) * noise_scale

        # 按tie-breaker排序
        sorted_indices = np.argsort(tie_breaker)
        sorted_noise = np.sort(noise)

        # 对齐噪声：tie-breaker高的获得更高的噪声
        aligned_noise = np.zeros(n)
        for rank, idx in enumerate(sorted_indices):
            aligned_noise[idx] = sorted_noise[rank]

        return aligned_noise


# === 测试代码 ===
if __name__ == "__main__":
    print("="*60)
    print("WA-GRPO 算法测试")
    print("="*60)

    wa_grpo = WAGRPOAdvantageComputer(
        alpha=0.12,
        diversity_weight=0.35,
        revise_gain_weight=0.25,
        exec_success_weight=0.20,
        efficiency_weight=0.10,
        op_variety_weight=0.10,
    )

    # 测试场景1：全零优势组（最难的情况）
    print("\n" + "="*60)
    print("测试1：全零优势组（所有样本奖励相同）")
    print("="*60)

    # 模拟3个组，每组4个样本，奖励完全相同
    rewards = [
        1.0, 1.0, 1.0, 1.0,  # 组1: 全对
        0.0, 0.0, 0.0, 0.0,  # 组2: 全错
        0.7, 0.7, 0.7, 0.7,  # 组3: 部分正确
    ]

    # 模拟不同的workflow（即使答案相同，实现方式不同）
    workflows = [
        # 组1：全对但实现不同
        "def solve(): return x + y  # simple",
        "def solve(): return sum([x, y])  # using sum",
        "from functools import reduce; def solve(): return reduce(lambda a,b: a+b, [x,y])",
        "def solve(): result = x; result += y; return result  # verbose",
        # 组2：全错但实现不同
        "def solve(): return x - y  # wrong operation",
        "def solve(): return x * y  # multiplication instead",
        "def solve(): return 0  # always zero",
        "def solve(): return x - y  # same as first",
        # 组3：部分正确
        "def solve():\n    if x > 0:\n        return x + y\n    return 0",
        "def solve():\n    try:\n        return x + y\n    except: return None",
        "def solve(): return int(x) + int(y)  # type conversion",
        "def solve():\n    for i in range(1):\n        return x + y",
    ]

    # 模拟执行元信息
    exec_metas = [
        {'success': True, 'execution_time': 0.01, 'has_revision': False},
        {'success': True, 'execution_time': 0.02, 'has_revision': True, 'revision_improved': True, 'revision_delta': 0.3},
        {'success': True, 'execution_time': 0.05, 'has_revision': False},
        {'success': True, 'execution_time': 0.03, 'has_revision': True, 'revision_improved': False},
        {'success': False, 'error_type': 'ValueError', 'execution_time': 0.01},
        {'success': False, 'error_type': 'TypeError', 'execution_time': 0.02},
        {'success': False, 'error_type': 'SyntaxError', 'execution_time': 0.0},
        {'success': False, 'error_type': 'RuntimeError', 'execution_time': 0.01},
        {'success': True, 'execution_time': 0.02, 'has_revision': True, 'revision_improved': True, 'revision_delta': 0.2},
        {'success': True, 'execution_time': 0.03, 'has_revision': False},
        {'success': True, 'execution_time': 0.01, 'has_revision': False},
        {'success': True, 'execution_time': 0.04, 'has_revision': True, 'revision_improved': True, 'revision_delta': 0.1},
    ]

    advantages, info = wa_grpo.compute_advantages(
        rewards=rewards,
        group_size=4,
        workflows=workflows,
        exec_metas=exec_metas,
    )

    print(f"\n原始奖励: {rewards}")
    print(f"计算优势: {[f'{a:.3f}' for a in advantages]}")
    print(f"\n诊断信息:")
    print(f"  原始零方差组: {info['original_zero_var_groups']}/3")
    print(f"  Alpha应用次数: {info['alpha_applied']}")
    print(f"  噪声应用次数: {info['noise_applied']}")
    print(f"  最终零优势组: {info['final_zero_adv_groups']}/3")
    print(f"\n特征统计:")
    for key, value in info['tie_breaker_stats'].items():
        print(f"  {key}: {value:.3f}")

    # 验证：检查是否有效打破了对称性
    print("\n" + "="*60)
    print("验证：检查每组优势的方差")
    print("="*60)
    for g in range(3):
        group_adv = advantages[g*4:(g+1)*4]
        print(f"组{g+1}: 优势={[f'{a:.3f}' for a in group_adv]}, 方差={np.var(group_adv):.6f}")

    # 测试场景2：混合场景
    print("\n" + "="*60)
    print("测试2：混合场景（部分组有差异，部分组无差异）")
    print("="*60)

    rewards_mixed = [
        1.0, 0.7, 0.4, 0.0,  # 组1: 有差异
        0.5, 0.5, 0.5, 0.5,  # 组2: 无差异
        0.8, 0.8, 0.2, 0.2,  # 组3: 部分差异
    ]

    workflows_mixed = workflows  # 复用之前的workflow

    advantages_mixed, info_mixed = wa_grpo.compute_advantages(
        rewards=rewards_mixed,
        group_size=4,
        workflows=workflows_mixed,
        exec_metas=exec_metas,
    )

    print(f"\n原始奖励: {rewards_mixed}")
    print(f"计算优势: {[f'{a:.3f}' for a in advantages_mixed]}")
    print(f"\n诊断信息:")
    print(f"  原始零方差组: {info_mixed['original_zero_var_groups']}/3")
    print(f"  Alpha应用次数: {info_mixed['alpha_applied']}")
    print(f"  最终零优势组: {info_mixed['final_zero_adv_groups']}/3")

    # 对比标准GRPO
    print("\n" + "="*60)
    print("对比：标准GRPO vs WA-GRPO")
    print("="*60)

    def standard_grpo_advantages(rewards, group_size):
        """标准GRPO优势计算（无处理）"""
        batch_size = len(rewards) // group_size
        advantages = []
        zero_groups = 0

        for g in range(batch_size):
            group_r = np.array(rewards[g*group_size:(g+1)*group_size])
            mean_r = np.mean(group_r)
            std_r = np.std(group_r)

            if std_r < 1e-6:
                zero_groups += 1
                group_adv = np.zeros(group_size)
            else:
                group_adv = (group_r - mean_r) / std_r

            advantages.extend(group_adv.tolist())

        return advantages, zero_groups

    # 全同分场景对比
    std_adv, std_zero = standard_grpo_advantages(rewards, 4)
    print(f"\n全同分场景:")
    print(f"  标准GRPO - 零优势组: {std_zero}/3, 优势方差: {np.var(std_adv):.6f}")
    print(f"  WA-GRPO  - 零优势组: {info['final_zero_adv_groups']}/3, 优势方差: {np.var(advantages):.6f}")

    # 混合场景对比
    std_adv_mixed, std_zero_mixed = standard_grpo_advantages(rewards_mixed, 4)
    print(f"\n混合场景:")
    print(f"  标准GRPO - 零优势组: {std_zero_mixed}/3")
    print(f"  WA-GRPO  - 零优势组: {info_mixed['final_zero_adv_groups']}/3")

    print("\n" + "="*60)
    print("✅ WA-GRPO测试完成")
    print("="*60)
