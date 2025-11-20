#!/usr/bin/env python3
"""
Unified Evaluation Functions for All Datasets
Includes evaluation metrics for GSM8K, HumanEval, MBPP, CommonsenseQA, HotpotQA, MMLU
"""

import re
import json
import math
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
import subprocess
import sys
import tempfile
import multiprocessing
from functools import partial
import traceback


class UnifiedEvaluator:
    """Unified evaluator for all dataset types"""

    def __init__(self):
        self.evaluators = {
            "math": self.evaluate_math,
            "code": self.evaluate_code,
            "qa": self.evaluate_qa,
            "multi_hop": self.evaluate_multi_hop,
            "multiple_choice": self.evaluate_multiple_choice
        }

    # ============== MATH EVALUATION (GSM8K) ==============

    def extract_math_answer(self, text: str) -> Optional[float]:
        """Extract numeric answer from math solution"""
        # Look for #### pattern (GSM8K style)
        if "####" in text:
            answer_str = text.split("####")[-1].strip()
            match = re.search(r'-?[\d,]+\.?\d*', answer_str)
            if match:
                return float(match.group().replace(',', ''))

        # Look for boxed answer (\boxed{answer})
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
        if boxed_match:
            answer_str = boxed_match.group(1)
            match = re.search(r'-?[\d,]+\.?\d*', answer_str)
            if match:
                return float(match.group().replace(',', ''))

        # Look for final number in the text
        numbers = re.findall(r'-?[\d,]+\.?\d*', text)
        if numbers:
            try:
                return float(numbers[-1].replace(',', ''))
            except:
                pass

        return None

    def evaluate_math(self, prediction: str, ground_truth: str, **kwargs) -> Dict[str, Any]:
        """Evaluate math answer"""
        pred_answer = self.extract_math_answer(prediction)
        gt_answer = self.extract_math_answer(ground_truth)

        if pred_answer is None or gt_answer is None:
            return {"correct": False, "score": 0.0, "pred": pred_answer, "gt": gt_answer}

        # Check with tolerance
        correct = abs(pred_answer - gt_answer) < 1e-4

        return {
            "correct": correct,
            "score": 1.0 if correct else 0.0,
            "pred": pred_answer,
            "gt": gt_answer
        }

    # ============== CODE EVALUATION (HumanEval/MBPP) ==============

    def execute_code_with_tests(self, code: str, test: str, timeout: float = 5.0) -> Tuple[bool, str]:
        """Execute code with test cases"""
        def unsafe_execute():
            try:
                # Create temporary directory for execution
                with tempfile.TemporaryDirectory() as tmpdir:
                    # Combine code and tests
                    full_program = code + "\n" + test

                    # Execute
                    exec_globals = {}
                    exec(full_program, exec_globals)
                    return True, "passed"
            except AssertionError as e:
                return False, f"assertion failed: {e}"
            except Exception as e:
                return False, f"error: {e}"

        # Run in separate process for safety
        try:
            manager = multiprocessing.Manager()
            result = manager.list()

            p = multiprocessing.Process(target=lambda: result.extend(unsafe_execute()))
            p.start()
            p.join(timeout=timeout)

            if p.is_alive():
                p.kill()
                return False, "timeout"

            if result:
                return result[0], result[1]
            else:
                return False, "execution failed"
        except Exception as e:
            return False, f"process error: {e}"

    def evaluate_code(self, prediction: str, ground_truth: str, test: str = None, **kwargs) -> Dict[str, Any]:
        """Evaluate code generation"""
        if test is None:
            # Simple string match as fallback
            return {
                "correct": prediction.strip() == ground_truth.strip(),
                "score": 1.0 if prediction.strip() == ground_truth.strip() else 0.0,
                "method": "string_match"
            }

        # Execute with tests
        passed, message = self.execute_code_with_tests(prediction, test)

        return {
            "correct": passed,
            "score": 1.0 if passed else 0.0,
            "message": message,
            "method": "execution"
        }

    def calculate_pass_at_k(self, results: List[bool], k: int) -> float:
        """Calculate pass@k metric for code evaluation"""
        n = len(results)
        c = sum(results)

        if n < k:
            return c / n

        # Unbiased estimator
        if c < k:
            return 0.0

        return 1.0 - math.comb(n - c, k) / math.comb(n, k)

    # ============== QA EVALUATION (CommonsenseQA/General QA) ==============

    def evaluate_qa(self, prediction: str, ground_truth: str, choices: Dict = None, **kwargs) -> Dict[str, Any]:
        """Evaluate QA answer"""
        # Multiple choice extraction
        if choices:
            pred_label = self.extract_choice_label(prediction, choices)
            correct = pred_label == ground_truth
            return {
                "correct": correct,
                "score": 1.0 if correct else 0.0,
                "pred": pred_label,
                "gt": ground_truth,
                "method": "multiple_choice"
            }

        # Token-level F1 score for open-ended QA
        pred_tokens = prediction.lower().split()
        gt_tokens = ground_truth.lower().split()

        if not pred_tokens or not gt_tokens:
            return {
                "correct": False,
                "score": 0.0,
                "exact_match": False,
                "f1": 0.0
            }

        # Exact match
        exact_match = prediction.strip().lower() == ground_truth.strip().lower()

        # F1 score
        common = set(pred_tokens) & set(gt_tokens)
        num_common = len(common)

        if num_common == 0:
            f1 = 0.0
        else:
            precision = num_common / len(pred_tokens)
            recall = num_common / len(gt_tokens)
            f1 = 2 * (precision * recall) / (precision + recall)

        return {
            "correct": exact_match,
            "score": f1,
            "exact_match": exact_match,
            "f1": f1,
            "method": "token_f1"
        }

    def extract_choice_label(self, text: str, choices: Dict) -> Optional[str]:
        """Extract multiple choice answer label"""
        text = text.upper()

        # Direct label matching
        for label in choices.get('label', []):
            if f"{label}." in text or f"{label})" in text or f"({label})" in text:
                return label

        # Check if answer text is mentioned
        text_lower = text.lower()
        for i, choice_text in enumerate(choices.get('text', [])):
            if choice_text.lower() in text_lower:
                return choices['label'][i]

        # Check first character
        first_char = text.strip()[0] if text.strip() else ''
        if first_char in choices.get('label', []):
            return first_char

        return None

    # ============== MULTI-HOP EVALUATION (HotpotQA) ==============

    def evaluate_multi_hop(self, prediction: str, ground_truth: str,
                          supporting_facts: List = None, pred_facts: List = None,
                          **kwargs) -> Dict[str, Any]:
        """Evaluate multi-hop QA with supporting facts"""
        # Answer evaluation
        answer_result = self.evaluate_qa(prediction, ground_truth)

        if supporting_facts is None or pred_facts is None:
            return answer_result

        # Supporting facts evaluation
        pred_set = set(map(tuple, pred_facts)) if pred_facts else set()
        gt_set = set(map(tuple, supporting_facts)) if supporting_facts else set()

        if not gt_set:
            sp_score = 0.0
        else:
            common = pred_set & gt_set
            precision = len(common) / len(pred_set) if pred_set else 0.0
            recall = len(common) / len(gt_set)
            sp_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Joint score
        joint_score = answer_result['score'] * sp_score

        return {
            "correct": answer_result['correct'],
            "answer_score": answer_result['score'],
            "sp_score": sp_score,
            "joint_score": joint_score,
            "method": "multi_hop"
        }

    # ============== MULTIPLE CHOICE EVALUATION (MMLU) ==============

    def evaluate_multiple_choice(self, prediction: str, ground_truth: str,
                                subject: str = None, **kwargs) -> Dict[str, Any]:
        """Evaluate multiple choice answer (MMLU style)"""
        # Extract answer from prediction
        pred_answer = None
        prediction = prediction.strip().upper()

        # Look for answer patterns
        patterns = [
            r'(?:^|\s)([A-D])(?:\.|\)|\s|$)',  # Standalone letter
            r'answer is ([A-D])',                # "answer is X"
            r'choose ([A-D])',                   # "choose X"
            r'select ([A-D])',                   # "select X"
            r'^([A-D])$'                        # Just the letter
        ]

        for pattern in patterns:
            match = re.search(pattern, prediction, re.IGNORECASE)
            if match:
                pred_answer = match.group(1).upper()
                break

        # If no pattern matched, check first character
        if pred_answer is None and prediction:
            first_char = prediction[0]
            if first_char in 'ABCD':
                pred_answer = first_char

        correct = pred_answer == ground_truth.upper()

        result = {
            "correct": correct,
            "score": 1.0 if correct else 0.0,
            "pred": pred_answer,
            "gt": ground_truth,
            "method": "multiple_choice"
        }

        if subject:
            result["subject"] = subject

        return result

    # ============== UNIFIED EVALUATION ==============

    def evaluate(self, prediction: str, ground_truth: str,
                problem_type: str, **kwargs) -> Dict[str, Any]:
        """Unified evaluation interface"""
        if problem_type not in self.evaluators:
            raise ValueError(f"Unknown problem type: {problem_type}")

        return self.evaluators[problem_type](prediction, ground_truth, **kwargs)

    def evaluate_batch(self, predictions: List[str], ground_truths: List[str],
                      problem_types: List[str], **kwargs) -> Dict[str, Any]:
        """Evaluate a batch of predictions"""
        results = []
        scores_by_type = defaultdict(list)

        for pred, gt, ptype in zip(predictions, ground_truths, problem_types):
            result = self.evaluate(pred, gt, ptype, **kwargs)
            results.append(result)
            scores_by_type[ptype].append(result['score'])

        # Calculate aggregate metrics
        overall_accuracy = np.mean([r['correct'] for r in results])
        overall_score = np.mean([r['score'] for r in results])

        # Calculate per-type metrics
        type_metrics = {}
        for ptype, scores in scores_by_type.items():
            type_metrics[ptype] = {
                "accuracy": np.mean([s == 1.0 for s in scores]),
                "avg_score": np.mean(scores),
                "count": len(scores)
            }

        return {
            "overall_accuracy": overall_accuracy,
            "overall_score": overall_score,
            "type_metrics": type_metrics,
            "detailed_results": results,
            "total": len(results)
        }


class DatasetSpecificEvaluator:
    """Dataset-specific evaluation wrapper"""

    def __init__(self):
        self.evaluator = UnifiedEvaluator()

    def evaluate_gsm8k(self, predictions: List[str], dataset_path: str) -> Dict[str, Any]:
        """Evaluate on GSM8K dataset"""
        with open(dataset_path, 'r') as f:
            data = [json.loads(line) for line in f]

        ground_truths = [self._extract_gsm8k_answer(item['answer']) for item in data]
        problem_types = ['math'] * len(data)

        return self.evaluator.evaluate_batch(predictions, ground_truths, problem_types)

    def evaluate_humaneval(self, predictions: List[str], dataset_path: str,
                          k_values: List[int] = [1, 10, 100]) -> Dict[str, Any]:
        """Evaluate on HumanEval dataset"""
        with open(dataset_path, 'r') as f:
            data = [json.loads(line) for line in f]

        results = []
        for pred, item in zip(predictions, data):
            test = item.get('test', '')
            result = self.evaluator.evaluate_code(pred, item.get('canonical_solution', ''), test)
            results.append(result)

        # Calculate pass@k metrics
        pass_at_k = {}
        for k in k_values:
            if len(results) >= k:
                pass_at_k[f'pass@{k}'] = self.evaluator.calculate_pass_at_k(
                    [r['correct'] for r in results], k
                )

        return {
            "results": results,
            "pass_at_k": pass_at_k,
            "accuracy": np.mean([r['correct'] for r in results])
        }

    def evaluate_commonsenseqa(self, predictions: List[str], dataset_path: str) -> Dict[str, Any]:
        """Evaluate on CommonsenseQA dataset"""
        with open(dataset_path, 'r') as f:
            data = [json.loads(line) for line in f]

        results = []
        for pred, item in zip(predictions, data):
            result = self.evaluator.evaluate_qa(
                pred, item['answerKey'],
                choices=item.get('choices')
            )
            results.append(result)

        return {
            "results": results,
            "accuracy": np.mean([r['correct'] for r in results]),
            "avg_score": np.mean([r['score'] for r in results])
        }

    def evaluate_mmlu(self, predictions: List[str], dataset_path: str) -> Dict[str, Any]:
        """Evaluate on MMLU dataset"""
        with open(dataset_path, 'r') as f:
            data = [json.loads(line) for line in f]

        results = []
        subject_scores = defaultdict(list)

        for pred, item in zip(predictions, data):
            subject = item.get('subject', 'unknown')
            result = self.evaluator.evaluate_multiple_choice(
                pred, item['answer'], subject=subject
            )
            results.append(result)
            subject_scores[subject].append(result['correct'])

        # Calculate per-subject accuracy
        subject_accuracies = {
            subj: np.mean(scores) for subj, scores in subject_scores.items()
        }

        return {
            "results": results,
            "overall_accuracy": np.mean([r['correct'] for r in results]),
            "subject_accuracies": subject_accuracies,
            "num_subjects": len(subject_accuracies)
        }

    def _extract_gsm8k_answer(self, answer_text: str) -> str:
        """Extract answer from GSM8K format"""
        if "####" in answer_text:
            return answer_text.split("####")[-1].strip()
        return answer_text


if __name__ == "__main__":
    # Example usage
    evaluator = UnifiedEvaluator()

    # Test math evaluation
    math_result = evaluator.evaluate(
        "The answer is 42.",
        "#### 42",
        "math"
    )
    print(f"Math evaluation: {math_result}")

    # Test QA evaluation
    qa_result = evaluator.evaluate(
        "The capital is Paris.",
        "Paris",
        "qa"
    )
    print(f"QA evaluation: {qa_result}")

    # Test multiple choice
    mc_result = evaluator.evaluate(
        "The answer is B.",
        "B",
        "multiple_choice"
    )
    print(f"Multiple choice evaluation: {mc_result}")

    print("\n✅ Evaluation functions ready!")
