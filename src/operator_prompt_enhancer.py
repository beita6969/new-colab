#!/usr/bin/env python3
"""
Operator提示词增强器 - Layer 2: Operator执行提示词直接优化
针对gpt-oss-120b本地模型的提示词优化（无few-shot依赖）
"""
from typing import Dict, List


class OperatorPromptEnhancer:
    """
    Operator级提示词增强器（简化版）

    功能：
    1. 为gpt-oss-120b优化operator调用参数
    2. 增强instruction/prompt的清晰度
    3. 提供问题特定的指导
    4. 提高本地模型的执行质量
    """

    def __init__(self, enable_enhancement: bool = True):
        """
        Args:
            enable_enhancement: 是否启用增强
        """
        self.enable_enhancement = enable_enhancement
        self.enhancement_templates = self._load_enhancement_templates()

    def enhance_operator_call(
        self,
        operator_name: str,
        original_kwargs: Dict,
        problem_type: str,
        current_problem: str = None
    ) -> Dict:
        """
        增强operator调用的参数（直接优化instruction/prompt）

        Args:
            operator_name: operator名称（如"Custom", "Programmer"等）
            original_kwargs: 原始调用参数
            problem_type: 问题类型
            current_problem: 当前问题文本（可选）

        Returns:
            增强后的参数字典
        """
        if not self.enable_enhancement:
            return original_kwargs

        # 复制参数避免修改原始
        enhanced_kwargs = original_kwargs.copy()

        # 根据operator类型增强不同参数
        if operator_name == "Custom":
            enhanced_kwargs = self._enhance_custom(enhanced_kwargs, problem_type)
        elif operator_name == "AnswerGenerate":
            enhanced_kwargs = self._enhance_answer_generate(enhanced_kwargs, problem_type)
        elif operator_name == "Programmer":
            enhanced_kwargs = self._enhance_programmer(enhanced_kwargs, problem_type)
        elif operator_name == "Test":
            enhanced_kwargs = self._enhance_test(enhanced_kwargs, problem_type)
        elif operator_name == "Review":
            enhanced_kwargs = self._enhance_review(enhanced_kwargs, problem_type)
        elif operator_name == "Revise":
            enhanced_kwargs = self._enhance_revise(enhanced_kwargs, problem_type)
        # ScEnsemble不需要增强

        return enhanced_kwargs

    def _enhance_custom(
        self,
        kwargs: Dict,
        problem_type: str
    ) -> Dict:
        """
        增强Custom operator的instruction参数

        Custom API: custom(input=str, instruction=str)
        """
        if 'instruction' not in kwargs:
            return kwargs

        original_instruction = kwargs['instruction']

        # 为gpt-oss-120b优化instruction
        enhancement = self._build_direct_enhancement(
            operator_name="Custom",
            problem_type=problem_type,
            instruction=original_instruction
        )

        if enhancement:
            kwargs['instruction'] = f"{enhancement}\n\n{original_instruction}"

        return kwargs

    def _enhance_answer_generate(
        self,
        kwargs: Dict,
        problem_type: str
    ) -> Dict:
        """
        增强AnswerGenerate operator

        AnswerGenerate API: answer_generate(input=str)
        """
        if 'input' not in kwargs:
            return kwargs

        original_input = kwargs['input']

        # 为gpt-oss-120b添加推理指导
        hint = "[For gpt-oss-120b: Provide step-by-step reasoning]\n\n"
        kwargs['input'] = f"{hint}{original_input}"

        return kwargs

    def _enhance_programmer(
        self,
        kwargs: Dict,
        problem_type: str
    ) -> Dict:
        """
        增强Programmer operator的analysis参数

        Programmer API: programmer(problem=str, analysis=str)
        """
        if 'analysis' not in kwargs:
            return kwargs

        original_analysis = kwargs['analysis']

        # 为gpt-oss-120b添加代码生成指导
        enhancement = "[For gpt-oss-120b: Generate complete, executable code]\n\n"
        kwargs['analysis'] = f"{enhancement}{original_analysis}"

        return kwargs

    def _enhance_test(
        self,
        kwargs: Dict,
        problem_type: str
    ) -> Dict:
        """
        增强Test operator

        Test API: test(problem=str, solution=str, entry_point=str, test=str)
        """
        # Test operator依赖test cases质量，不需要额外增强
        return kwargs

    def _enhance_review(
        self,
        kwargs: Dict,
        problem_type: str
    ) -> Dict:
        """
        增强Review operator

        Review API: review(problem=str, solution=str)
        """
        if 'problem' not in kwargs:
            return kwargs

        original_problem = kwargs['problem']

        # 为gpt-oss-120b添加review指导
        review_hint = "\n\n[Review checklist: correctness, edge cases, clarity]"
        kwargs['problem'] = f"{original_problem}{review_hint}"

        return kwargs

    def _enhance_revise(
        self,
        kwargs: Dict,
        problem_type: str
    ) -> Dict:
        """
        增强Revise operator

        Revise API: revise(problem=str, solution=str, feedback=str)
        """
        # Revise operator的feedback已经包含指导，不需要额外增强
        return kwargs

    def _build_direct_enhancement(
        self,
        operator_name: str,
        problem_type: str,
        instruction: str = None
    ) -> str:
        """
        为gpt-oss-120b构建直接增强文本（无few-shot）

        Args:
            operator_name: operator名称
            problem_type: 问题类型
            instruction: 原始instruction（可选）

        Returns:
            增强文本（如果有），否则返回空字符串
        """
        template = self.enhancement_templates.get(operator_name, {})
        if not template:
            return ""

        guidance = template.get("guidance", "")
        return guidance if guidance else ""

    def _load_enhancement_templates(self) -> Dict:
        """
        加载operator增强模板（针对gpt-oss-120b）

        Returns:
            增强模板字典
        """
        return {
            "Custom": {
                "enhancement_target": "instruction",
                "guidance": "[For gpt-oss-120b: Be explicit and clear in your instructions]"
            },
            "AnswerGenerate": {
                "enhancement_target": "input",
                "guidance": "[For gpt-oss-120b: Provide step-by-step reasoning]"
            },
            "Programmer": {
                "enhancement_target": "analysis",
                "guidance": "[For gpt-oss-120b: Generate complete, executable code]"
            },
            "ScEnsemble": {
                "enhancement_target": "none",
                "guidance": ""
            },
            "Test": {
                "enhancement_target": "test_cases",
                "guidance": ""
            },
            "Review": {
                "enhancement_target": "problem",
                "guidance": "[Review checklist: correctness, edge cases, clarity]"
            },
            "Revise": {
                "enhancement_target": "feedback",
                "guidance": ""
            }
        }
