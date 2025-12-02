#!/usr/bin/env python3
"""
兼容AFlow的operators模块
提供工作流算子实现
"""
import re
import asyncio
from typing import Dict, Any, List, Optional
import subprocess
import tempfile
import os


class BaseOperator:
    """算子基类"""

    def __init__(self, llm):
        self.llm = llm

    async def __call__(self, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError


class Custom(BaseOperator):
    """自定义算子 - 最灵活的通用算子"""

    async def __call__(self, input: str, instruction: str) -> Dict[str, str]:
        prompt = f"{instruction}\n\nInput:\n{input}"
        response = await self.llm(prompt)
        return {"response": response}


class AnswerGenerate(BaseOperator):
    """答案生成算子 - 步骤推理"""

    async def __call__(self, input: str) -> Dict[str, str]:
        prompt = f"""Solve the following problem step by step.

Problem: {input}

Provide your solution in the following format:
Thought: <your step-by-step reasoning>
Answer: <your final answer>"""

        response = await self.llm(prompt)

        # 解析思考和答案
        thought = ""
        answer = ""

        if "Thought:" in response and "Answer:" in response:
            parts = response.split("Answer:")
            thought = parts[0].replace("Thought:", "").strip()
            answer = parts[1].strip() if len(parts) > 1 else ""
        else:
            # 如果格式不对，整个响应作为答案
            answer = response.strip()
            thought = "Direct answer provided"

        return {"thought": thought, "answer": answer}


class Programmer(BaseOperator):
    """编程算子 - 自动生成并执行Python代码"""

    async def __call__(self, problem: str, analysis: str = "None") -> Dict[str, str]:
        prompt = f"""Write Python code to solve the following problem.

Problem: {problem}
Analysis: {analysis}

Provide only the Python code that solves this problem.
The code should print the final answer.

```python
# Your code here
```"""

        response = await self.llm(prompt)

        # 提取代码
        code = self._extract_code(response)

        # 执行代码
        output = await self._execute_code(code)

        return {"code": code, "output": output}

    def _extract_code(self, text: str) -> str:
        """从响应中提取代码"""
        # 尝试提取markdown代码块
        pattern = r'```python\s*([\s\S]*?)```'
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()

        # 尝试提取```代码块
        pattern = r'```\s*([\s\S]*?)```'
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()

        # 直接返回文本
        return text.strip()

    async def _execute_code(self, code: str, timeout: int = 30) -> str:
        """执行Python代码"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name

            try:
                result = subprocess.run(
                    ['python3', temp_file],
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )

                if result.returncode == 0:
                    return result.stdout.strip()
                else:
                    return f"Error: {result.stderr.strip()}"
            finally:
                os.unlink(temp_file)

        except subprocess.TimeoutExpired:
            return "Error: Execution timeout"
        except Exception as e:
            return f"Error: {str(e)}"


class Test(BaseOperator):
    """测试算子 - 测试代码执行"""

    async def __call__(
        self,
        problem: str,
        solution: str,
        entry_point: str = "solution"
    ) -> Dict[str, Any]:
        # 简单执行测试
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                # 组合代码和测试
                test_code = f"{solution}\n\n# Test\nprint({entry_point}())"
                f.write(test_code)
                temp_file = f.name

            try:
                result = subprocess.run(
                    ['python3', temp_file],
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                success = result.returncode == 0
                return {
                    "result": success,
                    "solution": solution,
                    "output": result.stdout if success else result.stderr
                }
            finally:
                os.unlink(temp_file)

        except Exception as e:
            return {
                "result": False,
                "solution": solution,
                "output": str(e)
            }


class Review(BaseOperator):
    """审查算子 - 审查和验证解决方案"""

    async def __call__(self, problem: str, solution: str) -> Dict[str, Any]:
        prompt = f"""Review the following solution and determine if it is correct.

Problem: {problem}

Solution: {solution}

Provide your review:
1. Is the solution correct? (Yes/No)
2. What feedback do you have?

Format your response as:
Correct: Yes/No
Feedback: <your feedback>"""

        response = await self.llm(prompt)

        # 解析结果
        is_correct = "yes" in response.lower().split("correct:")[1].split("\n")[0].lower() if "correct:" in response.lower() else False
        feedback = response.split("Feedback:")[1].strip() if "Feedback:" in response else response

        return {
            "review_result": is_correct,
            "feedback": feedback
        }


class Revise(BaseOperator):
    """修订算子 - 根据反馈修订解决方案"""

    async def __call__(
        self,
        problem: str,
        solution: str,
        feedback: str
    ) -> Dict[str, str]:
        prompt = f"""Revise the following solution based on the feedback.

Problem: {problem}

Original Solution: {solution}

Feedback: {feedback}

Provide your revised solution:"""

        response = await self.llm(prompt)

        return {"solution": response.strip()}


class ScEnsemble(BaseOperator):
    """自洽性集成算子 - 选择最一致的答案"""

    async def __call__(self, solutions: List[str], problem: str) -> Dict[str, str]:
        if not solutions:
            return {"response": ""}

        if len(solutions) == 1:
            return {"response": solutions[0]}

        # 使用LLM选择最佳答案
        solutions_text = "\n\n".join([f"Solution {i+1}: {s}" for i, s in enumerate(solutions)])

        prompt = f"""Given the following problem and multiple solutions, select the best one.

Problem: {problem}

{solutions_text}

Which solution is most likely correct? Provide only the final answer."""

        response = await self.llm(prompt)

        return {"response": response.strip()}


class MdEnsemble(BaseOperator):
    """
    多数投票集成算子 - 通过多次打乱和投票选择最佳答案

    Paper: Can Generalist Foundation Models Outcompete Special-Purpose Tuning? Case Study in Medicine
    Link: https://arxiv.org/abs/2311.16452
    """

    def __init__(self, llm, vote_count: int = 5):
        super().__init__(llm)
        self.vote_count = vote_count

    async def __call__(self, solutions: List[str], problem: str) -> Dict[str, str]:
        """执行多数投票集成"""
        if not solutions:
            return {"solution": ""}

        if len(solutions) == 1:
            return {"solution": solutions[0]}

        from collections import Counter
        import random

        all_responses = []

        for _ in range(self.vote_count):
            # 打乱顺序
            shuffled_solutions = solutions.copy()
            random.shuffle(shuffled_solutions)
            answer_mapping = {chr(65 + i): solutions.index(sol) for i, sol in enumerate(shuffled_solutions)}

            # 构建选项文本
            solution_text = ""
            for index, solution in enumerate(shuffled_solutions):
                solution_text += f"{chr(65 + index)}: \n{str(solution)}\n\n"

            prompt = f"""Given the following problem and multiple candidate solutions, select the BEST one.

Problem: {problem}

Candidate Solutions:
{solution_text}

Analyze each solution carefully and select the best one.
Respond with ONLY the letter (A, B, C, etc.) of the best solution.

Best solution:"""

            response = await self.llm(prompt)

            # 解析答案
            answer = response.strip().upper()
            if len(answer) > 0:
                answer = answer[0]  # 取第一个字符

            if answer in answer_mapping:
                original_index = answer_mapping[answer]
                all_responses.append(original_index)

        if not all_responses:
            return {"solution": solutions[0]}  # 默认返回第一个

        # 选择出现次数最多的
        most_frequent_index = Counter(all_responses).most_common(1)[0][0]
        return {"solution": solutions[most_frequent_index]}


class Decompose(BaseOperator):
    """问题分解算子 - 将复杂问题分解为更简单的子问题"""

    async def __call__(self, problem: str) -> Dict[str, Any]:
        """分解复杂问题为子问题"""
        prompt = f"""Decompose the following problem into simpler sub-problems that can be solved step by step.

Problem: {problem}

Provide your decomposition in the following format:
<subproblems>
1. [First sub-problem]
2. [Second sub-problem]
3. [Third sub-problem]
...
</subproblems>

<reasoning>
[Explain why this decomposition helps solve the original problem]
</reasoning>"""

        response = await self.llm(prompt)

        # 解析子问题
        subproblems = []
        reasoning = ""

        # 提取子问题列表
        if "<subproblems>" in response and "</subproblems>" in response:
            start = response.find("<subproblems>") + len("<subproblems>")
            end = response.find("</subproblems>")
            subproblems_text = response[start:end].strip()

            # 解析编号的子问题
            lines = subproblems_text.split("\n")
            for line in lines:
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith("-")):
                    # 移除编号和符号
                    cleaned = re.sub(r'^[\d\.\-\)\]]+\s*', '', line)
                    if cleaned:
                        subproblems.append(cleaned)

        # 提取推理
        if "<reasoning>" in response and "</reasoning>" in response:
            start = response.find("<reasoning>") + len("<reasoning>")
            end = response.find("</reasoning>")
            reasoning = response[start:end].strip()

        # 如果解析失败，尝试简单分割
        if not subproblems:
            lines = response.split("\n")
            for line in lines:
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith("-")):
                    cleaned = re.sub(r'^[\d\.\-\)\]]+\s*', '', line)
                    if cleaned and len(cleaned) > 5:
                        subproblems.append(cleaned)

        return {
            "subproblems": subproblems if subproblems else [problem],
            "reasoning": reasoning,
            "count": len(subproblems) if subproblems else 1
        }


class Verify(BaseOperator):
    """验证算子 - 验证答案是否满足问题的所有约束条件"""

    async def __call__(self, problem: str, answer: str) -> Dict[str, Any]:
        """验证答案是否正确"""
        prompt = f"""Verify whether the following answer correctly solves the problem.

Problem: {problem}

Proposed Answer: {answer}

Carefully check:
1. Does the answer address all parts of the question?
2. Are the calculations correct (if applicable)?
3. Does the answer satisfy all constraints mentioned in the problem?
4. Is the answer in the correct format?

Provide your verification in the following format:
<is_valid>Yes/No</is_valid>
<issues>
[List any issues found, or "None" if valid]
</issues>
<corrected_answer>
[If invalid, provide a corrected answer. If valid, repeat the original answer]
</corrected_answer>"""

        response = await self.llm(prompt)

        # 解析验证结果
        is_valid = True
        issues = []
        corrected_answer = answer

        # 提取 is_valid
        if "<is_valid>" in response and "</is_valid>" in response:
            start = response.find("<is_valid>") + len("<is_valid>")
            end = response.find("</is_valid>")
            valid_text = response[start:end].strip().lower()
            is_valid = "yes" in valid_text or "true" in valid_text or "correct" in valid_text

        # 提取 issues
        if "<issues>" in response and "</issues>" in response:
            start = response.find("<issues>") + len("<issues>")
            end = response.find("</issues>")
            issues_text = response[start:end].strip()
            if issues_text.lower() != "none" and issues_text:
                issues = [i.strip() for i in issues_text.split("\n") if i.strip()]

        # 提取 corrected_answer
        if "<corrected_answer>" in response and "</corrected_answer>" in response:
            start = response.find("<corrected_answer>") + len("<corrected_answer>")
            end = response.find("</corrected_answer>")
            corrected_answer = response[start:end].strip()

        return {
            "is_valid": is_valid,
            "issues": issues,
            "corrected_answer": corrected_answer,
            "original_answer": answer
        }


# 导出所有算子
__all__ = [
    'BaseOperator',
    'Custom',
    'AnswerGenerate',
    'Programmer',
    'Test',
    'Review',
    'Revise',
    'ScEnsemble',
    'MdEnsemble',
    'Decompose',
    'Verify'
]
