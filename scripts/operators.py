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


# 导出所有算子
__all__ = [
    'BaseOperator',
    'Custom',
    'AnswerGenerate',
    'Programmer',
    'Test',
    'Review',
    'Revise',
    'ScEnsemble'
]
