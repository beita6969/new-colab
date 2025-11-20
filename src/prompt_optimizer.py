#!/usr/bin/env python3
"""
提示词优化器 - Layer 1: Workflow生成提示词动态优化
"""
from typing import Dict, List


class PromptOptimizer:
    """
    动态提示词优化器

    功能：
    1. 提供完整7个operator模板
    2. Few-shot示例学习（从experience_buffer检索）
    3. 问题类型自适应指导
    4. 动态组合生成最优提示词
    """

    def __init__(self):
        """
        初始化提示词优化器
        """
        self.operator_templates = self._load_operator_templates()
        self.type_guidance = self._load_type_guidance()

    def build_dynamic_prompt(
        self,
        problem: str,
        problem_type: str
    ) -> str:
        """
        构建动态优化的提示词

        Args:
            problem: 问题文本
            problem_type: 问题类型 (math/code/qa)

        Returns:
            优化后的完整提示词
        """
        # 1. 基础模板（完整7算子）
        base_template = self._get_full_operator_template()

        # 2. 类型自适应指导
        type_guidance_section = self.type_guidance.get(
            problem_type,
            self.type_guidance["qa"]  # 默认使用QA指导
        )

        # 4. 根据问题类型确定workflow签名
        if problem_type == "code":
            call_signature = "async def __call__(self, problem: str, entry_point: str, test: str):"
            call_comment = """# Solve: {problem}
        # entry_point: The function name to test (HumanEval format)
        # test: The test code containing check() function (HumanEval format)
        # MUST return (solution, cost) tuple
        # Example: return code, self.llm.get_usage_summary()["total_cost"]"""
        else:
            call_signature = "async def __call__(self, problem: str):"
            call_comment = """# Solve: {problem}
        # MUST return (solution, cost) tuple
        # Safe access: return solution.get('response', ''), self.llm.get_usage_summary().get("total_cost", 0.0)"""

        # 3. 组合提示词
        prompt = f"""Generate a Python Workflow class. Follow the exact template and API signatures.

🚨 CRITICAL RULES FOR OPERATOR INITIALIZATION AND CALLS:

1️⃣ OPERATOR CLASS NAMES (PascalCase - VERY IMPORTANT):
   ✅ CORRECT: self.custom = operator.Custom(self.llm)
   ✅ CORRECT: self.answer_generate = operator.AnswerGenerate(self.llm)
   ✅ CORRECT: self.test = operator.Test(self.llm)
   ❌ WRONG: self.custom = operator.custom(self.llm)
   ❌ WRONG: self.answer_generate = operator.answer_generate(self.llm)

⚡ PERFORMANCE CRITICAL - AVOID REDUNDANT CALLS:
   ✅ CORRECT: Cache operator results and reuse them
   result = await self.answer_generate(input=problem)
   answer = result.get('answer', '')
   # Use 'answer' variable multiple times - DO NOT call again!

   ❌ WRONG: Calling the same operator multiple times with same inputs
   result1 = await self.answer_generate(input=problem)  # First call
   # ... some code ...
   result2 = await self.answer_generate(input=problem)  # Redundant! Wastes time!

2️⃣ OPERATOR CALL RULES:
   ├─ EVERY operator call MUST include ALL required parameters
   ├─ DO NOT skip any required parameters - this will cause TypeError
   ├─ Use the exact parameter names shown in the Interface
   └─ Follow the Example call format exactly

3️⃣ Example INCORRECT calls (WILL FAIL):
   ❌ await self.test(problem=problem)  # Missing solution and entry_point!
   ❌ await self.review(solution=code)  # Missing problem!
   ❌ self.custom = operator.custom(self.llm)  # Wrong case!

4️⃣ Example CORRECT calls (WILL WORK):
   ✅ self.custom = operator.Custom(self.llm)  # Correct case!
   ✅ await self.test(problem=problem, solution=solution, entry_point=entry_point)
   ✅ await self.review(problem=problem, solution=code)

5️⃣ VARIABLE SCOPE CRITICAL RULE:
   ⚠️  ALWAYS initialize variables at function start, BEFORE any if/else blocks!

   ❌ WRONG - UnboundLocalError risk:
   if condition:
       result_var = await self.revise(...)  # Only defined in if block
   return result_var  # ERROR if condition is False!

   ✅ CORRECT - Always define first:
   result_var = initial_value  # Initialize at function start
   if condition:
       result_var = await self.revise(...)  # Update if needed
   return result_var  # Always safe to use

   Common variable names to initialize:
   - revised_code = code
   - final_answer = answer
   - solution = initial_solution

{base_template}

{type_guidance_section}

Template (complete the __call__ method):

import workspace.{problem_type}.workflows.template.operator as operator
from scripts.async_llm import create_llm_instance
from scripts.evaluator import DatasetType

class Workflow:
    def __init__(self, name: str, llm_config, dataset: DatasetType):
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        # Initialize operators you need, e.g.:
        # self.custom = operator.Custom(self.llm)
        # self.answer_generate = operator.AnswerGenerate(self.llm)
        # self.programmer = operator.Programmer(self.llm)
        # self.sc_ensemble = operator.ScEnsemble(self.llm)
        # self.test = operator.Test(self.llm)
        # self.review = operator.Review(self.llm)
        # self.revise = operator.Revise(self.llm)

    {call_signature}
        {call_comment}
        pass
"""

        return prompt

    def _get_full_operator_template(self) -> str:
        """
        返回完整operator列表定义（基于operator_templates）

        Returns:
            完整operator列表说明
        """
        lines = ["Available Operators (use intelligently based on problem type):\n"]

        for i, (op_name, op_info) in enumerate(self.operator_templates.items(), 1):
            lines.append(f"{i}. {op_name}")
            lines.append(f"   Description: {op_info['description']}")
            lines.append(f"   Interface: {op_info['interface']}")
            lines.append(f"   Returns: {op_info['returns']}")
            if 'required_params' in op_info:
                lines.append(f"   ⚠️ REQUIRED params: {', '.join(op_info['required_params'])}")
            if 'example_call' in op_info:
                lines.append(f"   Example call: {op_info['example_call']}")
            if 'note' in op_info:
                lines.append(f"   Note: {op_info['note']}")
            lines.append("")

        return "\n".join(lines)

    def _load_operator_templates(self) -> Dict:
        """
        加载operator模板定义

        Returns:
            operator模板字典
        """
        return {
            "Custom": {
                "description": "Most flexible, for any custom task",
                "interface": "Custom(input: str, instruction: str)",
                "returns": "{'response': str}"
            },
            "AnswerGenerate": {
                "description": "Step-by-step reasoning",
                "interface": "AnswerGenerate(input: str)",
                "returns": "{'thought': str, 'answer': str}",
                "example_call": "ans_result = await self.answer_generate(input=problem)\nanswer = ans_result.get('answer', '')  # Extract 'answer' not 'thought'",
                "required_params": ["input"],
                "note": "Returns dict with 'thought' (reasoning) and 'answer' (final result) - use 'answer' for final output"
            },
            "Programmer": {
                "description": "Auto-generate and execute Python code",
                "interface": "Programmer(problem: str, analysis: str)",
                "returns": "{'code': str, 'output': str}",
                "example_call": "prog_result = await self.programmer(problem=problem, analysis='Analyze and solve')\ncode = prog_result.get('code', '')  # Extract code string from dict",
                "note": "Returns dict with 'code' key - must extract before passing to Test"
            },
            "ScEnsemble": {
                "description": "Self-consistency ensemble",
                "interface": "ScEnsemble(solutions: List[str], problem: str)",
                "returns": "{'response': str}"
            },
            "Test": {
                "description": "Test the solution with test cases, if correct return 'no error'; if incorrect, reflect on the error",
                "interface": "Test(problem: str, solution: str, entry_point: str)",
                "returns": "{'result': bool, 'solution': str}",
                "note": "For HumanEval format - automatically extracts test cases using entry_point",
                "example_call": "result = await self.test(problem=problem, solution=solution, entry_point=entry_point)",
                "required_params": ["problem", "solution", "entry_point"]
            },
            "Review": {
                "description": "Review and verify solution",
                "interface": "Review(problem: str, solution: str)",
                "returns": "{'review_result': str, 'feedback': str}",
                "example_call": "review_result = await self.review(problem=problem, solution=code)\nfeedback = review_result.get('feedback', review_result.get('review_result', 'No feedback'))  # Handle multiple formats",
                "required_params": ["problem", "solution"],
                "note": "May return 'feedback' OR 'review_result' key - use nested .get() for safety"
            },
            "Revise": {
                "description": "Revise based on feedback",
                "interface": "Revise(problem: str, solution: str, feedback: str)",
                "returns": "{'solution': str}",
                "example_call": "revised = await self.revise(problem=problem, solution=code, feedback=feedback)\nrevised_code = revised.get('solution', code)  # Extract 'solution' with fallback",
                "required_params": ["problem", "solution", "feedback"],
                "note": "Returns dict with 'solution' key - always use .get() to avoid KeyError"
            }
        }

    def _load_type_guidance(self) -> Dict:
        """
        加载问题类型自适应指导

        Returns:
            类型指导字典
        """
        return {
            "math": """
✅ MATH Problem Requirements:
- Return final answer in \\boxed{} notation
- You may use AnswerGenerate, Programmer, Test, Review, Revise operators
- IMPORTANT: If using Test operator, MUST call with ALL parameters:
    result = await self.test(problem=problem, solution=solution, entry_point="solve")
- Choose operators based on problem requirements
""",
            "code": """
🚨🚨🚨 CRITICAL - CODE WORKFLOW SIGNATURE 🚨🚨🚨
Your __call__ method MUST accept exactly these parameters:
  async def __call__(self, problem: str, entry_point: str, test: str):

  - problem: The problem description
  - entry_point: Function name to test (e.g., "has_close_elements")
  - test: HumanEval test code containing check() function

⚠️⚠️⚠️ TEST OPERATOR CRITICAL REQUIREMENTS ⚠️⚠️⚠️
The Test operator MUST be called with ALL 3 PARAMETERS or it will fail with TypeError:

✅ CORRECT (WILL WORK):
  result = await self.test(problem=problem, solution=code, entry_point=entry_point)

❌ WRONG (WILL FAIL WITH TypeError):
  await self.test(problem=problem)  # Missing solution and entry_point!
  await self.test(solution=code)    # Missing problem and entry_point!
  await self.test(problem=problem, solution=code)  # Missing entry_point!

📝 RESPONSE FORMAT - USE .get() FOR SAFETY:
```python
# ✅ ALWAYS use .get() with defaults to avoid KeyError
result = await self.some_operator(...)

# Safe access patterns:
content = result.get('response', result.get('answer', ''))
code = result.get('code', '')
feedback = result.get('feedback', result.get('review_result', 'No feedback'))
success = result.get('success', result.get('result', False))

# For nested access:
standardized = result.get('__standardized__', {})
if standardized:
    content = standardized.get('content', '')
    success = standardized.get('success', False)
```
```python
async def __call__(self, problem: str, entry_point: str, test: str):
    # Step 1: Generate code ONCE - cache result
    prog_result = await self.programmer(problem=problem, analysis="Analyze and solve")
    code = prog_result.get('code', '')

    # Step 2: Test with ALL 3 PARAMETERS
    test_result = await self.test(problem=problem, solution=code, entry_point=entry_point)

    # Step 3: If failed, review ONCE and revise ONCE
    if not test_result.get('result', False):
        review_result = await self.review(problem=problem, solution=code)
        feedback = review_result.get('feedback', review_result.get('review_result', 'Review completed'))

        revised = await self.revise(problem=problem, solution=code, feedback=feedback)
        final_code = revised.get('solution', code)

        # Optional: Test revised code (remove if time is critical)
        # final_test = await self.test(problem=problem, solution=final_code, entry_point=entry_point)

        return final_code, self.llm.get_usage_summary()["total_cost"]

    return code, self.llm.get_usage_summary()["total_cost"]
```

✅ CODE Problem Requirements:
- Your solution MUST pass the test cases
- ALWAYS use Programmer to generate code
- ALWAYS test with Test operator (with ALL 3 parameters)
- Use Review and Revise if tests fail
- NEVER call Test with only problem parameter - it WILL FAIL with TypeError
- Use Programmer to generate code, Test to verify it
- Generated code must be self-contained and executable
""",
            "qa": """
✅ QA Problem Requirements:
- Provide accurate and concise answers
- You may use AnswerGenerate, Custom, Review operators
- IMPORTANT: DO NOT call Test operator for QA problems (they don't have test cases)
- If using Test operator with custom test, MUST call with ALL parameters:
    result = await self.test(problem=problem, solution=answer, entry_point="execute")
- Choose operators based on problem complexity
"""
        }

    def get_operator_count(self, workflow_code: str) -> Dict[str, int]:
        """
        统计workflow中使用的operator数量

        Args:
            workflow_code: workflow代码

        Returns:
            operator使用计数字典
        """
        operator_count = {name: 0 for name in self.operator_templates.keys()}

        for op_name in self.operator_templates.keys():
            # 检查初始化和调用
            if f'operator.{op_name}' in workflow_code or \
               f'self.{op_name.lower()}' in workflow_code:
                operator_count[op_name] += 1

        return operator_count
