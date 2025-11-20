#!/usr/bin/env python3
"""
RL工作流生成器 - 使用RL训练的Qwen2.5-7B生成优化的工作流
"""
import torch
import json
import ast
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
import os

class RLWorkflowGenerator:
    """使用RL训练的Qwen2.5-7B生成优化的工作流"""

    def __init__(
        self,
        base_model: str = "Qwen/Qwen2.5-7B-Instruct",
        lora_checkpoint: Optional[str] = None,
        device_ids: List[int] = [2, 3],
        operator_descriptions_path: Optional[str] = None,
        config: Optional[Dict] = None
    ):
        """
        Args:
            base_model: 基座模型路径
            lora_checkpoint: LoRA检查点路径（None表示使用基座模型）
            device_ids: 使用的GPU ID列表
            operator_descriptions_path: AFlow算子描述文件路径
            config: 额外配置
        """
        self.base_model = base_model
        self.lora_checkpoint = lora_checkpoint
        self.device_ids = device_ids
        self.device = f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu"
        self.config = config or {}

        # 设置CUDA设备
        if torch.cuda.is_available():
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, device_ids))

        print(f"🔧 初始化RL工作流生成器")
        print(f"  设备: {self.device}")
        print(f"  GPU: {device_ids}")

        # 加载tokenizer
        print(f"📥 加载tokenizer: {base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True
        )

        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载模型
        print(f"📥 加载基座模型: {base_model}")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map={"": self.device},
            trust_remote_code=True
        )

        # 加载LoRA权重（如果有）
        if lora_checkpoint:
            print(f"📥 加载LoRA检查点: {lora_checkpoint}")
            self.model = PeftModel.from_pretrained(self.model, lora_checkpoint)
            self.model.eval()

        # 加载算子描述
        self.operator_descriptions = self._load_operator_descriptions(operator_descriptions_path)

        print(f"✅ RL工作流生成器初始化完成")

    def _load_operator_descriptions(self, descriptions_path: Optional[str]) -> Dict:
        """加载AFlow算子描述"""
        if descriptions_path and Path(descriptions_path).exists():
            with open(descriptions_path, 'r') as f:
                return json.load(f)

        # 默认算子描述
        return {
            "Custom": {
                "description": "Generates anything based on customized input and instruction.",
                "interface": "custom(input: str, instruction: str) -> dict with key 'response'"
            },
            "AnswerGenerate": {
                "description": "Generates step-by-step reasoning and final answer.",
                "interface": "answer_generate(input: str) -> dict with keys 'thought' and 'answer'"
            },
            "Programmer": {
                "description": "Automatically writes and executes Python code.",
                "interface": "programmer(problem: str, analysis: str = 'None') -> dict with keys 'code' and 'output'"
            },
            "ScEnsemble": {
                "description": "Uses self-consistency to select the most frequent solution.",
                "interface": "sc_ensemble(solutions: List[str], problem: str) -> dict with key 'response'"
            },
            "Review": {
                "description": "Reviews and provides feedback on a solution.",
                "interface": "review(problem: str, solution: str) -> dict with keys 'review_result' and 'feedback'"
            },
            "Revise": {
                "description": "Revises solution based on feedback.",
                "interface": "revise(problem: str, solution: str, feedback: str) -> dict with key 'solution'"
            }
        }

    def _build_generation_prompt(self, problem: str, problem_type: str) -> str:
        """构建提示词，明确算子 API"""

        prompt = f"""Generate a Python Workflow class. Follow the exact template and API signatures.

CRITICAL: Only use operators listed below with their EXACT parameters!
CRITICAL: Initialize ALL variables before using them! Never return undefined variables!
CRITICAL: If a variable is defined inside an if-block, either initialize it before the if-block OR handle both branches!

Available Operators:

1. Custom(llm) - Most flexible, for any custom task
   Call: await self.custom(input=str, instruction=str)
   Returns: {{'response': str}}

2. AnswerGenerate(llm) - Step-by-step reasoning
   Call: await self.answer_generate(input=str)  ← NO instruction parameter!
   Returns: {{'thought': str, 'answer': str}}

3. Programmer(llm) - Auto-generate and execute Python code
   Call: await self.programmer(problem=str, analysis=str)
   Returns: {{'code': str, 'output': str}}

4. Test(llm) - Test code with test cases
   Call: await self.test(problem=str, solution=str, entry_point=str)
   Returns: {{'result': bool, 'solution': str}}

5. Review(llm) - Review and validate solution
   Call: await self.review(problem=str, solution=str)
   Returns: {{'review_result': bool, 'feedback': str}}

6. Revise(llm) - Revise solution based on feedback
   Call: await self.revise(problem=str, solution=str, feedback=str)
   Returns: {{'solution': str}}

7. ScEnsemble(llm) - Self-consistency ensemble voting
   Call: await self.sc_ensemble(solutions=list, problem=str)
   Returns: {{'response': str}}

Template (complete the __call__ method):

import workspace.{problem_type}.workflows.template.operator as operator
from scripts.async_llm import create_llm_instance
from scripts.evaluator import DatasetType

class Workflow:
    def __init__(self, name: str, llm_config, dataset: DatasetType):
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        # Initialize operators you need (ONLY the ones you will use):
        # self.custom = operator.Custom(self.llm)
        # self.answer_generate = operator.AnswerGenerate(self.llm)
        # self.programmer = operator.Programmer(self.llm)
        # self.test = operator.Test(self.llm)
        # self.review = operator.Review(self.llm)
        # self.revise = operator.Revise(self.llm)
        # self.sc_ensemble = operator.ScEnsemble(self.llm)

    async def __call__(self, problem: str):
        # Solve: {problem}
        # MUST return (solution, cost) tuple
        # Example: return solution['response'], self.llm.get_usage_summary()["total_cost"]

        # IMPORTANT: Initialize solution variable before any if-blocks!
        # Good example:
        #   solution = await self.answer_generate(input=problem)
        #   answer = solution.get('answer', '')
        #   if some_condition:
        #       answer = improved_answer  # Modify existing variable
        #   return answer, cost  # Always defined
        #
        # Bad example (NEVER do this):
        #   if some_condition:
        #       answer = ...  # Only defined in if-block
        #   return answer, cost  # ERROR: answer may be undefined!

        pass
"""

        return prompt

    def generate_workflow(
        self,
        problem: str,
        problem_type: str = "math",
        temperature: float = 0.7,
        max_new_tokens: int = 2048,
        return_full_output: bool = False,
        custom_prompt: Optional[str] = None
    ) -> Dict:
        """
        生成优化的工作流

        Args:
            problem: 问题文本
            problem_type: 问题类型 (math/code/qa)
            temperature: 采样温度
            max_new_tokens: 最大生成token数
            return_full_output: 是否返回完整输出
            custom_prompt: 自定义提示词（如果提供，将覆盖默认提示词）

        Returns:
            {
                "workflow_code": "Python代码",
                "valid": bool,
                "error": Optional[str],
                "metadata": {...}
            }
        """

        # 构建提示词（支持动态注入）
        if custom_prompt is not None:
            prompt = custom_prompt
        else:
            prompt = self._build_generation_prompt(problem, problem_type)

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=self.config.get('top_p', 0.95),
                top_k=self.config.get('top_k', 50),
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # 解码
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        # 解析输出
        workflow_code, is_valid, error = self._parse_workflow_code(generated_text, problem_type)

        result = {
            "workflow_code": workflow_code,
            "valid": is_valid,
            "error": error,
            "metadata": {
                "problem": problem,
                "problem_type": problem_type,
                "temperature": temperature,
                "tokens_generated": outputs.shape[1] - inputs['input_ids'].shape[1]
            }
        }

        if return_full_output:
            result["full_output"] = generated_text
            result["prompt"] = prompt

        return result

    def _parse_workflow_code(self, generated_text: str, problem_type: str) -> Tuple[str, bool, Optional[str]]:
        """解析生成的文本，提取并验证工作流代码"""

        # DEBUG: 打印 Qwen 生成的原始文本
        print(f"\n{'='*60}")
        print(f"🔍 DEBUG: Qwen 生成的原始文本 (完整):")
        print(f"{'='*60}")
        print(generated_text)  # 打印完整文本
        print(f"{'='*60}\n")

        # 提取代码块
        code_start = generated_text.find("```python")
        if code_start == -1:
            # 没有markdown代码块，尝试直接查找class定义
            code_start = generated_text.find("class Workflow:")
            if code_start == -1:
                print(f"⚠️  未找到 'class Workflow:'，使用默认工作流")
                return self._get_default_workflow(problem_type), False, "No Workflow class found in output"

            code = generated_text[code_start:]
        else:
            code_start += len("```python\n")
            code_end = generated_text.find("```", code_start)

            if code_end == -1:
                code = generated_text[code_start:]
            else:
                code = generated_text[code_start:code_end]

        # 去除首尾空白
        code = code.strip()

        # 验证语法
        try:
            ast.parse(code)
            is_valid = True
            error = None
        except SyntaxError as e:
            is_valid = False
            error = f"Syntax error: {str(e)}"
            # 返回默认工作流
            code = self._get_default_workflow(problem_type)

        return code, is_valid, error

    def _get_default_workflow(self, problem_type: str = "math") -> str:
        """默认工作流（当生成失败时）"""
        return f"""import workspace.{problem_type}.workflows.template.operator as operator
from scripts.async_llm import create_llm_instance
from scripts.evaluator import DatasetType

class Workflow:
    def __init__(self, name: str, llm_config, dataset: DatasetType):
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)

    async def __call__(self, problem: str):
        solution = await self.custom(input=problem, instruction="Solve this problem step by step.")
        return solution['response'], self.llm.get_usage_summary()["total_cost"]
"""


def test_generator():
    """测试生成器"""
    print("\n" + "=" * 60)
    print("🧪 测试RL工作流生成器")
    print("=" * 60)

    # 注意：这需要Qwen模型，如果没有下载会很慢
    generator = RLWorkflowGenerator(
        base_model="Qwen/Qwen2.5-7B-Instruct",
        device_ids=[2, 3],
        operator_descriptions_path="/home/yijia/.claude/11/AFlow/workspace/MATH/workflows/template/operator.json"
    )

    # 测试问题
    test_problem = "What is 15 + 27?"

    print(f"\n📝 测试问题: {test_problem}")

    # 生成工作流
    result = generator.generate_workflow(
        problem=test_problem,
        problem_type="math",
        temperature=0.7,
        max_new_tokens=1024
    )

    print(f"\n✅ 生成结果:")
    print(f"  有效性: {result['valid']}")
    if result['error']:
        print(f"  错误: {result['error']}")

    print(f"\n📄 生成的工作流代码:")
    print(result['workflow_code'])


if __name__ == "__main__":
    test_generator()
