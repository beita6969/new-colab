#!/usr/bin/env python3
"""
vLLM工作流生成器 - 使用vLLM API进行并发推理（Fallback: 使用transformers）
"""
import asyncio
import torch
from openai import AsyncOpenAI
from typing import Dict, List, Optional, Tuple
import json
import ast
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

class VLLMWorkflowGenerator:
    """使用vLLM API生成优化的工作流（支持并发）

    支持两种模式：
    1. vLLM API模式（推荐）：通过AsyncOpenAI客户端调用vLLM服务
    2. Transformers模式（Fallback）：直接使用transformers库
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8003/v1",
        api_key: str = "EMPTY",
        model_name: str = "/home/yijia/verl-agent/models/qwen/Qwen2___5-7B-Instruct",
        max_concurrent: int = 6,
        operator_descriptions_path: Optional[str] = None,
        config: Optional[Dict] = None,
        use_vllm_api: bool = False,  # 默认使用transformers模式
        device: str = "cuda:0"
    ):
        """
        Args:
            base_url: vLLM服务器地址
            api_key: API密钥（vLLM不需要真实密钥）
            model_name: 模型名称/路径
            max_concurrent: 最大并发请求数
            operator_descriptions_path: AFlow算子描述文件路径
            config: 额外配置
            use_vllm_api: 是否使用vLLM API（False则使用transformers）
            device: 设备（transformers模式）
        """
        self.model_name = model_name
        self.max_concurrent = max_concurrent
        self.config = config or {}
        self.use_vllm_api = use_vllm_api
        self.device = device

        # 加载算子描述
        self.operator_descriptions = self._load_operator_descriptions(operator_descriptions_path)

        if use_vllm_api:
            # vLLM API模式
            self.client = AsyncOpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=300.0,  # 5分钟超时
                max_retries=2
            )
            self.semaphore = asyncio.Semaphore(max_concurrent)
            print(f"✅ 初始化vLLM工作流生成器（API模式）")
            print(f"  服务器: {base_url}")
            print(f"  最大并发: {max_concurrent}")
        else:
            # Transformers模式（直接使用已加载的模型）
            self.model = None  # 将由外部设置（避免重复加载）
            self.tokenizer = None
            # ⚠️ 关键修复：使用锁保护GPU访问（同一时间只允许一个推理）
            self._generation_lock = asyncio.Lock()
            print(f"✅ 初始化workflow生成器（Transformers模式）")
            print(f"  模型: {model_name}")
            print(f"  设备: {device}")
            print(f"  ⚠️  GPU推理将串行执行（避免CUDA冲突）")

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
        """构建生成提示词"""
        prompt = f"""Generate a Python Workflow class to solve the given problem.

IMPORTANT: Consider the problem's difficulty and complexity when designing your workflow.
- Simple problems may only need one operator
- Complex problems may benefit from review loops or multiple attempts

CRITICAL RULES:
- Only use operators listed below with their EXACT parameters
- Initialize ALL operators in __init__ before using them in __call__
- Never return undefined variables
- Design your workflow freely - you decide which operators to use and how to combine them

Available Operators:

1. Custom(llm) - Most flexible, for any custom task
   Call: await self.custom(input=str, instruction=str)
   Returns: {{'response': str}}

2. AnswerGenerate(llm) - Step-by-step reasoning
   Call: await self.answer_generate(input=str)
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

Template:

import workspace.{problem_type}.workflows.template.operator as operator
from scripts.async_llm import create_llm_instance
from scripts.evaluator import DatasetType

class Workflow:
    def __init__(self, name: str, llm_config, dataset: DatasetType):
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)

        # Initialize operators you need (examples):
        # self.answer_generate = operator.AnswerGenerate(self.llm)
        # self.review = operator.Review(self.llm)
        # self.revise = operator.Revise(self.llm)
        # self.programmer = operator.Programmer(self.llm)
        # self.test = operator.Test(self.llm)
        # self.custom = operator.Custom(self.llm)
        # self.sc_ensemble = operator.ScEnsemble(self.llm)

    async def __call__(self, problem: str):
        # Solve: {problem}
        # MUST return (solution, cost) tuple
        # Example patterns:
        #
        # Pattern 1 - Simple:
        # solution = await self.answer_generate(input=problem)
        # return solution['answer'], self.llm.get_usage_summary()["total_cost"]
        #
        # Pattern 2 - Review+Revise loop:
        # solution = await self.answer_generate(input=problem)
        # review = await self.review(problem=problem, solution=solution['answer'])
        # if not review['review_result']:
        #     revised = await self.revise(problem=problem, solution=solution['answer'], feedback=review['feedback'])
        #     return revised['solution'], self.llm.get_usage_summary()["total_cost"]
        # return solution['answer'], self.llm.get_usage_summary()["total_cost"]
        #
        # Pattern 3 - Code with test:
        # code_result = await self.programmer(problem=problem, analysis='')
        # test_result = await self.test(problem=problem, solution=code_result['code'], entry_point='solution')
        # return test_result['solution'], self.llm.get_usage_summary()["total_cost"]
        pass
"""
        return prompt

    async def generate_workflow(
        self,
        problem: str,
        problem_type: str = "math",
        temperature: float = 0.7,
        max_new_tokens: int = 2048,
        custom_prompt: Optional[str] = None
    ) -> Dict:
        """
        生成单个工作流（异步）

        Returns:
            {
                "workflow_code": "Python代码",
                "valid": bool,
                "error": Optional[str],
                "metadata": {...}
            }
        """
        if self.use_vllm_api:
            return await self._generate_with_vllm_api(
                problem, problem_type, temperature, max_new_tokens, custom_prompt
            )
        else:
            return await self._generate_with_transformers(
                problem, problem_type, temperature, max_new_tokens, custom_prompt
            )

    async def _generate_with_vllm_api(
        self,
        problem: str,
        problem_type: str,
        temperature: float,
        max_tokens: int,
        custom_prompt: Optional[str]
    ) -> Dict:
        """使用vLLM API生成"""
        async with self.semaphore:  # 控制并发数
            try:
                # 构建提示词
                prompt = custom_prompt or self._build_generation_prompt(problem, problem_type)

                # 调用vLLM API
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a workflow generation expert."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=self.config.get('top_p', 0.95),
                )

                # 提取生成的代码
                generated_text = response.choices[0].message.content
                workflow_code, is_valid, error = self._parse_workflow_code(generated_text, problem_type)

                return {
                    "workflow_code": workflow_code,
                    "valid": is_valid,
                    "error": error,
                    "metadata": {
                        "tokens": response.usage.total_tokens if response.usage else 0,
                        "model": self.model_name
                    }
                }

            except Exception as e:
                return {
                    "workflow_code": "",
                    "valid": False,
                    "error": str(e),
                    "metadata": {}
                }

    async def _generate_with_transformers(
        self,
        problem: str,
        problem_type: str,
        temperature: float,
        max_new_tokens: int,
        custom_prompt: Optional[str]
    ) -> Dict:
        """使用transformers生成（使用锁保护GPU访问）"""
        # ⚠️ 关键：使用锁确保同一时间只有一个推理在执行
        async with self._generation_lock:
            loop = asyncio.get_event_loop()

            def _sync_generate():
                """同步生成函数（在线程池中执行）"""
                # 构建提示词
                prompt = custom_prompt or self._build_generation_prompt(problem, problem_type)

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

                return generated_text

            try:
                # 在默认executor中运行（CPU密集型操作）
                generated_text = await loop.run_in_executor(None, _sync_generate)

                # 解析输出
                workflow_code, is_valid, error = self._parse_workflow_code(generated_text, problem_type)

                return {
                    "workflow_code": workflow_code,
                    "valid": is_valid,
                    "error": error,
                    "metadata": {
                        "problem": problem,
                        "problem_type": problem_type,
                        "temperature": temperature
                    }
                }
            except Exception as e:
                return {
                    "workflow_code": "",
                    "valid": False,
                    "error": str(e),
                    "metadata": {}
                }

    def _parse_workflow_code(self, generated_text: str, problem_type: str) -> Tuple[str, bool, Optional[str]]:
        """解析生成的文本，提取并验证工作流代码"""
        # 提取代码块
        code_start = generated_text.find("```python")
        if code_start == -1:
            code_start = generated_text.find("class Workflow:")
            if code_start == -1:
                return self._get_default_workflow(problem_type), False, "No Workflow class found"
            code = generated_text[code_start:]
        else:
            code_start += len("```python\n")
            code_end = generated_text.find("```", code_start)
            code = generated_text[code_start:code_end] if code_end != -1 else generated_text[code_start:]

        code = code.strip()

        # ⚠️ Auto-Fix：自动修复缺失的operator初始化
        code = self._validate_and_fix_workflow(code, problem_type)

        # 验证语法
        try:
            ast.parse(code)
            is_valid = True
            error = None
        except SyntaxError as e:
            is_valid = False
            error = f"Syntax error: {str(e)}"
            code = self._get_default_workflow(problem_type)

        return code, is_valid, error

    def _validate_and_fix_workflow(self, code: str, problem_type: str) -> str:
        """验证并自动修复workflow中缺失的operator初始化

        Args:
            code: 生成的workflow代码
            problem_type: 问题类型

        Returns:
            修复后的代码
        """
        import re

        # 1. 提取__init__中已初始化的operators
        initialized_ops = set()
        init_section = re.search(r'def __init__\([^)]+\):[\s\S]+?(?=\n    async def|\n    def|$)', code)
        if init_section:
            init_code = init_section.group(0)
            # 匹配 self.xxx = operator.XXX(self.llm)
            init_patterns = re.findall(r'self\.(\w+)\s*=\s*operator\.(\w+)\(', init_code)
            for attr_name, op_name in init_patterns:
                initialized_ops.add(attr_name)

        # 2. 提取__call__中使用的operators
        used_ops = set()
        call_section = re.search(r'async def __call__\([^)]+\):[\s\S]+', code)
        if call_section:
            call_code = call_section.group(0)
            # 匹配 await self.xxx(...)
            used_patterns = re.findall(r'await self\.(\w+)\(', call_code)
            for op_name in used_patterns:
                used_ops.add(op_name)

        # 3. 找出缺失的operators
        missing_ops = used_ops - initialized_ops

        if missing_ops:
            print(f"\n⚠️  检测到缺失的operator初始化: {missing_ops}")
            print(f"   已初始化: {initialized_ops}")
            print(f"   已使用: {used_ops}")

            # 4. 自动添加缺失的初始化代码
            # 找到 self.llm = create_llm_instance(...) 的位置
            llm_init_match = re.search(r'(\s+)(self\.llm = create_llm_instance\([^)]+\))', code)
            if llm_init_match:
                indent = llm_init_match.group(1)
                llm_init_line = llm_init_match.group(2)

                # 构建缺失的初始化代码
                missing_inits = []
                for op_name in sorted(missing_ops):
                    # 推断operator类名（首字母大写+驼峰命名）
                    # answer_generate -> AnswerGenerate
                    # review -> Review
                    op_class_name = ''.join(word.capitalize() for word in op_name.split('_'))

                    # 检查是否是有效的operator（从prompt中获取）
                    valid_operators = ['Custom', 'AnswerGenerate', 'Programmer', 'Test', 'Review', 'Revise', 'ScEnsemble']
                    if op_class_name in valid_operators:
                        missing_inits.append(f"{indent}self.{op_name} = operator.{op_class_name}(self.llm)")

                if missing_inits:
                    # 在 self.llm = ... 之后插入
                    insert_code = '\n' + '\n'.join(missing_inits)
                    code = code.replace(llm_init_line, llm_init_line + insert_code)
                    print(f"✅ 自动添加了 {len(missing_inits)} 个缺失的operator初始化")

        return code

    def _get_default_workflow(self, problem_type: str = "math") -> str:
        """默认工作流"""
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

    async def generate_workflows_batch(
        self,
        problems: List[str],
        problem_types: List[str],
        temperatures: List[float],
        custom_prompts: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        批量并发生成工作流（优化版：使用GPU batch推理）

        Args:
            problems: 问题列表
            problem_types: 问题类型列表
            temperatures: 温度列表
            custom_prompts: 自定义提示词列表

        Returns:
            结果列表
        """
        if self.use_vllm_api:
            # vLLM API模式：并发调用
            tasks = []
            for i in range(len(problems)):
                task = self.generate_workflow(
                    problem=problems[i],
                    problem_type=problem_types[i],
                    temperature=temperatures[i],
                    custom_prompt=custom_prompts[i] if custom_prompts else None
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            processed_results = []
            for result in results:
                if isinstance(result, Exception):
                    processed_results.append({
                        "workflow_code": "",
                        "valid": False,
                        "error": str(result),
                        "metadata": {}
                    })
                else:
                    processed_results.append(result)

            return processed_results
        else:
            # Transformers模式：使用GPU batch推理（关键优化！）
            return await self._batch_generate_with_transformers(
                problems, problem_types, temperatures, custom_prompts
            )

    async def _batch_generate_with_transformers(
        self,
        problems: List[str],
        problem_types: List[str],
        temperatures: List[float],
        custom_prompts: Optional[List[str]]
    ) -> List[Dict]:
        """使用transformers批量生成（GPU batch推理）"""
        loop = asyncio.get_event_loop()

        def _sync_batch_generate():
            """同步批量生成函数"""
            # 构建所有prompts
            prompts = []
            for i in range(len(problems)):
                if custom_prompts and custom_prompts[i]:
                    prompt = custom_prompts[i]
                else:
                    prompt = self._build_generation_prompt(problems[i], problem_types[i])
                prompts.append(prompt)

            # 批量tokenize（关键：padding对齐）
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,  # 对齐到最长序列
                truncation=True,
                max_length=3072
            ).to(self.device)

            # 批量生成（6个prompt一起在GPU上并行）
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.get('max_new_tokens', 2048),
                    temperature=temperatures[0],  # 假设temperature相同
                    top_p=self.config.get('top_p', 0.95),
                    top_k=self.config.get('top_k', 50),
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # 批量解码
            generated_texts = self.tokenizer.batch_decode(
                outputs[:, inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            return generated_texts

        try:
            # 在线程池执行批量推理
            generated_texts = await loop.run_in_executor(None, _sync_batch_generate)

            # 解析所有结果
            results = []
            for i, generated_text in enumerate(generated_texts):
                workflow_code, is_valid, error = self._parse_workflow_code(
                    generated_text, problem_types[i]
                )
                results.append({
                    "workflow_code": workflow_code,
                    "valid": is_valid,
                    "error": error,
                    "metadata": {
                        "problem": problems[i],
                        "problem_type": problem_types[i],
                        "temperature": temperatures[i]
                    }
                })

            return results

        except Exception as e:
            # 出错时返回空结果
            return [{
                "workflow_code": "",
                "valid": False,
                "error": str(e),
                "metadata": {}
            } for _ in problems]
