#!/usr/bin/env python3
"""
测试改进的AFlow执行器
"""
import asyncio
import sys
sys.path.append('/home/yijia/.claude/11/integrated_aflow_roll')

from src.aflow_executor import AFlowExecutor

async def test_executor():
    """测试执行器的各种场景"""

    # 创建执行器（启用Fallback）
    executor = AFlowExecutor(
        llm_config_path="/home/yijia/.claude/11/integrated_aflow_roll/config/aflow_llm.yaml",
        enable_fallback=True
    )

    print("\n" + "="*60)
    print("测试1: 正确的工作流代码")
    print("="*60)

    good_code = '''
import operator
from scripts.async_llm import create_llm_instance

class Workflow:
    def __init__(self, name, llm_config, dataset):
        self.name = name
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)

    async def __call__(self, problem):
        result = await self.custom(input=problem, instruction="Solve step by step")
        return result.get('response', ''), self.llm.get_usage_summary()["total_cost"]
'''

    answer, cost, metadata = await executor.execute_workflow(
        workflow_code=good_code,
        problem="What is 2 + 2?",
        problem_type="math"
    )

    print(f"✅ 答案: {answer[:100]}...")
    print(f"   成本: ${cost:.6f}")
    print(f"   元数据: {metadata}")

    print("\n" + "="*60)
    print("测试2: 有语法错误的工作流（测试自动修复）")
    print("="*60)

    bad_code = '''
class Workflow:
    def __init__(self, name, llm_config, dataset):
        self.custom = operator.custom(self.llm)  # 小写错误

    async def __call__(self, problem):
        result = self.custom(input=problem)  # 缺少await
        return result['response']  # 缺少cost
'''

    answer, cost, metadata = await executor.execute_workflow(
        workflow_code=bad_code,
        problem="What is 3 + 3?",
        problem_type="math"
    )

    print(f"✅ 答案: {answer[:100] if answer else 'None'}...")
    print(f"   成本: ${cost:.6f}")
    print(f"   元数据: {metadata}")

    print("\n" + "="*60)
    print("测试3: 完全无效的代码（测试Fallback）")
    print("="*60)

    invalid_code = '''
This is not even Python code!
'''

    answer, cost, metadata = await executor.execute_workflow(
        workflow_code=invalid_code,
        problem="What is 4 + 4?",
        problem_type="math"
    )

    print(f"✅ 答案: {answer[:100] if answer else 'None'}...")
    print(f"   成本: ${cost:.6f}")
    print(f"   元数据: {metadata}")

    print("\n" + "="*60)
    print("测试4: Code类型问题（测试entry_point处理）")
    print("="*60)

    code_workflow = '''
class Workflow:
    def __init__(self, name, llm_config, dataset):
        self.llm = create_llm_instance(llm_config)
        self.programmer = operator.Programmer(self.llm)

    async def __call__(self, problem, entry_point):
        result = await self.programmer(problem=problem, analysis="Solve this")
        code = result.get('code', '')
        return code, self.llm.get_usage_summary()["total_cost"]
'''

    answer, cost, metadata = await executor.execute_workflow(
        workflow_code=code_workflow,
        problem="Write a function to calculate factorial",
        problem_type="code",
        entry_point="factorial"
    )

    print(f"✅ 答案: {answer[:200] if answer else 'None'}...")
    print(f"   成本: ${cost:.6f}")
    print(f"   元数据: {metadata}")

if __name__ == "__main__":
    asyncio.run(test_executor())
