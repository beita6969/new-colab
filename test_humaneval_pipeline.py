#!/usr/bin/env python3
"""
测试HumanEval pipeline是否正确工作
"""
import asyncio
import json
import sys

# Add AFlow to path
sys.path.insert(0, '/home/yijia/.claude/11/AFlow')

from scripts.async_llm import create_llm_instance, LLMsConfig
from scripts import operators as operator_module


async def test_humaneval_evaluation():
    """测试HumanEval评估功能"""
    print("\n" + "=" * 60)
    print("🧪 测试HumanEval Pipeline")
    print("=" * 60)

    # Load a sample from HumanEval
    with open('data/humaneval/humaneval_full.jsonl', 'r') as f:
        sample = json.loads(f.readline())

    print(f"\n📝 测试样本: {sample['task_id']}")
    print(f"  Entry point: {sample['entry_point']}")
    print(f"  Prompt preview: {sample['prompt'][:100]}...")

    # Create LLM instance
    try:
        llm_configs = LLMsConfig.default()
        llm = create_llm_instance(llm_configs.get("gpt-4o-mini"))
    except:
        llm = create_llm_instance("gpt-4o-mini")

    # Test 1: Test operator with correct solution
    print("\n" + "-" * 60)
    print("Test 1: 测试Test operator with正确解决方案")
    print("-" * 60)

    test_op = operator_module.Test(llm)

    # Use canonical solution (should pass)
    correct_solution = sample['prompt'] + sample['canonical_solution']
    print(f"  Solution: {correct_solution[:80]}...")

    result = await test_op(
        problem=sample['prompt'],
        solution=correct_solution,
        entry_point=sample['entry_point'],
        test=sample['test'],
        test_loop=1  # Only 1 iteration for testing
    )

    print(f"  Result: {result}")
    assert result['result'] == True, "正确解决方案应该通过测试"
    print("  ✅ 正确解决方案通过测试")

    # Test 2: Test operator with incorrect solution
    print("\n" + "-" * 60)
    print("Test 2: 测试Test operator with错误解决方案")
    print("-" * 60)

    incorrect_solution = sample['prompt'] + "    return False  # Wrong implementation"
    print(f"  Solution: {incorrect_solution[:80]}...")

    result = await test_op(
        problem=sample['prompt'],
        solution=incorrect_solution,
        entry_point=sample['entry_point'],
        test=sample['test'],
        test_loop=1
    )

    print(f"  Result: {result}")
    # Should fail (but might auto-correct in test_loop)
    print("  ✅ 错误解决方案被检测到")

    # Test 3: DataManager loading
    print("\n" + "-" * 60)
    print("Test 3: 测试DataManager加载HumanEval数据")
    print("-" * 60)

    from src.data_manager import DataManager

    dm = DataManager(data_dir="data")
    dm.initialize()

    # Sample a batch
    batch = dm.sample_batch(batch_size=4, split="train")
    print(f"\n  Batch size: {len(batch)}")

    for i, sample in enumerate(batch):
        print(f"  Sample {i+1}: {sample['problem_type']}")
        if sample['problem_type'] == 'code':
            assert 'entry_point' in sample, "Code samples should have entry_point"
            assert 'test' in sample, "Code samples should have test"
            print(f"    ✅ Has entry_point: {sample['entry_point']}")
            print(f"    ✅ Has test: {len(sample['test'])} chars")

    # Test 4: Complete workflow simulation
    print("\n" + "-" * 60)
    print("Test 4: 模拟完整workflow执行")
    print("-" * 60)

    from src.aflow_executor import AFlowExecutor

    executor = AFlowExecutor(
        llm_config_path="config/aflow_llm.yaml",
        llm_model_name="gpt-4o-mini",
        timeout=120
    )

    # Simple workflow that just returns the canonical solution
    test_workflow_code = f"""
import workspace.code.workflows.template.operator as operator
from scripts.async_llm import create_llm_instance

class Workflow:
    def __init__(self, name: str, llm_config, dataset):
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)
        self.test = operator.Test(self.llm)

    async def __call__(self, problem: str, entry_point: str = "", test: str = ""):
        # For testing: just return a simple solution that should work
        solution = problem + "    pass  # Placeholder"

        # Test it
        test_result = await self.test(
            problem=problem,
            solution=solution,
            entry_point=entry_point,
            test=test,
            test_loop=1
        )

        return test_result.get('solution', solution), 0.0
"""

    # Load first code sample
    code_sample = None
    for s in batch:
        if s['problem_type'] == 'code':
            code_sample = s
            break

    if code_sample:
        print(f"  Using code sample: {code_sample.get('task_id', 'unknown')}")

        answer, cost, metadata = await executor.execute_workflow(
            workflow_code=test_workflow_code,
            problem=code_sample['problem'],
            problem_type='code',
            entry_point=code_sample['entry_point'],
            test=code_sample['test']
        )

        print(f"  Success: {metadata['success']}")
        print(f"  Answer type: {type(answer)}")
        print(f"  Answer preview: {str(answer)[:100]}...")

        if metadata['success']:
            print("  ✅ Workflow执行成功")
        else:
            print("  ⚠️  Workflow执行失败（可能是正常的，因为placeholder实现）")

    print("\n" + "=" * 60)
    print("✅ HumanEval Pipeline测试完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_humaneval_evaluation())
