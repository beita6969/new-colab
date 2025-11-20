#!/usr/bin/env python3
"""
简单测试HumanEval Test operator
"""
import json
import sys

# Add AFlow to path
sys.path.insert(0, '/home/yijia/.claude/11/AFlow')

from scripts import operators as operator_module


def test_humaneval_test_operator():
    """测试Test operator的HumanEval评估功能"""
    print("\n" + "=" * 60)
    print("🧪 测试HumanEval Test Operator")
    print("=" * 60)

    # Load a sample from HumanEval
    with open('data/humaneval/humaneval_full.jsonl', 'r') as f:
        sample = json.loads(f.readline())

    print(f"\n📝 测试样本: {sample['task_id']}")
    print(f"  Entry point: {sample['entry_point']}")
    print(f"  Prompt: {sample['prompt'][:100]}...")

    # Create Test operator (without LLM for basic testing)
    class MockLLM:
        pass

    test_op = operator_module.Test(MockLLM())

    # Test 1: Correct solution
    print("\n" + "-" * 60)
    print("Test 1: 正确解决方案应该通过测试")
    print("-" * 60)

    correct_solution = sample['prompt'] + sample['canonical_solution']
    print(f"  Solution length: {len(correct_solution)} chars")

    result = test_op.exec_code_humaneval(
        solution=correct_solution,
        test=sample['test'],
        entry_point=sample['entry_point']
    )

    print(f"  Result: {result}")

    if result == "no error":
        print("  ✅ Test 1 PASSED: 正确解决方案通过测试")
        test1_pass = True
    else:
        print(f"  ❌ Test 1 FAILED: {result}")
        test1_pass = False

    # Test 2: Incorrect solution
    print("\n" + "-" * 60)
    print("Test 2: 错误解决方案应该失败")
    print("-" * 60)

    incorrect_solution = sample['prompt'] + "    return False  # Wrong"
    print(f"  Solution length: {len(incorrect_solution)} chars")

    result = test_op.exec_code_humaneval(
        solution=incorrect_solution,
        test=sample['test'],
        entry_point=sample['entry_point']
    )

    print(f"  Result type: {type(result)}")

    if result != "no error":
        print("  ✅ Test 2 PASSED: 错误解决方案被正确检测")
        test2_pass = True
    else:
        print("  ❌ Test 2 FAILED: 错误解决方案没有被检测到")
        test2_pass = False

    # Test 3: DataManager loading
    print("\n" + "-" * 60)
    print("Test 3: DataManager加载HumanEval数据")
    print("-" * 60)

    from src.data_manager import DataManager

    dm = DataManager(data_dir="data")
    dm.initialize()

    batch = dm.sample_batch(batch_size=8, split="train")
    print(f"\n  Batch size: {len(batch)}")

    code_count = 0
    for sample in batch:
        if sample['problem_type'] == 'code':
            code_count += 1
            has_entry_point = 'entry_point' in sample
            has_test = 'test' in sample and len(sample['test']) > 0
            print(f"    Code sample: entry_point={has_entry_point}, test={has_test}")

            if not (has_entry_point and has_test):
                print(f"    ❌ Missing required fields!")
                test3_pass = False
                break
    else:
        if code_count > 0:
            print(f"  ✅ Test 3 PASSED: {code_count} code samples with proper format")
            test3_pass = True
        else:
            print("  ⚠️  Test 3 SKIPPED: No code samples in batch")
            test3_pass = True

    # Summary
    print("\n" + "=" * 60)
    print("📊 测试总结")
    print("=" * 60)
    print(f"  Test 1 (Correct solution):   {'✅ PASSED' if test1_pass else '❌ FAILED'}")
    print(f"  Test 2 (Incorrect solution): {'✅ PASSED' if test2_pass else '❌ FAILED'}")
    print(f"  Test 3 (DataManager):        {'✅ PASSED' if test3_pass else '❌ FAILED'}")

    all_pass = test1_pass and test2_pass and test3_pass
    print(f"\n{'✅ 所有测试通过!' if all_pass else '❌ 有测试失败'}")
    print("=" * 60)

    return all_pass


if __name__ == "__main__":
    success = test_humaneval_test_operator()
    sys.exit(0 if success else 1)
