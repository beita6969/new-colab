#!/usr/bin/env python3
"""
小样本测试脚本 - 验证reward_computer修复的正确性

测试范围:
1. MBPP entry_point自动推断
2. Math空boxed评分修正
3. QA任务评分
4. HumanEval代码评分
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from src.reward_computer import RewardComputer


def test_mbpp_entry_point_inference():
    """测试MBPP entry_point自动推断功能"""
    print("\n" + "="*60)
    print("测试1: MBPP entry_point自动推断")
    print("="*60)

    rc = RewardComputer(use_llm_judge=False, debug_logging=True)

    # 测试用例1: MBPP样本 - 从test中的assert推断
    mbpp_solution_1 = '''def unique_Element(arr):
    s = set(arr)
    return len(s) == 1
'''
    mbpp_test_1 = '''assert unique_Element([1,1,1]) == True
assert unique_Element([1,2,1,2]) == False
assert unique_Element([1,2,3,4,5]) == False'''

    print("\n测试1.1: MBPP正确代码 + 从assert推断entry_point")
    print(f"代码: {mbpp_solution_1[:50]}...")
    print(f"测试: {mbpp_test_1[:50]}...")

    score = rc.compute_reward(
        problem="Write a python function to check whether a list of numbers contains only one distinct element or not.",
        prediction=mbpp_solution_1,
        ground_truth="def unique_Element(arr): ...",
        problem_type="code",
        test=mbpp_test_1,
        entry_point="",  # 空entry_point，需要自动推断
        source="mbpp"
    )
    print(f"评分结果: {score}")
    assert score >= 0.9, f"MBPP正确代码应得高分，实际得分: {score}"
    print("✅ 通过: MBPP正确代码得分 >= 0.9")

    # 测试用例2: MBPP错误代码
    mbpp_wrong_solution = '''def unique_Element(arr):
    return True  # 错误实现
'''
    print("\n测试1.2: MBPP错误代码")
    score_wrong = rc.compute_reward(
        problem="Write a python function to check whether a list of numbers contains only one distinct element or not.",
        prediction=mbpp_wrong_solution,
        ground_truth="def unique_Element(arr): ...",
        problem_type="code",
        test=mbpp_test_1,
        entry_point="",
        source="mbpp"
    )
    print(f"评分结果: {score_wrong}")
    assert score_wrong < 0.5, f"MBPP错误代码应得低分，实际得分: {score_wrong}"
    print("✅ 通过: MBPP错误代码得分 < 0.5")

    # 测试用例3: 从solution中的def推断
    mbpp_solution_3 = '''def noprofit_noloss(actual_cost, sale_amount):
    if sale_amount == actual_cost:
        return True
    else:
        return False
'''
    mbpp_test_3 = '''assert noprofit_noloss(1500,1200)==False
assert noprofit_noloss(100,100)==True
assert noprofit_noloss(2000,5000)==False'''

    print("\n测试1.3: 从solution的def推断entry_point")
    score_3 = rc.compute_reward(
        problem="Write a function to check whether the given amount has no profit and no loss",
        prediction=mbpp_solution_3,
        ground_truth="...",
        problem_type="code",
        test=mbpp_test_3,
        entry_point="",
        source="mbpp"
    )
    print(f"评分结果: {score_3}")
    assert score_3 >= 0.9, f"应得高分，实际: {score_3}"
    print("✅ 通过: 从solution推断entry_point成功")

    return True


def test_math_empty_boxed():
    """测试Math空boxed评分修正"""
    print("\n" + "="*60)
    print("测试2: Math空boxed评分修正")
    print("="*60)

    rc = RewardComputer(use_llm_judge=False, debug_logging=True)

    # 测试用例1: 空boxed
    print("\n测试2.1: 空\\boxed{}应得0分")
    score_empty = rc.compute_reward(
        problem="What is 2 + 2?",
        prediction="Let me think... \\boxed{}",
        ground_truth="\\boxed{4}",
        problem_type="math",
        source="math"
    )
    print(f"评分结果: {score_empty}")
    assert score_empty == 0.0, f"空boxed应得0分，实际: {score_empty}"
    print("✅ 通过: 空boxed得0分")

    # 测试用例2: 正确答案
    print("\n测试2.2: 正确答案应得1.0分")
    score_correct = rc.compute_reward(
        problem="What is 2 + 2?",
        prediction="The answer is \\boxed{4}",
        ground_truth="\\boxed{4}",
        problem_type="math",
        source="math"
    )
    print(f"评分结果: {score_correct}")
    assert score_correct == 1.0, f"正确答案应得1.0分，实际: {score_correct}"
    print("✅ 通过: 正确答案得1.0分")

    # 测试用例3: 接近但不完全正确
    print("\n测试2.3: 数值接近答案")
    score_close = rc.compute_reward(
        problem="Calculate 100/3",
        prediction="\\boxed{33.33}",
        ground_truth="\\boxed{33.333333}",
        problem_type="math",
        source="math"
    )
    print(f"评分结果: {score_close}")
    assert score_close >= 0.7, f"接近答案应得高分，实际: {score_close}"
    print("✅ 通过: 接近答案得高分")

    # 测试用例4: 完全错误
    print("\n测试2.4: 完全错误答案")
    score_wrong = rc.compute_reward(
        problem="What is 2 + 2?",
        prediction="\\boxed{100}",
        ground_truth="\\boxed{4}",
        problem_type="math",
        source="math"
    )
    print(f"评分结果: {score_wrong}")
    assert score_wrong <= 0.4, f"错误答案应得低分，实际: {score_wrong}"
    print("✅ 通过: 错误答案得低分")

    # 测试用例5: 只有空格的boxed
    print("\n测试2.5: 只有空格的boxed{}应得0分")
    score_space = rc.compute_reward(
        problem="What is 2 + 2?",
        prediction="\\boxed{   }",
        ground_truth="\\boxed{4}",
        problem_type="math",
        source="math"
    )
    print(f"评分结果: {score_space}")
    assert score_space == 0.0, f"空格boxed应得0分，实际: {score_space}"
    print("✅ 通过: 空格boxed得0分")

    return True


def test_qa_scoring():
    """测试QA任务评分"""
    print("\n" + "="*60)
    print("测试3: QA任务评分")
    print("="*60)

    rc = RewardComputer(use_llm_judge=False, debug_logging=True)

    # 测试用例1: 完全匹配
    print("\n测试3.1: 完全匹配")
    score_exact = rc.compute_reward(
        problem="Who wrote Romeo and Juliet?",
        prediction="William Shakespeare",
        ground_truth="William Shakespeare",
        problem_type="qa",
        source="hotpotqa"
    )
    print(f"评分结果: {score_exact}")
    assert score_exact == 1.0, f"完全匹配应得1.0分，实际: {score_exact}"
    print("✅ 通过: 完全匹配得1.0分")

    # 测试用例2: 大小写不敏感匹配
    print("\n测试3.2: 大小写不敏感匹配")
    score_case = rc.compute_reward(
        problem="Who wrote Romeo and Juliet?",
        prediction="william shakespeare",
        ground_truth="William Shakespeare",
        problem_type="qa",
        source="hotpotqa"
    )
    print(f"评分结果: {score_case}")
    assert score_case >= 0.7, f"大小写不敏感应得高分，实际: {score_case}"
    print("✅ 通过: 大小写不敏感得高分")

    # 测试用例3: 部分匹配
    print("\n测试3.3: 部分匹配")
    score_partial = rc.compute_reward(
        problem="Who wrote Romeo and Juliet?",
        prediction="Shakespeare wrote it",
        ground_truth="William Shakespeare",
        problem_type="qa",
        source="hotpotqa"
    )
    print(f"评分结果: {score_partial}")
    # 部分匹配应该有一定分数
    print(f"部分匹配得分: {score_partial}")

    # 测试用例4: 完全错误
    print("\n测试3.4: 完全错误")
    score_wrong = rc.compute_reward(
        problem="Who wrote Romeo and Juliet?",
        prediction="Charles Dickens",
        ground_truth="William Shakespeare",
        problem_type="qa",
        source="hotpotqa"
    )
    print(f"评分结果: {score_wrong}")
    assert score_wrong <= 0.4, f"完全错误应得低分，实际: {score_wrong}"
    print("✅ 通过: 完全错误得低分")

    # 测试用例5: Yes/No问题
    print("\n测试3.5: Yes/No问题")
    score_yesno = rc.compute_reward(
        problem="Are both Print and National Journal periodicals?",
        prediction="no",
        ground_truth="no",
        problem_type="qa",
        source="hotpotqa"
    )
    print(f"评分结果: {score_yesno}")
    assert score_yesno >= 0.9, f"Yes/No正确应得高分，实际: {score_yesno}"
    print("✅ 通过: Yes/No正确得高分")

    return True


def test_humaneval_code():
    """测试HumanEval代码评分(有entry_point的情况)"""
    print("\n" + "="*60)
    print("测试4: HumanEval代码评分")
    print("="*60)

    rc = RewardComputer(use_llm_judge=False, debug_logging=True)

    # 测试用例1: 正确代码
    print("\n测试4.1: HumanEval正确代码")
    humaneval_solution = '''def fib4(n: int):
    results = [0, 0, 2, 0]
    if n < 4:
        return results[n]
    for _ in range(4, n + 1):
        results.append(results[-1] + results[-2] + results[-3] + results[-4])
        results.pop(0)
    return results[-1]
'''
    humaneval_test = '''
def check(candidate):
    assert candidate(5) == 4
    assert candidate(8) == 28
    assert candidate(10) == 104
    assert candidate(12) == 386
'''

    score = rc.compute_reward(
        problem="def fib4(n: int): ...",
        prediction=humaneval_solution,
        ground_truth="...",
        problem_type="code",
        test=humaneval_test,
        entry_point="fib4",  # HumanEval有entry_point
        source="humaneval"
    )
    print(f"评分结果: {score}")
    assert score >= 0.9, f"HumanEval正确代码应得高分，实际: {score}"
    print("✅ 通过: HumanEval正确代码得高分")

    # 测试用例2: 错误代码
    print("\n测试4.2: HumanEval错误代码")
    wrong_solution = '''def fib4(n: int):
    return n  # 错误实现
'''
    score_wrong = rc.compute_reward(
        problem="def fib4(n: int): ...",
        prediction=wrong_solution,
        ground_truth="...",
        problem_type="code",
        test=humaneval_test,
        entry_point="fib4",
        source="humaneval"
    )
    print(f"评分结果: {score_wrong}")
    assert score_wrong < 0.5, f"错误代码应得低分，实际: {score_wrong}"
    print("✅ 通过: HumanEval错误代码得低分")

    return True


def test_entry_point_inference_function():
    """测试_infer_entry_point函数本身"""
    print("\n" + "="*60)
    print("测试5: _infer_entry_point函数")
    print("="*60)

    rc = RewardComputer(use_llm_judge=False)

    # 测试1: 从test中的assert推断
    print("\n测试5.1: 从assert推断")
    test_code = "assert unique_Element([1,1,1]) == True"
    solution = "def unique_Element(arr): return len(set(arr)) == 1"
    result = rc._infer_entry_point(solution, test_code)
    print(f"推断结果: {result}")
    assert result == "unique_Element", f"应推断为unique_Element，实际: {result}"
    print("✅ 通过")

    # 测试2: 从solution的def推断(排除solve)
    print("\n测试5.2: 从solution推断(排除solve)")
    solution_with_solve = '''def solve():
    pass

def my_function(x):
    return x * 2
'''
    result2 = rc._infer_entry_point(solution_with_solve, None)
    print(f"推断结果: {result2}")
    assert result2 == "my_function", f"应推断为my_function(跳过solve)，实际: {result2}"
    print("✅ 通过")

    # 测试3: 只有solve时返回solve
    print("\n测试5.3: 只有solve时返回solve")
    solution_only_solve = '''def solve():
    return 42
'''
    result3 = rc._infer_entry_point(solution_only_solve, None)
    print(f"推断结果: {result3}")
    assert result3 == "solve", f"应推断为solve，实际: {result3}"
    print("✅ 通过")

    return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "#"*60)
    print("# RewardComputer 修复验证测试")
    print("#"*60)

    tests = [
        ("MBPP entry_point推断", test_mbpp_entry_point_inference),
        ("Math空boxed评分", test_math_empty_boxed),
        ("QA任务评分", test_qa_scoring),
        ("HumanEval代码评分", test_humaneval_code),
        ("_infer_entry_point函数", test_entry_point_inference_function),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, "✅ 通过", None))
        except AssertionError as e:
            results.append((name, "❌ 失败", str(e)))
        except Exception as e:
            results.append((name, "💥 异常", str(e)))

    # 打印汇总
    print("\n" + "="*60)
    print("测试汇总")
    print("="*60)

    passed = 0
    failed = 0
    for name, status, error in results:
        print(f"{status} {name}")
        if error:
            print(f"   └── {error}")
        if "通过" in status:
            passed += 1
        else:
            failed += 1

    print(f"\n总计: {passed} 通过, {failed} 失败")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
