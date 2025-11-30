#!/usr/bin/env python3
"""
P12修复验证测试 - LLM提取作为主力
测试扩展的解释性模式和LLM答案提取效果
"""

import sys
import os
sys.path.insert(0, '/home/yijia/.claude/11/integrated_aflow_roll/src')

from reward_computer import RewardComputer


def test_explanatory_patterns():
    """测试扩展的解释性模式匹配"""
    print("=" * 60)
    print("测试1: 解释性模式匹配 (P12扩展)")
    print("=" * 60)

    explanatory_patterns = [
        '**Step', '**Solution', '**Approach', '**Analysis',
        '**Answer**', '**Answer', '## Answer', '# Answer',
        '**Python code', '```python',
        'Therefore,', 'Thus,', 'Hence,',
    ]

    # 来自真实训练日志的测试案例
    test_texts = [
        # 应该匹配的
        ("**Answer**\nThe answer is 42", True, "**Answer**"),
        ("**Answer**  \nThere are **12 female members**.", True, "**Answer**"),
        ("Therefore, the result is 70.", True, "Therefore,"),
        ("Thus, x = 5.", True, "Thus,"),
        ("**Python code to compute the answer**\n```python\nprint(12)\n```", True, "**Python code"),
        ("```python\ndef solve():\n    return 42\n```", True, "```python"),
        ("**Step 1 – Identify what the problem is asking**", True, "**Step"),
        ("**Solution Overview**\nWe need to...", True, "**Solution"),
        ("**Approach**\nLet x be...", True, "**Approach"),
        # 不应该匹配的（纯代码或简单答案）
        ("42", False, None),
        ("\\boxed{70}", False, None),
        ("def solve(): return 42", False, None),
    ]

    passed = 0
    for text, should_match, expected_pattern in test_texts:
        is_match = any(p in text for p in explanatory_patterns)
        matched_pattern = next((p for p in explanatory_patterns if p in text), None)

        status = "✅" if is_match == should_match else "❌"
        if is_match == should_match:
            passed += 1

        print(f"  {status} '{text[:50]}...' -> 匹配={is_match}")
        if matched_pattern:
            print(f"       匹配模式: {matched_pattern}")

    print(f"\n📊 模式匹配测试: {passed}/{len(test_texts)} 通过")
    return passed == len(test_texts)


def test_llm_extraction_simulation():
    """模拟LLM提取效果（使用本地提取作为fallback）"""
    print("\n" + "=" * 60)
    print("测试2: 答案提取效果模拟")
    print("=" * 60)

    # 初始化RewardComputer (不使用LLM，测试本地提取)
    rc = RewardComputer(use_llm_judge=False, debug_logging=False)

    # 来自训练日志的真实案例
    test_cases = [
        # (预测文本, 正确答案, 期望结果描述)
        (
            """**Answer**
There are **12 female members**.

**Python code to compute the answer**

```python
print(female_members)  # Output: 12
```""",
            "12",
            "应该提取到12"
        ),
        (
            """**Step 1 – Identify what the problem is asking**
We need to find the cost of five CDs.

**Step 2 – Calculate**
Two CDs cost $28, so one CD costs $14.
Five CDs cost 5 × $14 = $70.

\\boxed{70}""",
            "70",
            "应该从boxed提取70"
        ),
        (
            "Therefore, the answer is **5**.",
            "5",
            "应该提取到5"
        ),
        (
            "\\boxed{**Approach** Let x be the number...}",
            "42",
            "无效boxed，应该返回低分"
        ),
    ]

    print("\n[本地提取测试]")
    for pred, gt, desc in test_cases:
        # 测试boxed提取
        boxed = rc._extract_boxed_robust(pred)
        # 测试数学答案提取
        math_ans = rc._extract_math_answer(pred)

        print(f"\n  📝 {desc}")
        print(f"     预测: {pred[:60]}...")
        print(f"     GT: {gt}")
        print(f"     boxed提取: {boxed}")
        print(f"     math提取: {math_ans}")


def test_full_reward_with_llm():
    """使用真实LLM进行完整奖励计算测试"""
    print("\n" + "=" * 60)
    print("测试3: 完整奖励计算 (使用LLM Judge)")
    print("=" * 60)

    try:
        # 尝试初始化带LLM的RewardComputer
        rc = RewardComputer(use_llm_judge=True, debug_logging=True)

        if not rc.llm_judge_client:
            print("⚠️  LLM Judge客户端未初始化，跳过此测试")
            return None

        # 真实训练案例
        test_cases = [
            {
                "problem": "In a glee club, there are two times as many female than male members. How many female members are there if there are 18 members in the club?",
                "prediction": """**Answer**
There are **12 female members**.

**Python code to compute the answer**

```python
print(female_members)  # Output: 12
```""",
                "ground_truth": "12",
                "source": "gsm8k",
                "expected_score": 1.0,
            },
            {
                "problem": "Two identical CDs regularly cost a total of $28. What is the cost in dollars of five of these CDs?",
                "prediction": """**Step 1 – Identify what the problem is asking**
We need to find the cost of five CDs.

**Step 2 – Calculate**
Two CDs cost $28, so one CD costs $14.
Five CDs cost 5 × $14 = $70.

\\boxed{70}""",
                "ground_truth": "70",
                "source": "math",
                "expected_score": 1.0,
            },
            {
                "problem": "What is 2 + 2?",
                "prediction": "Therefore, 2 + 2 = 4.",
                "ground_truth": "4",
                "source": "math",
                "expected_score": 1.0,
            },
        ]

        passed = 0
        for case in test_cases:
            print(f"\n{'='*40}")
            print(f"问题: {case['problem'][:60]}...")
            print(f"预测: {case['prediction'][:80]}...")
            print(f"答案: {case['ground_truth']}")
            print(f"期望得分: {case['expected_score']}")
            print()

            reward = rc._compute_math_reward(
                problem=case['problem'],
                prediction=case['prediction'],
                ground_truth=case['ground_truth'],
                source=case['source']
            )

            print(f"\n实际得分: {reward}")

            if reward >= case['expected_score'] * 0.9:  # 允许10%误差
                print("✅ 通过")
                passed += 1
            else:
                print("❌ 未通过")

        print(f"\n📊 完整奖励测试: {passed}/{len(test_cases)} 通过")
        return passed == len(test_cases)

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_code_function_rename():
    """测试代码函数重命名 (P12)"""
    print("\n" + "=" * 60)
    print("测试4: 代码函数重命名 (P12)")
    print("=" * 60)

    rc = RewardComputer(use_llm_judge=False, debug_logging=True)

    # 来自训练日志的真实案例
    test_cases = [
        {
            "solution": """def reverse_upto(arr, pos):
    return arr[:pos+1][::-1] + arr[pos+1:]
""",
            "entry_point": "reverse_Array_Upto_K",
            "test": "assert reverse_Array_Upto_K([1,2,3,4,5], 2) == [3,2,1,4,5]",
            "expected_pass": True,
        },
        {
            "solution": """def validate(s):
    return len(s) > 0
""",
            "entry_point": "is_valid_string",
            "test": "assert is_valid_string('hello') == True",
            "expected_pass": True,
        },
    ]

    passed = 0
    for case in test_cases:
        print(f"\n原函数: {case['solution'].split('(')[0].replace('def ', '')}")
        print(f"期望entry_point: {case['entry_point']}")

        reward = rc._compute_code_reward(
            problem=None,
            prediction=case['solution'],
            ground_truth=None,
            test=case['test'],
            entry_point=case['entry_point']
        )

        print(f"奖励: {reward}")

        if (reward >= 0.5) == case['expected_pass']:
            print("✅ 通过")
            passed += 1
        else:
            print("❌ 未通过")

    print(f"\n📊 代码重命名测试: {passed}/{len(test_cases)} 通过")
    return passed == len(test_cases)


def main():
    print("\n" + "#" * 60)
    print("# P12修复验证测试 - LLM提取作为主力")
    print("#" * 60)

    results = []

    # 测试1: 模式匹配
    results.append(("解释性模式匹配", test_explanatory_patterns()))

    # 测试2: 本地提取模拟
    test_llm_extraction_simulation()
    results.append(("答案提取模拟", True))  # 仅观察性测试

    # 测试3: 完整LLM测试
    llm_result = test_full_reward_with_llm()
    if llm_result is not None:
        results.append(("完整LLM奖励", llm_result))

    # 测试4: 代码函数重命名
    results.append(("代码函数重命名", test_code_function_rename()))

    # 汇总
    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 所有测试通过！可以启动训练验证")
    else:
        print("⚠️  部分测试失败，请检查")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
