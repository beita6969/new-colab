#!/usr/bin/env python3
"""
P12小规模测试 - 直接测试LLM提取 (8002端口OSS模型)
"""

import sys
import os

# 禁用代理，确保直连localhost:8002
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ['no_proxy'] = 'localhost,127.0.0.1'

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
sys.path.insert(0, '/home/yijia/.claude/11/integrated_aflow_roll/src')

from reward_computer import RewardComputer


def main():
    print("\n" + "#" * 60)
    print("# P12小规模测试 - LLM提取验证 (8002端口OSS)")
    print("#" * 60)

    # 初始化
    print("\n[1/2] 初始化RewardComputer...")
    rc = RewardComputer(use_llm_judge=True, debug_logging=True)

    if not rc.llm_judge_client:
        print("❌ LLM Judge客户端初始化失败")
        return 1
    print(f"✅ LLM Judge客户端就绪: {rc.llm_judge_model}")

    # 测试案例 - 来自真实训练日志
    print("\n[2/2] 测试LLM提取效果...")

    test_cases = [
        # 案例1: 包含Python代码的数学解答 (之前返回0.2)
        {
            'name': 'Math+Python代码',
            'problem': 'In a glee club, there are two times as many female than male members. How many female members are there if there are 18 members in the club?',
            'prediction': '''**Answer**
There are **12 female members**.

**Python code to compute the answer**

```python
total_members = 18
male_members = total_members // 3
female_members = 2 * male_members
print(female_members)  # Output: 12
```''',
            'ground_truth': '12',
            'problem_type': 'math',
            'source': 'gsm8k',
            'old_score': 0.2,
            'expected_score': 1.0,
        },
        # 案例2: Step-by-step解答
        {
            'name': 'Math Step解答',
            'problem': 'Two identical CDs regularly cost a total of $28. What is the cost in dollars of five of these CDs?',
            'prediction': '''**Step 1 – Identify what the problem is asking**
We need to find the cost of five CDs.

**Step 2 – Calculate**
Two CDs cost $28, so one CD costs $14.
Five CDs cost 5 × $14 = $70.

Therefore, the answer is **70**.

\\boxed{70}''',
            'ground_truth': '70',
            'problem_type': 'math',
            'source': 'math',
            'old_score': 0.2,
            'expected_score': 1.0,
        },
        # 案例3: Therefore结论
        {
            'name': 'Therefore结论',
            'problem': 'What is 15 + 27?',
            'prediction': 'Let me calculate: 15 + 27 = 42. Therefore, the answer is 42.',
            'ground_truth': '42',
            'problem_type': 'math',
            'source': 'math',
            'old_score': 0.2,
            'expected_score': 1.0,
        },
        # 案例4: QA解释性回答
        {
            'name': 'QA解释性回答',
            'problem': 'What is the capital of France?',
            'prediction': '''**Answer**

The capital of France is **Paris**.

**Explanation**
Paris has been the capital since the 10th century and is the political, economic, and cultural center of France.''',
            'ground_truth': 'Paris',
            'problem_type': 'qa',
            'source': 'hotpotqa',
            'old_score': 0.2,
            'expected_score': 1.0,
        },
    ]

    results = {'improved': 0, 'same': 0, 'worse': 0}

    for i, test in enumerate(test_cases):
        print(f"\n{'='*55}")
        print(f"测试 {i+1}/{len(test_cases)}: {test['name']}")
        print(f"{'='*55}")
        print(f"问题: {test['problem'][:60]}...")
        print(f"答案: {test['ground_truth']}")
        print(f"预测: {test['prediction'][:80]}...")
        print(f"旧分数: {test['old_score']} -> 期望: {test['expected_score']}")
        print()

        try:
            if test['problem_type'] == 'math':
                reward = rc._compute_math_reward(
                    problem=test['problem'],
                    prediction=test['prediction'],
                    ground_truth=test['ground_truth'],
                    source=test['source']
                )
            else:
                reward = rc._compute_qa_reward(
                    problem=test['problem'],
                    prediction=test['prediction'],
                    ground_truth=test['ground_truth'],
                    source=test.get('source', 'hotpotqa')
                )

            print(f"\n新分数: {reward}")

            if reward > test['old_score']:
                print(f"✅ 改进! {test['old_score']} -> {reward}")
                results['improved'] += 1
            elif reward == test['old_score']:
                print(f"➡️  相同: {reward}")
                results['same'] += 1
            else:
                print(f"❌ 变差: {test['old_score']} -> {reward}")
                results['worse'] += 1

        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            results['worse'] += 1

    # 汇总
    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)
    print(f"  ✅ 改进: {results['improved']}/{len(test_cases)}")
    print(f"  ➡️  相同: {results['same']}/{len(test_cases)}")
    print(f"  ❌ 变差: {results['worse']}/{len(test_cases)}")

    # LLM统计
    print(f"\nLLM Judge统计:")
    print(f"  成功调用: {rc.eval_stats.get('llm_judge_success', 0)}")
    print(f"  API失败: {rc.eval_stats.get('llm_judge_api_failures', 0)}")

    print("\n" + "=" * 60)
    if results['improved'] >= len(test_cases) * 0.75:  # 75%改进
        print("🎉 P12修复有效！大部分案例得到改进")
        print("   建议启动完整训练验证")
        return 0
    elif results['worse'] == 0:
        print("✅ P12修复安全，没有案例变差")
        return 0
    else:
        print("⚠️  需要进一步检查")
        return 1


if __name__ == "__main__":
    sys.exit(main())
