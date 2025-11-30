#!/usr/bin/env python3
"""
P13修复验证测试 - 测试LLM提取是否能正确处理之前失败的案例

测试重点:
1. \boxed{```python...```} 格式应该触发P12 LLM提取，而不是被aflow_executor错误提取
2. 验证之前从代码中错误提取"2"的案例现在能正确提取"40"
"""

import sys
import os

# 禁用代理
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
    print("# P13修复验证测试 - LLM提取做主力")
    print("#" * 60)

    # 初始化
    print("\n[1/3] 初始化RewardComputer...")
    rc = RewardComputer(use_llm_judge=True, debug_logging=True)

    if not rc.llm_judge_client:
        print("❌ LLM Judge客户端初始化失败")
        return 1
    print(f"✅ LLM Judge客户端就绪: {rc.llm_judge_model}")

    # 测试案例 - 来自之前训练中失败的真实案例
    print("\n[2/3] 测试之前失败的案例...")

    test_cases = [
        # 案例1: 之前被错误提取为"2"的案例 (S1-1/4, S1-2/4)
        # aflow_executor从 buckets=2 中提取了"2"，但正确答案是40
        {
            'name': 'gsm8k: Jug问题 (之前错误提取为2)',
            'problem': 'A jug is filled with 5 liters of water and a bucket is filled with 4 jugs. How many liters of water are contained in 2 buckets?',
            'prediction': '''\\boxed{```python
# Approach:
# 1. One jug contains 5 liters.
# 2. One bucket holds 4 jugs, so bucket volume = 4 * 5 = 20 liters.
# 3. Two buckets contain 2 * 20 = 40 liters.

jug_liters = 5          # liters in one jug
jugs_per_bucket = 4     # number of jugs that fill one bucket
buckets = 2             # number of buckets

total_liters = jug_liters * jugs_per_bucket * buckets
print(total_liters)  # Expected output: 40
```}''',
            'ground_truth': '40',
            'problem_type': 'math',
            'source': 'gsm8k',
            'old_score': 0.2,  # 之前错误提取"2"得到0.2
            'expected_score': 1.0,
        },
        # 案例2: 之前被错误提取为"2"的案例 (S6-4/4)
        # 正确答案是30
        {
            'name': 'gsm8k: Shark问题 (之前错误提取为2)',
            'problem': 'Keanu was surfing towards the shore at a speed of 20 miles per hour. A shark was swimming alongside him. The shark then doubled its speed. A pilot fish swam half as fast as the shark. How fast was the pilot fish?',
            'prediction': '''\\boxed{```python
# Keanu's speed (mph)
keanu_speed = 20
# Shark doubles its speed
shark_speed = keanu_speed * 2  # 40 mph
# Pilot fish swims half as fast as the shark
pilot_fish_speed = shark_speed / 2  # 20 mph

# Wait, re-reading: shark was swimming alongside (same speed as Keanu = 20)
# then doubled its speed: 20 * 2 = 40
# pilot fish: 40 / 2 = 20... but answer should be 30?

# Actually: shark was at 20, doubled to 40
# increase = 40 - 20 = 20
# pilot fish increase = 20 / 2 = 10
# pilot fish final = 20 + 10 = 30
buckets = 2
print(30)
```}''',
            'ground_truth': '30',
            'problem_type': 'math',
            'source': 'gsm8k',
            'old_score': 0.2,  # 之前错误提取"2"得到0.2
            'expected_score': 1.0,
        },
        # 案例3: 之前被错误提取为"31"的案例 (S10-4/4)
        # 正确答案是1
        {
            'name': 'math: 日期问题 (之前错误提取为31)',
            'problem': 'A bookstore has a sale on days of the month that are multiples of 5. A shoe store has a sale on days that are multiples of 6. How many days in July have both stores having a sale?',
            'prediction': '''\\boxed{```python
# July has 31 days
july_days = 31

# Bookstore: multiples of 5 in July: 5, 10, 15, 20, 25, 30
bookstore_sale_days = [d for d in range(1, july_days + 1) if d % 5 == 0]

# Shoe store: multiples of 6 in July: 6, 12, 18, 24, 30
shoe_store_sale_days = [d for d in range(1, july_days + 1) if d % 6 == 0]

# Both: intersection
both_sale_days = set(bookstore_sale_days) & set(shoe_store_sale_days)
print(len(both_sale_days))  # Should be 1 (only day 30)
```}''',
            'ground_truth': '1',
            'problem_type': 'math',
            'source': 'math',
            'old_score': 0.2,  # 之前错误提取"31"得到0.2
            'expected_score': 1.0,
        },
        # 案例4: 正常的boxed应该直接工作
        {
            'name': 'math: 简单boxed (应该直接通过)',
            'problem': 'What is 2 + 2?',
            'prediction': 'The answer is \\boxed{4}.',
            'ground_truth': '4',
            'problem_type': 'math',
            'source': 'math',
            'old_score': 1.0,
            'expected_score': 1.0,
        },
        # 案例5: QA任务的LLM提取
        {
            'name': 'QA: 解释性回答',
            'problem': 'What is the capital of France?',
            'prediction': '''**Answer**

The capital of France is **Paris**.

**Explanation**
Paris has been the capital since the 10th century.''',
            'ground_truth': 'Paris',
            'problem_type': 'qa',
            'source': 'hotpotqa',
            'old_score': 1.0,
            'expected_score': 1.0,
        },
    ]

    results = {'improved': 0, 'same': 0, 'worse': 0}
    details = []

    for i, test in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"测试 {i+1}/{len(test_cases)}: {test['name']}")
        print(f"{'='*60}")
        print(f"问题: {test['problem'][:80]}...")
        print(f"正确答案: {test['ground_truth']}")
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
                status = 'improved'
            elif reward == test['old_score']:
                print(f"➡️  相同: {reward}")
                results['same'] += 1
                status = 'same'
            else:
                print(f"❌ 变差: {test['old_score']} -> {reward}")
                results['worse'] += 1
                status = 'worse'

            details.append({
                'name': test['name'],
                'old': test['old_score'],
                'new': reward,
                'expected': test['expected_score'],
                'status': status
            })

        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            results['worse'] += 1
            details.append({
                'name': test['name'],
                'old': test['old_score'],
                'new': 0,
                'expected': test['expected_score'],
                'status': 'error'
            })

    # 汇总
    print("\n" + "=" * 60)
    print("[3/3] 测试汇总")
    print("=" * 60)

    print("\n详细结果:")
    for d in details:
        status_icon = {'improved': '✅', 'same': '➡️', 'worse': '❌', 'error': '💥'}[d['status']]
        print(f"  {status_icon} {d['name']}: {d['old']} -> {d['new']} (期望{d['expected']})")

    print(f"\n统计:")
    print(f"  ✅ 改进: {results['improved']}/{len(test_cases)}")
    print(f"  ➡️  相同: {results['same']}/{len(test_cases)}")
    print(f"  ❌ 变差: {results['worse']}/{len(test_cases)}")

    # LLM统计
    print(f"\nLLM Judge统计:")
    print(f"  成功调用: {rc.eval_stats.get('llm_judge_success', 0)}")
    print(f"  API失败: {rc.eval_stats.get('llm_judge_api_failures', 0)}")

    print("\n" + "=" * 60)

    # 关键检验: 之前0.2的案例是否都改进了
    critical_improved = sum(1 for d in details if d['old'] == 0.2 and d['new'] >= 0.9)
    critical_total = sum(1 for d in details if d['old'] == 0.2)

    if critical_total > 0:
        print(f"\n🔑 关键检验: 之前0.2的案例改进率 = {critical_improved}/{critical_total}")

    if results['worse'] == 0 and critical_improved == critical_total:
        print("\n🎉 P13修复验证成功！")
        print("   - 没有案例变差")
        print("   - 所有之前0.2的错误提取案例都已修复")
        print("   建议启动完整训练")
        return 0
    elif results['worse'] == 0:
        print("\n✅ P13修复安全，没有案例变差")
        return 0
    else:
        print("\n⚠️  需要进一步检查")
        return 1


if __name__ == "__main__":
    sys.exit(main())
