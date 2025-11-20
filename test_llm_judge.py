#!/usr/bin/env python3
"""
测试LLM Judge功能
"""
import sys
sys.path.insert(0, '/home/yijia/.claude/11/integrated_aflow_roll/src')

from reward_computer import RewardComputer

def test_llm_judge():
    print("\n" + "=" * 60)
    print("🧪 测试LLM Judge (GPT OSS 120B @ port 8002)")
    print("=" * 60)

    # 初始化RewardComputer with LLM Judge
    print("\n🔧 初始化RewardComputer (LLM Judge模式)...")
    computer = RewardComputer(
        use_answer_extractor=False,  # 禁用答案提取器，直接测试LLM Judge
        use_llm_judge=True,
        llm_config={
            "base_url": "http://localhost:8002/v1",
            "api_key": "sk-dummy",
            "model_name": "/home/yijia/lhy/openai/gpt-oss-120b"  # 完整模型路径
        }
    )

    # 测试用例
    test_cases = [
        {
            "name": "数学 - 分数等价",
            "problem": "What is 1/2?",
            "prediction": "The answer is 0.5",
            "ground_truth": "1/2",
            "problem_type": "math",
            "expected": True
        },
        {
            "name": "数学 - 完全匹配",
            "problem": "What is 15 + 27?",
            "prediction": "<think>Let me calculate: 15 + 27 = 42</think><answer>42</answer>",
            "ground_truth": "42",
            "problem_type": "math",
            "expected": True
        },
        {
            "name": "数学 - 错误答案",
            "problem": "What is 15 + 27?",
            "prediction": "The answer is 50",
            "ground_truth": "42",
            "problem_type": "math",
            "expected": False
        },
        {
            "name": "QA - 语义等价",
            "problem": "What is the capital of France?",
            "prediction": "The capital of France is Paris.",
            "ground_truth": "Paris",
            "problem_type": "qa",
            "expected": True
        },
        {
            "name": "QA - 数值提取",
            "problem": "How many subscribers?",
            "prediction": "He makes $1,800 a month. He has 200 subscribers.",
            "ground_truth": "200",
            "problem_type": "qa",
            "expected": True
        },
        {
            "name": "数学 - 代数表达式",
            "problem": "Factor x^2 + x - 2",
            "prediction": "The factored form is (x+2)(x-1) or x^2+x-2",
            "ground_truth": "x^2+x-2",
            "problem_type": "math",
            "expected": True
        }
    ]

    print("\n" + "=" * 60)
    print("开始测试...")
    print("=" * 60)

    passed = 0
    failed = 0

    for i, case in enumerate(test_cases, 1):
        print(f"\n📝 测试 {i}/{len(test_cases)}: {case['name']}")
        print(f"  问题: {case['problem']}")
        print(f"  预测: {case['prediction'][:60]}...")
        print(f"  真值: {case['ground_truth']}")

        try:
            # 调用LLM Judge
            is_correct = computer._llm_judge_compare(
                problem=case['problem'],
                prediction=case['prediction'],
                ground_truth=case['ground_truth'],
                problem_type=case['problem_type']
            )

            print(f"  判决: {is_correct}")
            print(f"  期望: {case['expected']}")

            if is_correct == case['expected']:
                print(f"  ✅ 通过")
                passed += 1
            else:
                print(f"  ❌ 失败")
                failed += 1

        except Exception as e:
            print(f"  ❌ 错误: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    print(f"  通过: {passed}/{len(test_cases)}")
    print(f"  失败: {failed}/{len(test_cases)}")
    print(f"  准确率: {passed/len(test_cases)*100:.1f}%")

    if passed == len(test_cases):
        print("\n🎉 所有测试通过！LLM Judge工作正常。")
        return True
    else:
        print(f"\n⚠️  {failed}个测试失败，请检查。")
        return False


if __name__ == "__main__":
    success = test_llm_judge()
    sys.exit(0 if success else 1)
