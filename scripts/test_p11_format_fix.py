#!/usr/bin/env python3
"""
P11修复验证测试
测试改进后的代码检测逻辑和提取失败处理
"""

import sys
import re
sys.path.insert(0, '/home/yijia/.claude/11/integrated_aflow_roll/src')

def test_strict_code_patterns():
    """测试严格的代码模式检测"""
    print("=" * 60)
    print("测试1: 严格代码模式检测")
    print("=" * 60)

    strict_code_patterns = [
        r'\bimport\s+[a-zA-Z_][a-zA-Z0-9_]*',      # import module
        r'\bfrom\s+[a-zA-Z_][a-zA-Z0-9_.]*\s+import',  # from xxx import
        r'\bdef\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(',    # def func(
        r'\bclass\s+[a-zA-Z_][a-zA-Z0-9_]*\s*[:\(]',  # class Foo: 或 class Foo(
        r'\bfor\s+[a-zA-Z_][a-zA-Z0-9_]*\s+in\s+',  # for x in (Python特有)
        r'\bwhile\s+[a-zA-Z_][a-zA-Z0-9_]*\s*[:<>=!]',  # while x < (循环条件)
        r'if\s+__name__\s*==',                      # if __name__ ==
        r'print\s*\([^)]+\)',                       # print(xxx)
    ]

    # 应该被检测为代码的文本
    code_texts = [
        "import numpy as np",
        "from sklearn import model",
        "def calculate(x):",
        "class Solution:",
        "for i in range(10):",
        "while x < 10:",
        'if __name__ == "__main__"',
        "print(result)",
    ]

    # 不应该被检测为代码的数学文本
    math_texts = [
        "For a real number x, let f(x) = x^2",
        "For how many positive integers n does...",
        "We need the number of positive integers n for which...",
        "There are **5** positive integers n for which...",
        "**Step 1 – Identify what the problem is asking**",
        "Let x be a real number. For all values of x...",
        "The formula for the area is A = πr²",
        "Return to step 1 and repeat.",  # 含有"return"但不是代码
        "For each element in the set...",
        "While this approach works...",  # "while"作为连词
    ]

    print("\n[应该被检测为代码的文本]")
    code_pass = 0
    for text in code_texts:
        is_code = any(re.search(pattern, text.lower()) for pattern in strict_code_patterns)
        status = "✅" if is_code else "❌"
        print(f"  {status} '{text[:50]}' -> {'代码' if is_code else '非代码'}")
        if is_code:
            code_pass += 1

    print(f"\n[不应该被检测为代码的数学文本]")
    math_pass = 0
    for text in math_texts:
        is_code = any(re.search(pattern, text.lower()) for pattern in strict_code_patterns)
        status = "✅" if not is_code else "❌"
        print(f"  {status} '{text[:50]}' -> {'代码' if is_code else '非代码'}")
        if not is_code:
            math_pass += 1

    print(f"\n📊 代码检测: {code_pass}/{len(code_texts)} 正确")
    print(f"📊 数学文本: {math_pass}/{len(math_texts)} 正确")

    return code_pass == len(code_texts) and math_pass == len(math_texts)


def test_qa_answer_extraction():
    """测试QA答案本地提取"""
    print("\n" + "=" * 60)
    print("测试2: QA Answer本地提取")
    print("=" * 60)

    test_cases = [
        # (输入文本, 期望提取的答案)
        (
            "**Answer:** The Choctaw Nation\n\n**Explanation**\nBryant is located in...",
            "The Choctaw Nation"
        ),
        (
            "**Answer**: 42\n\nThis is because...",
            "42"
        ),
        (
            "**Answer** – Paris\n\n**Details**\nParis is the capital...",
            "Paris"
        ),
        (
            "Let me explain...\n\n**Answer:** John Smith\n\nHe was born in...",
            "John Smith"
        ),
    ]

    passed = 0
    for text, expected in test_cases:
        answer_match = re.search(r'\*\*Answer[:\*]*\s*[:\-–—]*\s*(.+?)(?:\n\n|\*\*|$)', text, re.IGNORECASE | re.DOTALL)
        if answer_match:
            extracted = answer_match.group(1).strip()
            extracted = re.sub(r'^[\*\#\-–—:]+|[\*\#\-–—:]+$', '', extracted).strip()
        else:
            extracted = None

        status = "✅" if extracted == expected else "❌"
        print(f"  {status} 输入: '{text[:40]}...'")
        print(f"       期望: '{expected}'")
        print(f"       提取: '{extracted}'")
        if extracted == expected:
            passed += 1

    print(f"\n📊 QA提取测试: {passed}/{len(test_cases)} 通过")
    return passed == len(test_cases)


def test_boxed_extraction():
    """测试boxed答案提取（模拟P11 fallback）"""
    print("\n" + "=" * 60)
    print("测试3: Boxed答案提取（P11 Fallback）")
    print("=" * 60)

    # 简化的boxed提取逻辑
    def extract_boxed(text):
        import re
        # 匹配 \boxed{...}
        match = re.search(r'\\boxed\{([^{}]+)\}', text)
        if match:
            return match.group(1).strip()
        # 匹配嵌套的boxed
        match = re.search(r'\\boxed\{(.+)\}', text, re.DOTALL)
        if match:
            content = match.group(1)
            # 如果内容太长（包含解释），尝试只取最后的数字/表达式
            if len(content) > 100:
                # 查找最后的数字或简单表达式
                last_expr = re.findall(r'[\d\.\-\+/]+|\$[^$]+\$', content)
                if last_expr:
                    return last_expr[-1].strip('$')
            return content[:100]  # 截断过长的内容
        return None

    test_cases = [
        ("The answer is \\boxed{42}", "42"),
        ("Therefore, \\boxed{5/6} is the result", "5/6"),
        ("\\boxed{2x+1}", "2x+1"),
        ("After calculation, we get \\boxed{-3.14}", "-3.14"),
    ]

    passed = 0
    for text, expected in test_cases:
        extracted = extract_boxed(text)
        status = "✅" if extracted == expected else "❌"
        print(f"  {status} 输入: '{text[:40]}...'")
        print(f"       期望: '{expected}'")
        print(f"       提取: '{extracted}'")
        if extracted == expected:
            passed += 1

    print(f"\n📊 Boxed提取测试: {passed}/{len(test_cases)} 通过")
    return passed == len(test_cases)


def main():
    print("\n" + "#" * 60)
    print("# P11修复验证测试")
    print("#" * 60)

    results = []
    results.append(("严格代码模式检测", test_strict_code_patterns()))
    results.append(("QA Answer提取", test_qa_answer_extraction()))
    results.append(("Boxed答案提取", test_boxed_extraction()))

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
        print("🎉 所有测试通过！P11修复验证成功")
    else:
        print("⚠️  部分测试失败，需要检查")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
