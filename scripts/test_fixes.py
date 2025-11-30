#!/usr/bin/env python3
"""测试所有关键修复是否正常工作"""
import sys
sys.path.insert(0, '/home/yijia/.claude/11/integrated_aflow_roll/src')

def test_invalid_boxed_detection():
    """测试无效boxed检测（修复后不应该过于激进）"""
    import re

    # 真正无效的模式
    truly_invalid_patterns = [
        (r'\\boxed\{\s*\}', '空boxed'),
        (r'\\boxed\{```python[\s\S]*```\s*\}', '完整代码块boxed'),
    ]

    # 测试用例
    test_cases = [
        # (输入, 应该是有效的?)
        (r'\boxed{8}', True),           # 数字，有效
        (r'\boxed{36}', True),          # 数字，有效
        (r'\boxed{**Answer** 8}', True),  # 包含Markdown，但答案有效（修复后）
        (r'\boxed{}', False),           # 空，无效
        (r'\boxed{   }', False),        # 只有空格，无效
        (r'\boxed{```python\ndef solve():\n    pass\n```}', False),  # 代码块，无效
    ]

    print("🔍 测试无效boxed检测:")
    all_passed = True
    for answer, should_be_valid in test_cases:
        is_invalid = False
        for pattern, desc in truly_invalid_patterns:
            if re.search(pattern, answer):
                is_invalid = True
                break

        detected_valid = not is_invalid
        status = "✅" if detected_valid == should_be_valid else "❌"
        print(f"  {status} '{answer[:40]}...' -> 检测为{'有效' if detected_valid else '无效'}，期望{'有效' if should_be_valid else '无效'}")
        if detected_valid != should_be_valid:
            all_passed = False

    return all_passed

def test_answer_extraction():
    """测试答案提取辅助函数"""
    import re

    def extract_valid_answer_from_text(text):
        # 尝试提取boxed中的内容
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
        if boxed_match:
            content = boxed_match.group(1).strip()
            content = re.sub(r'\*\*([^*]+)\*\*', r'\1', content)
            content = content.strip()
            # 如果内容是纯数字，直接返回
            if content and re.match(r'^-?\d+(?:\.\d+)?$', content):
                return content
            # 如果内容不是代码，尝试从中提取数字
            if content and not any(x in content for x in ['```', 'def ', 'import ', 'class ']):
                numbers = re.findall(r'-?\d+(?:\.\d+)?', content)
                if numbers:
                    return numbers[-1]  # 返回最后一个数字
                return content

        # 尝试提取最后一个数字
        numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
        if numbers:
            return numbers[-1]

        return None

    test_cases = [
        (r'\boxed{**Approach** The answer is 8}', '8'),  # 应该提取数字8
        (r'\boxed{36}', '36'),
        (r'The final answer is \boxed{42}', '42'),
        (r'Solution: The result is 100', '100'),
    ]

    print("\n🔍 测试答案提取:")
    all_passed = True
    for text, expected in test_cases:
        result = extract_valid_answer_from_text(text)
        status = "✅" if result == expected else "❌"
        print(f"  {status} '{text[:40]}...' -> 提取: {result}, 期望: {expected}")
        if result != expected:
            all_passed = False

    return all_passed

def test_code_leakage_detection():
    """测试代码泄漏检测"""
    code_indicators = ['def solve(', 'def main(', 'import ', 'return ', 'class ', 'if __name__']

    test_cases = [
        # (输入, 应该检测到代码泄漏?)
        ('def solve():\n    return 42', True),
        ('import math\nresult = math.sqrt(16)', True),
        ('The answer is 42', False),
        ('\\boxed{36}', False),
        ('class Solution:\n    pass', True),
    ]

    print("\n🔍 测试代码泄漏检测:")
    all_passed = True
    for answer, should_detect in test_cases:
        detected = any(indicator in answer for indicator in code_indicators)
        status = "✅" if detected == should_detect else "❌"
        print(f"  {status} '{answer[:40]}...' -> 检测: {detected}, 期望: {should_detect}")
        if detected != should_detect:
            all_passed = False

    return all_passed

def main():
    print("=" * 60)
    print("🧪 测试所有关键修复")
    print("=" * 60)

    results = []
    results.append(("无效boxed检测", test_invalid_boxed_detection()))
    results.append(("答案提取", test_answer_extraction()))
    results.append(("代码泄漏检测", test_code_leakage_detection()))

    print("\n" + "=" * 60)
    print("📊 测试结果汇总:")
    all_passed = True
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 所有测试通过！可以开始训练。")
    else:
        print("⚠️  部分测试失败，需要检查。")
    print("=" * 60)

    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())
