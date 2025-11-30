#!/usr/bin/env python3
"""测试 answer() 函数支持"""
import re

def run_isolated_code(code_str):
    """模拟 _execute_leaked_code 中的执行逻辑"""
    import io
    import sys
    global_namespace = {'__builtins__': __builtins__}

    # 添加常用数学库
    try:
        import math
        global_namespace['math'] = math
    except:
        pass

    # 捕获 stdout
    old_stdout = sys.stdout
    sys.stdout = captured_output = io.StringIO()

    try:
        exec(code_str, global_namespace)

        # 尝试调用常见函数名: solve(), main(), answer()
        for func_name in ['solve', 'main', 'answer']:
            if func_name in global_namespace and callable(global_namespace[func_name]):
                result = global_namespace[func_name]()
                if result is not None:
                    return f"函数 {func_name}() 返回: {result}"
                break  # 函数存在但返回None，继续检查stdout

        # 检查 stdout 输出
        stdout_content = captured_output.getvalue().strip()
        if stdout_content:
            lines = [l.strip() for l in stdout_content.split('\n') if l.strip()]
            if lines:
                return f"stdout 输出: {lines[-1]}"

        return None
    except Exception as e:
        print(f"代码执行异常: {e}", file=old_stdout)
        return None
    finally:
        sys.stdout = old_stdout


def test_execute_leaked_code(code_string):
    """测试代码泄漏执行"""
    code = code_string

    # 如果代码在代码块中，提取
    code_block_match = re.search(r'```python\s*([\s\S]*?)```', code)
    if code_block_match:
        code = code_block_match.group(1)
        print(f"  提取代码块")

    # 执行
    result = run_isolated_code(code)
    return result


if __name__ == '__main__':
    print("=" * 60)
    print("🧪 测试 answer() 函数支持")
    print("=" * 60)

    # 测试用例
    test_cases = [
        # (描述, 代码, 期望包含的结果)
        ("def solve() 返回数字", '''def solve():
    return 42
''', "42"),
        ("def main() 返回数字", '''def main():
    return 100
''', "100"),
        ("def answer() 返回字符串", '''def answer():
    return "Both are Supreme Court cases"
''', "Supreme Court"),
        ("def answer() 返回 yes/no", '''def answer():
    """Returns whether both are SC cases"""
    return "yes"
''', "yes"),
        ("代码块包装的 answer()", '''```python
def answer():
    return "test result"
```''', "test result"),
        ("只有 print 输出", '''def main():
    print("answer is 42")
main()
''', "42"),
    ]

    all_passed = True
    for desc, code, expected in test_cases:
        result = test_execute_leaked_code(code)
        if result and expected in str(result):
            status = "✅"
        else:
            status = "❌"
            all_passed = False
        print(f"{status} {desc}")
        print(f"   结果: {result}")
        print()

    print("=" * 60)
    if all_passed:
        print("🎉 所有测试通过！answer() 函数支持已添加。")
    else:
        print("⚠️ 部分测试失败，需要检查。")
    print("=" * 60)
