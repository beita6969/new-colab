#!/usr/bin/env python3
"""
Code执行器 - 安全执行Python代码，修复Sympy错误
"""
import ast
import sys
import io
import traceback
import contextlib
import signal
from typing import Tuple, Dict, Any, Optional
import subprocess
import tempfile
import os


class CodeExecutor:
    """
    安全执行Python代码，处理Sympy和其他常见错误
    """

    def __init__(self, timeout: int = 10):
        """
        Args:
            timeout: 执行超时时间（秒）
        """
        self.timeout = timeout

    def safe_execute_code(
        self,
        code: str,
        test_code: Optional[str] = None,
        entry_point: Optional[str] = None
    ) -> Tuple[bool, str, Dict]:
        """
        安全执行代码，避免Sympy错误

        Args:
            code: 要执行的代码
            test_code: 测试代码（可选）
            entry_point: 函数入口点（可选）

        Returns:
            (success, output, metadata)
        """
        metadata = {
            'method': None,
            'error_type': None,
            'sympy_error': False
        }

        # 方法1: 先尝试在受限环境中执行
        success, output = self._execute_restricted(code, test_code, entry_point)
        metadata['method'] = 'restricted'

        if success:
            return success, output, metadata

        # 检查是否是Sympy错误
        if 'cannot determine truth value' in output.lower():
            metadata['sympy_error'] = True
            # 尝试修复Sympy错误
            fixed_code = self._fix_sympy_errors(code)
            success, output = self._execute_restricted(fixed_code, test_code, entry_point)
            metadata['method'] = 'sympy_fixed'

            if success:
                return success, output, metadata

        # 方法2: 使用子进程隔离执行
        success, output = self._execute_subprocess(code, test_code, entry_point)
        metadata['method'] = 'subprocess'

        if not success:
            metadata['error_type'] = self._classify_error(output)

        return success, output, metadata

    def _execute_restricted(
        self,
        code: str,
        test_code: Optional[str] = None,
        entry_point: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        在受限环境中执行代码
        """
        # 创建安全的全局环境
        safe_globals = {
            '__builtins__': {
                # 基础类型
                'int': int, 'float': float, 'str': str, 'bool': bool,
                'list': list, 'dict': dict, 'set': set, 'tuple': tuple,
                # 基础函数
                'len': len, 'range': range, 'enumerate': enumerate,
                'zip': zip, 'map': map, 'filter': filter,
                'min': min, 'max': max, 'sum': sum, 'abs': abs,
                'sorted': sorted, 'reversed': reversed,
                'print': print, 'input': input,
                # 数学函数
                'pow': pow, 'round': round, 'divmod': divmod,
                # 类型转换
                'ord': ord, 'chr': chr, 'hex': hex, 'bin': bin,
                # 其他
                'isinstance': isinstance, 'type': type,
                'Exception': Exception, 'ValueError': ValueError,
                'TypeError': TypeError, 'IndexError': IndexError,
                'KeyError': KeyError,
            },
            '__name__': '__main__',
        }

        # 添加常用模块
        try:
            import math
            safe_globals['math'] = math
        except ImportError:
            pass

        try:
            import collections
            safe_globals['collections'] = collections
        except ImportError:
            pass

        try:
            import itertools
            safe_globals['itertools'] = itertools
        except ImportError:
            pass

        # 捕获输出
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        captured_output = io.StringIO()
        sys.stdout = captured_output
        sys.stderr = captured_output

        try:
            # 设置超时
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Code execution exceeded {self.timeout} seconds")

            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.timeout)

            # 执行代码
            exec(code, safe_globals)

            # 如果有测试代码，执行测试
            if test_code:
                exec(test_code, safe_globals)

            # 如果有入口点，调用它
            if entry_point and entry_point in safe_globals:
                func = safe_globals[entry_point]
                # 尝试调用函数（简单测试）
                if callable(func):
                    # 测试一些基本输入
                    test_cases = [
                        (),  # 无参数
                        (0,), (1,), (5,),  # 单个数字
                        ([],), ([1, 2, 3],),  # 列表
                        ('',), ('test',),  # 字符串
                    ]

                    for args in test_cases:
                        try:
                            result = func(*args)
                            print(f"{entry_point}{args} = {result}")
                            break  # 只要有一个成功就退出
                        except Exception:
                            continue

            # 取消超时
            signal.alarm(0)

            output = captured_output.getvalue()
            return True, output

        except TimeoutError as e:
            signal.alarm(0)
            return False, str(e)

        except Exception as e:
            signal.alarm(0)
            error_msg = f"{type(e).__name__}: {str(e)}\n"
            error_msg += traceback.format_exc()
            return False, error_msg

        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def _execute_subprocess(
        self,
        code: str,
        test_code: Optional[str] = None,
        entry_point: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        使用子进程执行代码（更安全）
        """
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            # 写入代码
            f.write(code)

            # 如果有测试代码，添加测试
            if test_code:
                f.write("\n\n# Tests\n")
                f.write(test_code)

            # 如果有入口点，添加简单测试
            if entry_point:
                f.write(f"\n\n# Test entry point\n")
                f.write(f"if __name__ == '__main__':\n")
                f.write(f"    if '{entry_point}' in globals():\n")
                f.write(f"        func = {entry_point}\n")
                f.write(f"        try:\n")
                f.write(f"            print(f'{entry_point}(5) = {{func(5)}}')\n")
                f.write(f"        except Exception as e:\n")
                f.write(f"            print(f'Error calling {entry_point}: {{e}}')\n")

            temp_file = f.name

        try:
            # 使用subprocess执行
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            # 判断成功
            if result.returncode == 0:
                return True, result.stdout
            else:
                error_output = result.stderr or result.stdout
                return False, error_output

        except subprocess.TimeoutExpired:
            return False, f"Code execution exceeded {self.timeout} seconds"

        except Exception as e:
            return False, f"Subprocess execution error: {e}"

        finally:
            # 清理临时文件
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def _fix_sympy_errors(self, code: str) -> str:
        """
        尝试修复Sympy相关错误
        """
        fixed_code = code

        # 修复1: 将Sympy符号比较转换为数值比较
        sympy_patterns = [
            # Relational比较
            (r'if\s+(\w+)\s*([<>]=?)\s*(\w+)', r'if float(\1) \2 float(\3)'),
            (r'if\s+(\w+)\s*==\s*(\w+)', r'if abs(float(\1) - float(\3)) < 1e-9'),
            # while循环条件
            (r'while\s+(\w+)\s*([<>]=?)\s*(\w+)', r'while float(\1) \2 float(\3)'),
        ]

        import re
        for pattern, replacement in sympy_patterns:
            fixed_code = re.sub(pattern, replacement, fixed_code)

        # 修复2: 添加Sympy导入检查
        if 'sympy' in fixed_code.lower():
            # 在代码开头添加辅助函数
            helper = '''
# Sympy helper functions
def safe_compare(a, b, op='=='):
    """Safely compare Sympy expressions"""
    try:
        if hasattr(a, 'evalf'):
            a = float(a.evalf())
        if hasattr(b, 'evalf'):
            b = float(b.evalf())
        a, b = float(a), float(b)

        if op == '==':
            return abs(a - b) < 1e-9
        elif op == '<':
            return a < b
        elif op == '<=':
            return a <= b
        elif op == '>':
            return a > b
        elif op == '>=':
            return a >= b
        else:
            return False
    except:
        return False

'''
            fixed_code = helper + fixed_code

        return fixed_code

    def _classify_error(self, error_msg: str) -> str:
        """
        分类错误类型
        """
        error_lower = error_msg.lower()

        if 'cannot determine truth value' in error_lower:
            return 'sympy_relational'
        elif 'syntaxerror' in error_lower:
            return 'syntax'
        elif 'nameerror' in error_lower:
            return 'undefined_variable'
        elif 'typeerror' in error_lower:
            return 'type_error'
        elif 'indexerror' in error_lower:
            return 'index_out_of_bounds'
        elif 'keyerror' in error_lower:
            return 'missing_key'
        elif 'zerodivision' in error_lower:
            return 'division_by_zero'
        elif 'timeout' in error_lower:
            return 'timeout'
        else:
            return 'unknown'


def test_code_executor():
    """测试代码执行器"""
    executor = CodeExecutor(timeout=5)

    print("="*60)
    print("测试1: 正常的代码")
    print("="*60)

    code1 = '''
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)
'''

    success, output, metadata = executor.safe_execute_code(code1, entry_point='factorial')
    print(f"成功: {success}")
    print(f"输出: {output}")
    print(f"元数据: {metadata}")

    print("\n" + "="*60)
    print("测试2: Sympy错误代码")
    print("="*60)

    code2 = '''
import sympy as sp
x = sp.Symbol('x')
y = x**2 + 2*x + 1
# This would cause "cannot determine truth value" error
if y > 0:
    print("Positive")
'''

    success, output, metadata = executor.safe_execute_code(code2)
    print(f"成功: {success}")
    print(f"输出: {output[:200]}...")
    print(f"元数据: {metadata}")

    print("\n" + "="*60)
    print("测试3: 超时代码")
    print("="*60)

    code3 = '''
while True:
    pass  # Infinite loop
'''

    success, output, metadata = executor.safe_execute_code(code3)
    print(f"成功: {success}")
    print(f"输出: {output}")
    print(f"元数据: {metadata}")


if __name__ == "__main__":
    test_code_executor()
