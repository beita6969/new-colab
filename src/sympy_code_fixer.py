#!/usr/bin/env python3
"""
SymPy代码预处理器 - 修复兼容性问题
"""
import re
from typing import Tuple


class SymPyCodeFixer:
    """
    修复生成代码中的SymPy API兼容性问题
    """

    @staticmethod
    def fix_code(code: str) -> Tuple[str, bool, list]:
        """
        修复代码中的SymPy问题

        Args:
            code: 原始代码

        Returns:
            (fixed_code, was_modified, fixes_applied)
        """
        original_code = code
        fixes_applied = []

        # 修复1: IntervalSet → Interval with union
        if 'IntervalSet' in code or 'interval_set' in code.lower():
            # IntervalSet已废弃，使用Union of Intervals
            code = re.sub(
                r'sp\.calculus\.util\.IntervalSet',
                'sp.Union',
                code
            )
            code = re.sub(
                r'IntervalSet\(',
                'Union(',
                code
            )
            if code != original_code:
                fixes_applied.append('IntervalSet → Union')

        # 修复2: Interval加法 → Interval并集
        code = re.sub(
            r'sp\.Interval\([^)]+\)\s*\+\s*sp\.Interval\([^)]+\)',
            lambda m: m.group(0).replace(' + ', ' | '),
            code
        )
        if 'Interval' in code and ' + ' in code:
            fixes_applied.append('Interval + → Interval |')

        # 修复3: 添加安全的SymPy比较
        if 'sympy' in code.lower() and ('if ' in code or 'while ' in code):
            # 在代码开头添加安全比较函数
            helper = '''\n# SymPy safe comparison helper\ndef _sp_bool(expr):\n    """Safely convert SymPy expression to bool"""\n    try:\n        if hasattr(expr, 'evalf'):\n            return bool(expr.evalf())\n        return bool(expr)\n    except:\n        return False\n\n'''

            if '# SymPy safe comparison helper' not in code:
                code = helper + code
                fixes_applied.append('Added safe comparison helper')

        # 修复4: 修复常见的SymPy错误用法
        replacements = [
            # 旧API → 新API
            ('sp.solve_univariate_inequality', 'sp.solve'),
            ('sp.calculus.singularities', 'sp.singularities'),
        ]

        for old, new in replacements:
            if old in code:
                code = code.replace(old, new)
                fixes_applied.append(f'{old} → {new}')

        # 修复5: 添加导入检查
        if 'import sympy' in code and 'sp.' in code:
            # 确保正确的别名
            if 'import sympy as sp' not in code:
                code = code.replace('import sympy', 'import sympy as sp')
                fixes_applied.append('Fixed sympy import alias')

        was_modified = (code != original_code)
        return code, was_modified, fixes_applied

    @staticmethod
    def add_safe_execution_wrapper(code: str) -> str:
        """
        为代码添加安全执行包装

        Args:
            code: 原始代码

        Returns:
            包装后的代码
        """
        wrapper = '''\ntry:\n{indented_code}\nexcept Exception as e:\n    print(f"Execution error: {{e}}")\n    import traceback\n    traceback.print_exc()\n'''

        # 缩进原始代码
        indented_lines = ['    ' + line for line in code.split('\n')]
        indented_code = '\n'.join(indented_lines)

        return wrapper.format(indented_code=indented_code)

    @staticmethod
    def validate_code_safety(code: str) -> Tuple[bool, list]:
        """
        验证代码安全性

        Args:
            code: 代码字符串

        Returns:
            (is_safe, warnings)
        """
        warnings = []

        # 检查危险模式
        dangerous_patterns = [
            (r'\bexec\(', 'Uses exec()'),
            (r'\beval\(', 'Uses eval()'),
            (r'\b__import__\(', 'Uses __import__()'),
            (r'\bopen\([^)]*["\']w', 'Opens file for writing'),
            (r'\bos\.system\(', 'Uses os.system()'),
            (r'\bsubprocess\.', 'Uses subprocess'),
        ]

        for pattern, warning in dangerous_patterns:
            if re.search(pattern, code):
                warnings.append(warning)

        is_safe = len(warnings) == 0
        return is_safe, warnings


def test_sympy_fixer():
    """测试SymPy修复器"""
    fixer = SymPyCodeFixer()

    # 测试1: IntervalSet修复
    code1 = '''
import sympy as sp
result = sp.calculus.util.IntervalSet(-sp.oo, 2)
print(result)
'''
    fixed1, modified1, fixes1 = fixer.fix_code(code1)
    print("测试1 - IntervalSet修复:")
    print(f"  修改: {modified1}")
    print(f"  应用的修复: {fixes1}")
    print(f"  修复后: {fixed1[:100]}...\n")

    # 测试2: Interval加法
    code2 = '''
import sympy as sp
interval = sp.Interval(-2, 2) + sp.Interval(3, 5)
print(interval)
'''
    fixed2, modified2, fixes2 = fixer.fix_code(code2)
    print("测试2 - Interval加法修复:")
    print(f"  修改: {modified2}")
    print(f"  应用的修复: {fixes2}\n")

    # 测试3: 安全性检查
    code3 = '''
import os
os.system("rm -rf /")
'''
    is_safe, warnings = fixer.validate_code_safety(code3)
    print("测试3 - 安全性检查:")
    print(f"  安全: {is_safe}")
    print(f"  警告: {warnings}")


if __name__ == "__main__":
    test_sympy_fixer()
