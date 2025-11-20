#!/usr/bin/env python3
"""
工作流代码验证器 - 确保生成的工作流代码质量
"""
import ast
import re
from typing import Tuple, Dict, List


class WorkflowValidator:
    """
    验证RL模型生成的工作流代码

    功能：
    1. 语法检查
    2. 必需元素检查
    3. 算子名称规范检查
    4. 异步调用检查
    """

    def __init__(self):
        # 已知的算子列表
        self.valid_operators = [
            'Custom', 'AnswerGenerate', 'Programmer', 'ScEnsemble',
            'Test', 'Review', 'Revise', 'CustomCodeGenerate',
            'Format', 'MdEnsemble'
        ]

        # 算子参数要求
        self.operator_requirements = {
            'Custom': ['input', 'instruction'],
            'AnswerGenerate': ['input'],
            'Programmer': ['problem', 'analysis'],
            'ScEnsemble': ['solutions', 'problem'],
            'Test': ['problem', 'solution', 'entry_point'],
            'Review': ['problem', 'solution'],
            'Revise': ['problem', 'solution', 'feedback'],
            'CustomCodeGenerate': ['problem', 'entry_point', 'instruction'],
            'Format': ['problem', 'solution'],
            'MdEnsemble': ['solutions', 'problem']
        }

    def validate_workflow_code(self, code: str, problem_type: str = 'math') -> Tuple[bool, str, Dict]:
        """
        验证工作流代码

        Args:
            code: 生成的Python代码
            problem_type: 问题类型 (math/code/qa)

        Returns:
            (is_valid, error_message, validation_details)
        """
        validation_details = {
            'syntax_valid': False,
            'has_workflow_class': False,
            'has_call_method': False,
            'has_return': False,
            'operators_valid': False,
            'async_calls_valid': False,
            'warnings': []
        }

        # 1. 语法检查
        try:
            tree = ast.parse(code)
            validation_details['syntax_valid'] = True
        except SyntaxError as e:
            return False, f"语法错误: {e}", validation_details

        # 2. 检查Workflow类
        has_workflow_class = any(
            isinstance(node, ast.ClassDef) and node.name == 'Workflow'
            for node in ast.walk(tree)
        )
        validation_details['has_workflow_class'] = has_workflow_class
        if not has_workflow_class:
            return False, "缺少Workflow类定义", validation_details

        # 3. 检查__call__方法
        has_call_method = self._has_call_method(tree)
        validation_details['has_call_method'] = has_call_method
        if not has_call_method:
            return False, "缺少async def __call__方法", validation_details

        # 4. 检查return语句
        has_return = self._has_return_in_call(tree)
        validation_details['has_return'] = has_return
        if not has_return:
            return False, "__call__方法缺少return语句", validation_details

        # 5. 检查算子使用
        operator_issues = self._check_operators(code)
        if operator_issues:
            validation_details['operators_valid'] = False
            validation_details['warnings'].extend(operator_issues)
            # 算子问题作为警告，不直接失败
        else:
            validation_details['operators_valid'] = True

        # 6. 检查异步调用
        async_issues = self._check_async_calls(code)
        if async_issues:
            validation_details['async_calls_valid'] = False
            validation_details['warnings'].extend(async_issues)
        else:
            validation_details['async_calls_valid'] = True

        # 7. 特定类型检查
        if problem_type == 'code':
            code_issues = self._check_code_workflow(tree, code)
            if code_issues:
                validation_details['warnings'].extend(code_issues)

        # 综合判断
        if validation_details['warnings']:
            warning_msg = '; '.join(validation_details['warnings'])
            return True, f"验证通过但有警告: {warning_msg}", validation_details

        return True, "验证通过", validation_details

    def _has_call_method(self, tree: ast.AST) -> bool:
        """检查是否有__call__方法"""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'Workflow':
                for item in node.body:
                    if isinstance(item, ast.AsyncFunctionDef) and item.name == '__call__':
                        return True
        return False

    def _has_return_in_call(self, tree: ast.AST) -> bool:
        """检查__call__方法是否有return语句"""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'Workflow':
                for item in node.body:
                    if isinstance(item, ast.AsyncFunctionDef) and item.name == '__call__':
                        for stmt in ast.walk(item):
                            if isinstance(stmt, ast.Return):
                                return True
        return False

    def _check_operators(self, code: str) -> List[str]:
        """检查算子使用问题"""
        issues = []

        # 检查小写算子名（常见错误）
        lowercase_pattern = r'operator\.([a-z][a-zA-Z_]*?)\('
        lowercase_matches = re.findall(lowercase_pattern, code)
        for match in lowercase_matches:
            issues.append(f"算子名应使用PascalCase: operator.{match} -> operator.{match.capitalize()}")

        # 检查未知算子
        operator_pattern = r'operator\.([A-Z][a-zA-Z_]*?)\('
        operator_matches = re.findall(operator_pattern, code)
        for op in operator_matches:
            if op not in self.valid_operators:
                issues.append(f"未知算子: {op}")

        # 检查Test算子参数（Code工作流常见错误）
        if 'self.test' in code:
            test_pattern = r'self\.test\([^)]*\)'
            test_calls = re.findall(test_pattern, code)
            for call in test_calls:
                # 检查是否包含所有必需参数
                if not all(param in call for param in ['problem', 'solution', 'entry_point']):
                    issues.append("Test算子缺少必需参数: 需要problem, solution, entry_point")

        return issues

    def _check_async_calls(self, code: str) -> List[str]:
        """检查异步调用问题"""
        issues = []

        # 检查算子调用是否使用await
        operator_call_pattern = r'(self\.[a-z_]+)\([^)]*\)'
        calls = re.findall(operator_call_pattern, code)

        for call in calls:
            # 排除非算子调用
            if call in ['self.llm', 'self.name', 'self.dataset']:
                continue

            # 检查是否有对应的await
            if f'await {call}' not in code:
                issues.append(f"异步调用缺少await: {call}")

        return issues

    def _check_code_workflow(self, tree: ast.AST, code: str) -> List[str]:
        """检查Code类型工作流的特殊要求"""
        issues = []

        # 检查__call__方法签名
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'Workflow':
                for item in node.body:
                    if isinstance(item, ast.AsyncFunctionDef) and item.name == '__call__':
                        # 检查参数
                        args = [arg.arg for arg in item.args.args]
                        if 'entry_point' not in args:
                            issues.append("Code工作流的__call__方法应包含entry_point参数")

        return issues

    def fix_common_issues(self, code: str) -> str:
        """
        尝试自动修复常见问题

        Args:
            code: 有问题的代码

        Returns:
            修复后的代码
        """
        fixed_code = code

        # 1. 修复小写算子名
        lowercase_pattern = r'operator\.([a-z][a-zA-Z_]*?)\('
        def fix_case(match):
            name = match.group(1)
            # 智能大写转换
            if name == 'custom':
                return 'operator.Custom('
            elif name == 'answergenerae' or name == 'answer_generate':
                return 'operator.AnswerGenerate('
            elif name == 'programmer':
                return 'operator.Programmer('
            elif name == 'test':
                return 'operator.Test('
            elif name == 'review':
                return 'operator.Review('
            elif name == 'revise':
                return 'operator.Revise('
            elif name.startswith('sc'):
                return 'operator.ScEnsemble('
            else:
                # 默认：首字母大写
                return f'operator.{name.capitalize()}('

        fixed_code = re.sub(lowercase_pattern, fix_case, fixed_code)

        # 2. 修复缺少await的算子调用
        # 查找所有self.xxx()调用
        call_pattern = r'^(\s*)(self\.(?:custom|answer_generate|programmer|test|review|revise|sc_ensemble)\([^)]*\))'
        lines = fixed_code.split('\n')
        fixed_lines = []

        for line in lines:
            if re.match(call_pattern, line) and 'await' not in line:
                # 添加await
                line = re.sub(call_pattern, r'\1await \2', line)
            fixed_lines.append(line)

        fixed_code = '\n'.join(fixed_lines)

        # 3. 确保Test算子有完整参数（针对Code问题）
        if 'self.test' in fixed_code and 'entry_point' not in fixed_code:
            # 尝试添加entry_point参数
            test_pattern = r'self\.test\(([^)]+)\)'
            def add_entry_point(match):
                params = match.group(1)
                if 'entry_point' not in params:
                    # 添加entry_point参数
                    return f'self.test({params}, entry_point=entry_point)'
                return match.group(0)

            fixed_code = re.sub(test_pattern, add_entry_point, fixed_code)

        return fixed_code


def test_validator():
    """测试验证器"""
    validator = WorkflowValidator()

    # 测试用例1：正确的工作流
    good_code = '''
import operator
from scripts.async_llm import create_llm_instance

class Workflow:
    def __init__(self, name, llm_config, dataset):
        self.name = name
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)

    async def __call__(self, problem):
        result = await self.custom(input=problem, instruction="Solve")
        return result['response'], self.llm.get_usage_summary()["total_cost"]
'''

    # 测试用例2：有问题的工作流
    bad_code = '''
class Workflow:
    def __init__(self, name, llm_config, dataset):
        self.custom = operator.custom(self.llm)  # 小写错误

    async def __call__(self, problem):
        result = self.custom(input=problem)  # 缺少await
        # 缺少return
'''

    print("测试正确的工作流:")
    valid, msg, details = validator.validate_workflow_code(good_code)
    print(f"  结果: {valid}, 消息: {msg}")

    print("\n测试有问题的工作流:")
    valid, msg, details = validator.validate_workflow_code(bad_code)
    print(f"  结果: {valid}, 消息: {msg}")

    print("\n尝试自动修复:")
    fixed = validator.fix_common_issues(bad_code)
    print("修复后的代码:")
    print(fixed)


if __name__ == "__main__":
    test_validator()
