#!/usr/bin/env python3
"""
答案提取器 - 从模型输出和ground truth中提取标准化答案
"""
import re
import json
from typing import Any, Optional, Tuple

class AnswerExtractor:
    """统一的答案提取器，用于标准化预测和真值"""

    def __init__(self, use_llm_fallback: bool = True, llm_client=None):
        """
        Args:
            use_llm_fallback: 是否使用LLM作为兜底提取器
            llm_client: LLM客户端（用于兜底提取）
        """
        self.use_llm_fallback = use_llm_fallback
        self.llm_client = llm_client

    def extract_answer(self, text: str, problem_type: str, is_ground_truth: bool = False) -> str:
        """
        主入口：从文本中提取标准化答案

        Args:
            text: 原始文本
            problem_type: 问题类型 (math/code/qa)
            is_ground_truth: 是否是ground truth（影响提取策略）

        Returns:
            标准化后的答案
        """
        if not text:
            return ""

        if problem_type == "math":
            return self._extract_math_answer(text, is_ground_truth)
        elif problem_type == "code":
            return self._extract_code_answer(text, is_ground_truth)
        elif problem_type == "qa":
            return self._extract_qa_answer(text, is_ground_truth)
        else:
            return str(text).strip()

    def _extract_math_answer(self, text: str, is_ground_truth: bool) -> str:
        """
        提取数学答案 - 通用方法（不针对特定数据集）

        策略（参考AgentFlow）:
        1. <answer>标签（取最后一个）
        2. \boxed{}（LaTeX格式）
        3. GSM8K的#### 格式（适配特定数据集）
        4. 明确的"Final Answer"标记
        5. 对于ground_truth: 使用LLM理解复杂文本
        6. 兜底：提取最后一个数字
        """
        text = str(text).strip()

        # 1. 优先提取<answer>标签（取最后一个，避免中间值）
        answer_matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
        if answer_matches:
            # 关键：取最后一个匹配（AgentFlow方法）
            answer_text = answer_matches[-1].strip()
            return self._clean_math_answer(answer_text)

        # 过滤workflow日志污染
        if "Revised Solution:" in text or "Based on the feedback" in text:
            # 清理 "Revised Solution:" 污染
            clean_text = re.sub(r'Revised Solution:.*?(?=\\boxed|\d|$)', '', text, flags=re.DOTALL)
            # 清理 "Based on the feedback" 污染
            clean_text = re.sub(r'Based on the feedback[^\\]*(?=\\boxed|$)', '', clean_text, flags=re.DOTALL)
            if clean_text.strip():
                text = clean_text

        # 2. 提取\boxed{}（标准LaTeX格式）- 增强检测代码泄漏和错误
        boxed = self._extract_boxed(text)
        if boxed:
            # 检测空输出（最先检查）
            if not boxed or boxed.strip() == '':
                # 空输出，继续尝试其他提取方法
                boxed = None
            # 🔧 P0-FIX: 检测代码泄漏（最重要的检查）
            # 如果boxed中包含def/return/import等关键字，说明错误地将代码作为答案
            elif any(keyword in boxed for keyword in ['def ', 'return ', 'import ', 'class ', 'if __name__', 'print(', 'for ', 'while ', 'elif ', ':\n', 'await ', 'async ']):
                # 🔧 代码泄漏！尝试执行代码获取真实答案
                print(f"  ⚠️ 检测到代码泄漏在\\boxed{{}}中，尝试执行...")
                executed_answer = self._execute_code_and_extract_answer(boxed, 'math')
                if executed_answer:
                    print(f"  ✅ 执行成功，提取答案: {executed_answer}")
                    return executed_answer
                # 执行失败，尝试静态分析
                code_answer = self._extract_answer_from_code_block(boxed)
                if code_answer and not any(kw in str(code_answer) for kw in ['def ', 'import ', 'class ']):
                    print(f"  ✅ 静态分析提取答案: {code_answer}")
                    return self._clean_math_answer(code_answer)
                # 无法提取有效答案，返回空（不要返回代码！）
                print(f"  ❌ 无法从代码中提取答案，返回空")
                boxed = None
            # 检测代码块标记：如果包含```python或```，说明是代码块而非答案
            elif '```python' in boxed or boxed.startswith('```'):
                # 策略1：尝试执行代码获取答案（仅对math问题）
                executed_answer = self._execute_code_and_extract_answer(boxed, 'math')
                if executed_answer:
                    return executed_answer

                # 策略2：静态分析提取答案
                code_answer = self._extract_answer_from_code_block(boxed)
                if code_answer:
                    return code_answer
                # 无法提取，跳过
                boxed = None
            # 检测执行错误：如果是Error信息，跳过
            elif boxed.startswith('Error:') or 'Traceback' in boxed or 'SyntaxError' in boxed:
                # 执行错误，继续尝试其他提取方法
                boxed = None
            # 检测无效输出 - 使用 'in' 而不是 startswith 来捕获更多情况
            elif 'Based on the feedback' in boxed or 'Revised Solution' in boxed:
                # 无效输出，跳过
                boxed = None
            else:
                return self._clean_math_answer(boxed)

        # 3. GSM8K格式：提取#### 后的数字（适配特定数据集）
        if is_ground_truth:
            gsm8k_match = re.search(r'####\s*(-?\d+\.?\d*)', text)
            if gsm8k_match:
                return self._clean_math_answer(gsm8k_match.group(1))

        # 4. 查找明确的"Final Answer"标记
        final_answer_patterns = [
            r"(?:the\s+final\s+answer\s+is)[：:]*\s*([-+]?\d+(?:/\d+)?(?:\.\d+)?)",
            r"(?:Final\s+Answer|最终答案)[：:]*\s*([-+]?\d+(?:/\d+)?(?:\.\d+)?)",
            r"(?:The\s+answer\s+is)[：:]*\s*([-+]?\d+(?:/\d+)?(?:\.\d+)?)",
        ]
        for pattern in final_answer_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return self._clean_math_answer(match.group(1))

        # 4. 对于ground_truth且文本复杂：使用LLM理解（通用方法）
        if is_ground_truth and self.use_llm_fallback and self.llm_client:
            # 检测复杂性：多个数字和运算符
            has_calculations = text.count('=') >= 2 or len(re.findall(r'\d+', text)) > 3
            if has_calculations:
                llm_result = self._llm_extract_math_ground_truth(text)
                if llm_result and llm_result != text:
                    return llm_result

        # 5. 检查是否为代数表达式（包含变量）
        # 如果包含字母变量（x, y, a, b等）且有运算符，保持原样
        has_variables = bool(re.search(r'[a-zA-Z]', text))
        has_operators = bool(re.search(r'[+\-*/\^]', text))
        if has_variables and has_operators:
            # 这是代数表达式，返回清理后的文本（去除空格等）
            cleaned = re.sub(r'\s+', '', text)  # 移除空格
            cleaned = cleaned.strip()
            return cleaned

        # 6. 兜底策略：提取数字
        if is_ground_truth:
            # Ground truth: 直接取最后一个数字（简单情况）
            numbers = self._extract_all_numbers(text)
            if numbers:
                return str(numbers[-1])
        else:
            # Prediction: 优先括号外的数字
            clean_text = re.sub(r'\([^)]*\)', '', text)
            clean_numbers = self._extract_all_numbers(clean_text)
            if clean_numbers:
                return str(clean_numbers[-1])
            numbers = self._extract_all_numbers(text)
            if numbers:
                return str(numbers[-1])

        # 最后兜底：检查是否有污染内容，如果有则返回空字符串
        # 避免返回包含 "Based on the feedback" 或代码块的污染文本
        if 'Based on the feedback' in text or 'Revised Solution' in text or '```python' in text:
            return ""
        # 否则返回清理后的文本（仅保留数字和基本符号）
        cleaned = re.sub(r'[^\d\-+./]', ' ', text).strip()
        if cleaned:
            # 尝试提取最后一个数字
            nums = re.findall(r'-?\d+\.?\d*', cleaned)
            if nums:
                return nums[-1]
        return ""

    def _extract_code_answer(self, text: str, is_ground_truth: bool) -> str:
        """
        提取代码答案

        对于Code任务:
        - prediction: 提取完整的函数实现代码
        - ground_truth: 同样提取函数实现代码
        - 评估: 通过test_result metadata而非字符串比较

        优先级：
        1. ```python...``` 代码块（带AST验证）
        2. def 函数定义
        3. 完整文本（如果是ground truth）
        """
        text = str(text).strip()

        # P0修复: 清理空代码块和无效占位符
        text = re.sub(r'```python\s*```', '', text)  # 清理空块
        text = re.sub(r'```\s*```', '', text)
        text = text.replace('No code provided', '').replace('No code', '')

        # 1. 提取代码块 (P1修复: 换行符可选)
        code_blocks = re.findall(r'```(?:python)?\s*\n?([^`]+)```', text)
        if code_blocks:
            # 尝试从后往前找第一个语法正确的代码块
            for block in reversed(code_blocks):
                block = block.strip()
                # 验证代码语法正确性
                if self._validate_code_syntax(block):
                    return block
            # 如果所有代码块都有语法错误，返回最后一个
            return code_blocks[-1].strip()

        # 2. 查找函数定义
        func_pattern = r'(def\s+\w+\s*\([^)]*\)[^:]*:[\s\S]+?)(?=\n(?:def\s|class\s|$))'
        funcs = re.findall(func_pattern, text)
        if funcs:
            # 验证第一个函数定义
            first_func = funcs[0].strip()
            if self._validate_code_syntax(first_func):
                return first_func
            return first_func  # 即使有语法错误也返回

        # 3. 如果是ground truth且看起来像代码，直接返回
        if is_ground_truth:
            return text

        # 4. LLM兜底
        if self.use_llm_fallback and self.llm_client:
            return self._llm_extract_code(text)

        return text

    def _validate_code_syntax(self, code: str) -> bool:
        """
        验证代码语法正确性

        Returns:
            True if valid Python syntax, False otherwise
        """
        try:
            import ast
            ast.parse(code)
            return True
        except SyntaxError:
            return False
        except Exception:
            return False

    def _extract_qa_answer(self, text: str, is_ground_truth: bool) -> str:
        """
        提取QA答案
        - 对于数值型问题: 提取最终数字答案
        - 对于文本型问题: 标准化文本
        - 对于选项题: 统一格式为单字母（A/B/C/D/E）
        """
        text = str(text).strip()

        # 0. 选项题标准化（优先处理）
        # 如果答案看起来像选项格式，标准化为单字母
        option_answer = self._normalize_option_answer(text)
        if option_answer:
            return option_answer

        # 1. 如果有明确的答案标记，先尝试提取
        answer_patterns = [
            r"(?:Answer|答案)[：:]*\s*([^\n.]+)",
            r"(?:The answer is)[：:]*\s*([^\n.]+)",
            r"(?:Final answer|Therefore)[：:]*\s*([^\n.]+)",
        ]

        for pattern in answer_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                answer_text = match.group(1).strip()
                # 再次检查是否是选项格式
                option_normalized = self._normalize_option_answer(answer_text)
                if option_normalized:
                    return option_normalized
                # 尝试从答案文本中提取数字
                numbers = self._extract_all_numbers(answer_text)
                if numbers:
                    return str(int(numbers[-1]) if numbers[-1] == int(numbers[-1]) else numbers[-1])
                return self._normalize_qa_answer(answer_text)

        # 2. 检查是否为数值型答案（通过检测文本中是否有数字计算）
        # 如果文本包含计算符号(+, -, *, /, =)，则尝试提取最终数字
        has_calculation = any(op in text for op in ['+', '-', '*', '/', '=', '<<', '>>'])
        if has_calculation or re.search(r'\d+', text):
            # 尝试提取最终答案
            # 策略1: 查找最后出现的数字(排除中间计算过程)
            numbers = self._extract_all_numbers(text)
            if numbers:
                # 取最后一个数字作为最终答案(通常是计算结果)
                final_number = numbers[-1]
                return str(int(final_number) if final_number == int(final_number) else final_number)

        # 3. 文本型答案 - 标准化整个文本
        normalized = self._normalize_qa_answer(text)

        # 4. 如果太长，尝试提取核心信息
        if len(normalized.split()) > 50 and not is_ground_truth:
            # 取最后关键句
            sentences = text.split('.')
            if len(sentences) > 2:
                key_text = sentences[-2] + '.' + sentences[-1]
                return self._normalize_qa_answer(key_text)

        return normalized

    def _normalize_option_answer(self, text: str) -> Optional[str]:
        """标准化选项答案为单字母格式

        支持的格式：
        - "A" → "A"
        - "A." → "A"
        - "A. ream" → "A"
        - "ream" (如果没有其他线索) → None
        - "Option A" → "A"
        - "(A)" → "A"
        """
        text = text.strip()

        # 格式1: 单个大写字母
        if len(text) == 1 and text.upper() in 'ABCDE':
            return text.upper()

        # 格式2: "A." 或 "A:" 或 "(A)"
        match = re.match(r'^[\(\[]?([A-E])[\)\]\.:]*\s*', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # 格式3: "Option A" 或 "选项A"
        match = re.search(r'(?:Option|选项)\s*([A-E])\b', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # 格式4: "The answer is A"
        match = re.search(r'\b([A-E])\b(?=\s*(?:is|为)\s*(?:correct|the answer)?)', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        return None

    def _execute_code_and_extract_answer(self, code_block: str, problem_type: str) -> Optional[str]:
        """执行代码并提取答案（用于数学问题）

        Args:
            code_block: 代码块文本
            problem_type: 问题类型（只对math问题执行代码）

        Returns:
            执行结果或None
        """
        # 只对math问题执行代码
        if problem_type != "math":
            return None

        import subprocess
        import tempfile
        import os

        # 移除代码块标记
        code = re.sub(r'^```python\n?', '', code_block)
        code = re.sub(r'```$', '', code)
        code = code.strip()

        # 安全检查：拒绝危险操作
        dangerous_keywords = ['os.system', 'subprocess', 'eval', 'exec', 'open', '__import__', 'rm ', 'del ']
        if any(kw in code for kw in dangerous_keywords):
            return None

        try:
            # 创建临时文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                # 修改代码：捕获最后的表达式结果
                # 如果代码包含print，直接运行
                # 如果代码只有计算，添加print输出最后的变量
                if 'print(' not in code:
                    # 查找最后的赋值语句
                    lines = code.split('\n')
                    last_var = None
                    for line in reversed(lines):
                        line = line.strip()
                        if '=' in line and not line.startswith('#'):
                            # 提取变量名
                            var_name = line.split('=')[0].strip()
                            if var_name.isidentifier():
                                last_var = var_name
                                break

                    if last_var:
                        code += f'\nprint({last_var})'

                f.write(code)
                temp_path = f.name

            # 执行代码（5秒超时）
            result = subprocess.run(
                ['python3', temp_path],
                capture_output=True,
                text=True,
                timeout=5
            )

            # 删除临时文件
            os.unlink(temp_path)

            # 提取输出
            if result.returncode == 0 and result.stdout:
                output = result.stdout.strip()
                # 提取最后一行（通常是结果）
                if output:
                    last_line = output.split('\n')[-1].strip()
                    # 验证是数字
                    try:
                        # 尝试转换为数字
                        if '/' in last_line:
                            parts = last_line.split('/')
                            float(parts[0])
                            float(parts[1])
                            return last_line
                        else:
                            num = float(last_line)
                            return str(int(num) if num == int(num) else num)
                    except:
                        # 不是数字，但仍然返回（可能是表达式）
                        return last_line

            return None

        except subprocess.TimeoutExpired:
            # 超时，清理临时文件
            try:
                os.unlink(temp_path)
            except:
                pass
            return None
        except Exception:
            # 执行失败
            try:
                os.unlink(temp_path)
            except:
                pass
            return None

    def _extract_answer_from_code_block(self, code_block: str) -> Optional[str]:
        """从代码块中提取答案（静态分析）

        策略：
        1. 查找print语句的参数
        2. 查找return语句的值
        3. 查找最后的计算结果

        注意：这个方法只做静态分析，不执行代码
        如果需要执行代码，调用 _execute_code_and_extract_answer
        """
        code_block = code_block.strip()

        # 移除代码块标记
        code_block = re.sub(r'^```python\n?', '', code_block)
        code_block = re.sub(r'```$', '', code_block)

        # 策略1: 查找print语句
        print_pattern = r'print\(([^)]+)\)'
        print_matches = re.findall(print_pattern, code_block)
        if print_matches:
            # 取最后一个print的内容
            last_print = print_matches[-1].strip()
            # 如果是变量名，尝试继续提取
            if last_print.isidentifier():
                # 查找这个变量的赋值
                var_pattern = rf'{last_print}\s*=\s*(.+)'
                var_match = re.search(var_pattern, code_block)
                if var_match:
                    return var_match.group(1).strip()
            return last_print

        # 策略2: 查找return语句
        return_pattern = r'return\s+(.+?)\s*(?:\n|$)'
        return_matches = re.findall(return_pattern, code_block)
        if return_matches:
            return return_matches[-1].strip()

        # 策略3: 查找最后的赋值语句
        assignment_lines = [line for line in code_block.split('\n') if '=' in line and not line.strip().startswith('#')]
        if assignment_lines:
            last_assignment = assignment_lines[-1]
            # 提取等号右边的值
            if '=' in last_assignment:
                value = last_assignment.split('=', 1)[1].strip()
                return value

        return None

    def _extract_boxed(self, text: str) -> Optional[str]:
        """提取\boxed{}中的内容"""
        # 处理嵌套括号的情况
        pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
        return None

    def _extract_all_numbers(self, text: str) -> list:
        """提取所有数字（支持整数、小数、分数、负数）

        返回: 字符串列表（保持原始格式，特别是分数）
        """
        numbers = []

        # 优先匹配分数（完整保留格式，避免转换精度损失）
        fraction_pattern = r'-?\d+/\d+'
        fraction_matches = re.findall(fraction_pattern, text)
        for frac in fraction_matches:
            numbers.append(frac)  # 保持字符串格式，如 "5/324"

        # 匹配其他数字格式
        other_patterns = [
            r'-?\d+\.?\d*(?:[eE][+-]?\d+)?',  # 科学计数法
            r'-?\d{1,3}(?:,\d{3})*(?:\.\d+)?',  # 带千分位
        ]

        for pattern in other_patterns:
            matches = re.findall(pattern, text)
            for m in matches:
                # 跳过已经作为分数一部分的数字
                if any(m in frac for frac in fraction_matches):
                    continue
                try:
                    # 移除千分位
                    clean_m = m.replace(',', '')
                    numbers.append(clean_m)  # 保持字符串格式
                except:
                    pass

        return numbers

    def _clean_math_answer(self, answer: str) -> str:
        """
        清理数学答案（去单位、标准化格式）

        重要: 保持分数形式便于比较，避免浮点精度问题
        """
        answer = str(answer).strip()

        # 修复 "i42" 问题 - 可能是"is 42"被错误处理
        if answer.startswith('i') and len(answer) > 1 and answer[1:].replace('.', '', 1).replace('/', '').isdigit():
            answer = answer[1:]

        # 移除LaTeX命令但保留内容
        answer = re.sub(r'\\boxed\{(.+?)\}', r'\1', answer)
        answer = re.sub(r'\\frac\{(.+?)\}\{(.+?)\}', r'\1/\2', answer)  # \frac{a}{b} → a/b
        answer = re.sub(r'\\text\{(.+?)\}', r'\1', answer)

        # 移除常见单位
        units = ['grams', 'gram', 'g', 'kg', 'meters', 'meter', 'm', 'cm',
                 'seconds', 'second', 's', 'minutes', 'minute', 'min',
                 'dollars', 'dollar', '$', '元', '个', '只', 'km', 'hours', 'hour']

        for unit in units:
            answer = re.sub(rf'\s*{re.escape(unit)}\b', '', answer, flags=re.IGNORECASE)

        # 移除多余的标点和空格（但保留'/'用于分数）
        answer = re.sub(r'[,\s]+', '', answer)

        # 尝试规范化数字
        try:
            # 处理分数 - 保持分数形式或化简
            if '/' in answer:
                parts = answer.split('/')
                if len(parts) == 2:
                    try:
                        numerator = float(parts[0])
                        denominator = float(parts[1])

                        # 如果分母是1，直接返回分子
                        if denominator == 1:
                            return str(int(numerator) if numerator == int(numerator) else numerator)

                        # 化简分数（使用gcd）
                        from math import gcd
                        if numerator == int(numerator) and denominator == int(denominator):
                            g = gcd(int(abs(numerator)), int(abs(denominator)))
                            if g > 1:
                                numerator /= g
                                denominator /= g
                            # 返回化简后的分数字符串
                            return f"{int(numerator)}/{int(denominator)}"

                        # 保持原分数形式
                        return answer
                    except:
                        return answer

            # 处理百分号
            if '%' in answer:
                return str(float(answer.replace('%', '')) / 100)

            # 普通数字 - 保持整数/小数格式
            num = float(answer)
            if num == int(num):
                return str(int(num))
            return str(num)
        except:
            # 无法转换，返回清理后的字符串
            return answer

    def _normalize_qa_answer(self, text: str) -> str:
        """标准化QA答案"""
        # 小写
        text = text.lower()
        # 移除标点
        text = re.sub(r'[^\w\s]', ' ', text)
        # 压缩空格
        text = ' '.join(text.split())
        return text.strip()

    def _llm_extract_math(self, text: str) -> str:
        """使用LLM提取数学答案（用于prediction）"""
        if not self.llm_client:
            return text

        prompt = f"""Extract ONLY the final numerical answer from this math solution.
Return JUST the number, no explanation.

Solution: {text[:1000]}

Final answer (number only):"""

        try:
            response = self.llm_client.generate(prompt, max_tokens=20, temperature=0)
            answer = response.strip()
            # 验证是否是数字
            float(answer.replace('/', '.').replace(',', ''))
            return answer
        except:
            return text

    def _llm_extract_math_ground_truth(self, text: str) -> str:
        """使用LLM理解ground truth中的最终答案（通用方法，参考AgentFlow）

        关键prompt设计:
        1. 明确指示"忽略推理过程"
        2. 寻找"结论性陈述"
        3. 识别最终答案vs中间计算
        """
        if not self.llm_client:
            return text

        prompt = f"""You are extracting the FINAL ANSWER from a mathematical solution text.

**Instructions:**
1. **Ignore intermediate calculations** - Focus only on the concluding answer
2. **Look for concluding statements** like "So the answer is...", "Therefore...", "The result is..."
3. **Extract the final numeric value** - Return JUST the number

**Text:**
{text[:800]}

**Output Format:**
- Return ONLY the final numerical answer
- No explanation, no intermediate values
- If multiple numbers exist, return the one from the final conclusion

**Final Answer (number only):**"""

        try:
            response = self.llm_client.generate(prompt, max_tokens=30, temperature=0)
            answer = response.strip()
            # 验证是否是数字或分数
            if '/' in answer:
                parts = answer.split('/')
                float(parts[0])
                float(parts[1])
            else:
                float(answer.replace(',', ''))
            return answer
        except:
            return text

    def _llm_extract_code(self, text: str) -> str:
        """使用LLM提取代码（兜底）"""
        if not self.llm_client:
            return text

        prompt = f"""Extract ONLY the Python function code from this text.
Return JUST the code, no explanation.

Text: {text[:1000]}

Code:"""

        try:
            response = self.llm_client.generate(prompt, max_tokens=500, temperature=0)
            # 验证是否包含def
            if 'def ' in response:
                return response.strip()
            return text
        except:
            return text


def test_extractor():
    """测试答案提取器"""
    extractor = AnswerExtractor(use_llm_fallback=False)

    # 测试用例
    test_cases = [
        # Math cases
        {
            "text": "The probability is $\\frac{1}{27}$. So the answer is \\boxed{\\frac{8}{9}}",
            "type": "math",
            "expected": "0.8888888888888888"  # 8/9 计算后
        },
        {
            "text": "After calculating, we get 586 grams",
            "type": "math",
            "expected": "586"  # 去除单位
        },
        {
            "text": "Therefore, the final answer is 42.",
            "type": "math",
            "expected": "42"  # 正确提取数字
        },
        # Code cases
        {
            "text": "```python\ndef solve(n):\n    return n * 2\n```",
            "type": "code",
            "expected": "def solve(n):\n    return n * 2"
        },
        # QA cases
        {
            "text": "The capital of France is Paris.",
            "type": "qa",
            "expected": "the capital of france is paris"
        },
    ]

    print("=" * 60)
    print("🧪 测试答案提取器")
    print("=" * 60)

    for i, case in enumerate(test_cases, 1):
        result = extractor.extract_answer(case["text"], case["type"])
        print(f"\nTest {i} ({case['type']}):")
        print(f"  输入: {case['text'][:50]}...")
        print(f"  提取: {result}")
        print(f"  期望: {case['expected']}")
        print(f"  ✅ 通过" if result == case["expected"] else f"  ❌ 不匹配")


if __name__ == "__main__":
    test_extractor()
