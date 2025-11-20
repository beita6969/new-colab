#!/usr/bin/env python3
"""
奖励计算器 - 改进版(借鉴ROLL和AgentFlow设计)
"""
import sys
import re
from typing import Any, Dict, Optional

# 添加AFlow到路径
sys.path.insert(0, '/home/yijia/.claude/11/AFlow')

# 导入答案提取器
try:
    from .answer_extractor import AnswerExtractor
except ImportError:
    from answer_extractor import AnswerExtractor


class RewardComputer:
    """
    改进的奖励计算器

    新增特性(借鉴ROLL):
    1. 格式奖励 - 检查<think>/<answer>标签
    2. 重复惩罚 - N-gram重复检测
    3. 改进的数学评估 - 支持LaTeX和boxed
    4. 更细粒度的评分阶梯
    5. LLM Judge - 使用GPT OSS 120B进行语义比较(AgentFlow方法)
    """

    def __init__(
        self,
        reward_weights: Optional[Dict[str, float]] = None,
        use_answer_extractor: bool = True,  # 是否使用答案提取器
        use_llm_judge: bool = False,  # 新增：是否使用LLM Judge
        llm_config: Optional[Dict] = None  # 新增：LLM配置
    ):
        """
        Args:
            reward_weights: 奖励权重配置（仅用于向后兼容，实际使用二元奖励）
            use_answer_extractor: 是否使用答案提取器来标准化答案
            use_llm_judge: 是否使用LLM Judge进行语义比较
            llm_config: LLM配置（用于LLM Judge）
        """
        # 保留用于向后兼容，但不再使用
        self.reward_weights = reward_weights or {
            "correctness": 1.0
        }

        # 初始化答案提取器
        self.use_answer_extractor = use_answer_extractor
        if use_answer_extractor:
            self.extractor = AnswerExtractor(use_llm_fallback=False)  # 暂时不使用LLM兜底
        else:
            self.extractor = None

        # 初始化LLM Judge
        self.use_llm_judge = use_llm_judge
        self.llm_judge_client = None
        if use_llm_judge:
            self._init_llm_judge_client(llm_config)

        print(f"✅ 10分制奖励计算器初始化完成")
        print(f"  模式: 正确性分数 [-10, 10] → 归一化奖励 [0, 1]")
        print(f"  答案提取器: {'启用' if use_answer_extractor else '禁用'}")
        print(f"  LLM Judge: {'启用 (GPT OSS 120B @ port 8002)' if use_llm_judge else '禁用'}")

    def _init_llm_judge_client(self, llm_config: Optional[Dict]):
        """初始化LLM Judge客户端（使用GPT OSS 120B）"""
        try:
            from openai import OpenAI

            # 使用port 8002的GPT OSS 120B模型
            default_config = {
                "base_url": "http://localhost:8002/v1",
                "api_key": "sk-dummy",  # vLLM不需要真实key
                "model_name": "/home/yijia/lhy/openai/gpt-oss-120b"  # 完整模型路径
            }

            config = llm_config or default_config

            self.llm_judge_client = OpenAI(
                base_url=config.get("base_url", default_config["base_url"]),
                api_key=config.get("api_key", default_config["api_key"])
            )
            self.llm_judge_model = config.get("model_name", default_config["model_name"])

            print(f"  ✅ LLM Judge客户端初始化成功")
            print(f"     模型: {self.llm_judge_model}")
            print(f"     URL: {config.get('base_url', default_config['base_url'])}")
        except Exception as e:
            print(f"  ⚠️  LLM Judge客户端初始化失败: {e}")
            self.use_llm_judge = False
            self.llm_judge_client = None

    def _llm_judge_compare(
        self,
        problem: str,
        prediction: str,
        ground_truth: str,
        problem_type: str
    ) -> bool:
        """
        使用LLM Judge进行语义比较（AgentFlow方法）

        Args:
            problem: 问题文本
            prediction: 模型预测（完整响应，未提取）
            ground_truth: Ground truth答案
            problem_type: 问题类型

        Returns:
            bool: True表示等价，False表示不等价
        """
        if not self.llm_judge_client:
            print("⚠️  LLM Judge客户端未初始化，降级为规则比较")
            return False

        # 构建AgentFlow风格的prompt（优化版 - 更明确的提取指导）
        query_prompt = f"""You are a precise mathematical and logical equivalence evaluator. Your task is to determine if the Model Response contains an answer equivalent to the Ground Truth.

**Step 1: Extract the Final Answer**
From the Model Response, extract ONLY the final answer, ignoring all reasoning steps, explanations, and intermediate calculations.

Look for answers in these formats (in order of priority):
1. Inside `\\boxed{{...}}` LaTeX notation
2. After phrases like "The answer is", "Therefore", "So", "Thus", "Final answer:"
3. In `<answer>...</answer>` tags
4. The last number, expression, or entity mentioned

**Step 2: Extract from Ground Truth**
Similarly extract the final answer from Ground Truth, which may contain:
- Step-by-step solutions (extract only the final result)
- Multiple numbers (take the last/final one)
- Explanatory text (ignore and find the answer)

**Step 3: Normalize Both Answers**
Before comparing, normalize both answers:
- **Numbers:** Convert to same format (0.5 == 1/2 == 50%)
- **Units/Currency:** Ignore ($30 == 30, 10 meters == 10)
- **Formatting:** Ignore spaces, case, punctuation
- **LaTeX:** Interpret mathematical meaning (\\frac{{1}}{{2}} == 0.5)

**Step 4: Compare Equivalence**
Answers are equivalent if:
- **Math:** Numerically/algebraically equal (even if different forms)
- **Text:** Same entity/concept (ignore synonyms, case)
- **Precision:** Allow reasonable rounding (42.0 == 42)

**Examples of CORRECT equivalence:**
- "1/2" == "0.5" ✓
- "$30" == "30" ✓
- "\\boxed{{42}}" == "42" ✓
- "x^2+2x+1" == "(x+1)^2" ✓ (algebraically equivalent)
- "10 meters" == "10" ✓

**Examples of INCORRECT equivalence:**
- "John Smith" == "Jane Doe" ✗ (different entities)
- "42" == "43" ✗ (different numbers)
- "Paris" == "London" ✗ (different locations)

**Inputs:**
Question: {problem}
Model Response: {prediction}
Ground Truth: {ground_truth}

**Required Output Format:**
<analysis>Your reasoning in 1-2 sentences</analysis>
<true_false>True or False</true_false>

Be LENIENT with formatting differences but STRICT with factual/numerical differences.
"""

        try:
            # 调用LLM Judge（最多重试1次）
            for attempt in range(2):  # 0=首次, 1=重试
                response = self.llm_judge_client.chat.completions.create(
                    model=self.llm_judge_model,
                    messages=[
                        {"role": "system", "content": "You are a precise answer equivalence evaluator."},
                        {"role": "user", "content": query_prompt}
                    ],
                    temperature=0.0,
                    max_tokens=200
                )

                # 检查响应是否为空
                content = response.choices[0].message.content
                if content is None:
                    if attempt == 0:
                        print(f"⚠️  LLM Judge首次返回空内容，重试中...")
                        continue  # 重试
                    else:
                        print(f"⚠️  LLM Judge重试后仍返回空内容，fallback判定为False")
                        return False

                # 成功获取内容，跳出重试循环
                result_text = content.strip()
                break

            # 解析<true_false>标签 - 增强的鲁棒性匹配
            import re
            # 匹配多种格式（按优先级尝试）：
            # 1. <true_false>True</true_false>
            # 2. <true_false>: True
            # 3. **true_false**: True
            # 4. true_false: True
            # 5. 直接在文本中查找True/False（最后手段）

            # 尝试1: 标准XML标签
            true_false_match = re.search(
                r'<true_false>\s*(True|False)\s*</true_false>',
                result_text,
                re.IGNORECASE
            )

            # 尝试2: 冒号分隔的标签
            if not true_false_match:
                true_false_match = re.search(
                    r'<true_false>\s*:\s*(True|False)',
                    result_text,
                    re.IGNORECASE
                )

            # 尝试3: Markdown粗体格式
            if not true_false_match:
                true_false_match = re.search(
                    r'\*\*true_false\*\*\s*:?\s*(True|False)',
                    result_text,
                    re.IGNORECASE
                )

            # 尝试4: 简单的key: value格式
            if not true_false_match:
                true_false_match = re.search(
                    r'true_false\s*:?\s*(True|False)',
                    result_text,
                    re.IGNORECASE
                )

            # 尝试5: 查找独立的True/False（最后手段）
            if not true_false_match:
                # 只在响应末尾查找，避免误匹配分析文本中的True/False
                last_200_chars = result_text[-200:]
                true_false_match = re.search(
                    r'\b(True|False)\b',
                    last_200_chars,
                    re.IGNORECASE
                )

            if true_false_match:
                verdict = true_false_match.group(1).lower() == "true"

                # 调试输出（20%采样）
                import random
                if random.random() < 0.2:
                    print(f"\n🤖 LLM Judge结果 ({problem_type}):")
                    print(f"  问题: {problem[:60]}...")
                    print(f"  预测: {str(prediction)[:60]}...")
                    print(f"  真值: {str(ground_truth)[:60]}...")
                    print(f"  判决: {verdict}")
                    print(f"  LLM响应: {result_text[:150]}...")

                return verdict
            else:
                # 完全无法解析时，打印完整响应用于调试
                print(f"⚠️  无法解析LLM Judge响应（尝试了5种格式）")
                print(f"  完整响应: {result_text}")
                return False

        except Exception as e:
            print(f"⚠️  LLM Judge调用失败: {e}")
            return False

    def compute_reward(
        self,
        problem: str,
        prediction: Any,
        ground_truth: Any,
        problem_type: str = "math",
        metadata: Optional[Dict] = None
    ) -> float:
        """
        计算奖励 - 支持LLM Judge和答案提取两种模式

        Returns:
            reward: 范围 [0.0, 1.0] (归一化后的奖励)
        """
        metadata = metadata or {}

        # 使用LLM Judge进行语义比较（所有任务类型）
        is_correct = self._llm_judge_compare(
            problem=problem,
            prediction=str(prediction),
            ground_truth=str(ground_truth),
            problem_type=problem_type
        )

        # 二元奖励：正确=10分，错误=-5分
        correctness_score = 10.0 if is_correct else -5.0

        if metadata is not None:
            metadata['correctness_score'] = correctness_score
            metadata['used_llm_judge'] = True

        # 归一化到[0, 1]用于GRPO
        # 使用简单的二元映射，避免复杂的sigmoid
        normalized_reward = 1.0 if is_correct else 0.0

        return normalized_reward

    def _is_correct(
        self,
        prediction: Any,
        ground_truth: Any,
        problem_type: str
    ) -> bool:
        """
        判断预测是否正确

        Returns:
            bool: True if correct, False otherwise
        """
        if prediction is None:
            return False

        if problem_type == "math":
            return self._is_math_correct(prediction, ground_truth)
        elif problem_type == "code":
            return self._is_code_correct(prediction, ground_truth)
        elif problem_type == "qa":
            return self._is_qa_correct(prediction, ground_truth)
        else:
            return self._is_general_correct(prediction, ground_truth)

    def _is_math_correct(self, prediction: str, ground_truth: str) -> bool:
        """
        判断数学答案是否正确

        支持:
        - 数字比较（含浮点误差）
        - 分数比较（如 5/324 vs 0.0154...）
        - 字符串匹配
        """
        try:
            pred_str = str(prediction).strip()
            gt_str = str(ground_truth).strip()

            # 字符串完全匹配
            if pred_str == gt_str:
                return True

            # 解析为数值比较（支持分数）
            def parse_number(s: str) -> float:
                """解析数字，支持分数格式"""
                if '/' in s:
                    parts = s.split('/')
                    return float(parts[0]) / float(parts[1])
                return float(s)

            try:
                pred_num = parse_number(pred_str)
                gt_num = parse_number(gt_str)

                # 使用相对误差比较（处理浮点精度）
                rel_error = abs(pred_num - gt_num) / (abs(gt_num) + 1e-9)
                return rel_error < 1e-6
            except:
                pass

            # 方法1: boxed 格式
            pred_boxed = self._extract_boxed(pred_str)
            gt_boxed = self._extract_boxed(gt_str)
            if pred_boxed and gt_boxed:
                try:
                    pred_num = parse_number(pred_boxed)
                    gt_num = parse_number(gt_boxed)
                    rel_error = abs(pred_num - gt_num) / (abs(gt_num) + 1e-9)
                    if rel_error < 1e-6:
                        return True
                except:
                    pass

            # 方法2: 数字提取
            pred_numbers = self._extract_numbers(pred_str)
            gt_numbers = self._extract_numbers(gt_str)

            if not gt_numbers:
                # 无法提取数字，用字符串匹配
                return gt_str.strip().lower() in pred_str.strip().lower()

            if not pred_numbers:
                return False

            # 比较最后一个数字
            pred_answer = pred_numbers[-1]
            gt_answer = gt_numbers[-1]

            return abs(pred_answer - gt_answer) < 1e-4

        except Exception:
            return False

    def _is_code_correct(self, prediction: str, ground_truth: str) -> bool:
        """判断代码答案是否正确"""
        try:
            pred_str = str(prediction).strip()
            gt_str = str(ground_truth).strip()

            if not pred_str:
                return False

            # 精确匹配
            if pred_str.lower() == gt_str.lower():
                return True

            # 包含匹配
            if gt_str.lower() in pred_str.lower():
                return True

            return False

        except Exception:
            return False

    def _is_qa_correct(self, prediction: str, ground_truth: str) -> bool:
        """判断QA答案是否正确"""
        try:
            pred_str = str(prediction).strip().lower()
            gt_str = str(ground_truth).strip().lower()

            # 精确匹配
            if pred_str == gt_str:
                return True

            # 包含匹配
            if gt_str in pred_str or pred_str in gt_str:
                return True

            # Token重叠阈值
            pred_tokens = set(pred_str.split())
            gt_tokens = set(gt_str.split())

            if len(gt_tokens) == 0:
                return False

            overlap_ratio = len(pred_tokens & gt_tokens) / len(gt_tokens)
            return overlap_ratio > 0.8

        except Exception:
            return False

    def _is_general_correct(self, prediction: str, ground_truth: str) -> bool:
        """通用正确性判断"""
        try:
            pred_str = str(prediction).strip().lower()
            gt_str = str(ground_truth).strip().lower()

            return pred_str == gt_str or gt_str in pred_str

        except Exception:
            return False

    def _compute_correctness_reward(
        self,
        prediction: Any,
        ground_truth: Any,
        problem_type: str
    ) -> float:
        """
        计算正确性奖励（保留用于向后兼容）

        Returns:
            reward: [-10, 10]
        """
        if prediction is None:
            return -10.0  # 执行失败

        if problem_type == "math":
            return self._compute_math_correctness(prediction, ground_truth)
        elif problem_type == "code":
            return self._compute_code_correctness(prediction, ground_truth)
        elif problem_type == "qa":
            return self._compute_qa_correctness(prediction, ground_truth)
        else:
            return self._compute_general_correctness(prediction, ground_truth)

    def _compute_math_correctness(self, prediction: str, ground_truth: str) -> float:
        """
        数学问题正确性(改进版 - 借鉴ROLL)

        改进:
        1. 支持LaTeX \boxed{}格式
        2. 更细粒度的评分阶梯
        3. 更好的数字提取
        """
        try:
            pred_str = str(prediction)
            gt_str = str(ground_truth)

            # 方法1: 提取boxed答案(ROLL风格)
            pred_boxed = self._extract_boxed(pred_str)
            gt_boxed = self._extract_boxed(gt_str)

            if pred_boxed and gt_boxed:
                try:
                    pred_num = float(pred_boxed)
                    gt_num = float(gt_boxed)
                    diff = abs(pred_num - gt_num)

                    if diff < 1e-4:
                        return 10.0   # 完全正确
                    elif diff < 0.1:
                        return 8.0    # 非常接近(新增阶梯)
                    elif diff < 1.0:
                        return 5.0    # 接近
                    elif diff < 10.0:
                        return 2.0    # 数量级正确(新增阶梯)
                    else:
                        return -5.0   # 错误
                except:
                    pass

            # 方法2: 数字提取(改进版)
            pred_numbers = self._extract_numbers(pred_str)
            gt_numbers = self._extract_numbers(gt_str)

            if not gt_numbers:
                # 无法提取ground truth数字,使用字符串匹配
                if gt_str.strip().lower() in pred_str.strip().lower():
                    return 10.0
                else:
                    return -5.0

            if not pred_numbers:
                # 无法提取预测数字
                return -8.0

            # 取最后一个数字作为答案
            pred_answer = pred_numbers[-1]
            gt_answer = gt_numbers[-1]

            # 比较(更细粒度)
            diff = abs(pred_answer - gt_answer)

            if diff < 1e-4:
                return 10.0   # 完全正确
            elif diff < 0.1:
                return 8.0    # 非常接近
            elif diff < 1.0:
                return 5.0    # 接近
            elif diff < 10.0:
                return 2.0    # 数量级正确
            else:
                return -5.0   # 错误

        except Exception as e:
            print(f"⚠️  数学评估错误: {e}")
            return -5.0

    def _extract_boxed(self, text: str) -> Optional[str]:
        """提取\boxed{}中的内容(ROLL风格)"""
        match = re.search(r'\\boxed\{([^}]+)\}', text)
        if match:
            return match.group(1).strip()
        return None

    def _extract_numbers(self, text: str) -> list:
        """从文本中提取所有数字(改进版 + 文字数字识别)"""
        numbers = []

        # Method 1: Numeric extraction (existing)
        # 匹配整数、小数、负数、科学计数法
        pattern = r'-?\d+\.?\d*(?:[eE][+-]?\d+)?'
        matches = re.findall(pattern, text)
        for m in matches:
            if m:
                try:
                    numbers.append(float(m))
                except:
                    pass

        # Method 2: Word-to-number recognition (NEW - fixes ~15-20% QA errors)
        # Aligns with SQuAD/HotpotQA standards for text-based answers
        word_to_num = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
            'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
            'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
            'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
            'eighteen': 18, 'nineteen': 19, 'twenty': 20, 'thirty': 30,
            'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70,
            'eighty': 80, 'ninety': 90, 'hundred': 100, 'thousand': 1000
        }

        text_lower = text.lower()
        for word, num in word_to_num.items():
            if word in text_lower:
                numbers.append(float(num))

        return numbers

    def _compute_code_correctness(self, prediction: str, ground_truth: str) -> float:
        """
        代码问题正确性(改进版)

        改进说明：
        - 区分fallback占位符 (返回-3.0) vs 真正的空预测 (返回-10.0)
        - fallback占位符表示至少尝试了，给予部分学习信号
        - 真正的空预测说明彻底失败，给予严厉惩罚
        """
        try:
            pred_str = str(prediction).strip()
            gt_str = str(ground_truth).strip()

            # 如果预测为空
            if not pred_str:
                return -10.0  # 彻底失败

            # 检查是否为fallback占位符
            if "[Fallback placeholder for problem:" in pred_str:
                # Fallback机制成功触发，至少返回了某些内容
                # 给予部分学习信号，而不是完全惩罚
                return -3.0

            # 完全匹配(最高分)
            if pred_str.lower() == gt_str.lower():
                return 10.0

            # 包含匹配
            if gt_str.lower() in pred_str.lower():
                return 10.0

            # 提取函数定义
            pred_funcs = self._extract_function_names(pred_str)
            gt_funcs = self._extract_function_names(gt_str)

            # 检查函数名是否匹配
            if pred_funcs and gt_funcs:
                if any(pf == gf for pf in pred_funcs for gf in gt_funcs):
                    return 5.0  # 部分正确

            # 检查是否至少包含Python代码特征
            if "def " in pred_str and ("return" in pred_str or "print" in pred_str):
                # 至少看起来像代码，给予中等惩罚
                return -2.0

            return -5.0

        except Exception as e:
            print(f"⚠️  代码评估错误: {e}")
            return -5.0

    def _extract_function_names(self, code: str) -> list:
        """从代码中提取函数名"""
        pattern = r'def\s+(\w+)\s*\('
        matches = re.findall(pattern, code)
        return matches

    def _compute_qa_correctness(self, prediction: str, ground_truth: str) -> float:
        """
        QA问题正确性(ROLL风格改进)
        """
        try:
            pred_str = str(prediction).strip().lower()
            gt_str = str(ground_truth).strip().lower()

            if not pred_str:
                return -10.0

            # 精确匹配
            if pred_str == gt_str:
                return 10.0

            # 包含匹配
            if gt_str in pred_str:
                return 8.0

            # Token重叠
            pred_tokens = set(pred_str.split())
            gt_tokens = set(gt_str.split())

            if not gt_tokens:
                return -5.0

            overlap_ratio = len(pred_tokens & gt_tokens) / len(gt_tokens)

            if overlap_ratio > 0.8:
                return 6.0
            elif overlap_ratio > 0.5:
                return 3.0
            elif overlap_ratio > 0.2:
                return 0.0
            else:
                return -5.0

        except Exception as e:
            print(f"⚠️  QA评估错误: {e}")
            return -5.0

    def _compute_general_correctness(self, prediction: str, ground_truth: str) -> float:
        """通用正确性评估"""
        return self._compute_qa_correctness(prediction, ground_truth)

    def _compute_efficiency_reward(self, cost: float) -> float:
        """
        计算效率奖励(基于API成本) - ROLL风格

        Returns:
            reward: [-8, 10]
        """
        if cost == 0.0:
            return 0.0

        # ROLL风格的成本阈值
        if cost <= 0.001:
            return 10.0
        elif cost <= 0.005:
            return 5.0
        elif cost <= 0.01:
            return 0.0
        elif cost <= 0.05:
            return -3.0
        else:
            return -8.0

    def _compute_simplicity_reward(
        self,
        execution_time: float,
        num_operators: int = 1
    ) -> float:
        """
        计算简洁性奖励 - ROLL风格

        Returns:
            reward: [-5, 10]
        """
        # 基于执行时间
        if execution_time <= 5.0:
            time_reward = 10.0
        elif execution_time <= 15.0:
            time_reward = 5.0
        elif execution_time <= 30.0:
            time_reward = 0.0
        elif execution_time <= 60.0:
            time_reward = -3.0
        else:
            time_reward = -5.0

        # 基于算子数量
        if num_operators <= 2:
            operator_reward = 10.0
        elif num_operators <= 4:
            operator_reward = 5.0
        elif num_operators <= 6:
            operator_reward = 0.0
        else:
            operator_reward = -5.0

        # 平均
        return (time_reward + operator_reward) / 2.0

    def _compute_format_reward(self, response: str, problem_type: str) -> float:
        """
        格式奖励(新增 - ROLL风格)

        检查响应格式规范性

        Returns:
            reward: [-2, 2]
        """
        if not response:
            return -2.0

        if problem_type == "math":
            # 检查是否有思考过程+答案
            has_think = bool(re.search(r'<think>.*?</think>', response, re.DOTALL))
            has_answer = bool(re.search(r'<answer>.*?</answer>', response, re.DOTALL))

            if has_think and has_answer:
                return 2.0    # 完美格式
            elif has_answer:
                return 0.0    # 基本格式
            else:
                return -2.0   # 格式混乱

        elif problem_type == "code":
            # 检查是否有代码块
            has_code_block = bool(re.search(r'```.*?```', response, re.DOTALL))

            if has_code_block:
                return 2.0
            else:
                return -2.0

        elif problem_type == "qa":
            # 检查答案长度合理性
            if 10 < len(response) < 500:
                return 2.0
            elif len(response) > 0:
                return 0.0
            else:
                return -2.0

        return 0.0

    def _compute_repetition_penalty(self, response: str, ngram_size: int = 3) -> float:
        """
        重复惩罚(新增 - ROLL风格)

        计算N-gram重复度并给予惩罚

        Args:
            response: 响应文本
            ngram_size: N-gram大小(默认3)

        Returns:
            penalty: [-2, 0]
        """
        if not response:
            return 0.0

        words = response.split()

        if len(words) < ngram_size:
            return 0.0

        # 生成所有N-grams
        ngrams = []
        for i in range(len(words) - ngram_size + 1):
            ngram = tuple(words[i:i+ngram_size])
            ngrams.append(ngram)

        if not ngrams:
            return 0.0

        # 计算唯一N-grams比例
        unique_ratio = len(set(ngrams)) / len(ngrams)

        # 转换为惩罚
        if unique_ratio > 0.9:
            return 0.0      # 几乎无重复
        elif unique_ratio > 0.7:
            return -0.5     # 轻微重复
        elif unique_ratio > 0.5:
            return -1.0     # 中度重复
        else:
            return -2.0     # 严重重复


def test_reward_computer():
    """测试改进版奖励计算器"""
    print("\n" + "=" * 60)
    print("🧪 测试改进版奖励计算器")
    print("=" * 60)

    computer = RewardComputer()

    # 测试案例
    test_cases = [
        {
            "name": "数学 - 完美格式+正确",
            "problem": "What is 15 + 27?",
            "prediction": "<think>Let me calculate: 15 + 27 = 42</think><answer>\\boxed{42}</answer>",
            "ground_truth": "42",
            "problem_type": "math",
            "metadata": {"cost": 0.002, "execution_time": 3.5}
        },
        {
            "name": "数学 - 正确但无格式",
            "problem": "What is 15 + 27?",
            "prediction": "The answer is 42.",
            "ground_truth": "42",
            "problem_type": "math",
            "metadata": {"cost": 0.002, "execution_time": 3.0}
        },
        {
            "name": "数学 - 接近答案",
            "problem": "What is 15 + 27?",
            "prediction": "<think>Calculating</think><answer>42.1</answer>",
            "ground_truth": "42",
            "problem_type": "math",
            "metadata": {"cost": 0.001, "execution_time": 2.0}
        },
        {
            "name": "代码 - 正确+格式",
            "problem": "Write a function to square a number",
            "prediction": "```python\ndef square(x):\n    return x * x\n```",
            "ground_truth": "def square(x):\n    return x * x",
            "problem_type": "code",
            "metadata": {"cost": 0.003, "execution_time": 5.0}
        },
        {
            "name": "QA - 正确",
            "problem": "What is the capital of France?",
            "prediction": "The capital of France is Paris.",
            "ground_truth": "Paris",
            "problem_type": "qa",
            "metadata": {"cost": 0.001, "execution_time": 2.0}
        },
        {
            "name": "严重重复",
            "problem": "Test",
            "prediction": "answer answer answer answer answer answer",
            "ground_truth": "answer",
            "problem_type": "qa",
            "metadata": {"cost": 0.001, "execution_time": 1.0}
        }
    ]

    for case in test_cases:
        reward = computer.compute_reward(
            problem=case["problem"],
            prediction=case["prediction"],
            ground_truth=case["ground_truth"],
            problem_type=case["problem_type"],
            metadata=case["metadata"]
        )

        print(f"\n📝 {case['name']}")
        print(f"  预测: {case['prediction'][:60]}...")
        print(f"  正确答案: {case['ground_truth']}")
        print(f"  奖励: {reward:.2f}/10.0")


if __name__ == "__main__":
    test_reward_computer()
