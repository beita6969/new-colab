# 奖励函数对比分析与改进方案

## 📊 当前训练状态 (2025-11-16)

**训练进度:** Step 3/500 (0.6%)

**准确率统计:**
- **Step 1:** 3/16 = 18.8%
- **Step 2:** 6/16 = 37.5% ✅ **提升19%**
- **Step 3:** 数据收集中...

**质量指标:**
- 格式正确率: 97.1% (34/35)
- Fallback使用率: 0.0%
- eval()错误率: 8.6%
- 平均API成本: $0.000091/调用

**结论:** 训练刚开始,模型正在快速学习中,准确率从18.8%提升到37.5%显示了良好的学习趋势。

---

## 🔍 奖励函数对比分析

### 1. 当前系统 (integrated_aflow_roll)

**文件:** `src/reward_computer.py` (185行)

**奖励公式:**
```python
total_reward = 0.7 × correctness + 0.2 × efficiency + 0.1 × simplicity
```

**维度分析:**

| 维度 | 权重 | 范围 | 计算方法 | 优点 | 缺点 |
|------|------|------|---------|------|------|
| **Correctness** | 70% | [-10, 10] | 数学:数字匹配<br>代码:字符串包含<br>QA:Token重叠 | 简单快速 | 粗粒度评估 |
| **Efficiency** | 20% | [-8, 10] | 基于API成本分级 | 鼓励低成本 | 未考虑执行时间 |
| **Simplicity** | 10% | [-5, 10] | 执行时间+算子数 | 鼓励简洁 | 权重可能过低 |

**特点:**
- ✅ 简单明确,易于理解
- ✅ 快速计算,无需额外API调用
- ✅ 多维度覆盖(结果+效率+质量)
- ❌ 正确性评估粗糙(字符串匹配)
- ❌ 无过程奖励(process reward)
- ❌ 权重硬编码,无法自适应

---

### 2. ROLL框架

**文件:** `ROLL/roll/pipeline/rlvr/rewards/` (8个Worker,4408行)

**架构特点:**
```
多域路由系统
├── MathRuleRewardWorker (272行) - 数学题专用
├── CodeSandboxRewardWorker (865行) - 代码执行沙箱
├── LLMJudgeRewardWorker (249行) - LLM智能评判
├── IFEvalRuleRewardWorker (700行) - 指令遵循(27个约束函数)
├── CrossThinkQARewardWorker (266行) - 推理QA
├── GeneralValRewardWorker (189行) - 通用验证
├── MultipleChoiceRewardWorker (164行) - 选择题
└── DetectionRewardWorker (704行) - 目标检测
```

**MathRuleRewardWorker详细分析:**

| 维度 | 权重 | 计算方法 | 示例 |
|------|------|---------|------|
| **verify_answer** | 主要 | math_verify库验证(LaTeX+表达式) | 0/1二值 |
| **repetition_penalty** | 辅助 | N-gram重复度(ngram=3) | -0.1 max |
| **format_reward** | 辅助 | 正则匹配`<think>...</think><answer>...</answer>` | 0/-1 |
| **long_block_penalty** | 辅助 | 单词最大长度>100 | -1 |
| **response_length** | 归一化 | len(response)/20000 | [0,1] |

**公式:**
```python
response_level_reward = verify_answer + repetition_penalty + format_reward
token_level_reward = 全0 (稀疏奖励设计)
```

**CodeSandboxRewardWorker详细分析:**

**核心能力:** 真实代码执行测试

| 测试类型 | 描述 | 示例 |
|---------|------|------|
| **Input/Output** | stdin→stdout对比 | `"input": "5\n", "expected": "25\n"` |
| **Assert测试** | Python assert语句 | `assert solution(5) == 25` |
| **Pytest集成** | 完整pytest框架 | 支持fixtures和参数化 |
| **Check函数** | 自定义检查函数 | `def check(candidate): ...` |

**奖励计算:**
```python
reward = pass_test_ratio (通过测试数/总测试数)
+ format_validation (0/1)
+ think_tag_check (0/1)
- error_penalty (SyntaxError=-1, LogicError=-2)
```

**错误分类系统:**
- ✅ `SyntaxError`: 编译错误 → -1惩罚
- ✅ `LogicError`: 逻辑错误 → -2惩罚
- ✅ `ReturnCode`: 运行时错误 → 记录code

**ROLL的优势:**
- ✅ **域特异性设计:** 为每种任务类型定制专用Worker
- ✅ **真实执行验证:** 代码沙箱实际运行测试用例
- ✅ **细粒度评估:** 27个指令约束函数
- ✅ **配置驱动:** 通过tag自动路由到对应Worker
- ✅ **对抗鲁棒性:** 检测BPE攻击(长文本块)
- ✅ **多维度组合:** 正确性+格式+重复+长度

**ROLL的局限:**
- ❌ **复杂度高:** 4408行代码,维护成本大
- ❌ **硬编码权重:** 各维度权重不可学习
- ❌ **无过程奖励:** token_level全为0
- ❌ **计算开销:** 代码沙箱执行耗时

---

### 3. AgentFlow框架

**文件:** `AgentFlow/agentflow/reward.py` (67行)

**设计哲学:** 装饰器模式 + LLM评判

**核心代码:**
```python
@reward
def compute_reward(response: str, correct_answer: str) -> float:
    """
    装饰器自动处理:
    1. 异步/同步函数兼容
    2. 返回值类型验证(float/int/None)
    3. AgentOps追踪集成
    4. RewardSpanData封装
    """
    return 1.0 if response == correct_answer else 0.0
```

**LLM评判系统 (calculate_score_unified.py):**

```python
class ResultScorer:
    def __init__(self):
        # 使用GPT-4o作为评判模型
        self.llm_engine = ChatOpenAI(
            model_string="gpt-4o",
            is_multimodal=False,
            enable_cache=True  # 缓存优化
        )

    def answer_verification(self, question, response, correct_answer):
        """
        评判流程:
        1. 提取<answer>标签内容
        2. GPT-4o判断是否正确
        3. 返回分析+布尔判断
        """
        prompt = f"""
        Question: {question}
        Response: {response}
        Correct Answer: {correct_answer}

        Is the response correct? (Yes/No)
        Provide analysis.
        """

        llm_result = self.llm_engine.call(prompt)
        return parse_yes_no(llm_result)
```

**并行评分系统:**
```python
def score_results(self, results, max_workers=10):
    """
    特点:
    - ThreadPoolExecutor并行处理
    - 最多10个worker同时评分
    - 进度条实时显示
    - 支持缓存避免重复调用
    """
    with ThreadPoolExecutor(max_workers) as executor:
        futures = [
            executor.submit(self.answer_verification, r)
            for r in results
        ]

        for future in tqdm(as_completed(futures)):
            result = future.result()
            # 统计正确/错误
```

**AgentFlow的优势:**
- ✅ **极简设计:** 核心只有67行代码
- ✅ **智能评判:** GPT-4o理解复杂推理
- ✅ **高扩展性:** 装饰器模式易于添加新奖励
- ✅ **深度集成:** 与AgentOps无缝追踪
- ✅ **缓存优化:** 避免重复LLM调用
- ✅ **并行处理:** 10 workers同时评分

**AgentFlow的局限:**
- ❌ **API依赖:** 需要调用GPT-4o,成本高
- ❌ **离线评估:** 不适合在线RL训练
- ❌ **单一维度:** 主要只有正确性评判
- ❌ **延迟问题:** LLM调用耗时(2-5秒/样本)

---

## 📈 三者对比总结表

| 特性 | 当前系统 | ROLL | AgentFlow |
|------|---------|------|-----------|
| **代码规模** | 185行 | 4408行 | 67行 |
| **奖励维度** | 3个(固定) | 5-8个(可配置) | 1个(可扩展) |
| **评估方法** | 规则匹配 | 规则+沙箱执行 | LLM评判 |
| **计算速度** | 快(ms级) | 中(秒级) | 慢(2-5秒) |
| **准确性** | 中等 | 高 | 最高 |
| **适用场景** | 在线RL训练 | 大规模多域RL | 离线分析 |
| **API成本** | 低 | 低 | 高(GPT-4o) |
| **可扩展性** | 中 | 低(需加Worker) | 高(装饰器) |
| **过程奖励** | 无 | 无(token_level全0) | 可选 |
| **权重学习** | 无 | 无 | 无 |
| **多域支持** | 手动if | 自动路由 | 手动注册 |

---

## 🚀 改进建议方案

### 方案A: 渐进式改进(推荐优先实施)

**目标:** 在当前系统基础上,吸收ROLL和AgentFlow的优点

#### A1. 改进正确性评估(借鉴ROLL)

**当前问题:** 数学题只提取最后一个数字,代码题只做字符串包含

**改进方案:**
```python
class ImprovedCorrectnessEvaluator:
    """改进的正确性评估器"""

    def __init__(self):
        # 添加math_verify库
        from sympy import sympify, simplify
        self.math_verifier = self._math_verify

    def _math_verify(self, pred_str: str, gt_str: str) -> float:
        """
        ROLL风格的数学验证
        支持:
        1. 数字提取(保留当前方法)
        2. LaTeX表达式解析
        3. 符号表达式验证
        """
        try:
            # 方法1: 数字提取(快速)
            pred_nums = self._extract_numbers(pred_str)
            gt_nums = self._extract_numbers(gt_str)

            if pred_nums and gt_nums:
                if abs(pred_nums[-1] - gt_nums[-1]) < 1e-4:
                    return 10.0

            # 方法2: LaTeX解析(精确)
            pred_expr = self._parse_latex(pred_str)
            gt_expr = self._parse_latex(gt_str)

            if pred_expr and gt_expr:
                if simplify(pred_expr - gt_expr) == 0:
                    return 10.0

            # 方法3: 字符串相似度(兜底)
            if self._string_similarity(pred_str, gt_str) > 0.9:
                return 8.0

            return -5.0

        except Exception as e:
            # 降级到原始方法
            return self._original_math_correctness(pred_str, gt_str)

    def _parse_latex(self, text: str):
        """提取并解析LaTeX表达式"""
        # 提取 \boxed{...} 或 $...$
        import re

        boxed = re.search(r'\\boxed\{([^}]+)\}', text)
        if boxed:
            return sympify(boxed.group(1))

        dollar = re.search(r'\$([^$]+)\$', text)
        if dollar:
            return sympify(dollar.group(1))

        return None
```

**预期效果:**
- ✅ 支持更多数学表达式格式
- ✅ 提升数学题评估准确性
- ✅ 保持快速计算速度
- **实施成本:** 低(1-2小时)

---

#### A2. 添加格式奖励维度(借鉴ROLL)

**当前问题:** 无格式检查,模型可能生成混乱输出

**改进方案:**
```python
def _compute_format_reward(self, response: str, problem_type: str) -> float:
    """
    检查响应格式规范性

    返回:
        +2.0: 完美格式
        +0.0: 基本格式
        -2.0: 格式混乱
    """

    if problem_type == "math":
        # 检查是否有思考过程+答案
        has_think = bool(re.search(r'<think>.*?</think>', response, re.DOTALL))
        has_answer = bool(re.search(r'<answer>.*?</answer>', response, re.DOTALL))

        if has_think and has_answer:
            return 2.0
        elif has_answer:
            return 0.0
        else:
            return -2.0

    elif problem_type == "code":
        # 检查是否有代码块
        has_code_block = bool(re.search(r'```python.*?```', response, re.DOTALL))

        if has_code_block:
            return 2.0
        else:
            return -2.0

    elif problem_type == "qa":
        # 检查答案长度合理性
        if 10 < len(response) < 500:
            return 2.0
        else:
            return 0.0

    return 0.0
```

**集成到总奖励:**
```python
# 更新reward_computer.py的compute_reward方法
total_reward = (
    0.65 * correctness_reward +      # 降低5%给格式
    0.20 * efficiency_reward +
    0.10 * simplicity_reward +
    0.05 * format_reward             # 新增5%
)
```

**预期效果:**
- ✅ 鼓励模型遵循格式规范
- ✅ 提升输出可读性
- ✅ 减少解析错误
- **实施成本:** 低(1小时)

---

#### A3. 添加重复惩罚(借鉴ROLL)

**当前问题:** 模型可能生成大量重复内容获取长度奖励

**改进方案:**
```python
def _compute_repetition_penalty(self, response: str, ngram_size: int = 3) -> float:
    """
    计算N-gram重复度惩罚

    Args:
        response: 响应文本
        ngram_size: N-gram大小(默认3)

    Returns:
        惩罚值: [-2.0, 0.0]
    """
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
        return 0.0          # 几乎无重复
    elif unique_ratio > 0.7:
        return -0.5         # 轻微重复
    elif unique_ratio > 0.5:
        return -1.0         # 中度重复
    else:
        return -2.0         # 严重重复
```

**预期效果:**
- ✅ 防止模型生成重复内容
- ✅ 鼓励多样化表达
- ✅ 避免reward hacking
- **实施成本:** 低(1小时)

---

#### A4. 可选LLM评判(借鉴AgentFlow)

**应用场景:** 对准确率要求极高的场景,或离线验证

**改进方案:**
```python
class OptionalLLMJudge:
    """可选的LLM评判器"""

    def __init__(self, enable: bool = False, model: str = "gpt-4o-mini"):
        self.enable = enable
        self.model = model

        if enable:
            from openai import OpenAI
            self.client = OpenAI()

    def judge(self, question: str, response: str, ground_truth: str) -> float:
        """
        使用LLM评判答案质量

        Returns:
            [0, 10] 的评分
        """
        if not self.enable:
            return None

        prompt = f"""
You are an expert evaluator. Rate the response quality on scale 0-10.

Question: {question}

Response: {response}

Ground Truth: {ground_truth}

Evaluate based on:
1. Correctness (most important)
2. Reasoning quality
3. Clarity

Format: Score: X.X
"""

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )

            result = completion.choices[0].message.content

            # 提取分数
            import re
            match = re.search(r'Score:\s*([0-9.]+)', result)
            if match:
                return float(match.group(1))

        except Exception as e:
            print(f"LLM judge error: {e}")

        return None

# 在reward_computer.py中集成
class RewardComputer:
    def __init__(self, reward_weights, use_llm_judge: bool = False):
        self.llm_judge = OptionalLLMJudge(enable=use_llm_judge)

    def compute_reward(self, problem, prediction, ground_truth, problem_type, metadata):
        # 原有奖励计算
        rule_based_reward = ...

        # 可选LLM评判
        if self.llm_judge.enable:
            llm_score = self.llm_judge.judge(problem, prediction, ground_truth)

            if llm_score is not None:
                # 混合：70% 规则 + 30% LLM
                final_reward = 0.7 * rule_based_reward + 0.3 * llm_score
                return final_reward

        return rule_based_reward
```

**使用建议:**
- 训练时: `use_llm_judge=False` (快速)
- 验证时: `use_llm_judge=True` (准确)

**预期效果:**
- ✅ 离线评估更准确
- ✅ 可验证规则奖励的质量
- ⚠️  训练时成本高,不推荐
- **实施成本:** 中(2-3小时)

---

### 方案B: 代码执行验证(借鉴ROLL,高级)

**目标:** 为代码题添加真实执行测试

**当前问题:** 代码题只做字符串包含,无法验证代码正确性

**改进方案:**
```python
class CodeExecutionValidator:
    """代码执行验证器"""

    def __init__(self, timeout: int = 5):
        self.timeout = timeout

    def validate_code(self, workflow_output: str, test_cases: List[Dict]) -> float:
        """
        执行代码并测试

        Args:
            workflow_output: 工作流输出(包含代码)
            test_cases: 测试用例列表
                [
                    {"input": "2 3", "expected_output": "5"},
                    {"assert": "assert solution(2, 3) == 5"}
                ]

        Returns:
            通过率 [0, 1]
        """
        # 提取代码
        code = self._extract_code(workflow_output)

        if not code:
            return 0.0

        # 运行测试
        passed = 0
        total = len(test_cases)

        for test in test_cases:
            try:
                if 'input' in test and 'expected_output' in test:
                    # Input/Output测试
                    result = self._run_with_io(
                        code,
                        test['input'],
                        timeout=self.timeout
                    )

                    if result.strip() == test['expected_output'].strip():
                        passed += 1

                elif 'assert' in test:
                    # Assert测试
                    success = self._run_assert(
                        code,
                        test['assert'],
                        timeout=self.timeout
                    )

                    if success:
                        passed += 1

            except Exception as e:
                # 测试失败
                continue

        return passed / total if total > 0 else 0.0

    def _extract_code(self, text: str) -> str:
        """提取代码块"""
        import re

        # 提取```python ... ```
        match = re.search(r'```python\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            return match.group(1)

        # 提取```... ```
        match = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            return match.group(1)

        return ""

    def _run_with_io(self, code: str, input_str: str, timeout: int) -> str:
        """执行代码并捕获输出"""
        import subprocess
        import tempfile

        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_path = f.name

        try:
            # 执行
            result = subprocess.run(
                ['python3', temp_path],
                input=input_str,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            return result.stdout

        except subprocess.TimeoutExpired:
            return ""

        finally:
            import os
            os.unlink(temp_path)

    def _run_assert(self, code: str, assert_stmt: str, timeout: int) -> bool:
        """执行断言测试"""
        import subprocess
        import tempfile

        # 组合代码和断言
        full_code = code + "\n\n" + assert_stmt

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(full_code)
            temp_path = f.name

        try:
            result = subprocess.run(
                ['python3', temp_path],
                capture_output=True,
                timeout=timeout
            )

            # 返回码0表示断言通过
            return result.returncode == 0

        except:
            return False

        finally:
            import os
            os.unlink(temp_path)

# 集成到RewardComputer
class RewardComputer:
    def __init__(self, reward_weights, enable_code_execution: bool = False):
        self.code_validator = CodeExecutionValidator() if enable_code_execution else None

    def _compute_code_correctness(self, prediction: str, ground_truth: str, test_cases: List[Dict] = None) -> float:
        """改进的代码正确性评估"""

        # 方法1: 代码执行测试(如果有测试用例)
        if self.code_validator and test_cases:
            pass_rate = self.code_validator.validate_code(prediction, test_cases)

            if pass_rate == 1.0:
                return 10.0       # 所有测试通过
            elif pass_rate >= 0.8:
                return 7.0        # 大部分通过
            elif pass_rate >= 0.5:
                return 4.0        # 一半通过
            elif pass_rate > 0:
                return 1.0        # 部分通过
            else:
                return -5.0       # 全部失败

        # 方法2: 字符串匹配(原有方法,兜底)
        if ground_truth.lower() in prediction.lower():
            return 10.0

        # 方法3: 函数名匹配
        pred_funcs = re.findall(r'def\s+(\w+)\s*\(', prediction)
        gt_funcs = re.findall(r'def\s+(\w+)\s*\(', ground_truth)

        if pred_funcs and gt_funcs and pred_funcs[0] == gt_funcs[0]:
            return 5.0

        return -5.0
```

**数据格式扩展:**

需要在数据集中添加test_cases字段:
```json
{
  "problem": "Write a function to add two numbers",
  "problem_type": "code",
  "ground_truth": "def add(a, b):\n    return a + b",
  "test_cases": [
    {"assert": "assert add(2, 3) == 5"},
    {"assert": "assert add(0, 0) == 0"},
    {"assert": "assert add(-1, 1) == 0"}
  ]
}
```

**预期效果:**
- ✅ 代码题评估准确性大幅提升
- ✅ 真实执行验证代码正确性
- ⚠️  执行耗时增加(每题5秒timeout)
- ⚠️  需要安全沙箱(防止恶意代码)
- **实施成本:** 高(1-2天)

---

### 方案C: 自适应权重学习(高级)

**目标:** 让奖励函数权重随训练动态调整

**当前问题:** 权重硬编码为 {correctness:0.7, efficiency:0.2, simplicity:0.1}

**改进方案:**
```python
class AdaptiveRewardWeighting:
    """自适应奖励权重学习"""

    def __init__(self, initial_weights: Dict[str, float] = None):
        self.weights = initial_weights or {
            'correctness': 0.7,
            'efficiency': 0.2,
            'simplicity': 0.1
        }

        # 权重历史
        self.weight_history = []

        # 性能指标历史
        self.performance_history = []

    def update_weights(self, step: int, accuracy: float, avg_reward: float):
        """
        根据训练性能动态调整权重

        策略:
        1. 训练早期(step < 100): 提高correctness权重
        2. 训练中期(100-300): 平衡三者
        3. 训练后期(step > 300): 提高efficiency权重
        """

        # 记录性能
        self.performance_history.append({
            'step': step,
            'accuracy': accuracy,
            'avg_reward': avg_reward
        })

        # 动态调整策略
        if step < 100:
            # 早期: 专注正确性
            self.weights = {
                'correctness': 0.8,
                'efficiency': 0.1,
                'simplicity': 0.1
            }

        elif step < 300:
            # 中期: 平衡
            if accuracy < 0.5:
                # 准确率低,继续提高correctness
                self.weights['correctness'] = 0.75
                self.weights['efficiency'] = 0.15
                self.weights['simplicity'] = 0.10
            else:
                # 准确率高,开始优化效率
                self.weights['correctness'] = 0.65
                self.weights['efficiency'] = 0.25
                self.weights['simplicity'] = 0.10

        else:
            # 后期: 优化效率和简洁性
            self.weights['correctness'] = 0.60
            self.weights['efficiency'] = 0.30
            self.weights['simplicity'] = 0.10

        # 记录权重
        self.weight_history.append({
            'step': step,
            'weights': self.weights.copy()
        })

    def get_weights(self) -> Dict[str, float]:
        """获取当前权重"""
        return self.weights.copy()

# 集成到GRPO训练器
class GRPOTrainer:
    def __init__(self, config_path):
        # ... 原有初始化 ...

        # 添加自适应权重
        self.adaptive_weights = AdaptiveRewardWeighting()

    async def train_step(self, step: int):
        # ... 执行工作流,计算奖励 ...

        # 更新权重
        self.adaptive_weights.update_weights(
            step=step,
            accuracy=accuracy,
            avg_reward=np.mean(all_rewards)
        )

        # 应用新权重到reward_computer
        new_weights = self.adaptive_weights.get_weights()
        self.reward_computer.reward_weights = new_weights

        print(f"📊 当前权重: {new_weights}")
```

**预期效果:**
- ✅ 训练早期专注学习正确答案
- ✅ 训练后期优化效率和质量
- ✅ 适应不同训练阶段的需求
- **实施成本:** 中(3-4小时)

---

## 📋 实施优先级和路线图

### Phase 1: 快速改进(1周内)

**优先级:** ⭐⭐⭐⭐⭐

1. **A1. 改进正确性评估** (1-2小时)
   - 添加LaTeX解析
   - 支持更多数学表达式格式

2. **A2. 添加格式奖励** (1小时)
   - 检查<think>/<answer>标签
   - 检查代码块格式

3. **A3. 添加重复惩罚** (1小时)
   - N-gram重复检测
   - 防止reward hacking

**预期提升:** 准确率 +10-15%

---

### Phase 2: 中级优化(2-3周内)

**优先级:** ⭐⭐⭐⭐

4. **A4. 可选LLM评判** (2-3小时)
   - 集成GPT-4o-mini评判
   - 仅用于离线验证

5. **C. 自适应权重学习** (3-4小时)
   - 根据训练阶段动态调整
   - 记录权重变化历史

**预期提升:** 准确率 +5-10%

---

### Phase 3: 高级功能(1-2个月)

**优先级:** ⭐⭐⭐

6. **B. 代码执行验证** (1-2天)
   - 真实代码执行沙箱
   - 测试用例验证

7. **过程奖励(Process Reward)** (3-5天)
   - 评估推理步骤质量
   - Token级别奖励设计

**预期提升:** 准确率 +10-20% (尤其是代码题)

---

## 💡 立即可实施的Quick Wins

### 1. 调整当前权重(0成本)

**建议修改 `config/training.yaml`:**
```yaml
reward_weights:
  correctness: 0.75    # 从0.7提升到0.75
  efficiency: 0.15     # 从0.2降低到0.15
  simplicity: 0.10     # 保持0.1
```

**理由:** 训练初期应更重视正确性

---

### 2. 降低correctness阈值(10分钟)

**修改 `src/reward_computer.py`:**
```python
# 当前阈值过严
def _compute_math_correctness(self, prediction: str, ground_truth: str) -> float:
    # 旧代码
    if abs(pred_answer - gt_answer) < 1e-4:
        return 10.0
    elif abs(pred_answer - gt_answer) < 1.0:    # 阈值1.0
        return 5.0
    else:
        return -5.0

# 改进建议
def _compute_math_correctness(self, prediction: str, ground_truth: str) -> float:
    diff = abs(pred_answer - gt_answer)

    if diff < 1e-4:
        return 10.0       # 完全正确
    elif diff < 0.1:      # 新增: 非常接近
        return 8.0
    elif diff < 1.0:
        return 5.0        # 接近
    elif diff < 10.0:     # 新增: 数量级正确
        return 2.0
    else:
        return -5.0       # 完全错误
```

**预期效果:** 对接近正确的答案给予部分奖励,加速学习

---

## 📊 监控和验证

**添加详细的奖励分解日志:**

```python
# 在grpo_trainer.py中
print(f"""
🎯 奖励分解:
  - 正确性: {correctness:.2f}/10.0 (权重70%)
  - 效率:   {efficiency:.2f}/10.0 (权重20%)
  - 简洁性: {simplicity:.2f}/10.0 (权重10%)
  - 总奖励: {total_reward:.2f}/10.0
""")
```

**添加奖励分布统计:**
```python
# 在analyze_training.py中
def analyze_reward_distribution(self):
    """分析各维度奖励分布"""

    # 提取各维度分数
    correctness_scores = re.findall(r'正确性: ([\d.-]+)/10\.0', content)
    efficiency_scores = re.findall(r'效率: ([\d.-]+)/10\.0', content)

    print(f"\n📊 奖励分布:")
    print(f"  正确性: μ={np.mean(correctness):.2f}, σ={np.std(correctness):.2f}")
    print(f"  效率:   μ={np.mean(efficiency):.2f}, σ={np.std(efficiency):.2f}")
```

---

## 🎯 总结

### 当前系统优势
- ✅ 简洁高效(185行代码)
- ✅ 适合在线RL训练
- ✅ 多维度覆盖(结果+效率+质量)

### 主要改进方向
1. **正确性评估** - 借鉴ROLL的math_verify和LaTeX解析
2. **格式规范** - 添加格式检查奖励维度
3. **防reward hacking** - 添加重复惩罚
4. **代码验证** - 可选的真实执行测试
5. **自适应权重** - 根据训练阶段动态调整

### 推荐实施顺序
```
Phase 1 (立即): A1 + A2 + A3 → 预期准确率 37.5% → 50%+
Phase 2 (1周): A4 + C → 预期准确率 50% → 60%+
Phase 3 (1月): B + 过程奖励 → 预期准确率 60% → 75%+
```

### 风险和注意事项
- ⚠️  代码执行需要安全沙箱(docker/firejail)
- ⚠️  LLM评判成本高,仅用于验证
- ⚠️  权重调整需要A/B测试验证效果
- ⚠️  过程奖励设计复杂,需要大量实验

---

**文档版本:** v1.0
**创建时间:** 2025-11-16
**下次更新:** 实施Phase 1后
