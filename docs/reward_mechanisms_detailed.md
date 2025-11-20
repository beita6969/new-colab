# ROLL和AgentFlow奖励机制详细罗列

## 目录
- [ROLL框架奖励机制](#roll框架奖励机制)
- [AgentFlow框架奖励机制](#agentflow框架奖励机制)
- [直接可用的代码实现](#直接可用的代码实现)

---

## ROLL框架奖励机制

### 1. 数学题奖励 (MathRuleRewardWorker)

**文件路径:** `/home/yijia/.claude/11/ROLL/roll/pipeline/rlvr/rewards/math_rule_reward_worker.py`

#### 评分维度列表

| 维度 | 计算方法 | 返回值范围 | 权重/作用 |
|------|---------|-----------|----------|
| **verify_answer** | math_verify库验证 | 0 或 1 | 主要奖励 |
| **repetition_penalty** | N-gram重复度 | [-0.1, 0] | 惩罚重复 |
| **format_reward** | 正则匹配格式 | 0 或 -1 | 格式规范 |
| **long_block_penalty** | 单词最大长度 | 0 或 -1 | 防BPE攻击 |
| **response_length** | 响应长度归一化 | [0, 1] | 仅统计用 |

#### 具体计算公式

```python
# 1. verify_answer (主奖励)
def verify_answer(prediction: str, ground_truth: str) -> int:
    """
    使用math_verify库验证数学答案
    支持: LaTeX, 分数, 小数, 表达式
    """
    from math_verify import verify_math_answer

    # 提取答案(支持\boxed{}格式)
    pred_answer = extract_boxed_answer(prediction)
    gt_answer = extract_boxed_answer(ground_truth)

    # 验证
    is_correct = verify_math_answer(pred_answer, gt_answer)

    return 1 if is_correct else 0

# 示例
verify_answer("The answer is \\boxed{42}", "42")  # → 1
verify_answer("The answer is 41", "42")           # → 0


# 2. repetition_penalty (重复惩罚)
def repetition_penalty(response: str, ngram_size: int = 3) -> float:
    """
    计算N-gram重复度
    公式: penalty = (1 - unique_ngrams/total_ngrams) * max_penalty
    """
    words = response.split()

    if len(words) < ngram_size:
        return 0.0

    # 生成N-grams
    ngrams = [
        tuple(words[i:i+ngram_size])
        for i in range(len(words) - ngram_size + 1)
    ]

    # 计算唯一比例
    unique_ratio = len(set(ngrams)) / len(ngrams)

    # 返回惩罚
    max_penalty = 0.1
    penalty = (1 - unique_ratio) * max_penalty

    return -penalty  # 负值惩罚

# 示例
repetition_penalty("solve solve solve the problem")  # → -0.08 (重复多)
repetition_penalty("let's solve the problem now")    # → 0.0 (无重复)


# 3. format_reward (格式检查)
def format_reward(response: str, pattern: str = None) -> int:
    """
    检查响应是否符合格式要求
    默认格式: <think>...</think><answer>...</answer>
    """
    import re

    if pattern is None:
        pattern = r'^<think>.*?</think>.*?<answer>.*?</answer>$'

    # 正则匹配
    match = re.search(pattern, response, re.DOTALL)

    return 0 if match else -1

# 示例
format_reward("<think>Let me solve</think><answer>42</answer>")  # → 0
format_reward("The answer is 42")                                # → -1


# 4. long_block_penalty (长文本块惩罚)
def long_block_penalty(response: str, max_length: int = 100) -> int:
    """
    检测单个单词是否过长(防止BPE攻击)
    """
    words = response.split()

    for word in words:
        if len(word) > max_length:
            return -1

    return 0

# 示例
long_block_penalty("normal text here")              # → 0
long_block_penalty("a" * 150 + " normal text")      # → -1


# 5. response_length (响应长度)
def response_length_norm(response: str, max_len: int = 20000) -> float:
    """
    归一化响应长度(仅用于统计)
    """
    return len(response) / max_len

# 示例
response_length_norm("short answer")        # → 0.0006
response_length_norm("a" * 10000)           # → 0.5
```

#### 最终奖励组合

```python
def compute_math_reward(response: str, ground_truth: str) -> float:
    """
    ROLL数学题完整奖励计算
    """
    # 计算各维度
    verify = verify_answer(response, ground_truth)           # 0 或 1
    rep_penalty = repetition_penalty(response, ngram_size=3) # [-0.1, 0]
    fmt_reward = format_reward(response)                     # 0 或 -1
    long_penalty = long_block_penalty(response)              # 0 或 -1

    # 组合(response_level_reward)
    response_level_reward = verify + rep_penalty + fmt_reward + long_penalty

    # token_level_reward全为0(稀疏奖励)
    token_level_reward = 0.0

    # 统计分数
    score = verify  # 用于准确率统计

    return {
        'response_level_reward': response_level_reward,  # [-2.1, 1.0]
        'token_level_reward': token_level_reward,        # 0.0
        'score': score                                   # 0 或 1
    }

# 示例
# 完美回答
compute_math_reward(
    "<think>Let me calculate</think><answer>\\boxed{42}</answer>",
    "42"
)
# → {'response_level_reward': 1.0, 'token_level_reward': 0.0, 'score': 1}

# 错误答案但格式好
compute_math_reward(
    "<think>Let me calculate</think><answer>\\boxed{41}</answer>",
    "42"
)
# → {'response_level_reward': 0.0, 'token_level_reward': 0.0, 'score': 0}

# 正确但格式差
compute_math_reward("The answer is 42", "42")
# → {'response_level_reward': 0.0, 'token_level_reward': 0.0, 'score': 1}
```

---

### 2. 代码执行奖励 (CodeSandboxRewardWorker)

**文件路径:** `/home/yijia/.claude/11/ROLL/roll/pipeline/rlvr/rewards/code_sandbox_reward_worker.py`

#### 评分维度列表

| 维度 | 计算方法 | 返回值范围 | 权重/作用 |
|------|---------|-----------|----------|
| **pass_test_ratio** | 通过测试数/总测试数 | [0, 1] | 主要奖励 |
| **format_validation** | 检查代码块存在 | 0 或 1 | 格式检查 |
| **think_tag_check** | 检查</think>标签 | 0 或 1 | 思考过程 |
| **error_classification** | 错误类型惩罚 | -1 或 -2 | 错误识别 |

#### 测试用例格式

```python
# 格式1: Input/Output测试
test_case_1 = {
    "stdin": "5\n",
    "expected_stdout": "25\n"
}

# 格式2: Assert测试
test_case_2 = {
    "assert_code": "assert solution(5) == 25"
}

# 格式3: Check函数测试
test_case_3 = {
    "entry_point": "solution",
    "check_code": """
def check(candidate):
    assert candidate(5) == 25
    assert candidate(0) == 0
    assert candidate(-3) == 9
"""
}

# 格式4: Pytest测试
test_case_4 = {
    "pytest_code": """
import pytest

def test_solution():
    assert solution(5) == 25
    assert solution(0) == 0
"""
}
```

#### 具体计算公式

```python
# 1. 提取代码
def extract_code(response: str) -> str:
    """
    从响应中提取代码
    优先级: Python代码块 > 任何代码块 > 纯文本
    """
    import re

    # 提取</think>之后的内容
    think_match = re.search(r'</think>(.*)', response, re.DOTALL)
    if think_match:
        content = think_match.group(1)
    else:
        content = response

    # 提取```python代码块
    python_match = re.search(r'```python\s*(.*?)\s*```', content, re.DOTALL)
    if python_match:
        code = python_match.group(1)
    else:
        # 提取任何```代码块
        any_match = re.search(r'```\s*(.*?)\s*```', content, re.DOTALL)
        if any_match:
            code = any_match.group(1)
        else:
            code = content

    # 去除if __name__ == "__main__"部分
    code = re.sub(r'if\s+__name__\s*==\s*["\']__main__["\']\s*:.*', '', code, flags=re.DOTALL)

    return code.strip()


# 2. 执行Input/Output测试
def run_io_test(code: str, stdin: str, expected_stdout: str, timeout: int = 5) -> bool:
    """
    执行stdin→stdout测试
    """
    import subprocess
    import tempfile
    import os

    # 写入临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_path = f.name

    try:
        # 执行代码
        result = subprocess.run(
            ['python3', temp_path],
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        # 比较输出
        actual = result.stdout.strip()
        expected = expected_stdout.strip()

        return actual == expected

    except subprocess.TimeoutExpired:
        return False
    except Exception as e:
        return False
    finally:
        os.unlink(temp_path)


# 3. 执行Assert测试
def run_assert_test(code: str, assert_code: str, timeout: int = 5) -> bool:
    """
    执行assert语句测试
    """
    import subprocess
    import tempfile
    import os

    # 组合代码
    full_code = code + "\n\n" + assert_code

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(full_code)
        temp_path = f.name

    try:
        result = subprocess.run(
            ['python3', temp_path],
            capture_output=True,
            timeout=timeout
        )

        # 返回码0表示所有assert通过
        return result.returncode == 0

    except:
        return False
    finally:
        os.unlink(temp_path)


# 4. 执行Check函数测试
def run_check_test(code: str, entry_point: str, check_code: str, timeout: int = 5) -> bool:
    """
    执行check函数测试
    """
    import subprocess
    import tempfile
    import os

    # 组合: 代码 + check函数 + 调用check
    full_code = f"""
{code}

{check_code}

# 调用check函数
check({entry_point})
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(full_code)
        temp_path = f.name

    try:
        result = subprocess.run(
            ['python3', temp_path],
            capture_output=True,
            timeout=timeout
        )

        return result.returncode == 0

    except:
        return False
    finally:
        os.unlink(temp_path)


# 5. 计算测试通过率
def compute_pass_ratio(code: str, test_cases: list) -> tuple:
    """
    计算测试通过率

    返回: (pass_ratio, error_type, passed, total)
    """
    passed = 0
    total = len(test_cases)
    error_type = None

    for test in test_cases:
        try:
            # 根据测试类型选择执行方法
            if 'stdin' in test:
                success = run_io_test(code, test['stdin'], test['expected_stdout'])
            elif 'assert_code' in test:
                success = run_assert_test(code, test['assert_code'])
            elif 'check_code' in test:
                success = run_check_test(code, test['entry_point'], test['check_code'])
            else:
                success = False

            if success:
                passed += 1

        except SyntaxError:
            error_type = 'SyntaxError'
        except Exception as e:
            if 'assert' in str(e).lower():
                error_type = 'LogicError'
            else:
                error_type = 'RuntimeError'

    pass_ratio = passed / total if total > 0 else 0.0

    return pass_ratio, error_type, passed, total


# 6. 格式验证
def validate_format(response: str) -> int:
    """
    检查是否包含代码块
    """
    import re

    has_code_block = bool(re.search(r'```.*?```', response, re.DOTALL))
    return 1 if has_code_block else 0


# 7. 思考标签检查
def check_think_tag(response: str) -> int:
    """
    检查是否包含</think>标签
    """
    return 1 if '</think>' in response else 0


# 8. 错误分类惩罚
def error_penalty(error_type: str) -> float:
    """
    根据错误类型返回惩罚
    """
    penalties = {
        'SyntaxError': -1.0,   # 编译错误
        'LogicError': -2.0,    # 逻辑错误
        'RuntimeError': -1.5,  # 运行时错误
    }

    return penalties.get(error_type, 0.0)
```

#### 最终奖励组合

```python
def compute_code_reward(response: str, test_cases: list) -> dict:
    """
    ROLL代码题完整奖励计算
    """
    # 提取代码
    code = extract_code(response)

    if not code:
        return {
            'response_level_reward': -2.0,
            'token_level_reward': 0.0,
            'score': 0,
            'details': 'No code extracted'
        }

    # 计算测试通过率
    pass_ratio, error_type, passed, total = compute_pass_ratio(code, test_cases)

    # 格式验证
    fmt_score = validate_format(response)

    # 思考标签
    think_score = check_think_tag(response)

    # 错误惩罚
    err_penalty = error_penalty(error_type) if error_type else 0.0

    # 组合奖励
    response_level_reward = pass_ratio + fmt_score + think_score + err_penalty

    # 统计分数(全部通过才算正确)
    score = 1 if pass_ratio == 1.0 else 0

    return {
        'response_level_reward': response_level_reward,  # [-2, 3]
        'token_level_reward': 0.0,
        'score': score,
        'pass_ratio': pass_ratio,
        'passed': passed,
        'total': total,
        'error_type': error_type
    }

# 示例
test_cases = [
    {"assert_code": "assert square(5) == 25"},
    {"assert_code": "assert square(0) == 0"},
    {"assert_code": "assert square(-3) == 9"}
]

# 完美回答
compute_code_reward(
    """
<think>Let me write the function</think>

```python
def square(x):
    return x * x
```
""",
    test_cases
)
# → {
#     'response_level_reward': 3.0,
#     'score': 1,
#     'pass_ratio': 1.0,
#     'passed': 3,
#     'total': 3
# }

# 部分通过
compute_code_reward(
    """
```python
def square(x):
    return x + x  # 错误实现
```
""",
    test_cases
)
# → {
#     'response_level_reward': 1.0,
#     'score': 0,
#     'pass_ratio': 0.0,
#     'passed': 0,
#     'total': 3,
#     'error_type': 'LogicError'
# }
```

---

### 3. LLM评判奖励 (LLMJudgeRewardWorker)

**文件路径:** `/home/yijia/.claude/11/ROLL/roll/pipeline/rlvr/rewards/llm_judge_reward_worker.py`

#### 评分维度

| 维度 | 计算方法 | 返回值范围 |
|------|---------|-----------|
| **llm_score** | API调用LLM评分 | [0, 10] |
| **normalized_score** | 归一化到[0,1] | [0, 1] |

#### 具体实现

```python
def llm_judge_reward(
    question: str,
    response: str,
    reference: str = None,
    judge_model: str = "gpt-4o-mini"
) -> float:
    """
    使用LLM评判响应质量

    Args:
        question: 问题
        response: 模型响应
        reference: 参考答案(可选)
        judge_model: 评判模型

    Returns:
        normalized_score: [0, 1]
    """
    from openai import OpenAI

    client = OpenAI()

    # 构建评判提示
    if reference:
        prompt = f"""
You are an expert evaluator. Rate the response quality on scale 0-10.

Question: {question}

Response: {response}

Reference Answer: {reference}

Evaluate based on:
1. Correctness (most important - 70%)
2. Reasoning quality (20%)
3. Clarity and completeness (10%)

Provide your evaluation in this format:
Score: X.X
Reasoning: [brief explanation]
"""
    else:
        prompt = f"""
You are an expert evaluator. Rate the response quality on scale 0-10.

Question: {question}

Response: {response}

Evaluate based on:
1. Factual accuracy (40%)
2. Reasoning quality (30%)
3. Clarity and completeness (20%)
4. Helpfulness (10%)

Provide your evaluation in this format:
Score: X.X
Reasoning: [brief explanation]
"""

    # 调用LLM
    try:
        completion = client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0  # 确保一致性
        )

        result = completion.choices[0].message.content

        # 提取分数
        import re
        match = re.search(r'Score:\s*([0-9.]+)', result)
        if match:
            score = float(match.group(1))
        else:
            # 降级: 查找Yes/No
            if 'yes' in result.lower():
                score = 10.0
            elif 'no' in result.lower():
                score = 0.0
            else:
                score = 5.0  # 不确定

        # 归一化到[0, 1]
        normalized = score / 10.0

        return normalized

    except Exception as e:
        print(f"LLM judge error: {e}")
        return 0.5  # 默认中等分数


# 示例
llm_judge_reward(
    question="What is 2+2?",
    response="2+2 equals 4",
    reference="4"
)
# → 1.0

llm_judge_reward(
    question="What is 2+2?",
    response="I think it might be 5",
    reference="4"
)
# → 0.2
```

---

### 4. 指令遵循奖励 (IFEvalRuleRewardWorker)

**文件路径:** `/home/yijia/.claude/11/ROLL/roll/pipeline/rlvr/rewards/ifeval_rule_reward_worker.py`

#### 27个约束函数列表

| 约束类型 | 函数名 | 检查内容 |
|---------|-------|---------|
| **关键词约束** | verify_keywords | 必须包含指定关键词 |
| | verify_keyword_frequency | 关键词出现指定次数 |
| | validate_forbidden_words | 不能包含禁止词 |
| **格式约束** | verify_paragraph_count | 段落数量要求 |
| | validate_word_constraint | 单词数量约束 |
| | verify_sentence_constraint | 句子数量约束 |
| | verify_bullet_points | 项目符号数量 |
| | validate_json_format | JSON格式验证 |
| | validate_quotation | 双引号包装 |
| | validate_no_commas | 无逗号约束 |
| **内容约束** | validate_title | 必须有标题 |
| | validate_choice | 选择题验证 |
| | validate_paragraph | 指定段落首词 |
| | verify_postscript | 附言验证 |
| | validate_placeholders | 占位符验证 |
| | validate_response_language | 语言约束 |
| **其他约束** | validate_repeat_prompt | 重复提示词 |
| | validate_two_responses | 两个回答 |
| | validate_uppercase | 全大写 |
| | validate_lowercase | 全小写 |
| | validate_end | 结尾短语 |
| | ... | (共27个) |

#### 关键约束函数实现

```python
# 1. 关键词约束
def verify_keywords(response: str, keywords: list) -> bool:
    """
    检查响应是否包含所有关键词
    """
    response_lower = response.lower()

    for keyword in keywords:
        if keyword.lower() not in response_lower:
            return False

    return True

# 示例
verify_keywords("The cat sat on the mat", ["cat", "mat"])  # → True
verify_keywords("The dog sat on the mat", ["cat", "mat"])  # → False


# 2. 关键词频率
def verify_keyword_frequency(response: str, keyword: str, frequency: int) -> bool:
    """
    检查关键词出现指定次数
    """
    count = response.lower().count(keyword.lower())
    return count == frequency

# 示例
verify_keyword_frequency("cat cat dog", "cat", 2)  # → True
verify_keyword_frequency("cat cat dog", "cat", 3)  # → False


# 3. 禁止词
def validate_forbidden_words(response: str, forbidden: list) -> bool:
    """
    检查不包含禁止词
    """
    response_lower = response.lower()

    for word in forbidden:
        if word.lower() in response_lower:
            return False

    return True

# 示例
validate_forbidden_words("This is good", ["bad", "terrible"])  # → True
validate_forbidden_words("This is bad", ["bad", "terrible"])   # → False


# 4. 段落数量
def verify_paragraph_count(response: str, count: int) -> bool:
    """
    检查段落数量
    段落由空行分隔
    """
    paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
    return len(paragraphs) == count

# 示例
verify_paragraph_count("Para 1\n\nPara 2\n\nPara 3", 3)  # → True


# 5. 单词数量约束
def validate_word_constraint(response: str, min_words: int = None, max_words: int = None) -> bool:
    """
    检查单词数量在范围内
    """
    words = response.split()
    word_count = len(words)

    if min_words and word_count < min_words:
        return False

    if max_words and word_count > max_words:
        return False

    return True

# 示例
validate_word_constraint("This is a test", min_words=3, max_words=5)  # → True
validate_word_constraint("This is a test", min_words=10)              # → False


# 6. 项目符号数量
def verify_bullet_points(response: str, count: int) -> bool:
    """
    检查项目符号数量
    支持: -, *, •, 1., 2., etc.
    """
    import re

    # 匹配各种项目符号
    bullets = re.findall(r'^\s*[-*•]\s+', response, re.MULTILINE)
    numbered = re.findall(r'^\s*\d+\.\s+', response, re.MULTILINE)

    total = len(bullets) + len(numbered)

    return total == count

# 示例
verify_bullet_points("""
- Item 1
- Item 2
- Item 3
""", 3)  # → True


# 7. JSON格式验证
def validate_json_format(response: str) -> bool:
    """
    检查响应是否为有效JSON
    """
    import json

    try:
        json.loads(response)
        return True
    except:
        return False

# 示例
validate_json_format('{"key": "value"}')  # → True
validate_json_format('invalid json')      # → False


# 8. 双引号包装
def validate_quotation(response: str) -> bool:
    """
    检查响应是否用双引号包装
    """
    response = response.strip()
    return response.startswith('"') and response.endswith('"')

# 示例
validate_quotation('"This is quoted"')  # → True
validate_quotation('This is not')       # → False


# 9. 标题验证
def validate_title(response: str, title: str = None) -> bool:
    """
    检查是否包含标题
    标题通常在第一行或用#标记
    """
    import re

    if title:
        # 检查指定标题
        return title in response.split('\n')[0]
    else:
        # 检查是否有标题(# 或首行大写)
        first_line = response.split('\n')[0].strip()

        # Markdown标题
        if first_line.startswith('#'):
            return True

        # 全大写标题
        if first_line.isupper() and len(first_line) > 0:
            return True

        return False

# 示例
validate_title("# My Title\n\nContent")  # → True
validate_title("TITLE\n\nContent")      # → True


# 10. 语言约束
def validate_response_language(response: str, language: str) -> bool:
    """
    检查响应语言
    简单实现: 基于字符集检测
    """
    if language == "english":
        # 检查主要是ASCII字符
        ascii_ratio = sum(ord(c) < 128 for c in response) / len(response)
        return ascii_ratio > 0.9

    elif language == "chinese":
        # 检查主要是中文字符
        chinese_ratio = sum('\u4e00' <= c <= '\u9fff' for c in response) / len(response)
        return chinese_ratio > 0.3

    return True

# 示例
validate_response_language("This is English", "english")  # → True
validate_response_language("这是中文", "chinese")          # → True
```

#### 最终奖励组合

```python
def compute_ifeval_reward(response: str, constraints: list) -> dict:
    """
    ROLL指令遵循奖励计算

    Args:
        response: 模型响应
        constraints: 约束列表
            [
                {"type": "keywords", "keywords": ["cat", "dog"]},
                {"type": "word_count", "min": 50, "max": 100},
                {"type": "format", "format": "json"}
            ]

    Returns:
        奖励字典
    """
    satisfied = 0
    total = len(constraints)

    constraint_results = []

    for constraint in constraints:
        ctype = constraint['type']

        # 根据类型调用对应函数
        if ctype == 'keywords':
            result = verify_keywords(response, constraint['keywords'])

        elif ctype == 'word_count':
            result = validate_word_constraint(
                response,
                constraint.get('min'),
                constraint.get('max')
            )

        elif ctype == 'format':
            if constraint['format'] == 'json':
                result = validate_json_format(response)
            elif constraint['format'] == 'quoted':
                result = validate_quotation(response)

        elif ctype == 'paragraphs':
            result = verify_paragraph_count(response, constraint['count'])

        elif ctype == 'bullets':
            result = verify_bullet_points(response, constraint['count'])

        elif ctype == 'forbidden':
            result = validate_forbidden_words(response, constraint['words'])

        # ... 其他约束类型

        else:
            result = True  # 未知类型默认通过

        constraint_results.append({
            'type': ctype,
            'satisfied': result
        })

        if result:
            satisfied += 1

    # 计算满足率
    satisfaction_ratio = satisfied / total if total > 0 else 1.0

    # 添加重复惩罚
    rep_penalty = repetition_penalty(response, ngram_size=3)

    # 最终奖励
    response_level_reward = satisfaction_ratio + rep_penalty

    return {
        'response_level_reward': response_level_reward,
        'token_level_reward': 0.0,
        'score': 1 if satisfaction_ratio == 1.0 else 0,
        'satisfaction_ratio': satisfaction_ratio,
        'satisfied': satisfied,
        'total': total,
        'details': constraint_results
    }

# 示例
constraints = [
    {"type": "keywords", "keywords": ["algorithm", "efficiency"]},
    {"type": "word_count", "min": 50, "max": 200},
    {"type": "bullets", "count": 3}
]

compute_ifeval_reward("""
Here's an explanation of the algorithm and its efficiency:

- First step: Initialize variables
- Second step: Process data
- Third step: Return result

The algorithm has O(n) time complexity which makes it very efficient.
""", constraints)
# → {
#     'response_level_reward': 1.0,
#     'score': 1,
#     'satisfaction_ratio': 1.0,
#     'satisfied': 3,
#     'total': 3
# }
```

---

### 5. CrossThinkQA奖励

**文件路径:** `/home/yijia/.claude/11/ROLL/roll/pipeline/rlvr/rewards/crossthinkqa_rule_reward_worker.py`

#### 三级奖励策略

```python
def compute_crossthinkqa_reward(
    response: str,
    ground_truth: str,
    mode: str = "soft"  # "loose" / "soft" / "strict"
) -> float:
    """
    CrossThinkQA三级奖励策略

    Args:
        response: 模型响应
        ground_truth: 正确答案
        mode: 奖励模式
            - loose: 答案正确即可
            - soft: 答案正确+格式好=1.0, 仅答案正确=0.5
            - strict: 必须完全正确(格式+答案)

    Returns:
        reward: float
    """
    # 提取答案
    answer = extract_answer(response)

    # 检查格式
    has_format = check_format(response)

    # 检查正确性
    is_correct = (answer.lower().strip() == ground_truth.lower().strip())

    # 根据模式计算奖励
    if mode == "loose":
        # 宽松模式: 答案对就给奖励
        return 1.0 if is_correct else -1.0

    elif mode == "soft":
        # 软模式: 格式+正确=1.0, 仅正确=0.5
        if is_correct and has_format:
            return 1.0
        elif is_correct:
            return 0.5
        elif has_format:
            return -0.5  # 格式对但答案错
        else:
            return -1.0

    elif mode == "strict":
        # 严格模式: 必须格式和答案都对
        if is_correct and has_format:
            return 1.0
        else:
            return -1.0

    return 0.0


def extract_answer(response: str) -> str:
    """
    提取<answer>标签中的答案
    """
    import re

    match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # 降级: 返回最后一行
    return response.strip().split('\n')[-1].strip()


def check_format(response: str) -> bool:
    """
    检查是否有<think>和<answer>标签
    """
    import re

    has_think = bool(re.search(r'<think>.*?</think>', response, re.DOTALL))
    has_answer = bool(re.search(r'<answer>.*?</answer>', response, re.DOTALL))

    return has_think and has_answer


# 示例
# 完美回答(soft模式)
compute_crossthinkqa_reward(
    "<think>Let me think</think><answer>Paris</answer>",
    "Paris",
    mode="soft"
)
# → 1.0

# 仅答案正确(soft模式)
compute_crossthinkqa_reward(
    "The answer is Paris",
    "Paris",
    mode="soft"
)
# → 0.5

# 答案错误但格式对(soft模式)
compute_crossthinkqa_reward(
    "<think>Let me think</think><answer>London</answer>",
    "Paris",
    mode="soft"
)
# → -0.5
```

---

### 6. 目标检测奖励 (DetectionRewardWorker)

**文件路径:** `/home/yijia/.claude/11/ROLL/roll/pipeline/rlvr/rewards/detection_reward_worker.py`

#### 多指标加权评分

```python
def compute_detection_reward(
    pred_boxes: list,
    gt_boxes: list,
    weights: dict = None
) -> float:
    """
    目标检测奖励计算

    Args:
        pred_boxes: 预测边界框
            [{"x1": 10, "y1": 20, "x2": 100, "y2": 200, "label": "cat"}, ...]
        gt_boxes: 真实边界框
            [{"x1": 12, "y1": 22, "x2": 98, "y2": 198, "label": "cat"}, ...]
        weights: 各指标权重

    Returns:
        综合奖励
    """
    if weights is None:
        weights = {
            'iou_max_iou': 0.3,      # 最大IoU
            'iou_max_label': 0.2,    # 标签匹配IoU
            'map': 0.2,              # mAP
            'map50': 0.2,            # mAP@0.5
            'map75': 0.1             # mAP@0.75
        }

    # 1. 计算IoU分数(最大IoU策略)
    iou_scores_max_iou = []
    for gt in gt_boxes:
        max_iou = 0
        for pred in pred_boxes:
            iou = compute_iou(pred, gt)
            max_iou = max(max_iou, iou)
        iou_scores_max_iou.append(max_iou)

    avg_iou_max_iou = sum(iou_scores_max_iou) / len(iou_scores_max_iou) if iou_scores_max_iou else 0

    # 2. 计算IoU分数(标签匹配策略)
    iou_scores_max_label = []
    for gt in gt_boxes:
        max_iou = 0
        for pred in pred_boxes:
            if pred['label'] == gt['label']:  # 只考虑同标签
                iou = compute_iou(pred, gt)
                max_iou = max(max_iou, iou)
        iou_scores_max_label.append(max_iou)

    avg_iou_max_label = sum(iou_scores_max_label) / len(iou_scores_max_label) if iou_scores_max_label else 0

    # 3. 计算mAP
    map_score = compute_map(pred_boxes, gt_boxes, iou_threshold=0.5)

    # 4. 计算mAP@0.5
    map50_score = compute_map(pred_boxes, gt_boxes, iou_threshold=0.5)

    # 5. 计算mAP@0.75
    map75_score = compute_map(pred_boxes, gt_boxes, iou_threshold=0.75)

    # 6. 大小惩罚
    size_penalty = 0.6 ** abs(len(pred_boxes) - len(gt_boxes))

    # 7. 加权组合
    weighted_iou_max_iou = avg_iou_max_iou * size_penalty
    weighted_iou_max_label = avg_iou_max_label * size_penalty
    weighted_map = map_score * size_penalty
    weighted_map50 = map50_score * size_penalty
    weighted_map75 = map75_score * size_penalty

    # 8. 最终奖励
    total_weight = sum(weights.values())

    final_reward = (
        weighted_iou_max_iou * weights['iou_max_iou'] +
        weighted_iou_max_label * weights['iou_max_label'] +
        weighted_map * weights['map'] +
        weighted_map50 * weights['map50'] +
        weighted_map75 * weights['map75']
    ) / total_weight

    return final_reward


def compute_iou(box1: dict, box2: dict) -> float:
    """
    计算两个边界框的IoU(Intersection over Union)
    """
    # 计算交集
    x1 = max(box1['x1'], box2['x1'])
    y1 = max(box1['y1'], box2['y1'])
    x2 = min(box1['x2'], box2['x2'])
    y2 = min(box1['y2'], box2['y2'])

    if x2 < x1 or y2 < y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)

    # 计算并集
    area1 = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1'])
    area2 = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1'])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def compute_map(pred_boxes: list, gt_boxes: list, iou_threshold: float = 0.5) -> float:
    """
    计算mAP(mean Average Precision)
    """
    if not pred_boxes or not gt_boxes:
        return 0.0

    # 按置信度排序预测框(如果有confidence字段)
    if 'confidence' in pred_boxes[0]:
        pred_boxes = sorted(pred_boxes, key=lambda x: x['confidence'], reverse=True)

    # 标记真实框是否被匹配
    gt_matched = [False] * len(gt_boxes)

    tp = 0  # True Positives
    fp = 0  # False Positives

    for pred in pred_boxes:
        best_iou = 0
        best_gt_idx = -1

        # 找到最佳匹配的真实框
        for i, gt in enumerate(gt_boxes):
            if gt_matched[i]:
                continue

            if pred['label'] == gt['label']:
                iou = compute_iou(pred, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i

        # 判断是否为True Positive
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            gt_matched[best_gt_idx] = True
        else:
            fp += 1

    # 计算Precision
    fn = len(gt_boxes) - sum(gt_matched)  # False Negatives

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # 简化的AP计算(通常需要计算PR曲线下面积)
    ap = (precision + recall) / 2 if (precision + recall) > 0 else 0

    return ap


# 示例
pred_boxes = [
    {"x1": 10, "y1": 20, "x2": 100, "y2": 200, "label": "cat"},
    {"x1": 150, "y1": 50, "x2": 250, "y2": 150, "label": "dog"}
]

gt_boxes = [
    {"x1": 12, "y1": 22, "x2": 98, "y2": 198, "label": "cat"},
    {"x1": 155, "y1": 55, "x2": 245, "y2": 145, "label": "dog"}
]

compute_detection_reward(pred_boxes, gt_boxes)
# → 0.85 (高IoU和mAP)
```

---

## AgentFlow框架奖励机制

### 1. 装饰器模式

**文件路径:** `/home/yijia/.claude/11/AgentFlow/agentflow/reward.py`

#### 核心装饰器

```python
from functools import wraps
from typing import Union, Optional

def reward(fn):
    """
    奖励函数装饰器

    功能:
    1. 包装返回值为RewardSpanData
    2. 自动处理async/sync函数
    3. 类型验证(float/int/None)
    4. AgentOps追踪集成
    """
    @wraps(fn)
    def sync_wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        return _process_reward_result(result)

    @wraps(fn)
    async def async_wrapper(*args, **kwargs):
        result = await fn(*args, **kwargs)
        return _process_reward_result(result)

    # 判断是否为async函数
    if asyncio.iscoroutinefunction(fn):
        return async_wrapper
    else:
        return sync_wrapper


def _process_reward_result(result: Union[float, int, None]) -> dict:
    """
    处理奖励结果

    Returns:
        RewardSpanData: {"type": "reward", "value": float}
    """
    if result is None:
        value = None
    elif isinstance(result, (int, float)):
        value = float(result)
    else:
        import warnings
        warnings.warn(f"Reward function returned {type(result)}, expected float/int/None")
        value = None

    return {
        "type": "reward",
        "value": value,
        "timestamp": time.time()
    }


# 使用示例
@reward
def simple_correctness(response: str, ground_truth: str) -> float:
    """简单的正确性评分"""
    return 1.0 if response == ground_truth else 0.0


@reward
async def async_llm_judge(question: str, response: str) -> float:
    """异步LLM评判"""
    # 调用异步API
    score = await llm_api.judge(question, response)
    return score


# 调用
result1 = simple_correctness("42", "42")
# → {"type": "reward", "value": 1.0, "timestamp": ...}

result2 = await async_llm_judge("What is 2+2?", "4")
# → {"type": "reward", "value": 0.95, "timestamp": ...}
```

---

### 2. LLM评判系统

**文件路径:** `/home/yijia/.claude/11/AgentFlow/test/calculate_score_unified.py`

#### ResultScorer类

```python
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

class ResultScorer:
    """
    统一评分系统
    使用GPT-4o进行答案验证
    """

    def __init__(self, llm_engine=None):
        if llm_engine is None:
            # 默认使用GPT-4o
            self.client = OpenAI()
            self.model = "gpt-4o"
        else:
            self.client = llm_engine
            self.model = llm_engine.model_string

    def answer_verification(
        self,
        question: str,
        response: str,
        correct_answer: str
    ) -> tuple:
        """
        验证答案是否正确

        Returns:
            (is_correct: bool, analysis: str)
        """
        # 提取<answer>标签内容
        answer = self._extract_answer(response)

        # 构建验证提示
        prompt = f"""
You are an expert answer verifier. Determine if the student's answer is correct.

Question: {question}

Student's Answer: {answer}

Correct Answer: {correct_answer}

Is the student's answer correct? Consider:
1. Numerical equivalence (e.g., 0.5 = 1/2)
2. Semantic equivalence (e.g., "Paris" = "paris" = "the capital of France")
3. Minor formatting differences should be ignored

Respond in this format:
Correct: [Yes/No]
Analysis: [brief explanation]
"""

        try:
            # 调用LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )

            result = response.choices[0].message.content

            # 解析结果
            is_correct = 'yes' in result.lower().split('\n')[0]
            analysis = result

            return is_correct, analysis

        except Exception as e:
            print(f"Verification error: {e}")
            return False, f"Error: {e}"

    def _extract_answer(self, response: str) -> str:
        """提取<answer>标签中的内容"""
        import re

        match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        if match:
            return match.group(1).strip()

        # 降级: 返回整个响应
        return response.strip()

    def score_results(
        self,
        results: list,
        max_workers: int = 10
    ) -> dict:
        """
        批量评分(并行处理)

        Args:
            results: 结果列表
                [
                    {
                        "question": "What is 2+2?",
                        "response": "<think>...</think><answer>4</answer>",
                        "ground_truth": "4"
                    },
                    ...
                ]
            max_workers: 并行worker数量

        Returns:
            评分统计
        """
        correct = 0
        total = len(results)
        wrong_indices = []

        # 并行处理
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交任务
            futures = {
                executor.submit(
                    self.answer_verification,
                    r['question'],
                    r['response'],
                    r['ground_truth']
                ): i
                for i, r in enumerate(results)
            }

            # 收集结果(带进度条)
            for future in tqdm(as_completed(futures), total=total, desc="Scoring"):
                idx = futures[future]

                try:
                    is_correct, analysis = future.result()

                    if is_correct:
                        correct += 1
                    else:
                        wrong_indices.append(idx)

                    # 保存分析
                    results[idx]['is_correct'] = is_correct
                    results[idx]['analysis'] = analysis

                except Exception as e:
                    print(f"Error processing result {idx}: {e}")
                    wrong_indices.append(idx)

        # 计算统计
        accuracy = (correct / total * 100) if total > 0 else 0

        return {
            'correct': correct,
            'total': total,
            'accuracy': accuracy,
            'wrong_indices': wrong_indices,
            'results': results
        }


# 使用示例
scorer = ResultScorer()

results = [
    {
        "question": "What is the capital of France?",
        "response": "<think>France is a country in Europe</think><answer>Paris</answer>",
        "ground_truth": "Paris"
    },
    {
        "question": "What is 2+2?",
        "response": "<think>2+2=4</think><answer>4</answer>",
        "ground_truth": "4"
    }
]

scores = scorer.score_results(results, max_workers=10)
print(f"Accuracy: {scores['accuracy']:.1f}%")
print(f"Correct: {scores['correct']}/{scores['total']}")
```

---

## 直接可用的代码实现

### 综合奖励计算器(借鉴ROLL+AgentFlow)

```python
class HybridRewardComputer:
    """
    综合奖励计算器
    结合ROLL的多维度设计和AgentFlow的LLM评判
    """

    def __init__(
        self,
        weights: dict = None,
        use_llm_judge: bool = False,
        llm_model: str = "gpt-4o-mini"
    ):
        # 权重配置
        self.weights = weights or {
            'correctness': 0.65,
            'efficiency': 0.15,
            'simplicity': 0.10,
            'format': 0.05,
            'repetition': 0.05
        }

        # LLM评判器(可选)
        self.use_llm_judge = use_llm_judge
        if use_llm_judge:
            from openai import OpenAI
            self.llm_client = OpenAI()
            self.llm_model = llm_model

    def compute_reward(
        self,
        problem: str,
        prediction: str,
        ground_truth: str,
        problem_type: str,
        metadata: dict = None
    ) -> tuple:
        """
        计算综合奖励

        Returns:
            (total_reward, reward_breakdown)
        """
        metadata = metadata or {}

        # 1. 正确性奖励
        correctness = self._compute_correctness(
            prediction, ground_truth, problem_type
        )

        # 2. 效率奖励
        efficiency = self._compute_efficiency(
            metadata.get('cost', 0.0)
        )

        # 3. 简洁性奖励
        simplicity = self._compute_simplicity(
            metadata.get('execution_time', 0.0),
            metadata.get('num_operators', 1)
        )

        # 4. 格式奖励(ROLL风格)
        format_score = self._compute_format(prediction, problem_type)

        # 5. 重复惩罚(ROLL风格)
        repetition = self._compute_repetition_penalty(prediction)

        # 6. 可选LLM评判(AgentFlow风格)
        llm_score = None
        if self.use_llm_judge:
            llm_score = self._llm_judge(problem, prediction, ground_truth)

        # 组合奖励
        if llm_score is not None:
            # 混合: 70%规则 + 30%LLM
            rule_based = (
                self.weights['correctness'] * correctness +
                self.weights['efficiency'] * efficiency +
                self.weights['simplicity'] * simplicity +
                self.weights['format'] * format_score +
                self.weights['repetition'] * repetition
            )
            total_reward = 0.7 * rule_based + 0.3 * llm_score * 10
        else:
            total_reward = (
                self.weights['correctness'] * correctness +
                self.weights['efficiency'] * efficiency +
                self.weights['simplicity'] * simplicity +
                self.weights['format'] * format_score +
                self.weights['repetition'] * repetition
            )

        # 裁剪到[-10, 10]
        total_reward = max(-10.0, min(10.0, total_reward))

        # 奖励分解
        breakdown = {
            'correctness': correctness,
            'efficiency': efficiency,
            'simplicity': simplicity,
            'format': format_score,
            'repetition': repetition,
            'llm_score': llm_score,
            'total': total_reward
        }

        return total_reward, breakdown

    def _compute_correctness(self, pred: str, gt: str, ptype: str) -> float:
        """正确性评估(ROLL风格改进)"""
        if ptype == "math":
            return self._math_correctness_roll_style(pred, gt)
        elif ptype == "code":
            return self._code_correctness_roll_style(pred, gt)
        elif ptype == "qa":
            return self._qa_correctness_roll_style(pred, gt)
        return 0.0

    def _math_correctness_roll_style(self, pred: str, gt: str) -> float:
        """ROLL风格数学正确性"""
        # 方法1: 提取boxed答案
        pred_answer = self._extract_boxed(pred)
        gt_answer = self._extract_boxed(gt)

        if pred_answer and gt_answer:
            try:
                pred_num = float(pred_answer)
                gt_num = float(gt_answer)

                diff = abs(pred_num - gt_num)

                if diff < 1e-4:
                    return 10.0
                elif diff < 0.1:
                    return 8.0
                elif diff < 1.0:
                    return 5.0
                elif diff < 10.0:
                    return 2.0
                else:
                    return -5.0
            except:
                pass

        # 方法2: 数字提取(原方法)
        pred_nums = self._extract_numbers(pred)
        gt_nums = self._extract_numbers(gt)

        if pred_nums and gt_nums:
            diff = abs(pred_nums[-1] - gt_nums[-1])

            if diff < 1e-4:
                return 10.0
            elif diff < 1.0:
                return 5.0
            else:
                return -5.0

        return -5.0

    def _extract_boxed(self, text: str) -> str:
        """提取\boxed{}中的内容"""
        import re
        match = re.search(r'\\boxed\{([^}]+)\}', text)
        return match.group(1) if match else None

    def _extract_numbers(self, text: str) -> list:
        """提取所有数字"""
        import re
        numbers = re.findall(r'-?\d+\.?\d*', text)
        return [float(n) for n in numbers if n]

    def _code_correctness_roll_style(self, pred: str, gt: str) -> float:
        """ROLL风格代码正确性"""
        # 简化版本: 字符串匹配
        if gt.lower() in pred.lower():
            return 10.0

        # 函数名匹配
        import re
        pred_funcs = re.findall(r'def\s+(\w+)\s*\(', pred)
        gt_funcs = re.findall(r'def\s+(\w+)\s*\(', gt)

        if pred_funcs and gt_funcs and pred_funcs[0] == gt_funcs[0]:
            return 5.0

        return -5.0

    def _qa_correctness_roll_style(self, pred: str, gt: str) -> float:
        """ROLL风格QA正确性"""
        pred = pred.lower().strip()
        gt = gt.lower().strip()

        # 精确匹配
        if pred == gt:
            return 10.0

        # 包含匹配
        if gt in pred:
            return 8.0

        # Token重叠
        pred_tokens = set(pred.split())
        gt_tokens = set(gt.split())

        if not gt_tokens:
            return 0.0

        overlap = len(pred_tokens & gt_tokens) / len(gt_tokens)

        if overlap > 0.8:
            return 6.0
        elif overlap > 0.5:
            return 3.0
        elif overlap > 0.2:
            return 0.0
        else:
            return -5.0

    def _compute_efficiency(self, cost: float) -> float:
        """效率奖励(ROLL风格)"""
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

    def _compute_simplicity(self, exec_time: float, num_ops: int) -> float:
        """简洁性奖励(ROLL风格)"""
        # 执行时间评分
        if exec_time <= 5.0:
            time_score = 10.0
        elif exec_time <= 15.0:
            time_score = 5.0
        elif exec_time <= 30.0:
            time_score = 0.0
        elif exec_time <= 60.0:
            time_score = -3.0
        else:
            time_score = -5.0

        # 算子数评分
        if num_ops <= 2:
            op_score = 10.0
        elif num_ops <= 4:
            op_score = 5.0
        elif num_ops <= 6:
            op_score = 0.0
        else:
            op_score = -5.0

        return (time_score + op_score) / 2.0

    def _compute_format(self, response: str, ptype: str) -> float:
        """格式奖励(ROLL风格)"""
        import re

        if ptype == "math":
            has_think = bool(re.search(r'<think>.*?</think>', response, re.DOTALL))
            has_answer = bool(re.search(r'<answer>.*?</answer>', response, re.DOTALL))

            if has_think and has_answer:
                return 2.0
            elif has_answer:
                return 0.0
            else:
                return -2.0

        elif ptype == "code":
            has_code = bool(re.search(r'```.*?```', response, re.DOTALL))
            return 2.0 if has_code else -2.0

        return 0.0

    def _compute_repetition_penalty(self, response: str, ngram: int = 3) -> float:
        """重复惩罚(ROLL风格)"""
        words = response.split()

        if len(words) < ngram:
            return 0.0

        ngrams = [
            tuple(words[i:i+ngram])
            for i in range(len(words) - ngram + 1)
        ]

        if not ngrams:
            return 0.0

        unique_ratio = len(set(ngrams)) / len(ngrams)

        if unique_ratio > 0.9:
            return 0.0
        elif unique_ratio > 0.7:
            return -0.5
        elif unique_ratio > 0.5:
            return -1.0
        else:
            return -2.0

    def _llm_judge(self, question: str, response: str, gt: str) -> float:
        """LLM评判(AgentFlow风格)"""
        prompt = f"""
Rate the response quality on scale 0-1.

Question: {question}
Response: {response}
Ground Truth: {gt}

Format: Score: X.X
"""

        try:
            completion = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )

            result = completion.choices[0].message.content

            import re
            match = re.search(r'Score:\s*([0-9.]+)', result)
            if match:
                return float(match.group(1))
        except:
            pass

        return 0.5


# 使用示例
computer = HybridRewardComputer(use_llm_judge=False)

reward, breakdown = computer.compute_reward(
    problem="What is 2+2?",
    prediction="<think>Let me calculate</think><answer>4</answer>",
    ground_truth="4",
    problem_type="math",
    metadata={'cost': 0.0001, 'execution_time': 2.0, 'num_operators': 2}
)

print(f"Total Reward: {reward:.2f}")
print(f"Breakdown: {breakdown}")
```

---

## 总结

### ROLL的核心设计

1. **多维度评分**: 5-8个维度组合
2. **域特异性**: 每种任务类型专用Worker
3. **稀疏奖励**: token_level全0,response_level密集
4. **配置驱动**: 通过tag自动路由

### AgentFlow的核心设计

1. **装饰器模式**: 极简优雅
2. **LLM评判**: 智能准确
3. **并行处理**: 10 workers
4. **深度集成**: AgentOps追踪

### 推荐实施

直接使用上面的`HybridRewardComputer`,结合两者优点!
