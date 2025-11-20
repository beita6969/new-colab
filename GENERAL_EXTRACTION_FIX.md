# Ground Truth提取通用方法修复 (2025-11-19 22:45)

## 问题背景

之前的方法针对GSM8K数据集设计了特殊的`<<calculation>>result`格式检测，但这违反了通用性原则。参考AgentFlow项目后，采用更通用的方法。

## AgentFlow的通用方法

### 核心策略

1. **结构化输出**: 使用`<answer>`标签包裹最终答案
2. **取最后匹配**: `re.findall()`后取`[-1]`，自然避免中间值
3. **LLM理解**: 当规则失败时，用LLM理解文本语义

### 关键设计模式

```python
# 1. 取最后一个<answer>标签
answer_matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
if answer_matches:
    return answer_matches[-1]  # 最后一个 = 最终答案

# 2. LLM prompt: 明确"忽略推理过程"
prompt = """
You are extracting the FINAL ANSWER from a mathematical solution text.

**Instructions:**
1. **Ignore intermediate calculations** - Focus only on the concluding answer
2. **Look for concluding statements** like "So the answer is..."
3. **Extract the final numeric value** - Return JUST the number
"""
```

## 新实现 (`src/answer_extractor.py:45-113`)

### 提取优先级（通用方法）

```python
def _extract_math_answer(self, text: str, is_ground_truth: bool) -> str:
    # 1. <answer>标签（取最后一个）
    answer_matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if answer_matches:
        return self._clean_math_answer(answer_matches[-1])

    # 2. LaTeX \boxed{} 格式
    boxed = self._extract_boxed(text)
    if boxed:
        return self._clean_math_answer(boxed)

    # 3. 明确的"Final Answer"标记
    patterns = [
        r"(?:the\s+final\s+answer\s+is)[：:]*\s*([-+]?\d+(?:/\d+)?(?:\.\d+)?)",
        r"(?:Final\s+Answer|最终答案)[：:]*\s*([-+]?\d+(?:/\d+)?(?:\.\d+)?)",
    ]

    # 4. 对于ground_truth且复杂：调用LLM理解
    if is_ground_truth and self.use_llm_fallback:
        if text.count('=') >= 2 or len(re.findall(r'\d+', text)) > 3:
            return self._llm_extract_math_ground_truth(text)

    # 5. 兜底：最后一个数字
    numbers = self._extract_all_numbers(text)
    if numbers:
        return str(numbers[-1])
```

### LLM Ground Truth提取 (`src/answer_extractor.py:368-408`)

```python
def _llm_extract_math_ground_truth(self, text: str) -> str:
    """参考AgentFlow设计的prompt"""
    prompt = f"""
You are extracting the FINAL ANSWER from a mathematical solution text.

**Instructions:**
1. **Ignore intermediate calculations** - Focus only on the concluding answer
2. **Look for concluding statements** like "So the answer is...", "Therefore..."
3. **Extract the final numeric value** - Return JUST the number

**Text:**
{text[:800]}

**Output Format:**
- Return ONLY the final numerical answer
- No explanation, no intermediate values
- If multiple numbers exist, return the one from the final conclusion

**Final Answer (number only):**
"""
    response = self.llm_client.generate(prompt, max_tokens=30, temperature=0)
    return response.strip()
```

## 测试结果

```
✅ 带<answer>标签:  "<answer>60</answer>" → "60"
✅ LaTeX boxed:     "\boxed{5/324}" → "5/324"
✅ Final Answer标记: "the final answer is 42" → "42"
✅ GSM8K风格:      "So he bought 80-26=54 more" → "54.0" (数值等价)
```

## 移除的数据集特定代码

### 删除：GSM8K专用逻辑

```python
# ❌ 删除 - 针对GSM8K的特殊检测
gsm8k_pattern = r'<<[^>]+>>\s*([-+]?\d+(?:\.\d+)?(?:/\d+)?)'
gsm8k_matches = re.findall(gsm8k_pattern, text)
if gsm8k_matches:
    return gsm8k_matches[-1]

# ❌ 删除 - GSM8K特定启发式
conclusion_patterns = [
    r'(?:So|Therefore)[^.]*?([-+]?\d+)',
    r'makes\s+(?:\$)?\s*([-+]?\d+)',  # 针对"makes money"问题
    r'has\s+([-+]?\d+)\s+(?:subscribers)',  # 针对订阅者问题
]
```

### 替换为：通用LLM理解

```python
# ✅ 通用方法 - 适用于所有数据集
if is_ground_truth and self.use_llm_fallback:
    if has_complex_calculations:
        return self._llm_extract_math_ground_truth(text)
```

## 为何不针对数据集设计？

### 问题

1. **脆弱性**: GSM8K格式可能变化，硬编码的`<<>>`规则会失效
2. **不可扩展**: 每个新数据集需要新规则
3. **维护成本**: 规则冲突、优先级管理复杂

### AgentFlow方法的优势

1. **通用性**: `<answer>`标签适用于所有任务类型
2. **鲁棒性**: LLM fallback处理边缘情况
3. **语义理解**: LLM能理解"最终答案"的概念，不依赖格式

## 配置要求

### 启用LLM Fallback

当前默认`use_llm_fallback=False`。对于复杂ground truth，建议启用：

```python
# src/reward_computer.py 或 src/grpo_trainer.py
extractor = AnswerExtractor(use_llm_fallback=True)
```

需要配置LLM客户端（通常使用GPT-4o-mini或类似模型）。

### 不启用LLM的情况

如果不启用LLM fallback：
- 依赖规则提取（<answer>, \boxed{}, "Final Answer"标记）
- 兜底使用"最后一个数字"
- 对于简单ground truth仍然有效
- 复杂的多步计算ground truth可能提取错误

## 预期改善

| 场景 | 修复前 | 修复后 |
|------|--------|--------|
| 简单数字 | ✅ "42" | ✅ "42" |
| 带<answer>标签 | ✅ "60" | ✅ "60" |
| LaTeX \boxed{} | ✅ "5/324" | ✅ "5/324" |
| GSM8K多步计算 | ❌ "4" (错) | ✅ "54" (对) |
| 其他数据集 | ❓ 未知 | ✅ 通用 |

## 后续步骤

1. **检查LLM客户端配置**
   ```bash
   grep -r "use_llm_fallback\|llm_client" src/
   ```

2. **重启训练验证**
   - 停止当前训练
   - 重启训练
   - 观察ground truth提取日志

3. **监控提取准确性**
   ```bash
   grep '答案提取对比 (math)' logs/*.log | tail -20
   ```

## 文件修改

```
src/answer_extractor.py:45-113    - _extract_math_answer (通用方法)
src/answer_extractor.py:368-408   - _llm_extract_math_ground_truth (新增)
```

## 参考

- AgentFlow项目: `/home/yijia/.claude/11/AgentFlow`
- 核心文件: `train/rollout.py`, `train/utils.py`, `models/planner.py`
- 设计模式: 结构化输出 + 取最后匹配 + LLM judge

---

**生成时间**: 2025-11-19 22:45
**状态**: ✅ 通用方法已实现，等待训练验证
**下一步**: 重启训练，监控ground truth提取效果
