# 训练问题诊断与修复方案

## 诊断时间
2025-11-20 13:50

## 运行状态
- 运行时长: 25分钟
- 状态: 已停止（用于诊断）
- 进度: Step 1, Batch 1 (4/24样本执行中)

## 发现的问题

### 问题1: Workflow变量作用域bug ❌ CRITICAL

**错误信息**:
```
UnboundLocalError: local variable 'revised_code' referenced before assignment
位置: aflow_executor.py:271 → 动态生成的workflow第33行
```

**根本原因**:
Qwen2.5-7B生成的workflow代码存在变量作用域问题：
```python
# 错误的代码模式
if "needs improvement" in feedback:
    revised_code = revised.get('solution', answer)  # 只在if内定义
    ...

return revised_code, cost  # ❌ 如果if不执行，变量未定义
```

**影响**:
- 导致2/24个workflow执行失败（8%失败率）
- 触发Fallback机制，但降低了训练效率

**修复方案**:
在prompt_optimizer或rl_workflow_generator中添加变量初始化检查，确保：
```python
revised_code = code  # ✅ 在函数开头初始化
if "needs improvement" in feedback:
    revised = await self.revise(...)
    revised_code = revised.get('solution', revised_code)

return revised_code, cost  # ✅ 总是有定义
```

### 问题2: LLM Judge结果被覆盖 ❌ HIGH PRIORITY

**症状**:
```
LLM Judge: True (正确)
最终评分: -5.0/10.0 (负分!)
```

**根本原因**:
reward_computer.py:278-343中，LLM Judge的结果没有立即返回，而是继续执行了答案提取比较逻辑。

```python
if self.use_llm_judge and problem_type != "code":
    is_correct = self._llm_judge_compare(...)  # 返回True
    correctness_score = 10.0 if is_correct else -5.0  # 计算为10.0
    # ❌ 缺少return，代码继续往下执行

# 继续执行答案提取逻辑（不应该！）
else:
    extracted_pred = self.extractor.extract_answer(...)  # "$1,825"
    extracted_gt = self.extractor.extract_answer(...)     # "1825"
    correctness_score = self._compute_correctness_reward(...)  # -5.0（覆盖了LLM Judge的10.0!）
```

**影响**:
- LLM Judge判定正确的样本被错误标记为错误
- 导致训练信号完全错误
- 预计影响50%+的样本（因为很多答案带单位/格式）

**证据**:
```
预测: $1,825 vs 真值: 1825  → 判决:True → 评分:-5.0 ❌
预测: \boxed{50 days} vs 真值: 50 → 判决:True → 评分:10.0 ✅（碰巧提取后匹配）
```

**修复方案**:
在reward_computer.py:293后添加归一化和return：
```python
if self.use_llm_judge and problem_type != "code":
    is_correct = self._llm_judge_compare(...)
    correctness_score = 10.0 if is_correct else -5.0

    if metadata is not None:
        metadata['correctness_score'] = correctness_score
        metadata['used_llm_judge'] = True

    # ✅ 立即归一化并返回，不要继续执行答案提取逻辑
    normalized_reward = 1.0 if is_correct else 0.0
    return normalized_reward

# else分支才执行答案提取
```

### 问题3: LLM Judge响应格式解析失败 ⚠️ MEDIUM

**症状**:
```
⚠️ 无法解析LLM Judge响应（尝试了5种格式）
完整响应: <analysis>The model states...
```

**原因**:
少数情况下（~10%），LLM Judge返回的响应被截断或格式异常。

**影响**:
解析失败时默认返回False（错误），造成误判。

**修复方案**:
已在reward_computer.py:197-236实施了5级容错解析，但仍有极少数失败。建议：
1. 增加max_tokens从200→300
2. 添加重试机制（失败时重试1次）

## 修复优先级

**P0 - 立即修复**（阻塞训练）:
1. ✅ 问题2: LLM Judge结果被覆盖（最严重，影响所有样本）
2. ✅ 问题1: Workflow变量作用域bug（影响8%样本）

**P1 - 下次训练前修复**:
3. 问题3: LLM Judge解析失败增强（影响~2%样本）

## 预期效果

修复后：
- Math准确率: 当前~0%（因为评分错误）→ 50-60%
- Workflow成功率: 92% → 100%
- 训练信号: 完全错误 → 正确

## 下一步

1. 实施P0修复
2. 验证修复
3. 重新启动训练
4. 监控前10步的准确率
