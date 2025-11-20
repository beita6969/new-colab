# 训练稳定性与准确率修复方案

## 执行摘要

本文档提供完整的修复方案，解决当前训练中的三大核心问题：
1. 训练不稳定（准确率波动±13%）
2. Math任务准确率未达预期（当前40%，目标75-80%）
3. Code/QA任务持续低准确率

**预期效果**：
- Math准确率：40% → 75-80%
- 训练波动：±13% → ±5%
- Code准确率：0% → 40-50%
- 整体准确率：25% → 60%+

---

## P0 - 立即修复（今天完成）

### ✅ 1. 增强LLM Judge响应解析鲁棒性

**问题**：GPT OSS 120B返回的判决格式不统一，导致18%误判率。

**修改文件**：`src/reward_computer.py:154-222`

**改动内容**：
```python
# 原代码：单一正则匹配
true_false_match = re.search(
    r'(?:<true_false>|<true_false>:|\*\*true_false\*\*:?)\s*(True|False)',
    result_text,
    re.IGNORECASE
)

# 新代码：5级容错匹配
# 尝试1: 标准XML标签
true_false_match = re.search(r'<true_false>\s*(True|False)\s*</true_false>', ...)
# 尝试2: 冒号分隔
if not true_false_match:
    true_false_match = re.search(r'<true_false>\s*:\s*(True|False)', ...)
# 尝试3-5: Markdown/key-value/末尾兜底
...
```

**状态**：✅ 已完成

**预期效果**：LLM Judge误判率从18% → <5%

---

### ✅ 2. 修复Answer Extractor分数提取bug

**问题**：`"5/324"` 被提取为计算后的小数，导致精度损失和比较错误。

**修改文件**：`src/answer_extractor.py:248-280`

**改动内容**：
```python
# 原代码：分数转浮点
if '/' in clean_m:
    parts = clean_m.split('/')
    value = float(parts[0]) / float(parts[1])  # 精度损失！
    numbers.append(value)

# 新代码：保持字符串格式
fraction_pattern = r'-?\d+/\d+'
fraction_matches = re.findall(fraction_pattern, text)
for frac in fraction_matches:
    numbers.append(frac)  # 保持 "5/324" 字符串格式
```

**状态**：✅ 已完成

**预期效果**：分数比较准确率100%，数学题准确率提升10-15%

---

### ⚠️ 3. 确保test_result正确传递到RewardComputer

**问题**：Code任务持续0%准确率，因为test_result未被RewardComputer使用。

**当前状态**：
- ✅ RewardComputer已支持：`src/reward_computer.py:246-259`
- ✅ Trainer已传递metadata：`src/grpo_trainer.py:349-365`
- ❌ **metadata['test_result']未被正确设置**

**需要的修改** - 两个选项：

#### 选项A：工作流返回test_result（推荐）

修改生成的Workflow模板，让Code任务返回test结果：

```python
# 当前Workflow返回
return code, self.llm.get_usage_summary().get("total_cost", 0.0)

# 修改为
test_result = await self.test(problem=problem, solution=code, entry_point=entry_point)
test_passed = test_result.get('result', False)
metadata = {
    'test_result': test_passed,
    'test_details': test_result
}
return code, self.llm.get_usage_summary().get("total_cost", 0.0), metadata
```

修改`src/aflow_executor.py:292-310`处理3元组返回：
```python
# 安全地解包结果（可能返回2个或3个值）
if isinstance(result, tuple):
    if len(result) >= 3:
        answer, cost, extra_metadata = result[0], result[1], result[2]
        # 合并metadata
        if isinstance(extra_metadata, dict):
            metadata.update(extra_metadata)
    elif len(result) >= 2:
        answer, cost = result[0], result[1]
```

#### 选项B：执行器注入test_result（过渡方案）

在`src/aflow_executor.py:220-274`的Code分支中包装Test operator调用：

```python
if problem_type == "code":
    # 执行workflow
    result = await asyncio.wait_for(
        workflow(problem, kwargs["entry_point"], kwargs["test"]),
        timeout=self.timeout
    )

    # 注入test_result到metadata
    # 尝试从workflow的test operator结果中提取
    if hasattr(workflow, 'test') and hasattr(workflow.test, '_last_result'):
        metadata['test_result'] = workflow.test._last_result.get('result', False)
```

**状态**：🔄 待决定选项A或B

**预期效果**：Code准确率从0% → 40-50%

---

## P1 - 本周完成（稳定性优化）

### 4. 增大批量大小，降低统计噪声

**问题**：24样��/Step太小，导致准确率波动±13%。

**修改文件**：`config/training.yaml`

**改动内容**：
```yaml
# 选项1：增加每问题的workflow数
num_return_sequences_in_group: 6  # 改为 10-12

# 选项2：增加问题数
rollout_batch_size: 4  # 改为 6-8

# 推荐：两者都增加
num_return_sequences_in_group: 10  # 6 → 10
rollout_batch_size: 6              # 4 → 6
# 总样本：4×6=24 → 6×10=60
```

**预期效果**：波动从±13% → ±5%

---

### 5. 学习率与温度保守化

**修改文件**：`config/training.yaml`

**改动内容**：
```yaml
# 学习率降低
learning_rate: 1.0e-5  # 改为 5.0e-6

# 添加梯度裁剪
max_grad_norm: 1.0  # 新增

# KL正则化权重增加
kl_coef: 0.02  # 原0.01 → 0.02

# 温度保持固定（不调度）
temperature: 0.3  # 保持不变
```

---

### 6. 增加验证频率与早停

**修改文件**：`config/training.yaml`

**改动内容**：
```yaml
# 验证频率
eval_interval: 5  # 原10 → 5

# 早停配置
early_stopping:
  enabled: true
  patience: 3  # 3次验证不提升则停止
  min_delta: 0.01  # 最小改善幅度1%
```

---

### 7. 修复Workflow变量作用域bug

**问题**：`UnboundLocalError: local variable 'revised_code' referenced before assignment`

**影响位置**：生成的Workflow代码

**修复模板**（伪代码）：
```python
# 错误的模式
if not test_result.get('result', False):
    review_result = await self.review(...)
    revised = await self.revise(...)
    revised_code = revised.get('solution', code)  # 只在if内定义

return revised_code  # ❌ revised_code可能未定义

# 正确的模式
revised_code = code  # ✅ 初始化
if not test_result.get('result', False):
    review_result = await self.review(...)
    revised = await self.revise(...)
    revised_code = revised.get('solution', revised_code)  # 更新

return revised_code  # ✅ 总是有定义
```

**修改位置**：
- `src/prompt_optimizer.py` - 更新Few-shot模板
- 在Workflow生成器的后处理中添加变量初始化验证

---

## P2 - 数学专项增强

### 8. 启用Ground Truth的LLM辅助提取

**修改文件**：`src/reward_computer.py:50-55`

**改动内容**：
```python
# 当前
self.extractor = AnswerExtractor(use_llm_fallback=False)

# 修改为
self.extractor = AnswerExtractor(use_llm_fallback=True, llm_client=...)
```

**效果**：复杂GT文本（"Each part is 30/6=5..."）提取更准确

---

### 9. Math任务快路径优化

在Workflow生成时，对Math任务优先使用简单流程：

```python
if problem_type == "math":
    # 快路径：AnswerGenerate → 提取答案 → 返回
    ans_result = await self.answer_generate(input=problem)
    answer = ans_result.get('answer', '')
    # 不调用Programmer/Test（Math不需要代码执行）
    return answer, cost
```

**效果**：Math任务从8秒降到3秒，准确率保持或提升

---

## 配置文件完整修改建议

### `config/training.yaml` 修改摘要

```yaml
# === 批量大小（降低噪声）===
rollout_batch_size: 6  # 4 → 6
num_return_sequences_in_group: 10  # 6 → 10

# === 学习率与稳定性 ===
learning_rate: 5.0e-6  # 1e-5 → 5e-6
max_grad_norm: 1.0  # 新增
kl_coef: 0.02  # 0.01 → 0.02

# === 验证与早停 ===
eval_interval: 5  # 10 → 5
early_stopping:
  enabled: true
  patience: 3
  min_delta: 0.01

# === 检查点保存 ===
save_steps: 5  # 保存最佳验证集模型
save_total_limit: 5  # 保留最近5个checkpoint
```

---

## 验证计划

### 回归测试

1. **LLM Judge测试**
```bash
python test_llm_judge.py
# 预期：6/6通过
```

2. **Answer Extractor测试**
```python
# 测试用例
test_cases = [
    ("5/324", "math", "5/324"),  # 分数保持
    ("\\boxed{36}", "math", "36"),  # LaTeX提取
    ("The answer is 42.5", "math", "42.5"),  # 文本提取
    ("$30", "qa", "30"),  # 单位剥离
]
```

3. **Code任务test_result验证**
```bash
# 采样10个Code任务，检查日志中是否出现
"💻 使用测试结果: 通过 → 10.0分"
"💻 使用测试结果: 失败 → 0.0分"
```

### 监控指标

训练恢复后，重点观察：

| 指标 | 当前 | 目标（10步内） | 目标（50步） |
|------|------|---------------|-------------|
| Math准确率 | 40% | 60-70% | 75-85% |
| Code准确率 | 0% | 20-30% | 40-50% |
| QA准确率 | 30% | 40-50% | 55-65% |
| 总体准确率 | 25% | 45-55% | 60-70% |
| 准确率波动 | ±13% | ±8% | ±5% |
| 验证集准确率 | 40% | 45-50% | 55-65% |
| 训练-验证差距 | -13% | -8% | -5% |

---

## 实施顺序

**第1阶段（今天）**：
1. ✅ 应用P0修复（Judge解析、Answer Extractor）
2. 🔄 决定Code test_result方案（选项A或B）
3. 🔄 应用选定的test_result修复

**第2阶段（明天）**：
4. 修改`config/training.yaml`（批量、学习率、验证）
5. 修复Workflow变量作用域bug
6. 重启训练，观察前10步

**第3阶段（本周）**：
7. 启用GT LLM fallback
8. 优化Math快路径
9. 持续监控到Step 50

---

## 决策点

请确认以下决策，以便我完成剩余修复：

### 1. Code任务test_result传递方式

- [ ] **选项A**：修改Workflow模板，返回3元组 `(code, cost, metadata)`
- [ ] **选项B**：在执行器中包装Test operator，注入metadata

**推荐**：选项A（更清晰，长期可维护）

### 2. AnswerExtractor LLM fallback

- [ ] **启用**：对复杂GT文本使用LLM辅助提取
- [ ] **暂不启用**：保持当前纯规则提取

**推荐**：启用（仅对GT且复杂文本，成本可控）

### 3. 批量大小调整

- [ ] **激进**：rollout_batch_size=8, num_sequences=12 (96样本/step)
- [ ] **保守**：rollout_batch_size=6, num_sequences=10 (60样本/step)
- [ ] **最小**：rollout_batch_size=4, num_sequences=10 (40样本/step)

**推荐**：保守方案（60样本/step，平衡噪声与速度）

---

## 文件修改清单

### 已完成
- [x] `src/reward_computer.py` - LLM Judge解析增强
- [x] `src/answer_extractor.py` - 分数提取修复

### 待完成
- [ ] `src/aflow_executor.py` - test_result传递（选项A或B）
- [ ] `config/training.yaml` - 批量、学习率、验证配置
- [ ] `src/prompt_optimizer.py` - Workflow变量初始化模板
- [ ] `src/reward_computer.py` - 启用LLM fallback（可选）

---

## 联系与反馈

如有问题或需要调整方案，请及时反馈。修复完成后建议：

1. 清理旧checkpoint，从干净状态重启
2. 密切监控前20步的准确率和波动
3. Step 10和Step 20进行完整验证集评估
4. 基于实际效果微调学习率和批量大小

**预期**：完成所有修复后，Math准确率将在20步内稳定到70%+，50步内达到80%+。
