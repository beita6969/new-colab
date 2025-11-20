# GRPO 训练完整流程详解

**文档时间**: 2025-11-16
**训练状态**: ✅ 正常运行 (PID 2255398)

---

## 🎯 整体架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                      GRPO 训练循环主流程                           │
│                                                                   │
│  1. 数据加载     2. Qwen生成    3. AFlow执行    4. 评估奖励    5. 策略更新 │
│     ⬇️              ⬇️              ⬇️              ⬇️              ⬇️      │
│  MATH数据集  →  生成代码  →  调用GPT执行  →  计算reward  →  更新Qwen   │
│  (500题)      (7B本地)     (4o-mini API)   (组相对优势)   (LoRA微调)  │
└─────────────────────────────────────────────────────────────────┘
```

**关键点**:
- **Qwen 模型**: 本地运行在 GPU 2-3，生成 Python Workflow 代码
- **OpenAI API**: 仅在执行生成的 Workflow 时调用，用于求解问题
- **评估方法**: 正确性 (70%) + 效率 (20%) + 简洁性 (10%)
- **训练方法**: GRPO（组相对策略优化），每步 4 个候选，选优更新

---

## 📊 完整数据流详解

### Phase 1: 数据采样
```python
# train.py:206-214
batch = dataset.sample(batch_size=4)  # 采样 4 个问题
# 示例问题: "If a = 3 and b = 4, what is a^2 + b^2?"
```

**输出**:
```json
{
  "problem": "If a = 3 and b = 4, what is a^2 + b^2?",
  "answer": "25",          # 标准答案（用于评估）
  "type": "algebra",
  "difficulty": "level_1"
}
```

---

### Phase 2: Qwen 生成 Workflow 代码

#### 2.1 构建 Prompt
```python
# src/rl_workflow_generator.py:113-154
prompt = f"""Generate a Python Workflow class. Follow the exact template and API signatures.

CRITICAL: Only use operators listed below with their EXACT parameters!

Available Operators:

1. Custom(llm) - Most flexible, for any custom task
   Call: await self.custom(input=str, instruction=str)
   Returns: {'response': str}

2. AnswerGenerate(llm) - Step-by-step reasoning
   Call: await self.answer_generate(input=str)  ← NO instruction parameter!
   Returns: {'thought': str, 'answer': str}

3. Programmer(llm) - Auto-generate and execute Python code
   Call: await self.programmer(problem=str, analysis=str)
   Returns: {'code': str, 'output': str}

Template (complete the __call__ method):

import workspace.math.workflows.template.operator as operator
from scripts.async_llm import create_llm_instance
from scripts.evaluator import DatasetType

class Workflow:
    def __init__(self, name: str, llm_config, dataset: DatasetType):
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        # Example: self.custom = operator.Custom(self.llm)

    async def __call__(self, problem: str):
        # Solve: {problem}
        # MUST return (solution, cost) tuple
        # Example: return solution['response'], self.llm.get_usage_summary()["total_cost"]
        pass
"""
```

#### 2.2 Qwen 模型推理（本地 GPU）
```python
# src/rl_workflow_generator.py:187-199
inputs = tokenizer(prompt, return_tensors="pt").to("cuda:2")

# 关键参数
outputs = model.generate(
    **inputs,
    max_new_tokens=2048,
    temperature=0.1,      # ✨ 低温度 = 更严格遵循模板
    top_p=0.95,
    do_sample=True
)

generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

**Qwen 生成的实际代码示例**（从日志中提取）:
```python
class Workflow:
    def __init__(self, name: str, llm_config, dataset: DatasetType):
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)
        self.answer_generate = operator.AnswerGenerate(self.llm)
        self.programmer = operator.Programmer(self.llm)

    async def __call__(self, problem: str):
        # Step 1: 使用 AnswerGenerate 生成推理过程
        thought_response = await self.answer_generate(input=problem)
        thought = thought_response['thought']

        # Step 2: 使用 Programmer 生成并执行代码
        programmer_response = await self.programmer(problem=problem, analysis=thought)
        code = programmer_response['code']
        output = programmer_response['output']

        # Step 3: 提取解决方案
        solution = output.strip() if output else "No solution found"

        # Step 4: 返回 (solution, cost) 元组
        return solution, self.llm.get_usage_summary()["total_cost"]
```

**关键点**:
- ✅ 生成了正确的 `class Workflow:` 结构
- ✅ 使用了正确的算子 API（`answer_generate(input=problem)` 没有 instruction）
- ✅ 返回了 `(solution, cost)` 元组
- ⚠️ 可能包含一些冗余代码，但语法正确

---

### Phase 3: AFlow 执行 Workflow

#### 3.1 动态加载生成的代码
```python
# src/aflow_executor.py:213-242
namespace = {
    "operator": operator_module,
    "create_llm_instance": create_llm_instance,
    "DatasetType": str
}

# 执行代码创建 Workflow 类
exec(workflow_code, namespace)
WorkflowClass = namespace["Workflow"]
```

#### 3.2 实例化 Workflow
```python
# src/aflow_executor.py:109-121
llm_config = self._get_llm_config()  # 返回 LLMConfig 实例
workflow = WorkflowClass(
    name="rl_generated_workflow",
    llm_config=llm_config,         # gpt-4o-mini 配置
    dataset="math"
)
```

**这时发生了什么**:
```python
# 在 Workflow.__init__ 中
self.llm = create_llm_instance(llm_config)
# 创建了 AsyncLLM 实例，配置为 gpt-4o-mini

self.answer_generate = operator.AnswerGenerate(self.llm)
# 创建了算子，内部持有 gpt-4o-mini 的 AsyncLLM
```

#### 3.3 调用 Workflow 求解问题（调用 OpenAI API）
```python
# src/aflow_executor.py:154-157
result = await asyncio.wait_for(
    workflow(problem),  # 调用 Workflow.__call__
    timeout=300
)
answer, cost = result
```

**Workflow 内部执行流程**（会调用 OpenAI API）:

**Step 1**: `await self.answer_generate(input=problem)`
```python
# scripts/operators.py: AnswerGenerate
async def __call__(self, input: str):
    # ⚠️ 这里会调用 OpenAI API！
    response = await self.llm.call_with_format(
        prompt=f"Solve step by step:\n{input}",
        format_type="answer_generate"
    )
    return {'thought': '...', 'answer': '25'}
```

**实际 API 调用**（从日志中看到）:
```
Token usage: 138 input + 126 output = 264 total
Cost: $0.000096 ($0.000021 for input, $0.000076 for output)
```
→ **这是调用 gpt-4o-mini 的成本**

**Step 2**: `await self.programmer(problem=problem, analysis=thought)`
```python
# scripts/operators.py: Programmer
async def __call__(self, problem: str, analysis: str):
    # ⚠️ 这里又会调用 OpenAI API！
    code_response = await self.llm.call_with_format(
        prompt=f"Generate Python code to solve:\n{problem}\n\nAnalysis: {analysis}",
        format_type="code"
    )

    # 执行生成的代码
    output = exec(code_response['code'])

    return {'code': code_response['code'], 'output': output}
```

**实际 API 调用**:
```
Token usage: 383 input + 108 output = 491 total
Cost: $0.000122 ($0.000057 for input, $0.000065 for output)
```

**总成本**: $0.000096 + $0.000122 = **$0.000218** per problem

---

### Phase 4: 评估与奖励计算

#### 4.1 正确性评估
```python
# src/reward_calculator.py:46-62
def _evaluate_correctness(self, answer, ground_truth):
    # 标准化答案（移除空格、标点）
    pred = self._normalize_answer(answer)
    gt = self._normalize_answer(ground_truth)

    if pred == gt:
        return 1.0  # 完全正确
    elif self._partial_match(pred, gt):
        return 0.5  # 部分正确
    else:
        return 0.0  # 错误
```

**示例**:
```python
ground_truth = "25"
answer = "25"  # Workflow 输出
correctness = 1.0  # ✅ 完全正确
```

#### 4.2 效率评估
```python
# src/reward_calculator.py:64-73
def _evaluate_efficiency(self, cost, execution_time):
    # 标准化：目标成本 $0.001，目标时间 10秒
    cost_score = max(0, 1 - (cost / 0.001))
    time_score = max(0, 1 - (execution_time / 10))

    return 0.5 * cost_score + 0.5 * time_score
```

**示例**:
```python
cost = $0.000218  # 实际成本
cost_score = 1 - (0.000218 / 0.001) = 0.782

execution_time = 2.3秒
time_score = 1 - (2.3 / 10) = 0.77

efficiency = 0.5 * 0.782 + 0.5 * 0.77 = 0.776
```

#### 4.3 简洁性评估
```python
# src/reward_calculator.py:75-86
def _evaluate_simplicity(self, workflow_code):
    # 统计使用的算子数量
    operators_used = 0
    for op in ['Custom', 'AnswerGenerate', 'Programmer', 'ScEnsemble']:
        if f"operator.{op}" in workflow_code:
            operators_used += 1

    # 理想情况：1-2 个算子
    if operators_used <= 2:
        return 1.0
    else:
        return max(0, 1 - 0.2 * (operators_used - 2))
```

**示例**:
```python
workflow_code = "... AnswerGenerate ... Programmer ..."
operators_used = 2
simplicity = 1.0  # ✅ 完美简洁
```

#### 4.4 综合奖励
```python
# src/reward_calculator.py:31-44
reward = (
    0.7 * correctness +   # 0.7 * 1.0 = 0.7
    0.2 * efficiency +    # 0.2 * 0.776 = 0.155
    0.1 * simplicity      # 0.1 * 1.0 = 0.1
)
# Total = 0.955 ✅ 高分！
```

---

### Phase 5: GRPO 策略更新

#### 5.1 生成 4 个候选 Workflow
```python
# train.py:233-246
outputs = []
for i in range(4):  # 每个问题生成 4 个候选
    workflow_code = rl_generator.generate_workflow(
        problem=problem,
        problem_type="math"
    )

    answer, cost, metadata = await executor.execute_workflow(
        workflow_code, problem
    )

    reward = reward_calc.calculate_reward(answer, gt, cost, ...)
    outputs.append((workflow_code, reward))
```

**示例输出**:
```python
[
    ("Workflow_1", reward=0.955),  # AnswerGenerate + Programmer
    ("Workflow_2", reward=0.823),  # Custom only
    ("Workflow_3", reward=0.701),  # ScEnsemble (慢)
    ("Workflow_4", reward=0.645)   # 语法错误，用fallback
]
```

#### 5.2 计算组相对优势（Group-Relative Advantage）
```python
# train.py:278-284
rewards = [0.955, 0.823, 0.701, 0.645]
mean_reward = np.mean(rewards)  # 0.781
std_reward = np.std(rewards)    # 0.131

advantages = [(r - mean_reward) / (std_reward + 1e-8) for r in rewards]
# [1.33, 0.32, -0.61, -1.04]
```

**解释**:
- `Workflow_1` (adv=1.33): **大幅优于平均** → 增加其概率
- `Workflow_2` (adv=0.32): **略优于平均** → 小幅增加
- `Workflow_3` (adv=-0.61): **略差于平均** → 小幅降低
- `Workflow_4` (adv=-1.04): **远差于平均** → 大幅降低

#### 5.3 计算策略梯度损失
```python
# train.py:290-306
loss = 0
for i, (workflow_code, advantage) in enumerate(zip(outputs, advantages)):
    # 重新计算 log_prob
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model(**inputs, labels=tokenizer(workflow_code, return_tensors="pt").input_ids)

    log_prob = -outputs.loss  # 负对数似然

    # GRPO 损失：-log_prob * advantage
    loss += -log_prob * advantage

loss = loss / 4  # 平均
```

**数学解释**:
```
L = -Σ log π(a|s) * A(s,a)

对于 Workflow_1 (A=1.33):
  增加 log π(Workflow_1|problem) → 更容易生成类似代码

对于 Workflow_4 (A=-1.04):
  减少 log π(Workflow_4|problem) → 更难生成类似代码
```

#### 5.4 反向传播更新 LoRA 参数
```python
# train.py:310-315
optimizer.zero_grad()
loss.backward()  # 计算梯度
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
optimizer.step()  # 更新参数
```

**更新的参数**:
- LoRA 参数: 20,000,000 个（仅 0.26% 的总参数）
- 基座模型: 冻结，不更新

**效果**:
- Qwen 学习到：**AnswerGenerate + Programmer** 是高奖励的模式
- 逐渐减少生成 **ScEnsemble**（太慢）和错误代码

---

## 🔄 训练循环完整时序图

```
时刻 T=0 (Step 1 开始)
├─ [00:00] 采样 4 个问题
│
├─ [00:05] Qwen 生成 4×4=16 个 Workflow（本地 GPU）
│   ├─ Problem 1 → [Workflow_1a, 1b, 1c, 1d]
│   ├─ Problem 2 → [Workflow_2a, 2b, 2c, 2d]
│   ├─ Problem 3 → [Workflow_3a, 3b, 3c, 3d]
│   └─ Problem 4 → [Workflow_4a, 4b, 4c, 4d]
│
├─ [00:10] 执行 16 个 Workflow（调用 OpenAI API）
│   ├─ Workflow_1a(problem_1) → API call → answer, cost
│   ├─ Workflow_1b(problem_1) → API call → answer, cost
│   └─ ... (16 次 API 调用，并发执行)
│
├─ [00:35] 计算 16 个奖励
│   └─ 每个 Workflow → [correctness, efficiency, simplicity] → reward
│
├─ [00:40] 计算组相对优势（4 组，每组 4 个）
│   ├─ Problem 1: advantages = [1.2, 0.5, -0.8, -0.9]
│   └─ Problem 2-4: 类似
│
├─ [00:45] 计算损失并反向传播
│   └─ loss = -Σ log_prob * advantage
│
├─ [00:50] 更新 LoRA 参数
│   └─ optimizer.step()
│
└─ [00:55] 保存检查点 → Step 2 开始

总耗时：约 55 秒/step
总 API 调用：16 次/step（batch_size=4, candidates=4）
总 API 成本：约 $0.0035/step（16 × $0.00022）
```

---

## 🤖 Qwen vs OpenAI API 职责划分

### Qwen2.5-7B-Instruct (本地 GPU 2-3)
**职责**: 生成 Workflow 代码（策略网络）

**输入**:
```
Prompt: "Generate a Workflow to solve: If a=3, b=4, what is a^2+b^2?"
```

**输出**:
```python
class Workflow:
    async def __call__(self, problem: str):
        result = await self.answer_generate(input=problem)
        return result['answer'], self.llm.get_usage_summary()["total_cost"]
```

**运行频率**:
- 每个问题生成 4 次（探索不同策略）
- 每步 4 个问题 → 16 次生成
- 500 步 → 8000 次生成

**成本**: 免费（本地运行）

---

### OpenAI gpt-4o-mini (API)
**职责**: 执行 Workflow 中的算子（求解实际问题）

**调用场景**:
1. **Custom 算子**: 自定义指令
   ```python
   await self.custom(input="What is 3^2 + 4^2?", instruction="Solve step by step")
   # API 调用: gpt-4o-mini
   ```

2. **AnswerGenerate 算子**: 推理求解
   ```python
   await self.answer_generate(input="What is 3^2 + 4^2?")
   # API 调用: gpt-4o-mini
   ```

3. **Programmer 算子**: 生成并执行代码
   ```python
   await self.programmer(problem="What is 3^2 + 4^2?", analysis="...")
   # API 调用: gpt-4o-mini（生成代码）
   ```

**运行频率**:
- 每个 Workflow 调用 1-3 次（取决于算子数量）
- 平均 2 次/Workflow
- 每步 16 个 Workflow × 2 = 32 次 API 调用
- 500 步 → 16,000 次 API 调用

**成本**:
- 单次: $0.0001-0.0003
- 每步: $0.005-0.01
- 总计（500步）: **$2.5-5.0**

---

## 📈 评估方法详解

### 1. 正确性评估（70% 权重）

#### 方法 1: 精确匹配
```python
def exact_match(pred, gt):
    return 1.0 if normalize(pred) == normalize(gt) else 0.0

# 示例
exact_match("25", "25") → 1.0
exact_match("25.0", "25") → 1.0 (标准化后相同)
exact_match("24", "25") → 0.0
```

#### 方法 2: 数值容差匹配（数学问题）
```python
def numerical_match(pred, gt, tolerance=1e-4):
    try:
        pred_num = float(extract_number(pred))
        gt_num = float(extract_number(gt))
        return 1.0 if abs(pred_num - gt_num) < tolerance else 0.0
    except:
        return exact_match(pred, gt)

# 示例
numerical_match("25.0001", "25") → 1.0
numerical_match("25.1", "25") → 0.0
```

#### 方法 3: 部分匹配（复杂答案）
```python
def partial_match(pred, gt):
    # 检查关键字是否存在
    pred_tokens = set(tokenize(pred))
    gt_tokens = set(tokenize(gt))

    overlap = len(pred_tokens & gt_tokens) / len(gt_tokens)

    if overlap > 0.8:
        return 1.0
    elif overlap > 0.5:
        return 0.5
    else:
        return 0.0
```

---

### 2. 效率评估（20% 权重）

#### 公式
```python
efficiency = 0.5 * cost_score + 0.5 * time_score

cost_score = max(0, 1 - cost / cost_target)
time_score = max(0, 1 - time / time_target)
```

#### 目标值
- `cost_target` = $0.001（单个问题）
- `time_target` = 10秒

#### 示例
```python
# Workflow A: AnswerGenerate only
cost = $0.0001, time = 1.5秒
cost_score = 1 - 0.0001/0.001 = 0.9
time_score = 1 - 1.5/10 = 0.85
efficiency = 0.5 * 0.9 + 0.5 * 0.85 = 0.875

# Workflow B: AnswerGenerate + Programmer + ScEnsemble
cost = $0.0015, time = 15秒
cost_score = 1 - 0.0015/0.001 = 0 (超预算)
time_score = 1 - 15/10 = 0 (超时)
efficiency = 0.0
```

**奖励信号**: Qwen 学习避免使用昂贵的算子组合

---

### 3. 简洁性评估（10% 权重）

#### 算子计数
```python
def count_operators(code):
    count = 0
    for op in ['Custom', 'AnswerGenerate', 'Programmer', 'ScEnsemble', 'Review', 'Revise']:
        if f"operator.{op}" in code:
            count += 1
    return count

def simplicity_score(count):
    if count <= 2:
        return 1.0
    elif count == 3:
        return 0.8
    elif count == 4:
        return 0.6
    else:
        return max(0, 0.6 - 0.2 * (count - 4))
```

#### 示例
```python
# Workflow A: 只用 Custom
count = 1 → simplicity = 1.0

# Workflow B: AnswerGenerate + Programmer
count = 2 → simplicity = 1.0

# Workflow C: AnswerGenerate + Programmer + Review + Revise
count = 4 → simplicity = 0.6
```

**奖励信号**: Qwen 学习用最少的算子完成任务

---

## 💡 奖励函数设计理念

### 组合策略
```python
reward = 0.7 * correctness + 0.2 * efficiency + 0.1 * simplicity
```

### 权重设计原因

#### 70% 正确性
- **理由**: 错误答案无价值，必须优先保证正确
- **效果**: 即使效率低，只要正确也能得 0.7 分
- **训练目标**: Qwen 首先学会生成能得到正确答案的 Workflow

#### 20% 效率
- **理由**: 在保证正确的前提下，优化成本和速度
- **效果**: 正确且高效的 Workflow 得分 0.9+
- **训练目标**: Qwen 学会避免不必要的多步调用

#### 10% 简洁性
- **理由**: 简洁代码更易维护，也暗示更好的策略
- **效果**: 简单直接的解法比复杂的略优
- **训练目标**: Qwen 学会用最简单的方式解决问题

### 设计权衡

#### 场景 1: 复杂但准确 vs 简单但错误
```python
Workflow A: AnswerGenerate + Programmer + Review (复杂)
  correctness = 1.0, efficiency = 0.7, simplicity = 0.8
  reward = 0.7×1.0 + 0.2×0.7 + 0.1×0.8 = 0.92

Workflow B: Custom only (简单)
  correctness = 0.6, efficiency = 0.9, simplicity = 1.0
  reward = 0.7×0.6 + 0.2×0.9 + 0.1×1.0 = 0.70

✅ Workflow A 获胜（正确性为王）
```

#### 场景 2: 两者都正确，但效率不同
```python
Workflow A: AnswerGenerate only
  correctness = 1.0, efficiency = 0.9, simplicity = 1.0
  reward = 0.7×1.0 + 0.2×0.9 + 0.1×1.0 = 0.98

Workflow B: AnswerGenerate + Programmer + ScEnsemble
  correctness = 1.0, efficiency = 0.3, simplicity = 0.6
  reward = 0.7×1.0 + 0.2×0.3 + 0.1×0.6 = 0.82

✅ Workflow A 获胜（简单高效）
```

---

## 🔧 GRPO 算法核心

### 为什么用 GRPO 而不是 PPO？

#### PPO 问题
- 需要大量样本（millions）
- 需要 Value Network（额外 7B 参数）
- 训练不稳定（KL 散度难控制）

#### GRPO 优势
- 只需对比组内相对优劣（4 个候选即可）
- 不需要 Value Network（节省内存）
- 稳定性更好（组归一化）

### GRPO 数学原理

#### 标准 RL 目标
```
maximize E[R(s,a)]  # 最大化期望奖励
```

#### PPO 目标
```
L = E[min(
    π_new(a|s)/π_old(a|s) * A(s,a),
    clip(π_new(a|s)/π_old(a|s), 1-ε, 1+ε) * A(s,a)
)]
```

#### GRPO 目标
```
对于每组 {a1, a2, a3, a4}:
  mean_R = mean([R(s,a1), R(s,a2), R(s,a3), R(s,a4)])
  std_R = std([R(s,a1), R(s,a2), R(s,a3), R(s,a4)])

  A(s,ai) = (R(s,ai) - mean_R) / (std_R + 1e-8)

L = E[log π(a|s) * A(s,a)]
```

**关键区别**: 优势 A 基于**组内相对表现**，而非全局 baseline

---

## 📊 训练监控指标

### 主要指标

```python
metrics = {
    "avg_reward": 0.82,           # 平均奖励
    "avg_correctness": 0.75,      # 平均正确率
    "avg_cost": 0.00025,          # 平均API成本
    "avg_execution_time": 3.2,    # 平均执行时间
    "fallback_rate": 0.15,        # Fallback使用率
    "valid_generation_rate": 0.85 # 有效生成率
}
```

### 期望趋势

#### 成功训练的信号
- ✅ `avg_reward` 从 0.6 → 0.9+
- ✅ `avg_correctness` 从 0.5 → 0.95+
- ✅ `fallback_rate` 从 0.5 → 0.05
- ✅ `valid_generation_rate` 从 0.6 → 0.98+
- ✅ `avg_cost` 保持稳定或降低

#### 失败训练的信号
- ❌ `avg_reward` 震荡不收敛
- ❌ `fallback_rate` 居高不下 (>0.3)
- ❌ `avg_cost` 持续上升
- ❌ Loss 爆炸或 NaN

---

## 🎯 当前训练状态

**进程**: PID 2255398
**GPU**: 2-3 (CUDA_VISIBLE_DEVICES)
**Step**: 1/500
**模型**: Qwen2.5-7B-Instruct + LoRA
**温度**: 0.1（严格模板遵循）

**最新生成质量**:
- ✅ 生成正确的 `class Workflow:` 结构
- ✅ 使用正确的算子 API
- ✅ 返回正确的 `(solution, cost)` 元组
- ⚠️ 偶尔添加不必要的辅助方法（将通过训练改进）

**API 调用情况**:
- ✅ 每个 Workflow 执行时调用 gpt-4o-mini
- ✅ 典型成本: $0.0002-0.0003/问题
- ✅ Qwen 本身不调用 API（完全本地运行）

---

## 📋 总结

### Qwen 在做什么？
**生成 Python 代码**（Workflow 类），用于组合 AFlow 算子来解决问题。

### OpenAI API 在做什么？
**执行算子中的实际推理**（求解数学题、生成代码等）。

### 评估方法是什么？
**三维评分**: 正确性（70%）+ 效率（20%）+ 简洁性（10%）。

### 奖励函数如何设计？
**组相对优势**: 同一问题的 4 个候选相互比较，优者增强、劣者抑制。

### 完整流程总结
```
1. 采样问题 → 2. Qwen生成代码(本地) → 3. 执行代码(调用API) →
4. 评估奖励 → 5. 计算优势 → 6. 更新Qwen参数 → 重复
```

---

**文档完成** ✅
**下一步**: 持续监控训练进度，观察 `avg_reward` 和 `fallback_rate` 趋势
