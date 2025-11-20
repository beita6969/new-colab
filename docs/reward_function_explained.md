# 奖励函数详解 - 准确率与评估机制

**文档时间**: 2025-11-16
**文件**: `src/reward_computer.py`

---

## 🎯 奖励函数设计概览

### **三维加权奖励**

```python
总奖励 = 0.7 × 正确性 + 0.2 × 效率 + 0.1 × 简洁性
```

**范围**: `[-10, 10]`

---

## 📊 1. 正确性奖励 (70% 权重)

### **数学题评估** (`_compute_math_correctness`)

#### 方法：数字提取 + 精确比较

```python
def _compute_math_correctness(prediction, ground_truth):
    # 1. 从答案中提取所有数字
    pred_numbers = extract_numbers(prediction)  # 如: "答案是42" → [42]
    gt_numbers = extract_numbers(ground_truth)   # "42" → [42]

    # 2. 取最后一个数字作为最终答案
    pred_answer = pred_numbers[-1]
    gt_answer = gt_numbers[-1]

    # 3. 比较（允许数值误差）
    if abs(pred_answer - gt_answer) < 1e-4:
        return 10.0   # ✅ 完全正确
    elif abs(pred_answer - gt_answer) < 1.0:
        return 5.0    # 🟡 接近正确
    else:
        return -5.0   # ❌ 错误
```

#### 示例

| 预测 | 标准答案 | 提取数字 | 评分 | 说明 |
|------|---------|---------|------|------|
| "The answer is 42" | "42" | 42 vs 42 | **+10.0** | 完全正确 |
| "I think it's 42.0001" | "42" | 42.0001 vs 42 | **+10.0** | 误差<1e-4 |
| "About 43" | "42" | 43 vs 42 | **+5.0** | 接近 |
| "The result is 50" | "42" | 50 vs 42 | **-5.0** | 错误 |
| "I don't know" | "42" | [] vs [42] | **-8.0** | 无法提取 |
| None (执行失败) | "42" | - | **-10.0** | 失败 |

---

### **代码题评估** (`_compute_code_correctness`)

#### 方法：字符串匹配 + 函数名提取

```python
def _compute_code_correctness(prediction, ground_truth):
    # 1. 字符串包含检查
    if ground_truth.lower() in prediction.lower():
        return 10.0  # ✅ 包含正确答案

    # 2. 提取函数名
    pred_funcs = extract_function_names(prediction)   # ["add", "multiply"]
    gt_funcs = extract_function_names(ground_truth)  # ["add"]

    # 3. 检查函数名匹配
    if any(pf in gt_funcs for pf in pred_funcs):
        return 5.0   # 🟡 部分正确

    return -5.0      # ❌ 错误
```

---

### **QA题评估** (`_compute_qa_correctness`)

#### 方法：多级字符串匹配

```python
def _compute_qa_correctness(prediction, ground_truth):
    pred = prediction.strip().lower()
    gt = ground_truth.strip().lower()

    # 1. 精确匹配
    if pred == gt:
        return 10.0  # ✅ 完美

    # 2. 包含匹配
    if gt in pred:
        return 8.0   # ✅ 包含答案

    # 3. Token重叠
    overlap_ratio = len(pred_tokens & gt_tokens) / len(gt_tokens)

    if overlap_ratio > 0.8:
        return 6.0   # 🟢 高重叠
    elif overlap_ratio > 0.5:
        return 3.0   # 🟡 中等重叠
    elif overlap_ratio > 0.2:
        return 0.0   # 🟠 低重叠
    else:
        return -5.0  # ❌ 几乎无关
```

#### 示例

| 预测 | 标准答案 | 评分 | 说明 |
|------|---------|------|------|
| "Paris" | "Paris" | **+10.0** | 精确匹配 |
| "The capital is Paris" | "Paris" | **+8.0** | 包含答案 |
| "Paris is the capital of France" | "Paris" | **+8.0** | 包含答案 |
| "It's in France, the city is Paris" | "Paris" | **+6.0** | Token重叠>80% |

---

## 💰 2. 效率奖励 (20% 权重)

### **基于API成本**

```python
def _compute_efficiency_reward(cost: float):
    """
    评估API调用成本的效率

    成本阈值:
    - 最优: $0.001
    - 可接受: $0.005
    - 中等: $0.01
    - 较高: $0.05
    """
    if cost <= 0.001:
        return 10.0   # ⭐⭐⭐⭐⭐ 最优
    elif cost <= 0.005:
        return 5.0    # ⭐⭐⭐⭐ 良好
    elif cost <= 0.01:
        return 0.0    # ⭐⭐⭐ 中等
    elif cost <= 0.05:
        return -3.0   # ⭐⭐ 较高
    else:
        return -8.0   # ⭐ 过高
```

### **实际成本分布** (Step 5)

根据训练数据:
- 平均成本: **$0.000101** / 调用
- 评分范围: **+5.0 到 +10.0**
- 评级: **良好到最优**

---

## ⚡ 3. 简洁性奖励 (10% 权重)

### **双因素评估**

```python
def _compute_simplicity_reward(execution_time, num_operators):
    # 1. 时间评分
    time_reward = {
        "≤5秒": 10.0,   # 快速
        "≤15秒": 5.0,   # 正常
        "≤30秒": 0.0,   # 稍慢
        "≤60秒": -3.0,  # 较慢
        ">60秒": -5.0   # 很慢
    }

    # 2. 算子数评分
    operator_reward = {
        "≤2个": 10.0,   # 简洁
        "≤4个": 5.0,    # 适中
        "≤6个": 0.0,    # 复杂
        ">6个": -5.0    # 过度复杂
    }

    # 3. 平均
    return (time_reward + operator_reward) / 2.0
```

### **当前典型表现** (Step 5)

**执行时间**: 约 3-5 秒
**算子数量**: 2-3 个 (AnswerGenerate + Programmer)
**简洁性评分**: **+7.5 到 +10.0**

---

## 🧮 综合奖励计算示例

### **示例 1: 完美执行**

```python
问题: "What is 15 + 27?"
预测: "The answer is 42"
标准答案: "42"
成本: $0.0001
执行时间: 3秒
算子数: 2

# 计算:
correctness = 10.0  (完全正确)
efficiency = 10.0   (成本 $0.0001 ≤ $0.001)
simplicity = 10.0   (3秒, 2算子)

总奖励 = 0.7×10.0 + 0.2×10.0 + 0.1×10.0
      = 7.0 + 2.0 + 1.0
      = 10.0  ⭐⭐⭐⭐⭐
```

### **示例 2: 正确但低效**

```python
问题: "What is 15 + 27?"
预测: "The answer is 42"
标准答案: "42"
成本: $0.006
执行时间: 20秒
算子数: 5

# 计算:
correctness = 10.0  (完全正确)
efficiency = 0.0    (成本 $0.006, 在中等范围)
simplicity = 2.5    ((5.0 + 0.0) / 2 = 2.5)

总奖励 = 0.7×10.0 + 0.2×0.0 + 0.1×2.5
      = 7.0 + 0.0 + 0.25
      = 7.25  ⭐⭐⭐⭐
```

### **示例 3: 错误答案**

```python
问题: "What is 15 + 27?"
预测: "The answer is 50"
标准答案: "42"
成本: $0.0001
执行时间: 2秒
算子数: 2

# 计算:
correctness = -5.0  (错误答案)
efficiency = 10.0   (成本低)
simplicity = 10.0   (快速简洁)

总奖励 = 0.7×(-5.0) + 0.2×10.0 + 0.1×10.0
      = -3.5 + 2.0 + 1.0
      = -0.5  ⭐⭐
```

**关键发现**: 即使效率和简洁性完美，错误答案仍会导致负奖励！

### **示例 4: 执行失败**

```python
问题: "What is 15 + 27?"
预测: None  (执行失败)
标准答案: "42"
成本: $0.0
执行时间: 0秒

# 计算:
correctness = -10.0  (执行失败)
efficiency = 0.0     (无成本)
simplicity = 0.0     (无数据)

总奖励 = 0.7×(-10.0) + 0.2×0.0 + 0.1×0.0
      = -7.0
      = -7.0  ❌
```

---

## 📈 当前训练表现预估

### **典型 Workflow 表现** (Step 5)

**模式**: AnswerGenerate + Programmer

```
正确性: 8-10 分 (假设80-100%准确)
效率: 8-10 分 (成本 $0.0001)
简洁性: 7.5-10 分 (3秒, 2-3算子)

预估平均奖励:
= 0.7 × 9.0 + 0.2 × 9.0 + 0.1 × 8.5
= 6.3 + 1.8 + 0.85
= 8.95 ⭐⭐⭐⭐⭐
```

### **问题准确率监控**

目前日志中**未明确记录准确率**，但可以通过以下方式推断：

#### 间接指标

1. **Fallback 使用率**: 0%
   - 说明: 所有生成的 Workflow 都成功执行
   - 含义: 至少能产生答案（正确性待验证）

2. **eval() 错误率**: 3.1%
   - 说明: 97% 的 Workflow 代码逻辑正确
   - 含义: 高代码质量

3. **格式正确率**: 100%
   - 说明: Qwen 完全掌握代码结构
   - 含义: 基础能力扎实

#### 建议监控方案

为了明确跟踪准确率，建议：

**方案 1: 添加日志输出**
```python
# 在 grpo_trainer.py 中
print(f"正确性评分: {correctness_reward:.2f}")
print(f"问题准确: {'✓' if correctness_reward > 5 else '✗'}")
```

**方案 2: 统计聚合**
```bash
# 从日志中提取并统计
grep "correctness_reward" logs/training_output.log | \
  awk '{if ($2 > 5) correct++; total++} END {print "准确率:", correct/total*100"%"}'
```

---

## 🎯 GRPO 如何使用奖励

### **组相对优势计算**

```python
# 对于每个问题，生成4个候选Workflow
rewards = [8.95, 7.25, -0.5, -7.0]  # 4个候选的奖励

# 计算组内统计
mean = np.mean(rewards)  # 2.175
std = np.std(rewards)    # 6.52

# 计算优势
advantages = [(r - mean) / std for r in rewards]
# [1.04, 0.78, -0.41, -1.41]
```

### **策略更新**

```python
# GRPO 损失
loss = -Σ log π(workflow_i | problem) × advantage_i

# 效果:
- Workflow_1 (adv=1.04): ⬆️ 大幅增加生成概率
- Workflow_2 (adv=0.78): ⬆️ 增加生成概率
- Workflow_3 (adv=-0.41): ⬇️ 降低生成概率
- Workflow_4 (adv=-1.41): ⬇️ 大幅降低生成概率
```

---

## 💡 设计理念

### **为什么 70% 正确性？**

**原因**: 错误答案完全无价值

**效果**:
- 即使速度快、成本低，错误答案仍得负分
- 强制 Qwen 优先学习生成**正确**的 Workflow
- 避免学到"快速但错误"的捷径

### **为什么 20% 效率？**

**原因**: 在保证正确的前提下，优化成本

**效果**:
- 引导 Qwen 选择成本更低的算子
- 避免不必要的多步调用
- 如: Custom (1步) vs AnswerGenerate+Programmer (2步)

### **为什么 10% 简洁性？**

**原因**: 简洁代码更易维护

**效果**:
- 鼓励使用最少的算子解决问题
- 避免过度设计（如不必要的 Review+Revise 循环）
- 提高可读性

---

## 📊 预期学习曲线

### **Step 1-20: 学习正确性**
```
平均奖励: 3-6 分
正确性: 50-70%
效率: 良好 (成本低)
简洁性: 良好 (简单模式)
```

### **Step 20-100: 优化效率**
```
平均奖励: 6-8 分
正确性: 70-85%
效率: 开始差异化 (学会选择算子)
简洁性: 提升 (减少不必要的步骤)
```

### **Step 100-500: 策略多样化**
```
平均奖励: 8-9.5 分
正确性: 85-95%
效率: 显著优化 (根据问题选择)
简洁性: 自适应 (简单问题用简单策略)
```

---

## 🔧 可调参数

当前配置可通过 `config/training.yaml` 调整：

```yaml
reward:
  weights:
    correctness: 0.7   # 可调整 [0.5-0.9]
    efficiency: 0.2    # 可调整 [0.1-0.3]
    simplicity: 0.1    # 可调整 [0.05-0.2]

  thresholds:
    cost_optimal: 0.001      # 最优成本阈值
    cost_acceptable: 0.01    # 可接受成本
    time_optimal: 5.0        # 最优时间 (秒)
    operator_optimal: 2      # 最优算子数
```

---

## 📋 总结

### **核心设计**

1. ✅ **正确性第一** (70%): 错误答案无价值
2. ✅ **效率其次** (20%): 优化成本和时间
3. ✅ **简洁最后** (10%): 保持代码简洁

### **优势**

- ✅ 明确的优先级
- ✅ 平衡多个目标
- ✅ 可解释的评分
- ✅ 自动化评估

### **当前状态**

根据 Step 5 数据:
- 📊 格式正确率: **100%**
- 💰 成本控制: **优秀** ($0.0001)
- ⚡ 执行速度: **快** (3-5秒)
- 🎯 **预估平均奖励**: **8-9分**

**评级**: ⭐⭐⭐⭐⭐ **优秀**

---

**文档完成**: 2025-11-16
**更新频率**: 随训练进展更新
