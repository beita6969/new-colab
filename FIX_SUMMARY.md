# 修复摘要

## 修复时间
2025-11-20 14:00

## 修复内容

### 修复1: LLM Judge结果被覆盖 ✅

**文件**: `src/reward_computer.py:276-297`

**修改**:
- 移除了答案提取器的if-else分支逻辑
- LLM Judge现在是唯一的评分方式（所有任务类型）
- 计算评分后立即归一化并返回
- 使用简单的二元映射: 正确=1.0, 错误=0.0

**预期效果**:
- LLM Judge的判决直接决定最终评分
- 消除答案提取器与LLM Judge的冲突
- Math准确率: 0% → 50-60%

### 修复2: Workflow变量作用域指导 ✅

**文件**: `src/prompt_optimizer.py:102-121`

**修改**:
添加第5条规则 - 变量作用域关键指导:
```python
5️⃣ VARIABLE SCOPE CRITICAL RULE:
   ⚠️  ALWAYS initialize variables at function start, BEFORE any if/else blocks!

   ❌ WRONG:
   if condition:
       result_var = await self.revise(...)
   return result_var  # ERROR if condition is False!

   ✅ CORRECT:
   result_var = initial_value  # Initialize first
   if condition:
       result_var = await self.revise(...)
   return result_var  # Always safe
```

**预期效果**:
- Qwen生成的workflow代码将正确初始化变量
- Workflow成功率: 92% → 100%

## 测试计划

1. 快速测试（1个step，4个样本）
2. 验证指标：
   - Workflow执行成功率 = 100%
   - LLM Judge正确评分（True → 1.0, False → 0.0）
   - 无UnboundLocalError
3. 如果测试通过，重启完整训练

## 文件修改清单

- [x] src/reward_computer.py - 简化为LLM Judge only
- [x] src/prompt_optimizer.py - 添加变量作用域规则
