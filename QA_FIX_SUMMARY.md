# QA答案提取修复总结

## 修复日期
2025-11-19 21:51

## 问题描述

QA任务准确率为0%，根本原因是答案提取逻辑有缺陷：
- **症状**: 提取中间计算值而非最终答案
- **示例**: 问题"How many subscribers?"，模型输出包含"$1,800"和"200 subscribers"，提取器错误地提取了"1800"而非"200"

## 修复内容

**文件**: `src/answer_extractor.py:174-222`

### 修改前逻辑
```python
def _extract_qa_answer(self, text: str, is_ground_truth: bool) -> str:
    # 1. 检查Answer:标记
    # 2. 标准化全文本
    # 3. 如果太长，提取最后两句
```

### 修改后逻辑
```python
def _extract_qa_answer(self, text: str, is_ground_truth: bool) -> str:
    # 1. 检查Answer:/The answer is:/Final answer:等标记
    #    - 如果找到，从标记内容中提取数字

    # 2. 检测是否为数值型问题（包含计算符号+,-,*,/,=,<<,>>）
    #    - 如果是，提取所有数字，返回最后一个数字
    #    - 策略：最后一个数字通常是最终答案

    # 3. 否则，按文本QA处理（标准化）
```

### 关键改进

1. **数值型QA识别**: 通过检测计算符号判断是否为数值型问题
2. **最后数字优先**: 从所有数字中选取最后一个（通常是最终结果）
3. **Answer标记增强**: 支持更多答案标记格式

## 测试结果

```python
# 测试1: 数值QA with 中间值
Input:  "He made $1,800. He needs 50 more. Final count is 200 subscribers."
Output: "200" ✅

# 测试2: 计算型QA
Input:  "3 touchdowns in first half and 3 in second. Total: 6 touchdowns."
Output: "6" ✅

# 测试3: Answer:前缀
Input:  "Calculating... Answer: 42"
Output: "42" ✅

# 测试4: 文本QA
Input:  "The capital of France is Paris."
Output: "the capital of france is paris" (标准化)
```

## 数据集分析

### QA问题类型

数据集`train/mixed_dataset.jsonl`包含两类QA：

1. **数值型QA** (answer_type="number")
   - "How many touchdowns..."
   - "How many more troops..."
   - "How many months..."
   - Ground Truth: "6", "200", 等纯数字

2. **文本型QA** (answer_type="span" 或其他)
   - "Who was sidelined..."
   - "What album was..."
   - Ground Truth: 文本片段

### 提取策略

| 类型 | 识别方式 | 提取策略 | 示例 |
|------|----------|----------|---------|
| 数值型 | 包含计算符号或数字 | 返回最后一个数字 | "$1,800...200 subscribers" → "200" |
| 文本型 | 无计算符号 | 标准化全文本 | "The capital is Paris" → "the capital is paris" |

## 预期效果

### 训练指标改善

| 指标 | 修复前 | 修复后（预期） | 提升 |
|------|--------|----------------|------|
| QA准确率 | 0% | 50-70% | +50-70% |
| QA平均得分 | -5.0 | 5.0-8.0 | +10-13 |
| QA奖励 | 0.0 | 0.7-0.9 | +0.7-0.9 |

### 修复验证方法

1. **观察QA任务日志**:
   ```bash
   grep -A 5 '答案提取对比 (qa)' logs/train_qa_fixed_20251119_215136.log
   ```

2. **检查提取结果**:
   - 「提取预测」应为单个数字（数值型）或简短文本（文本型）
   - 不应出现"1800"之类的中间值

3. **检查正确性评分**:
   - QA任务评分应从-5.0提升至正值
   - 准确率应从0%提升

## 重启信息

**时间**: 2025-11-19 21:51:36
**日志文件**: `logs/train_qa_fixed_20251119_215136.log`
**进程PID**: 609445
**GPU**: CUDA_VISIBLE_DEVICES=2

## 监控命令

实时查看训练日志:
```bash
tail -f logs/train_qa_fixed_20251119_215136.log
```

查看QA任务结果:
```bash
grep -E '(答案提取对比 \(qa\)|正确性评分.*qa)' logs/train_qa_fixed_20251119_215136.log | tail -50
```

检查进程状态:
```bash
ps aux | grep 609445
```

## 后续观察重点

1. **QA准确率提升**: 从0%提升至50%+
2. **数字提取准确性**: 不再提取中间值（如1800），正确提取最终答案（如200）
3. **文本QA稳定性**: 标准化提取是否一致
4. **与Math任务对比**: QA数值型问题应接近Math准确率

## 相关文件

修改的文件:
```
src/answer_extractor.py:174-222  - QA提取逻辑
```

历史修复:
```
P0_FIXES_SUMMARY.md              - P0修复总结
RESTART_SUMMARY.md               - 上次重启记录
```

---

生成时间: 2025-11-19 21:52
状态: ✅ 修复已部署，训练重启中
