# 混合数据集说明文档

## 数据集位置

**路径**: `/home/yijia/.claude/11/integrated_aflow_roll/data/mixed/`

### 文件列表
- `train_mixed.jsonl` - 训练集（8,120个样本）
- `val_mixed.jsonl` - 验证集（1,264个样本）
- `dataset_stats.json` - 统计信息

## 数据集分布

### 训练集（8,120个样本）
- **数学（math）**: 2,000个样本（24.6%）来自GSM8K
- **代码（code）**: 120个样本（1.5%）来自MBPP（全部保留）
- **问答（qa）**: 3,000个样本（36.9%）来自CommonsenseQA和HotpotQA
- **混合（mixed）**: 3,000个样本（36.9%）来自MMLU

### 验证集（1,264个样本）
- **数学（math）**: 200个样本（15.8%）
- **代码（code）**: 464个样本（36.7%）全部保留（HumanEval + MBPP测试集）
- **问答（qa）**: 300个样本（23.7%）
- **混合（mixed）**: 300个样本（23.7%）

## 数据格式

所有样本采用统一的JSON格式：

```json
{
  "source": "数据集来源",
  "problem_type": "问题类型（math/code/qa/mixed）",
  "problem": "问题文本（包含选项）",
  "ground_truth": "正确答案",
  ... 其他字段根据类型不同而不同
}
```

### 各类型样本示例

#### 数学题（GSM8K）
```json
{
  "source": "gsm8k",
  "problem_type": "math",
  "problem": "Amara had 100 pieces of clothing...",
  "ground_truth": "65",
  "solution": "完整的解题步骤..."
}
```

#### 代码题（MBPP/HumanEval）
```json
{
  "source": "mbpp",
  "problem_type": "code",
  "problem": "Write a function to...",
  "ground_truth": "def function_name():\n    ...",
  "test_list": ["assert ...", "assert ..."]
}
```

#### 问答题（CommonsenseQA）
```json
{
  "source": "commonsenseqa",
  "problem_type": "qa",
  "problem": "What would a person expect...? Choices: A. compliments B. passing grade C. intellectual challenge D. sticker E. avoid pain",
  "ground_truth": "B",
  "choices": {
    "label": ["A", "B", "C", "D", "E"],
    "text": ["compliments", "passing grade", ...]
  }
}
```

#### 混合题（MMLU）
```json
{
  "source": "mmlu",
  "problem_type": "mixed",
  "problem": "A listener from Brazil... Choices: A. a radio announcer B. a journalist C. a university professor D. a tour guide",
  "ground_truth": "A",
  "subject": "reading comprehension",
  "choices": ["a radio announcer", "a journalist", "a university professor", "a tour guide"]
}
```

## 关键特性

### 1. 选择题完整处理
- **CommonsenseQA**: 选项正确从嵌套的question.choices中提取
- **MMLU**: 答案从数字索引（0,1,2,3）转换为字母（A,B,C,D）
- 所有选择题的选项都包含在problem文本中，格式为 "Choices: A. ... B. ... C. ..."

### 2. 小样本保留策略
- 样本数少于500的数据集全部保留
- MBPP训练集（120个）全部用于训练
- HumanEval（164个）和MBPP测试集（300个）全部用于验证

### 3. 验证集来源
- 所有验证集样本均来自原始数据集的测试/dev分割
- 验证集保持与训练集相同的类型比例（代码类除外，因小样本全保留）

### 4. 统一格式
- 所有样本转换为统一格式
- 保留原始数据的重要字段（如choices、test_list等）
- 便于统一的评估和训练流程

## 使用方式

在训练配置文件中使用：

```yaml
data:
  train_path: data/mixed/train_mixed.jsonl
  val_path: data/mixed/val_mixed.jsonl

domain_ratios:
  math: 0.246    # 实际训练集比例
  code: 0.015
  qa: 0.369
  mixed: 0.369
```

## 生成脚本

脚本位置: `scripts/create_mixed_dataset.py`

重新生成数据集:
```bash
python3 scripts/create_mixed_dataset.py
```

修改参数:
- `--target-size`: 目标训练集大小（默认10000）
- `--keep-small-threshold`: 小数据集阈值（默认500）
- `--seed`: 随机种子（默认42）

## 数据质量检查

所有选择题已验证:
- ✅ CommonsenseQA: 选项完整，答案为字母
- ✅ MMLU: 选项完整，答案已转换为字母
- ✅ 所有样本格式统一
- ✅ 验证集与训练集比例一致
