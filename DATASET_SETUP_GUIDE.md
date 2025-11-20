# 📚 Complete Dataset Setup Guide for AFlow+ROLL

## 快速开始

### 1. 下载所有数据集

```bash
# 进入项目目录
cd /home/yijia/.claude/11/integrated_aflow_roll

# 安装依赖
pip install datasets transformers tqdm requests

# 运行下载脚本
python scripts/download_all_datasets.py
```

这将下载以下数据集：
- **GSM8K** (8,500题) - 数学推理
- **HumanEval** (164题) - 代码生成
- **MBPP** (1,000题) - 基础编程
- **CommonsenseQA** (12,247题) - 常识推理
- **HotpotQA** (113,000题) - 多跳推理
- **MMLU** (15,000题) - 多领域知识

### 2. 数据集结构

下载完成后，数据集将按以下结构组织：

```
data/
├── raw/                    # 原始数据集
│   ├── gsm8k/
│   │   ├── train.jsonl    # 7,473 训练样本
│   │   └── test.jsonl     # 1,319 测试样本
│   ├── humaneval/
│   │   └── HumanEval.jsonl # 164 编程题
│   ├── mbpp/
│   │   ├── train.jsonl
│   │   ├── validation.jsonl
│   │   └── test.jsonl
│   ├── commonsenseqa/
│   │   ├── train.jsonl    # 9,741 训练
│   │   ├── validation.jsonl # 1,221 验证
│   │   └── test.jsonl     # 1,285 测试
│   ├── hotpotqa/
│   │   └── dev_distractor.json
│   └── mmlu/
│       ├── train.jsonl
│       ├── validation.jsonl
│       └── test.jsonl
│
└── processed/              # 处理后的混合数据
    ├── train_mixed.jsonl   # 1000 训练样本
    ├── val_mixed.jsonl     # 100 验证样本
    └── test_mixed.jsonl    # 100 测试样本
```

## 数据集详情

### 数学推理 - GSM8K

**格式示例**：
```json
{
  "question": "Natalia sold clips to 48 of her friends...",
  "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips...\n#### 72"
}
```

**评估方法**：
- 提取`####`后的数值答案
- 数值容差比较 (1e-4)
- 准确率计算

### 代码生成 - HumanEval & MBPP

**HumanEval格式**：
```json
{
  "task_id": "HumanEval/0",
  "prompt": "def has_close_elements(numbers: List[float], threshold: float) -> bool:",
  "entry_point": "has_close_elements",
  "test": "def check(candidate):\n    assert candidate(...)"
}
```

**评估方法**：
- Pass@k指标 (k=1,10,100)
- 代码执行测试
- 超时保护 (5秒)

### 问答 - CommonsenseQA

**格式示例**：
```json
{
  "question": "Where do you put groceries?",
  "choices": {
    "label": ["A", "B", "C", "D", "E"],
    "text": ["pantry", "shelf", "refrigerator", "cabinet", "kitchen"]
  },
  "answerKey": "C"
}
```

**评估方法**：
- 多选题准确率
- 选项匹配

### 多跳推理 - HotpotQA

**格式示例**：
```json
{
  "question": "What government position was held by...",
  "answer": "Chief of Protocol",
  "supporting_facts": [["title1", 0], ["title2", 2]],
  "context": [["title", ["sentence1", "sentence2"]]]
}
```

**评估方法**：
- 答案F1分数
- 支撑事实F1分数
- 联合评分

### 综合评估 - MMLU

**格式示例**：
```json
{
  "question": "Question text",
  "choices": ["A", "B", "C", "D"],
  "answer": "B",
  "subject": "abstract_algebra"
}
```

**评估方法**：
- 57个学科分类准确率
- 整体准确率
- 领域别准确率 (STEM/人文/社科)

## 使用评估函数

### 基础使用

```python
from src.unified_evaluator import UnifiedEvaluator

# 创建评估器
evaluator = UnifiedEvaluator()

# 评估数学题
math_result = evaluator.evaluate(
    prediction="The answer is 42.",
    ground_truth="#### 42",
    problem_type="math"
)
print(f"Math correct: {math_result['correct']}")

# 评估代码
code_result = evaluator.evaluate(
    prediction="def add(a, b): return a + b",
    ground_truth="def add(a, b): return a + b",
    problem_type="code",
    test="assert add(1, 2) == 3"
)
print(f"Code passed: {code_result['correct']}")

# 评估问答
qa_result = evaluator.evaluate(
    prediction="The answer is B.",
    ground_truth="B",
    problem_type="multiple_choice"
)
print(f"QA correct: {qa_result['correct']}")
```

### 批量评估

```python
from src.unified_evaluator import DatasetSpecificEvaluator

# 创建数据集评估器
ds_evaluator = DatasetSpecificEvaluator()

# 评估GSM8K
gsm8k_results = ds_evaluator.evaluate_gsm8k(
    predictions=model_predictions,
    dataset_path="./data/raw/gsm8k/test.jsonl"
)
print(f"GSM8K Accuracy: {gsm8k_results['overall_accuracy']:.2%}")

# 评估HumanEval
humaneval_results = ds_evaluator.evaluate_humaneval(
    predictions=model_predictions,
    dataset_path="./data/raw/humaneval/HumanEval.jsonl",
    k_values=[1, 10, 100]
)
print(f"Pass@1: {humaneval_results['pass_at_k']['pass@1']:.2%}")

# 评估MMLU
mmlu_results = ds_evaluator.evaluate_mmlu(
    predictions=model_predictions,
    dataset_path="./data/raw/mmlu/test.jsonl"
)
print(f"MMLU Accuracy: {mmlu_results['overall_accuracy']:.2%}")
```

## 集成到训练流程

### 1. 更新训练配置

编辑 `config/training.yaml`，添加数据集配置：

```yaml
data:
  train_path: ./data/processed/train_mixed.jsonl
  val_path: ./data/processed/val_mixed.jsonl
  test_path: ./data/processed/test_mixed.jsonl

  # 领域比例
  domain_ratios:
    math: 0.30    # GSM8K
    code: 0.25    # HumanEval + MBPP
    qa: 0.25      # CommonsenseQA + HotpotQA
    mixed: 0.20   # MMLU
```

### 2. 运行训练

```bash
# 使用新数据集训练
python train.py --config config/training.yaml
```

### 3. 监控训练

```bash
# 查看训练日志
tail -f logs/training.log

# 查看评估结果
python analyze_training.py --checkpoint checkpoints/step_50
```

## 数据集统计

| 数据集 | 训练 | 验证 | 测试 | 类型 | 评估指标 |
|--------|------|------|------|------|----------|
| GSM8K | 7,473 | - | 1,319 | 数学 | 准确率 |
| HumanEval | - | - | 164 | 代码 | Pass@k |
| MBPP | ~374 | 90 | 500 | 代码 | Pass@k |
| CommonsenseQA | 9,741 | 1,221 | 1,285 | QA | 准确率 |
| HotpotQA | 90,000+ | 7,000+ | - | 多跳 | F1分数 |
| MMLU | - | 285 | 14,042 | 综合 | 准确率 |

## 性能基准

预期性能指标（基于Qwen2.5-7B + LoRA）：

- **GSM8K**: 60-70% 准确率
- **HumanEval**: Pass@1 30-40%
- **MBPP**: Pass@1 40-50%
- **CommonsenseQA**: 70-75% 准确率
- **HotpotQA**: F1 60-65%
- **MMLU**: 55-60% 准确率

经过GRPO训练后，预期提升10-15%。

## 常见问题

### Q: 下载失败怎么办？
A: 检查网络连接，使用代理或镜像源：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Q: 内存不足？
A: 使用流式加载：
```python
dataset = load_dataset("dataset_name", streaming=True)
```

### Q: 如何添加新数据集？
A: 编辑 `config/datasets.yaml` 添加新数据集配置，然后更新下载脚本。

## 下一步

1. ✅ 数据集下载完成
2. ✅ 评估函数就绪
3. 🎯 开始训练：`python train.py`
4. 📊 监控进度：使用WandB或TensorBoard
5. 🔬 调优超参数：基于验证集性能

---

**提示**：建议先用小批量数据（100-1000样本）测试流程，确认无误后再全量训练。
