# WA-GRPO 训练数据集总结文档

**更新时间**: 2025-01-XX
**数据集版本**: train_10k_final.jsonl (P6修复版)

---

## 1. 数据集概览

### 1.1 基本统计

| 项目 | 数值 |
|------|------|
| **总样本数** | 9,500 |
| **训练集文件** | `data/ready_to_train/train_10k_final.jsonl` |
| **类型分布** | Math: 33.3%, Code: 33.4%, QA: 33.2% |

### 1.2 类型分布

| 类型 | 样本数 | 占比 |
|------|--------|------|
| Code | 3,176 | 33.4% |
| Math | 3,166 | 33.3% |
| QA | 3,158 | 33.2% |

---

## 2. 数据源分布

### 2.1 按来源统计

| 数据源 | 样本数 | 占比 | 类型 |
|--------|--------|------|------|
| math | 1,956 | 20.6% | Math |
| hotpotqa | 1,662 | 17.5% | QA |
| squad_v2 | 1,496 | 15.7% | QA |
| gsm8k | 1,210 | 12.7% | Math |
| bigcodebench | 1,089 | 11.5% | Code |
| code_exercises | 809 | 8.5% | Code |
| mbpp | 762 | 8.0% | Code |
| mbppplus | 364 | 3.8% | Code |
| humanevalplus | 151 | 1.6% | Code |
| humaneval | 1 | 0.0% | Code |

### 2.2 来源说明

**Math类 (3,166样本)**:
- `math`: MATH竞赛数学题，包含完整LaTeX解答
- `gsm8k`: Grade School Math，小学应用题，答案格式为 `#### <number>`

**Code类 (3,176样本)**:
- `bigcodebench`: HuggingFace BigCodeBench，有测试用例
- `mbpp`: Mostly Basic Python Problems
- `mbppplus`: MBPP+增强版，有测试用例无ground_truth
- `humanevalplus`: HumanEval+增强版
- `code_exercises`: 代码练习题，无测试用例，使用LLM Judge

**QA类 (3,158样本)**:
- `hotpotqa`: 多跳推理问答
- `squad_v2`: SQuAD阅读理解

---

## 3. 数据格式规范

### 3.1 通用字段

```json
{
  "problem": "问题描述",
  "problem_type": "math|code|qa",
  "source": "数据来源标识",
  "ground_truth": "标准答案",
  "difficulty": "easy|medium|hard (可选)",
  "meta": {
    "test": "测试用例 (Code)",
    "entry_point": "入口函数名 (Code)",
    "use_llm_judge": true/false,
    "short_answer": "简化答案 (Math)"
  }
}
```

### 3.2 各类型示例

**Math示例 (gsm8k)**:
```json
{
  "problem": "Natalia sold clips to 48 of her friends in April...",
  "problem_type": "math",
  "source": "gsm8k",
  "ground_truth": "Natalia sold 48/2 = <<48/2=24>>24...\n#### 72"
}
```

**Code示例 (bigcodebench)**:
```json
{
  "problem": "def triangle_area(a, h):\n    \"\"\"Given length of a side and height, return the area of a triangle.\"\"\"",
  "problem_type": "code",
  "source": "bigcodebench",
  "ground_truth": "return 0.5 * a * h",
  "meta": {
    "test": "assert triangle_area(5, 3) == 7.5",
    "entry_point": "triangle_area"
  }
}
```

**QA示例 (hotpotqa)**:
```json
{
  "problem": "In what year did the historical film drama...",
  "problem_type": "qa",
  "source": "hotpotqa",
  "ground_truth": "1982"
}
```

---

## 4. Code样本评估路径

### 4.1 评估方式统计

| 评估方式 | 样本数 | 占比 |
|----------|--------|------|
| 测试执行 (test_execution) | 2,367 | 74.5% |
| LLM Judge | 821 | 25.9% |

### 4.2 字段完整性

| 字段 | 有值样本数 | 占比 |
|------|------------|------|
| test用例 | 2,367 | 74.5% |
| entry_point | 3,162 | 99.6% |

### 4.3 评估逻辑

```
有test + 有entry_point → 测试执行 (精确评估)
有test + 无entry_point → 从test中提取entry_point → 测试执行
无test → LLM Judge (语义比较)
```

---

## 5. 特殊处理说明

### 5.1 HumanEval格式处理 (P6修复)

HumanEval系列数据集格式特殊:
- `problem`: 包含函数签名 + docstring
- `ground_truth`: 只包含函数体 (不含def行)

**处理方式**: 在`reward_computer.py`中自动合并:
```python
# 检测并合并函数签名与函数体
if not has_def_in_solution and has_def_in_problem:
    # 从problem提取签名，与solution合并
    solution = func_signature + '\n' + indented_body
```

### 5.2 MBPP+无ground_truth处理

MBPP+ (364样本) 没有ground_truth，但有测试用例:
- 评估方式: 测试执行
- 奖励计算: 根据测试通过率给分

### 5.3 无测试用例代码处理 (P5修复)

code_exercises等数据集无测试用例:
- 评估方式: LLM Judge语义比较
- Judge配置: `config/judge_prompts.yaml` → `code_llm_judge`

---

## 6. 奖励计算规则

### 6.1 5档细粒度奖励

| 等级 | 奖励值 | 判定条件 |
|------|--------|----------|
| 完美 | 1.0 | 精确匹配/测试全通过/LLM判定等价 |
| 接近 | 0.7 | 包含关系/F1≥0.7/>80%测试通过 |
| 部分 | 0.4 | F1≥0.4/>50%测试通过 |
| 格式 | 0.2 | 有输出但错误/>20%测试通过 |
| 错误 | 0.0 | 无输出/完全错误 |

### 6.2 各类型评估策略

**Math**:
1. 提取答案 (`\boxed{}`, `####`)
2. 数值等价比较 (允许误差0.1%)
3. LLM Judge语义比较

**Code**:
1. 代码提取和清理 (移除markdown块)
2. P6: HumanEval格式签名合并
3. 测试用例执行 (multiprocessing隔离) 或 LLM Judge
4. 根据通过率/语义等价给分

**QA**:
1. 精确匹配 (标准化后)
2. 数值等价检查
3. 包含关系检查 (词级别)
4. F1评分 (SQuAD标准)
5. LLM Judge语义比较

---

## 7. 配置说明

### 7.1 训练配置 (training.yaml)

```yaml
# 数据集路径
train_dataset: "11/integrated_aflow_roll/data/ready_to_train/train_10k_final.jsonl"

# 混合采样比例
domain_ratios:
  math: 0.333
  code: 0.333
  qa: 0.334
```

### 7.2 Judge配置 (judge_prompts.yaml)

- `gsm8k`: GSM8K专属，处理`####`格式
- `math`: MATH竞赛题，处理LaTeX
- `code_llm_judge`: 代码语义比较
- `hotpotqa`: 多跳推理QA
- `squad_v2`: 阅读理解，支持上下位词

---

## 8. 数据质量检查清单

- [x] 所有样本有`problem`字段
- [x] 所有样本有`problem_type`字段
- [x] 所有样本有`source`字段
- [x] 96.2%样本有`ground_truth` (3.8% mbppplus用test执行)
- [x] Code样本99.6%有`entry_point`
- [x] 类型分布均匀 (33%±0.4%)
- [x] 无重复样本
- [x] 所有Code样本有评估路径 (test或LLM Judge)

---

## 9. 更新记录

### P7修复 (2025-01-XX) - AFlow评估一致性
**目标**: 确保评估方法与AFlow完全一致

**对比分析**:
| 项目 | AFlow | 修复前 | 修复后 |
|------|-------|--------|--------|
| Math容差(MATH) | `abs_tol=1e-3` | `rel_tol=1e-6` | `abs_tol=1e-3` ✅ |
| Math容差(GSM8K) | `abs_tol=1e-6` | `rel_tol=1e-6` | `abs_tol=1e-6` ✅ |
| Code超时 | 15秒 | 10秒 | 15秒 ✅ |
| Code Sanitize | AST解析+依赖分析 | 简单移除markdown | AST解析+依赖分析 ✅ |
| HumanEval辅助函数 | encode_cyclic等 | 无 | 已添加 ✅ |
| QA标准化 | 移除冠词/标点/小写 | 相同 | ✅ |
| F1计算 | SQuAD标准 | SQuAD标准 | ✅ |

**代码修改**:
1. `_compute_math_reward()`: 根据source使用不同容差
2. `_execute_code_isolated()`: 超时从10秒改为15秒
3. 新增`_sanitize_code()`: AST解析代码清理
4. 新增`_get_dependencies()`: 获取AST节点依赖
5. 新增`_find_reachable()`: BFS查找可达定义
6. 新增`HUMANEVAL_HELPERS`: decode_cyclic/decode_shift/find_zero辅助函数

### P6修复 (2025-01-XX)
- HumanEval格式支持：自动合并函数签名与函数体
- data_manager支持配置数据集路径
- 默认训练集更新为`train_10k_final.jsonl`

### P5修复
- 无测试用例代码支持LLM Judge
- 添加`code_llm_judge`配置

### P4修复
- SQuAD词级别包含检查
- 上下位词支持

---

**文档路径**: `docs/DATASET_SUMMARY.md`
