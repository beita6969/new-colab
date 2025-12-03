# AFlow-GRPO: 开放式工作流组合训练系统

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **核心创新**：让模型自主学习如何组合 Operators 来解决问题，而不是从预定义选项中选择

## 🎯 项目理念

```
传统方法: "请选择最佳工作流: A) Custom B) Programmer C) Custom->Review"
本项目方法: "这是可用的Operators，请设计最优工作流 DSL"
```

模型学习生成 DSL (Domain Specific Language) 来组合 Operators，实现真正的**开放式工作流组合**。

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                     AFlow-GRPO 训练系统                      │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐  │
│  │   vLLM      │───>│   DSL       │───>│   Workflow      │  │
│  │  Generator  │    │   Parser    │    │   Executor      │  │
│  └─────────────┘    └─────────────┘    └─────────────────┘  │
│         │                                      │            │
│         v                                      v            │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐  │
│  │   GRPO      │<───│   Reward    │<───│   Evaluator     │  │
│  │   Trainer   │    │   Computer  │    │   (gpt-4o-mini) │  │
│  └─────────────┘    └─────────────┘    └─────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 训练流程

1. **输入问题** → 模型根据问题类型生成 DSL 工作流
2. **DSL 解析** → 转换为可执行的 Python 代码
3. **工作流执行** → 按照 DSL 逻辑执行各个 Operator (通过 OpenAI API)
4. **奖励计算** → 评估答案正确性、效率等
5. **GRPO 更新** → 使用 WA-GRPO 更新模型参数

---

## 🔧 DSL 语法

模型生成的工作流使用 DSL (Domain Specific Language) 表示：

| 语法 | 含义 | 示例 |
|------|------|------|
| `->` | 顺序执行 | `Custom -> Review -> Revise` |
| `[...]` | 并行执行 | `[Custom, Custom, Custom] -> ScEnsemble` |
| `?` | 条件分支 | `Review ? Revise : done` |
| `* n` | 循环执行 | `(Review -> Revise) * 3` |

### 示例工作流

```python
# 数学问题 - 编程验证
"Custom -> Programmer -> Review ? Revise : done"

# 代码生成 - 测试驱动
"CustomCodeGenerate -> Test -> Format"

# 复杂问题 - 多路投票
"[Custom, Custom, Custom] -> ScEnsemble -> Review"

# 迭代优化
"AnswerGenerate -> (Review -> Revise) * 2 -> Format"
```

---

## 🛠️ 可用 Operators

| Operator | 功能 | 输入 → 输出 |
|----------|------|-------------|
| **Custom** | 通用生成 | `(input, instruction)` → `response` |
| **AnswerGenerate** | 思维链推理 | `(input)` → `thought, answer` |
| **Programmer** | 代码执行 | `(problem, analysis)` → `code, output` |
| **CustomCodeGenerate** | 代码生成 | `(problem, entry_point, instruction)` → `code` |
| **Test** | 测试验证 | `(problem, solution, entry_point)` → `result, solution` |
| **Review** | 解答审查 | `(problem, solution)` → `review_result, feedback` |
| **Revise** | 解答修改 | `(problem, solution, feedback)` → `solution` |
| **Format** | 格式化输出 | `(problem, solution)` → `solution` |
| **ScEnsemble** | 自洽集成 | `(solutions, problem)` → `response` |
| **MdEnsemble** | 多数投票 | `(solutions, problem)` → `solution` |

---

## 📦 项目结构

```
.
├── train.py                    # 训练入口
├── config/
│   ├── training.yaml           # 主训练配置
│   ├── operator.json           # Operator 定义
│   └── aflow_llm.yaml          # LLM API 配置
├── src/
│   ├── vllm_workflow_generator.py  # 🔥 核心：工作流生成器 + DSL解析
│   ├── grpo_trainer.py             # GRPO 训练器
│   ├── wa_grpo.py                  # WA-GRPO 优势估计
│   ├── aflow_executor.py           # 工作流执行器
│   ├── reward_computer.py          # 奖励计算
│   └── unified_evaluator.py        # 评估器
├── data/
│   └── ready_to_train/
│       ├── train_10k_final.jsonl   # 训练集 (10K样本)
│       └── test_500_preprocessed.jsonl  # 测试集
└── scripts/                    # 工具脚本
```

---

## 🚀 快速开始

### 环境要求

| 组件 | 最低配置 | 推荐配置 |
|------|---------|----------|
| GPU | V100 16GB | A100 40GB |
| Python | 3.10+ | 3.10.12 |
| CUDA | 12.0+ | 12.6 |

### 1. 克隆仓库

```bash
git clone https://github.com/beita6969/new-colab.git
cd new-colab

# 如果有 LFS 大文件
git lfs pull
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 API Key

```bash
export OPENAI_API_KEY="your-openai-api-key"
export LD_LIBRARY_PATH=/usr/lib64-nvidia:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### 4. 启动训练

```bash
python train.py --config config/training.yaml
```

---

## 🖥️ Google Colab 一键启动

```python
#@title 🚀 AFlow-GRPO 一键启动
OPENAI_API_KEY = "sk-your-api-key"  #@param {type:"string"}

import os

# 检查 GPU
!nvidia-smi --query-gpu=name,memory.total --format=csv

# 克隆仓库
!git clone https://github.com/beita6969/new-colab.git 2>/dev/null || (cd new-colab && git pull)
%cd new-colab
!git lfs pull

# 安装依赖
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
!pip install -q transformers>=4.40.0 accelerate>=0.27.0 peft>=0.10.0
!pip install -q bitsandbytes>=0.42.0 scipy safetensors openai httpx pyyaml tqdm

# 配置环境
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ['LD_LIBRARY_PATH'] = '/usr/lib64-nvidia:/usr/local/cuda/lib64'
os.environ['WANDB_DISABLED'] = 'true'

# 启动训练
!python3 train.py --config config/training.yaml
```

---

## ⚙️ 配置详解

### 主要参数 (`config/training.yaml`)

```yaml
# GRPO 算法配置
num_return_sequences_in_group: 2   # K值: 每个问题生成K个工作流
rollout_batch_size: 5              # B值: 每批处理B个问题
learning_rate: 2.0e-5              # 学习率
kl_loss_coef: 0.005                # KL 散度惩罚系数
clip_range: 0.20                   # PPO 裁剪范围

# LoRA 配置
lora_rank: 64
lora_alpha: 64
lora_target_modules: "q_proj,k_proj,v_proj,o_proj"

# WA-GRPO (Workflow-Aware)
wa_grpo:
  diversity_weight: 0.35           # 工作流多样性权重
  revise_gain_weight: 0.25         # 改进幅度权重
  exec_success_weight: 0.20        # 执行成功率权重

# 温度调度
temperature_schedule:
  enabled: true
  initial: 0.5                     # 早期高温探索
  final: 0.15                      # 后期低温利用
```

### 显存配置建议

| GPU | 显存 | K | B | grad_accum |
|-----|------|---|---|------------|
| T4 | 16GB | 2 | 2 | 8 |
| V100 | 16GB | 2 | 3 | 6 |
| A100 | 40GB | 2 | 5 | 4 |

---

## 📊 奖励系统

**5级奖励**：`[0, 0.2, 0.4, 0.7, 1.0]`

```yaml
reward_weights:
  correctness: 0.65    # 答案正确性
  efficiency: 0.15     # 执行效率
  simplicity: 0.10     # 工作流简洁度
  format: 0.05         # 输出格式
  repetition: 0.05     # 重复惩罚
```

---

## 📂 数据集格式

```json
{
  "question": "问题文本",
  "answer": "标准答案",
  "domain": "math|code|qa",
  "entry_point": "函数名 (仅code)"
}
```

**数据分布**：Math 33.3% / Code 33.3% / QA 33.4%

---

## 🔍 常见问题

### Q: DSL 解析失败？

系统会自动处理常见问题：
- `X ? Y : done` → 自动转换为 `X -> Y`
- `-> done` 后缀 → 自动移除

### Q: OOM (显存不足)？

```yaml
gradient_accumulation_steps: 8     # 增加累积
gradient_checkpointing: true       # 启用检查点
rollout_batch_size: 2              # 减少批次
```

### Q: OpenAI API 超时？

调整 `execution_timeout: 600` 或减少 `num_return_sequences_in_group`

---

## 📈 监控训练

```bash
# 实时日志
tail -f logs/training.log

# 查看关键指标
grep -E "Step|reward|loss" logs/training.log | tail -50
```

---

## 🙏 致谢

- [AFlow](https://github.com/geekan/MetaGPT) - 工作流框架
- [GRPO](https://arxiv.org/abs/2402.03300) - 训练算法
- [Qwen2.5](https://github.com/QwenLM/Qwen2.5) - 基础模型
- [PEFT](https://github.com/huggingface/peft) - LoRA 实现

---

## 📄 License

MIT License

---

**核心创新**：让模型学习 "如何组合工具"，而不是 "选择哪个预设方案"
