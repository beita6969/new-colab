# LLM as Judge - GRPO Training Framework

GRPO (Group Relative Policy Optimization) 训练框架，使用LLM Judge进行奖励评估。

## 架构

- **策略模型**: Qwen 2.5-7B (LoRA微调)
- **执行模型**: GPT OSS 120B @ port 8002
- **评估模型**: GPT OSS 120B (LLM Judge)

## 主要功能

✅ 完整的GRPO训练流程
✅ LLM Judge评分系统 (直接返回True/False)
✅ 动态Workflow生成与执行
✅ 变量作用域自动修复 (100% workflow成功率)
✅ LLM Judge重试机制 (应对空响应)
✅ 混合数据集支持 (Math/Code/QA)

## 核心文件

- `src/reward_computer.py`: LLM Judge only架构
- `src/prompt_optimizer.py`: Workflow生成提示词优化
- `src/grpo_trainer.py`: GRPO训练主流程
- `src/aflow_executor.py`: 动态workflow执行器

## 数据集

由于文件大小限制，大型数据集文件未上传到GitHub。请使用以下方式获取：

### 方法1: 下载脚本

```bash
python scripts/download_all_datasets.py
python scripts/create_mixed_dataset.py
```

### 方法2: 手动下载

已包含的小型数据集:
- `data/mixed/train_mixed.jsonl` (7.6MB) - 优化后的混合训练集
- `data/mixed/val_mixed.jsonl` - 验证集
- `data/humaneval/` - HumanEval代码测试
- `data/gsm8k/` - GSM8K数学推理

未包含的大型文件 (需手动下载):
- `data/train/mixed_dataset.jsonl` (493MB)
- `data/raw/mmlu/auxiliary_train.jsonl` (158MB)
- `data/raw/hotpotqa/dev_distractor.json` (45MB)

数据集来源:
- HumanEval: https://github.com/openai/human-eval
- GSM8K: https://github.com/openai/grade-school-math
- MMLU: https://github.com/hendrycks/test
- HotpotQA: https://hotpotqa.github.io/
- CommonsenseQA: https://www.tau-nlp.org/commonsenseqa

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据集

```bash
# 使用优化后的混合数据集 (已包含)
ls data/mixed/

# 或下载完整数据集
python scripts/download_all_datasets.py
```

### 3. 启动vLLM服务 (GPT OSS 120B)

```bash
bash launch_vllm_safe.sh
```

### 4. 开始训练

```bash
python train.py --config config/training.yaml
```

## 配置

### 训练配置 (`config/training.yaml`)

```yaml
# 模型
strategy_model: "Qwen/Qwen2.5-7B-Instruct"
aflow_executor_model: "gpt-oss-120b"  # vLLM服务

# 数据集
train_dataset: "data/mixed/train_mixed.jsonl"
val_dataset: "data/mixed/val_mixed.jsonl"

# GPU
physical_gpus: [2]
```

### LLM配置 (`config/aflow_llm.yaml`)

需要手动创建（已在.gitignore中排除，防止泄露token）:

```yaml
models:
  "gpt-oss-120b":
    api_type: "openai"
    base_url: "http://localhost:8002/v1"
    api_key: "sk-dummy"
    model_name: "/path/to/gpt-oss-120b"
    temperature: 0
    top_p: 1
    no_proxy: "localhost,127.0.0.1"
```

## 训练监控

```bash
# 实时监控
bash monitor.sh

# 或使用Python脚本
python monitor_training_live.py
```

## 关键修复

### 1. Workflow变量作用域修复

- **问题**: UnboundLocalError导致8%失败率
- **修复**: `src/prompt_optimizer.py:103-120` 添加变量初始化指导
- **效果**: 100% workflow成功率

### 2. LLM Judge架构简化

- **问题**: 答案提取器覆盖LLM Judge结果
- **修复**: `src/reward_computer.py:276-297` 只保留LLM Judge
- **效果**: 准确率从0%提升到48%

### 3. LLM Judge重试机制

- **问题**: GPT OSS 120B偶尔返回空内容（13%频率）
- **修复**: `src/reward_computer.py:174-199` 重试1次+fallback
- **效果**: 成功率从87%提升到预计>95%

详见:
- `FIX_SUMMARY.md`: P0修复总结
- `DIAGNOSIS_REPORT.md`: 问题诊断报告
- `LLM_JUDGE_RETRY_FIX.md`: 重试机制实现

## 当前训练状态

- 进度: Step 4/500
- 准确率: 48% (47/97样本)
- Workflow成功率: 100%
- GPU: GPU 2 (50GB/80GB)

## 项目结构

```
.
├── src/
│   ├── grpo_trainer.py          # GRPO训练器
│   ├── reward_computer.py       # LLM Judge评分
│   ├── prompt_optimizer.py      # 提示词优化
│   ├── aflow_executor.py        # Workflow执行
│   ├── rl_workflow_generator.py # Workflow生成
│   └── data_manager.py          # 数据管理
├── config/
│   ├── training.yaml            # 训练配置
│   └── aflow_llm.yaml.example   # LLM配置示例
├── data/
│   ├── mixed/                   # 优化后的混合数据集 ✅
│   ├── humaneval/              # HumanEval代码测试 ✅
│   ├── gsm8k/                  # GSM8K数学 ✅
│   └── raw/                    # 原始数据（部分需下载）
├── scripts/
│   ├── download_all_datasets.py # 数据集下载
│   └── create_mixed_dataset.py  # 混合数据集创建
└── docs/
    └── *.md                     # 修复和诊断文档
```

## 许可证

MIT

## 致谢

- Qwen Team: Qwen 2.5-7B模型
- vLLM: 高效LLM推理服务
- HuggingFace: Transformers库
