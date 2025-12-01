# AFlow + GRPO 训练环境配置指南

## 项目简介

本项目实现了 **AFlow + ROLL GRPO** 训练框架，用于训练智能体生成工作流（Workflow）。

- **基础模型**: Qwen2.5-7B-Instruct
- **训练算法**: GRPO (Group Relative Policy Optimization) + WA-GRPO (Workflow-Aware)
- **LLM Judge**: OpenAI gpt-4o-mini (用于评估工作流执行结果)
- **微调方法**: LoRA (Low-Rank Adaptation)

---

## Google Colab 环境配置

### 1. 运行时设置

在 Colab 中选择:
- **运行时类型**: Python 3
- **硬件加速器**: GPU
- **GPU 类型**: A100 (推荐) 或 V100/T4

### 2. 检查 GPU 和 CUDA

```python
# 检查 GPU
!nvidia-smi

# 检查 CUDA 版本
!nvcc --version
```

预期输出:
- GPU: NVIDIA A100-SXM4-40GB (或其他可用 GPU)
- CUDA: 12.x

### 3. 克隆仓库

```bash
# 克隆项目
!git clone https://github.com/beita6969/colab.git
%cd colab
```

### 4. 安装依赖

```bash
# 安装 Python 依赖
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install transformers>=4.40.0 accelerate>=0.27.0 peft>=0.10.0
!pip install bitsandbytes>=0.42.0 scipy safetensors
!pip install openai httpx pyyaml tqdm wandb
!pip install datasets sentencepiece tiktoken
```

### 5. 设置环境变量

```python
import os

# 设置 OpenAI API Key (必需)
os.environ['OPENAI_API_KEY'] = 'your-openai-api-key-here'

# 设置 CUDA 库路径
os.environ['LD_LIBRARY_PATH'] = '/usr/lib64-nvidia:/usr/local/cuda/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')

# 可选: 设置 WandB API Key (用于监控)
os.environ['WANDB_API_KEY'] = 'your-wandb-api-key-here'
```

### 6. 下载模型 (可选，首次运行会自动下载)

```python
from huggingface_hub import snapshot_download

# 下载 Qwen2.5-7B-Instruct
snapshot_download(
    repo_id='Qwen/Qwen2.5-7B-Instruct',
    local_dir='/root/.cache/huggingface/Qwen2.5-7B-Instruct',
    local_dir_use_symlinks=False,
    resume_download=True
)
```

---

## 开始训练

### 快速启动

```bash
# 设置环境变量并启动训练
export LD_LIBRARY_PATH=/usr/lib64-nvidia:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export OPENAI_API_KEY='your-openai-api-key-here'
export PYTHONUNBUFFERED=1

python3 train.py --config config/training.yaml 2>&1 | tee training.log
```

### 使用 Notebook 启动

```python
!export LD_LIBRARY_PATH=/usr/lib64-nvidia:/usr/local/cuda/lib64:$LD_LIBRARY_PATH && \
 export OPENAI_API_KEY='your-openai-api-key-here' && \
 python3 train.py --config config/training.yaml
```

---

## 配置说明

### 训练配置 (`config/training.yaml`)

| 参数 | 值 | 说明 |
|------|-----|------|
| `num_return_sequences_in_group` | 2 | GRPO K值，每个样本生成2个序列 |
| `rollout_batch_size` | 5 | 批大小，每步处理5个样本 |
| `learning_rate` | 2e-5 | 学习率 |
| `lora_rank` | 64 | LoRA 秩 |
| `lora_alpha` | 64 | LoRA alpha |
| `max_steps` | 500 | 最大训练步数 |
| `warmup_steps` | 100 | 预热步数 |

### 关键文件结构

```
.
├── train.py                 # 训练入口
├── config/
│   ├── training.yaml        # 训练配置
│   ├── aflow_llm.yaml       # LLM 配置
│   └── operator.json        # 算子描述
├── src/
│   ├── grpo_trainer.py      # GRPO 训练器
│   ├── aflow_executor.py    # AFlow 执行器
│   ├── reward_computer.py   # 奖励计算
│   └── ...
├── scripts/
│   ├── async_llm.py         # 异步 LLM 客户端
│   ├── operators.py         # 工作流算子
│   └── evaluator.py         # 评估器
└── data/
    └── ready_to_train/      # 训练数据
```

---

## 常见问题

### Q1: CUDA 库找不到

```bash
# 解决方案: 设置 LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib64-nvidia:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Q2: OpenAI API 错误

确保设置了正确的 API Key:
```python
import os
os.environ['OPENAI_API_KEY'] = 'sk-your-actual-key'
```

### Q3: 显存不足 (OOM)

修改 `config/training.yaml`:
```yaml
rollout_batch_size: 3      # 减小批大小
gradient_accumulation_steps: 8  # 增加累积步数
gradient_checkpointing: true    # 启用梯度检查点
```

### Q4: 模型下载慢

使用镜像:
```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

---

## 恢复训练

如果训练中断，可以从 checkpoint 恢复:

```bash
# 查看已保存的 checkpoints
ls checkpoints/

# 恢复训练 (修改 config 中的 resume_from 参数)
# 或直接加载最新 checkpoint
python3 train.py --config config/training.yaml --resume checkpoints/step_50
```

---

## 监控训练

### 使用 WandB

1. 在 `config/training.yaml` 中启用:
```yaml
wandb:
  enabled: true
  project: "agent-prompt"
  api_key: "your-wandb-api-key"
```

2. 访问 [wandb.ai](https://wandb.ai) 查看训练曲线

### 本地日志

```bash
# 实时查看日志
tail -f training.log

# 查看最近的训练指标
grep "Step" training.log | tail -20
```

---

## 环境要求总结

| 组件 | 要求 |
|------|------|
| Python | 3.10+ |
| CUDA | 12.x |
| GPU 显存 | 40GB+ (A100) 或 16GB+ (V100/T4 需调参) |
| PyTorch | 2.0+ |
| transformers | 4.40+ |
| peft | 0.10+ |

---

## 一键启动脚本 (Colab)

将以下代码复制到 Colab 单元格中运行:

```python
# === AFlow + GRPO 一键启动 ===

# 1. 克隆仓库
!git clone https://github.com/beita6969/colab.git
%cd colab

# 2. 安装依赖
!pip install -q torch transformers accelerate peft bitsandbytes
!pip install -q openai httpx pyyaml tqdm wandb datasets

# 3. 设置环境变量
import os
os.environ['OPENAI_API_KEY'] = 'your-openai-api-key'  # 替换为你的 key
os.environ['LD_LIBRARY_PATH'] = '/usr/lib64-nvidia:/usr/local/cuda/lib64'

# 4. 启动训练
!python3 train.py --config config/training.yaml
```

---

## 联系方式

如有问题，请提交 Issue 或联系维护者。
