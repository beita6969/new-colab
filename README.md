# AFlow + ROLL 深度融合项目

**基于强化学习的工作流自动优化系统**

## 🎯 项目概述

本项目将三个先进框架深度融合，实现基于强化学习的工作流自动优化：

- **AFlow (FoundationAgents)**: Workflow框架，提供10个算子和提示词系统
- **ROLL (Alibaba)**: 强化学习框架，使用GRPO算法微调Qwen2.5-7B
- **AgentFlow (lupantech)**: 参考的模块化Agent架构

### 核心创新

用**强化学习驱动的Qwen2.5-7B模型**替换AFlow中的API调用和随机算法，实现：
- ✅ **在线学习**（Online Learning）- 实时优化工作流
- ✅ **自适应提示词生成** - RL模型学习优化提示词
- ✅ **智能算子调用控制** - 模型决定算子序列和边的关系
- ✅ **迭代升级** - 通过强化学习不断改进

---

## 📁 项目结构

```
integrated_aflow_roll/
├── src/
│   ├── gpu_manager.py              # GPU管理（保护PID 3819483，使用GPU 2-3）
│   ├── data_manager.py             # 混合数据集管理（math 40%, code 30%, qa 30%）
│   ├── rl_workflow_generator.py    # RL模型生成工作流（Qwen2.5-7B + LoRA）
│   ├── aflow_executor.py           # AFlow执行引擎适配（使用gpt-4o-mini）
│   ├── reward_computer.py          # 奖励计算（正确性70% + 效率20% + 简洁性10%）
│   └── grpo_trainer.py             # GRPO训练器（在线学习）
├── config/
│   ├── training.yaml               # 训练配置
│   └── aflow_llm.yaml             # AFlow LLM配置（OpenAI API）
├── data/
│   ├── train/mixed_dataset.jsonl  # 训练集（80样本）
│   ├── val/mixed_dataset.jsonl    # 验证集（10样本）
│   └── test/mixed_dataset.jsonl   # 测试集（10样本）
├── checkpoints/                    # 模型检查点
├── logs/                          # 训练日志
├── scripts/
│   ├── create_sample_data.py      # 创建示例数据
│   └── prepare_data.py            # 数据准备（从AFlow/ROLL提取）
├── train.py                        # 训练入口
├── inference.py                    # 推理测试
├── test_integration.py             # 集成测试
└── README.md                       # 本文件
```

---

## ✅ 已完成功能

### 1. GPU管理 ✓
- **自动清理GPU 2-3**（保护代理进程PID 3819483）
- **环境验证**（检查GPU可用性和受保护进程）
- **CUDA设备隔离**（仅使用指定GPU）

### 2. 数据管理 ✓
- **混合数据集**（数学、代码、QA三种类型）
- **按比例采样**（数学40%、代码30%、QA30%）
- **在线循环采样**（无限迭代，自动重新打乱）

### 3. RL工作流生成器 ✓
- **Qwen2.5-7B + LoRA**（仅训练1%参数）
- **直接生成Python代码**（完整Workflow类）
- **语法验证**（ast.parse检查）
- **默认工作流回退**（生成失败时使用）

### 4. AFlow执行适配器 ✓
- **无缝集成AFlow算子**（使用现成代码）
- **动态工作流加载**（从Python代码创建类）
- **超时保护**（默认300秒）
- **gpt-4o-mini执行**（使用OpenAI API）

### 5. 奖励计算器 ✓
- **多维度奖励**：
  - 正确性（70%）：数学/代码/QA不同策略
  - 效率（20%）：基于API成本
  - 简洁性（10%）：基于执行时间和算子数
- **奖励归一化**：[-10, 10]范围

### 6. GRPO训练器 ✓
- **在线学习模式**：ppo_epochs=1, no replay buffer
- **GRPO算法**：组相对优势，降低方差
- **梯度累积**：支持大batch训练
- **KL正则化**：防止策略偏离
- **检查点保存**：定期保存LoRA权重

---

## 🚀 快速开始

### 前置条件

```bash
# 1. 确保GPU 2-3可用
nvidia-smi

# 2. 代理进程正在运行
ps -p 3819483

# 3. 安装依赖（如果还没安装）
cd /home/yijia/.claude/11/ROLL && pip install -e .
cd /home/yijia/.claude/11/AFlow && pip install -r requirements.txt
pip install transformers accelerate peft deepspeed torch
```

### 测试系统

```bash
cd /home/yijia/.claude/11/integrated_aflow_roll

# 运行集成测试
python3 test_integration.py
```

### 启动训练

```bash
# 方法1：使用环境变量
CUDA_VISIBLE_DEVICES=2,3 python3 train.py

# 方法2：自动GPU管理（推荐）
python3 train.py --config config/training.yaml
```

### 监控训练

```bash
# 查看GPU使用情况
watch -n 1 nvidia-smi

# 查看训练日志
tail -f logs/training.log

# 监控GPU 2-3
python3 src/gpu_manager.py --gpus 2 3 --monitor
```

### 测试模型

```bash
# 使用训练好的检查点
python3 inference.py \
    --checkpoint checkpoints/step_50 \
    --problem "What is 15 + 27?" \
    --problem-type math \
    --ground-truth "42"
```

---

## 📊 配置说明

### 训练配置 (`config/training.yaml`)

```yaml
# 核心参数
max_steps: 500                          # 总训练步数
rollout_batch_size: 8                   # 每批问题数
num_return_sequences_in_group: 8        # GRPO组大小
learning_rate: 1.0e-5                   # 学习率

# GPU配置
device_mapping: [2, 3]                  # 仅使用GPU 2-3
protected_pids: [3819483]               # 受保护的进程

# 数据集混合比例
domain_ratios:
  math: 0.4                             # 40%数学
  code: 0.3                             # 30%代码
  qa: 0.3                               # 30%QA

# LoRA配置
lora_rank: 32
lora_alpha: 32
lora_target_modules: "q_proj,k_proj,v_proj,o_proj"
```

### AFlow LLM配置 (`config/aflow_llm.yaml`)

```yaml
models:
  "gpt-4o-mini":
    api_type: "openai"
    base_url: "https://api.openai.com/v1"
    api_key: "sk-proj-..."  # 你的API密钥
```

---

## 🔬 技术细节

### GRPO训练流程

```
1. 采样Batch（混合数据集）
   ├─ 数学问题 40%
   ├─ 代码问题 30%
   └─ QA问题 30%

2. 为每个问题生成8个工作流（GRPO组）
   ├─ RL模型生成Python代码
   ├─ 语法验证
   └─ 记录log概率

3. 执行工作流
   ├─ AFlow引擎执行
   ├─ gpt-4o-mini调用算子
   └─ 返回答案和成本

4. 计算奖励
   ├─ 正确性评估
   ├─ 效率评估（成本）
   └─ 简洁性评估（时间）

5. GRPO优势计算
   ├─ 组内奖励归一化
   └─ Advantage = reward - group_mean

6. 策略更新
   ├─ 计算新log概率
   ├─ 重要性采样比
   ├─ PPO裁剪损失
   ├─ KL正则化
   └─ 梯度下降

7. 重复步骤1-6（在线学习，无replay）
```

### 关键算法

**GRPO组相对优势**：
```python
# 为每个问题生成K=8个工作流
rewards = [r1, r2, r3, r4, r5, r6, r7, r8]

# 组内归一化
mean_reward = np.mean(rewards)
advantages = [r - mean_reward for r in rewards]

# 降低方差，更稳定的训练
```

**PPO裁剪损失**：
```python
ratio = exp(new_log_prob - old_log_prob)
clipped_ratio = clip(ratio, 1-ε, 1+ε)  # ε=0.2
loss = -min(ratio * advantage, clipped_ratio * advantage)
```

---

## 📈 预期效果

- **性能提升**：相比固定工作流提升15-20%正确率
- **成本降低**：相比穷举搜索降低50%以上API成本
- **泛化能力**：可适应新问题类型
- **可解释性**：生成的工作流代码可读易调试

---

## 🛡️ 安全保护

### GPU保护机制
- ✅ 自动检测并清理GPU 2-3上的进程
- ✅ 白名单保护代理进程（PID 3819483）
- ✅ 不影响其他GPU（0, 1, 4, 5, 6, 7）
- ✅ 清理前验证进程信息

### 故障恢复
- ✅ 工作流执行超时保护（300秒）
- ✅ 语法错误自动回退到默认工作流
- ✅ API调用失败自动重试
- ✅ 定期保存检查点

---

## 📝 使用示例

### 示例1：数学问题

```bash
python3 inference.py \
    --checkpoint checkpoints/step_100 \
    --problem "Solve for x: 2x + 5 = 15" \
    --problem-type math \
    --ground-truth "5"
```

**预期输出**：
```
生成的工作流代码:
  - 使用AnswerGenerate算子
  - 分步推理求解
  - 返回最终答案

执行结果:
  - 答案: x = 5
  - 成本: $0.002
  - 奖励: 9.5/10
```

### 示例2：代码问题

```bash
python3 inference.py \
    --checkpoint checkpoints/step_100 \
    --problem "Write a function that returns the sum of two numbers" \
    --problem-type code
```

**预期输出**：
```
生成的工作流代码:
  - 使用Programmer算子
  - 自动编写和执行代码
  - 测试验证

执行结果:
  - 代码: def add(a, b): return a + b
  - 测试: 通过
```

---

## 🐛 故障排查

### GPU不可用
```bash
# 检查GPU状态
nvidia-smi

# 检查CUDA环境
echo $CUDA_VISIBLE_DEVICES

# 手动清理GPU 2-3
python3 src/gpu_manager.py --gpus 2 3 --force-clean
```

### 代理进程问题
```bash
# 检查进程是否运行
ps -p 3819483

# 如果进程不存在，更新配置
vim config/training.yaml
# 修改 protected_pids: []
```

### 数据集问题
```bash
# 重新创建示例数据
python3 scripts/create_sample_data.py

# 验证数据
python3 src/data_manager.py
```

### 训练失败
```bash
# 检查日志
cat logs/training.log

# 降低batch size
vim config/training.yaml
# rollout_batch_size: 4

# 使用更小的模型测试
# base_model: "Qwen/Qwen2.5-1.5B-Instruct"
```

---

## 📚 参考文档

- AFlow框架：`/home/yijia/.claude/11/AFlow/`
- ROLL框架：`/home/yijia/.claude/11/ROLL/`
- AgentFlow框架：`/home/yijia/.claude/11/AgentFlow/`
- 详细设计文档：`/home/yijia/.claude/11/claude.md`

---

## 🎉 项目状态

✅ **所有核心功能已实现并测试通过**

- [x] GPU管理和清理
- [x] 混合数据集管理
- [x] RL工作流生成（Qwen2.5-7B + LoRA）
- [x] AFlow执行适配
- [x] 奖励计算
- [x] GRPO训练器
- [x] 在线学习模式
- [x] 集成测试

**系统准备就绪，可以开始训练！** 🚀
