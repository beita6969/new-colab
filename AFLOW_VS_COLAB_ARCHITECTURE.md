# AFlow vs Colab 深度架构对比文档

> 此文档为修改代码前的详细技术分析，覆盖两个项目的核心架构差异、设计模式和实现细节。

---

## 1. 项目定位与目标对比

| 维度 | AFlow | Colab |
|------|-------|-------|
| **核心目标** | LLM驱动的自动化工作流优化 | 用GRPO训练LLM生成工作流代码 |
| **优化方法** | 进化搜索（LLM生成变异） | 强化学习（策略梯度） |
| **输入** | 数据集 + 初始工作流模板 | 问题 + 训练数据 |
| **输出** | 优化后的工作流代码（Python文件） | 训练好的LoRA权重 |
| **运行模式** | 离线优化（多轮迭代） | 在线学习（实时采样+更新） |
| **LLM角色** | 优化器（生成变异）+ 执行器（运行算子） | 策略模型（生成工作流）+ 执行器（运行算子） |

---

## 2. 核心架构对比

### 2.1 AFlow 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         AFlow 架构                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  run.py                                                         │
│     │                                                           │
│     ▼                                                           │
│  Optimizer (scripts/optimizer.py)                               │
│     │                                                           │
│     ├──► GraphUtils: 工作流代码管理                              │
│     │      ├── create_round_directory()                         │
│     │      ├── load_graph() - 动态import Workflow类              │
│     │      ├── write_graph_files() - 写入优化后的代码            │
│     │      └── create_graph_optimize_prompt()                   │
│     │                                                           │
│     ├──► ExperienceUtils: 经验管理                               │
│     │      ├── check_modification() - 检查修改是否重复           │
│     │      └── format_experience() - 格式化历史经验              │
│     │                                                           │
│     ├──► EvaluationUtils: 评估                                   │
│     │      └── evaluate_graph() - 在验证集上评估                 │
│     │                                                           │
│     └──► AsyncLLM: 优化LLM                                       │
│            └── call_with_format() - 生成优化后的Graph代码        │
│                                                                  │
│  workspace/{DATASET}/workflows/round_{N}/                       │
│     ├── graph.py  - Workflow类定义                               │
│     └── prompt.py - 自定义提示词                                 │
│                                                                  │
│  benchmarks/{dataset}.py                                        │
│     └── 数据集特定的评估逻辑                                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Colab 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         Colab 架构                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  train.py                                                       │
│     │                                                           │
│     ▼                                                           │
│  GRPOTrainer (src/grpo_trainer.py)                              │
│     │                                                           │
│     ├──► DataManager: 混合数据集采样                             │
│     │      ├── sample_batch() - 按比例采样math/code/qa           │
│     │      └── domain_ratios: {math: 0.333, code: 0.333, qa: 0.334}│
│     │                                                           │
│     ├──► VLLMWorkflowGenerator: 工作流生成                       │
│     │      ├── Qwen2.5-7B + LoRA 生成代码                        │
│     │      ├── _build_generation_prompt() - 构建详细提示词       │
│     │      └── generate_workflows_batch() - 批量GPU推理          │
│     │                                                           │
│     ├──► AFlowExecutor: 执行工作流                               │
│     │      ├── _create_workflow_class() - 动态编译               │
│     │      ├── execute_workflow() - 带超时执行                   │
│     │      └── _execute_fallback_workflow() - 失败处理           │
│     │                                                           │
│     ├──► RewardComputer: 5层奖励计算                             │
│     │      ├── _compute_math_reward()                            │
│     │      ├── _compute_code_reward()                            │
│     │      ├── _compute_qa_reward()                              │
│     │      └── _llm_judge_equivalence() - 语义判断               │
│     │                                                           │
│     ├──► WAGRPOAdvantageComputer: 优势计算                       │
│     │      ├── compute_advantages() - 组归一化                   │
│     │      └── _compute_tie_breaker() - 打破同分困境             │
│     │                                                           │
│     ├──► ExperienceBuffer: 高质量样本缓存                        │
│     │      └── add_sample() / retrieve_top_k()                   │
│     │                                                           │
│     └──► Optimizer + Scheduler: 策略更新                         │
│            ├── AdamW + Cosine Schedule                           │
│            └── PPO-style clipped loss + KL regularization        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 核心组件详细对比

### 3.1 Operators（算子）

#### AFlow: `scripts/operators.py` (411行)

| 算子 | 接口 | 功能 |
|------|------|------|
| `Custom` | `__call__(input, instruction)` | 通用LLM调用 |
| `AnswerGenerate` | `__call__(input)` | 步骤推理，返回thought+answer |
| `CustomCodeGenerate` | `__call__(problem, entry_point, instruction)` | 代码生成（带函数名约束） |
| `Programmer` | `__call__(problem, analysis)` | 代码生成+执行，带3次重试 |
| `Test` | `__call__(problem, solution, entry_point, test_loop)` | 测试代码，带反思修正 |
| `Review` | `__call__(problem, solution)` | 审查解决方案 |
| `Revise` | `__call__(problem, solution, feedback)` | 根据反馈修订 |
| `ScEnsemble` | `__call__(solutions, problem)` | 自洽性集成（选择最一致答案） |
| `MdEnsemble` | `__call__(solutions, problem)` | 多数投票（多轮打乱+投票） |
| `Format` | `__call__(problem, solution)` | 格式化提取答案 |

**特点**:
- 使用 `_fill_node()` 统一调用LLM
- 支持 XML/Code/Text 三种Formatter
- Programmer有进程池执行代码
- 使用tenacity重试机制

#### Colab: `scripts/operators.py` (465行)

| 算子 | 接口 | 功能 |
|------|------|------|
| `Custom` | `__call__(input, instruction)` | 通用LLM调用 |
| `AnswerGenerate` | `__call__(input)` | 步骤推理 |
| `Programmer` | `__call__(problem, analysis)` | 代码生成+执行 |
| `Test` | `__call__(problem, solution, entry_point)` | 简单代码测试 |
| `Review` | `__call__(problem, solution)` | 审查 |
| `Revise` | `__call__(problem, solution, feedback)` | 修订 |
| `ScEnsemble` | `__call__(solutions, problem)` | 自洽性集成 |
| `MdEnsemble` | `__call__(solutions, problem)` | 多数投票 |
| `Decompose` | `__call__(problem)` | **新增** 问题分解 |
| `Verify` | `__call__(problem, answer)` | **新增** 答案验证 |

**特点**:
- 简化版实现（无Formatter抽象）
- 直接调用 `self.llm(prompt)`
- 使用subprocess执行代码（无进程池）
- 新增 Decompose 和 Verify 算子

### 3.2 AsyncLLM

#### AFlow: `scripts/async_llm.py` (280行)

```python
class AsyncLLM:
    def __init__(self, config, system_msg=None):
        self.config = config  # LLMConfig对象
        self.aclient = AsyncOpenAI(...)
        self.usage_tracker = TokenUsageTracker()

    async def __call__(self, prompt) -> str:
        # 直接调用OpenAI API
        response = await self.aclient.chat.completions.create(...)
        return response.choices[0].message.content

    async def call_with_format(self, prompt, formatter) -> dict:
        # 带格式化器的调用（XML/Code/Text）
        formatted_prompt = formatter.prepare_prompt(prompt)
        response = await self.__call__(formatted_prompt)
        is_valid, parsed = formatter.validate_response(response)
        return parsed
```

**特点**:
- 完整的Token使用追踪和成本计算
- Formatter抽象（XmlFormatter, CodeFormatter, TextFormatter）
- 从YAML配置文件加载多个模型配置
- 单例模式的LLMsConfig

#### Colab: `scripts/async_llm.py` (229行)

```python
class AsyncLLM:
    def __init__(self, api_key, base_url, model, temperature, ...):
        self.client = AsyncOpenAI(...)

    async def __call__(self, prompt, system_prompt=None, ...) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = await self.client.chat.completions.create(...)
        return response.choices[0].message.content

    async def batch_call(self, prompts, max_concurrent=10) -> List[tuple]:
        # 批量调用（带信号量限流）
```

**特点**:
- 简化版实现（无Formatter）
- 支持system_prompt参数
- 支持batch_call批量调用
- 简化的Token统计（无成本计算）

### 3.3 工作流执行

#### AFlow: `benchmarks/benchmark.py` + 各数据集类

```python
class BaseBenchmark:
    async def run_evaluation(self, agent, va_list, ...):
        data = await self.load_data(va_list)
        results = await self.evaluate_all_problems(data, agent, ...)
        return average_score, average_cost, total_cost

    async def evaluate_problem(self, data, graph) -> Tuple:
        # 抽象方法，由子类实现
        pass
```

**执行流程**:
1. Optimizer动态import `workspace/{DATASET}/workflows/round_{N}/graph.py`
2. 直接实例化Workflow类并调用
3. 在Benchmark类中评估结果

#### Colab: `src/aflow_executor.py` (1097行)

```python
class AFlowExecutor:
    async def execute_workflow(self, workflow_code, problem, problem_type, **kwargs):
        # 1. 验证工作流代码
        is_valid, msg, _ = self.validator.validate_workflow_code(...)

        # 2. 修复SymPy兼容性问题
        fixed_code, _, _ = self.sympy_fixer.fix_code(workflow_code)

        # 3. 动态创建Workflow类
        workflow_class = self._create_workflow_class(workflow_code, problem_type)

        # 4. 实例化并执行（带超时）
        workflow = workflow_class(name, llm_config, dataset)
        result = await asyncio.wait_for(workflow(problem), timeout=self.timeout)

        return answer, cost, metadata

    def _create_workflow_class(self, workflow_code, problem_type):
        # 动态编译代码为Python类
        # 包含大量修复逻辑：
        # - 过滤禁止的import
        # - 修复typo (self.lllm → self.llm)
        # - 修复顶层await
        # - 清理无效类型注解
        # - 自动初始化变量防止UnboundLocalError
        namespace = {"operator": operator_module, ...}
        exec(modified_code, namespace)
        return namespace["Workflow"]
```

**特点**:
- WorkflowValidator验证代码有效性
- SymPyCodeFixer修复兼容性问题
- ResponseStandardizer标准化输出
- 大量自动修复逻辑（P0-P14修复）
- Fallback机制（执行失败返回错误信息供学习）

---

## 4. 关键差异详解

### 4.1 优化方法

#### AFlow: 进化搜索

```
Round 1 → Round 2 → Round 3 → ... → Round N
   │         │         │
   ▼         ▼         ▼
初始工作流  变异1     变异2    ...   最优工作流
```

- 使用LLM（Claude-3.5）生成工作流变异
- 每轮在验证集上评估
- 选择高分工作流作为下一轮基础
- 收敛检测（top-3得分稳定）

#### Colab: 强化学习（GRPO）

```
  ┌─────────────────────────────────────────┐
  │               GRPO训练循环              │
  │  ┌─────┐   ┌─────────┐   ┌─────────┐   │
  │  │采样 │──►│生成K个  │──►│执行+奖励│   │
  │  │问题 │   │工作流   │   │计算     │   │
  │  └─────┘   └─────────┘   └────┬────┘   │
  │                                │        │
  │  ┌─────────┐   ┌─────────────────┐     │
  │  │策略更新 │◄──│WA-GRPO优势计算  │     │
  │  │(PPO)   │   │(组归一化+tie-breaker)│  │
  │  └─────────┘   └─────────────────┘     │
  └─────────────────────────────────────────┘
```

- 每步采样新问题（在线学习）
- 为每个问题生成K=2个工作流
- 组内归一化计算优势
- PPO风格策略更新

### 4.2 奖励设计

#### AFlow: 二元奖励

```python
# benchmarks/humaneval.py
result = self.check_solution(prediction, data["test"], data["entry_point"])
score = 1.0 if ret[0] == self.PASS else 0.0  # 只有0或1
```

#### Colab: 5层粒度奖励

```python
# src/reward_computer.py
# Level 5: 精确匹配     → 1.0
# Level 4: LLM Judge等价 → 0.7
# Level 3: 部分正确     → 0.4
# Level 2: 有效尝试     → 0.2
# Level 1: 执行失败     → 0.0
```

### 4.3 数据处理

#### AFlow:
- 每个数据集独立处理
- 固定验证集评估
- 单一问题类型（math OR code OR qa）

#### Colab:
- 混合数据集采样（math:code:qa = 1:1:1）
- 支持HumanEval特殊格式（entry_point, test）
- 支持HotpotQA的context注入
- 自动过滤MBPP数据（质量问题）

### 4.4 模型配置

#### AFlow:
```yaml
# config/config2.yaml
models:
  gpt-4o-mini:
    api_key: ${OPENAI_API_KEY}
    base_url: https://oneapi.deepwisdom.ai/v1
    temperature: 0.7
  claude-3-5-sonnet:
    ...
```

- 优化LLM: Claude-3.5-Sonnet
- 执行LLM: GPT-4o-mini
- 使用外部API

#### Colab:
```yaml
# config/training.yaml
base_model: "Qwen/Qwen2.5-7B-Instruct"
use_lora: true
lora_rank: 64
lora_alpha: 64
lora_target_modules: "q_proj,k_proj,v_proj,o_proj"

aflow_executor_model: "gpt-4o-mini"  # 执行器用OpenAI
```

- 策略模型: Qwen2.5-7B + LoRA（本地）
- 执行器LLM: GPT-4o-mini（API）
- LLM Judge: GPT-4o-mini（API）

---

## 5. 文件结构对比

### 5.1 AFlow

```
AFlow/
├── run.py                          # 主入口
├── run_baseline.py                 # 基线测试
├── config/
│   └── config2.yaml                # LLM配置
├── scripts/
│   ├── optimizer.py                # 优化器核心 (256行)
│   ├── operators.py                # 算子实现 (411行)
│   ├── async_llm.py                # LLM客户端 (280行)
│   ├── formatter.py                # 输出格式化 (242行)
│   ├── evaluator.py                # 评估器 (66行)
│   ├── logs.py                     # 日志
│   ├── workflow.py                 # 工作流基类
│   ├── prompts/
│   │   ├── prompt.py               # 算子提示词
│   │   └── optimize_prompt.py      # 优化提示词
│   ├── optimizer_utils/
│   │   ├── graph_utils.py          # 图操作
│   │   ├── data_utils.py           # 数据操作
│   │   ├── experience_utils.py     # 经验管理
│   │   ├── evaluation_utils.py     # 评估工具
│   │   └── convergence_utils.py    # 收敛检测
│   └── utils/
│       ├── sanitize.py             # 代码消毒
│       ├── code.py                 # 代码工具
│       └── common.py               # 通用工具
├── benchmarks/
│   ├── benchmark.py                # 基类 (117行)
│   ├── humaneval.py                # HumanEval
│   ├── gsm8k.py                    # GSM8K
│   ├── math.py                     # MATH
│   ├── hotpotqa.py                 # HotpotQA
│   └── ...                         # 其他数据集
├── workspace/
│   ├── HumanEval/workflows/
│   │   ├── template/
│   │   │   ├── operator.py         # 数据集特定算子
│   │   │   └── operator.json       # 算子描述
│   │   └── round_1/
│   │       ├── graph.py            # 工作流定义
│   │       └── prompt.py           # 自定义提示词
│   ├── MATH/workflows/...
│   ├── GSM8K/workflows/...
│   └── HotpotQA/workflows/...
└── data/
    └── datasets/                   # 数据集文件
```

### 5.2 Colab

```
colab/
├── train.py                        # 主入口
├── config/
│   ├── training.yaml               # 训练配置 (155行)
│   ├── aflow_llm.yaml              # 执行器LLM配置
│   └── operator.json               # 算子描述
├── src/
│   ├── grpo_trainer.py             # GRPO训练器 (1330行) ⭐核心
│   ├── wa_grpo.py                  # WA-GRPO算法 (750行) ⭐核心
│   ├── reward_computer.py          # 奖励计算 (2000+行) ⭐核心
│   ├── vllm_workflow_generator.py  # 工作流生成 (784行)
│   ├── aflow_executor.py           # 工作流执行 (1097行)
│   ├── data_manager.py             # 数据管理 (368行)
│   ├── experience_buffer.py        # 经验缓冲 (297行)
│   ├── workflow_validator.py       # 工作流验证
│   ├── response_standardizer.py    # 响应标准化
│   ├── sympy_code_fixer.py         # SymPy修复
│   ├── prompt_optimizer.py         # 提示词优化
│   ├── operator_prompt_enhancer.py # 算子增强
│   ├── gpu_manager.py              # GPU管理
│   ├── code_executor.py            # 代码执行
│   ├── answer_extractor.py         # 答案提取
│   ├── judge_prompt_loader.py      # Judge提示词
│   ├── unified_evaluator.py        # 统一评估
│   └── wandb_metrics_collectors.py # 监控
├── scripts/
│   ├── operators.py                # 算子实现 (465行)
│   └── async_llm.py                # LLM客户端 (229行)
└── data/
    ├── ready_to_train/             # 预处理数据
    │   ├── train_10k_final.jsonl
    │   └── test_500_preprocessed.jsonl
    └── experience_buffer/          # 高质量样本缓存
```

---

## 6. 修改建议与注意事项

### 6.1 如果要修改 AFlow

**关键文件**:
1. `scripts/optimizer.py` - 优化循环逻辑
2. `scripts/operators.py` - 算子实现
3. `scripts/prompts/optimize_prompt.py` - 优化提示词
4. `workspace/{DATASET}/workflows/template/operator.json` - 算子描述

**注意事项**:
- 工作流代码会被写入文件系统（`workspace/`）
- 使用动态import加载工作流类
- Formatter抽象层很重要（XML/Code/Text）
- 收敛检测基于top-3得分稳定性

### 6.2 如果要修改 Colab

**关键文件**:
1. `src/grpo_trainer.py` - 训练主循环
2. `src/wa_grpo.py` - 优势计算算法
3. `src/reward_computer.py` - 奖励函数
4. `src/aflow_executor.py` - 执行器
5. `config/training.yaml` - 配置

**注意事项**:
- 大量的P0-P14修复逻辑，修改前要理解原因
- WA-GRPO的tie-breaker逻辑是解决全零优势的关键
- 5层奖励设计是为了提供更丰富的学习信号
- 代码执行使用subprocess隔离，有超时控制
- Fallback机制返回错误信息而非静默失败

### 6.3 共享代码迁移

如果要从AFlow迁移功能到Colab，注意以下差异：

| AFlow | Colab | 说明 |
|-------|-------|------|
| `_fill_node(op_class, prompt, mode)` | `await self.llm(prompt)` | Formatter抽象层 |
| `LLMConfig` + `LLMsConfig` | 简单dict配置 | 配置管理 |
| `ProcessPoolExecutor` | `subprocess.run` | 代码执行隔离 |
| 固定验证集 | 在线采样 | 数据流 |
| 文件系统存储工作流 | 动态编译执行 | 工作流管理 |

---

## 7. 总结

### 7.1 AFlow 优势
- 更成熟的抽象（Formatter、Benchmark基类）
- 更稳定的代码执行（ProcessPool）
- 工作流持久化（可复现）
- 完整的Token/成本追踪

### 7.2 Colab 优势
- 端到端强化学习训练
- 更细粒度的奖励设计
- WA-GRPO解决同分困境
- 丰富的自动修复逻辑
- 实时监控（wandb集成）

### 7.3 两者结合的潜力
- 用AFlow的Formatter抽象改进Colab的算子实现
- 用Colab的5层奖励改进AFlow的评估
- 用AFlow的进化搜索初始化Colab的策略
- 用Colab训练出的模型作为AFlow的执行器

---

*文档生成时间: 2024-12-03*
*作者: Claude Code Analysis*
