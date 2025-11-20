# 当前问题记录

**时间**: 2025-11-16
**训练状态**: Step 1/500 (75%)
**PID**: 2203674

## 🔴 核心问题

### 问题 1: LLM 配置类型错误（间歇性）

**错误信息**:
```
AttributeError: 'dict' object has no attribute 'call_with_format'
```

**错误堆栈**:
```python
File "/home/yijia/.claude/11/integrated_aflow_roll/src/aflow_executor.py", line 154
File "<string>", line 15, in __call__
File "/home/yijia/.claude/11/AFlow/scripts/operators.py", line 61, in _fill_node
    response = await self.llm.call_with_format(prompt, formatter)
AttributeError: 'dict' object has no attribute 'call_with_format'
```

**问题分析**:
- `create_llm_instance()` 接收到了 dict 而不是 LLMConfig 实例
- 导致创建的 llm 对象没有 `call_with_format` 方法
- 已应用的修复（`_load_llm_config` 和 `_get_llm_config`）不够彻底

**已尝试的修复**:
1. ✅ `_load_llm_config()`: 失败时使用 `LLMsConfig.default()`
2. ✅ `_get_llm_config()`: 返回字符串模型名而不是 None
3. ✅ `execute_workflow()`: 添加 None 检查
4. ❌ 仍然间歇性发生

**需要研究**:
- AFlow 中 `create_llm_instance()` 的正确使用方式
- LLMConfig 和 LLMsConfig 的正确初始化流程
- 其他项目如何处理 LLM 配置

---

### 问题 2: Qwen 生成代码的 API 使用错误

**错误信息**:
```
TypeError: AnswerGenerate.__call__() got an unexpected keyword argument 'instruction'
```

**错误堆栈**:
```python
File "<string>", line 15, in __call__
TypeError: AnswerGenerate.__call__() got an unexpected keyword argument 'instruction'
```

**问题分析**:
- Qwen 生成的 Workflow 代码中错误地给 `AnswerGenerate` 传入了 `instruction` 参数
- AnswerGenerate 不接受 instruction 参数
- 说明 Qwen 不清楚每个算子的正确 API

**Qwen 生成的错误示例**:
```python
# 错误：AnswerGenerate 不接受 instruction 参数
result = await self.answer_generate(
    input=problem,
    instruction="Solve step by step"  # ❌ 错误参数
)
```

**正确的 API**:
```python
# AnswerGenerate 只接受 input 参数
result = await self.answer_generate(input=problem)
```

**需要研究**:
- AFlow 中每个算子的正确 API 签名
- 其他项目如何在 prompt 中描述算子用法
- 是否有现成的算子使用示例

---

## 🟡 次要问题

### 问题 3: Qwen 生成格式改善但仍不完美

**进展**:
- ✅ Temperature=0.1 生效，Qwen 现在生成 `class Workflow:` 而不是 `def solve()`
- ✅ 这是重大进步

**仍存在的问题**:
- Qwen 在代码前添加解释文字（"The provided code is..."）
- 代码可能被截断

**当前生成示例**:
```
🔍 DEBUG: Qwen 生成的原始文本:
The provided code is almost complete but lacks the necessary imports...

```python
import workspace.math.workflows.template.operator as operator
from scripts.async_llm import create_llm_instance
from scripts.evaluator import DatasetType

class Workflow:
    def __init__(self, name: str, llm_config, dataset: DatasetType):
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        ...
```

**需要优化**:
- Prompt 设计：更明确地要求只输出代码
- 可能需要使用 Chat template 强制格式

---

## 🔍 需要研究的问题

### 1. AFlow 仓库研究重点

**目录**: `/home/yijia/.claude/11/AFlow`

**需要查看**:
- [ ] `scripts/async_llm.py`:
  - `create_llm_instance()` 的实现
  - `LLMConfig` 和 `LLMsConfig` 的定义
  - 正确的初始化流程

- [ ] `scripts/operators.py`:
  - 每个算子的 `__call__` 方法签名
  - AnswerGenerate, Custom, Programmer 等的正确用法

- [ ] `workspace/*/workflows/`:
  - 现有的 Workflow 实现示例
  - 如何正确使用算子

- [ ] `workspace/*/workflows/template/`:
  - 官方的 Workflow 模板
  - 推荐的实现模式

### 2. AgentFlow 仓库研究重点

**目录**: 需要找到 AgentFlow 的位置

**需要查看**:
- [ ] 如何处理 LLM 配置
- [ ] 如何生成和执行 Workflow
- [ ] 是否有类似的问题和解决方案

---

## 📋 解决方案优先级

### 高优先级（必须解决）

1. **彻底修复 LLM 配置错误**
   - 研究 AFlow 中的正确用法
   - 确保所有情况下 llm_config 都是有效的
   - 可能需要重构 `_get_llm_config()`

2. **修复 Qwen 生成代码的 API 错误**
   - 在 prompt 中明确每个算子的 API 签名
   - 提供正确的使用示例
   - 可能需要添加代码验证步骤

### 中优先级（改善体验）

3. **优化 Qwen 生成格式**
   - 改进 prompt 设计
   - 考虑使用 Chat template
   - 移除不必要的解释文字

### 低优先级（长期优化）

4. **提高生成质量**
   - Few-shot 示例
   - 更详细的算子描述
   - Curriculum learning

---

## 🎯 下一步行动

1. **立即**: 使用 Explore agent 深入研究 AFlow 和 AgentFlow 仓库
2. **然后**: 根据研究结果修复 LLM 配置错误
3. **接着**: 优化 prompt 以修复 API 使用错误
4. **最后**: 重启训练验证修复效果

---

## 📚 参考资料

- AFlow 仓库: `/home/yijia/.claude/11/AFlow`
- 当前系统分析: `docs/system_analysis.md`
- 配置文件: `config/training.yaml`, `config/aflow_llm.yaml`
- 核心代码:
  - `src/aflow_executor.py`
  - `src/rl_workflow_generator.py`
  - `src/grpo_trainer.py`
