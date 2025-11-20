# AFlow 深入研究成果

**研究时间**: 2025-11-16
**研究文件数**: 38+ Python 文件
**研究仓库**: `/home/yijia/.claude/11/AFlow`

---

## 🎯 核心发现

### 发现 1: LLM 配置的正确流程

**问题根源**: `'dict' object has no attribute 'call_with_format'`

**正确的数据流**:
```
YAML → dict → LLMsConfig(dict) → LLMsConfig.get(name) → LLMConfig 实例 →
create_llm_instance(LLMConfig) → AsyncLLM 实例 → operator.Custom(AsyncLLM) →
AsyncLLM.call_with_format() ✓
```

**关键代码**:
```python
# 1. LLMsConfig.get() 返回 LLMConfig 实例（不是 dict！）
llm_config = LLMsConfig(models_dict).get("gpt-4o-mini")
# 返回类型: LLMConfig 实例

# 2. create_llm_instance() 支持三种输入
def create_llm_instance(llm_config):
    if isinstance(llm_config, LLMConfig):  # 推荐
        return AsyncLLM(llm_config)
    elif isinstance(llm_config, str):      # 也支持
        return AsyncLLM(llm_config)
    elif isinstance(llm_config, dict):     # 会转换
        return AsyncLLM(LLMConfig(llm_config))
```

**修复方案**: 在 `_get_llm_config()` 中添加类型检查和转换。

---

### 发现 2: AnswerGenerate 不接受 instruction 参数！

**错误**: `TypeError: AnswerGenerate.__call__() got an unexpected keyword argument 'instruction'`

**算子 API 签名汇总**:

| 算子 | 参数 | 返回值 |
|------|------|--------|
| `Custom` | `input: str`<br>`instruction: str` | `{'response': str}` |
| `AnswerGenerate` | `input: str` **（只有这一个！）** | `{'thought': str, 'answer': str}` |
| `Programmer` | `problem: str`<br>`analysis: str = "None"` | `{'code': str, 'output': str}` |
| `ScEnsemble` | `solutions: List[str]`<br>`problem: str` | `{'response': str}` |
| `Review` | `problem: str`<br>`solution: str` | `{'review_result': bool, 'feedback': str}` |
| `Revise` | `problem: str`<br>`solution: str`<br>`feedback: str` | `{'solution': str}` |

**正确用法**:
```python
# ❌ 错误
result = await self.answer_generate(input=problem, instruction="...")

# ✅ 正确
result = await self.answer_generate(input=problem)
thought = result['thought']
answer = result['answer']
```

**如果需要自定义指令，应该使用 Custom 算子**:
```python
result = await self.custom(input=problem, instruction="Solve step by step")
answer = result['response']
```

---

### 发现 3: Workflow 标准模板

**所有 Workflow 都遵循相同结构**:

```python
import workspace.{problem_type}.workflows.template.operator as operator
from scripts.async_llm import create_llm_instance
from scripts.evaluator import DatasetType

class Workflow:
    def __init__(self, name: str, llm_config, dataset: DatasetType):
        self.name = name
        self.dataset = dataset
        # 关键：使用 create_llm_instance
        self.llm = create_llm_instance(llm_config)
        # 初始化算子（传入 AsyncLLM 实例）
        self.custom = operator.Custom(self.llm)

    async def __call__(self, problem: str):
        """执行工作流"""
        solution = await self.custom(input=problem, instruction="")
        # 关键：必须返回 (answer, cost) 元组
        return solution['response'], self.llm.get_usage_summary()["total_cost"]
```

**关键点**:
1. 使用 `create_llm_instance(llm_config)` 创建 LLM
2. 算子初始化传入 `self.llm`（AsyncLLM 实例）
3. 算子返回值是字典，需要访问对应键（如 `['response']`）
4. 必须返回 `(solution, cost)` 元组
5. Cost 通过 `self.llm.get_usage_summary()["total_cost"]` 获取

---

## 🔧 具体修复方案

### 修复 1: 彻底修复 LLM 配置错误

**位置**: `src/aflow_executor.py:251-265`

**问题**: `_get_llm_config()` 可能返回错误类型

**修复代码**:
```python
def _get_llm_config(self):
    """获取LLM配置（确保返回正确类型）"""
    from scripts.async_llm import LLMsConfig, LLMConfig

    try:
        if self.llm_configs:
            result = self.llm_configs.get(self.llm_model_name)
        else:
            result = LLMsConfig.default().get(self.llm_model_name)

        # 类型验证（关键！）
        if isinstance(result, LLMConfig):
            return result
        elif isinstance(result, dict):
            # 如果意外返回了 dict，转换为 LLMConfig
            print(f"⚠️  警告：get() 返回了 dict，正在转换为 LLMConfig")
            return LLMConfig(result)
        elif isinstance(result, str):
            return result
        else:
            print(f"⚠️  未知类型: {type(result)}，降级为字符串")
            return self.llm_model_name

    except Exception as e:
        print(f"⚠️  获取LLM配置失败: {e}")
        import traceback
        traceback.print_exc()
        return self.llm_model_name
```

---

### 修复 2: 优化 Workflow 生成 Prompt

**位置**: `src/rl_workflow_generator.py:113-139`

**问题**: Prompt 没有明确说明每个算子的精确 API

**修复代码**:
```python
def _build_generation_prompt(self, problem: str, problem_type: str) -> str:
    """构建提示词，明确算子 API"""

    prompt = f"""Generate a Python Workflow class. Follow the exact template and API signatures.

IMPORTANT: Only use operators listed below with their EXACT parameters.

Available Operators:

1. Custom(llm) - Most flexible
   Call: await self.custom(input=str, instruction=str)
   Returns: {{'response': str}}

2. AnswerGenerate(llm) - Step-by-step reasoning
   Call: await self.answer_generate(input=str)  ← NO instruction parameter!
   Returns: {{'thought': str, 'answer': str}}

3. Programmer(llm) - Auto-generate and execute code
   Call: await self.programmer(problem=str, analysis=str)
   Returns: {{'code': str, 'output': str}}

Template (fill in the logic):

import workspace.{problem_type}.workflows.template.operator as operator
from scripts.async_llm import create_llm_instance
from scripts.evaluator import DatasetType

class Workflow:
    def __init__(self, name: str, llm_config, dataset: DatasetType):
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        # Initialize operators (e.g., self.custom = operator.Custom(self.llm))

    async def __call__(self, problem: str):
        # Use operators to solve: {problem}
        # Must return (solution, cost) tuple
        # Example: return solution['response'], self.llm.get_usage_summary()["total_cost"]
        pass
"""
    return prompt
```

---

### 修复 3: 添加代码验证

**位置**: `src/rl_workflow_generator.py:213-256`

**在 `_parse_workflow_code` 中添加验证**:

```python
def _parse_workflow_code(self, generated_text: str, problem_type: str):
    """解析并验证工作流代码"""

    # ... 现有代码提取逻辑 ...

    # 新增：验证常见错误
    if code:
        # 检查 AnswerGenerate 错误用法
        if "answer_generate(" in code and "instruction=" in code:
            print(f"⚠️  检测到错误：AnswerGenerate 不接受 instruction 参数")
            # 自动修复
            code = code.replace(
                "await self.answer_generate(input=problem, instruction=",
                "await self.answer_generate(input=problem) # Fixed: removed instruction="
            )
            print(f"  已自动修复")

        # 检查是否返回了 cost
        if "return" in code and "get_usage_summary" not in code:
            print(f"⚠️  警告：可能缺少 cost 计算")

    # ... 继续语法验证 ...
```

---

## 📝 推荐的工作流模式

### 模式 1: 简单单步（推荐用于大多数情况）

```python
class Workflow:
    def __init__(self, name: str, llm_config, dataset: DatasetType):
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)

    async def __call__(self, problem: str):
        solution = await self.custom(
            input=problem,
            instruction="Solve this problem step by step and provide the final answer."
        )
        return solution['response'], self.llm.get_usage_summary()["total_cost"]
```

### 模式 2: 使用 Programmer（数学问题自动执行代码）

```python
class Workflow:
    def __init__(self, name: str, llm_config, dataset: DatasetType):
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.programmer = operator.Programmer(self.llm)

    async def __call__(self, problem: str):
        result = await self.programmer(problem=problem, analysis="None")
        return result['output'], self.llm.get_usage_summary()["total_cost"]
```

### 模式 3: Self-Consistency（生成多个答案并选择最佳）

```python
class Workflow:
    def __init__(self, name: str, llm_config, dataset: DatasetType):
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)

    async def __call__(self, problem: str):
        # 生成多个候选答案
        solutions = []
        for _ in range(3):
            sol = await self.custom(input=problem, instruction="")
            solutions.append(sol['response'])

        # 选择最一致的答案
        final = await self.sc_ensemble(solutions=solutions, problem=problem)
        return final['response'], self.llm.get_usage_summary()["total_cost"]
```

---

## 🎯 立即行动计划

### 优先级 1（立即修复）

1. ✅ **更新 `_get_llm_config()`** - 添加类型检查和转换
2. ✅ **优化生成 Prompt** - 明确算子 API 签名
3. ✅ **添加代码验证** - 自动检测和修复常见错误

### 优先级 2（短期优化）

4. 添加 Few-shot 示例到 Prompt
5. 实现代码自动修复机制
6. 优化 temperature 和采样参数

### 优先级 3（长期改进）

7. 实现奖励信号修正（惩罚无效生成）
8. 添加 Curriculum Learning
9. 收集和分析成功的 Workflow 模式

---

## 📚 参考资料

- **LLM 配置**: `/home/yijia/.claude/11/AFlow/scripts/async_llm.py`
- **算子定义**: `/home/yijia/.claude/11/AFlow/scripts/operators.py`
- **官方示例**: `/home/yijia/.claude/11/AFlow/workspace/*/workflows/round_1/graph.py`
- **算子 JSON**: `/home/yijia/.claude/11/AFlow/workspace/*/workflows/template/operator.json`

---

**研究完成** ✅
