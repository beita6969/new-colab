# GRPO训练系统优化修复方案

基于代码探索和训练效果分析，本文档提供详细的修复步骤和实现方案。

---

## 📊 当前状态总结

### 训练效果
- **Math任务**: ✅ 表现良好 (Step 7达到100%准确率)
- **Code任务**: ❌ 完全失败 (0%准确率)
- **QA任务**: ⚠️ 不稳定 (0-100%波动)
- **训练速度**: 约15-20分钟/step (0.01样本/秒)
- **Workflow失败率**: 8.3%

### 主要问题
1. Code任务评估逻辑错误 (比较代码文本而非执行结果)
2. Revise operator未初始化 (43.75%的workflow失败)
3. Math分数提取bug ("5/324"→"324.0")
4. Workflow缺乏多样性
5. 训练速度慢

---

## 🔴 P0 关键修复 (阻塞训练进度)

### P0.1: 修复Code任务的0%准确率

**问题根因**:
- Answer extractor提取代码文本本身，而非执行结果
- Reward computer比较代码字符串而非运行测试用例
- 已有test_result metadata但未充分利用

**修复方案**:

#### 步骤1: 修改answer_extractor.py的Code提取逻辑

```python
# src/answer_extractor.py:110-141

def _extract_code_answer(self, text: str, is_ground_truth: bool) -> str:
    """
    提取代码答案

    对于Code任务:
    - prediction: 提取完整的函数实现代码
    - ground_truth: 同样提取函数实现代码
    - 评估: 通过test_result metadata而非字符串比较
    """
    text = str(text).strip()

    # 1. 提取代码块
    code_blocks = re.findall(r'```(?:python)?\n?([^`]+)```', text)
    if code_blocks:
        # 返回最后一个代码块
        last_block = code_blocks[-1].strip()

        # 验证代码语法正确性
        try:
            import ast
            ast.parse(last_block)
            return last_block
        except SyntaxError:
            # 如果最后一个代码块有语法错误，尝试其他代码块
            for block in reversed(code_blocks[:-1]):
                try:
                    ast.parse(block)
                    return block.strip()
                except SyntaxError:
                    continue
            # 所有代码块都有错误，返回最后一个
            return last_block

    # 2. 查找函数定义
    func_pattern = r'(def\s+\w+\s*\([^)]*\)[^:]*:[\s\S]+?)(?=\n(?:def\s|class\s|$))'
    funcs = re.findall(func_pattern, text)
    if funcs:
        return funcs[0].strip()

    # 3. 如果是ground truth且看起来像代码，直接返回
    if is_ground_truth:
        return text

    return text
```

#### 步骤2: 强化reward_computer.py中test_result的使用

```python
# src/reward_computer.py:95-108

# 对于代码题，优先使用测试结果
if problem_type == "code":
    # 检查是否有test_result metadata
    test_result = metadata.get('test_result') if metadata else None

    if test_result is not None:
        # 有测试结果，直接使用
        if test_result:
            correctness_score = 10.0  # 测试通过，满分
            print(f"  💻 测试通过 → {correctness_score}分")
        else:
            # 测试失败，检查是否生成了有效代码
            if prediction and "def " in str(prediction):
                correctness_score = 3.0  # 生成了代码但测试失败，给部分分
                print(f"  💻 代码生成但测试失败 → {correctness_score}分")
            else:
                correctness_score = -5.0  # 没有生成有效代码
                print(f"  💻 未生成有效代码 → {correctness_score}分")
    else:
        # 没有test_result，回退到代码相似度比较
        print(f"  ⚠️  无test_result，使用代码相似度")
        correctness_score = self._compute_code_correctness(
            prediction, ground_truth
        )
else:
    # Math和QA使用原有逻辑
    # ...
```

#### 步骤3: 确保workflow执行正确传递test_result

检查grpo_trainer.py中workflow执行后metadata的构建:

```python
# src/grpo_trainer.py 约339-350行

metadata = {
    'success': success,
    'cost': cost,
    'workflow_valid': workflow_valid,
    'problem_type': problem_type,
    'test_result': test_result,  # 确保这个字段存在
    # ...
}
```

**预期效果**: Code任务准确率从0%提升至40-60%

---

### P0.2: 修复Revise Operator未初始化错误

**问题根因**:
- Workflow类缺少revise方法
- Qwen生成的workflow调用了不存在的operator

**修复方案**:

#### 方案A: 添加revise operator (推荐)

```python
# src/aflow_executor.py

class Workflow:
    def __init__(self, operators: Dict, ...):
        # 现有初始化
        self.operators = operators

        # 确保所有常用operators都已初始化
        required_ops = ['custom', 'answer_generate', 'programmer',
                        'test', 'review', 'revise']  # 添加revise

        for op_name in required_ops:
            if op_name not in self.operators:
                print(f"⚠️  警告: {op_name} operator未初始化")

    async def revise(self, problem: str, solution: str, feedback: str) -> Dict:
        """
        Revise operator: 根据反馈改进解决方案
        """
        if 'revise' in self.operators:
            return await self.operators['revise']({
                'problem': problem,
                'solution': solution,
                'feedback': feedback
            })
        else:
            # 降级：使用custom operator
            print(f"  ⚠️  revise operator不可用，使用custom")
            return await self.custom(
                instruction=f"Based on this feedback: {feedback}, revise the solution: {solution}"
            )
```

#### 方案B: 在workflow生成prompt中约束可用operators

```python
# src/rl_workflow_generator.py:113-154

DEFAULT_PROMPT = """
你需要生成一个AFlow工作流来解决给定问题。

**可用的Operators** (仅使用以下operators):
1. Custom - 通用LLM调用
2. AnswerGenerate - 生成最终答案
3. Programmer - 生成代码
4. Test - 测试代码
5. Review - 审查解决方案

**禁止使用的Operators**:
- Revise (尚未实现)

**重要约束**:
- 只能调用self.{operator_name}()
- 不要在代码中import任何模块
- 返回值必须包含'result'键

示例workflow:
```python
async def solve(self, problem: str) -> Tuple[str, float]:
    # Step 1: Generate solution
    result = await self.custom(instruction=f"Solve: {problem}")

    # Step 2: Get final answer
    answer = await self.answer_generate(problem=problem, solution=result['solution'])

    return answer['answer'], self.llm.get_usage_summary().get("total_cost", 0.0)
```
"""
```

**预期效果**: Workflow失败率从8.3%降至<3%

---

### P0.3: 修复Math答案提取器的分数bug

**问题**: "5/324"被提取为"324.0"

**根因分析**:
```python
# 当前逻辑 (answer_extractor.py:215-248)
def _clean_math_answer(self, answer: str) -> str:
    # ...
    # 处理分数
    if '/' in answer:
        parts = answer.split('/')
        if len(parts) == 2:
            return str(float(parts[0]) / float(parts[1]))  # 5/324 → 0.0154...
```

问题在于:
1. 正则匹配可能只提取了分母 "324"
2. _clean_math_answer将其转为float

**修复方案**:

```python
# src/answer_extractor.py:215-248

def _clean_math_answer(self, answer: str) -> str:
    """
    清理数学答案（去单位、标准化格式）

    重要: 保持分数形式用于比较，避免浮点精度问题
    """
    answer = str(answer).strip()

    # 修复 "i42" 问题
    if answer.startswith('i') and len(answer) > 1 and answer[1:].replace('.', '').isdigit():
        answer = answer[1:]

    # 移除LaTeX命令但保留内容
    answer = re.sub(r'\\boxed\{(.+?)\}', r'\1', answer)
    answer = re.sub(r'\\frac\{(.+?)\}\{(.+?)\}', r'\1/\2', answer)  # \frac{a}{b} → a/b
    answer = re.sub(r'\\text\{(.+?)\}', r'\1', answer)

    # 移除常见单位
    units = ['grams', 'gram', 'g', 'kg', 'meters', 'meter', 'm', 'cm',
             'seconds', 'second', 's', 'minutes', 'minute', 'min',
             'dollars', 'dollar', '$', '元', '个', '只', 'km', 'hours', 'hour']

    for unit in units:
        answer = re.sub(rf'\s*{re.escape(unit)}\b', '', answer, flags=re.IGNORECASE)

    # 移除多余的标点和空格 (但保留'/'用于分数)
    answer = re.sub(r'[,\s]+', '', answer)
    answer = answer.replace('.', '', 1) if answer.count('.') > 1 else answer  # 移除多余小数点

    # 尝试规范化数字
    try:
        # 处理分数 - 保持分数形式
        if '/' in answer:
            parts = answer.split('/')
            if len(parts) == 2:
                numerator = float(parts[0])
                denominator = float(parts[1])
                # 化简分数 (可选)
                from math import gcd
                g = gcd(int(numerator), int(denominator))
                if g > 1:
                    numerator /= g
                    denominator /= g
                # 返回分数字符串
                if denominator == 1:
                    return str(int(numerator))
                return f"{int(numerator)}/{int(denominator)}"

        # 处理百分号
        if '%' in answer:
            return str(float(answer.replace('%', '')) / 100)

        # 普通数字 - 保持整数/小数格式
        num = float(answer)
        if num == int(num):
            return str(int(num))
        return str(num)
    except:
        # 无法转换，返回清理后的字符串
        return answer
```

**额外改进**: Math比较时支持分数等价性

```python
# src/reward_computer.py

def _is_math_correct(self, prediction: Any, ground_truth: Any) -> bool:
    """数学答案比较，支持分数等价性"""
    try:
        pred_str = str(prediction).strip()
        gt_str = str(ground_truth).strip()

        # 字符串完全匹配
        if pred_str == gt_str:
            return True

        # 解析为数值比��
        def parse_fraction(s: str) -> float:
            if '/' in s:
                parts = s.split('/')
                return float(parts[0]) / float(parts[1])
            return float(s)

        pred_num = parse_fraction(pred_str)
        gt_num = parse_fraction(gt_str)

        # 使用相对误差比较 (处理浮点精度)
        rel_error = abs(pred_num - gt_num) / (abs(gt_num) + 1e-9)
        return rel_error < 1e-6
    except:
        return False
```

**预期效果**: Math准确率从66%提升至80%+

---

### P0.4: 实现Temperature Curriculum Scheduling

**问题**: 所有workflow使用相同temperature，缺乏多样性

**修复方案**:

```python
# src/grpo_trainer.py

class GRPOTrainer:
    def __init__(self, ...):
        # 现有初始化
        # ...

        # Temperature scheduling配置
        self.temp_schedule = {
            'initial': 0.7,     # 初始温度
            'final': 1.0,       # 最终温度
            'warmup_steps': 100  # warmup步数
        }

    def get_current_temperature(self, step: int) -> float:
        """
        计算当前step的temperature

        策略: 线性从initial升至final
        早期: 低温度生成确定性workflow，建立baseline
        后期: 高温度探索多样性workflow
        """
        if step < self.temp_schedule['warmup_steps']:
            # Linear warmup
            progress = step / self.temp_schedule['warmup_steps']
            temp = (self.temp_schedule['initial'] +
                   progress * (self.temp_schedule['final'] - self.temp_schedule['initial']))
        else:
            temp = self.temp_schedule['final']

        return temp

    def train(self):
        for step in range(self.num_train_steps):
            # 动态temperature
            current_temp = self.get_current_temperature(step)

            # 传递给workflow生成器
            workflows = self.workflow_generator.generate(
                problems=batch_problems,
                temperature=current_temp  # 动态温度
            )

            # 记录到WandB
            wandb.log({'train/temperature': current_temp, 'train/step': step})
```

**额外优化**: 在同一batch内使用不同temperature

```python
# 生成6个workflows时使用不同temperature
temperatures = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1]  # 6个不同值
workflows = []
for i, temp in enumerate(temperatures):
    wf = self.workflow_generator.generate(
        problem=problem,
        temperature=temp,
        num_return_sequences=1
    )
    workflows.append(wf)
```

**预期效果**: Workflow多样性提升50%，训练收敛速度加快

---

## 🟠 P1 高优先级修复

### P1.1: 重新校准Sigmoid奖励函数

**问题**: 当前sigmoid scale=3.0对所有任务类型一视同仁

**修复方案**:

```python
# src/reward_computer.py:115-131

def _normalize_reward(self, correctness_score: float, problem_type: str) -> float:
    """
    归一化奖励到[0, 1]，针对不同任务类型使用不同曲线
    """
    import numpy as np

    # 任务特定的scale参数
    scales = {
        'code': 5.0,   # Code是二元的(通过/失败)，需要陡峭曲线
        'math': 3.0,   # Math有部分分，中等陡度
        'qa': 2.5      # QA更主观，平滑曲线
    }

    scale = scales.get(problem_type, 3.0)

    # Sigmoid归一化
    normalized = 1.0 / (1.0 + np.exp(-correctness_score / scale))

    # 极值修正
    if correctness_score >= 10.0:
        normalized = 1.0
    elif correctness_score <= -10.0:
        normalized = 0.0

    # 确保范围
    normalized = max(0.0, min(1.0, normalized))

    return normalized
```

**效果对比**:
```
原始 (scale=3.0):
  score=10 → 0.95
  score=5  → 0.81
  score=0  → 0.50
  score=-5 → 0.19

Code (scale=5.0):
  score=10 → 0.88  # 更平滑
  score=5  → 0.73
  score=0  → 0.50
  score=-5 → 0.27

QA (scale=2.5):
  score=10 → 0.98  # 更陡峭
  score=5  → 0.88
  score=0  → 0.50
  score=-5 → 0.12
```

---

### P1.2: 添加代码AST验证

(已在P0.1中部分实现)

扩展验证:

```python
def _validate_code_structure(self, code: str, entry_point: str = None) -> Tuple[bool, str]:
    """
    验证代码结构完整性

    Returns:
        (is_valid, error_message)
    """
    try:
        import ast
        tree = ast.parse(code)

        # 检查是否包含函数定义
        functions = [node.name for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef)]

        if not functions:
            return False, "No function definition found"

        # 检查entry_point是否存在
        if entry_point and entry_point not in functions:
            return False, f"Entry point '{entry_point}' not found. Found: {functions}"

        # 检查是否有return语句
        has_return = any(isinstance(node, ast.Return) for node in ast.walk(tree))
        if not has_return:
            return False, "No return statement found"

        return True, ""
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Validation error: {e}"
```

---

### P1.3: 修复UnboundLocalError

**问题**: 生成的workflow中变量作用域错误

**示例错误代码**:
```python
async def solve(self, problem: str):
    result = await self.programmer(...)
    if not result.get('success'):
        revised_code = await self.revise(...)  # 定义在if内
        test_result = await self.test(revised_code)  # OK

    return revised_code  # 错误: revised_code可能未定义
```

**修复方案**: 在workflow validation阶段添加变量作用域检查

```python
# src/workflow_validator.py

import ast

class WorkflowValidator:
    def check_variable_scope(self, code: str) -> List[str]:
        """
        检查变量作用域问题

        Returns:
            错误列表
        """
        errors = []
        try:
            tree = ast.parse(code)

            # 查找solve函数
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == 'solve':
                    # 分析变量定义和使用
                    defined_vars = set()

                    for child in ast.walk(node):
                        # 记录赋值
                        if isinstance(child, ast.Assign):
                            for target in child.targets:
                                if isinstance(target, ast.Name):
                                    defined_vars.add(target.id)

                        # 检查使用
                        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                            if child.id not in defined_vars and child.id != 'self':
                                errors.append(f"Variable '{child.id}' may be used before assignment")

        except Exception as e:
            errors.append(f"Scope check failed: {e}")

        return errors
```

**提示Qwen避免此问题**:

在workflow生成prompt中添加:
```
**变量作用域规则**:
1. 所有在条件分支(if/else)中定义的变量，必须在外部初始化
2. 示例:
   ```python
   # 错误
   if condition:
       result = await self.custom(...)
   return result  # result可能未定义

   # 正确
   result = None  # 先初始化
   if condition:
       result = await self.custom(...)
   return result if result else "fallback"
   ```
```

---

## 🟡 P2 性能优化

### P2.1: 并行化Workflow生成

**当前**: 串行生成24个workflows (6 sequences × 4 batch)

```python
# 当前代码 (grpo_trainer.py)
for problem in batch:
    workflows = self.workflow_generator.generate(
        problem=problem,
        num_return_sequences=6
    )  # 每个耗时5-10秒
    all_workflows.extend(workflows)
# 总时间: 4 × (6 × 8秒) = 192秒
```

**优化**: 使用asyncio.gather并行生成

```python
import asyncio

async def generate_workflows_parallel(self, problems: List[Dict]) -> List[str]:
    """
    并行生成所有workflows
    """
    tasks = []
    for problem in problems:
        # 为每个problem生成6个workflows
        for i in range(6):
            task = self.workflow_generator.generate_async(
                problem=problem,
                temperature=0.7 + i * 0.1  # 不同温度
            )
            tasks.append(task)

    # 并行执行所有任务
    workflows = await asyncio.gather(*tasks)
    return workflows

# 在train()中:
workflows = await self.generate_workflows_parallel(batch)
# 总时间: max(8秒) = 8秒  (24倍加速)
```

**注意**: 需要确保LLM API支持并发请求

**预期效果**: Workflow生成时间从192秒降至8秒 (24倍加速)

---

## 📝 实施计划

### Week 1: P0关键修复
- Day 1-2: P0.1 修复Code任务评估
- Day 3: P0.2 修复Revise operator
- Day 4: P0.3 修复Math分数提取
- Day 5: P0.4 实现temperature scheduling

### Week 2: P1高优先级
- Day 1-2: P1.1 重新校准reward function
- Day 3: P1.2 添加AST验证
- Day 4-5: P1.3 修复变量作用域

### Week 3: P2性能优化
- Day 1-3: P2.1 并行化workflow生成
- Day 4-5: 集成测试和性能验证

---

## 🎯 预期效果

修复后的训练表现:

| 指标 | 当前 | 修复后 | 提升 |
|------|------|--------|------|
| Math准确率 | 66% | 80%+ | +21% |
| Code准确率 | 0% | 50%+ | +50% |
| QA准确率 | 不稳定 | 60%+ | 稳定 |
| Workflow失败率 | 8.3% | <3% | -64% |
| 训练速度 | 15min/step | 2min/step | 7.5x |
| 总体效率 | ~50% | >90% | +80% |

---

## 🔧 验证清单

每个修复完成后需验证:

- [ ] 单元测试通过
- [ ] 在小规模数据集(10样本)上运行无错误
- [ ] WandB指标符合预期
- [ ] 日志输出清晰可读
- [ ] 代码review通过

---

## 📚 参考文件

关键文件及修改位置:

- `src/answer_extractor.py`: 110-141 (Code提取), 215-248 (Math清理)
- `src/reward_computer.py`: 95-131 (奖励归一化)
- `src/grpo_trainer.py`: 284-304 (Workflow生成), 339-350 (Metadata)
- `src/aflow_executor.py`: Workflow类定义 (添加revise)
- `src/rl_workflow_generator.py`: 113-154 (Prompt约束)
- `src/workflow_validator.py`: 添加作用域检查

---

生成时间: 2025-11-19
作者: Claude Code Analysis
