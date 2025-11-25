# 显存泄漏修复 - 实施说明

## 修复列表

### 1. src/grpo_trainer.py:288 - 添加显存监控
在 train_step 方法开始处添加:
```python
import torch
import gc

# 显存监控开始
torch.cuda.reset_peak_memory_stats()
mem_before = torch.cuda.memory_allocated() / 1e9
```

### 2. src/grpo_trainer.py:590 - 添加显存清理和监控记录
在 return metrics 之前添加:
```python
# 清理张量列表,释放显存
del all_workflows, all_problems, all_answers, all_rewards, all_log_probs, correctness_scores
torch.cuda.empty_cache()
gc.collect()

# 显存监控结束
mem_after = torch.cuda.memory_allocated() / 1e9
mem_peak = torch.cuda.max_memory_allocated() / 1e9
print(f"显存: 前={mem_before:.2f}GB, 后={mem_after:.2f}GB, 峰值={mem_peak:.2f}GB, 增长={(mem_after-mem_before):.3f}GB")

# 记录到wandb
wandb_log_data["memory/allocated_gb"] = mem_after
wandb_log_data["memory/peak_gb"] = mem_peak
wandb_log_data["memory/growth_gb"] = mem_after - mem_before
```

### 3. src/grpo_trainer.py:696 - 修复梯度清理
将:
```python
self.optimizer.zero_grad()
```
改为:
```python
self.optimizer.zero_grad(set_to_none=True)
```

### 4. src/experience_buffer.py - 限制样本大小
在 add_sample 方法中添加截断逻辑

## 实施步骤

由于修改较多,建议使用脚本批量修改。