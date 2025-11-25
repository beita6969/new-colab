#!/usr/bin/env python3
"""
显存泄漏修复脚本
自动为grpo_trainer.py添加显存监控和内存清理
"""
import re

def fix_grpo_trainer():
    file_path = "src/grpo_trainer.py"

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 修复1: 在train_step开始添加显存监控
    content = content.replace(
        '    async def train_step(self, step: int) -> Dict:\n        """\n        单步GRPO训练（在线学习）\n\n        Returns:\n            metrics: 训练指标\n        """\n\n        # 1. 采样batch',
        '''    async def train_step(self, step: int) -> Dict:
        """
        单步GRPO训练（在线学习）

        Returns:
            metrics: 训练指标
        """
        import torch
        import gc

        # 显存监控开始
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated() / 1e9

        # 1. 采样batch'''
    )

    # 修复2: 在return metrics前添加清理代码
    content = content.replace(
        '        wandb.log(wandb_log_data, step=step)\n\n        return metrics',
        '''        # 清理张量列表,释放显存
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

        wandb.log(wandb_log_data, step=step)

        return metrics'''
    )

    # 修复3: 梯度清理优化
    content = content.replace(
        '                self.optimizer.zero_grad()',
        '                self.optimizer.zero_grad(set_to_none=True)'
    )

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print("✅ grpo_trainer.py 修复完成")

def fix_experience_buffer():
    file_path = "src/experience_buffer.py"

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 在add_sample中添加截断
    old_pattern = r"(def add_sample\(self, sample: Dict, problem_type: str\):[^}]+?)\n(\s+# 添加到对应类型的buffer)"

    new_code = r'''\1
        # 截断长文本,防止内存膨胀
        if 'workflow_code' in sample and len(str(sample['workflow_code'])) > 5000:
            sample['workflow_code'] = str(sample['workflow_code'])[:5000] + "...[truncated]"
        if 'answer' in sample and len(str(sample['answer'])) > 2000:
            sample['answer'] = str(sample['answer'])[:2000] + "...[truncated]"

\2'''

    content = re.sub(old_pattern, new_code, content)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print("✅ experience_buffer.py 修复完成")

if __name__ == "__main__":
    print("开始修复显存泄漏问题...")
    fix_grpo_trainer()
    fix_experience_buffer()
    print("\n所有修复完成!")
