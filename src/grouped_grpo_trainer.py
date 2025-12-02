#!/usr/bin/env python3
"""
分组GRPO训练器 - 每组多问题加权评分 + 多样性打破平局

核心改进:
1. 每 step 包含 3 个问题组（math/qa/code 各 1 组）
2. 每组 4 个问题（2 easy + 2 hard）
3. 每个 workflow 在组内所有问题上评分，加权计算
4. 相同分数时用多样性打破平局，保证有梯度
"""

import os
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ['no_proxy'] = 'localhost,127.0.0.1'

import gc
import torch
import torch.nn.functional as F
import asyncio
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import time
import json
import wandb

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

# 导入新的分组数据管理器和奖励计算器
from grouped_data_manager import GroupedDataManager
from grouped_reward import GroupedRewardCalculator
from vllm_workflow_generator import VLLMWorkflowGenerator
from aflow_executor import AFlowExecutor
from reward_computer import RewardComputer
from gpu_manager import GPUManager
from experience_buffer import ExperienceBuffer
from prompt_optimizer import PromptOptimizer
from operator_prompt_enhancer import OperatorPromptEnhancer


class GroupedGRPOTrainer:
    """分组GRPO训练器：每组多问题评分，确保每 step 包含三种类型"""

    def __init__(self, config_path: str = "config/grouped_training.yaml"):
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        print("=" * 60)
        print("🚀 初始化分组GRPO训练器")
        print("=" * 60)

        # GPU管理
        physical_gpus = self.config.get('physical_gpus', self.config['device_mapping'])
        self.gpu_manager = GPUManager(
            target_gpus=physical_gpus,
            protected_pids=self.config.get('protected_pids', []),
            auto_clean=False
        )
        print(f"✅ 使用GPU {physical_gpus}")

        # Temperature scheduling
        temp_config = self.config.get('temperature_schedule', {})
        self.temp_schedule = {
            'enabled': temp_config.get('enabled', True),
            'initial': temp_config.get('initial', 0.5),
            'final': temp_config.get('final', 0.15),
            'warmup_steps': temp_config.get('warmup_steps', 150)
        }

        # 初始化wandb
        self._initialize_wandb()

        # 初始化组件
        self._initialize_components()

        print("=" * 60)
        print("✅ 分组GRPO训练器初始化完成")
        print("=" * 60)

    def _initialize_wandb(self):
        """初始化wandb"""
        wandb_config = self.config.get('wandb', {})
        wandb_api_key = wandb_config.get('api_key', '')

        try:
            if wandb_api_key and len(wandb_api_key) == 40:
                wandb.login(key=wandb_api_key)
                mode = "online"
            else:
                print("⚠️  wandb API key无效，使用offline模式")
                mode = "offline"
        except Exception as e:
            print(f"⚠️  wandb登录失败: {e}")
            mode = "offline"

        wandb.init(
            project=wandb_config.get('project', 'grouped-grpo'),
            name=wandb_config.get('run_name', f"grouped-{time.strftime('%Y%m%d-%H%M%S')}"),
            mode=mode,
            config={
                "base_model": self.config['base_model'],
                "learning_rate": self.config['learning_rate'],
                "groups_per_domain": self.config.get('groups_per_domain', 1),
                "num_sequences": self.config['num_return_sequences_in_group'],
                "weight_easy": self.config.get('grouped_reward', {}).get('weight_easy', 0.3),
                "weight_hard": self.config.get('grouped_reward', {}).get('weight_hard', 0.7),
            }
        )
        print(f"✅ wandb初始化完成 (mode: {mode})")

    def _initialize_components(self):
        """初始化所有组件"""

        # 1. 分组数据管理器
        print("\n📂 初始化分组数据管理器...")
        self.data_manager = GroupedDataManager(
            data_dir=self.config.get('grouped_data_dir', 'data/grouped'),
            groups_per_domain=self.config.get('groups_per_domain', 1),
            shuffle=True
        )
        self.data_manager.initialize()

        # 2. 分组奖励计算器
        print("\n🎯 初始化分组奖励计算器...")
        reward_config = self.config.get('grouped_reward', {})
        self.grouped_reward = GroupedRewardCalculator(
            weight_easy=reward_config.get('weight_easy', 0.3),
            weight_hard=reward_config.get('weight_hard', 0.7),
            diversity_threshold=reward_config.get('diversity_threshold', 0.05),
            diversity_weight=reward_config.get('diversity_weight', 0.1),
            debug=self.config.get('debug', False)
        )
        print(f"  Easy权重: {reward_config.get('weight_easy', 0.3):.0%}")
        print(f"  Hard权重: {reward_config.get('weight_hard', 0.7):.0%}")
        print(f"  多样性阈值: {reward_config.get('diversity_threshold', 0.05)}")

        # 3. RL模型
        print("\n🤖 加载RL模型...")
        self._load_rl_model()

        # 4. Workflow生成器
        print("\n🔧 初始化Workflow生成器...")
        self.generator = VLLMWorkflowGenerator(
            model_name=self.config['base_model'],
            max_concurrent=self.config['num_return_sequences_in_group'],
            operator_descriptions_path=self.config.get('aflow_operator_descriptions_path'),
            use_vllm_api=False,
            device=f"cuda:{self.config['device_mapping'][0]}"
        )
        self.generator.model = self.model
        self.generator.tokenizer = self.tokenizer

        # 5. ExperienceBuffer
        print("\n📚 初始化ExperienceBuffer...")
        exp_config = self.config.get('experience_buffer', {})
        self.experience_buffer = ExperienceBuffer(
            buffer_size=exp_config.get('buffer_size', 100),
            reward_threshold=exp_config.get('reward_threshold', 8.0),
            persistence_dir=exp_config.get('persistence_dir', 'data/experience_buffer'),
            problem_types=["math", "code", "qa"]
        )

        # 6. PromptOptimizer
        print("\n✨ 初始化PromptOptimizer...")
        self.prompt_optimizer = PromptOptimizer()
        self.use_dynamic_prompts = self.config.get('prompt_optimizer', {}).get('enabled', True)

        # 7. OperatorPromptEnhancer
        print("\n🔧 初始化OperatorPromptEnhancer...")
        self.operator_enhancer = OperatorPromptEnhancer(
            enable_enhancement=self.config.get('operator_prompt_enhancer', {}).get('enabled', True)
        )

        # 8. AFlow执行器
        print("\n⚙️  初始化AFlow执行器...")
        self.executor = AFlowExecutor(
            llm_config_path=self.config['aflow_config_path'],
            timeout=self.config.get('execution_timeout', 600),
            operator_enhancer=self.operator_enhancer
        )

        # 9. 基础奖励计算器（用于单问题评分）
        print("\n🎯 初始化基础奖励计算器...")
        self.reward_computer = RewardComputer(
            reward_weights=self.config.get('reward_weights'),
            use_llm_judge=True,
            llm_config={
                "base_url": "https://api.openai.com/v1",
                "api_key": os.environ.get('OPENAI_API_KEY', 'sk-dummy'),
                "model_name": "gpt-4o-mini"
            },
            debug_logging=self.config.get('debug', False)
        )

        # 10. 优化器
        print("\n🔬 初始化优化器...")
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 0.01)
        )

        # 11. 学习率调度器
        warmup_steps = self.config.get('warmup_steps', 100)
        max_steps = self.config['max_steps']
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps
        )

    def _load_rl_model(self):
        """加载RL模型"""
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['base_model'],
            torch_dtype=torch.bfloat16 if self.config.get('bf16', True) else torch.float16,
            device_map=f"cuda:{self.config['device_mapping'][0]}",
            trust_remote_code=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['base_model'],
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 梯度检查点
        if self.config.get('gradient_checkpointing', True):
            self.model.gradient_checkpointing_enable()
            print("✅ 梯度检查点已启用")

        # LoRA
        if self.config.get('use_lora', True):
            lora_config = LoraConfig(
                r=self.config['lora_rank'],
                lora_alpha=self.config['lora_alpha'],
                target_modules=self.config['lora_target_modules'].split(','),
                lora_dropout=self.config.get('lora_dropout', 0.05),
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.model = get_peft_model(self.model, lora_config)
            print("✅ LoRA应用完成")
            self.model.print_trainable_parameters()

    def get_current_temperature(self, step: int) -> float:
        """获取当前温度"""
        if not self.temp_schedule['enabled']:
            return self.config['generation_config']['temperature']

        if step < self.temp_schedule['warmup_steps']:
            progress = step / self.temp_schedule['warmup_steps']
            temp = (self.temp_schedule['initial'] +
                   progress * (self.temp_schedule['final'] - self.temp_schedule['initial']))
        else:
            temp = self.temp_schedule['final']

        return temp

    async def train_step(self, step: int) -> Dict:
        """
        单步分组GRPO训练

        流程:
        1. 采样 3 个问题组（math/qa/code 各 1 组）
        2. 为每组生成 K 个 workflow
        3. 每个 workflow 在组内 4 个问题上评分
        4. 计算加权得分 + 多样性打破平局
        5. 计算优势值，更新模型
        """
        torch.cuda.reset_peak_memory_stats()

        # 1. 采样问题组（确保包含三种类型）
        groups = self.data_manager.sample_step_groups(split="train")
        stats = self.data_manager.get_step_stats(groups)

        print(f"\n{'='*60}")
        print(f"📍 Step {step}/{self.config['max_steps']}")
        print(f"{'='*60}")
        print(f"📦 采样 {stats['total_groups']} 组 ({stats['total_problems']} 问题)")
        print(f"   组分布: {stats['groups']}")
        print(f"   问题分布: {stats['problems']}")

        current_temp = self.get_current_temperature(step)
        print(f"🌡️  Temperature: {current_temp:.3f}")

        num_sequences = self.config['num_return_sequences_in_group']  # K

        # 收集所有结果
        all_rewards = []
        all_log_probs = []
        all_advantages = []
        step_metrics = {
            'rewards': [],
            'correctness': {'easy': [], 'hard': []},
            'diversity_scores': [],
            'by_domain': {'math': [], 'qa': [], 'code': []}
        }

        # 2. 处理每个问题组
        for group_idx, group in enumerate(groups):
            group_id = group['group_id']
            domain = group['domain']
            problems = group['problems']

            print(f"\n📂 处理组 {group_idx+1}/{len(groups)}: {group_id} ({domain})")

            # 2.1 为该组生成 K 个 workflow
            # 使用组的第一个问题作为代表性输入
            representative_problem = problems[0]['question']

            # 生成 workflow prompt
            if self.use_dynamic_prompts:
                custom_prompt = self.prompt_optimizer.build_dynamic_prompt(
                    problem=representative_problem,
                    problem_type=domain
                )
            else:
                custom_prompt = None

            workflow_results = await self.generator.generate_workflows_batch(
                problems=[representative_problem] * num_sequences,
                problem_types=[domain] * num_sequences,
                temperatures=[current_temp] * num_sequences,
                custom_prompts=[custom_prompt] * num_sequences if custom_prompt else None
            )

            workflows = [r['workflow_code'] for r in workflow_results]

            # 2.2 每个 workflow 在组内所有问题上评分
            problem_scores_per_workflow = [[] for _ in range(num_sequences)]

            for prob_idx, problem in enumerate(problems):
                difficulty = problem['difficulty']
                weight = problem['weight']
                question = problem['question']
                answer = problem['answer']

                print(f"  📝 问题 {prob_idx+1}/{len(problems)} ({difficulty}, weight={weight:.2f})")

                for wf_idx, workflow_code in enumerate(workflows):
                    # 执行 workflow
                    try:
                        pred_answer, cost, exec_meta = await self.executor.execute_workflow(
                            workflow_code=workflow_code,
                            problem=question,
                            problem_type=domain,
                            entry_point=problem.get('entry_point', ''),
                            test=problem.get('test_cases', []),
                            source=problem.get('source', domain),
                            context=problem.get('context', '')
                        )

                        # 计算正确性
                        if exec_meta.get('success', False):
                            correctness = self.reward_computer.compute_reward(
                                problem=question,
                                prediction=pred_answer,
                                ground_truth=answer,
                                problem_type=domain,
                                metadata=exec_meta,
                                test=problem.get('test_cases', []),
                                entry_point=problem.get('entry_point', ''),
                                source=problem.get('source', domain)
                            )
                        else:
                            correctness = 0.0

                    except Exception as e:
                        print(f"    ⚠️  WF{wf_idx+1} 执行失败: {e}")
                        correctness = 0.0

                    # 记录问题得分
                    problem_scores_per_workflow[wf_idx].append({
                        'problem_id': problem['id'],
                        'difficulty': difficulty,
                        'weight': weight,
                        'correctness': correctness
                    })

                    # 统计
                    step_metrics['correctness'][difficulty].append(correctness)

            # 2.3 计算组内奖励（加权 + 多样性）
            group_rewards, diag = self.grouped_reward.calculate_group_rewards(
                workflows=workflows,
                problem_scores_per_workflow=problem_scores_per_workflow
            )

            print(f"\n  🎯 组 {group_id} 奖励:")
            print(f"     加权分: {[f'{s:.3f}' for s in diag['weighted_scores']]}")
            print(f"     多样性: {[f'{s:.3f}' for s in diag['diversity_scores']]}")
            print(f"     最终奖励: {[f'{r:.3f}' for r in group_rewards]}")
            print(f"     需要多样性tiebreak: {diag['need_diversity_tiebreak']}")

            # 2.4 计算优势值
            advantages = self.grouped_reward.compute_advantages(group_rewards)
            print(f"     优势值: {[f'{a:.3f}' for a in advantages]}")

            # 2.5 计算 log prob
            for wf_idx, workflow_code in enumerate(workflows):
                log_prob = await self._compute_log_prob(
                    representative_problem, workflow_code, domain
                )
                all_log_probs.append(log_prob)
                all_rewards.append(group_rewards[wf_idx])
                all_advantages.append(advantages[wf_idx])

            # 统计
            step_metrics['rewards'].extend(group_rewards)
            step_metrics['diversity_scores'].extend(diag['diversity_scores'])
            step_metrics['by_domain'][domain].extend(group_rewards)

        # 3. 梯度更新
        if len(all_advantages) > 0 and any(a != 0 for a in all_advantages):
            loss = self._compute_grpo_loss(
                log_probs=all_log_probs,
                advantages=all_advantages
            )

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.get('max_grad_norm', 1.0)
            )

            self.optimizer.step()
            self.scheduler.step()

            loss_value = loss.item()
        else:
            loss_value = 0.0
            print("⚠️  所有优势值为零，跳过梯度更新")

        # 4. 日志
        metrics = {
            'step': step,
            'loss': loss_value,
            'mean_reward': np.mean(step_metrics['rewards']) if step_metrics['rewards'] else 0,
            'mean_advantage': np.mean(all_advantages) if all_advantages else 0,
            'std_advantage': np.std(all_advantages) if all_advantages else 0,
            'easy_correctness': np.mean(step_metrics['correctness']['easy']) if step_metrics['correctness']['easy'] else 0,
            'hard_correctness': np.mean(step_metrics['correctness']['hard']) if step_metrics['correctness']['hard'] else 0,
            'mean_diversity': np.mean(step_metrics['diversity_scores']) if step_metrics['diversity_scores'] else 0,
            'lr': self.scheduler.get_last_lr()[0],
            'temperature': current_temp
        }

        # 分域统计
        for domain in ['math', 'qa', 'code']:
            if step_metrics['by_domain'][domain]:
                metrics[f'{domain}_reward'] = np.mean(step_metrics['by_domain'][domain])

        wandb.log(metrics)

        print(f"\n📊 Step {step} 总结:")
        print(f"   Loss: {loss_value:.4f}")
        print(f"   平均奖励: {metrics['mean_reward']:.3f}")
        print(f"   Easy正确率: {metrics['easy_correctness']:.3f}")
        print(f"   Hard正确率: {metrics['hard_correctness']:.3f}")
        print(f"   优势std: {metrics['std_advantage']:.4f}")

        # 清理
        gc.collect()
        torch.cuda.empty_cache()

        return metrics

    async def _compute_log_prob(
        self,
        problem: str,
        workflow_code: str,
        problem_type: str
    ) -> torch.Tensor:
        """计算 workflow 的 log 概率"""
        # 构建输入
        input_text = f"Problem type: {problem_type}\nProblem: {problem}\n\n"
        full_text = input_text + workflow_code

        # tokenize
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config['prompt_max_length'] + self.config['response_max_length']
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        input_ids = inputs['input_ids']
        prompt_len = len(self.tokenizer.encode(input_text))

        # 前向传播 - P14修复: 移除torch.no_grad()以保留梯度用于训练
        outputs = self.model(**inputs)
        logits = outputs.logits

        # 计算生成部分的 log prob
        shift_logits = logits[:, prompt_len-1:-1, :]
        shift_labels = input_ids[:, prompt_len:]

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

        return token_log_probs.sum()

    def _compute_grpo_loss(
        self,
        log_probs: List[torch.Tensor],
        advantages: List[float]
    ) -> torch.Tensor:
        """计算 GRPO 损失"""
        losses = []
        for log_prob, advantage in zip(log_probs, advantages):
            if advantage != 0:
                loss = -advantage * log_prob
                losses.append(loss)

        if losses:
            return torch.stack(losses).mean()
        return torch.tensor(0.0, device=self.model.device)

    async def train(self):
        """主训练循环"""
        print("\n" + "="*60)
        print("🎓 开始分组GRPO训练")
        print("="*60)

        max_steps = self.config['max_steps']
        save_every = self.config.get('save_every', 50)
        log_every = self.config.get('log_every', 5)

        for step in range(1, max_steps + 1):
            try:
                metrics = await self.train_step(step)

                # 保存检查点
                if step % save_every == 0:
                    self._save_checkpoint(step)

            except Exception as e:
                print(f"\n❌ Step {step} 出错: {e}")
                import traceback
                traceback.print_exc()
                continue

        print("\n" + "="*60)
        print("✅ 训练完成")
        print("="*60)

        # 保存最终模型
        self._save_checkpoint(max_steps, final=True)
        wandb.finish()

    def _save_checkpoint(self, step: int, final: bool = False):
        """保存检查点"""
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        if final:
            save_path = output_dir / "final_model"
        else:
            save_path = output_dir / f"checkpoint-{step}"

        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        print(f"\n💾 检查点已保存: {save_path}")


async def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="分组GRPO训练")
    parser.add_argument(
        '--config',
        type=str,
        default='config/grouped_training.yaml',
        help='配置文件路径'
    )
    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     分组GRPO训练 - 多问题加权评分 + 多样性打破平局          ║
║                                                              ║
║     每 step: 3 组 × 4 问题 = 12 问题                        ║
║     确保包含: math + qa + code                               ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)

    trainer = GroupedGRPOTrainer(config_path=args.config)
    await trainer.train()


if __name__ == "__main__":
    asyncio.run(main())
