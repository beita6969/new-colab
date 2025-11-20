#!/usr/bin/env python3
"""
GRPO训练器 - 在线学习模式的强化学习训练器
"""
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
import wandb  # ✨ 新增wandb集成

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_manager import DataManager
from rl_workflow_generator import RLWorkflowGenerator
from aflow_executor import AFlowExecutor
from reward_computer import RewardComputer
from gpu_manager import GPUManager
from experience_buffer import ExperienceBuffer
from prompt_optimizer import PromptOptimizer
from operator_prompt_enhancer import OperatorPromptEnhancer


class GRPOTrainer:
    """GRPO训练器：在线学习模式"""

    def __init__(self, config_path: str = "config/training.yaml"):
        """
        Args:
            config_path: 训练配置文件路径
        """
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        print("=" * 60)
        print("🚀 初始化GRPO训练器")
        print("=" * 60)

        # GPU管理（使用物理GPU ID）
        physical_gpus = self.config.get('physical_gpus', self.config['device_mapping'])
        self.gpu_manager = GPUManager(
            target_gpus=physical_gpus,
            protected_pids=self.config.get('protected_pids', []),
            auto_clean=False  # 禁用自动清理
        )

        # 跳过GPU环境验证，直接使用
        print(f"✅ 使用GPU {physical_gpus}（已禁用清理和验证）")

        # Temperature scheduling配置
        temp_config = self.config.get('temperature_schedule', {})
        self.temp_schedule = {
            'enabled': temp_config.get('enabled', True),
            'initial': temp_config.get('initial', 0.3),
            'final': temp_config.get('final', 0.8),
            'warmup_steps': temp_config.get('warmup_steps', 100)
        }
        print(f"\n🌡️  Temperature Scheduling:")
        print(f"  Enabled: {self.temp_schedule['enabled']}")
        if self.temp_schedule['enabled']:
            print(f"  Range: {self.temp_schedule['initial']} → {self.temp_schedule['final']}")
            print(f"  Warmup: {self.temp_schedule['warmup_steps']} steps")

        # ✨ 初始化wandb
        self._initialize_wandb()

        # 初始化组件
        self._initialize_components()

        print("=" * 60)
        print("✅ GRPO训练器初始化完成")
        print("=" * 60)

    def _initialize_wandb(self):
        """初始化wandb监控"""
        # 从配置或环境变量获取wandb设置
        wandb_config = self.config.get('wandb', {})

        # 设置API key(如果提供的话)
        wandb_api_key = wandb_config.get('api_key', 'b42ca0000cf06f97b05eba34f58823ad5f3122a4')

        # 尝试登录,如果失败则使用offline模式
        try:
            if wandb_api_key and len(wandb_api_key) == 40:
                wandb.login(key=wandb_api_key)
                mode = "online"
            else:
                print("⚠️  wandb API key无效或未提供,使用offline模式")
                mode = "offline"
        except Exception as e:
            print(f"⚠️  wandb登录失败: {e}, 使用offline模式")
            mode = "offline"

        # 初始化wandb run
        wandb.init(
            project=wandb_config.get('project', 'aflow-roll-integration'),
            name=wandb_config.get('run_name', f"grpo-training-{time.strftime('%Y%m%d-%H%M%S')}"),
            mode=mode,  # online或offline
            config={
                # 训练配置
                "base_model": self.config['base_model'],
                "learning_rate": self.config['learning_rate'],
                "batch_size": self.config['rollout_batch_size'],
                "num_sequences": self.config['num_return_sequences_in_group'],
                "max_steps": self.config['max_steps'],
                "lora_rank": self.config['lora_rank'],
                "lora_alpha": self.config['lora_alpha'],
                # 数据配置
                "domain_ratios": self.config['domain_ratios'],
                # 奖励配置
                "reward_weights": self.config.get('reward_weights', {}),
            },
            tags=["grpo", "aflow", "roll", "workflow-generation"],
            notes="GRPO training with improved reward function (ROLL+AgentFlow design)"
        )

        print("\n✅ wandb初始化完成")
        print(f"  模式: {mode}")
        print(f"  项目: {wandb.run.project}")
        print(f"  Run名称: {wandb.run.name}")
        if mode == "online":
            print(f"  Run URL: {wandb.run.url}")
        else:
            print(f"  离线日志: wandb/offline-run-*")

    def _initialize_components(self):
        """初始化所有组件"""

        # 1. 数据管理器
        print("\n📂 初始化数据管理器...")
        self.data_manager = DataManager(
            data_dir=self.config['data_dir'],
            domain_ratios=self.config['domain_ratios']
        )
        self.data_manager.initialize()

        # 2. RL模型（Qwen2.5-7B + LoRA）
        print("\n🤖 加载RL模型...")
        self._load_rl_model()

        # 3. RL工作流生成器（共享已加载的模型）
        print("\n🔧 初始化工作流生成器...")
        self.generator = RLWorkflowGenerator(
            base_model=self.config['base_model'],  # 传递路径用于加载tokenizer
            device_ids=self.config['device_mapping'],
            operator_descriptions_path=self.config.get('aflow_operator_descriptions_path')
        )
        # 共享已加载的模型（避免重复加载）
        self.generator.model = self.model
        self.generator.tokenizer = self.tokenizer

        # 4. ExperienceBuffer - 高质量样本管理（需先初始化，用于后续组件）
        print("\n📚 初始化ExperienceBuffer...")
        experience_config = self.config.get('experience_buffer', {})
        self.experience_buffer = ExperienceBuffer(
            buffer_size=experience_config.get('buffer_size', 100),
            reward_threshold=experience_config.get('reward_threshold', 8.0),
            persistence_dir=experience_config.get('persistence_dir', 'data/experience_buffer'),
            problem_types=["math", "code", "qa"]
        )
        print(f"  Buffer大小: {self.experience_buffer.buffer_size}")
        print(f"  奖励阈值: {self.experience_buffer.reward_threshold}")

        # 5. PromptOptimizer - Layer 1动态提示词优化
        print("\n✨ 初始化PromptOptimizer (Layer 1)...")
        prompt_config = self.config.get('prompt_optimizer', {})
        self.prompt_optimizer = PromptOptimizer()
        self.use_dynamic_prompts = prompt_config.get('enabled', True)
        print(f"  动态提示词: {'启用' if self.use_dynamic_prompts else '禁用'}")

        # 6. OperatorPromptEnhancer - Layer 2 operator提示词增强
        print("\n🔧 初始化OperatorPromptEnhancer (Layer 2)...")
        operator_config = self.config.get('operator_prompt_enhancer', {})
        self.operator_enhancer = OperatorPromptEnhancer(
            enable_enhancement=operator_config.get('enabled', True)
        )
        print(f"  Operator增强: {'启用' if self.operator_enhancer.enable_enhancement else '禁用'}")

        # 7. AFlow执行器（传入operator_enhancer）
        print("\n⚙️  初始化AFlow执行器...")
        timeout = self.config.get('execution_timeout', 180)  # 默认180秒
        self.executor = AFlowExecutor(
            llm_config_path=self.config['aflow_config_path'],
            timeout=timeout,
            operator_enhancer=self.operator_enhancer  # 传递Layer 2增强器
        )
        print(f"  执行超时: {timeout}秒")

        # 8. 奖励计算器
        print("\n🎯 初始化奖励计算器...")
        self.reward_computer = RewardComputer(
            reward_weights=self.config.get('reward_weights'),
            use_llm_judge=True,  # 启用LLM Judge (GPT OSS 120B @ port 8002)
            llm_config={
                "base_url": "http://localhost:8002/v1",
                "api_key": "sk-dummy",
                "model_name": "/home/yijia/lhy/openai/gpt-oss-120b"
            }
        )

        # 9. 优化器
        print("\n🔬 初始化优化器...")
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 0.01)
        )

    def _load_rl_model(self):
        """加载RL模型（Qwen2.5-7B + LoRA）"""
        device = f"cuda:{self.config['device_mapping'][0]}"

        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['base_model'],
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载基座模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['base_model'],
            torch_dtype=torch.bfloat16 if self.config.get('bf16') else torch.float16,
            device_map={"": device},
            trust_remote_code=True
        )

        # 应用LoRA
        if self.config.get('use_lora', True):
            lora_config = LoraConfig(
                r=self.config['lora_rank'],
                lora_alpha=self.config['lora_alpha'],
                target_modules=self.config['lora_target_modules'].split(','),
                lora_dropout=self.config['lora_dropout'],
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.model = get_peft_model(self.model, lora_config)

            print(f"✅ LoRA应用完成")
            self.model.print_trainable_parameters()

    def get_current_temperature(self, step: int) -> float:
        """
        计算当前step的temperature

        策略: 线性从initial升至final
        - 早期: 低温度生成确定性workflow，建立baseline
        - 后期: 高温度探索多样性workflow

        Args:
            step: 当前训练步数

        Returns:
            当前的temperature值
        """
        if not self.temp_schedule['enabled']:
            return self.config['generation_config']['temperature']

        if step < self.temp_schedule['warmup_steps']:
            # Linear warmup
            progress = step / self.temp_schedule['warmup_steps']
            temp = (self.temp_schedule['initial'] +
                   progress * (self.temp_schedule['final'] - self.temp_schedule['initial']))
        else:
            temp = self.temp_schedule['final']

        return temp

    async def train_step(self, step: int) -> Dict:
        """
        单步GRPO训练（在线学习）

        Returns:
            metrics: 训练指标
        """

        # 1. 采样batch
        batch = self.data_manager.sample_batch(
            batch_size=self.config['rollout_batch_size'],
            split="train"
        )

        # 统计
        batch_stats = self.data_manager.get_batch_stats(batch)
        print(f"\n📦 Batch {step}: {len(batch)} 样本, 分布: {batch_stats}")

        # 获取当前temperature（动态调度）
        current_temp = self.get_current_temperature(step)
        print(f"🌡️  Temperature: {current_temp:.3f}")

        # 2. 为每个问题生成K个工作流（GRPO组）
        all_workflows = []
        all_problems = []
        all_answers = []
        all_rewards = []
        all_log_probs = []

        # ✨ 新增：准确率统计
        correctness_scores = []  # 存储所有正确性分数

        num_sequences = self.config['num_return_sequences_in_group']

        for sample_idx, sample in enumerate(tqdm(batch, desc="生成和执行工作流"), 1):
            problem = sample['problem']
            ground_truth = sample['ground_truth']
            problem_type = sample['problem_type']

            # GRPO组
            group_workflows = []
            group_answers = []
            group_rewards = []
            group_log_probs = []
            group_correctness = []

            for i in range(num_sequences):
                # 构建动态提示词（如果启用）
                custom_prompt = None
                if self.use_dynamic_prompts:
                    custom_prompt = self.prompt_optimizer.build_dynamic_prompt(
                        problem=problem,
                        problem_type=problem_type
                    )

                # 生成工作流
                result = self.generator.generate_workflow(
                    problem=problem,
                    problem_type=problem_type,
                    temperature=current_temp,  # 使用动态temperature
                    custom_prompt=custom_prompt
                )

                workflow_code = result['workflow_code']

                # 计算log概率（旧策略）
                log_prob = await self._compute_log_prob(problem, workflow_code, problem_type)

                # 执行工作流
                try:
                    answer, cost, metadata = await self.executor.execute_workflow(
                        workflow_code=workflow_code,
                        problem=problem,
                        problem_type=problem_type,
                        entry_point=sample.get('entry_point', ''),
                        test=sample.get('test', '')  # NEW: pass test cases for HumanEval
                    )

                    # 计算奖励
                    if metadata['success']:
                        reward = self.reward_computer.compute_reward(
                            problem=problem,
                            prediction=answer,
                            ground_truth=ground_truth,
                            problem_type=problem_type,
                            metadata=metadata
                        )

                        # ✨ 新增：显式计算并记录正确性
                        correctness = self.reward_computer._compute_correctness_reward(
                            prediction=answer,
                            ground_truth=ground_truth,
                            problem_type=problem_type
                        )
                        correctness_scores.append(correctness)
                        group_correctness.append(correctness)

                        # 判断是否正确（correctness > 5.0 表示接近正确或完全正确）
                        is_correct = correctness >= 5.0
                        status_icon = "✅" if is_correct else "❌"

                        # 实时日志到 wandb (样本级别)
                        wandb.log({
                            f"sample/{problem_type}/correctness": correctness,
                            f"sample/{problem_type}/reward": reward,
                            f"sample/step": step,
                            f"sample/sample_id": sample_idx * 4 + i,
                        })

                        print(f"  {status_icon} 正确性评分: {correctness:.1f}/10.0 | 预测: {str(answer)[:50]} | 真值: {str(ground_truth)[:50]}")
                    else:
                        reward = -10.0  # 执行失败惩罚
                        correctness_scores.append(-10.0)
                        group_correctness.append(-10.0)
                        print(f"  ❌ 执行失败 | 真值: {str(ground_truth)[:50]}")

                except Exception as e:
                    print(f"  ⚠️  执行错误: {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()
                    answer = None
                    reward = -10.0
                    correctness_scores.append(-10.0)
                    group_correctness.append(-10.0)

                group_workflows.append(workflow_code)
                group_answers.append(answer)
                group_rewards.append(reward)
                group_log_probs.append(log_prob)

            # GRPO关键：组内奖励归一化
            mean_reward = np.mean(group_rewards)
            group_advantages = [r - mean_reward for r in group_rewards]

            # 💾 收集高质量样本到ExperienceBuffer
            for idx, (workflow, answer, reward) in enumerate(zip(group_workflows, group_answers, group_rewards)):
                # 只收集原始奖励高的样本（非advantage）
                if reward >= self.experience_buffer.reward_threshold:
                    sample = {
                        'problem': problem,
                        'workflow_code': workflow,
                        'answer': answer,
                        'ground_truth': ground_truth,
                        'reward': reward,
                        'correctness_score': correctness_scores[-len(group_rewards) + idx] if correctness_scores else 0,
                        'metadata': {
                            'problem_type': problem_type,
                            'step': step
                        },
                        'step': step
                    }
                    self.experience_buffer.add_sample(sample, problem_type)

            # 收集
            all_workflows.extend(group_workflows)
            all_problems.extend([problem] * num_sequences)
            all_answers.extend(group_answers)
            all_rewards.extend(group_advantages)  # 使用优势
            all_log_probs.extend(group_log_probs)

        # 3. 策略梯度更新
        print(f"\n🔄 更新策略...")
        loss, kl_div = await self._update_policy(
            problems=all_problems,
            workflows=all_workflows,
            old_log_probs=all_log_probs,
            advantages=all_rewards,
            problem_types=[s['problem_type'] for s in batch for _ in range(num_sequences)]
        )

        # 4. 指标
        # ✨ 新增：计算准确率统计
        num_correct = sum(1 for score in correctness_scores if score >= 5.0)
        num_total = len(correctness_scores)
        accuracy = (num_correct / num_total * 100) if num_total > 0 else 0.0
        avg_correctness = np.mean(correctness_scores) if correctness_scores else 0.0

        # 计算问题类型分布的准确率
        problem_type_stats = {}
        for problem_type in ['math', 'code', 'qa']:
            type_scores = [s for s, p in zip(correctness_scores,
                          [s['problem_type'] for s in batch for _ in range(num_sequences)])
                          if p == problem_type]
            if type_scores:
                type_correct = sum(1 for s in type_scores if s >= 5.0)
                type_accuracy = (type_correct / len(type_scores) * 100)
                type_avg = np.mean(type_scores)
                problem_type_stats[problem_type] = {
                    "accuracy": type_accuracy,
                    "avg_score": type_avg,
                    "count": len(type_scores)
                }

        metrics = {
            "step": step,
            "loss": loss,
            "kl_div": kl_div,
            "avg_reward": np.mean(all_rewards),
            "max_reward": np.max(all_rewards),
            "min_reward": np.min(all_rewards),
            "num_samples": len(all_workflows),
            # ✨ 新增准确率指标
            "accuracy": accuracy,
            "num_correct": num_correct,
            "num_total": num_total,
            "avg_correctness_score": avg_correctness
        }

        print(f"\n🎯 准确率统计: {num_correct}/{num_total} = {accuracy:.1f}% (平均正确性评分: {avg_correctness:.2f}/10.0)")
        print(f"\n📊 问题类型分布:")
        for ptype, stats in problem_type_stats.items():
            print(f"  {ptype}: {stats['accuracy']:.1f}% (avg: {stats['avg_score']:.2f}, n={stats['count']})")

        # ✨ 详细 wandb logging (实时仪表板)
        wandb_log_data = {
            "train/loss": loss,
            "train/kl_div": kl_div,
            "train/avg_reward": np.mean(all_rewards),
            "train/max_reward": np.max(all_rewards),
            "train/min_reward": np.min(all_rewards),
            "train/accuracy": accuracy,
            "train/avg_correctness_score": avg_correctness,
            "train/num_correct": num_correct,
            "train/num_total": num_total,
            "train/temperature": current_temp,  # 记录当前temperature
            "train/step": step,
        }

        # 添加问题类型的分布指标
        for ptype, stats in problem_type_stats.items():
            wandb_log_data[f"train/accuracy_{ptype}"] = stats['accuracy']
            wandb_log_data[f"train/avg_score_{ptype}"] = stats['avg_score']
            wandb_log_data[f"train/count_{ptype}"] = stats['count']

        wandb.log(wandb_log_data, step=step)

        return metrics

    async def _compute_log_prob(
        self,
        problem: str,
        workflow_code: str,
        problem_type: str
    ) -> torch.Tensor:
        """计算工作流的log概率（旧策略）"""

        self.model.eval()

        with torch.no_grad():
            # 构建完整文本
            prompt = self.generator._build_generation_prompt(problem, problem_type)
            full_text = prompt + workflow_code

            # Tokenize
            inputs = self.tokenizer(full_text, return_tensors="pt").to(self.model.device)

            # 前向传播
            outputs = self.model(**inputs, labels=inputs["input_ids"])

            # 负对数似然 -> log概率
            log_prob = -outputs.loss

            return log_prob.detach().cpu()

    async def _update_policy(
        self,
        problems: List[str],
        workflows: List[str],
        old_log_probs: List[torch.Tensor],
        advantages: List[float],
        problem_types: List[str]
    ) -> Tuple[float, float]:
        """更新策略（GRPO）"""

        self.model.train()

        total_loss = 0.0
        total_kl = 0.0
        num_updates = 0

        # 梯度累积
        grad_accum_steps = self.config.get('gradient_accumulation_steps', 1)

        for i in range(0, len(workflows), grad_accum_steps):
            batch_slice = slice(i, min(i + grad_accum_steps, len(workflows)))

            batch_loss = 0.0
            batch_kl = 0.0

            for j in range(i, min(i + grad_accum_steps, len(workflows))):
                problem = problems[j]
                workflow = workflows[j]
                old_log_prob = old_log_probs[j]
                advantage = advantages[j]
                problem_type = problem_types[j]

                # 计算新log概率
                new_log_prob = await self._compute_log_prob_trainable(problem, workflow, problem_type)

                # 重要性采样比
                ratio = torch.exp(new_log_prob - old_log_prob.to(self.model.device))

                # PPO裁剪损失
                clip_range = self.config['clip_range']
                clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)

                advantage_tensor = torch.tensor(advantage, device=self.model.device)
                policy_loss = -torch.min(
                    ratio * advantage_tensor,
                    clipped_ratio * advantage_tensor
                )

                # KL正则化
                if self.config.get('use_kl_loss'):
                    kl_loss = self.config['kl_loss_coef'] * (new_log_prob - old_log_prob.to(self.model.device)).pow(2)
                else:
                    kl_loss = 0.0

                # 总损失
                loss = policy_loss + kl_loss

                # 累积
                batch_loss += loss
                batch_kl += kl_loss if isinstance(kl_loss, torch.Tensor) else 0.0

            # 平均
            batch_loss = batch_loss / min(grad_accum_steps, len(workflows) - i)

            # 反向传播
            batch_loss.backward()

            total_loss += batch_loss.item()
            total_kl += batch_kl.item() if isinstance(batch_kl, torch.Tensor) else batch_kl
            num_updates += 1

            # 优化器步骤
            if (i + grad_accum_steps) % grad_accum_steps == 0:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('max_grad_norm', 1.0))
                self.optimizer.step()
                self.optimizer.zero_grad()

        avg_loss = total_loss / max(num_updates, 1)
        avg_kl = total_kl / max(num_updates, 1)

        return avg_loss, avg_kl

    async def _compute_log_prob_trainable(
        self,
        problem: str,
        workflow_code: str,
        problem_type: str
    ) -> torch.Tensor:
        """计算工作流的log概率（新策略，可训练）"""

        # 构建完整文本
        prompt = self.generator._build_generation_prompt(problem, problem_type)
        full_text = prompt + workflow_code

        # Tokenize
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.model.device)

        # 前向传播
        outputs = self.model(**inputs, labels=inputs["input_ids"])

        # 负对数似然 -> log概率
        log_prob = -outputs.loss

        return log_prob

    async def evaluate_on_val_set(self, num_samples: int = 50) -> Dict:
        """
        在验证集上评估模型性能

        Args:
            num_samples: 验证样本数量

        Returns:
            验证指标字典
        """
        print(f"\n{'='*60}")
        print(f"🧪 验证集评估 ({num_samples}个样本)")
        print(f"{'='*60}")

        # 采样验证集
        val_batch = self.data_manager.sample_batch(
            batch_size=num_samples,
            split="val"  # 使用验证集
        )

        # 统计
        batch_stats = self.data_manager.get_batch_stats(val_batch)
        print(f"📦 验证集分布: {batch_stats}")

        # 评估每个样本
        correctness_scores = []
        total_cost = 0.0
        successful_executions = 0

        for idx, sample in enumerate(tqdm(val_batch, desc="验证集评估"), 1):
            problem = sample['problem']
            ground_truth = sample['ground_truth']
            problem_type = sample['problem_type']

            try:
                # 使用当前策略生成workflow（使用动态提示词）
                custom_prompt = None
                if self.use_dynamic_prompts:
                    custom_prompt = self.prompt_optimizer.build_dynamic_prompt(
                        problem=problem,
                        problem_type=problem_type
                    )

                result = self.generator.generate_workflow(
                    problem=problem,
                    problem_type=problem_type,
                    temperature=self.config['generation_config']['temperature'],
                    custom_prompt=custom_prompt
                )

                workflow_code = result['workflow_code']

                # 执行workflow
                answer, cost, metadata = await self.executor.execute_workflow(
                    workflow_code=workflow_code,
                    problem=problem,
                    problem_type=problem_type,
                    entry_point=sample.get('entry_point', ''),
                    test=sample.get('test', '')  # NEW: pass test cases for HumanEval
                )

                # 计算正确性
                if metadata['success']:
                    correctness = self.reward_computer._compute_correctness_reward(
                        prediction=answer,
                        ground_truth=ground_truth,
                        problem_type=problem_type
                    )
                    correctness_scores.append(correctness)
                    total_cost += cost
                    successful_executions += 1

                    is_correct = correctness >= 5.0
                    status_icon = "✅" if is_correct else "❌"
                    if idx <= 5:  # 只打印前5个样本的详情
                        print(f"  {status_icon} [{idx}/{num_samples}] 正确性: {correctness:.1f}/10.0")
                else:
                    correctness_scores.append(0.0)
                    if idx <= 5:
                        print(f"  ❌ [{idx}/{num_samples}] 执行失败")

            except Exception as e:
                print(f"  ⚠️  [{idx}/{num_samples}] 错误: {type(e).__name__}")
                correctness_scores.append(0.0)

        # 计算指标
        num_correct = sum(1 for score in correctness_scores if score >= 5.0)
        val_accuracy = (num_correct / num_samples * 100) if num_samples > 0 else 0.0
        avg_correctness = np.mean(correctness_scores) if correctness_scores else 0.0
        avg_cost = total_cost / successful_executions if successful_executions > 0 else 0.0
        success_rate = (successful_executions / num_samples * 100) if num_samples > 0 else 0.0

        metrics = {
            "val_accuracy": val_accuracy,
            "val_num_correct": num_correct,
            "val_num_total": num_samples,
            "val_avg_correctness": avg_correctness,
            "val_avg_cost": avg_cost,
            "val_success_rate": success_rate
        }

        print(f"\n📊 验证集结果:")
        print(f"  准确率: {num_correct}/{num_samples} = {val_accuracy:.1f}%")
        print(f"  平均正确性: {avg_correctness:.2f}/10.0")
        print(f"  执行成功率: {success_rate:.1f}%")
        print(f"  平均成本: ${avg_cost:.4f}")
        print(f"{'='*60}\n")

        return metrics

    async def train(self):
        """完整训练循环"""
        print("\n" + "=" * 60)
        print("🎓 开始GRPO训练")
        print("=" * 60)

        max_steps = self.config['max_steps']
        save_every = self.config.get('save_every', 50)
        log_every = self.config.get('log_every', 5)
        eval_every = self.config.get('eval_every', 10)  # 每10步验证一次
        val_samples = self.config.get('val_samples', 50)  # 验证集样本数

        for step in range(1, max_steps + 1):
            print(f"\n{'=' * 60}")
            print(f"📍 Step {step}/{max_steps}")
            print(f"{'=' * 60}")

            # 训练步骤
            metrics = await self.train_step(step)

            # 日志
            if step % log_every == 0:
                print(f"\n📊 Metrics:")
                for key, value in metrics.items():
                    print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

                # 记录到wandb
                wandb.log(metrics, step=step)

            # 🧪 验证集评估（每N步）
            if step % eval_every == 0:
                val_metrics = await self.evaluate_on_val_set(num_samples=val_samples)

                # 合并验证指标到训练指标
                metrics.update(val_metrics)

                # 记录验证指标到wandb
                wandb.log(val_metrics, step=step)

                print(f"✅ 验证集评估完成 (Step {step})")

            # 保存检查点
            if step % save_every == 0:
                self.save_checkpoint(step)

        print("\n" + "=" * 60)
        print("✅ 训练完成")
        print("=" * 60)

    def save_checkpoint(self, step: int):
        """保存检查点"""
        checkpoint_dir = Path(self.config['output_dir']) / f"step_{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 保存LoRA权重
        self.model.save_pretrained(checkpoint_dir)

        # 💾 保存ExperienceBuffer
        self.experience_buffer.save(step=step)

        # 📊 打印ExperienceBuffer统计信息
        buffer_stats = self.experience_buffer.get_stats()
        print(f"\n📚 ExperienceBuffer统计:")
        for problem_type, stats in buffer_stats.items():
            if stats['count'] > 0:
                print(f"  {problem_type}: {stats['count']}样本, "
                      f"平均奖励={stats['avg_reward']:.2f}, "
                      f"最高奖励={stats['max_reward']:.2f}, "
                      f"平均正确性={stats['avg_correctness']:.2f}")

        print(f"💾 检查点已保存: {checkpoint_dir}")


async def main():
    """主函数"""
    trainer = GRPOTrainer(config_path="config/training.yaml")
    await trainer.train()


if __name__ == "__main__":
    asyncio.run(main())
