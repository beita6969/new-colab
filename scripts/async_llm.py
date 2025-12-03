#!/usr/bin/env python3
"""
兼容AFlow的async_llm模块
提供与AFlow/scripts/async_llm.py相同的接口
包含Formatter支持的call_with_format()方法
"""
import os
import asyncio
import yaml
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from dataclasses import dataclass, field
from openai import AsyncOpenAI
import httpx

if TYPE_CHECKING:
    from scripts.formatter import BaseFormatter


@dataclass
class LLMConfig:
    """单个LLM配置"""
    api_type: str = "openai"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 4096

    def __post_init__(self):
        # 支持从环境变量读取API key
        if self.api_key == "${OPENAI_API_KEY}" or not self.api_key:
            self.api_key = os.environ.get('OPENAI_API_KEY', 'sk-dummy')


@dataclass
class LLMsConfig:
    """多个LLM配置集合"""
    models: Dict[str, LLMConfig] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'LLMsConfig':
        """从YAML文件加载配置"""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        models = {}
        for name, config in data.get('models', {}).items():
            # 处理环境变量
            api_key = config.get('api_key', '')
            if api_key == "${OPENAI_API_KEY}":
                api_key = os.environ.get('OPENAI_API_KEY', '')

            models[name] = LLMConfig(
                api_type=config.get('api_type', 'openai'),
                base_url=config.get('base_url', 'https://api.openai.com/v1'),
                api_key=api_key,
                model_name=config.get('model_name', name),
                temperature=config.get('temperature', 0.7),
                top_p=config.get('top_p', 1.0),
                max_tokens=config.get('max_tokens', 4096)
            )

        return cls(models=models)


class AsyncLLM:
    """异步LLM客户端 - 兼容AFlow接口"""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 4096
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

        # 创建OpenAI客户端
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=300.0
        )

        # 统计
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0

    async def __call__(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """调用LLM生成响应"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                top_p=self.top_p,
                max_tokens=max_tokens or self.max_tokens
            )

            # 统计tokens
            if response.usage:
                self.total_input_tokens += response.usage.prompt_tokens
                self.total_output_tokens += response.usage.completion_tokens
            self.total_calls += 1

            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"LLM调用失败: {e}")
            raise

    async def batch_call(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        max_concurrent: int = 10
    ) -> List[tuple]:
        """批量调用"""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def call_with_semaphore(prompt: str):
            async with semaphore:
                try:
                    result = await self(prompt, system_prompt)
                    return (True, result)
                except Exception as e:
                    return (False, str(e))

        tasks = [call_with_semaphore(p) for p in prompts]
        return await asyncio.gather(*tasks)

    def get_usage_summary(self) -> Dict[str, Any]:
        """获取使用统计"""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_calls": self.total_calls,
            "total_cost": 0.0  # 简化：不计算成本
        }

    async def call_with_format(
        self,
        prompt: str,
        formatter: "BaseFormatter",
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        使用Formatter调用LLM - AFlow兼容接口

        Args:
            prompt: 原始提示词
            formatter: Formatter实例（XmlFormatter/CodeFormatter/TextFormatter）
            system_prompt: 可选的系统提示词
            temperature: 可选的温度参数
            max_tokens: 可选的最大token数

        Returns:
            解析后的响应字典

        Raises:
            FormatError: 当响应格式验证失败时
        """
        from scripts.formatter import FormatError

        # 使用formatter准备带格式要求的提示词
        formatted_prompt = formatter.prepare_prompt(prompt)

        # 调用LLM获取响应
        response = await self(
            formatted_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # 使用formatter验证和解析响应
        is_valid, parsed = formatter.validate_response(response)

        if not is_valid:
            raise FormatError(formatter.format_error_message())

        return parsed


def create_llm_instance(
    config: Dict[str, Any],
    **kwargs
) -> AsyncLLM:
    """
    创建LLM实例的工厂函数 - 兼容AFlow接口

    Args:
        config: LLM配置字典或LLMConfig对象

    Returns:
        AsyncLLM实例
    """
    if isinstance(config, LLMConfig):
        return AsyncLLM(
            api_key=config.api_key,
            base_url=config.base_url,
            model=config.model_name,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens
        )

    # 处理字符串配置（模型名称）- 使用OpenAI API
    if isinstance(config, str):
        api_key = os.environ.get('OPENAI_API_KEY', 'sk-dummy')
        # 映射模型名称到实际的OpenAI模型
        model_mapping = {
            'gpt-oss-120b': 'gpt-4o-mini',  # 映射到gpt-4o-mini
            'gpt-4o': 'gpt-4o',
            'gpt-4o-mini': 'gpt-4o-mini',
            'gpt-3.5-turbo': 'gpt-3.5-turbo',
        }
        model_name = model_mapping.get(config, 'gpt-4o-mini')
        return AsyncLLM(
            api_key=api_key,
            base_url='https://api.openai.com/v1',
            model=model_name,
            temperature=0.7,
            top_p=1.0,
            max_tokens=4096
        )

    # 处理字典配置
    api_key = config.get('api_key', '')
    if api_key == "${OPENAI_API_KEY}" or not api_key:
        api_key = os.environ.get('OPENAI_API_KEY', 'sk-dummy')

    return AsyncLLM(
        api_key=api_key,
        base_url=config.get('base_url', 'https://api.openai.com/v1'),
        model=config.get('model_name', config.get('model', 'gpt-4o-mini')),
        temperature=config.get('temperature', 0.7),
        top_p=config.get('top_p', 1.0),
        max_tokens=config.get('max_tokens', 4096)
    )


# 测试
if __name__ == "__main__":
    async def test():
        config = {
            "api_key": os.environ.get('OPENAI_API_KEY'),
            "base_url": "https://api.openai.com/v1",
            "model_name": "gpt-4o-mini"
        }
        llm = create_llm_instance(config)
        result = await llm("Say hello!")
        print(f"Result: {result}")

    asyncio.run(test())
