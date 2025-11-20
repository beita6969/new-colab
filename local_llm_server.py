#!/usr/bin/env python3
"""
轻量级本地LLM服务器 - OpenAI API兼容
使用HuggingFace transformers直接推理，替代gpt-4o-mini加速训练

特性:
- OpenAI API兼容接口
- Flash Attention 2加速（如果可用）
- 批处理优化
- 零网络延迟

用法:
    python local_llm_server.py --port 8000 --gpu 2
"""

import argparse
import asyncio
import time
import uuid
from typing import List, Optional, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# API模型定义（OpenAI兼容）
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "qwen2.5-7b-local"
    messages: List[Message]
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 1.0
    n: int = 1
    stream: bool = False

class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 1699000000
    owned_by: str = "local"

class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


class LocalLLMServer:
    """本地LLM服务器"""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        max_length: int = 4096,
        use_flash_attention: bool = True
    ):
        """
        初始化本地LLM服务器

        Args:
            model_path: 模型路径
            device: 设备（cuda:0, cuda:1等）
            max_length: 最大序列长度
            use_flash_attention: 是否使用Flash Attention 2
        """
        self.model_path = model_path
        self.device = device
        self.max_length = max_length

        print(f"🚀 初始化本地LLM服务器...")
        print(f"  模型: {model_path}")
        print(f"  设备: {device}")
        print(f"  最大长度: {max_length}")

        # 加载tokenizer
        print("📥 加载tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        # 模型加载配置
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": device,
            "trust_remote_code": True,
        }

        # 尝试使用Flash Attention 2
        if use_flash_attention:
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                print("  尝试启用Flash Attention 2...")
            except Exception as e:
                print(f"  Flash Attention 2不可用: {e}")
                print("  使用标准attention")

        # 加载模型
        print("📥 加载模型...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs
        )
        self.model.eval()  # 推理模式

        print(f"✅ 模型加载完成")
        print(f"  参数量: {self.model.num_parameters() / 1e9:.2f}B")
        print(f"  内存占用: ~{self.model.get_memory_footprint() / 1e9:.2f}GB")

    def format_messages(self, messages: List[Message]) -> str:
        """
        格式化消息为Qwen2.5格式

        Args:
            messages: 消息列表

        Returns:
            格式化后的提示词
        """
        # Qwen2.5使用ChatML格式
        formatted = ""
        for msg in messages:
            role = msg.role
            content = msg.content
            if role == "system":
                formatted += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                formatted += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                formatted += f"<|im_start|>assistant\n{content}<|im_end|>\n"

        # 添加assistant开始标记
        formatted += "<|im_start|>assistant\n"
        return formatted

    @torch.inference_mode()
    def generate(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 1.0
    ) -> Dict[str, Any]:
        """
        生成回复

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大生成token数
            top_p: nucleus sampling参数

        Returns:
            包含生成文本和token统计的字典
        """
        # 格式化输入
        prompt = self.format_messages(messages)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        ).to(self.device)

        prompt_tokens = inputs.input_ids.shape[1]

        # 生成配置
        gen_config = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        # 生成
        start_time = time.time()
        outputs = self.model.generate(
            **inputs,
            **gen_config
        )
        generation_time = time.time() - start_time

        # 解码
        generated_ids = outputs[0][prompt_tokens:]  # 只保留新生成的部分
        generated_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        )

        completion_tokens = len(generated_ids)
        total_tokens = prompt_tokens + completion_tokens

        # 计算速度
        tokens_per_sec = completion_tokens / generation_time if generation_time > 0 else 0

        print(f"  生成: {completion_tokens} tokens @ {tokens_per_sec:.1f} tok/s")

        return {
            "text": generated_text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "generation_time": generation_time
        }


# 全局服务器实例
server: Optional[LocalLLMServer] = None


# FastAPI应用
app = FastAPI(
    title="Local LLM Server",
    description="OpenAI API兼容的本地LLM服务器",
    version="1.0.0"
)


@app.on_event("startup")
async def startup_event():
    """启动时初始化模型"""
    global server

    # 从命令行参数获取配置
    model_path = app.state.model_path
    device = app.state.device

    server = LocalLLMServer(
        model_path=model_path,
        device=device,
        max_length=4096,
        use_flash_attention=True
    )

    print("\n" + "="*60)
    print("✅ 本地LLM服务器就绪")
    print("="*60)
    print(f"  Base URL: http://127.0.0.1:{app.state.port}/v1")
    print(f"  健康检查: http://127.0.0.1:{app.state.port}/health")
    print("="*60 + "\n")


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "ok", "model_loaded": server is not None}


@app.get("/v1/models")
async def list_models():
    """列出可用模型"""
    return ModelsResponse(
        data=[
            ModelInfo(
                id="qwen2.5-7b-local",
                owned_by="local"
            ),
            ModelInfo(
                id="gpt-4o-mini",  # 兼容旧配置
                owned_by="local"
            )
        ]
    )


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """创建聊天补全（OpenAI API兼容）"""
    if server is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # 生成回复
        result = server.generate(
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p
        )

        # 构造OpenAI兼容响应
        response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=Message(
                        role="assistant",
                        content=result["text"]
                    ),
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=result["prompt_tokens"],
                completion_tokens=result["completion_tokens"],
                total_tokens=result["total_tokens"]
            )
        )

        return response

    except Exception as e:
        print(f"❌ 生成失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="本地LLM服务器")
    parser.add_argument(
        "--model",
        type=str,
        default="/home/yijia/verl-agent/models/qwen/Qwen2___5-7B-Instruct",
        help="模型路径"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=2,
        help="使用的GPU编号"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="服务器端口"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="服务器host"
    )

    args = parser.parse_args()

    # 设置CUDA设备
    device = f"cuda:{args.gpu}"

    # 存储配置到app state
    app.state.model_path = args.model
    app.state.device = device
    app.state.port = args.port

    # 启动服务器
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )


if __name__ == "__main__":
    main()
