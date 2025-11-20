#!/usr/bin/env python3
"""
GPT-OSS-20B 本地服务器 - OpenAI API兼容
使用llama-cpp-python加载GGUF格式模型

特性:
- OpenAI API兼容接口
- GGUF格式优化加载
- GPU加速 (CUDA)
- 零网络延迟
- 预期速度: 25-30 tokens/s (2x faster than Qwen2.5-7B)

用法:
    python gptoss_20b_server.py --port 8000 --gpu 2
"""

import argparse
import time
import uuid
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# API模型定义（OpenAI兼容）
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "gpt-oss-20b"
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
    owned_by: str = "openai"

class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


class GPTOSSServer:
    """GPT-OSS-20B服务器 (GGUF)"""

    def __init__(
        self,
        model_path: str,
        n_gpu_layers: int = -1,  # -1 = 全部offload到GPU
        n_ctx: int = 4096,
        n_batch: int = 512,
        verbose: bool = False
    ):
        """
        初始化GPT-OSS-20B服务器

        Args:
            model_path: GGUF模型文件路径
            n_gpu_layers: GPU层数 (-1表示全部)
            n_ctx: 上下文长度
            n_batch: 批处理大小
            verbose: 是否详细日志
        """
        self.model_path = model_path

        print(f"🚀 初始化GPT-OSS-20B服务器...")
        print(f"  模型: {model_path}")
        print(f"  GPU层数: {n_gpu_layers} (全部offload到GPU)")
        print(f"  上下文长度: {n_ctx}")
        print(f"  批处理: {n_batch}")

        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "需要安装 llama-cpp-python:\n"
                "CMAKE_ARGS=\"-DGGML_CUDA=on\" pip install llama-cpp-python"
            )

        # 加载GGUF模型
        print("📥 加载GGUF模型 (GPU加速)...")
        start_time = time.time()

        self.model = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,  # 全部offload到GPU
            n_ctx=n_ctx,
            n_batch=n_batch,
            verbose=verbose,
            logits_all=False,  # 节省内存
            use_mmap=True,  # 使用内存映射
            use_mlock=False,  # 不锁定内存（允许swap）
        )

        load_time = time.time() - start_time

        print(f"✅ 模型加载完成 ({load_time:.1f}秒)")
        print(f"  架构: Mixture-of-Experts (21B total, 3.6B active)")
        print(f"  量化: MXFP4 (~12GB)")
        print(f"  预期速度: 25-30 tokens/s")

    def format_messages(self, messages: List[Message]) -> str:
        """
        格式化消息为GPT-OSS格式 (Harmony response format)

        GPT-OSS使用类似ChatML的格式
        """
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

        # 生成
        start_time = time.time()
        output = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            echo=False,  # 不回显输入
            stop=["<|im_end|>", "<|endoftext|>"],
        )
        generation_time = time.time() - start_time

        # 提取结果
        generated_text = output["choices"][0]["text"]
        prompt_tokens = output["usage"]["prompt_tokens"]
        completion_tokens = output["usage"]["completion_tokens"]
        total_tokens = output["usage"]["total_tokens"]

        # 计算速度
        tokens_per_sec = completion_tokens / generation_time if generation_time > 0 else 0

        print(f"  生成: {completion_tokens} tokens @ {tokens_per_sec:.1f} tok/s")

        return {
            "text": generated_text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "generation_time": generation_time,
            "tokens_per_sec": tokens_per_sec
        }


# 全局服务器实例
server: Optional[GPTOSSServer] = None


# FastAPI应用
app = FastAPI(
    title="GPT-OSS-20B Server",
    description="OpenAI API兼容的GPT-OSS-20B本地服务器",
    version="1.0.0"
)


@app.on_event("startup")
async def startup_event():
    """启动时初始化模型"""
    global server

    # 从命令行参数获取配置
    model_path = app.state.model_path
    n_gpu_layers = app.state.n_gpu_layers

    server = GPTOSSServer(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        n_ctx=4096,
        n_batch=512,
        verbose=False
    )

    print("\n" + "="*60)
    print("✅ GPT-OSS-20B服务器就绪")
    print("="*60)
    print(f"  Base URL: http://127.0.0.1:{app.state.port}/v1")
    print(f"  健康检查: http://127.0.0.1:{app.state.port}/health")
    print(f"  模型: GPT-OSS-20B (21B params, 3.6B active)")
    print(f"  预期性能: 25-30 tokens/s (2x faster than Qwen2.5-7B)")
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
                id="gpt-oss-20b",
                owned_by="openai"
            ),
            ModelInfo(
                id="gpt-4o-mini",  # 兼容旧配置
                owned_by="openai"
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
    parser = argparse.ArgumentParser(description="GPT-OSS-20B本地服务器")
    parser.add_argument(
        "--model",
        type=str,
        default="models/gpt-oss-20b-gguf/openai_gpt-oss-20b-MXFP4.gguf",
        help="GGUF模型文件路径"
    )
    parser.add_argument(
        "--gpu-layers",
        type=int,
        default=-1,
        help="GPU层数 (-1表示全部offload到GPU)"
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

    # 存储配置到app state
    app.state.model_path = args.model
    app.state.n_gpu_layers = args.gpu_layers
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
