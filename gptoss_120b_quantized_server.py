#!/usr/bin/env python3
"""
GPT-OSS-120B 量化服务器 - OpenAI API兼容
使用bitsandbytes 4-bit量化，显存占用~15-20GB

特性:
- 4-bit量化 (61GB → ~15-20GB)
- OpenAI API兼容
- GPU加速
- 预期速度: 30-40 tokens/s (MoE架构，只激活5.1B参数)

用法:
    python gptoss_120b_quantized_server.py --port 8000 --gpu 0
"""

import argparse
import time
import uuid
from typing import List, Optional, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# API模型定义（OpenAI兼容）
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "gpt-oss-120b"
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


class GPTOSS120BServer:
    """GPT-OSS-120B 4-bit量化服务器"""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        max_length: int = 4096,
        quantization_bits: int = 4
    ):
        """
        初始化GPT-OSS-120B量化服务器

        Args:
            model_path: 模型路径
            device: 设备
            max_length: 最大序列长度
            quantization_bits: 量化位数 (4 or 8)
        """
        self.model_path = model_path
        self.device = device
        self.max_length = max_length

        print(f"🚀 初始化GPT-OSS-120B量化服务器...")
        print(f"  模型: {model_path}")
        print(f"  量化: {quantization_bits}-bit (bitsandbytes)")
        print(f"  设备: {device}")
        print(f"  预期显存: ~15-20GB (vs 原始61GB)")
        print(f"  预期速度: 30-40 tokens/s (MoE 5.1B active)")

        # 配置4-bit量化
        if quantization_bits == 4:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,  # 双重量化，进一步节省内存
                bnb_4bit_quant_type="nf4"  # NF4量化类型（推荐）
            )
        elif quantization_bits == 8:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
        else:
            raise ValueError(f"Unsupported quantization_bits: {quantization_bits}")

        # 加载tokenizer (使用slow tokenizer避免fast tokenizer错误)
        print("📥 加载tokenizer (slow tokenizer)...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False  # 使用slow tokenizer避免fast tokenizer损坏问题
        )

        # 加载量化模型
        print(f"📥 加载{quantization_bits}-bit量化模型 (这可能需要2-3分钟)...")
        start_time = time.time()

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map=device,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        self.model.eval()

        load_time = time.time() - start_time

        print(f"✅ 模型加载完成 ({load_time:.1f}秒)")
        print(f"  架构: Mixture-of-Experts (117B total, 5.1B active)")
        print(f"  量化: {quantization_bits}-bit NF4")

        # 获取实际显存占用
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(device) / 1e9
            print(f"  实际显存: {memory_allocated:.2f}GB")

    def format_messages(self, messages: List[Message]) -> str:
        """
        格式化消息为GPT-OSS格式 (Harmony response format)
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

    @torch.inference_mode()
    def generate(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 1.0
    ) -> Dict[str, Any]:
        """生成回复"""
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
        generated_ids = outputs[0][prompt_tokens:]
        generated_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        )

        completion_tokens = len(generated_ids)
        total_tokens = prompt_tokens + completion_tokens
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
server: Optional[GPTOSS120BServer] = None

# FastAPI应用
app = FastAPI(
    title="GPT-OSS-120B Quantized Server",
    description="OpenAI API兼容的GPT-OSS-120B 4-bit量化服务器",
    version="1.0.0"
)


@app.on_event("startup")
async def startup_event():
    """启动时初始化模型"""
    global server

    model_path = app.state.model_path
    device = app.state.device
    quantization_bits = app.state.quantization_bits

    server = GPTOSS120BServer(
        model_path=model_path,
        device=device,
        max_length=4096,
        quantization_bits=quantization_bits
    )

    print("\n" + "="*60)
    print("✅ GPT-OSS-120B量化服务器就绪")
    print("="*60)
    print(f"  Base URL: http://127.0.0.1:{app.state.port}/v1")
    print(f"  健康检查: http://127.0.0.1:{app.state.port}/health")
    print(f"  模型: GPT-OSS-120B (117B params, 5.1B active)")
    print(f"  量化: {quantization_bits}-bit (bitsandbytes)")
    print(f"  预期性能: 30-40 tokens/s")
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
                id="gpt-oss-120b",
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
        result = server.generate(
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p
        )

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
    parser = argparse.ArgumentParser(description="GPT-OSS-120B量化服务器")
    parser.add_argument(
        "--model",
        type=str,
        default="/home/yijia/lhy/openai/gpt-oss-120b",
        help="模型路径"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="使用的GPU编号（逻辑编号）"
    )
    parser.add_argument(
        "--quantization-bits",
        type=int,
        default=4,
        choices=[4, 8],
        help="量化位数 (4 or 8)"
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

    device = f"cuda:{args.gpu}"

    # 存储配置到app state
    app.state.model_path = args.model
    app.state.device = device
    app.state.quantization_bits = args.quantization_bits
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
