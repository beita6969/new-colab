#!/usr/bin/env python3
"""测试本地LLM API速度"""

import time
import requests
import json

API_URL = "http://127.0.0.1:8000/v1/chat/completions"

# 测试请求（类似operator调用）
test_request = {
    "model": "qwen2.5-7b-local",
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful math tutor."
        },
        {
            "role": "user",
            "content": "Solve: What is 25 * 4?"
        }
    ],
    "temperature": 0,
    "max_tokens": 512
}

print("🧪 测试本地LLM API速度\n")
print("="*60)

# 热身（第一次调用通常较慢）
print("🔥 热身调用...")
response = requests.post(API_URL, json=test_request, timeout=60)
result = response.json()
print(f"  结果: {result['choices'][0]['message']['content'][:100]}...")
print()

# 正式测试（3次）
times = []
for i in range(3):
    print(f"⏱️  测试 {i+1}/3...")
    start = time.time()
    response = requests.post(API_URL, json=test_request, timeout=60)
    elapsed = time.time() - start

    result = response.json()
    tokens = result['usage']['completion_tokens']
    tokens_per_sec = tokens / elapsed

    times.append(elapsed)

    print(f"  耗时: {elapsed:.2f}秒")
    print(f"  Token数: {tokens}")
    print(f"  速度: {tokens_per_sec:.1f} tokens/s")
    print()

print("="*60)
print("📊 统计结果:")
print(f"  平均耗时: {sum(times)/len(times):.2f}秒")
print(f"  最快: {min(times):.2f}秒")
print(f"  最慢: {max(times):.2f}秒")
print()

# 对比OpenAI API
print("💡 预期加速效果:")
print(f"  OpenAI API (网络延迟): ~1-2秒 + 推理时间")
print(f"  本地API (无网络延迟): {sum(times)/len(times):.2f}秒")
print(f"  加速比: ~{2 / (sum(times)/len(times)):.1f}x - {1 / (sum(times)/len(times)):.1f}x")
print("="*60)
