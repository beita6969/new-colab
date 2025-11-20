#!/usr/bin/env python3
"""
下载所有AFlow数据集
"""
import os
import json
from datasets import load_dataset

# 创建数据目录
os.makedirs("data/gsm8k", exist_ok=True)
os.makedirs("data/math", exist_ok=True)
os.makedirs("data/hotpotqa", exist_ok=True)
os.makedirs("data/drop", exist_ok=True)
os.makedirs("data/mbpp", exist_ok=True)

print("=" * 60)
print("开始下载所有AFlow数据集")
print("=" * 60)

# 1. GSM8K
print("\n【1/5】下载 GSM8K...")
try:
    ds = load_dataset("openai/gsm8k", "main", split="train")
    with open("data/gsm8k/train.jsonl", "w") as f:
        for item in ds:
            f.write(json.dumps({"question": item["question"], "answer": item["answer"]}) + "\n")
    print(f"✅ GSM8K train: {len(ds)} 样本")

    ds_test = load_dataset("openai/gsm8k", "main", split="test")
    with open("data/gsm8k/test.jsonl", "w") as f:
        for item in ds_test:
            f.write(json.dumps({"question": item["question"], "answer": item["answer"]}) + "\n")
    print(f"✅ GSM8K test: {len(ds_test)} 样本")
except Exception as e:
    print(f"❌ GSM8K 下载失败: {e}")

# 2. MATH
print("\n【2/5】下载 MATH...")
try:
    ds = load_dataset("lighteval/MATH", split="train")
    with open("data/math/train.jsonl", "w") as f:
        for item in ds:
            f.write(json.dumps({
                "problem": item.get("problem", ""),
                "solution": item.get("solution", ""),
                "answer": item.get("answer", "")
            }) + "\n")
    print(f"✅ MATH train: {len(ds)} 样本")

    ds_test = load_dataset("lighteval/MATH", split="test")
    with open("data/math/test.jsonl", "w") as f:
        for item in ds_test:
            f.write(json.dumps({
                "problem": item.get("problem", ""),
                "solution": item.get("solution", ""),
                "answer": item.get("answer", "")
            }) + "\n")
    print(f"✅ MATH test: {len(ds_test)} 样本")
except Exception as e:
    print(f"❌ MATH 下载失败: {e}")

# 3. HotpotQA
print("\n【3/5】下载 HotpotQA...")
try:
    ds = load_dataset("hotpot_qa", "distractor", split="train")
    with open("data/hotpotqa/train.jsonl", "w") as f:
        for item in ds[:5000]:  # 取前5000个
            f.write(json.dumps({
                "question": item["question"],
                "answer": item["answer"],
                "context": item.get("context", "")
            }) + "\n")
    print(f"✅ HotpotQA train: 5000 样本")

    ds_val = load_dataset("hotpot_qa", "distractor", split="validation")
    with open("data/hotpotqa/validation.jsonl", "w") as f:
        for item in ds_val:
            f.write(json.dumps({
                "question": item["question"],
                "answer": item["answer"],
                "context": item.get("context", "")
            }) + "\n")
    print(f"✅ HotpotQA validation: {len(ds_val)} 样本")
except Exception as e:
    print(f"❌ HotpotQA 下载失败: {e}")

# 4. DROP
print("\n【4/5】下载 DROP...")
try:
    ds = load_dataset("ucinlp/drop", split="train")
    with open("data/drop/train.jsonl", "w") as f:
        for item in ds[:3000]:  # 取前3000个
            f.write(json.dumps({
                "question": item["question"],
                "answer": str(item.get("answers_spans", {}).get("spans", [""])[0] if item.get("answers_spans") else ""),
                "passage": item.get("passage", "")
            }) + "\n")
    print(f"✅ DROP train: 3000 样本")

    ds_val = load_dataset("ucinlp/drop", split="validation")
    with open("data/drop/validation.jsonl", "w") as f:
        for item in ds_val[:500]:  # 取前500个
            f.write(json.dumps({
                "question": item["question"],
                "answer": str(item.get("answers_spans", {}).get("spans", [""])[0] if item.get("answers_spans") else ""),
                "passage": item.get("passage", "")
            }) + "\n")
    print(f"✅ DROP validation: 500 样本")
except Exception as e:
    print(f"❌ DROP 下载失败: {e}")

# 5. MBPP
print("\n【5/5】下载 MBPP...")
try:
    ds = load_dataset("mbpp", "sanitized", split="train")
    with open("data/mbpp/train.jsonl", "w") as f:
        for item in ds:
            f.write(json.dumps({
                "task_id": item.get("task_id", 0),
                "prompt": item["text"],
                "code": item["code"],
                "test_list": item.get("test_list", []),
                "test_setup_code": item.get("test_setup_code", ""),
                "challenge_test_list": item.get("challenge_test_list", [])
            }) + "\n")
    print(f"✅ MBPP train: {len(ds)} 样本")

    ds_test = load_dataset("mbpp", "sanitized", split="test")
    with open("data/mbpp/test.jsonl", "w") as f:
        for item in ds_test:
            f.write(json.dumps({
                "task_id": item.get("task_id", 0),
                "prompt": item["text"],
                "code": item["code"],
                "test_list": item.get("test_list", []),
                "test_setup_code": item.get("test_setup_code", ""),
                "challenge_test_list": item.get("challenge_test_list", [])
            }) + "\n")
    print(f"✅ MBPP test: {len(ds_test)} 样本")
except Exception as e:
    print(f"❌ MBPP 下载失败: {e}")

print("\n" + "=" * 60)
print("所有数据集下载完成！")
print("=" * 60)
