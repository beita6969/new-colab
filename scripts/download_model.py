#!/usr/bin/env python3
"""
下载Qwen2.5-7B模型
"""
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

def download_model():
    """下载Qwen2.5-7B模型到缓存"""

    print("=" * 60)
    print("📥 下载Qwen2.5-7B-Instruct模型")
    print("=" * 60)

    model_name = "Qwen/Qwen2.5-7B-Instruct"

    # 设置HuggingFace token (从环境变量读取)
    token = os.environ.get('HF_TOKEN', None)
    if not token:
        print("⚠️  警告: 未设置HF_TOKEN环境变量")
        print("   请运行: export HF_TOKEN='your_token_here'")
        print("   或在代码中设置 token 变量")

    try:
        print(f"\n📥 下载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=token
        )
        print("✅ Tokenizer下载完成")

        print(f"\n📥 下载模型（约14GB，请耐心等待）...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="cpu",  # 先下载到CPU
            trust_remote_code=True,
            token=token
        )
        print("✅ 模型下载完成")

        # 显示缓存路径
        from transformers.utils import TRANSFORMERS_CACHE
        print(f"\n📂 模型缓存路径: {TRANSFORMERS_CACHE}")

        print("\n" + "=" * 60)
        print("✅ 所有文件下载完成！")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        print("\n可能的原因:")
        print("  1. 网络连接问题")
        print("  2. HuggingFace token无效")
        print("  3. 磁盘空间不足（需要至少20GB）")
        return False

if __name__ == "__main__":
    download_model()
