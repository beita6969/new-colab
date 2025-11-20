#!/usr/bin/env python3
"""
检查训练环境
"""
import sys
import os
import subprocess
from pathlib import Path

def check_environment():
    """全面检查训练环境"""

    print("=" * 70)
    print(" 🔍 AFlow + ROLL 训练环境检查")
    print("=" * 70)

    all_ok = True

    # 1. Python版本
    print("\n[1/10] 🐍 Python版本")
    python_version = sys.version_info
    if python_version >= (3, 8):
        print(f"  ✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print(f"  ❌ Python版本过低: {python_version}")
        all_ok = False

    # 2. PyTorch
    print("\n[2/10] 🔥 PyTorch")
    try:
        import torch
        print(f"  ✅ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"  ✅ CUDA {torch.version.cuda}")
            print(f"  ✅ 可用GPU: {torch.cuda.device_count()}")
        else:
            print("  ❌ CUDA不可用")
            all_ok = False
    except ImportError:
        print("  ❌ PyTorch未安装")
        all_ok = False

    # 3. Transformers
    print("\n[3/10] 🤗 Transformers")
    try:
        import transformers
        print(f"  ✅ Transformers {transformers.__version__}")
    except ImportError:
        print("  ❌ Transformers未安装")
        all_ok = False

    # 4. PEFT (LoRA)
    print("\n[4/10] 🎯 PEFT")
    try:
        import peft
        print(f"  ✅ PEFT {peft.__version__}")
    except ImportError:
        print("  ❌ PEFT未安装: pip install peft")
        all_ok = False

    # 5. GPU 2-3
    print("\n[5/10] 🖥️  GPU 2-3")
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.total', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            check=True
        )
        gpus = result.stdout.strip().split('\n')
        gpu_2_ok = False
        gpu_3_ok = False
        for line in gpus:
            if line.startswith('2,'):
                print(f"  ✅ GPU 2: {line}")
                gpu_2_ok = True
            if line.startswith('3,'):
                print(f"  ✅ GPU 3: {line}")
                gpu_3_ok = True

        if not (gpu_2_ok and gpu_3_ok):
            print("  ❌ GPU 2或3不可用")
            all_ok = False
    except Exception as e:
        print(f"  ❌ 无法检查GPU: {e}")
        all_ok = False

    # 6. 代理进程
    print("\n[6/10] 🛡️  代理进程 (PID 3819483)")
    try:
        os.kill(3819483, 0)
        print("  ✅ 代理进程正在运行")
    except OSError:
        print("  ⚠️  代理进程未找到（可以继续，但需更新配置）")

    # 7. AFlow
    print("\n[7/10] 📦 AFlow框架")
    aflow_path = Path('/home/yijia/.claude/11/AFlow')
    if aflow_path.exists():
        print(f"  ✅ AFlow路径存在: {aflow_path}")
        if (aflow_path / 'scripts/operators.py').exists():
            print("  ✅ operators.py存在")
        else:
            print("  ❌ operators.py不存在")
            all_ok = False
    else:
        print("  ❌ AFlow路径不存在")
        all_ok = False

    # 8. ROLL
    print("\n[8/10] 📦 ROLL框架")
    roll_path = Path('/home/yijia/.claude/11/ROLL')
    if roll_path.exists():
        print(f"  ✅ ROLL路径存在: {roll_path}")
    else:
        print("  ❌ ROLL路径不存在")
        all_ok = False

    # 9. 配置文件
    print("\n[9/10] ⚙️  配置文件")
    config_files = [
        'config/training.yaml',
        'config/aflow_llm.yaml'
    ]
    for cf in config_files:
        if Path(cf).exists():
            print(f"  ✅ {cf}")
        else:
            print(f"  ❌ {cf} 不存在")
            all_ok = False

    # 10. 数据集
    print("\n[10/10] 📂 数据集")
    dataset_files = [
        'data/train/mixed_dataset.jsonl',
        'data/val/mixed_dataset.jsonl'
    ]
    for df in dataset_files:
        if Path(df).exists():
            # 统计行数
            with open(df) as f:
                lines = sum(1 for _ in f)
            print(f"  ✅ {df} ({lines} 样本)")
        else:
            print(f"  ❌ {df} 不存在")
            all_ok = False

    # 11. Qwen模型
    print("\n[11/10] 🤖 Qwen2.5-7B模型")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct",
            trust_remote_code=True
        )
        print("  ✅ Qwen2.5-7B-Instruct已下载")
    except Exception as e:
        print(f"  ❌ Qwen2.5-7B模型未下载")
        print(f"     运行: python3 scripts/download_model.py")
        all_ok = False

    # 总结
    print("\n" + "=" * 70)
    if all_ok:
        print(" ✅ 所有检查通过！可以开始训练")
        print("=" * 70)
        print("\n🚀 启动训练:")
        print("  python3 train.py")
        return 0
    else:
        print(" ❌ 存在问题，请先解决")
        print("=" * 70)
        return 1

if __name__ == "__main__":
    sys.exit(check_environment())
