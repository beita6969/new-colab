#!/usr/bin/env python3
"""下载MBPP和MBPP+数据集"""

import os
import json
from datasets import load_dataset

# 设置代理
os.environ['http_proxy'] = 'http://127.0.0.1:10808'
os.environ['https_proxy'] = 'http://127.0.0.1:10808'

print("代理已设置: http://127.0.0.1:10808")
print("="*60)

# 设置输出目录
output_dir = "/home/yijia/.claude/11/integrated_aflow_roll/data/raw"
os.makedirs(output_dir, exist_ok=True)

print("\n[1/2] 下载MBPP原版数据集...")
print("-"*60)
try:
    # 下载MBPP数据集
    mbpp = load_dataset("mbpp", "full")

    # 保存为JSON格式
    mbpp_path = os.path.join(output_dir, "mbpp_full.json")

    # 合并所有split
    all_data = []
    for split in mbpp.keys():
        print(f"处理 {split} split: {len(mbpp[split])} 个样本")
        for item in mbpp[split]:
            item['split'] = split
            all_data.append(item)

    # 保存
    with open(mbpp_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)

    print(f"✅ MBPP原版已保存到: {mbpp_path}")
    print(f"   总样本数: {len(all_data)}")

    # 保存每个split单独的文件
    for split in mbpp.keys():
        split_path = os.path.join(output_dir, f"mbpp_{split}.json")
        split_data = [item for item in mbpp[split]]
        with open(split_path, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)
        print(f"   - {split}: {len(split_data)} 样本 -> {split_path}")

except Exception as e:
    print(f"❌ MBPP下载失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("[2/2] 下载MBPP+增强版数据集...")
print("-"*60)
try:
    # 下载MBPP+数据集
    mbpp_plus = load_dataset("evalplus/mbppplus")

    # 保存为JSON格式
    mbpp_plus_path = os.path.join(output_dir, "mbpp_plus_full.json")

    # 合并所有split
    all_data_plus = []
    for split in mbpp_plus.keys():
        print(f"处理 {split} split: {len(mbpp_plus[split])} 个样本")
        for item in mbpp_plus[split]:
            item['split'] = split
            all_data_plus.append(item)

    # 保存
    with open(mbpp_plus_path, 'w', encoding='utf-8') as f:
        json.dump(all_data_plus, f, indent=2, ensure_ascii=False)

    print(f"✅ MBPP+已保存到: {mbpp_plus_path}")
    print(f"   总样本数: {len(all_data_plus)}")

    # 保存每个split单独的文件
    for split in mbpp_plus.keys():
        split_path = os.path.join(output_dir, f"mbpp_plus_{split}.json")
        split_data = [item for item in mbpp_plus[split]]
        with open(split_path, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)
        print(f"   - {split}: {len(split_data)} 样本 -> {split_path}")

except Exception as e:
    print(f"❌ MBPP+下载失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("✅ 数据集下载完成！")
print("="*60)
print(f"\n数据保存位置: {output_dir}")
print("\n文件列表:")
for file in sorted(os.listdir(output_dir)):
    if file.startswith('mbpp'):
        filepath = os.path.join(output_dir, file)
        size = os.path.getsize(filepath) / 1024  # KB
        print(f"  - {file} ({size:.2f} KB)")
