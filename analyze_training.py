#!/usr/bin/env python3
"""
训练日志分析脚本 - 提取关键指标和统计信息
"""
import re
from collections import defaultdict
from pathlib import Path

def parse_training_log(log_file: str = "logs/training_output.log"):
    """解析训练日志并提取关键指标"""
    
    if not Path(log_file).exists():
        print(f"❌ 日志文件不存在: {log_file}")
        return
    
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # 提取所有步骤信息
    steps = re.findall(r'Step (\d+)/500', content)
    accuracies = re.findall(r'准确率统计: (\d+)/(\d+) = ([\d.]+)% \(平均正确性评分: ([-\d.]+)', content)
    
    print("\n" + "=" * 80)
    print("📊 GRPO训练分析报告")
    print("=" * 80)
    
    # 基本统计
    print(f"\n📈 训练进度:")
    print(f"  • 已完成步骤: {len(set(steps))} 步")
    if steps:
        print(f"  • 当前步骤: Step {steps[-1]}/500")
        print(f"  • 完成百分比: {int(steps[-1])/500*100:.1f}%")
    
    # 准确率趋势
    if accuracies:
        print(f"\n🎯 准确率趋势:")
        print(f"  {'Step':<8} {'正确/总数':<12} {'准确率':<10} {'平均评分'}")
        print(f"  {'-'*8} {'-'*12} {'-'*10} {'-'*10}")
        
        for i, (correct, total, acc, score) in enumerate(accuracies[-10:], 1):
            print(f"  Step {i:<3} {correct}/{total:<8} {acc}%{' '*(8-len(acc))} {score}")
        
        # 计算提升
        if len(accuracies) > 1:
            first_acc = float(accuracies[0][2])
            last_acc = float(accuracies[-1][2])
            improvement = last_acc - first_acc
            print(f"\n  💡 准确率变化: {first_acc}% → {last_acc}% (Δ{improvement:+.1f}%)")
    
    # 错误分析
    errors = re.findall(r'(TypeError|AttributeError|ValueError): (.+)', content)
    if errors:
        print(f"\n⚠️  错误统计:")
        error_counts = defaultdict(int)
        for error_type, _ in errors:
            error_counts[error_type] += 1
        
        for error_type, count in sorted(error_counts.items(), key=lambda x: -x[1]):
            print(f"  • {error_type}: {count} 次")
        
        print(f"\n  💡 总错误数: {len(errors)} (Qwen生成的代码会因错误获得低奖励，促进学习)")
    
    # wandb状态
    wandb_matches = re.findall(r'wandb: (.*)', content)
    if wandb_matches:
        print(f"\n📊 wandb状态:")
        for match in wandb_matches[-5:]:
            print(f"  {match}")
    
    # 性能统计
    times = re.findall(r'(\d+\.\d+)s/it', content)
    if times:
        times = [float(t) for t in times]
        avg_time = sum(times) / len(times)
        print(f"\n⏱️  性能统计:")
        print(f"  • 平均每样本耗时: {avg_time:.1f}秒")
        print(f"  • 估计Step 10耗时: {avg_time * 4 * 4 * 10 / 60:.1f}分钟")
        print(f"  • 估计完整训练(500步)耗时: {avg_time * 4 * 4 * 500 / 3600:.1f}小时")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    parse_training_log()
