#!/usr/bin/env python3
"""
训练监控脚本 - 实时查看训练进展
"""
import os
import time
import sys
from datetime import datetime
import subprocess


def clear_screen():
    """清屏"""
    os.system('clear' if os.name == 'posix' else 'cls')


def get_gpu_usage():
    """获取GPU使用情况"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True
        )
        lines = result.stdout.strip().split('\n')
        gpu_info = []
        for line in lines:
            parts = line.split(', ')
            if len(parts) >= 5:
                gpu_info.append({
                    'id': parts[0],
                    'name': parts[1],
                    'mem_used': float(parts[2]),
                    'mem_total': float(parts[3]),
                    'util': float(parts[4])
                })
        return gpu_info
    except:
        return []


def get_latest_log_lines(log_file, n=20):
    """获取最新的日志行"""
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            return lines[-n:]
    except:
        return []


def parse_metrics_from_log(lines):
    """从日志中解析指标"""
    metrics = {
        'step': None,
        'math_acc': None,
        'code_acc': None,
        'qa_acc': None,
        'overall_acc': None,
        'reward': None,
        'loss': None
    }

    for line in lines:
        if 'Step' in line and '/' in line:
            try:
                # 解析 Step X/Y
                parts = line.split('Step')[1].split('/')[0].strip()
                metrics['step'] = int(parts)
            except:
                pass

        if '准确率' in line or 'accuracy' in line.lower():
            # 解析准确率
            if 'Math' in line or '数学' in line:
                try:
                    acc = float(line.split(':')[1].split('%')[0].strip())
                    metrics['math_acc'] = acc
                except:
                    pass
            elif 'Code' in line or '代码' in line:
                try:
                    acc = float(line.split(':')[1].split('%')[0].strip())
                    metrics['code_acc'] = acc
                except:
                    pass
            elif 'QA' in line or '问答' in line:
                try:
                    acc = float(line.split(':')[1].split('%')[0].strip())
                    metrics['qa_acc'] = acc
                except:
                    pass
            elif 'Overall' in line or '整体' in line:
                try:
                    acc = float(line.split(':')[1].split('%')[0].strip())
                    metrics['overall_acc'] = acc
                except:
                    pass

        if '奖励' in line or 'reward' in line.lower():
            try:
                reward = float(line.split(':')[1].strip().split()[0])
                metrics['reward'] = reward
            except:
                pass

        if '损失' in line or 'loss' in line.lower():
            try:
                loss = float(line.split(':')[1].strip().split()[0])
                metrics['loss'] = loss
            except:
                pass

    return metrics


def display_dashboard():
    """显示监控仪表板"""
    log_file = '/home/yijia/.claude/11/integrated_aflow_roll/logs/train_improved_v3.log'

    while True:
        clear_screen()

        # 标题
        print("="*70)
        print("📊 GRPO训练监控仪表板".center(70))
        print("="*70)
        print(f"🕐 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # GPU状态
        print("🖥️  GPU状态:")
        print("-"*70)
        gpu_info = get_gpu_usage()
        for gpu in gpu_info:
            if gpu['id'] in ['2', '3']:  # 只显示GPU 2和3
                mem_percent = (gpu['mem_used'] / gpu['mem_total']) * 100
                print(f"  GPU {gpu['id']}: {gpu['name']}")
                print(f"    显存: {gpu['mem_used']:.0f}/{gpu['mem_total']:.0f} MB ({mem_percent:.1f}%)")
                print(f"    利用率: {gpu['util']:.0f}%")
        print()

        # 训练指标
        print("📈 训练指标:")
        print("-"*70)
        lines = get_latest_log_lines(log_file, 50)
        metrics = parse_metrics_from_log(lines)

        if metrics['step']:
            print(f"  当前步数: {metrics['step']}/500")
            progress = (metrics['step'] / 500) * 100
            bar_length = 30
            filled = int(bar_length * metrics['step'] / 500)
            bar = '█' * filled + '░' * (bar_length - filled)
            print(f"  进度: [{bar}] {progress:.1f}%")
        else:
            print(f"  当前步数: 等待中...")

        print()
        print("  准确率:")
        if metrics['math_acc'] is not None:
            print(f"    Math: {metrics['math_acc']:.1f}%")
        else:
            print(f"    Math: --")

        if metrics['code_acc'] is not None:
            print(f"    Code: {metrics['code_acc']:.1f}%")
        else:
            print(f"    Code: --")

        if metrics['qa_acc'] is not None:
            print(f"    QA:   {metrics['qa_acc']:.1f}%")
        else:
            print(f"    QA:   --")

        if metrics['overall_acc'] is not None:
            print(f"    整体: {metrics['overall_acc']:.1f}%")
        else:
            print(f"    整体: --")

        print()
        if metrics['reward'] is not None:
            print(f"  平均奖励: {metrics['reward']:.3f}")
        else:
            print(f"  平均奖励: --")

        if metrics['loss'] is not None:
            print(f"  损失: {metrics['loss']:.4f}")
        else:
            print(f"  损失: --")

        # 最新日志
        print()
        print("📝 最新日志:")
        print("-"*70)
        recent_lines = lines[-5:] if lines else []
        for line in recent_lines:
            line = line.strip()
            if line:
                # 截断长行
                if len(line) > 67:
                    line = line[:64] + "..."
                print(f"  {line}")

        # 状态栏
        print()
        print("="*70)
        print("  [Q] 退出  |  [R] 刷新  |  自动刷新: 5秒")

        # 等待刷新
        time.sleep(5)


if __name__ == "__main__":
    try:
        display_dashboard()
    except KeyboardInterrupt:
        print("\n\n👋 监控已停止")
        sys.exit(0)
