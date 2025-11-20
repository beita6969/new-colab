#!/usr/bin/env python3
"""
实时训练监控器 - 显示关键指标和错误
"""
import time
import re
import sys
from collections import defaultdict

def parse_log_line(line):
    """解析日志行提取关键信息"""
    info = {}

    # Step信息
    if 'Step' in line and '/500' in line:
        match = re.search(r'Step (\d+)/(\d+)', line)
        if match:
            info['step'] = int(match.group(1))

    # 准确率
    if '准确率统计:' in line:
        match = re.search(r'(\d+)/(\d+) = ([\d.]+)%', line)
        if match:
            info['correct'] = int(match.group(1))
            info['total'] = int(match.group(2))
            info['accuracy'] = float(match.group(3))

    # 正确性评分
    if '平均正确性评分:' in line:
        match = re.search(r'平均正确性评分: ([\d.]+)/10\.0', line)
        if match:
            info['avg_score'] = float(match.group(1))

    # 问题类型分布
    if 'math:' in line and 'avg:' in line:
        match = re.search(r'(\w+): ([\d.]+)% \(avg: ([\d.-]+), n=(\d+)\)', line)
        if match:
            info['task_type'] = match.group(1)
            info['task_accuracy'] = float(match.group(2))
            info['task_avg_score'] = float(match.group(3))
            info['task_count'] = int(match.group(4))

    # 错误
    if '❌ Workflow执行异常:' in line:
        match = re.search(r'❌ Workflow执行异常: (\w+)', line)
        if match:
            info['error_type'] = match.group(1)

    # Fallback
    if '✅ Fallback成功' in line:
        info['fallback'] = True

    # 正确性评分详情
    if '正确性评分:' in line and '|' in line:
        match = re.search(r'正确性评分: ([\d.-]+)/10\.0', line)
        if match:
            info['sample_score'] = float(match.group(1))

    return info

def monitor_training(log_file):
    """实时监控训练日志"""
    print("🔍 开始监控训练...")
    print("=" * 80)

    # 统计信息
    stats = {
        'step': 0,
        'samples_processed': 0,
        'errors': defaultdict(int),
        'fallbacks': 0,
        'task_stats': defaultdict(lambda: {'correct': 0, 'total': 0, 'scores': []})
    }

    # 打开日志文件
    with open(log_file, 'r') as f:
        # 跳到文件末尾
        f.seek(0, 2)

        last_update = time.time()

        while True:
            line = f.readline()
            if not line:
                # 没有新内容，等待
                time.sleep(0.5)

                # 每5秒显示一次汇总
                if time.time() - last_update > 5:
                    print_summary(stats)
                    last_update = time.time()
                continue

            # 解析行
            info = parse_log_line(line)

            if 'step' in info:
                stats['step'] = info['step']
                print(f"\n{'='*80}")
                print(f"📍 Step {info['step']}/500")
                print(f"{'='*80}")

            if 'accuracy' in info:
                print(f"\n✅ 准确率: {info['correct']}/{info['total']} = {info['accuracy']:.1f}%")
                if 'avg_score' in info:
                    print(f"   平均评分: {info['avg_score']:.2f}/10.0")
                stats['samples_processed'] = info['total']

            if 'task_type' in info:
                task = info['task_type']
                print(f"   {task}: {info['task_accuracy']:.1f}% (avg: {info['task_avg_score']:.2f}, n={info['task_count']})")
                stats['task_stats'][task]['total'] = info['task_count']
                stats['task_stats'][task]['accuracy'] = info['task_accuracy']
                stats['task_stats'][task]['avg_score'] = info['task_avg_score']

            if 'error_type' in info:
                error_type = info['error_type']
                stats['errors'][error_type] += 1
                print(f"❌ 错误: {error_type} (累计: {stats['errors'][error_type]}次)")

            if 'fallback' in info:
                stats['fallbacks'] += 1
                print(f"🔄 Fallback触发 (累计: {stats['fallbacks']}次)")

            if 'sample_score' in info:
                score = info['sample_score']
                if score >= 8:
                    emoji = "🟢"
                elif score >= 5:
                    emoji = "🟡"
                elif score >= 0:
                    emoji = "🟠"
                else:
                    emoji = "🔴"
                print(f"{emoji} 样本得分: {score:.1f}/10.0", end=' ')

def print_summary(stats):
    """打印汇总信息"""
    print(f"\n{'─'*80}")
    print(f"📊 训练汇总 (Step {stats['step']}/500)")
    print(f"   已处理样本: {stats['samples_processed']}")
    print(f"   累计错误: {sum(stats['errors'].values())}次")
    if stats['errors']:
        print(f"   错误分布: {dict(stats['errors'])}")
    print(f"   Fallback次数: {stats['fallbacks']}次")
    if stats['task_stats']:
        print(f"   任务类型:")
        for task, data in stats['task_stats'].items():
            if data['total'] > 0:
                print(f"      {task}: {data.get('accuracy', 0):.1f}% (avg: {data.get('avg_score', 0):.2f})")
    print(f"{'─'*80}")

if __name__ == '__main__':
    import glob

    # 找到最新的日志文件
    log_files = sorted(glob.glob('logs/train_restart_*.log'), reverse=True)
    if not log_files:
        print("❌ 找不到训练日志文件")
        sys.exit(1)

    log_file = log_files[0]
    print(f"📄 监控日志: {log_file}")

    try:
        monitor_training(log_file)
    except KeyboardInterrupt:
        print("\n\n⏹️  监控已停止")
        sys.exit(0)
