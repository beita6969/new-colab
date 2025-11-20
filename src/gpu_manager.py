#!/usr/bin/env python
"""
GPU Manager - 管理GPU 2-3的使用，保护代理进程
"""
import os
import subprocess
import sys
from typing import List, Dict, Tuple
import time

class GPUManager:
    """GPU管理器：清理、保护、监控"""

    def __init__(
        self,
        target_gpus: List[int] = [2, 3],
        protected_pids: List[int] = [3819483],
        auto_clean: bool = True
    ):
        """
        Args:
            target_gpus: 目标GPU列表（仅使用这些GPU）
            protected_pids: 受保护的进程ID列表（不会被清理）
            auto_clean: 是否自动清理目标GPU上的其他进程
        """
        self.target_gpus = target_gpus
        self.protected_pids = protected_pids
        self.auto_clean = auto_clean

    def check_gpu_available(self) -> bool:
        """检查目标GPU是否可用"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,memory.total', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                check=True
            )

            gpu_info = result.stdout.strip().split('\n')
            available_gpus = [int(line.split(',')[0]) for line in gpu_info]

            for gpu_id in self.target_gpus:
                if gpu_id not in available_gpus:
                    print(f"❌ GPU {gpu_id} 不可用！")
                    return False

            print(f"✅ 目标GPU {self.target_gpus} 可用")
            return True

        except Exception as e:
            print(f"❌ 检查GPU失败: {e}")
            return False

    def get_gpu_processes(self, gpu_id: int) -> List[Dict]:
        """获取指定GPU上的所有进程"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                check=True
            )

            # 获取每个GPU的进程
            pmon_result = subprocess.run(
                ['nvidia-smi', 'pmon', '-c', '1'],
                capture_output=True,
                text=True
            )

            # 解析nvidia-smi输出
            processes = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 3:
                        try:
                            pid = int(parts[0])
                            process_name = parts[1]
                            memory = parts[2]

                            # 检查进程在哪个GPU上
                            # 这里简化处理，假设nvidia-smi输出按GPU顺序
                            processes.append({
                                'pid': pid,
                                'name': process_name,
                                'memory': memory,
                                'gpu': gpu_id  # 简化：需要更精确的GPU分配检测
                            })
                        except ValueError:
                            continue

            return processes

        except Exception as e:
            print(f"⚠️  获取GPU进程失败: {e}")
            return []

    def get_all_target_gpu_processes(self) -> Dict[int, List[Dict]]:
        """获取所有目标GPU上的进程"""
        try:
            # 使用nvidia-smi查询所有GPU进程
            result = subprocess.run(
                ['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,process_name,used_memory',
                 '--format=csv,noheader'],
                capture_output=True,
                text=True,
                check=True
            )

            # 获取GPU UUID到ID的映射
            uuid_result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                check=True
            )

            uuid_to_id = {}
            for line in uuid_result.stdout.strip().split('\n'):
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 2:
                        gpu_id = int(parts[0])
                        gpu_uuid = parts[1]
                        uuid_to_id[gpu_uuid] = gpu_id

            # 按GPU分组进程
            gpu_processes = {gpu_id: [] for gpu_id in self.target_gpus}

            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 4:
                        try:
                            pid = int(parts[0])
                            gpu_uuid = parts[1]
                            process_name = parts[2]
                            memory = parts[3]

                            gpu_id = uuid_to_id.get(gpu_uuid)
                            if gpu_id in self.target_gpus:
                                gpu_processes[gpu_id].append({
                                    'pid': pid,
                                    'name': process_name,
                                    'memory': memory,
                                    'gpu': gpu_id
                                })
                        except ValueError:
                            continue

            return gpu_processes

        except subprocess.CalledProcessError as e:
            print(f"⚠️  nvidia-smi命令失败: {e}")
            return {gpu_id: [] for gpu_id in self.target_gpus}
        except Exception as e:
            print(f"⚠️  获取GPU进程失败: {e}")
            return {gpu_id: [] for gpu_id in self.target_gpus}

    def check_protected_processes(self) -> bool:
        """检查受保护的进程是否存在"""
        for pid in self.protected_pids:
            try:
                # 检查进程是否存在
                os.kill(pid, 0)  # 信号0不会杀死进程，只检查是否存在
                print(f"✅ 受保护进程 {pid} 正在运行")
            except OSError:
                print(f"⚠️  受保护进程 {pid} 未找到（可能已停止）")
                return False

        return True

    def clean_target_gpus(self, force: bool = False) -> Tuple[int, int]:
        """
        清理目标GPU上的进程（保护受保护进程）

        Returns:
            (成功清理数, 失败数)
        """
        if not self.auto_clean and not force:
            print("⚠️  自动清理已禁用，使用force=True强制清理")
            return 0, 0

        gpu_processes = self.get_all_target_gpu_processes()

        success_count = 0
        fail_count = 0

        for gpu_id, processes in gpu_processes.items():
            if not processes:
                print(f"✅ GPU {gpu_id} 无活动进程")
                continue

            print(f"\n🔍 GPU {gpu_id} 上的进程:")
            for proc in processes:
                pid = proc['pid']
                name = proc['name']
                memory = proc['memory']

                # 检查是否为受保护进程
                if pid in self.protected_pids:
                    print(f"  🛡️  PID {pid} ({name}, {memory}) - 受保护，跳过")
                    continue

                print(f"  🎯 PID {pid} ({name}, {memory}) - 准备清理")

                try:
                    # 尝试优雅终止
                    os.kill(pid, 15)  # SIGTERM
                    time.sleep(0.5)

                    # 检查是否已终止
                    try:
                        os.kill(pid, 0)
                        # 如果还在运行，强制杀死
                        print(f"    ⚡ 强制杀死 PID {pid}")
                        os.kill(pid, 9)  # SIGKILL
                        time.sleep(0.2)
                    except OSError:
                        # 进程已终止
                        pass

                    print(f"    ✅ 成功清理 PID {pid}")
                    success_count += 1

                except PermissionError:
                    print(f"    ❌ 权限不足，无法清理 PID {pid}")
                    fail_count += 1
                except ProcessLookupError:
                    print(f"    ✅ PID {pid} 已经不存在")
                    success_count += 1
                except Exception as e:
                    print(f"    ❌ 清理 PID {pid} 失败: {e}")
                    fail_count += 1

        print(f"\n📊 清理完成: 成功 {success_count}, 失败 {fail_count}")
        return success_count, fail_count

    def setup_cuda_devices(self):
        """设置CUDA环境变量，仅使用目标GPU"""
        gpu_str = ','.join(map(str, self.target_gpus))
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
        print(f"✅ 设置 CUDA_VISIBLE_DEVICES={gpu_str}")

    def verify_environment(self) -> bool:
        """验证整个环境配置"""
        print("=" * 60)
        print("🔧 验证GPU环境")
        print("=" * 60)

        # 1. 检查GPU可用性
        if not self.check_gpu_available():
            return False

        # 2. 检查受保护进程
        self.check_protected_processes()

        # 3. 获取当前GPU进程
        gpu_processes = self.get_all_target_gpu_processes()
        total_processes = sum(len(procs) for procs in gpu_processes.values())

        if total_processes > 0:
            print(f"\n⚠️  检测到 {total_processes} 个进程在目标GPU上运行")

            if self.auto_clean:
                print("🧹 开始自动清理...")
                success, fail = self.clean_target_gpus()

                if fail > 0:
                    print(f"❌ 有 {fail} 个进程清理失败")
                    return False
            else:
                print("❌ 请手动清理GPU进程或启用auto_clean")
                return False

        # 4. 设置CUDA环境变量
        self.setup_cuda_devices()

        print("\n" + "=" * 60)
        print("✅ GPU环境验证通过")
        print("=" * 60)

        return True

    def monitor_gpu_usage(self, interval: int = 5):
        """持续监控GPU使用情况"""
        try:
            while True:
                print(f"\n{'=' * 60}")
                print(f"⏰ GPU监控 - {time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'=' * 60}")

                for gpu_id in self.target_gpus:
                    result = subprocess.run(
                        [
                            'nvidia-smi',
                            f'--id={gpu_id}',
                            '--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total',
                            '--format=csv,noheader'
                        ],
                        capture_output=True,
                        text=True,
                        check=True
                    )

                    info = result.stdout.strip()
                    print(f"GPU {gpu_id}: {info}")

                time.sleep(interval)

        except KeyboardInterrupt:
            print("\n⏹️  监控已停止")


def main():
    """测试GPU管理器"""
    import argparse

    parser = argparse.ArgumentParser(description="GPU管理器")
    parser.add_argument('--gpus', type=int, nargs='+', default=[2, 3], help='目标GPU列表')
    parser.add_argument('--protected-pids', type=int, nargs='+', default=[3819483], help='受保护的进程ID')
    parser.add_argument('--no-auto-clean', action='store_true', help='禁用自动清理')
    parser.add_argument('--monitor', action='store_true', help='持续监控GPU')
    parser.add_argument('--force-clean', action='store_true', help='强制清理GPU')

    args = parser.parse_args()

    manager = GPUManager(
        target_gpus=args.gpus,
        protected_pids=args.protected_pids,
        auto_clean=not args.no_auto_clean
    )

    if args.force_clean:
        manager.clean_target_gpus(force=True)
    elif args.monitor:
        if manager.verify_environment():
            manager.monitor_gpu_usage()
    else:
        manager.verify_environment()


if __name__ == "__main__":
    main()
