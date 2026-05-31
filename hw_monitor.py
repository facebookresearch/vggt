"""
VGGT 训练/推理硬件监控脚本

功能:
  1. 记录 GPU 显存使用、利用率、温度等指标
  2. 记录 CPU 内存使用
  3. 支持后台运行，定时采样
  4. 输出硬件使用报告

用法:
  # 后台监控训练过程（每 2 秒采样一次，保存到文件）
  python hw_monitor.py --interval 2 --output hw_stats.json

  # 在另一个终端运行训练，监控会自动记录

  # 读取并打印监控报告
  python hw_monitor.py --report hw_stats.json
"""

import argparse
import json
import os
import time
import threading
import subprocess
import sys
from datetime import datetime


def get_gpu_info():
    """获取 GPU 信息（通过 nvidia-smi）"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,utilization.memory,"
             "memory.used,memory.total,memory.free,temperature.gpu,power.draw,power.limit",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 10:
                    gpus.append({
                        "index": int(parts[0]),
                        "name": parts[1],
                        "gpu_util_pct": float(parts[2]),
                        "mem_util_pct": float(parts[3]),
                        "mem_used_mb": float(parts[4]),
                        "mem_total_mb": float(parts[5]),
                        "mem_free_mb": float(parts[6]),
                        "temp_c": float(parts[7]),
                        "power_w": float(parts[8]) if parts[8] != 'N/A' else None,
                        "power_limit_w": float(parts[9]) if parts[9] != 'N/A' else None,
                    })
        return gpus
    except Exception as e:
        return [{"error": str(e)}]


def get_cpu_memory():
    """获取 CPU 内存信息"""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return {
            "total_gb": mem.total / (1024**3),
            "used_gb": mem.used / (1024**3),
            "available_gb": mem.available / (1024**3),
            "percent": mem.percent,
        }
    except ImportError:
        return {"error": "psutil not installed. Run: pip install psutil"}


def get_pytorch_gpu_info():
    """获取 PyTorch 报告的 GPU 显存信息"""
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
                "max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
                "max_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3,
            }
    except ImportError:
        pass
    return None


class HWMonitor:
    """硬件监控器，在后台线程中定时采样"""

    def __init__(self, interval=2.0):
        self.interval = interval
        self.samples = []
        self.running = False
        self.thread = None

    def start(self):
        """启动后台监控"""
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        print(f"[HW Monitor] Started (interval={self.interval}s)")

    def stop(self):
        """停止监控"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        print(f"[HW Monitor] Stopped ({len(self.samples)} samples)")

    def _monitor_loop(self):
        """监控循环"""
        while self.running:
            sample = {
                "timestamp": datetime.now().isoformat(),
                "gpus": get_gpu_info(),
                "cpu_mem": get_cpu_memory(),
                "pytorch_gpu": get_pytorch_gpu_info(),
            }
            self.samples.append(sample)
            time.sleep(self.interval)

    def save(self, output_path):
        """保存监控数据到文件"""
        with open(output_path, 'w') as f:
            json.dump(self.samples, f, indent=2)
        print(f"[HW Monitor] Data saved to {output_path}")

    def generate_report(self):
        """生成硬件使用报告"""
        if not self.samples:
            return "No samples recorded."

        # 提取 GPU 指标序列
        gpu_metrics = {}
        for sample in self.samples:
            for gpu in sample.get("gpus", []):
                idx = gpu.get("index", 0)
                if idx not in gpu_metrics:
                    gpu_metrics[idx] = {
                        "name": gpu.get("name", "Unknown"),
                        "mem_used": [],
                        "mem_total": [],
                        "gpu_util": [],
                        "temp": [],
                        "power": [],
                        "timestamps": [],
                    }
                gpu_metrics[idx]["mem_used"].append(gpu.get("mem_used_mb", 0))
                gpu_metrics[idx]["mem_total"].append(gpu.get("mem_total_mb", 0))
                gpu_metrics[idx]["gpu_util"].append(gpu.get("gpu_util_pct", 0))
                gpu_metrics[idx]["temp"].append(gpu.get("temp_c", 0))
                gpu_metrics[idx]["power"].append(gpu.get("power_w", 0) or 0)
                gpu_metrics[idx]["timestamps"].append(sample["timestamp"])

        # 生成报告
        report = []
        report.append("=" * 70)
        report.append("VGGT 硬件使用报告")
        report.append("=" * 70)
        report.append(f"采样时间: {self.samples[0]['timestamp']} -> {self.samples[-1]['timestamp']}")
        report.append(f"采样次数: {len(self.samples)}")
        report.append(f"采样间隔: {self.interval}s")
        report.append("")

        for idx, metrics in sorted(gpu_metrics.items()):
            report.append(f"--- GPU {idx}: {metrics['name']} ---")
            mem_used = metrics["mem_used"]
            mem_total = metrics["mem_total"]
            report.append(f"  显存总量: {mem_total[0]:.0f} MB")
            report.append(f"  显存峰值: {max(mem_used):.0f} MB ({max(mem_used)/mem_total[0]*100:.1f}%)")
            report.append(f"  显存均值: {sum(mem_used)/len(mem_used):.0f} MB")
            report.append(f"  GPU 利用率峰值: {max(metrics['gpu_util']):.1f}%")
            report.append(f"  GPU 利用率均值: {sum(metrics['gpu_util'])/len(metrics['gpu_util']):.1f}%")
            report.append(f"  温度峰值: {max(metrics['temp']):.1f}°C")
            report.append(f"  功耗峰值: {max(metrics['power']):.1f}W")
            report.append("")

        # PyTorch 显存
        pytorch_mem = [s.get("pytorch_gpu", {}) for s in self.samples if s.get("pytorch_gpu")]
        if pytorch_mem:
            max_alloc = max(m.get("max_allocated_gb", 0) or 0 for m in pytorch_mem if m)
            max_resv = max(m.get("max_reserved_gb", 0) or 0 for m in pytorch_mem if m)
            report.append("--- PyTorch 显存统计 ---")
            report.append(f"  峰值分配 (allocated): {max_alloc:.2f} GB")
            report.append(f"  峰值保留 (reserved):  {max_resv:.2f} GB")
            report.append("")

        # CPU 内存
        cpu_samples = [s.get("cpu_mem", {}) for s in self.samples if s.get("cpu_mem") and "error" not in s["cpu_mem"]]
        if cpu_samples:
            cpu_max_pct = max(s.get("percent", 0) for s in cpu_samples)
            cpu_max_used = max(s.get("used_gb", 0) for s in cpu_samples)
            report.append("--- CPU 内存 ---")
            report.append(f"  峰值使用: {cpu_max_used:.2f} GB ({cpu_max_pct:.1f}%)")
            report.append("")

        return "\n".join(report)


def print_report_from_file(filepath):
    """从保存的文件中读取并打印报告"""
    with open(filepath, 'r') as f:
        samples = json.load(f)

    monitor = HWMonitor()
    monitor.samples = samples
    # 从采样间隔推断
    if len(samples) >= 2:
        t1 = datetime.fromisoformat(samples[0]["timestamp"])
        t2 = datetime.fromisoformat(samples[1]["timestamp"])
        monitor.interval = (t2 - t1).total_seconds()
    print(monitor.generate_report())


def main():
    parser = argparse.ArgumentParser(description="VGGT 硬件监控工具")
    parser.add_argument("--interval", type=float, default=2.0,
                        help="采样间隔（秒）")
    parser.add_argument("--output", type=str, default="hw_stats.json",
                        help="输出文件路径")
    parser.add_argument("--duration", type=float, default=None,
                        help="监控时长（秒），默认持续运行直到 Ctrl+C")
    parser.add_argument("--report", type=str, default=None,
                        help="从已有的 JSON 文件生成报告")
    args = parser.parse_args()

    # 报告模式
    if args.report:
        print_report_from_file(args.report)
        return

    # 监控模式
    monitor = HWMonitor(interval=args.interval)
    monitor.start()

    print(f"\n{'='*60}")
    print("硬件监控运行中...")
    print(f"输出文件: {args.output}")
    if args.duration:
        print(f"将运行 {args.duration} 秒")
    print("按 Ctrl+C 停止")
    print(f"{'='*60}\n")

    try:
        if args.duration:
            time.sleep(args.duration)
        else:
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\n正在停止监控...")

    monitor.stop()
    monitor.save(args.output)
    print(monitor.generate_report())


if __name__ == "__main__":
    main()
