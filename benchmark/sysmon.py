"""
System metrics sampler — run as a subprocess alongside a training job.

Usage:
    python benchmark/sysmon.py results/exp_name/sysmon.jsonl

Samples CPU, RAM, and GPU stats at ~2 Hz and writes JSONL records.
Exits cleanly on SIGTERM or SIGINT.
"""
import json
import os
import signal
import subprocess
import sys
import time

import psutil


def _gpu_stats() -> dict:
    """Query nvidia-smi for per-GPU utilization and VRAM."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=2,
        )
    except Exception:
        return {}

    result = {}
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4:
            try:
                idx = int(parts[0])
                result[f"gpu{idx}_util_pct"] = float(parts[1])
                result[f"gpu{idx}_vram_used_mb"] = float(parts[2])
                result[f"gpu{idx}_vram_total_mb"] = float(parts[3])
            except ValueError:
                pass
    return result


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: sysmon.py <output.jsonl>", file=sys.stderr)
        sys.exit(1)

    output_path = sys.argv[1]
    running = True

    def _stop(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    # Prime psutil so first reading is accurate
    psutil.cpu_percent(percpu=True)
    time.sleep(0.1)

    with open(output_path, "w") as f:
        while running:
            try:
                cpu_per_core = psutil.cpu_percent(percpu=True)
                mem = psutil.virtual_memory()
                record = {
                    "timestamp": time.time(),
                    "cpu_mean_pct": sum(cpu_per_core) / len(cpu_per_core),
                    "cpu_per_core": cpu_per_core,
                    "ram_used_mb": mem.used / 1e6,
                    "ram_total_mb": mem.total / 1e6,
                    **_gpu_stats(),
                }
                f.write(json.dumps(record) + "\n")
                f.flush()
            except Exception:
                pass  # never crash the monitor

            time.sleep(0.5)


if __name__ == "__main__":
    main()
