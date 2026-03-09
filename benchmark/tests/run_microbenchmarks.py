"""
Orchestrator for all microbenchmarks — runs each benchmark × framework × config.

Usage:
    # Run all benchmarks for all frameworks:
    python benchmark/tests/run_microbenchmarks.py

    # Single framework, single benchmark:
    python benchmark/tests/run_microbenchmarks.py --frameworks ray --benchmarks scheduling

    # Dry run:
    python benchmark/tests/run_microbenchmarks.py --dry-run
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
TESTS_DIR = ROOT / "benchmark" / "tests"
RESULTS_DIR = ROOT / "benchmark" / "results" / "microbenchmarks"

BENCHMARKS = {
    "scheduling": {
        "script": TESTS_DIR / "bench_scheduling_latency.py",
        "extra_args": ["--num-pings", "100", "--warmup", "10"],
    },
    "serialization": {
        "script": TESTS_DIR / "bench_serialization.py",
        "extra_args": ["--num-trials", "50"],
    },
    "worker_scaling": {
        "script": TESTS_DIR / "bench_worker_scaling.py",
        "extra_args": ["--num-rollouts", "10"],
    },
}


def _venv_python() -> str:
    venv = ROOT / ".venv" / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable


def run_benchmark(name: str, framework: str, output_file: Path) -> int:
    cfg = BENCHMARKS[name]
    py = _venv_python()
    cmd = [
        py, str(cfg["script"]),
        "--framework", framework,
        "--output", str(output_file),
        *cfg["extra_args"],
    ]
    env = os.environ.copy()
    result = subprocess.run(cmd, cwd=str(ROOT), env=env)
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all microbenchmarks")
    parser.add_argument("--frameworks", nargs="+", default=["ray", "monarch"],
                        choices=["ray", "monarch"])
    parser.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS.keys()),
                        choices=list(BENCHMARKS.keys()))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    experiments = [
        (bench, fw)
        for bench in args.benchmarks
        for fw in args.frameworks
    ]

    if args.dry_run:
        print(f"Would run {len(experiments)} microbenchmarks:")
        for bench, fw in experiments:
            print(f"  {bench:20s}  framework={fw}")
        return

    summary = []
    for i, (bench, fw) in enumerate(experiments, 1):
        output_file = RESULTS_DIR / f"{bench}_{fw}.json"
        print(f"\n[{i}/{len(experiments)}] {bench} / {fw}", flush=True)

        t0 = time.time()
        rc = run_benchmark(bench, fw, output_file)
        elapsed = time.time() - t0

        status = "OK" if rc == 0 else f"FAILED (rc={rc})"
        print(f"  {status}  {elapsed:.1f}s", flush=True)
        summary.append({
            "benchmark": bench,
            "framework": fw,
            "status": status,
            "elapsed_s": elapsed,
            "output": str(output_file),
        })

    manifest = RESULTS_DIR / "manifest.json"
    manifest.write_text(json.dumps(summary, indent=2))
    print(f"\nDone. Results in {RESULTS_DIR}")
    print(f"Manifest: {manifest}")
    print(f"\nNext: python benchmark/tests/plot_microbenchmarks.py")


if __name__ == "__main__":
    main()
