"""
Benchmark orchestrator — runs Ray and Monarch PPO across worker/horizon scales.

Usage:
    # Full run (all 24 experiments, ~45 min):
    python benchmark/run_all.py

    # Quick sanity check:
    python benchmark/run_all.py --frameworks ray --workers 1 2 --horizons 128 --iterations 5

    # Single framework:
    python benchmark/run_all.py --frameworks monarch --workers 1 2 4

Results are written to benchmark/results/<framework>_<N>w_<H>h/
  config.json     — experiment parameters
  metrics.jsonl   — per-iteration training metrics
  sysmon.jsonl    — system resource samples (2 Hz)
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = ROOT / "benchmark"
RESULTS_DIR = BENCHMARK_DIR / "results"

# Env server lives in external/OpenEnv; envs/ must be importable from there.
OPENENV_ROOT = ROOT / "external" / "OpenEnv"
SERVER_PORT = 8000
SERVER_WORKERS = 8  # Keep high so it never bottlenecks our experiments


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _venv_python() -> str:
    venv = ROOT / ".venv" / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable


def _subprocess_env(device: str = "auto") -> dict:
    """Return os.environ with PYTHONPATH and optional device override."""
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{OPENENV_ROOT}:{existing}" if existing else str(OPENENV_ROOT)
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def start_env_server(port: int = SERVER_PORT, workers: int = SERVER_WORKERS) -> subprocess.Popen:
    """Start the OpenSpiel blackjack uvicorn server."""
    py = _venv_python()
    env = _subprocess_env()
    env["OPENSPIEL_GAME"] = "blackjack"
    proc = subprocess.Popen(
        [
            py, "-m", "uvicorn",
            "envs.openspiel_env.server.app:app",
            "--workers", str(workers),
            "--port", str(port),
            "--host", "127.0.0.1",
            "--log-level", "warning",
        ],
        cwd=str(OPENENV_ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _wait_for_server(f"http://127.0.0.1:{port}", timeout=20)
    return proc


def _wait_for_server(url: str, timeout: int = 20) -> None:
    import urllib.request
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            return
        except Exception:
            time.sleep(0.5)
    print(f"  WARNING: server at {url} did not become ready within {timeout}s", flush=True)


def start_sysmon(output_file: Path) -> subprocess.Popen:
    py = _venv_python()
    return subprocess.Popen(
        [py, str(BENCHMARK_DIR / "sysmon.py"), str(output_file)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _build_cmd(framework: str, workers: int, horizon: int, hidden_dim: int,
               env_mode: str, iterations: int, eval_games: int,
               metrics_file: Path) -> list:
    py = _venv_python()
    if framework == "ray":
        script = ROOT / "ray_blackjack" / "ppo_ray.py"
    else:
        script = ROOT / "monarch_blackjack" / "ppo_monarch.py"
    cmd = [
        py, str(script),
        "--server-url", f"http://127.0.0.1:{SERVER_PORT}",
        "--num-workers", str(workers),
        "--horizon", str(horizon),
        "--hidden-dim", str(hidden_dim),
        "--iterations", str(iterations),
        "--eval-games", str(eval_games),
        "--metrics-file", str(metrics_file),
    ]
    if env_mode == "local":
        cmd.append("--local-env")
    return cmd


def run_experiment(framework: str, workers: int, horizon: int, hidden_dim: int,
                   env_mode: str, iterations: int, eval_games: int,
                   exp_dir: Path, device: str = "auto") -> int:
    metrics_file = exp_dir / "metrics.jsonl"
    sysmon_file = exp_dir / "sysmon.jsonl"

    cmd = _build_cmd(framework, workers, horizon, hidden_dim, env_mode,
                     iterations, eval_games, metrics_file)
    env = _subprocess_env(device=device)

    sysmon = start_sysmon(sysmon_file)
    try:
        result = subprocess.run(cmd, cwd=str(ROOT), env=env)
        return result.returncode
    finally:
        sysmon.terminate()
        sysmon.wait(timeout=5)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Ray vs Monarch benchmark matrix")
    parser.add_argument("--frameworks", nargs="+", default=["ray", "monarch"],
                        choices=["ray", "monarch"])
    parser.add_argument("--workers", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--horizons", nargs="+", type=int, default=[128, 256, 512])
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[64],
                        help="Model hidden dim(s). Use 64 for CPU-bound, 1024+ for GPU-bound.")
    parser.add_argument("--env-modes", nargs="+", default=["websocket"],
                        choices=["websocket", "local"],
                        help="'websocket': I/O-bound via server. 'local': CPU-bound in-process.")
    parser.add_argument("--iterations", type=int, default=30,
                        help="Training iterations per experiment")
    parser.add_argument("--eval-games", type=int, default=50)
    parser.add_argument("--run-name", type=str, default=None,
                        help="Name for this run (default: timestamp). Results go to "
                             "benchmark/results/<run-name>/")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu"],
                        help="'cpu' forces CPU-only (sets CUDA_VISIBLE_DEVICES=\"\"). "
                             "'auto' uses GPU if available.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print experiment list without running")
    args = parser.parse_args()

    run_name = args.run_name or time.strftime("%Y-%m-%d_%H%M%S")
    results_dir = RESULTS_DIR / run_name
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run: {run_name}  →  {results_dir}  (device={args.device})", flush=True)

    experiments = [
        (fw, w, h, hd, em)
        for fw in args.frameworks
        for w in args.workers
        for h in args.horizons
        for hd in args.hidden_dims
        for em in args.env_modes
    ]
    total = len(experiments)

    if args.dry_run:
        print(f"Would run {total} experiments:")
        for fw, w, h, hd, em in experiments:
            print(f"  {fw:8s}  workers={w}  horizon={h:4d}  hidden={hd:4d}  env={em}")
        return

    need_server = any(em == "websocket" for _, _, _, _, em in experiments)
    server = None
    if need_server:
        print(f"Starting env server (port={SERVER_PORT}, workers={SERVER_WORKERS})...", flush=True)
        server = start_env_server()

    results_summary = []
    try:
        for i, (framework, workers, horizon, hidden_dim, env_mode) in enumerate(experiments, 1):
            exp_name = f"{framework}_{workers}w_{horizon}h_{hidden_dim}d_{env_mode}"
            exp_dir = results_dir / exp_name
            exp_dir.mkdir(exist_ok=True)

            config = {
                "framework": framework,
                "workers": workers,
                "horizon": horizon,
                "hidden_dim": hidden_dim,
                "env_mode": env_mode,
                "device": args.device,
                "iterations": args.iterations,
                "eval_games": args.eval_games,
                "server_port": SERVER_PORT,
                "run_name": run_name,
            }
            (exp_dir / "config.json").write_text(json.dumps(config, indent=2))

            print(f"\n[{i}/{total}] {exp_name}", flush=True)
            t0 = time.time()
            rc = run_experiment(
                framework, workers, horizon, hidden_dim, env_mode,
                args.iterations, args.eval_games, exp_dir,
                device=args.device,
            )
            elapsed = time.time() - t0

            status = "OK" if rc == 0 else f"FAILED (rc={rc})"
            print(f"  {status}  {elapsed:.0f}s", flush=True)
            results_summary.append({**config, "status": status, "elapsed_s": elapsed})

    finally:
        if server is not None:
            print("\nStopping env server...", flush=True)
            server.terminate()
            server.wait(timeout=10)

    manifest_path = results_dir / "manifest.json"
    manifest_path.write_text(json.dumps(results_summary, indent=2))
    print(f"\nDone. Results in {results_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"\nNext: python benchmark/plot.py --runs {run_name}")


if __name__ == "__main__":
    main()
