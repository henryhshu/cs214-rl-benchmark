# Microbenchmarks

Standalone benchmarks that isolate specific framework-level overheads (scheduling, serialization, per-worker variance) from end-to-end PPO training, enabling a direct comparison of Ray's bottom-up scheduler vs Monarch's top-down scheduler.

## Why Microbenchmarks?

The main `run_all.py` benchmark measures aggregate throughput and iteration timing, but conflates scheduling overhead, serialization cost, and per-worker variance into single numbers. To understand **why** one framework is faster than another at a given scale, we need to isolate each component:

- **Scheduling latency**: Time from task dispatch to execution start on the worker. Reveals the cost of each framework's task scheduler independent of workload.
- **Serialization cost**: Time to marshal tensors (model weights, rollout data) across process boundaries. Shows whether data transfer is a bottleneck at large model/horizon sizes.
- **Per-worker variance**: Distribution of rollout times across workers. High variance means straggler workers dominate wall-clock time, reducing parallelism efficiency.

## Benchmarks

### `bench_scheduling_latency.py`

No-op ping tasks that measure pure dispatch-to-execution latency.

```bash
python benchmark/tests/bench_scheduling_latency.py --framework ray --worker-counts 1 2 4 8
python benchmark/tests/bench_scheduling_latency.py --framework monarch --worker-counts 1 2 4 8
```

**What it measures**: A lightweight `TimingWorker` receives `dispatch_time` and returns `time.time() - dispatch_time`. After a warmup round, N batches are dispatched. Reports mean, median, p99, std, min, max latency in milliseconds.

### `bench_serialization.py`

Serialize/deserialize model state dicts and rollout tensors at varying sizes.

```bash
python benchmark/tests/bench_serialization.py --framework ray --num-trials 50
python benchmark/tests/bench_serialization.py --framework monarch --num-trials 50
```

**What it measures**: Round-trip time for sending data to a worker and getting it back. Tests model states at hidden_dims [64, 256, 1024] and rollout tensors at horizons [128, 256, 512, 1024]. For Ray, also separately times `ray.put()` and `ray.get()`.

### `bench_worker_scaling.py`

Per-worker rollout time distributions using real `RolloutWorker` + local environments.

```bash
python benchmark/tests/bench_worker_scaling.py --framework ray --worker-counts 1 2 4 8
python benchmark/tests/bench_worker_scaling.py --framework monarch --worker-counts 1 2 4 8
```

**What it measures**: Actual rollout times from the instrumented `collect()` method, wall-clock time, and straggler ratio (max/mean per-worker time). Uses the real PPO model and Blackjack environment.

## Running All Benchmarks

```bash
# Full suite (both frameworks, all benchmarks):
python benchmark/tests/run_microbenchmarks.py

# Single framework:
python benchmark/tests/run_microbenchmarks.py --frameworks ray

# Single benchmark:
python benchmark/tests/run_microbenchmarks.py --benchmarks scheduling

# Preview without running:
python benchmark/tests/run_microbenchmarks.py --dry-run
```

Results are written to `benchmark/results/microbenchmarks/`.

## Generating Plots

```bash
python benchmark/tests/plot_microbenchmarks.py
```

Produces 4 figures in `benchmark/results/microbenchmarks/figures/`:

| Figure | Description |
|--------|-------------|
| `figM1_scheduling_latency.png` | Scheduling latency bar chart (Ray vs Monarch, per worker count) |
| `figM2_serialization.png` | Serialization round-trip time vs payload size (log-scale X) |
| `figM3_worker_times.png` | Per-worker rollout time box plot |
| `figM4_straggler_ratio.png` | Straggler ratio (max/mean) vs worker count |

## In-Loop Instrumentation

The PPO training scripts (`ppo_ray.py`, `ppo_monarch.py`) also emit per-iteration timing fields that are plotted by `benchmark/plot.py` as figures 8 and 9 (generated only when instrumented data is present):

| Metric Field | Description |
|---|---|
| `model_serialize_time_s` | Time to put model state into object store (Ray) or clone (Monarch) |
| `scheduling_latency_mean_s` | Mean dispatch-to-start latency across workers |
| `scheduling_latency_max_s` | Max dispatch-to-start latency |
| `scheduling_latency_min_s` | Min dispatch-to-start latency |
| `scheduling_latency_std_s` | Std dev of dispatch-to-start latency |
| `worker_rollout_time_mean_s` | Mean per-worker rollout execution time |
| `worker_rollout_time_max_s` | Max per-worker rollout time (straggler) |
| `worker_rollout_time_min_s` | Min per-worker rollout time |
| `worker_rollout_time_std_s` | Std dev of per-worker rollout times |
| `load_state_time_mean_s` | Mean time workers spend loading model weights |
| `serialization_overhead_s` | `rollout_time - max(worker_rollout_times)` — framework overhead beyond compute |
