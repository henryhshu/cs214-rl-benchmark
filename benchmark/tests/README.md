# Microbenchmarks

Standalone benchmarks that isolate specific architectural differences between
Ray (bottom-up, decentralized scheduler + object store) and Monarch (top-down,
mesh-based controller), enabling a direct comparison independent of the PPO
workload.

## Benchmarks

### M1: Bulk Dispatch Overhead (`bench_dispatch.py`)

Measures the cost of dispatching no-op tasks to N workers and collecting results.

- **Ray**: N separate `.remote()` calls + `ray.get(N futures)` — overhead scales with N
- **Monarch**: 1 `.call()` to the mesh — overhead should be roughly constant in N

Sweeps worker counts [1, 2, 4, 8, 16] with 200 batches each.

```bash
python benchmark/tests/bench_dispatch.py --framework ray
python benchmark/tests/bench_dispatch.py --framework monarch
```

### M2: Broadcast Efficiency (`bench_broadcast.py`)

Sends the same large tensor (model weights) to all N workers simultaneously.

- **Ray**: `ray.put()` into shared-memory object store once, workers get zero-copy reads
- **Monarch**: Serialized through the controller to each worker

Tests payload sizes [50KB, 200KB, 800KB, 3.2MB] x worker counts [1, 2, 4, 8].

```bash
python benchmark/tests/bench_broadcast.py --framework ray
python benchmark/tests/bench_broadcast.py --framework monarch
```

### M3: Straggler Sensitivity (`bench_straggler.py`)

Workers do variable-duration work (simulating heterogeneous episode lengths).

- **Ray**: Independent futures — framework overhead on top of slowest worker
- **Monarch**: Mesh-level synchronous collection — same

Measures `overhead = wall_clock - max(worker_times)` across variance levels.

```bash
python benchmark/tests/bench_straggler.py --framework ray
python benchmark/tests/bench_straggler.py --framework monarch
```

### M4: Iteration Loop Overhead (`bench_iteration_loop.py`)

Simulates the PPO training loop: broadcast weights -> workers compute -> gather
results -> repeat. Measures amortized per-cycle framework tax.

Runs 100 rapid dispatch-gather cycles with realistic payload sizes.

```bash
python benchmark/tests/bench_iteration_loop.py --framework ray
python benchmark/tests/bench_iteration_loop.py --framework monarch
```

## Running All Benchmarks

```bash
python benchmark/tests/run_microbenchmarks.py
python benchmark/tests/run_microbenchmarks.py --frameworks ray
python benchmark/tests/run_microbenchmarks.py --dry-run
```

Results are written to `benchmark/results/microbenchmarks/`.

## Generating Plots

```bash
python benchmark/tests/plot_microbenchmarks.py
```

Produces 4 figures in `benchmark/results/microbenchmarks/figures/`:

| Figure | What it shows |
|--------|---------------|
| `figM1_dispatch_overhead.png` | Dispatch+collect time vs worker count (per-task vs bulk) |
| `figM2_broadcast_efficiency.png` | Broadcast time vs payload size at different worker counts |
| `figM3_straggler_overhead.png` | Framework overhead beyond slowest worker, across variance levels |
| `figM4_iteration_loop.png` | Per-cycle overhead in rapid dispatch-gather loops |
