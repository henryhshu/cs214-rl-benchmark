# Microbenchmark Analysis

Results from single-machine benchmarks (Threadripper 24-core, 2x RTX 3090, CPU-only mode).

## M1: Dispatch Overhead (no-op tasks)

Tests per-task dispatch (Ray) vs bulk mesh dispatch (Monarch).

| Workers | Ray (ms) | Monarch (ms) | Ray/Monarch |
|---------|----------|--------------|-------------|
| 1       | 0.60     | 0.73         | 0.83x       |
| 2       | 0.76     | 1.08         | 0.71x       |
| 4       | 1.19     | 1.35         | 0.88x       |
| 8       | 1.80     | 1.68         | 1.07x       |
| 16      | 3.15     | 2.33         | 1.35x       |

**Crossover at ~8 workers.** Ray is faster at low worker counts due to lower per-call baseline. But Ray's cost grows 5.25x for 16x workers while Monarch grows only 3.19x. Monarch's single `.call()` to the mesh amortizes dispatch overhead better at scale — consistent with top-down vs bottom-up scheduling.

## M2: Broadcast Efficiency (same tensor to all workers)

Tests Ray's shared-memory object store vs Monarch's controller-mediated transfer.

### Small payload (50KB)

| Workers | Ray (ms) | Monarch (ms) |
|---------|----------|--------------|
| 1       | 1.79     | 0.95         |
| 4       | 2.50     | 1.70         |
| 8       | 3.43     | 2.13         |

Monarch wins at small payloads. Ray's object store has a fixed overhead (~0.5ms `ray.put()`) that doesn't pay off for small data.

### Large payload (3.2MB, realistic model weights)

| Workers | Ray (ms) | Ray put (ms) | Monarch (ms) |
|---------|----------|-------------- |--------------|
| 1       | 7.81     | 4.39          | 3.06         |
| 2       | 6.27     | 3.09          | 5.16         |
| 4       | 8.29     | 3.24          | 6.83         |
| 8       | 9.86     | 3.73          | 9.23         |

Ray has a high fixed cost from `ray.put()` (~3-4ms for 3.2MB) but scales well: only 1.26x from 1->8 workers because all workers do zero-copy reads from shared memory. Monarch scales 3.0x because it serializes through the controller to each worker. They converge at 8 workers (~9-10ms).

**Key insight**: Ray's object store is designed for exactly this broadcast pattern. The put-once-read-many model avoids redundant serialization. Monarch doesn't have an equivalent mechanism on a single node.

## M3: Straggler Sensitivity (variable-duration work)

Tests framework overhead when workers finish at different times.

| Workers | Variance  | Ray overhead (ms) | Monarch overhead (ms) |
|---------|-----------|--------------------|-----------------------|
| 2       | uniform   | 1.05               | 1.82                  |
| 2       | extreme   | 1.43               | 1.70                  |
| 8       | uniform   | 2.23               | 1.94                  |
| 8       | extreme   | 2.26               | 2.32                  |

**Non-differentiator.** Both frameworks show similar overhead (~1.5-2.3ms) that is largely insensitive to work variance. Neither has an advantage in handling stragglers — both block until the slowest worker returns. The overhead comes from dispatch + result collection, not from how they handle heterogeneous completion.

## M4: Iteration Loop (broadcast + compute + gather, repeated)

Simulates the PPO training loop. 20ms fixed compute per worker, ~200KB payload broadcast, 100 cycles. This is the most representative benchmark.

| Workers | Ray cycle (ms) | Ray overhead (ms) | Ray OH% | Monarch cycle (ms) | Monarch overhead (ms) | Monarch OH% |
|---------|---------------|-------------------|---------|--------------------|-----------------------|-------------|
| 1       | 23.2          | 3.2               | 14%     | 22.5               | 2.5                   | 11%         |
| 2       | 23.2          | 3.2               | 14%     | 24.0               | 4.0                   | 17%         |
| 4       | 23.8          | 3.8               | 16%     | 39.5               | 19.5                  | **49%**     |
| 8       | 25.0          | 5.0               | 20%     | 67.8               | 44.1                  | **65%**     |

**The headline result.** Ray's cycle time barely increases from 1->8 workers (23->25ms) because:
1. `ray.put()` stores the payload in shared memory once
2. All 8 workers read it zero-copy
3. The 20ms compute dominates the cycle
4. Result gathering is cheap

Monarch's cycle time triples (22.5->68ms) because:
1. The controller serializes the payload to each worker individually
2. With 8 workers, that's roughly 8x the serialization cost
3. Gathering 8 results also passes through the controller

At 8 workers, 65% of Monarch's cycle is pure framework overhead. Only 20% of Ray's cycle is overhead.

## Summary

| Benchmark          | Winner at low N | Winner at high N | Key differentiator                       |
|--------------------|-----------------|------------------|------------------------------------------|
| Dispatch overhead  | Ray             | Monarch          | Bulk dispatch amortizes at scale          |
| Broadcast          | Monarch         | Tie (~8w)        | Ray's object store vs Monarch's controller|
| Straggler          | Tie             | Tie              | Neither handles stragglers better         |
| Iteration loop     | Tie (~1w)       | **Ray**          | Shared memory avoids redundant serialization |

**On a single machine, Ray's shared-memory object store gives it a fundamental advantage for the broadcast-compute-gather pattern used in PPO training.** Monarch's controller-mediated data transfer doesn't benefit from shared memory — it behaves as if workers are on separate nodes. This architecture is designed for multi-node setups where RDMA and high-bandwidth interconnects make the controller path competitive.

The dispatch overhead crossover (M1) suggests Monarch's mesh model would win for very large worker counts, but this advantage is overwhelmed by the serialization cost in realistic workloads (M4).

In the full PPO benchmark, the rollout compute (50-100ms+) is large enough to partially mask this overhead difference, which is why aggregate throughput appears similar. But the microbenchmarks reveal that Monarch pays 5-10x more framework tax per iteration, a cost that compounds with faster compute or higher iteration rates.
