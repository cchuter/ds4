# Multi-GPU pipeline-parallel inference for ds4 — v0 design

**Date:** 2026-05-26
**Status:** Approved for implementation (revised after codex spec review)
**Scope:** v0 — layout-only PP across N CUDA GPUs with CPU spill. No microbatch scheduler (v1), no expert-parallel (v2), no tensor-parallel (out of plan).

## Background

ds4 currently runs CUDA inference on a single device only:

- `ds4_cuda.cu:~1206` hardcodes `int dev = 0; cudaSetDevice(dev)` in `ds4_gpu_init`.
- `ds4_gpu_tensor` carries `ptr / bytes / owner` only — no device id.
- A single global cuBLAS handle, a single global scratch/workspace, and a single global model-map range are assumed.
- `ds4_gpu_set_model_map_range(...)` copies the full tensor-data span eagerly at engine startup; there is no selective per-device caching today.
- The public GPU API in `ds4_gpu.h` exposes zero device-id parameters.
- The high-level inference path in `ds4.c` has no per-layer device concept; the prefill/decode graph (`ds4_gpu_graph`) is built against the one-device assumption.
- CLIs expose only a binary `--cuda` switch — no `--tensor-split`, `--main-gpu`, `--n-gpu`, etc.

The target model is **DeepSeek-V4-Flash**: 284 B total parameters, 13 B active, 43 transformer layers, heavy MoE with routed experts in mixed quants (Q2_K / IQ2_XXS / Q4_K). Roughly 95 % of the parameter mass lives in routed experts.

The test box `192.168.50.231` has 2 × NVIDIA RTX 6000 Ada (48 GB each), connected by PCIe (NUMA NODE topology, no NVLink). Combined VRAM (~96 GB) is insufficient for the higher-quant variants of the model, so multi-GPU must coexist with CPU execution for spilled layers.

## Goal

Enable ds4 inference across N CUDA GPUs (tested at N = 1 and N = 2; N ≥ 3 covered by synthetic unit tests) with per-GPU VRAM budgets, falling back to CPU for any layer that doesn't fit.

The full roadmap targets capacity, batch = 1 latency, and batch > 1 throughput. **v0 delivers capacity only.** v1 (microbatch scheduler) and v2 (expert-parallel) deliver the latency and throughput wins.

### v0 performance expectations (explicit)

- **batch = 1 gen latency** is **expected to be marginally worse** than single-GPU (one PCIe activation hop per token added at the device boundary). Acceptable cost for accessing more capacity.
- **prefill latency** may be measurably worse: prefill activation at a boundary is `chunk_tokens × hidden × dtype` bytes — for a 256-token chunk at hidden = 7168 in fp32 that's ~7 MB per hop, ~280 µs at 25 GB/s effective PCIe. Still small but no longer trivial. v0 makes no prefill-overlap claim.
- **batch > 1 throughput** is unchanged versus single-GPU + CPU spill — concurrent execution on the two devices is a v1 deliverable.

## Non-goals (v0)

Explicitly deferred or out of plan:

- **v1 — microbatch scheduler:** concurrent execution of different microbatches on different GPUs.
- **v2 — expert-parallel routed FFN:** spreading routed experts of a single layer across multiple GPUs.
- **Tensor-parallel matmul split:** not planned. PCIe-only topology makes per-layer all-reduce more expensive than the speedup it would unlock.
- **NCCL / multi-host inference:** out of scope.
- **Hardware verification at N ≥ 3:** the available test hardware has 2 GPUs. N ≥ 3 is exercised by layer-packer unit tests with synthetic budgets only.
- **Dynamic re-packing at runtime:** layout is computed once at model load.
- **NUMA-aware pinned memory:** simple global pinned allocation in v0.
- **Optimal (non-greedy) packing:** v0 uses contiguous greedy packing. Power users get no override knob in v0.

## Architecture

### Three-tier, monotonic-contiguous layer placement

Each of the 43 transformer layers plus two pseudo-layers (embedding table, output head) is assigned to exactly one tier: `GPU0`, `GPU1`, …, `GPU{n-1}`, or `CPU`.

Placement is **monotonically non-decreasing** in tier index, where `CPU` is treated as the last tier. Concretely, if layer `L` is on tier `t`, all layers `L+1..n-1` are on tiers `t..CPU`. This gives a strictly contiguous topology — no `GPU0 → GPU1 → GPU0` ping-pong — and bounds device boundaries to at most `(n_gpus)` in any forward pass.

A `layer_placement[]` table is computed once at model-load time from the user's `--gpu-vram` budget.

### Per-device state

```c
typedef struct {
    int                device_id;       // CUDA device index
    cudaStream_t       stream;          // single stream per device in v0
    cublasHandle_t     cublas;          // per-device handle (was global)
    void              *scratch;         // per-device scratch / workspace pool
    size_t             scratch_bytes;
    void              *model_cache;     // per-device tensor cache (replaces global map range)
    size_t             model_cache_bytes;
    size_t             budget_bytes;    // configured budget
    size_t             used_bytes;      // running total of allocations
} ds4_gpu_ctx;

#define DS4_MAX_GPUS 16
extern ds4_gpu_ctx g_gpu[DS4_MAX_GPUS];
extern int         g_n_gpus;
```

`DS4_MAX_GPUS = 16` bounds NxN-table stack allocations. `g_n_gpus` is set at init from CLI or `cudaGetDeviceCount`.

### Device-aware tensor contract

`ds4_gpu_tensor` gains `int device_id`:

```c
struct ds4_gpu_tensor {
    void    *ptr;
    uint64_t bytes;
    int      owner;
    int      device_id;   // NEW: which device this tensor lives on
};
```

All tensor APIs (`ds4_gpu_tensor_alloc`, `_free`, `_read`, `_write`, `_copy`, slice/view operations) take the device id from the tensor where applicable, and call `cudaSetDevice` internally. `ds4_gpu_tensor_copy` between tensors of different devices uses the peer matrix (below); if the pair isn't peer-enabled, the copy stages through a pinned-host bounce buffer.

Allocation gains an explicit device argument so callers don't depend on ambient CUDA state:

```c
// New primary API
int ds4_gpu_tensor_alloc_on(ds4_gpu_tensor *t, int device_id, uint64_t bytes);

// Existing API kept as a single-device shim — equivalent to alloc_on(t, 0, bytes)
int ds4_gpu_tensor_alloc(ds4_gpu_tensor *t, uint64_t bytes);
```

`alloc_on` internally does `cudaSetDevice(device_id)`, allocates on that device's slab pool, and records `t->device_id = device_id`. All callers that need to pick a tier explicitly use `alloc_on`. The legacy `alloc` exists only for source compatibility with paths that have not yet been threaded; new code must not use it.

### Cross-device boundary synchronization

`cudaMemcpyPeerAsync` (or the pinned-host bounce variant) issued on the source device's stream completes asynchronously — the destination device's stream must not consume the activation before the copy completes. v0 uses a per-boundary event:

```c
// At a (src_dev, dst_dev) boundary:
WITH_DEVICE(src_dev) {
    issue copy on g_gpu[src_dev].stream into dst tensor
    cudaEventRecord(boundary_event, g_gpu[src_dev].stream);
}
WITH_DEVICE(dst_dev) {
    cudaStreamWaitEvent(g_gpu[dst_dev].stream, boundary_event, 0);
    // subsequent kernels on dst_dev.stream are safe
}
```

A single reusable `cudaEvent_t boundary_event[DS4_MAX_GPUS]` per source device is allocated at init. The boundary cost is one `cudaEventRecord` + one `cudaStreamWaitEvent` — well under a microsecond. This pattern lets v0 stay correct without forcing a `cudaDeviceSynchronize` on every hop, and leaves the door open for v1 to overlap independent microbatches across the two streams.

For GPU→CPU and CPU→GPU boundaries, the existing host-pinned path's synchronization remains in force (the CPU code only reads the destination after the relevant copy completes).

Kernel-launch wrappers in `ds4_cuda.cu` either consume a device id from the input tensor or run inside a `WITH_DEVICE(d) { ... }` scope macro that does `cudaSetDevice(d)` + restore on exit.

### Selective per-device model cache

`ds4_gpu_set_model_map_range()` semantics are replaced. Instead, after the layer placement is known, the engine instructs each device's model cache to populate only the tensors that the placement assigns to it:

```c
// New API
int ds4_gpu_device_cache_tensors(
    int                       device_id,
    const ds4_tensor_range   *ranges,
    int                       n_ranges);
```

CPU-tier layers are not cached to any device — they execute against the existing mmap-backed CPU path. This is what makes the per-device budget meaningful at load time, not just at runtime.

### Cross-device activation hop

At init, probe peer access for every ordered pair:

```c
int peer_ok[DS4_MAX_GPUS][DS4_MAX_GPUS];
for (int i = 0; i < g_n_gpus; i++)
  for (int j = 0; j < g_n_gpus; j++)
    if (i != j) {
      int can_access = 0;
      cudaDeviceCanAccessPeer(&can_access, g_gpu[i].device_id, g_gpu[j].device_id);
      if (can_access) {
        cudaSetDevice(g_gpu[i].device_id);
        cudaError_t e = cudaDeviceEnablePeerAccess(g_gpu[j].device_id, 0);
        peer_ok[i][j] = (e == cudaSuccess || e == cudaErrorPeerAccessAlreadyEnabled);
      } else {
        peer_ok[i][j] = 0;
      }
    }
```

Each pair is decided independently — handles mixed topologies (NVLink clusters of 4 + PCIe between clusters, etc.).

Boundary kinds and mechanisms:

| Boundary | Mechanism |
|---|---|
| GPU → GPU, `peer_ok[src][dst]` | `cudaMemcpyPeerAsync` on the source's stream |
| GPU → GPU, not peer-capable | Staged through per-pair pinned-host bounce buffer: `cudaMemcpyAsync` D→H on source stream, then `cudaStreamWaitEvent` on a source-recorded event to gate the H→D `cudaMemcpyAsync` on the destination stream — host memory must not be reused until the D→H completes |
| GPU → CPU | Hidden state and any layer-boundary tensors materialized to host pinned; CPU path takes over from there |
| CPU → GPU | Hidden state and any boundary tensors uploaded to destination device's pinned buffer |

RTX 6000 Ada P2P over PCIe is driver-dependent; recent NVIDIA driver releases disabled P2P on some Ada SKUs. The pinned-host fallback ensures correctness regardless of probe result.

### Mixed CPU / GPU execution mechanics

CPU spill is an **execution-mode change**, not just a placement label.

- **KV cache ownership:** Layer L's KV state lives on L's tier — GPU layers use the existing GPU KV cache; CPU layers use the existing CPU KV cache. Both already exist in the codebase. Sessions allocate per-layer KV state at the layer's tier.
- **HC-shaped activations and ratio-4 indexer state:** at a GPU↔CPU boundary, the engine materializes the hidden state on the destination tier. For ratio-4 selective-compressed-attention layers, the indexer-side state stays on the layer's tier and is freshly computed there; it does not need to be transferred across the boundary because the boundary is between layers, not within a layer.
- **Batch prefill buffers:** carried per-tier. When prefill crosses a tier boundary, the activation matrix is hopped using the same mechanism as decode, sized to `chunk_tokens × hidden × dtype`.
- **Logits and output head:** logits are computed on the tier where the output head pseudo-layer lives. If that's CPU, logits are returned from CPU; otherwise they are read from the actual device that owns the output-head tier (which may or may not be device 0 depending on placement).
- **MTP (multi-token prediction) second model:** the codebase supports an optional MTP head (`e->mtp_model`, `e->mtp_ready`) used for speculative decoding. **v0 disables MTP when the placement uses more than one tier** (i.e. when `n_gpus > 1` *or* any layer is on CPU while at least one is on GPU). In multi-tier mode, `e->mtp_ready` is forced to false at engine create time and the existing non-MTP code paths handle inference. Single-tier modes (`--cuda` alone, single-GPU `--gpu-vram N`, CPU-only `--gpu-vram 0`) keep MTP behavior unchanged. Multi-GPU + MTP co-design is deferred to v1 (it interacts directly with the microbatch scheduler).
- **Session lifecycle:** sessions allocate per-layer state at the layer's assigned tier. `ds4_session_create` consults the placement table; existing CPU and GPU-graph session paths remain — v0 introduces a mixed-mode path that walks the placement table.

## Layer-packing algorithm

### Per-layer footprint

```
layer_bytes(L) = weight_bytes(L)         // attention QKV/O, MoE router, shared experts,
                                          //   routed experts at configured quant
              + kv_cache_bytes(L)         // varies by attention type:
                                          //   raw window (layers 0,1), ratio-4 (with indexer KV),
                                          //   ratio-128 (compressed only); computed at max_context
              + session_graph_bytes(L)    // per-layer prefill/decode graph tensors
                                          //   (chunk-sized batch activations, intermediate scratch)
```

### Per-device fixed overhead (counted once per device, not per layer)

```
device_overhead_bytes =
    cublas_workspace_bytes              // per-device cuBLAS handle scratch (default ~4 MB; tunable)
  + global_scratch_bytes                // device-side scratch pool the model uses outside per-layer work
  + q8_batch_cache_bytes                // q8 f16/f32 caches used by prefill batch path
  + logits_buffer_bytes                 // only on the device that owns the output head
  + (mtp_ready ? mtp_overhead_bytes : 0)        // mtp_ready is always false in multi-tier mode (see below); kept here for the single-tier case
  + (steering ? steering_tensors_bytes : 0)
  + safety_margin_bytes                 // default ~1 GB
```

The packer subtracts `device_overhead_bytes` from each per-device budget up front, then performs layer allocation against the remaining headroom.

### Embedding and output head as pseudo-layers

`compute_layer_placement` operates on `n_layers + 2` entries in **forward order**: index `0` is the embedding table, indices `1..n_layers` are the 43 transformer layers, and index `n_layers + 1` is the output head. All three kinds are sized like layers and assigned via the same monotonic greedy fill. Typically embedding lands on the first GPU and output head on the last GPU (or CPU when GPU budget is fully consumed by transformer layers).

### Contiguous greedy fill

```
budget[d] = gpu_vram[d] - device_overhead_bytes[d]   for d in 0..n_gpus-1
device_for_entry[E] = CPU                            for all E in 0..n_entries-1
d = 0
for E in 0..n_entries-1:
    while d < n_gpus and entry_bytes(E) > budget[d]:
        d += 1                                       // advance — never return
    if d < n_gpus:
        device_for_entry[E] = d
        budget[d] -= entry_bytes(E)
    else:
        device_for_entry[E] = CPU                    // and all subsequent stay CPU
```

This is strictly monotonic: once we move from tier `d` to `d+1` or `CPU`, no later entry returns to a lower tier. Yields a contiguous PP layout with at most `n_gpus` device boundaries.

If a single entry exceeds every per-device budget, it is placed on the CPU tier; per the monotonicity rule, every entry after it also goes to CPU. There is no load-time error path here — all layers can execute on CPU (the existing CPU path is the fallback), so the packer always produces a valid placement. The only failure modes are (a) invalid CLI inputs and (b) per-device overhead exceeding the device's budget before any layer is placed, both surfaced as configuration errors at parse / init time, not from the packer itself.

### Layout print

```
multi-GPU layout:
  GPU0: layers 0-21 + embedding   (38.4 / 40.0 GB)
  GPU1: layers 22-31              (11.7 / 12.0 GB)
  CPU : layers 32-42 + output head
peer access: GPU0<->GPU1 PEER-OK   (or PINNED-HOST FALLBACK)
```

## Control surface

```
ds4-cli MODEL.gguf --gpu-vram 40,40,40,40,40,40,40,40   # explicit, variadic, GB per device
ds4-cli MODEL.gguf --gpu-vram 40,12                     # 2-GPU case
ds4-cli MODEL.gguf --gpu-vram 40                        # explicit single-GPU
ds4-cli MODEL.gguf --gpu-vram auto                      # cudaMemGetInfo per visible device
ds4-cli MODEL.gguf --gpu-vram 0                         # CPU-only
ds4-cli MODEL.gguf --gpu-devices 0,2,5                  # restrict to listed device indices
```

- `--gpu-vram` accepts a comma-separated GB list, the literal `auto`, or a single number.
- `--gpu-vram 0` is special-cased before any per-device setup: it sets `n_gpus = 0`, skips CUDA initialization entirely, and runs the existing CPU-only path. Per-device overhead validation (below) does not apply because there are no devices.
- `--gpu-devices` is optional; default is "all visible per `cudaGetDeviceCount`".
- `--cuda` backwards compatibility: **`--cuda` alone always selects device 0 only**, even on a multi-GPU box. This preserves existing behavior exactly. Multi-GPU is strictly opt-in via `--gpu-vram` (with `auto` or an explicit list) or `--gpu-devices`.
- Same flags wired into `ds4-cli`, `ds4-server`, `ds4-agent`, `ds4-bench`. In `ds4-server` the layout is fixed at process start.

## Public API changes (`ds4_gpu.h`)

```c
typedef struct {
    int    device_indices[DS4_MAX_GPUS];   // CUDA device IDs to use
    size_t vram_bytes[DS4_MAX_GPUS];       // per-device budget
    int    n_gpus;                         // entries used
    size_t safety_margin_bytes;            // per-GPU reserve, default ~1 GB
} ds4_gpu_config;

int ds4_gpu_init_multi(const ds4_gpu_config *cfg);

// Replaces ds4_gpu_set_model_map_range for multi-GPU; the old API is kept as a
// thin compatibility shim that targets device 0 only.
int ds4_gpu_device_cache_tensors(
    int                       device_id,
    const ds4_tensor_range   *ranges,
    int                       n_ranges);
```

`ds4_gpu_init(...)` becomes a thin wrapper around `ds4_gpu_init_multi` for source compatibility, populating a single-device config.

## Internal CUDA API changes (`ds4_cuda.cu`)

- Remove hardcoded `cudaSetDevice(0)` at `ds4_gpu_init`; replace with init loop over `cfg->n_gpus`.
- Introduce `WITH_DEVICE(d) { ... }` scope macro and use it at the top of every public `ds4_gpu_*` entry point that previously assumed device 0.
- Per-device cuBLAS handles (one per `ds4_gpu_ctx`). `ds4_gpu_get_cublas(int device_id)` accessor.
- Per-device scratch / workspace pool. Replace the global scratch pointer with `g_gpu[d].scratch`.
- Per-device model cache. Replace the global tensor-data map range with `g_gpu[d].model_cache`, populated selectively via `ds4_gpu_device_cache_tensors`.
- One stream per device (already part of `ds4_gpu_ctx`). v0 dispatches sequentially, but per-device streams now keep v1 from rewiring this layer.

## Engine-layer changes (`ds4.c`)

- Engine creation accepts a `ds4_gpu_config`. The placement table is computed during `ds4_engine_create_v` and stored on the engine.
- `ds4_session_create` for graph-backed backends walks the placement table; for each layer, allocate per-layer state (KV cache, graph tensors) on the assigned tier. Existing pure-CPU and pure-GPU session paths remain; the new "mixed" path is the multi-tier walker.
- Prefill and decode loops in `ds4.c` consult `device_for_entry[L]` per layer, switch device contexts (via `WITH_DEVICE`), and perform a boundary hop when the destination tier changes.
- Logits collection at the end of the forward pass reads from whichever tier owns the output head.

## Testing strategy

| Test | Where | Acceptance |
|---|---|---|
| **Layer-packer unit test** | `tests/test_layer_pack.c` (CPU-only) | Synthetic per-layer sizes + budgets across N ∈ {1, 2, 8, 16}, with: all-fit, partial-spill, zero-budget on one GPU, mixed sizes, one entry too large for any single budget (the entry and everything after it spill to CPU), per-device overhead larger than budget (init-time config error), embedding-and-output-head pseudo-layer placement at indices 0 and `n_entries-1`. Asserts strict monotonicity in output. |
| **Cross-device tensor copy** | `tests/test_gpu_xfer.c` (needs 2 GPUs) | Allocate a tensor on each device, `ds4_gpu_tensor_copy` between them, read back and compare. Run with peer enabled and explicitly disabled via env var to exercise both paths. |
| **Numerical equivalence — all-GPU** | `192.168.50.231` | Same prompt run single-GPU and PP-2-GPU with **no CPU spill**, cuBLAS and TF32 settings pinned identically (`CUBLAS_COMPUTE_32F`, `TF32_DISABLED=1` or matching env). Final FP32 logits **bit-identical**. |
| **Numerical equivalence — with CPU spill** | `192.168.50.231` | Same prompt run single-GPU and PP-2-GPU with one or more layers forced to CPU. Acceptance: **top-1 token identical**, top-5 set identical, `max(abs(delta))` of logits below tolerance ε (target: 1e-3 normalized; ε locked from empirical runs). Bit-exact is not expected — CPU kernel reduction order differs. |
| **VRAM accounting** | `192.168.50.231` | After model load, `nvidia-smi` per-device "used" matches the engine's `g_gpu[d].used_bytes + device_overhead_bytes` within ±256 MB. Catches budget regressions and leaks. |
| **Placement-derived load budgets** | `192.168.50.231` | Build a placement, instrument `ds4_gpu_device_cache_tensors` calls, assert each device caches only the tensors the placement assigns to it (no full-map fallthrough). |
| **End-to-end bench** | `192.168.50.231` | `ds4-bench` reports prefill + gen tok/s at splits 40/12, 24/24, 12/40. Baseline recorded for v1 comparison. Numbers logged but not pass/fail gated. Single-GPU baseline also recorded for the latency-overhead comparison. |

## Implementation decomposition

Revised from 3 to 6 dev-team tasks based on the spec-review feedback that `mgpu-foundation` understated the changes required in `ds4.c` and `mgpu-integration` was too large for one PR.

### Wave 1 — parallel (3 tasks)

| Task ID | Scope | Touches |
|---|---|---|
| **mgpu-device-aware-cuda** | `g_gpu[]` table; per-device cuBLAS handles, scratch pool, stream, mem pool; `device_id` field on `ds4_gpu_tensor`; `WITH_DEVICE` macro applied to existing `ds4_gpu_*` entry points; NxN peer probe matrix; cross-device tensor copy primitive (peer + pinned-host fallback); `ds4_gpu_init_multi`. **No behavior change at N=1** — placement is unchanged, model cache is still global. | `ds4_cuda.cu`, `ds4_gpu.h` |
| **mgpu-selective-model-cache** | Replace `ds4_gpu_set_model_map_range` semantics with `ds4_gpu_device_cache_tensors`. Per-device model-cache region. Old API kept as a single-device shim. Caller still passes "everything → device 0" by default; this task ships the mechanism, not the wiring. | `ds4_cuda.cu`, `ds4_gpu.h` |
| **mgpu-layer-packer** | Pure C function `compute_layer_placement(entries[], gpu_budgets[], device_overhead[], n_gpus) -> device_for_entry[]`. Embedding + output head as pseudo-layers. Strict monotonic-contiguous fill. Unit tests covering N=1,2,8,16 and edge cases. CPU-only, no CUDA dependency. | new `ds4_layer_pack.{c,h}`, `tests/test_layer_pack.c` |

The three wave-1 tasks touch disjoint enough files (the two CUDA tasks share `ds4_cuda.cu` and `ds4_gpu.h`; expected merge conflicts are small and localized to additions, but PRs will land sequentially within the wave to keep merges clean). The layer-packer is fully disjoint.

### Wave 2 — parallel (2 tasks)

| Task ID | Scope | Touches |
|---|---|---|
| **mgpu-graph-session-placement** | Engine accepts `ds4_gpu_config`. Placement table computed at `ds4_engine_create_v`. **Replaces the existing startup model-cache call (which targets device 0 only) with placement-derived per-device `ds4_gpu_device_cache_tensors` calls — one per GPU tier, each with the tensor ranges its placement assigns.** Per-layer KV / graph-tensor allocation walks the placement. Prefill and decode loops perform boundary hops at tier transitions; the mixed-tier session path materializes hidden state across GPU↔CPU boundaries. Logits collection from the output-head tier. | `ds4.c`, `ds4.h` |
| **mgpu-cli-wiring** | `--gpu-vram` and `--gpu-devices` flags across `ds4_cli.c`, `ds4_server.c`, `ds4_agent.c`, `ds4_bench.c`. Parsing, `auto` detection via `cudaMemGetInfo`. Build a `ds4_gpu_config` and pass to engine create. Layout print on load. `--cuda` preserved as device-0-only. Makefile / build verification. | `ds4_cli.c`, `ds4_server.c`, `ds4_agent.c`, `ds4_bench.c`, `Makefile` |

Wave-2 tasks share no files. Both depend on all three wave-1 tasks being merged.

### Wave 3 — sequential (1 task)

| Task ID | Scope | Touches |
|---|---|---|
| **mgpu-bench-qa** | Run the full test matrix on `192.168.50.231` at splits 40/12, 24/24, 12/40, plus a forced-spill config. Record numerical-equivalence verdicts (bit-exact for all-GPU, within-tolerance for spill), VRAM accounting deltas, peer-vs-host-bounce path, tok/s numbers. Update `docs/` with the v0 baseline report. | `tests/`, `docs/` |

Wave 3 depends on wave 2 being merged.

## Verification on hardware

The remote test box `192.168.50.231` is the target for all GPU-dependent tests. Pre-dispatch the orchestrator should verify:

- `ssh 192.168.50.231 nvidia-smi` shows both RTX 6000 Ada GPUs with sufficient free VRAM. Current state: GPU0 ~48.7 GB free, GPU1 ~18 GB free (other workload using ~31 GB).
- The build toolchain (CUDA, `nvcc`, codex CLI) is available on the box.

If GPU1 is occupied during a test window, that bench point uses whatever free budget is available; the integration task's QA step records what was used so results stay reproducible.

## Future work pointers

- **v1 — Microbatch scheduler:** introduce a request scheduler in `ds4_server.c` maintaining multiple in-flight microbatches; GPU0 computes layer 0–K for request `i+1` while GPU1 computes layer K+1–N for request `i`. Requires per-device streams (already in place after wave 1), per-request layer-progress tracking, and async completion handling.
- **v2 — Expert-parallel:** spread routed experts within a layer across devices. Routing-aware dispatch sends each token's activations to the GPU(s) hosting its top-K experts. Co-design with v1.
- **NUMA-aware host-bounce buffers:** for N ≥ 4 across NUMA nodes, allocate pinned bounce buffers on the correct node.
- **Placement hint file:** allow `--gpu-layout layout.json` for power users wanting non-greedy / non-contiguous layouts.
- **Prefill-overlap optimization:** chunk-pipelined prefill across two devices for first-token-latency wins even before v1's microbatch scheduler.
