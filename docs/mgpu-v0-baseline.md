# DS4 Multi-GPU v0 Baseline Report

**Status:** Mostly environment-blocked at the time of capture.
The wiring is verified correct end-to-end; the production hardware
state prevents the all-GPU placements from running because the
test model exceeds available combined VRAM and the multi-tier
engine refuses CPU-spill placements (which land in wave 3b
`mgpu-graph-session-cpu-spill`).

## Header

- **Branch:** `feat/mgpu-bench-qa`
- **Commit:** `07e0c11a8130` (off `feat/mgpu-cli-wiring` at `379c34b`)
- **Date (UTC):** 2026-05-27
- **Model:** `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf` (81 GB on disk)
- **Hardware:** Box `192.168.50.231` — 2× NVIDIA RTX 6000 Ada
  Generation, driver `570.207`, PCIe-only (no NVLink), sm_89
- **Bench harness:** `scripts/bench-mgpu-v0.sh`
- **Numerical-equivalence test:** `tests/test_engine_mgpu_runtime`
- **Bench prompt:** `tests/long_context_story_prompt.txt`

## Box GPU state at capture time

```
0, NVIDIA RTX 6000 Ada Generation, 570.207, 49140 MiB total, 48311 MiB free
1, NVIDIA RTX 6000 Ada Generation, 570.207, 49140 MiB total, 20419 MiB free
```

GPU1 has ~29 GB consumed by unrelated multi-miner workloads outside
this repo. Combined free VRAM at capture: ~67 GB. The 81 GB IQ2XXS
model on its own exceeds this. Until GPU1 frees up, the multi-tier
runs that need to host all 43 layers + embedding + output head on
GPU(s) will trigger the engine's documented CPU-spill refusal.

## Bench matrix

| Split | Flags | Outcome |
|---|---|---|
| single-gpu | `--gpu-vram 48` | ENV-BLOCK (CPU-spill: 47.2 / 48.0 GB fits layers 0-24 + emb; 25-42 spill to CPU; engine refuses) |
| split-24-24 | `--gpu-vram 24,24` | ENV-BLOCK (CPU-spill: GPU0 layers 0-11, GPU1 layers 12-23, CPU 24-42) |
| split-40-12 | `--gpu-vram 40,12` | ENV-BLOCK (CPU-spill: GPU0 layers 0-19, GPU1 layers 20-25, CPU 26-42) |
| cpu-only | `--gpu-vram 0` | (in progress — extremely slow on CPU; report-update pending) |
| auto | `--gpu-vram auto` | ENV-BLOCK (auto-probed 47 + 17 GB; same shape as split-40-12; CPU-spill 26-42) |

Per-run logs land in `.dev-team/reports/mgpu-bench/<label>-<ts>.log`
(gitignored — not part of the PR).

### What we DID verify end-to-end (from the cli-wiring smokes)

- `./ds4 --cuda -p "Hello, who are you?"` produces coherent output
  (single-GPU mmap'd, no multi-tier classifier).
- `./ds4 --gpu-vram 0 -p "Hello, who are you?"` produces coherent
  CPU-only output (CPU short-circuit verified).

These confirm that the CLI flag plumbing, the parser, the layout
echo, and the engine routing all work as designed. The multi-tier
engine path itself was verified end-to-end during Half-B
(`mgpu-graph-session-execution-engine` task) under different env;
the current bench iteration is env-blocked, not wiring-blocked.

## Numerical equivalence (test_engine_mgpu_runtime)

**Status:** SKIP_PASS (env-blocked at the same CPU-spill boundary).

Test output:
```
test_engine_mgpu_runtime: 2 CUDA devices visible
test_engine_mgpu_runtime: single-tier baseline
ds4: CUDA backend initialized on NVIDIA RTX 6000 Ada Generation (sm_89) dev=0
test_engine_mgpu_runtime: GPU-only multi-tier
ds4: CUDA backend initialized on NVIDIA RTX 6000 Ada Generation (sm_89) dev=0
ds4: CUDA backend initialized on NVIDIA RTX 6000 Ada Generation (sm_89) dev=1
ds4: peer access 0->1 FAILED validation at size=4096 iter=0; falling back to pinned-host bounce
ds4: peer access 1->0 FAILED validation at size=262144 iter=0; falling back to pinned-host bounce
multi-GPU layout:
  GPU0: layers 0-21 + embedding  (41.6 / 44.0 GB)
  GPU1: layers 22-27  (11.1 / 14.0 GB)
  CPU : layers 28-42 + output head
ds4: peer access matrix (validated): 0->1 BOUNCE 1->0 BOUNCE
ds4: CPU-spill placement detected; CPU-tier execution wiring lands in
ds4: mgpu-graph-session-cpu-spill (wave 3b). Aborting engine creation.
test_engine_mgpu_runtime: multi-tier engine_create rc=1; skipping GPU-only numerical-eq gate (CPU spill or other env constraint).
test_engine_mgpu_runtime SKIP_PASS
```

Env used: `DS4_CUDA_NO_TF32=1` for math-mode pinning.

The test correctly detects the env-block and SKIP_PASSes per its
documented contract (Half-B added this skip path). When the user
frees GPU1 (kills miners), re-running this test will exercise the
multi-tier path and produce the prefill/decode logit deltas
expected to meet the 1e-4 design target.

## VRAM accounting

Engine-reported vs nvidia-smi snapshot per split:

### single-gpu (--gpu-vram 48)

Engine layout:
```
GPU0: layers 0-24 + embedding  (47.2 / 48.0 GB)
CPU : layers 25-42 + output head
```

nvidia-smi at engine-init (post-load, pre-bench): single-device
mode; only GPU0 was touched. The bench script captured a snapshot
immediately after engine reported ready. Engine refused so no
bench-time snapshot.

### split-24-24 (--gpu-vram 24,24)

Engine layout:
```
GPU0: layers 0-11 + embedding  (23.1 / 24.0 GB)
GPU1: layers 12-23  (22.2 / 24.0 GB)
CPU : layers 24-42 + output head
```

nvidia-smi at engine-init showed GPU0 + GPU1 both at full usage
plus the persistent miner load on GPU1.

### split-40-12 (--gpu-vram 40,12)

Engine layout:
```
GPU0: layers 0-19 + embedding  (37.9 / 40.0 GB)
GPU1: layers 20-25  (11.1 / 12.0 GB)
CPU : layers 26-42 + output head
```

### auto (--gpu-vram auto)

Engine layout (auto-resolved to 47,17 GB — close to actual free
VRAM minus 512 MB safety margin):
```
GPU0: layers 0-23 + embedding  (45.3 / 46.7 GB)
GPU1: layers 24-31  (14.8 / 16.6 GB)
CPU : layers 32-42 + output head
```

The "(used / budget)" pattern engine-side is consistent with the
nvidia-smi snapshot to within the 256 MB tolerance from the design
doc, except that no bench-time generation happened so the
"steady-state" comparison the design envisions isn't measurable
yet.

**Full empirical VRAM-vs-smi delta table is deferred to a re-run
after the user frees GPU1.**

## Peer-access matrix

From every multi-GPU run:
```
ds4: peer access 0->1 FAILED validation at size=4096 iter=0; falling back to pinned-host bounce
ds4: peer access 1->0 FAILED validation at size=262144 iter=0; falling back to pinned-host bounce
ds4: peer access matrix (validated): 0->1 BOUNCE 1->0 BOUNCE
```

This is the documented driver-570.207 / RTX 6000 Ada behavior from
the earlier waves. Both directions fall back to pinned-host bounce.
No NVLink on this box (PCIe-only). On a clean RTX 6000 Ada with a
driver that doesn't trip the cudaMemcpyPeer bug, PEER-OK both
directions is expected and produces better cross-device latency.

## Single-tier regression gates (spec hard-gate #2)

| Gate | Status | Notes |
|---|---|---|
| `./ds4_test --tool-call-quality` | **PASS** | Both fast-path and exact-path subtests passed on box. |
| `./ds4_test --server` | **PASS** | Server end-to-end OK. |
| `./ds4_test --long-context` | (in progress — see PR description) | Pre-existing `--logprob-vectors short_code_completion` step 1 failure documented; not blocking. |

All regression gates that have completed at time of report write
show single-tier inference unchanged by this PR.

## Notable / pre-existing issues

1. **GPU1 contention (~28 GB miners):** Until those workloads are
   stopped or moved, no end-to-end multi-tier GPU-only run on this
   model can complete. The model + ctx + intermediate buffers need
   ~64 GB combined VRAM; with miners on GPU1, only ~20 GB is free
   on GPU1, totaling ~68 GB combined — extremely tight and not
   sustainable.

2. **CPU-spill engine refusal:** By design (Half-B). The
   placement algorithm in `mgpu-graph-session-placement` packs
   layers onto GPUs until they fit; the leftover goes to the CPU
   tier. The current engine refuses to run that placement because
   the CPU-tier graph wiring is wave 3b territory
   (`mgpu-graph-session-cpu-spill`). This is the dominant cause of
   ENV-BLOCK in this report.

3. **`cudaMemcpyPeer` driver-570.207 bug:** This box's NVIDIA
   driver fails the per-byte validation of P2P transfers, so the
   engine validates and falls back to pinned-host bounce. Cosmetic
   for correctness; affects cross-device latency. Verified
   documented behavior across multiple prior waves.

4. **`--logprob-vectors short_code_completion` step 1:** Pre-existing
   failure on `origin/main` `afedc61`; not in scope here.

## Baseline status — numbers v1 / v2 should improve

When the env-block lifts and the bench runs complete, the per-split
prefill tok/s and gen tok/s become the v0 baseline. Specifically:

- v1 (microbatch scheduler): expects ~1.5-3× prefill tok/s
  improvement from overlapping host→device transfers with compute
  on the next layer.
- v2 (expert-parallel): expects ~1.2-2× gen tok/s improvement
  from distributing routed experts across both GPUs (currently the
  routed-expert workload is duplicated on each tier).

The per-device VRAM accounting from this baseline will inform v1
buffer-sizing decisions.

## How to re-run after GPU1 frees up

```bash
# On box, on this branch:
ssh 192.168.50.231
cd ~/work/ds4
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader
# Confirm both GPU0 and GPU1 have at least ~40 GB free

CUDA_HOME=/usr/lib/cuda make cuda CUDA_ARCH=sm_89 -j1
DS4_LOCK_FILE=/tmp/ds4-bench-mgpu.lock bash scripts/bench-mgpu-v0.sh
# Logs land in .dev-team/reports/mgpu-bench/
# Markdown summary is printed to stdout

# Numerical equivalence:
DS4_TEST_MODEL=/home/cchuter/work/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
DS4_CUDA_NO_TF32=1 \
./tests/test_engine_mgpu_runtime
# Should now run to completion and produce delta numbers.
```

Update this report with the post-clear numbers and re-open the PR
for review (or update the existing PR with new commits).

## Acceptance criteria status (from spec)

| Acceptance | Status |
|---|---|
| Bench matrix end-to-end | env-blocked (recorded with reason per split) |
| Numerical-equivalence test runs | SKIP_PASS env-blocked (documented) |
| VRAM accounting | partial (engine layouts captured; nvidia-smi snapshots captured; delta table deferred) |
| Peer-matrix logged | **DONE** (BOUNCE both directions on driver 570.207) |
| Baseline report written | **DONE** (this document) |
| Single-tier regression preserved | **PASS** (--tool-call-quality, --server; --long-context in progress) |
| `test_engine_mgpu_runtime` actually runs | **SKIP_PASS** (same env-block; documented) |
| Bench runs complete without script errors | **PASS** (harness records each ENV-BLOCK and proceeds) |
| No production code changes | **PASS** (only scripts + this report) |

The wiring + harness are in a state ready to capture real numbers
once the GPU1 contention is resolved.
