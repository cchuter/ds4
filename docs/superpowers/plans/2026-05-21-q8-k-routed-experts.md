# Q8K Routed Experts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a true `DeepSeek-V4-Flash-Q8KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2.gguf` with routed expert tensors stored as GGUF type `q8_K`.

**Architecture:** Phase 2 builds on the pushed Q8_0 branch. The quantizer first learns to emit byte-compatible `q8_K` blocks for routed experts. The runtime then treats `Q8_K` as a stored expert weight type, with CPU dot paths and Metal routed MoE dispatch for decode and prefill.

**Tech Stack:** C99, Objective-C Metal host code, Metal Shading Language, local GGUF writer, DeepSeek V4 Flash safetensors under `/Users/cchuter/models/DeepSeek-V4-Flash`.

---

## File Structure

- `gguf-tools/quants.c`: add `q8_K` as an output quantization target and implement the `block_q8_K` writer.
- `gguf-tools/README.md`: document the Q8_K expert generation command and distinguish it from Q8_0.
- `gguf-tools/deepseek4-quantize.c`: update help text and use existing target planning with the new `ds4q_can_quantize(Q8_K)` behavior.
- `ds4.c`: add stored `Q8_K` expert validation, CPU dot kernels, routed MoE single-token decode, allocation-free decode, and batch prefill branches.
- `metal/moe.metal`: add `block_q8_K`, dequantization, routed matvec/matmul template exports, and optional direct sum6/pair kernels only where they are needed for correctness/performance.
- `ds4_metal.m`: add the Metal tensor enum and pipeline dispatch for `Q8_K` routed experts.
- `README.md`: update the local experiment section with the true Q8_K model command and smoke-test expectations.

## Task 1: Quantizer Q8_K Output

**Files:**
- Modify: `gguf-tools/quants.c`
- Modify: `gguf-tools/deepseek4-quantize.c`
- Modify: `gguf-tools/README.md`

- [x] **Step 1: Run the current red dry-run**

Run:

```sh
make -C gguf-tools
./gguf-tools/deepseek4-quantize \
  --hf /Users/cchuter/models/DeepSeek-V4-Flash \
  --template gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf \
  --out gguf/DeepSeek-V4-Flash-Q8KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2.gguf \
  --experts q8_K \
  --dry-run
```

Expected before this task is implemented: the command fails because `q8_K` is named but not quantizable.

Completed by `8add6c4`: red dry-run exited `1` with `error: unsupported planned tensor type`.

- [x] **Step 2: Add the q8_K quantizer**

In `gguf-tools/quants.c`, change the `DS4Q_TYPE_Q8_K` trait from non-quantizable to quantizable:

```c
[DS4Q_TYPE_Q8_K]    = { "q8_K",  QK_K, 292, true,  false },
```

Add this local block writer near `ds4q_quantize_q8_0()`:

```c
static size_t ds4q_quantize_q8_K(const float *src, void *dst, int64_t start,
                                 int64_t nrows, int64_t ncols) {
    if (ncols % QK_K != 0) abort();
    const size_t row_size = ds4q_row_size(DS4Q_TYPE_Q8_K, ncols);
    const int64_t start_row = start / ncols;
    const int64_t nblocks = ncols / QK_K;
    uint8_t *out = (uint8_t *)dst + (size_t)start_row * row_size;

    for (int64_t row = 0; row < nrows; row++) {
        const float *x = src + row * ncols;
        uint8_t *row_out = out + (size_t)row * row_size;
        for (int64_t b = 0; b < nblocks; b++) {
            uint8_t *block = row_out + (size_t)b * ds4q_type_traits[DS4Q_TYPE_Q8_K].type_size;
            float max = 0.0f;
            float amax = 0.0f;
            for (int j = 0; j < QK_K; j++) {
                const float ax = fabsf(x[j]);
                if (ax > amax) {
                    amax = ax;
                    max = x[j];
                }
            }
            float d = 0.0f;
            int8_t qs[QK_K];
            int16_t bsums[QK_K / 16];
            memset(qs, 0, sizeof(qs));
            memset(bsums, 0, sizeof(bsums));
            if (amax != 0.0f) {
                const float iscale = -127.0f / max;
                for (int j = 0; j < QK_K; j++) {
                    int v = ds4q_nearest_int(iscale * x[j]);
                    if (v > 127) v = 127;
                    if (v < -128) v = -128;
                    qs[j] = (int8_t)v;
                }
                for (int j = 0; j < QK_K / 16; j++) {
                    int sum = 0;
                    for (int i = 0; i < 16; i++) sum += qs[j * 16 + i];
                    bsums[j] = (int16_t)sum;
                }
                d = 1.0f / iscale;
            }
            memcpy(block, &d, sizeof(d));
            memcpy(block + 4, qs, sizeof(qs));
            memcpy(block + 4 + sizeof(qs), bsums, sizeof(bsums));
            x += QK_K;
        }
    }
    return (size_t)nrows * row_size;
}
```

Then add the switch branch in `ds4q_quantize_chunk()`:

```c
case DS4Q_TYPE_Q8_K:
    return ds4q_quantize_q8_K(src, dst, start, nrows, ncols);
```

- [x] **Step 3: Update quantizer help and docs**

In `gguf-tools/deepseek4-quantize.c`, update the type examples line:

```c
printf("\nTYPE examples: f16, f32, bf16, q8_0, q8_K, q4_k, q2_k, iq2_xxs\n");
```

In `gguf-tools/README.md`, add the Q8_K command after the Q4 example:

````markdown
True Q8_K routed experts:

```sh
gguf-tools/deepseek4-quantize \
  --hf /Users/cchuter/models/DeepSeek-V4-Flash \
  --template gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf \
  --out gguf/DeepSeek-V4-Flash-Q8KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2.gguf \
  --experts q8_K \
  --threads 8
```
````

- [x] **Step 4: Run quantizer verification**

Run:

```sh
make -C gguf-tools
./gguf-tools/deepseek4-quantize \
  --hf /Users/cchuter/models/DeepSeek-V4-Flash \
  --template gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf \
  --out gguf/DeepSeek-V4-Flash-Q8KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2.gguf \
  --experts q8_K \
  --dry-run
```

Expected: `type_changes: 129`, and every `type_change:` line for routed expert tensors ends in `-> q8_K`.

Completed by `8add6c4` and docs cleanup `4733071`: `make -C gguf-tools`
passed, q8_K dry-run reported `type_changes: 129`, a strict type-change check
found no bad type changes, and `git diff --check` passed.

- [x] **Step 5: Commit quantizer support**

Run:

```sh
git add gguf-tools/quants.c gguf-tools/deepseek4-quantize.c gguf-tools/README.md
git commit -m "feat: emit q8_k routed expert tensors"
```

## Task 2: CPU Q8_K Dot Kernels

**Files:**
- Modify: `ds4.c`

- [x] **Step 1: Add stored Q8_K type admission**

In `ds4.c`, add `DS4_TENSOR_Q8_K = 15` to the tensor enum and route helpers:

```c
DS4_TENSOR_Q8_K     = 15,
```

```c
static bool tensor_is_routed_expert_type(uint32_t type) {
    return type == DS4_TENSOR_Q8_0 ||
           type == DS4_TENSOR_Q8_K ||
           type == DS4_TENSOR_IQ2_XXS ||
           type == DS4_TENSOR_Q2_K ||
           type == DS4_TENSOR_Q4_K;
}
```

```c
case DS4_TENSOR_Q8_K:    return sizeof(block_q8_K);
```

- [x] **Step 2: Add CPU Q8_K x Q8_K dot products**

Add these helpers near `ds4_vec_dot_q2_K_q8_K()`:

```c
static void ds4_vec_dot_q8_K_q8_K(int n, float *s, const block_q8_K *x, const block_q8_K *y) {
    const int nb = n / QK_K;
    float sum = 0.0f;
    for (int b = 0; b < nb; b++) {
        const int8_t *xq = x[b].qs;
        const int8_t *yq = y[b].qs;
        int32_t isum = 0;
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
        const int32x4_t zero = vdupq_n_s32(0);
        int32x4_t acc = zero;
        for (int j = 0; j < QK_K; j += 16) {
            acc = vdotq_s32(acc, vld1q_s8(xq + j), vld1q_s8(yq + j));
        }
        isum = vaddvq_s32(acc);
#else
        for (int j = 0; j < QK_K; j++) isum += (int32_t)xq[j] * (int32_t)yq[j];
#endif
        sum += x[b].d * y[b].d * (float)isum;
    }
    *s = sum;
}

static void ds4_vec_dot_q8_K_pair_q8_K(
        int n,
        float *s0,
        float *s1,
        const block_q8_K *x0,
        const block_q8_K *x1,
        const block_q8_K *y) {
    const int nb = n / QK_K;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    for (int b = 0; b < nb; b++) {
        const int8_t *xq0 = x0[b].qs;
        const int8_t *xq1 = x1[b].qs;
        const int8_t *yq = y[b].qs;
        int32_t isum0 = 0;
        int32_t isum1 = 0;
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
        const int32x4_t zero = vdupq_n_s32(0);
        int32x4_t acc0 = zero;
        int32x4_t acc1 = zero;
        for (int j = 0; j < QK_K; j += 16) {
            const int8x16_t yv = vld1q_s8(yq + j);
            acc0 = vdotq_s32(acc0, vld1q_s8(xq0 + j), yv);
            acc1 = vdotq_s32(acc1, vld1q_s8(xq1 + j), yv);
        }
        isum0 = vaddvq_s32(acc0);
        isum1 = vaddvq_s32(acc1);
#else
        for (int j = 0; j < QK_K; j++) {
            isum0 += (int32_t)xq0[j] * (int32_t)yq[j];
            isum1 += (int32_t)xq1[j] * (int32_t)yq[j];
        }
#endif
        sum0 += x0[b].d * y[b].d * (float)isum0;
        sum1 += x1[b].d * y[b].d * (float)isum1;
    }
    *s0 = sum0;
    *s1 = sum1;
}
```

- [x] **Step 3: Add CPU expert mid and down helpers**

Add Q8_K versions alongside the existing `matvec_iq2_xxs_*` and `matvec_q8_0_*` helpers:

```c
typedef struct {
    float *mid;
    const uint8_t *gate_base[DS4_N_EXPERT_USED];
    const uint8_t *up_base[DS4_N_EXPERT_USED];
    const block_q8_K *xq;
    float expert_weight[DS4_N_EXPERT_USED];
    float clamp;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t gate_row_bytes[DS4_N_EXPERT_USED];
    uint64_t up_row_bytes[DS4_N_EXPERT_USED];
    int n_expert;
} matvec_q8_k_mid_ctx;
```

Implement `matvec_q8_k_mid_worker()`, `matvec_q8_k_experts_mid_prequant()`, `matvec_q8_k_accum_worker()`, and `matvec_q8_k_experts_accum_prequant()` by matching the Q8_0 helper structure but replacing `dot_q8_0_row_pair()` and `dot_q8_0_row()` with `ds4_vec_dot_q8_K_pair_q8_K()` and `ds4_vec_dot_q8_K_q8_K()`.

- [x] **Step 4: Build CPU**

Run:

```sh
make cpu
```

Expected: CPU binaries build without warnings from the new Q8_K functions.

Completed by `e38b6c2`: `make cpu` passed and `git diff --check` was clean.
Spec and code-quality reviews approved the Q8_K dot math, scalar/NEON parity,
helper validation, and lack of routed dispatch wiring.

- [x] **Step 5: Commit CPU dot kernels**

Run:

```sh
git add ds4.c
git commit -m "feat: add cpu q8_k routed expert kernels"
```

## Task 3: CPU Q8_K Routed Control Flow

**Files:**
- Modify: `ds4.c`

- [x] **Step 1: Add Q8_K routed type detection**

In `layer_routed_moe_one()`, `layer_routed_moe_one_prealloc()`, `layer_routed_moe_batch()`, and token-parallel routed contexts, add:

```c
const bool routed_q8_k =
    layer->ffn_gate_exps->type == DS4_TENSOR_Q8_K &&
    layer->ffn_up_exps->type == DS4_TENSOR_Q8_K &&
    layer->ffn_down_exps->type == DS4_TENSOR_Q8_K;
```

Validation for Q8_K uses `QK_K` alignment:

```c
if (routed_q8_k) {
    if (trace) ds4_die("Q8_K routed trace mode is not supported");
    if (expert_in_dim % QK_K != 0) ds4_die("Q8_K expert input is not QK_K aligned");
    if (down_in_dim != DS4_N_FF_EXP || down_in_dim % QK_K != 0) {
        ds4_die("Q8_K expert input has an unexpected layout");
    }
}
```

- [x] **Step 2: Wire single-token CPU Q8_K**

In `layer_routed_moe_one()`, allocate `block_q8_K *xq` and `block_q8_K *midq` for Q8_K exactly as the existing IQ2/Q2 branch does. Add this branch before the IQ2/Q2 branch:

```c
} else if (routed_q8_k && !trace) {
    matvec_q8_k_experts_mid_prequant(mid_all, model,
                                     layer->ffn_gate_exps,
                                     layer->ffn_up_exps,
                                     xq,
                                     selected,
                                     expert_weight,
                                     DS4_N_EXPERT_USED,
                                     clamp);
    for (int i = 0; i < DS4_N_EXPERT_USED; i++) {
        ds4_quantize_row_q8_K(mid_all + (uint64_t)i * down_in_dim,
                              midq + (uint64_t)i * (down_in_dim / QK_K),
                              (int64_t)down_in_dim);
    }
    matvec_q8_k_experts_accum_prequant(out, model, layer->ffn_down_exps,
                                       midq, selected, DS4_N_EXPERT_USED);
```

- [x] **Step 3: Wire allocation-free CPU decode**

In `layer_routed_moe_one_prealloc()`, reuse existing `xq` and `midq` scratch for Q8_K. Add this branch after `ds4_quantize_row_q8_K(x, xq, ...)` and before IQ2/Q2 dispatch:

```c
if (routed_q8_k) {
    matvec_q8_k_experts_mid_prequant(mid_all, model,
                                     layer->ffn_gate_exps,
                                     layer->ffn_up_exps,
                                     xq,
                                     selected,
                                     expert_weight,
                                     DS4_N_EXPERT_USED,
                                     clamp);
    for (int i = 0; i < DS4_N_EXPERT_USED; i++) {
        ds4_quantize_row_q8_K(mid_all + (uint64_t)i * down_in_dim,
                              midq + (uint64_t)i * (down_in_dim / QK_K),
                              (int64_t)down_in_dim);
    }
    matvec_q8_k_experts_accum_prequant(out, model, layer->ffn_down_exps,
                                       midq, selected, DS4_N_EXPERT_USED);
    (void)il;
    return;
}
```

- [x] **Step 4: Wire CPU batch prefill**

Add Q8_K batch contexts by matching the existing IQ2/Q2 batch path:

```c
typedef struct {
    float *mid;
    const uint8_t *gate_base[DS4_N_EXPERT];
    const uint8_t *up_base[DS4_N_EXPERT];
    const block_q8_K *xq;
    const ds4_expert_pair *pairs;
    const uint32_t *pair_ids;
    const uint32_t *expert_offset;
    const uint32_t *active_expert;
    const float *pair_weight;
    float clamp;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t xq_blocks;
    uint64_t gate_row_bytes[DS4_N_EXPERT];
    uint64_t up_row_bytes[DS4_N_EXPERT];
} matvec_q8_k_batch_mid_ctx;
```

Implement `matvec_q8_k_batch_mid_worker()` and `matvec_q8_k_batch_accum_rows_worker()` by matching the IQ2/Q2 batch workers and swapping in the Q8_K dot helpers.

- [x] **Step 5: Update routed quant bits**

In `ds4_engine_routed_quant_bits()`, make `Q8_K` report as 8:

```c
case DS4_TENSOR_Q8_0:
case DS4_TENSOR_Q8_K:
    return 8;
```

- [x] **Step 6: Build CPU and commit**

Run:

```sh
make cpu
```

Expected: CPU binaries build cleanly.

Completed by `ae66bc3` with cleanup `dc6eff1`: `make cpu` and
`git diff --check` passed. Spec review approved the CPU routed Q8_K control
flow, and code-quality re-review approved the scratch guard cleanup and removal
of dead token-parallel plumbing.

Commit:

```sh
git add ds4.c
git commit -m "feat: route q8_k experts on cpu"
```

## Task 4: Metal Q8_K Routed Experts

**Files:**
- Modify: `metal/moe.metal`
- Modify: `ds4_metal.m`

- [x] **Step 1: Add Metal Q8_K block and dequantization**

In the Metal source prelude in `ds4_metal.m`, add:

```metal
struct block_q8_K {
    float d;
    int8_t qs[QK_K];
    int16_t bsums[QK_K / 16];
};
```

In `metal/moe.metal`, add:

```metal
template <typename type4x4>
void dequantize_q8_K(device const block_q8_K *xb, short il, thread type4x4 &reg) {
    device const int8_t *qs = xb->qs + 16 * il;
    const float d = xb->d;
    float4x4 reg_f;
    for (int i = 0; i < 16; i++) {
        reg_f[i / 4][i % 4] = (float)qs[i] * d;
    }
    reg = (type4x4)reg_f;
}
```

- [x] **Step 2: Export routed matmul pipelines**

In `metal/moe.metal`, add host-visible Q8_K matmul templates next to the existing Q2/Q4/IQ2 templates:

```metal
template [[host_name("kernel_mul_mm_id_q8_K_f32")]] kernel mul_mm_id
kernel_mul_mm_id<half, half4x4, simdgroup_half8x8,
                 half, half2x4, simdgroup_half8x8,
                 block_q8_K, QK_NL, dequantize_q8_K,
                 float, float4x4, float, float2x4>;

template [[host_name("kernel_mul_mm_id_q8_K_f16")]] kernel mul_mm_id_f16_rhs
kernel_mul_mm_id<half, half4x4, simdgroup_half8x8,
                 half, half2x4, simdgroup_half8x8,
                 block_q8_K, QK_NL, dequantize_q8_K,
                 half, half4x4, half, half2x4>;
```

- [x] **Step 3: Add decode matvec pipeline**

Use the generic routed matvec template by adding a Q8_K implementation. Add `N_R0_Q8_K` near the other routed constants:

```metal
#define N_R0_Q8_K 2
```

Add the Q8_K matvec implementation and host export. The implementation can use a direct `float` activation dot like Q4/Q2 do, but each weight value is `(float)x[ib].qs[j] * x[ib].d`.

Host export:

```metal
template [[host_name("kernel_mul_mv_id_q8_K_f32")]] kernel kernel_mul_mv_id_q_t
kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q8_K_f32_impl<N_R0_Q8_K>>>;
```

- [x] **Step 4: Add Metal host type dispatch**

In `ds4_metal.m`, add:

```objc
DS4_METAL_TENSOR_Q8_K = 15,
```

Then update the routed dispatch helpers:

```objc
case DS4_METAL_TENSOR_Q8_K:
    return ds4_gpu_get_mul_mv_pipeline("kernel_mul_mv_id_q8_K_f32", 2);
```

```objc
case DS4_METAL_TENSOR_Q8_K:
    return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q8_K_f32", false);
```

```objc
case DS4_METAL_TENSOR_Q8_K:
    return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q8_K_f16", false);
```

For Q8_K, use the same routed MV shape as Q2/Q4:

```objc
static NSUInteger ds4_gpu_routed_mv_nsg(uint32_t type) {
    return type == DS4_METAL_TENSOR_Q8_0 ? 4u : 2u;
}

static bool ds4_gpu_routed_mv_rows_per_group_is_nr0(uint32_t type) {
    return type == DS4_METAL_TENSOR_Q8_0;
}
```

- [x] **Step 5: Build Metal**

Run:

```sh
make -B ds4 ds4_test
./ds4_test --metal-kernels
```

Expected: Metal library compilation succeeds and `metal-kernels: OK`.

Completed by `42aabd9` with fixes `9ede9e6`, `16dadd3`, and `7e1225e`:
`make -B ds4 ds4_test`, `./ds4_test --metal-kernels`, and `git diff --check`
passed. The added Metal kernel regression dispatches Q8_K routed MoE with a
nonzero expert id and multi-block Q8_K gate/up/down inputs, catching the
original partial-block matvec risk.

- [x] **Step 6: Commit Metal support**

Run:

```sh
git add ds4_metal.m metal/moe.metal
git commit -m "feat: route q8_k experts on metal"
```

## Task 5: Generate Q8KExperts GGUF

**Files:**
- Generated, ignored: `gguf/DeepSeek-V4-Flash-Q8KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2.gguf`

- [x] **Step 1: Run dry-run after runtime support builds**

Run:

```sh
./gguf-tools/deepseek4-quantize \
  --hf /Users/cchuter/models/DeepSeek-V4-Flash \
  --template gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf \
  --out gguf/DeepSeek-V4-Flash-Q8KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2.gguf \
  --experts q8_K \
  --dry-run
```

Expected: `type_changes: 129`, and `approx_file_bytes` is larger than the Q8_0Experts dry-run because each Q8_K block stores `292` bytes per `256` values.

Completed: dry-run reported `type_changes: 129` and
`approx_file_bytes: 324788806240`.

- [x] **Step 2: Generate the full model**

Run:

```sh
./gguf-tools/deepseek4-quantize \
  --hf /Users/cchuter/models/DeepSeek-V4-Flash \
  --template gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf \
  --out gguf/DeepSeek-V4-Flash-Q8KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2.gguf \
  --experts q8_K \
  --threads 8
```

Expected: the tool writes the GGUF without size mismatches.

Completed: the tool wrote
`gguf/DeepSeek-V4-Flash-Q8KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2.gguf`.

- [x] **Step 3: Record local artifact state**

Run:

```sh
ls -lh gguf/DeepSeek-V4-Flash-Q8KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2.gguf
df -h .
git status --short
```

Expected: the model exists locally and does not appear in `git status --short`.

Completed: the Q8_K model is `302G`, disk has `258Gi` free, and `git status`
shows the `gguf/` artifact as ignored rather than tracked.

## Task 6: Smoke Tests and Comparison

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/plans/2026-05-21-q8-k-routed-experts.md`

- [x] **Step 1: CPU smoke**

Run:

```sh
make cpu
DS4_LOCK_FILE=/tmp/ds4-q8k-smoke.lock ./ds4 \
  --cpu \
  -m gguf/DeepSeek-V4-Flash-Q8KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2.gguf \
  -p "What is the capital of France?" \
  --ctx 1024 \
  -n 8 \
  --nothink \
  --temp 0
```

Expected: output contains `Paris` and the process exits with code 0.

Completed: answered `The capital of France is Paris.` and exited 0.
Throughput: prefill `7.03 t/s`, generation `6.01 t/s`.

- [x] **Step 2: Metal smoke**

Run:

```sh
make -B ds4
DS4_LOCK_FILE=/tmp/ds4-q8k-smoke-metal.lock ./ds4 \
  --metal \
  -m gguf/DeepSeek-V4-Flash-Q8KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2.gguf \
  -p "What is the capital of France?" \
  --ctx 1024 \
  -n 8 \
  --nothink \
  --temp 0
```

Expected: output contains `Paris` and reports prefill/generation token rates.

Completed: answered `The capital of France is Paris.` and exited 0.
Throughput: prefill `39.29 t/s`, generation `20.39 t/s`.

- [x] **Step 3: Compare Q4, Q8_0, and Q8_K on the same smoke**

Run the same Metal command shape for:

```text
gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf
gguf/DeepSeek-V4-Flash-Q8_0Experts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2.gguf
gguf/DeepSeek-V4-Flash-Q8KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2.gguf
```

Record size, prefill t/s, generation t/s, and answer text in this plan.

Same prompt and command shape on Mac Studio M3 Ultra:

| Local GGUF | Size | Prefill | Generation | Answer |
| --- | ---: | ---: | ---: | --- |
| Q4KExperts imatrix | 153G | 49.55 t/s | 25.88 t/s | Paris |
| Q8_0Experts | 282G | 39.46 t/s | 20.94 t/s | Paris |
| Q8KExperts | 302G | 39.29 t/s | 20.39 t/s | Paris |

- [x] **Step 4: Run targeted tests**

Run:

```sh
make ds4_test
./ds4_test --server --metal-kernels
git diff --check
```

Expected: `ds4 tests: ok`, and `git diff --check` produces no output.

Completed: `make ds4_test`, `./ds4_test --server --metal-kernels`, and
`git diff --check` passed. The test output included `metal-kernels: OK`,
`server: OK`, and `ds4 tests: ok`.

- [x] **Step 5: Update README with Q8_K status**

Update `README.md` so the local experiment section contains both:

```markdown
`Q8_0Experts` uses GGUF type `q8_0` for routed experts and is the first
runnable high-precision checkpoint.

`Q8KExperts` uses GGUF type `q8_K` for routed experts. It is the true Phase 2
8-bit K-block format and is expected to be slightly larger than Q8_0 because
each `q8_K` block stores per-block sums in addition to the quantized bytes.
```

Completed: README now documents Q8_0Experts, Q8KExperts, generation commands,
and the Q4/Q8_0/Q8_K smoke comparison.

- [x] **Step 6: Commit verification docs**

Run:

```sh
git add README.md docs/superpowers/plans/2026-05-21-q8-k-routed-experts.md
git commit -m "docs: record q8_k routed expert verification"
```

## Task 7: Branch Handoff

**Files:**
- No code changes.

- [x] **Step 1: Final status check**

Run:

```sh
git status --short --branch
git log --oneline --decorate -8
ls -lh gguf/DeepSeek-V4-Flash-Q8KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2.gguf
```

Expected: the branch is clean, recent commits show Q8_K support, and the generated GGUF exists locally.

Completed: branch `feat/q8-k-routed-experts` is clean, recent commits show the
Q8_K quantizer/runtime/test/docs work, and the local generated Q8_K GGUF exists
at `302G` under ignored `gguf/`.

- [ ] **Step 2: Push when requested**

Run only after the user asks to push Phase 2:

```sh
git push -u origin feat/q8-k-routed-experts
```

Expected: the Phase 2 branch is available on `https://github.com/cchuter/ds4.git`.

## Self-Review

- Spec coverage: this plan covers quantizer output, CPU runtime, Metal runtime, model generation, CPU smoke, Metal smoke, comparison against Q4/Q8_0, documentation, and push handoff.
- Placeholder scan: no open-ended placeholder steps remain; each task has exact files, commands, and expected outputs.
- Type consistency: the plan uses `q8_K`/`DS4_TENSOR_Q8_K`/`DS4_METAL_TENSOR_Q8_K` consistently and keeps `Q8_0Experts` separate from `Q8KExperts`.
