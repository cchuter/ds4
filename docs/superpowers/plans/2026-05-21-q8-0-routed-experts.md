# Q8_0 Routed Experts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a `Q8_0Experts` DeepSeek V4 Flash GGUF load and run through both CPU and Metal DS4 paths.

**Architecture:** Keep Phase 1 scoped to GGUF type `q8_0` for all routed expert tensors. Reuse the existing quantizer `--experts q8_0` output, reuse existing CPU Q8_0 dense dot helpers for routed CPU projections, and wire existing Metal routed Q8_0 kernels into the routed MoE dispatch. Do not add true `q8_K` stored model weights in this plan.

**Tech Stack:** C99 DS4 runtime, Objective-C Metal glue, Metal kernels already present in `metal/moe.metal`, DS4 C test runner, `gguf-tools/deepseek4-quantize`.

---

## File Structure

- Modify `ds4.c`
  - Routed expert type validation.
  - CPU Q8_0 routed gate/up/down helpers.
  - CPU routed MoE single-token, preallocated decode, and batch prefill dispatch.
  - Routed quant-bit metadata.
- Modify `ds4_metal.m`
  - Metal routed expert type enum.
  - Q8_0 routed decode and prefill pipeline dispatch.
- Modify `README.md`
  - Add a short local-experiment note for `Q8_0Experts`.
- Use existing `gguf-tools/deepseek4-quantize`
  - No Phase 1 source changes expected.
- Generate, do not commit:
  - `gguf/DeepSeek-V4-Flash-Q8_0Experts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2.gguf`

## Task 1: Runtime Type Admission And Metadata

**Files:**
- Modify: `ds4.c:2295-2313`
- Modify: `ds4.c:15979-15984`

- [ ] **Step 1: Verify current rejection path**

Run:

```sh
rg -n "tensor_is_routed_expert_type|routed_expert_block_bytes|ds4_engine_routed_quant_bits" ds4.c
```

Expected: the routed helper accepts `IQ2_XXS`, `Q2_K`, and `Q4_K`, and `ds4_engine_routed_quant_bits()` returns `4` for Q4 and `2` for everything else.

- [ ] **Step 2: Admit Q8_0 routed expert tensors and compute type-aware row bytes**

In `ds4.c`, change the helpers to include `DS4_TENSOR_Q8_0`:

```c
static bool tensor_is_routed_expert_type(uint32_t type) {
    return type == DS4_TENSOR_Q8_0 ||
           type == DS4_TENSOR_IQ2_XXS ||
           type == DS4_TENSOR_Q2_K ||
           type == DS4_TENSOR_Q4_K;
}

static DS4_MAYBE_UNUSED uint64_t routed_expert_block_bytes(uint32_t type) {
    switch (type) {
    case DS4_TENSOR_Q8_0:    return 34;
    case DS4_TENSOR_IQ2_XXS: return sizeof(block_iq2_xxs);
    case DS4_TENSOR_Q2_K:    return sizeof(block_q2_K);
    case DS4_TENSOR_Q4_K:    return sizeof(block_q4_K);
    default:                 ds4_die("unsupported routed expert tensor type");
    }
    return 0;
}
```

Update `routed_expert_row_bytes()` so it uses the GGUF tensor type metadata for
the block element count. `Q8_0` has 32 elements per block, while the K-family
expert types have 256 elements per block:

```c
static DS4_MAYBE_UNUSED uint64_t routed_expert_row_bytes(const ds4_tensor *t) {
    const gguf_type_info *info = tensor_type(t->type);
    if (!info || info->block_elems == 0) ds4_die("unsupported routed expert tensor type");
    if ((t->dim[0] % info->block_elems) != 0) ds4_die("routed expert row is not quant block aligned");
    return (t->dim[0] / info->block_elems) * routed_expert_block_bytes(t->type);
}
```

- [ ] **Step 3: Report 8-bit routed expert metadata**

Replace `ds4_engine_routed_quant_bits()` with an explicit type switch:

```c
int ds4_engine_routed_quant_bits(ds4_engine *e) {
    if (!e) return 0;
    const ds4_tensor *gate = e->weights.layer[0].ffn_gate_exps;
    if (!gate) return 0;
    switch (gate->type) {
    case DS4_TENSOR_Q8_0: return 8;
    case DS4_TENSOR_Q4_K: return 4;
    case DS4_TENSOR_Q2_K:
    case DS4_TENSOR_IQ2_XXS:
        return 2;
    default:
        return 0;
    }
}
```

- [ ] **Step 4: Build to catch enum or compile errors**

Run:

```sh
make ds4_test
```

Expected: compile succeeds. If it fails, fix only compile errors introduced by this task.

- [ ] **Step 5: Commit**

```sh
git add ds4.c
git commit -m "feat: admit q8_0 routed expert tensors"
```

## Task 2: CPU Q8_0 Routed Helpers

**Files:**
- Modify: `ds4.c:3810-4235`

- [ ] **Step 1: Add a Q8_0 pair-mid context**

Near `matvec_iq2_xxs_mid_ctx`, add a Q8_0 context that stores activation bytes and scales:

```c
typedef struct {
    float *mid;
    const uint8_t *gate_base[DS4_N_EXPERT_USED];
    const uint8_t *up_base[DS4_N_EXPERT_USED];
    const int8_t *xq;
    const float *xscale;
    float expert_weight[DS4_N_EXPERT_USED];
    float clamp;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t blocks;
    uint64_t gate_row_bytes[DS4_N_EXPERT_USED];
    uint64_t up_row_bytes[DS4_N_EXPERT_USED];
    int n_expert;
} matvec_q8_0_mid_ctx;
```

- [ ] **Step 2: Add a Q8_0 pair-mid worker**

Add this worker after the IQ2 worker:

```c
static void matvec_q8_0_mid_worker(void *vctx, uint64_t row0, uint64_t row1) {
    matvec_q8_0_mid_ctx *ctx = vctx;

    for (uint64_t idx = row0; idx < row1; idx++) {
        const int slot = (int)(idx / ctx->out_dim);
        const uint64_t row = idx - (uint64_t)slot * ctx->out_dim;
        float gate = 0.0f;
        float up = 0.0f;

        const uint8_t *gate_row = ctx->gate_base[slot] + row * ctx->gate_row_bytes[slot];
        const uint8_t *up_row = ctx->up_base[slot] + row * ctx->up_row_bytes[slot];
        dot_q8_0_row_pair(gate_row, up_row, ctx->xq, ctx->xscale,
                          ctx->in_dim, ctx->blocks, &gate, &up);

        if (ctx->clamp > 1.0e-6f) {
            if (gate > ctx->clamp) gate = ctx->clamp;
            if (up > ctx->clamp) up = ctx->clamp;
            if (up < -ctx->clamp) up = -ctx->clamp;
        }
        ctx->mid[idx] = silu(gate) * up * ctx->expert_weight[slot];
    }
}
```

- [ ] **Step 3: Add the Q8_0 selected-experts mid function**

Add:

```c
static void matvec_q8_0_experts_mid_prequant(
        float            *mid,
        const ds4_model  *m,
        const ds4_tensor *gate_w,
        const ds4_tensor *up_w,
        const int8_t     *xq,
        const float      *xscale,
        const int        *selected,
        const float      *expert_weight,
        int               n_expert,
        float             clamp) {
    if (gate_w->type != DS4_TENSOR_Q8_0 || up_w->type != DS4_TENSOR_Q8_0) {
        ds4_die("expected Q8_0 expert tensors");
    }
    if (n_expert < 1 || n_expert > DS4_N_EXPERT_USED) ds4_die("unexpected routed expert count");

    uint64_t in_dim0 = 0;
    uint64_t out_dim0 = 0;
    matvec_q8_0_mid_ctx ctx = {
        .mid = mid,
        .xq = xq,
        .xscale = xscale,
        .clamp = clamp,
        .n_expert = n_expert,
    };

    for (int i = 0; i < n_expert; i++) {
        uint64_t gate_in_dim, gate_out_dim;
        uint64_t up_in_dim, up_out_dim;
        ctx.gate_base[i] = tensor_expert_bytes(m, gate_w, (uint32_t)selected[i],
                                               &gate_in_dim, &gate_out_dim, &ctx.gate_row_bytes[i]);
        ctx.up_base[i] = tensor_expert_bytes(m, up_w, (uint32_t)selected[i],
                                             &up_in_dim, &up_out_dim, &ctx.up_row_bytes[i]);
        if (gate_in_dim != up_in_dim || gate_out_dim != up_out_dim) {
            ds4_die("paired Q8_0 expert tensors do not match");
        }
        if (i == 0) {
            in_dim0 = gate_in_dim;
            out_dim0 = gate_out_dim;
        } else if (gate_in_dim != in_dim0 || gate_out_dim != out_dim0) {
            ds4_die("Q8_0 expert tensors do not share a layout");
        }
        ctx.expert_weight[i] = expert_weight[i];
    }
    if ((in_dim0 % 32u) != 0) ds4_die("Q8_0 expert row is not QK8_0 aligned");

    ctx.in_dim = in_dim0;
    ctx.out_dim = out_dim0;
    ctx.blocks = in_dim0 / 32u;
    ds4_parallel_for((uint64_t)n_expert * out_dim0, matvec_q8_0_mid_worker, &ctx);
}
```

- [ ] **Step 4: Add Q8_0 down accumulation**

Add a down accumulation context and worker:

```c
typedef struct {
    float *out;
    const uint8_t *base[DS4_N_EXPERT_USED];
    const int8_t *xq[DS4_N_EXPERT_USED];
    const float *xscale[DS4_N_EXPERT_USED];
    uint64_t in_dim;
    uint64_t row_bytes[DS4_N_EXPERT_USED];
    uint64_t blocks;
    int n_expert;
} matvec_q8_0_accum_ctx;

static void matvec_q8_0_accum_worker(void *vctx, uint64_t row0, uint64_t row1) {
    matvec_q8_0_accum_ctx *ctx = vctx;

    for (uint64_t row = row0; row < row1; row++) {
        float acc = 0.0f;
        for (int i = 0; i < ctx->n_expert; i++) {
            const uint8_t *br = ctx->base[i] + row * ctx->row_bytes[i];
            acc += dot_q8_0_row(br, ctx->xq[i], ctx->xscale[i], ctx->in_dim, ctx->blocks);
        }
        ctx->out[row] = acc;
    }
}
```

Then add:

```c
static void matvec_q8_0_experts_accum_prequant(
        float            *out,
        const ds4_model  *m,
        const ds4_tensor *w,
        const int8_t     *xq,
        const float      *xscale,
        const int        *selected,
        int               n_expert) {
    if (w->type != DS4_TENSOR_Q8_0) ds4_die("expected a Q8_0 expert tensor");
    if (n_expert < 1 || n_expert > DS4_N_EXPERT_USED) ds4_die("unexpected routed expert count");

    uint64_t in_dim0 = 0;
    uint64_t out_dim0 = 0;
    const uint64_t blocks0 = w->dim[0] / 32u;
    matvec_q8_0_accum_ctx ctx = {
        .out = out,
        .in_dim = w->dim[0],
        .blocks = blocks0,
        .n_expert = n_expert,
    };

    for (int i = 0; i < n_expert; i++) {
        uint64_t in_dim, out_dim;
        ctx.base[i] = tensor_expert_bytes(m, w, (uint32_t)selected[i], &in_dim, &out_dim, &ctx.row_bytes[i]);
        if (i == 0) {
            in_dim0 = in_dim;
            out_dim0 = out_dim;
        } else if (in_dim != in_dim0 || out_dim != out_dim0) {
            ds4_die("Q8_0 expert tensors do not share a layout");
        }
        ctx.xq[i] = xq + (uint64_t)i * blocks0 * 32u;
        ctx.xscale[i] = xscale + (uint64_t)i * blocks0;
    }
    if ((in_dim0 % 32u) != 0) ds4_die("Q8_0 expert row is not QK8_0 aligned");

    ds4_parallel_for(out_dim0, matvec_q8_0_accum_worker, &ctx);
}
```

- [ ] **Step 5: Build CPU object**

Run:

```sh
make ds4_cpu.o
```

Expected: `ds4_cpu.o` compiles.

- [ ] **Step 6: Commit**

```sh
git add ds4.c
git commit -m "feat: add cpu q8_0 routed kernels"
```

## Task 3: CPU Routed MoE Dispatch

**Files:**
- Modify: `ds4.c:5359-5505`
- Modify: `ds4.c:5509-5643`

- [ ] **Step 1: Update single-token routed MoE fast path**

In `layer_routed_moe_one()`, branch on the routed expert type after expert selection.

For Q8_0, allocate Q8_0 activation buffers:

```c
const bool routed_q8_0 =
    layer->ffn_gate_exps->type == DS4_TENSOR_Q8_0 &&
    layer->ffn_up_exps->type == DS4_TENSOR_Q8_0 &&
    layer->ffn_down_exps->type == DS4_TENSOR_Q8_0;
```

Use this Q8_0 non-trace block:

```c
if (!trace && routed_q8_0) {
    const uint64_t x_blocks = expert_in_dim / 32u;
    int8_t *xq8 = xmalloc((size_t)x_blocks * 32u);
    float *xscale8 = xmalloc((size_t)x_blocks * sizeof(float));
    quantize_q8_0_activation(x, xq8, xscale8, expert_in_dim);

    matvec_q8_0_experts_mid_prequant(mid_all, model,
                                     layer->ffn_gate_exps,
                                     layer->ffn_up_exps,
                                     xq8,
                                     xscale8,
                                     selected,
                                     expert_weight,
                                     DS4_N_EXPERT_USED,
                                     clamp);

    const uint64_t mid_blocks = down_in_dim / 32u;
    int8_t *midq8 = xmalloc((size_t)DS4_N_EXPERT_USED * mid_blocks * 32u);
    float *midscale8 = xmalloc((size_t)DS4_N_EXPERT_USED * mid_blocks * sizeof(float));
    for (int i = 0; i < DS4_N_EXPERT_USED; i++) {
        quantize_q8_0_activation(mid_all + (uint64_t)i * down_in_dim,
                                 midq8 + (uint64_t)i * mid_blocks * 32u,
                                 midscale8 + (uint64_t)i * mid_blocks,
                                 down_in_dim);
    }
    matvec_q8_0_experts_accum_prequant(out, model, layer->ffn_down_exps,
                                       midq8, midscale8, selected, DS4_N_EXPERT_USED);
    free(midscale8);
    free(midq8);
    free(xscale8);
    free(xq8);
} else if (!trace) {
    /* Existing IQ2/Q2 path remains here unchanged. */
}
```

Keep trace mode on the existing path for now. Q8_0 smoke testing does not require `--trace`, and this keeps Phase 1 smaller.

- [ ] **Step 2: Update preallocated decode path**

In `layer_routed_moe_one_prealloc()`, add the same `routed_q8_0` predicate. For the Q8_0 branch, allocate temporary Q8_0 activation buffers locally, compute mids and down accumulation, free buffers, and return before the existing IQ2/Q2 code:

```c
if (routed_q8_0) {
    const uint64_t x_blocks = expert_in_dim / 32u;
    int8_t *xq8 = xmalloc((size_t)x_blocks * 32u);
    float *xscale8 = xmalloc((size_t)x_blocks * sizeof(float));
    quantize_q8_0_activation(x, xq8, xscale8, expert_in_dim);

    matvec_q8_0_experts_mid_prequant(mid_all, model,
                                     layer->ffn_gate_exps,
                                     layer->ffn_up_exps,
                                     xq8,
                                     xscale8,
                                     selected,
                                     expert_weight,
                                     DS4_N_EXPERT_USED,
                                     clamp);

    const uint64_t mid_blocks = down_in_dim / 32u;
    int8_t *midq8 = xmalloc((size_t)DS4_N_EXPERT_USED * mid_blocks * 32u);
    float *midscale8 = xmalloc((size_t)DS4_N_EXPERT_USED * mid_blocks * sizeof(float));
    for (int i = 0; i < DS4_N_EXPERT_USED; i++) {
        quantize_q8_0_activation(mid_all + (uint64_t)i * down_in_dim,
                                 midq8 + (uint64_t)i * mid_blocks * 32u,
                                 midscale8 + (uint64_t)i * mid_blocks,
                                 down_in_dim);
    }
    matvec_q8_0_experts_accum_prequant(out, model, layer->ffn_down_exps,
                                       midq8, midscale8, selected, DS4_N_EXPERT_USED);
    free(midscale8);
    free(midq8);
    free(xscale8);
    free(xq8);
    (void)il;
    return;
}
```

- [ ] **Step 3: Add Q8_0 batch prefill contexts and workers**

Create Q8_0 batch equivalents for the existing IQ2 mid and Q2 down workers. Use these names so call sites are clear:

```c
typedef struct {
    float *mid;
    const uint8_t *gate_base[DS4_N_EXPERT];
    const uint8_t *up_base[DS4_N_EXPERT];
    const int8_t *xq;
    const float *xscale;
    const ds4_expert_pair *pairs;
    const uint32_t *pair_ids;
    const uint32_t *expert_offset;
    const uint32_t *active_expert;
    const float *pair_weight;
    float clamp;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t blocks;
    uint64_t gate_row_bytes[DS4_N_EXPERT];
    uint64_t up_row_bytes[DS4_N_EXPERT];
} matvec_q8_0_batch_mid_ctx;
```

Add `matvec_q8_0_batch_mid_worker()` immediately after the context:

```c
static void matvec_q8_0_batch_mid_worker(void *vctx, uint64_t task0, uint64_t task1) {
    matvec_q8_0_batch_mid_ctx *ctx = vctx;

    for (uint64_t task = task0; task < task1; task++) {
        const uint32_t active_idx = (uint32_t)(task / ctx->out_dim);
        const uint64_t row = task - (uint64_t)active_idx * ctx->out_dim;
        const uint32_t expert = ctx->active_expert[active_idx];
        const uint32_t begin = ctx->expert_offset[expert];
        const uint32_t end = ctx->expert_offset[expert + 1];

        const uint8_t *gate_row = ctx->gate_base[expert] + row * ctx->gate_row_bytes[expert];
        const uint8_t *up_row = ctx->up_base[expert] + row * ctx->up_row_bytes[expert];

        for (uint32_t i = begin; i < end; i++) {
            const uint32_t pair_id = ctx->pair_ids[i];
            const ds4_expert_pair pair = ctx->pairs[pair_id];
            float gate = 0.0f;
            float up = 0.0f;

            dot_q8_0_row_pair(gate_row, up_row,
                              ctx->xq + (uint64_t)pair.token * ctx->blocks * 32u,
                              ctx->xscale + (uint64_t)pair.token * ctx->blocks,
                              ctx->in_dim, ctx->blocks, &gate, &up);

            if (ctx->clamp > 1.0e-6f) {
                if (gate > ctx->clamp) gate = ctx->clamp;
                if (up > ctx->clamp) up = ctx->clamp;
                if (up < -ctx->clamp) up = -ctx->clamp;
            }

            ctx->mid[(uint64_t)pair_id * ctx->out_dim + row] = silu(gate) * up * ctx->pair_weight[pair_id];
        }
    }
}
```

Add the Q8_0 batch down context and worker:

```c
typedef struct {
    float *moe;
    const uint8_t *base[DS4_N_EXPERT];
    const int8_t *midq;
    const float *midscale;
    const ds4_expert_pair *pairs;
    const uint32_t *pair_ids;
    const uint32_t *expert_offset;
    const uint32_t *active_expert;
    uint32_t n_active;
    uint32_t n_tok;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t row_bytes[DS4_N_EXPERT];
    uint64_t blocks;
} matvec_q8_0_batch_accum_rows_ctx;

static void matvec_q8_0_batch_accum_rows_worker(void *vctx, uint64_t row0, uint64_t row1) {
    matvec_q8_0_batch_accum_rows_ctx *ctx = vctx;

    for (uint64_t row = row0; row < row1; row++) {
        for (uint32_t t = 0; t < ctx->n_tok; t++) {
            ctx->moe[(uint64_t)t * ctx->out_dim + row] = 0.0f;
        }

        for (uint32_t ai = 0; ai < ctx->n_active; ai++) {
            const uint32_t expert = ctx->active_expert[ai];
            const uint32_t begin = ctx->expert_offset[expert];
            const uint32_t end = ctx->expert_offset[expert + 1];
            const uint8_t *br = ctx->base[expert] + row * ctx->row_bytes[expert];

            for (uint32_t i = begin; i < end; i++) {
                const uint32_t pair_id = ctx->pair_ids[i];
                const ds4_expert_pair pair = ctx->pairs[pair_id];
                const int8_t *xq = ctx->midq + (uint64_t)pair_id * ctx->blocks * 32u;
                const float *xscale = ctx->midscale + (uint64_t)pair_id * ctx->blocks;
                const float v = dot_q8_0_row(br, xq, xscale, ctx->in_dim, ctx->blocks);
                ctx->moe[(uint64_t)pair.token * ctx->out_dim + row] += v;
            }
        }
    }
}
```

- [ ] **Step 4: Branch batch prefill for Q8_0**

In `layer_routed_moe_batch()`, add a Q8_0 branch after selected experts and `pair_ids` are built:

```c
const bool routed_q8_0 =
    layer->ffn_gate_exps->type == DS4_TENSOR_Q8_0 &&
    layer->ffn_up_exps->type == DS4_TENSOR_Q8_0 &&
    layer->ffn_down_exps->type == DS4_TENSOR_Q8_0;
if (routed_q8_0) {
    const uint64_t x_blocks = expert_in_dim / 32u;
    int8_t *xq8 = xmalloc((size_t)n_tok * x_blocks * 32u);
    float *xscale8 = xmalloc((size_t)n_tok * x_blocks * sizeof(float));
    for (uint32_t t = 0; t < n_tok; t++) {
        quantize_q8_0_activation(norm + (uint64_t)t * expert_in_dim,
                                 xq8 + (uint64_t)t * x_blocks * 32u,
                                 xscale8 + (uint64_t)t * x_blocks,
                                 expert_in_dim);
    }

    float *mid = xmalloc((size_t)total_pairs * expert_out_dim * sizeof(mid[0]));
    matvec_q8_0_batch_mid_ctx mid_ctx = {
        .mid = mid,
        .xq = xq8,
        .xscale = xscale8,
        .pairs = pairs,
        .pair_ids = pair_ids,
        .expert_offset = counts,
        .active_expert = active_expert,
        .pair_weight = pair_weight,
        .clamp = clamp,
        .in_dim = expert_in_dim,
        .out_dim = expert_out_dim,
        .blocks = x_blocks,
    };
    for (uint32_t ai = 0; ai < n_active; ai++) {
        const uint32_t e = active_expert[ai];
        uint64_t gate_in_dim, gate_out_dim;
        uint64_t up_in_dim, up_out_dim;
        mid_ctx.gate_base[e] = tensor_expert_bytes(model, layer->ffn_gate_exps, e,
                                                   &gate_in_dim, &gate_out_dim, &mid_ctx.gate_row_bytes[e]);
        mid_ctx.up_base[e] = tensor_expert_bytes(model, layer->ffn_up_exps, e,
                                                 &up_in_dim, &up_out_dim, &mid_ctx.up_row_bytes[e]);
        if (gate_in_dim != expert_in_dim || up_in_dim != expert_in_dim ||
            gate_out_dim != expert_out_dim || up_out_dim != expert_out_dim) {
            ds4_die("Q8_0 batch expert tensor layout mismatch");
        }
    }
    ds4_parallel_for((uint64_t)n_active * expert_out_dim, matvec_q8_0_batch_mid_worker, &mid_ctx);

    const uint64_t mid_blocks = down_in_dim / 32u;
    int8_t *midq8 = xmalloc((size_t)total_pairs * mid_blocks * 32u);
    float *midscale8 = xmalloc((size_t)total_pairs * mid_blocks * sizeof(float));
    for (uint32_t p = 0; p < total_pairs; p++) {
        quantize_q8_0_activation(mid + (uint64_t)p * down_in_dim,
                                 midq8 + (uint64_t)p * mid_blocks * 32u,
                                 midscale8 + (uint64_t)p * mid_blocks,
                                 down_in_dim);
    }
    free(mid);

    matvec_q8_0_batch_accum_rows_ctx down_ctx = {
        .moe = moe,
        .midq = midq8,
        .midscale = midscale8,
        .pairs = pairs,
        .pair_ids = pair_ids,
        .expert_offset = counts,
        .active_expert = active_expert,
        .n_active = n_active,
        .n_tok = n_tok,
        .in_dim = down_in_dim,
        .out_dim = down_out_dim,
        .blocks = mid_blocks,
    };
    for (uint32_t ai = 0; ai < n_active; ai++) {
        const uint32_t e = active_expert[ai];
        uint64_t in_dim, out_dim;
        down_ctx.base[e] = tensor_expert_bytes(model, layer->ffn_down_exps, e,
                                               &in_dim, &out_dim, &down_ctx.row_bytes[e]);
        if (in_dim != down_in_dim || out_dim != down_out_dim) {
            ds4_die("Q8_0 batch down expert tensor layout mismatch");
        }
    }
    ds4_parallel_for(down_out_dim, matvec_q8_0_batch_accum_rows_worker, &down_ctx);

    free(midscale8);
    free(midq8);
    free(xscale8);
    free(xq8);
    free(pair_ids);
    free(pairs);
    free(pair_weight);
    free(selected);
    (void)il;
    return;
}
```

Keep the existing IQ2/Q2 batch path untouched after the Q8_0 return.

- [ ] **Step 5: Build CPU-only DS4**

Run:

```sh
make cpu
```

Expected: CPU-only `./ds4`, `./ds4-server`, `./ds4-bench`, `./ds4-eval`, and `./ds4-agent` build.

- [ ] **Step 6: Commit**

```sh
git add ds4.c
git commit -m "feat: route q8_0 experts on cpu"
```

## Task 4: Metal Q8_0 Routed Dispatch

**Files:**
- Modify: `ds4_metal.m:29-31`
- Modify: `ds4_metal.m:11642-11703`
- Modify: `ds4_metal.m:12922-13066`
- Modify: `ds4_metal.m:13234-13318`

- [ ] **Step 1: Add the Metal Q8_0 tensor enum**

Update the enum:

```objc
enum {
    DS4_METAL_TENSOR_Q8_0    = 8,
    DS4_METAL_TENSOR_Q2_K    = 10,
    DS4_METAL_TENSOR_Q4_K    = 12,
    DS4_METAL_TENSOR_IQ2_XXS = 16,
};
```

- [ ] **Step 2: Wire Q8_0 decode matvec dispatch**

Update `ds4_gpu_routed_mv_nr0()`:

```objc
static uint32_t ds4_gpu_routed_mv_nr0(uint32_t type) {
    switch (type) {
    case DS4_METAL_TENSOR_Q8_0:    return 2;
    case DS4_METAL_TENSOR_Q4_K:    return 2;
    case DS4_METAL_TENSOR_Q2_K:
    case DS4_METAL_TENSOR_IQ2_XXS: return 4;
    default:                       return 0;
    }
}
```

Update `ds4_gpu_routed_mv_pipeline()`:

```objc
case DS4_METAL_TENSOR_Q8_0:
    return ds4_gpu_get_mul_mv_pipeline("kernel_mul_mv_id_q8_0_f32", 4);
```

Q8_0 uses no special threadgroup scratch, so leave `ds4_gpu_routed_mv_smem()` unchanged.

- [ ] **Step 3: Wire Q8_0 prefill matmul dispatch**

Update `ds4_gpu_routed_mm_pipeline()`:

```objc
case DS4_METAL_TENSOR_Q8_0:
    return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q8_0_f32", false);
```

Update `ds4_gpu_routed_mm_f16_rhs_pipeline()`:

```objc
case DS4_METAL_TENSOR_Q8_0:
    return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q8_0_f16", false);
```

- [ ] **Step 4: Leave Q8_0 pair/sum fusions disabled**

In the single-token and batch routed MoE sections, do not add Q8_0 to pair-swiglu or sum6 fused paths. The generic separate gate/up and generic down-then-sum paths already support a type once `ds4_gpu_routed_mv_pipeline()` returns a pipeline.

Verify with:

```sh
rg -n "pair_swiglu_pipeline|down_sum6_pipeline|direct_down_sum|use_tiny_pair_mv" ds4_metal.m
```

Expected: Q8_0 is absent from pair/sum fused conditions.

- [ ] **Step 5: Build Metal DS4**

Run:

```sh
make ds4 ds4-server ds4-bench ds4-eval
```

Expected: all targets build on macOS.

- [ ] **Step 6: Commit**

```sh
git add ds4_metal.m
git commit -m "feat: route q8_0 experts on metal"
```

## Task 5: Generate The Q8_0Experts GGUF

**Files:**
- Generate: `gguf/DeepSeek-V4-Flash-Q8_0Experts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2.gguf`

- [ ] **Step 1: Rebuild the quantizer**

Run:

```sh
make -C gguf-tools
```

Expected: `gguf-tools/deepseek4-quantize` exists.

- [ ] **Step 2: Confirm the output plan**

Run:

```sh
./gguf-tools/deepseek4-quantize \
  --hf /Users/cchuter/models/DeepSeek-V4-Flash \
  --template gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf \
  --dry-run \
  --experts q8_0 \
  --threads 8 | tail -n 6
```

Expected:

```text
n_tensors: 1328
meta_bytes: 5333536
tensor_bytes_unpadded: 303140862300
approx_file_bytes: 303146197600
type_changes: 129
```

- [ ] **Step 3: Generate the full model**

Run:

```sh
./gguf-tools/deepseek4-quantize \
  --hf /Users/cchuter/models/DeepSeek-V4-Flash \
  --template gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf \
  --out gguf/DeepSeek-V4-Flash-Q8_0Experts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2.gguf \
  --experts q8_0 \
  --threads 8
```

Expected: command completes and prints `wrote gguf/DeepSeek-V4-Flash-Q8_0Experts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2.gguf`.

- [ ] **Step 4: Check file size**

Run:

```sh
ls -lh gguf/DeepSeek-V4-Flash-Q8_0Experts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2.gguf
```

Expected: size is roughly `282G` to `303G` depending on decimal/binary display.

## Task 6: CPU And Metal Smoke Tests

**Files:**
- Modify: `README.md`

- [x] **Step 1: Run CPU smoke inference**

Run:

```sh
make cpu
./ds4 \
  -m gguf/DeepSeek-V4-Flash-Q8_0Experts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2.gguf \
  -p "What is the capital of France?" \
  --ctx 1024 \
  -n 8
```

Expected: the model opens, reports routed quant bits as 8 in any cache/log path that prints it, and generates text containing `Paris` or a plausible continuation. If macOS CPU virtual-memory instability appears, record the exact failure and continue to Metal; do not claim CPU runtime success without an actual completed prompt.

Completed with `DS4_LOCK_FILE=/tmp/ds4-q8-smoke.lock` to avoid the user's
already-running `ds4-server` default lock. Output contained
`The capital of France is Paris.` and reported generation at `3.94 t/s`.

- [x] **Step 2: Run Metal smoke inference**

Run:

```sh
make ds4
./ds4 \
  --metal \
  -m gguf/DeepSeek-V4-Flash-Q8_0Experts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2.gguf \
  -p "What is the capital of France?" \
  --ctx 1024 \
  -n 8
```

Expected: the model opens and generates text containing `Paris` or a plausible continuation.

Completed with `DS4_LOCK_FILE=/tmp/ds4-q8-smoke-metal.lock` to avoid the
user's already-running `ds4-server` default lock. Output contained
`The capital of France is Paris.` and reported generation at `14.88 t/s`.

- [x] **Step 3: Run project tests**

Run:

```sh
make test
```

Expected: `ds4_test` passes. If model-backed tests are too slow with the Q8_0 symlink, run targeted non-model tests:

```sh
./ds4_test --server --metal-kernels
```

Record which test command completed.

Completed targeted non-model coverage:

```sh
make ds4_test
./ds4_test --server --metal-kernels
```

Output ended with `ds4 tests: ok`.

- [x] **Step 4: Document the experimental model**

Add this note under the README model download section:

```markdown
### Local Q8_0 Routed Expert Experiment

The `feat/q8-routed-experts` branch can build a local experimental
`Q8_0Experts` GGUF from the official DeepSeek V4 Flash safetensors:

```sh
./gguf-tools/deepseek4-quantize \
  --hf /Users/cchuter/models/DeepSeek-V4-Flash \
  --template gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf \
  --out gguf/DeepSeek-V4-Flash-Q8_0Experts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2.gguf \
  --experts q8_0 \
  --threads 8
```

This is distinct from true `Q8_K` expert weights. `Q8_0Experts` uses GGUF
type `q8_0` for routed experts and is intended as the first runnable
high-precision routed-expert checkpoint before implementing `Q8_K`.
```

- [x] **Step 5: Final verification**

Run:

```sh
git diff --check
git status --short
```

Expected: no whitespace errors. Untracked/generated GGUF files may appear and must not be committed.

Completed:

```sh
git diff --check
make ds4_test
./ds4_test --server --metal-kernels
```

`git diff --check` produced no output, `make ds4_test` was up to date, and
`./ds4_test --server --metal-kernels` ended with `ds4 tests: ok`.

- [ ] **Step 6: Commit code and docs**

```sh
git add ds4.c ds4_metal.m README.md
git commit -m "feat: run q8_0 routed expert models"
```

## Task 7: Phase 1 Handoff

**Files:**
- No code changes unless verification uncovers a bug.

- [ ] **Step 1: Summarize generated model state**

Run:

```sh
ls -lh gguf/DeepSeek-V4-Flash-Q8_0Experts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2.gguf
git log --oneline --decorate -5
```

Expected: the generated GGUF exists locally and recent commits show Phase 1 runtime changes.

- [ ] **Step 2: Push branch when requested**

Only push after user approval:

```sh
git push -u origin feat/q8-routed-experts
```

Expected: branch is available on `https://github.com/cchuter/ds4.git`.

- [ ] **Step 3: Open Phase 2 planning**

After Phase 1 smoke tests pass, create a separate plan for true `Q8KExperts`.
The Phase 2 plan starts with `gguf-tools` support for emitted `q8_K` model
weights, then runtime CPU/Metal support for GGUF type `15`.
