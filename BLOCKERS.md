# upstream-sync-2 BLOCKERS — D1-D5 PR #6 multi-tier replant (needs-human)

**Branch:** `merge/upstream-sync-2` on `cchuter/ds4` (PR #17)
**Worktree:** `/Users/cchuter/work/ds4-merge-upstream-sync-2`
**HEAD:** `2d9ef04` (gitignore: ignore built test binaries) — matches `origin/merge/upstream-sync-2`
**Origin source-of-truth for PR #6:** commit `afedc61` (feat(mgpu): wave 3a Half-B — engine-side multi-tier dispatch (#6))

## TL;DR

PR #17 is currently in the "single-tier byte-equivalent + multi-tier ACCOUNTING-only" state — `ds4.c` has NONE of PR #6's `_by_tier` scaffolding:

```
$ grep -c "_by_tier" ds4.c
0
$ grep -E "metal_graph_set_active_tier|active_tier" ds4.c
(no hits)
```

Multi-tier execution fails at warmup with a cross-device pointer fault. Two prior bot sessions have attempted the D1-D5 replant of PR #6 onto upstream's restructured `ds4.c`. Both escalated `needs-human` with documented reasoning. **This is the third such escalation.**

## Empirical scope (re-verified 2026-06-08)

`git cherry-pick --no-commit afedc61` against current HEAD produces:

- **57 conflict regions in `ds4.c`** (independently counted via `grep -c "^<<<<<<< HEAD" ds4.c`)
- 2 conflict regions in `Makefile`
- 2 conflict regions in `ds4_cuda.cu`
- `ds4_gpu_mgpu.h` and `tests/test_engine_mgpu_refusal.c` auto-merge clean

Conflict-region sizes in `ds4.c` (excerpt of the larger ones, full list captured in this session):

```
 10484 -  10559 ( 75 lines)  struct Class P fields (first half)
 10582 -  10679 ( 97 lines)  struct Class E + Class P batch fields
 10702 -  10942 (240 lines)  graph_power_* helpers + accessors/tier-switch helpers
 10947 -  11051 (104 lines)  metal_graph_free
 11063 -  11141 ( 78 lines)  more metal_graph_free
 11763 -  11857 ( 94 lines)  metal_graph_alloc_raw_cap (head)
 11881 -  11978 ( 97 lines)  metal_graph_alloc_raw_cap (Class P)
 15748 -  16092 (344 lines)  metal_graph_encode_decode_layer (BIG)
 17718 -  17866 (148 lines)  prefill / batch dispatch
 19030 -  19133 (103 lines)  more prefill / batch dispatch
 19330 -  19535 (205 lines)  more prefill / batch dispatch
 20886 -  20945 ( 59 lines)  warmup
 21309 -  21396 ( 87 lines)  spec-decode / MTP
 21480 -  21550 ( 70 lines)  more spec-decode / MTP
 26594 -  27409              head / output / weights (4 regions)
```

Total conflict-line span in `ds4.c`: **~3,200 lines** across 57 regions.

Each large region overlays PR #6's per-tier scaffolding on TOP of fork-only code that PR #6 has no knowledge of: `graph_power_*` helpers, Q8_K compressor staging (`attn_comp_stage`), SplitKV-decode (`batch_q_half`), PR #12 selective-expert cache (`prefill_seed_*`), directional steering, streaming, etc.

## Phase-vs-build-gate CONTRADICTION (new finding, 2026-06-08)

The proposed D1-D5 phasing in the task spec says each phase gets its own
build gate (`make -j4 ds4.o`) and commit. **The phasing is architecturally
unrealizable as written** because:

1. Rule 1 of the conflict-resolution recipe says "Take ALL PR #6 `_by_tier[DS4_MAX_GPUS]` fields verbatim" — i.e., the single-tier field declarations (`ds4_gpu_tensor *cur_hc;`) are REMOVED and replaced by `cur_hc_by_tier[DS4_MAX_GPUS]`.

2. If D1 is committed in isolation (struct + Apple stubs only), the thousands of existing `g->cur_hc`, `g->ffn_out`, `g->prefill_tokens`, etc. references in dispatch code throughout `metal_graph_encode_decode_layer`, the batch dispatch family, and the warmup function will FAIL TO COMPILE because the fields no longer exist.

3. Those readers are precisely what D5 rewrites via the accessors. D2 introduces the accessors (and tier-switch helpers); D5 changes the call sites. So **D1's build gate cannot pass until D2 + D5 are also done**.

This is why PR #6 was originally a single atomic commit: it is one indivisible change. The 5-commit narrative is fine for a PR body, but the actual cherry-pick must be resolved + tested in one pass.

The "D5-MINIMAL is acceptable" interpretation from a prior session was an attempt to work around this contradiction by leaving D1-D4 actually unimplemented while declaring partial victory. The user explicitly rejected that interpretation.

## Workable paths forward (require human decision)

### Option A — Single atomic merge

Resolve all 57 ds4.c conflicts in ONE pass with full attention, do ONE Mac
build, ONE `test_engine_mgpu_placement` run, ONE multi-tier bench on the
box, ONE PR #17-body update. Commit message can narrate D1-D5 in the body
but the cherry-pick remains a single commit. This is what PR #6 originally
was. Realistic time: **8-15 focused hours** by an experienced engineer.

### Option B — Shadow fields (extend rule 1)

Modify rule 1 to ADD the `_by_tier` declarations alongside (not replacing)
the single-tier ones. D1 then compiles because dispatch still reads
`g->cur_hc`. D2 adds accessors. D3 makes the allocator fill BOTH. D4 wires
the embed/head tier. D5 progressively flips readers from single-tier fields
to accessors.

- **Cost:** ~80 extra pointer fields in the struct (~640 bytes per graph),
  a less-clean final state that differs subtly from upstream PR #6, and
  the allocator must keep both ptrs in sync until D5 is done.
- **Benefit:** Each phase is independently buildable, commitable, pushable.
- **Realistic time:** same total ~8-15h but split into 5 reviewable
  commits with intermediate state visible on the branch.

### Option C — Replant from upstream baseline

`git checkout afedc61 -- ds4.c` to get PR #6's `ds4.c` verbatim, then
manually overlay upstream's restructure (Q8_K, SplitKV, PR #12, etc.) on
top — the inverse direction. Likely just as hard; conflicts surface
differently.

### Option D — Defer multi-tier execution entirely

Accept the current state (PR #4 multi-tier ACCOUNTING but no PR #6
EXECUTION) as the merged baseline and treat PR #6 replant as a separate,
later PR off this branch's eventual merge. This requires updating the PR
#17 body to be honest about what's in and out, and accepting that
multi-tier execution stays broken on this branch.

## Why not just start hammering on Option A as a third bot

Both prior sessions ran into the same wall: the cherry-pick is mechanical
to *start* (struct fields, Apple stubs) but the 344-line conflict region
inside `metal_graph_encode_decode_layer` (lines 15748-16092 in the
cherry-picked state) requires understanding both upstream's restructured
attention + FFN dispatch AND PR #6's tier-switch + accessor changes
simultaneously. A single bot session does not have the context budget to
do that for ALL of the large regions (15748-16092, 17718-17866,
19330-19535, 21309-21396) without the work bleeding across multiple
sessions — at which point the `merge/upstream-sync-2` branch ends up in a
half-resolved state that is harder to recover from than starting clean.

A focused human-driven session with continuous file-level diff review
against PR #6 + upstream/main side-by-side is the realistic path.

## State at end of this session

- `ds4.c` clean at `2d9ef04`. All cherry-pick attempts aborted with
  `git reset --hard HEAD`.
- `PR6-BY-TIER-FIELDS.txt` (untracked) — the 87-line PR #6 `_by_tier`
  field enumeration, kept for the next session's reference.
- `/tmp/pr6.diff` (282 KB, 4599 lines) — full PR #6 diff.
- `/tmp/by_tier_fields.txt` (5 KB) — duplicate of the field list.
- This `BLOCKERS.md` updated with the consolidated handoff.
- No commits added, no pushes performed.

## Conflict-resolution recipe (preserved from prior session, for reference)

If/when Option A or B is pursued, the resolution rules for `ds4.c` were
distilled empirically by the previous session:

### Apple-stubs region (lines 63-85)

- Keep HEAD `ds4_gpu_init_multi` returning 0 (success).
- Add PR #6's new stubs (`ds4_gpu_tensor_alloc_ptr_on`,
  `_alloc_managed_on`, `_tensor_copy_xdev`, etc.) but have the `_on`
  pointer-allocators FALL BACK to single-tier `ds4_gpu_tensor_alloc(bytes)`
  for tier 0 — NOT return NULL.
- Keep HEAD's `ds4_gpu_ctx g_gpu[DS4_MAX_GPUS] = {{0}};` initializer.

### Struct fields (lines 10484-10679)

- Take ALL PR #6 `_by_tier[DS4_MAX_GPUS]` fields verbatim.
- Preserve fork-only fields:
  - `attn_comp_stage` → tier it: `attn_comp_stage_by_tier[DS4_MAX_GPUS]`
    (Q8_K compressor staging is Class P)
  - `batch_q_half` → keep as single-tier scalar with comment (SplitKV
    decode staging; out of PR #6 scope to tier it)
  - `prefill_seed_router_selected`, `prefill_seed_tokens`,
    `prefill_selected_profile_*` → single-tier (PR #12 selective-expert
    cache; not per-tier in PR #12 either)
  - `directional_steering_attn_scale`, `directional_steering_ffn_scale`,
    `power_percent`, `prefill_layer_avg_sec[]`, `decode_token_avg_sec`,
    `streaming_*`, `quality`, `ssd_streaming*`, `mtp_enabled`,
    `cpu_router_norm` → single-tier scalars; keep as-is
- Add PR #6's new scalar fields: `int active_tier`, `const int *placement`,
  `int emb_tier`, `int head_tier`.

### graph_power_* helpers (lines 10703-10737 inside the 10702-10942 region)

These are fork-only (upstream/PR #6 don't know about them) and MUST be
preserved verbatim. They precede the accessor/tier-switch helpers in the
cherry-picked output.

### Accessor macros + tier-switch helpers (lines 10738-10942)

Take PR #6's macros and helpers verbatim, but ALSO add an
`attn_comp_stage` Class P accessor (it's fork-only and PR #6 doesn't have
one for it).

### `metal_graph_free` (lines 10947-11210)

Tier-loop free PR #6's `_by_tier` slots; keep single-tier frees for the
preserved single-tier fields above.

### `metal_graph_alloc_raw_cap` (lines 11707-12101)

Take PR #6's per-used-tier allocation loop. Add `attn_comp_stage_by_tier`
allocation. Class E (`prefill_tokens`) → `emb_tier` only. Class H
(`output_pre/weights/embd/norm/logits`) → `head_tier` only. Class L (KV
caches) → `metal_graph_alloc_kv_cache_tensor_on(managed, layer_tier, bytes)`
per-layer.

### Dispatch bodies (lines 15047-21859)

Every `g->FIELD` read of a Class P field → `metal_graph_FIELD(g)`
accessor. PRESERVE upstream's new code paths (Q8_K compressor,
SplitKV-decode helpers, F16-gated routed mid, distributed-shadow paths).
At the top of each dispatch function add the tier-switch guard:

```c
if (g->placement) {
    if (!metal_graph_set_active_tier_decode(g, g->placement[il + 1])) return false;
}
```

(`_batch(g, tier, n_tokens)` for prefill dispatch.)

### `ds4_session_create` / engine-create

Pass `e->placement` (or NULL for single-tier) into
`metal_graph_alloc_raw_cap`. After `ds4_gpu_init_multi`, BEFORE warmup:
`g->active_tier = 0; ds4_gpu_set_current_device(0);` — this is the
proximate cause of the cross-device pointer fault reported in the spec.

### Warmup `metal_graph_warmup_prefill_kernels` (now at line 17341 after upstream restructure)

Same per-field wrapping as the regular dispatch path. This is where the
cross-device fault triggers when `active_tier` was leaking tier-1 device
state from a prior op.

### `Makefile` (2 regions)

Take the union: upstream's new ROCm/SSD/distributed objects + our fork's
`ds4_layer_pack`, `ds4_cpu_test_hooks`, test binaries.

### `ds4_cuda.cu` (2 regions, small)

Take PR #6's `_by_tier` parameter propagation through CUDA-side allocators;
keep upstream's perf wins (PR #13/#14/#16) untouched.

## Build-gate schedule under Option B (only viable if user picks Option B)

Otherwise build is ONE gate at the end of one cherry-pick resolution.

- After D1 (shadow fields + accessor scaffolding + Apple stubs): `make -j4 ds4.o`
- After D2 (accessors that read from shadow fields): `make -j4 ds4.o`
- After D3 (allocator fills both single-tier + tier slots): `make -j4 ds4.o`
- After D4 (call-site propagation + warmup tier-0 reset): `make -j4 ds4.o` + placement 81/81
- After D5 (flip readers + remove single-tier shadows): full `make -j4` + placement 81/81 + layer_pack 97/97

## After build passes (any option) — multi-tier bench on the box

```
ssh 192.168.50.231 'cd ~/work/ds4 && git fetch origin --quiet && \
  git reset --hard origin/merge/upstream-sync-2 && \
  DS4_LOCK_FILE=/tmp/ds4-pr17-d5full.lock CUDA_HOME=/usr/lib/cuda \
    make cuda CUDA_ARCH=sm_89 -j1 ds4-bench && \
  DS4_LOCK_FILE=/tmp/ds4-pr17-d5full.lock timeout 600 ./ds4-bench --cuda \
    --gpu-vram 47,47 \
    -m gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
    --prompt-file tests/long_context_story_prompt.txt \
    --ctx-start 2048 --ctx-max 2048 --gen-tokens 64 --show-output'
```

Acceptance: no cross-device fault, decoded text matches the canonical
"Mara sat outside the apothecary…" story, prefill ≥322 tps, gen ≥37.5 tps
default / ≥38.8 tps with `DS4_CUDA_SPLITKV_DECODE=1`.

If multi-tier bench FAILS after a clean build + placement-test PASS,
escalate again — do NOT paper over it in the PR body.

## Non-D5 carry-forward (still open, not in scope for this replant)

- CPU Q8_0/Q8_K routed-expert kernels — typedefs/functions present but
  unwired through `matvec_experts_mid_prequant` (signature mismatch).
- Transient-GPU-state issue: bench fails when GPU is busy (environmental).
