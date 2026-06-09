# upstream-sync-2 BLOCKERS — RESOLVED (PR #6 multi-tier replant)

**Branch:** `merge/upstream-sync-2` on `cchuter/ds4` (PR #17)
**Worktree:** `/Users/cchuter/work/ds4-merge-upstream-sync-2`
**HEAD:** `caacb14` (review: address codex round-1 findings)
**Origin source-of-truth for PR #6:** commit `afedc61` (feat(mgpu): wave 3a Half-B — engine-side multi-tier dispatch (#6))

## Status: RESOLVED — multi-tier dispatch working end-to-end

The PR #6 replant is complete and verified. Three prior builders had
escalated needs-human; the 4th attempt completed the work by treating it
as a single atomic merge (Option A from the prior BLOCKERS): take HEAD's
body in every conflict region, programmatically rewrite Class P/E/H
field references to accessor calls, and surgically fix the few
hardcoded `[0]` LHS writes that broke active-tier propagation.

### Verification (2026-06-08)

- **Mac build:** clean (1 pre-existing typedef-redefinition warning).
- **CUDA box build (sm_89):** clean.
- **tests/test_engine_mgpu_placement:** 81/81 PASS.
- **tests/test_layer_pack:** 97/97 PASS.
- **tests/test_gpu_args:** all PASS.
- **tests/test_engine_correctness single-tier (CUDA):** 8 tokens
  "The capital of France is Paris.<｜end▁of▁sentence｜>",
  IDs `671 6102 294 8760 344 11111 16 1`.
- **tests/test_engine_correctness multi-tier (47,47 GB split):** IDENTICAL
  8 tokens; full byte-equivalence between single-tier and multi-tier
  decodes.
- **ds4-bench multi-tier default:** prefill=342.21 tps (target ≥322 ✓),
  gen=39.92 tps (target ≥37.5 ✓). Decoded text matches canonical Mara
  story.
- **ds4-bench multi-tier with `DS4_CUDA_SPLITKV_DECODE=1`:**
  prefill=340.68 tps, gen=39.03 tps (target ≥38.8 ✓).
- **Codex code review round 2:** APPROVE.

## Commits added on this branch

```
caacb14 review: address codex round-1 findings
f2049e9 debug: revert xdev diagnostic logging — multi-tier now works
6c909c3 fix(mgpu): cur_hc/after_ffn_hc swap and head-tier saved_cur use active_tier
cc09230 debug: add xdev copy diagnostic logging
c37d4f0 fix(mgpu): force active_tier=0 and CUDA device=0 on graph alloc
b999797 fix(mgpu): layer_attn_state_kv/score on layer_tier not single-tier
7ea9adc fix(mgpu): tier-replicate Class P/E/H allocator across used tiers
fe6343d feat(mgpu): replant PR #6 multi-tier dispatch onto upstream-sync-2
3a4868e (prior baseline) docs: update BLOCKERS for upstream-sync-2 D1-D5 replant (3rd needs-human)
```

The replant is reviewable as 8 commits totaling ~1900 lines net diff on
ds4.c, plus tiny Makefile/ds4_cuda.cu hygiene changes.

## Strategy (working note for future reference)

1. **Cherry-pick atomically with `--no-commit`.** Do not attempt
   phase-by-phase commits within ds4.c; field renames mean any
   incomplete phase fails to compile.
2. **Take HEAD's body in every conflict region.** HEAD contains fork-only
   upstream restructure (Q8_K compressor, SplitKV decode, PR #12 selective-
   expert cache, fused-norm, F16-gated paths) that PR #6 has no
   awareness of. PR #6's body is a subset of HEAD's; HEAD wins on
   semantics.
3. **Programmatically rewrite `g->FIELD` → `metal_graph_FIELD(g)` for
   Class P/E/H fields, GLOBALLY (not just within conflict regions).**
   Use a fixed list of accessor fields and a regex with `\bg->FIELD\b`
   (negative lookahead `(?!_by_tier)`).
4. **For LHS writes (`g->FIELD = ...`) convert to `g->FIELD_by_tier[0] = ...`**
   first, then run the reader rewrite. The accessor returns a value, not
   an lvalue.
5. **Fork-only single-tier preservation:** `batch_q_half`,
   `prefill_seed_router_selected`, `prefill_seed_tokens`,
   `prefill_selected_profile_*` stay as single-tier scalars.
   `attn_comp_stage` becomes `attn_comp_stage_by_tier`.
6. **Tier-replicate the allocator.** Class P decode + Class P batch
   scratch loops over `used_tier[]` derived from placement. Class H goes
   on `head_tier = placement[DS4_N_LAYER + 1]`. Class E goes on
   `emb_tier = placement[0]`. Per-layer Class L (`layer_attn_state_kv/
   score`, `layer_index_state_kv/score`) goes on `layer_tier`.
7. **`cur_hc` / `after_ffn_hc` swap MUST use `g->active_tier` for both
   LHS and RHS.** The post-layer "swap saved cur with after_ffn" pattern
   recurs in 4+ sites. Hardcoded `[0]` here was the proximate cause of
   the multi-tier xdev failure: tier-1's `cur_hc_by_tier[0]` was
   overwritten with a tier-0 pointer interpreted as a tier-1 device VA.
8. **Output-head saved_cur capture:** record `saved_tier = g->active_tier`
   BEFORE `metal_graph_encode_output_head(...)` and use it for ALL
   subsequent cur_hc slot writes. The head encode can switch active_tier
   to head_tier internally.
9. **Apple stubs:** keep HEAD's `ds4_gpu_init_multi` returning 0; add
   short-circuiting `_alloc_ptr_on`/`_alloc_managed_on`/`_copy_xdev`/
   `_set_current_device` stubs that delegate to single-tier semantics.
10. **`metal_graph_alloc_raw_cap` must force `g->active_tier = 0` and
    `ds4_gpu_set_current_device(0)` immediately after `memset(g)`.** This
    is required because the per-tier allocation loops below switch
    devices via WITH_DEVICE() and would otherwise leave the active CUDA
    device at tier 1 when warmup runs.

## Non-D5 carry-forward (still open, not in scope for this PR)

- CPU Q8_0/Q8_K routed-expert kernels: typedefs and helper functions
  exist but are not yet wired through `matvec_experts_mid_prequant`
  (signature mismatch). Single-tier CUDA path uses Q8_0/Q8_K through the
  existing dispatch; CPU path does not.
- Peer-access validation fails on the box (RTX 6000 Ada x2). Currently
  falls back to pinned-host bounce. Functional but slower than peer
  copy would be. This is environmental / setup-dependent, not a code
  bug.
