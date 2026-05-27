/* test_engine_mgpu_placement — wave-2 placement-classification regression.
 *
 * Exercises the engine-side classify path (tensor_to_entry,
 * engine_compute_entry_bytes, engine_classify_multi_tier) via the
 * DS4_TEST_HOOKS-gated public helpers. Compiles only when ds4.c is
 * built with -DDS4_TEST_HOOKS (the test target adds this flag).
 *
 * Scenarios:
 *  1. NULL config: no_op, multi_tier == 0, n_entries == 0.
 *  2. Tensor classifier: bounded ds4_str parsing (no NUL).
 *  3. Forced multi-tier no-CPU placement: 2 GPUs, both budgets force a
 *     transition without CPU spill. multi_tier == 1, monotonic, both
 *     tiers used.
 *  4. CPU-spill placement: 2 GPUs with tiny budgets so some layers
 *     spill. multi_tier == 1 and at least one DS4_LAYER_PACK_CPU entry. */

#define DS4_TEST_HOOKS
#include "../ds4.h"
#include "../ds4_gpu_mgpu.h"
#include "../ds4_layer_pack.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* These match the typedef in ds4.c under DS4_TEST_HOOKS. */
typedef struct {
    const char *name;
    uint64_t    bytes;
} ds4_test_fake_tensor;

int ds4_test_classify_multi_tier(const ds4_test_fake_tensor *tensors,
                                  int n_tensors,
                                  const ds4_gpu_config *cfg,
                                  int placement_out[],
                                  int *out_multi_tier,
                                  int *out_n_entries);
int ds4_test_tensor_to_entry(const char *name, int name_len);

/* DS4_N_LAYER constant is private to ds4.c; for the test we use
 * the same value. (The packer header doesn't expose it.) */
#define DS4_N_LAYER_LOCAL 43
#define DS4_N_ENTRIES (DS4_N_LAYER_LOCAL + 2)

static int g_failures = 0;
static int g_checks = 0;

#define CHECK(cond, msg) do { \
    g_checks++; \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL: %s (line %d)\n", msg, __LINE__); \
        g_failures++; \
    } \
} while (0)

static void test_tensor_to_entry(void) {
    fprintf(stderr, "RUN: test_tensor_to_entry\n");
    /* Bounded name buffer to confirm we never read past name_len. */
    char buf[64];

    /* "blk.0.attn_norm.weight" should map to entry 1 (layer 0 + 1). */
    memcpy(buf, "blk.0.attn_norm.weight", 22);
    CHECK(ds4_test_tensor_to_entry(buf, 22) == 1, "blk.0.* -> entry 1");

    /* "blk.42.ffn_norm.weight" -> entry 43 (layer 42 + 1). */
    memcpy(buf, "blk.42.ffn_norm.weight", 22);
    CHECK(ds4_test_tensor_to_entry(buf, 22) == 43, "blk.42.* -> entry 43");

    /* "blk.43.x" — layer 43 is out of range (DS4_N_LAYER=43, layers are 0..42) */
    memcpy(buf, "blk.43.x", 8);
    CHECK(ds4_test_tensor_to_entry(buf, 8) == 0, "blk.43.* out of range");

    /* "output.weight" -> entry 44 (head). */
    memcpy(buf, "output.weight", 13);
    CHECK(ds4_test_tensor_to_entry(buf, 13) == 44, "output.weight -> entry 44");

    /* "output_norm.weight" -> entry 44. */
    memcpy(buf, "output_norm.weight", 18);
    CHECK(ds4_test_tensor_to_entry(buf, 18) == 44, "output_norm.weight -> entry 44");

    /* "token_embd.weight" -> entry 0. */
    memcpy(buf, "token_embd.weight", 17);
    CHECK(ds4_test_tensor_to_entry(buf, 17) == 0, "token_embd.weight -> entry 0");

    /* "mtp.0.foo" -> entry 44. */
    memcpy(buf, "mtp.0.foo", 9);
    CHECK(ds4_test_tensor_to_entry(buf, 9) == 44, "mtp.* -> head");

    /* Bounded parsing: pass a long buffer with garbage past name_len. */
    const char with_trailing[] = "blk.5.attn_norm.weightTRAILINGGARBAGE";
    CHECK(ds4_test_tensor_to_entry(with_trailing, 22) == 6,
          "bounded parsing ignores trailing bytes");

    /* Empty name -> entry 0. */
    CHECK(ds4_test_tensor_to_entry("", 0) == 0, "empty name -> entry 0");
}

static void test_null_config(void) {
    fprintf(stderr, "RUN: test_null_config\n");
    int placement[DS4_N_ENTRIES];
    int multi_tier = 99;
    int n_entries = 99;

    /* A trivial fake tensor list. */
    ds4_test_fake_tensor tensors[] = {
        {"token_embd.weight", 4096},
        {"output.weight", 4096},
    };
    int rc = ds4_test_classify_multi_tier(tensors,
                                           (int)(sizeof(tensors)/sizeof(tensors[0])),
                                           NULL,
                                           placement, &multi_tier, &n_entries);
    CHECK(rc == 0, "NULL cfg returns success");
    CHECK(multi_tier == 0, "NULL cfg -> multi_tier 0");
    CHECK(n_entries == 0, "NULL cfg -> n_entries 0");
}

/* Build a synthetic, model-shaped tensor list: 1 embedding + 43 layers
 * (each with 2 tensors of equal size) + 1 output head. Used by the
 * multi-tier tests to drive a realistic placement decision. */
static int build_synthetic_model(ds4_test_fake_tensor *out, int cap) {
    int n = 0;
    static char names[1024][32];

    /* Embedding. */
    snprintf(names[n], 32, "token_embd.weight");
    out[n].name = names[n]; out[n].bytes = (uint64_t)8ull * 1024 * 1024;
    n++;

    /* Per-layer tensors. */
    for (int il = 0; il < DS4_N_LAYER_LOCAL; il++) {
        snprintf(names[n], 32, "blk.%d.attn_q.weight", il);
        out[n].name = names[n]; out[n].bytes = (uint64_t)256ull * 1024 * 1024;
        n++;
        snprintf(names[n], 32, "blk.%d.ffn_down.weight", il);
        out[n].name = names[n]; out[n].bytes = (uint64_t)768ull * 1024 * 1024;
        n++;
        if (n + 2 > cap) return -1;
    }

    /* Output head. */
    snprintf(names[n], 32, "output.weight");
    out[n].name = names[n]; out[n].bytes = (uint64_t)16ull * 1024 * 1024;
    n++;
    snprintf(names[n], 32, "output_norm.weight");
    out[n].name = names[n]; out[n].bytes = (uint64_t)1ull * 1024 * 1024;
    n++;
    return n;
}

static void test_forced_two_tier_no_spill(void) {
    fprintf(stderr, "RUN: test_forced_two_tier_no_spill\n");
    ds4_test_fake_tensor tensors[256];
    int n = build_synthetic_model(tensors, 256);
    CHECK(n > 0, "synthetic model built");
    if (n <= 0) return;

    /* Sum approx total weights:
     *   1 embed + 43 layers * 1024 MiB + 1 head ~ 43 GiB.
     * Pick budgets that force a transition. The packer also adds a
     * per-layer KV estimate that the engine computes; using equal
     * budgets sized below the total guarantees a transition without
     * CPU spill. */
    ds4_gpu_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.n_gpus = 2;
    cfg.device_indices[0] = 0;
    cfg.device_indices[1] = 1;
    /* Total synthetic weights ~ 44 GiB plus per-layer KV estimate from
     * ds4_context_memory_estimate(CUDA, 4096). Pick budgets near half
     * the total so the packer is forced to split across both tiers
     * but with enough headroom on each to avoid CPU spill. */
    cfg.vram_bytes[0] = (size_t)28ull * 1024ull * 1024ull * 1024ull;
    cfg.vram_bytes[1] = (size_t)40ull * 1024ull * 1024ull * 1024ull;
    cfg.safety_margin_bytes = 0;

    int placement[DS4_N_ENTRIES];
    int multi_tier = 0;
    int n_entries = 0;
    int rc = ds4_test_classify_multi_tier(tensors, n, &cfg,
                                           placement, &multi_tier, &n_entries);
    CHECK(rc == 0, "classify succeeded");
    CHECK(n_entries == DS4_N_ENTRIES, "n_entries == DS4_N_LAYER + 2");
    CHECK(multi_tier == 1, "multi_tier set");

    /* Monotonic-contiguous (wave-1 packer guarantee): each successive
     * entry's tier is >= previous, with CPU treated as a higher
     * "spill" tier. We assert no decrease. */
    int prev = placement[0];
    int saw_0 = 0, saw_1 = 0, saw_cpu = 0;
    for (int i = 0; i < n_entries; i++) {
        int cur = placement[i];
        CHECK(cur == prev || cur > prev || cur == DS4_LAYER_PACK_CPU,
              "monotonic (cur >= prev or CPU)");
        if (cur == 0) saw_0 = 1;
        else if (cur == 1) saw_1 = 1;
        else if (cur == DS4_LAYER_PACK_CPU) saw_cpu = 1;
        prev = cur;
    }
    CHECK(saw_0 && saw_1, "both tiers used");
    CHECK(!saw_cpu, "no CPU spill for this budget");
}

static void test_cpu_spill(void) {
    fprintf(stderr, "RUN: test_cpu_spill\n");
    ds4_test_fake_tensor tensors[256];
    int n = build_synthetic_model(tensors, 256);
    if (n <= 0) return;

    ds4_gpu_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.n_gpus = 2;
    cfg.device_indices[0] = 0;
    cfg.device_indices[1] = 1;
    /* Tiny budgets: ~5 GiB each, but total weights are ~43 GiB +
     * per-layer KV estimate, so most layers spill to CPU. */
    cfg.vram_bytes[0] = (size_t)5ull * 1024ull * 1024ull * 1024ull;
    cfg.vram_bytes[1] = (size_t)5ull * 1024ull * 1024ull * 1024ull;

    int placement[DS4_N_ENTRIES];
    int multi_tier = 0;
    int n_entries = 0;
    int rc = ds4_test_classify_multi_tier(tensors, n, &cfg,
                                           placement, &multi_tier, &n_entries);
    CHECK(rc == 0, "classify succeeded");
    CHECK(multi_tier == 1, "multi_tier set with CPU spill");
    int any_cpu = 0;
    for (int i = 0; i < n_entries; i++) {
        if (placement[i] == DS4_LAYER_PACK_CPU) { any_cpu = 1; break; }
    }
    CHECK(any_cpu, "at least one CPU spill entry");
}

int main(void) {
    test_tensor_to_entry();
    test_null_config();
    test_forced_two_tier_no_spill();
    test_cpu_spill();

    fprintf(stderr, "\ntest_engine_mgpu_placement: %d/%d checks passed (%d failed)\n",
            g_checks - g_failures, g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
