/* Unit tests for the monotonic-contiguous layer placement packer.
 *
 * Pure C99: no CUDA, no Metal. Builds and runs on any host. */

#include "ds4_layer_pack.h"

#include <locale.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failed = 0;
static int g_total  = 0;

#define CHECK(cond, msg) do {                                                  \
    g_total++;                                                                 \
    if (!(cond)) {                                                             \
        fprintf(stderr, "  FAIL: %s (line %d)\n", (msg), __LINE__);            \
        g_failed++;                                                            \
    }                                                                          \
} while (0)

#define RUN(fn) do {                                                           \
    fprintf(stderr, "RUN: %s\n", #fn);                                         \
    int _before = g_failed;                                                    \
    (fn)();                                                                    \
    fprintf(stderr, "  %s\n", (_before == g_failed) ? "ok" : "FAIL");          \
} while (0)

/* Monotonicity invariant: tier(e) <= tier(e+1) where CPU is the max tier. */
static int check_monotonic(const int *dev, int n) {
    int prev_rank = -1;
    for (int i = 0; i < n; i++) {
        int rank = (dev[i] == DS4_LAYER_PACK_CPU) ? 999999 : dev[i];
        if (rank < prev_rank) return 0;
        prev_rank = rank;
    }
    return 1;
}

/* ----- scenarios ----- */

static void test_all_fit_n1(void) {
    const size_t entries[] = {10, 10, 10, 10, 10, 10};
    ds4_layer_pack_config cfg = { .n_gpus = 1 };
    cfg.gpu_budget_bytes[0] = 100;
    int dev[6];
    CHECK(ds4_compute_layer_placement(entries, 6, &cfg, dev) == 0, "rc 0");
    for (int i = 0; i < 6; i++) CHECK(dev[i] == 0, "all on dev 0");
    CHECK(check_monotonic(dev, 6), "monotonic");
}

static void test_all_fit_n2(void) {
    /* Budgets 50/50, entries {20,20,20,20}. Greedy: dev0 takes 20+20=40;
     * next 20 → 40+20=60 > 50, advance to dev1. dev1 takes 20+20=40.
     * Expected: {0,0,1,1}. */
    const size_t entries[] = {20, 20, 20, 20};
    ds4_layer_pack_config cfg = { .n_gpus = 2 };
    cfg.gpu_budget_bytes[0] = 50;
    cfg.gpu_budget_bytes[1] = 50;
    int dev[4];
    CHECK(ds4_compute_layer_placement(entries, 4, &cfg, dev) == 0, "rc 0");
    CHECK(dev[0] == 0 && dev[1] == 0, "entries 0,1 on dev 0");
    CHECK(dev[2] == 1 && dev[3] == 1, "entries 2,3 on dev 1");
    CHECK(check_monotonic(dev, 4), "monotonic");
}

static void test_partial_spill_n2(void) {
    /* Budgets 30/30, entries {15,15,15,15,15}. Sum 75 > 60 → tail to CPU.
     * dev0 takes 15+15=30; next 15 → 45 > 30, advance.
     * dev1 takes 15+15=30; next 15 → 45 > 30, advance → CPU. */
    const size_t entries[] = {15, 15, 15, 15, 15};
    ds4_layer_pack_config cfg = { .n_gpus = 2 };
    cfg.gpu_budget_bytes[0] = 30;
    cfg.gpu_budget_bytes[1] = 30;
    int dev[5];
    CHECK(ds4_compute_layer_placement(entries, 5, &cfg, dev) == 0, "rc 0");
    CHECK(dev[0] == 0 && dev[1] == 0, "0,1 on dev 0");
    CHECK(dev[2] == 1 && dev[3] == 1, "2,3 on dev 1");
    CHECK(dev[4] == DS4_LAYER_PACK_CPU, "4 on CPU");
    CHECK(check_monotonic(dev, 5), "monotonic");
}

static void test_zero_budget_one_gpu_n2(void) {
    /* Budgets {50, 0}, entries {10,10,10,10,10,10}. dev0 holds five (50/50).
     * Next 10 → 60 > 50, advance to dev1; dev1 budget 0, 10 > 0, advance →
     * CPU. The zero-budget device is effectively skipped. */
    const size_t entries[] = {10, 10, 10, 10, 10, 10};
    ds4_layer_pack_config cfg = { .n_gpus = 2 };
    cfg.gpu_budget_bytes[0] = 50;
    cfg.gpu_budget_bytes[1] = 0;
    int dev[6];
    CHECK(ds4_compute_layer_placement(entries, 6, &cfg, dev) == 0, "rc 0");
    for (int i = 0; i < 5; i++) CHECK(dev[i] == 0, "first five on dev 0");
    CHECK(dev[5] == DS4_LAYER_PACK_CPU, "sixth on CPU (dev 1 skipped)");
    for (int i = 0; i < 6; i++) CHECK(dev[i] != 1, "nothing on dev 1");
    CHECK(check_monotonic(dev, 6), "monotonic");
}

static void test_oversized_entry_n2(void) {
    /* Entry too large for any device → CPU, and the rule propagates. */
    const size_t entries[] = {10, 40, 10};
    ds4_layer_pack_config cfg = { .n_gpus = 2 };
    cfg.gpu_budget_bytes[0] = 30;
    cfg.gpu_budget_bytes[1] = 30;
    int dev[3];
    CHECK(ds4_compute_layer_placement(entries, 3, &cfg, dev) == 0, "rc 0");
    CHECK(dev[0] == 0, "small first on dev 0");
    CHECK(dev[1] == DS4_LAYER_PACK_CPU, "oversized on CPU");
    CHECK(dev[2] == DS4_LAYER_PACK_CPU, "post-oversized stays CPU");
    CHECK(check_monotonic(dev, 3), "monotonic");
}

static void test_mixed_n8(void) {
    /* Eight GPUs, varied budgets, varied entries. */
    const size_t entries[] = {
        50, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 50
    };
    ds4_layer_pack_config cfg = { .n_gpus = 8 };
    for (int d = 0; d < 8; d++) cfg.gpu_budget_bytes[d] = 200;
    /* Total = 50 + 14*100 + 50 = 1500. 8 budgets of 200 = 1600. Fits. */
    int dev[16];
    CHECK(ds4_compute_layer_placement(entries, 16, &cfg, dev) == 0, "rc 0");
    CHECK(check_monotonic(dev, 16), "monotonic");
    /* No entry exceeds 200 -> nothing on CPU. */
    for (int i = 0; i < 16; i++) {
        CHECK(dev[i] >= 0 && dev[i] < 8, "no CPU spill");
    }
}

static void test_n16_stress(void) {
    /* 16 GPUs each budget 100; 32 entries of 50 each → {0,0,1,1,...,15,15}. */
    ds4_layer_pack_config cfg = { .n_gpus = 16 };
    for (int d = 0; d < 16; d++) cfg.gpu_budget_bytes[d] = 100;
    size_t entries[32];
    for (int i = 0; i < 32; i++) entries[i] = 50;
    int dev[32];
    CHECK(ds4_compute_layer_placement(entries, 32, &cfg, dev) == 0, "rc 0");
    for (int i = 0; i < 32; i++) {
        CHECK(dev[i] == i / 2, "pair per device");
    }
    CHECK(check_monotonic(dev, 32), "monotonic");
}

static void test_pseudo_layers(void) {
    /* 43 layers + 2 pseudo. Embedding small, output head small, layers
     * medium-sized. Three 60-byte budgets. */
    const int n_layers = 43;
    const int n_entries = n_layers + 2;
    size_t entries[45];
    entries[0] = 5;                 /* embedding */
    for (int i = 1; i <= n_layers; i++) entries[i] = 10;
    entries[n_layers + 1] = 7;      /* output head */
    ds4_layer_pack_config cfg = { .n_gpus = 3 };
    cfg.gpu_budget_bytes[0] = 60;
    cfg.gpu_budget_bytes[1] = 60;
    cfg.gpu_budget_bytes[2] = 60;
    int dev[45];
    CHECK(ds4_compute_layer_placement(entries, n_entries, &cfg, dev) == 0, "rc 0");
    /* Embedding placed first → on dev 0 (5 bytes, plenty of room). */
    CHECK(dev[0] == 0, "embedding on dev 0");
    /* Total = 5 + 43*10 + 7 = 442. 3 budgets of 60 = 180. Mostly CPU.
     * The output head lands wherever the chain ends. With 5 + 10*5 = 55 on
     * dev 0 (5 embedding + 5 layers), next 10 → 65 > 60 → advance to dev 1.
     * Dev 1 holds 6 layers (60). Dev 2 holds 6 layers (60). After that
     * everything is CPU. Output head ends up on CPU. */
    CHECK(dev[n_layers + 1] == DS4_LAYER_PACK_CPU, "output head on CPU");
    CHECK(check_monotonic(dev, n_entries), "monotonic");
}

static void test_null_inputs(void) {
    size_t one = 0;
    int dev = 0;
    ds4_layer_pack_config cfg = { .n_gpus = 1 };
    cfg.gpu_budget_bytes[0] = 100;

    CHECK(ds4_compute_layer_placement(NULL, 1, &cfg, &dev) != 0, "null entries");
    CHECK(ds4_compute_layer_placement(&one, 1, NULL, &dev) != 0, "null cfg");
    CHECK(ds4_compute_layer_placement(&one, 1, &cfg, NULL) != 0, "null out");

    ds4_layer_pack_config bad = { .n_gpus = -1 };
    CHECK(ds4_compute_layer_placement(&one, 1, &bad, &dev) != 0, "n_gpus<0");
    bad.n_gpus = DS4_LAYER_PACK_MAX_GPUS + 1;
    CHECK(ds4_compute_layer_placement(&one, 1, &bad, &dev) != 0, "n_gpus too big");
}

/* Golden-string print test.
 *
 * Setup: n_layers=4 (so entries are indices 0..5), with placement
 *   entry 0  (embedding)   -> dev 0
 *   entry 1  (layer 0)     -> dev 0
 *   entry 2  (layer 1)     -> dev 0
 *   entry 3  (layer 2)     -> dev 1
 *   entry 4  (layer 3)     -> CPU
 *   entry 5  (output head) -> CPU
 *
 * Used/budget GB chosen so the printed numbers are stable (no rounding
 * ambiguity): used = 10 GiB, budget = 20 GiB → "10.0 / 20.0 GB" on each
 * device.  GiB = 1073741824. */
static void test_print_golden(void) {
    setlocale(LC_ALL, "C");
    const int n_layers = 4;
    const int n_entries = n_layers + 2;
    int dev[6] = { 0, 0, 0, 1, DS4_LAYER_PACK_CPU, DS4_LAYER_PACK_CPU };
    size_t entry_bytes[6] = {0, 0, 0, 0, 0, 0};
    size_t used[2]   = {10ull * 1073741824ull, 10ull * 1073741824ull};
    size_t budget[2] = {20ull * 1073741824ull, 20ull * 1073741824ull};

    char buf[1024];
    FILE *f = fmemopen(buf, sizeof(buf), "w");
    CHECK(f != NULL, "fmemopen");
    if (!f) return;
    ds4_layer_pack_print(f, dev, n_entries, n_layers, entry_bytes,
                          used, budget, 2);
    fflush(f);
    long pos = ftell(f);
    fclose(f);
    buf[pos < (long)sizeof(buf) ? pos : (long)sizeof(buf) - 1] = '\0';

    const char *expected =
        "multi-GPU layout:\n"
        "  GPU0: layers 0-1 + embedding  (10.0 / 20.0 GB)\n"
        "  GPU1: layer 2  (10.0 / 20.0 GB)\n"
        "  CPU : layer 3 + output head\n";
    if (strcmp(buf, expected) != 0) {
        fprintf(stderr, "  -- expected --\n%s", expected);
        fprintf(stderr, "  -- got --\n%s", buf);
    }
    CHECK(strcmp(buf, expected) == 0, "golden layout");
    CHECK(check_monotonic(dev, n_entries), "golden monotonic");
}

int main(void) {
    setlocale(LC_ALL, "C");
    RUN(test_all_fit_n1);
    RUN(test_all_fit_n2);
    RUN(test_partial_spill_n2);
    RUN(test_zero_budget_one_gpu_n2);
    RUN(test_oversized_entry_n2);
    RUN(test_mixed_n8);
    RUN(test_n16_stress);
    RUN(test_pseudo_layers);
    RUN(test_null_inputs);
    RUN(test_print_golden);

    fprintf(stderr, "\ntest_layer_pack: %d/%d checks passed (%d failed)\n",
            g_total - g_failed, g_total, g_failed);
    return g_failed == 0 ? 0 : 1;
}
