/* Cross-device GPU tensor copy test (mgpu-device-aware-cuda).
 *
 * Exercises:
 *   - ds4_gpu_init_multi single-device and multi-device init paths.
 *   - ds4_gpu_tensor_alloc_on / ds4_gpu_tensor_free_in_place.
 *   - ds4_gpu_tensor_copy_xdev same-device fast path.
 *   - ds4_gpu_tensor_copy_xdev peer-auto and DS4_FORCE_HOST_BOUNCE paths
 *     (when 2+ GPUs are visible). */

/* ds4_gpu_mgpu.h is standalone-C-compatible — it now provides the
 * complete ds4_gpu_tensor struct + typedef so callers can use the
 * bare type name. ds4_gpu.h is also included here only for the
 * legacy ds4_gpu_init / _cleanup / _tensor_read / _tensor_write
 * prototypes the test uses; the mgpu header does not duplicate them. */
#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "FAIL: %s (line %d)\n", (msg), __LINE__);       \
            return 1;                                                       \
        }                                                                   \
    } while (0)

static int run_one(int n_gpus_wanted, int force_bounce) {
    int dev_count = 0;
    (void)cudaGetDeviceCount(&dev_count);
    if (dev_count < 1) {
        fprintf(stderr, "no CUDA devices visible\n");
        return 0;
    }
    if (n_gpus_wanted > dev_count) {
        fprintf(stderr, "skip: wanted %d GPUs, have %d\n",
                n_gpus_wanted, dev_count);
        return 0;
    }

    ds4_gpu_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.n_gpus = n_gpus_wanted;
    for (int i = 0; i < n_gpus_wanted; i++) cfg.device_indices[i] = i;
    CHECK(ds4_gpu_init_multi(&cfg), "init_multi");

    if (force_bounce) setenv("DS4_FORCE_HOST_BOUNCE", "1", 1);
    else              unsetenv("DS4_FORCE_HOST_BOUNCE");

    const size_t N = 256 * 1024;
    float *host_src = (float *)malloc(N * sizeof(float));
    float *host_dst = (float *)malloc(N * sizeof(float));
    CHECK(host_src && host_dst, "host alloc");
    for (size_t i = 0; i < N; i++) host_src[i] = (float)(i % 997) * 0.5f;

    ds4_gpu_tensor a; memset(&a, 0, sizeof(a));
    ds4_gpu_tensor b; memset(&b, 0, sizeof(b));
    CHECK(ds4_gpu_tensor_alloc_on(&a, 0, N * sizeof(float)) == 0,
          "alloc_on dev 0");
    int dst_dev = (n_gpus_wanted == 2) ? 1 : 0;
    CHECK(ds4_gpu_tensor_alloc_on(&b, dst_dev, N * sizeof(float)) == 0,
          "alloc_on dst dev");

    CHECK(ds4_gpu_tensor_write(&a, 0, host_src, N * sizeof(float)),
          "tensor_write");
    CHECK(ds4_gpu_tensor_copy_xdev(&b, &a, N * sizeof(float)),
          "tensor_copy_xdev");
    CHECK(ds4_gpu_tensor_read(&b, 0, host_dst, N * sizeof(float)),
          "tensor_read");
    CHECK(memcmp(host_src, host_dst, N * sizeof(float)) == 0,
          "data mismatch");

    /* device_id round-trip. */
    CHECK(ds4_gpu_tensor_device(&a) == 0, "device_id a");
    CHECK(ds4_gpu_tensor_device(&b) == dst_dev, "device_id b");

    ds4_gpu_tensor_free_in_place(&a);
    ds4_gpu_tensor_free_in_place(&b);
    free(host_src);
    free(host_dst);
    ds4_gpu_cleanup();
    fprintf(stderr, "  test_gpu_xdev OK (n_gpus=%d, force_bounce=%d)\n",
            n_gpus_wanted, force_bounce);
    return 0;
}

/* Stress sub-test: repeated cross-device round-trips at realistic sizes.
 * Designed to catch the RTX 6000 Ada / driver bug where peer copies pass
 * a single small probe but corrupt data at larger sizes or on repeat. */
static int run_stress(int force_bounce) {
    int dev_count = 0;
    (void)cudaGetDeviceCount(&dev_count);
    if (dev_count < 2) {
        fprintf(stderr, "  skipping stress (need >= 2 devices)\n");
        return 0;
    }

    ds4_gpu_config cfg; memset(&cfg, 0, sizeof(cfg));
    cfg.n_gpus = 2;
    cfg.device_indices[0] = 0; cfg.device_indices[1] = 1;
    CHECK(ds4_gpu_init_multi(&cfg), "stress init_multi");

    if (force_bounce) setenv("DS4_FORCE_HOST_BOUNCE", "1", 1);
    else              unsetenv("DS4_FORCE_HOST_BOUNCE");

    const size_t sizes[] = {
        256u * 1024u,
        1u * 1024u * 1024u,
        16u * 1024u * 1024u,
    };
    const int    n_sizes = (int)(sizeof(sizes) / sizeof(sizes[0]));
    const int    iters   = 32;
    const size_t max_n   = sizes[n_sizes - 1];

    unsigned char *host_src = (unsigned char *)malloc(max_n);
    unsigned char *host_dst = (unsigned char *)malloc(max_n);
    CHECK(host_src && host_dst, "stress host alloc");

    ds4_gpu_tensor a; memset(&a, 0, sizeof(a));
    ds4_gpu_tensor b; memset(&b, 0, sizeof(b));
    CHECK(ds4_gpu_tensor_alloc_on(&a, 0, max_n) == 0, "stress alloc_on 0");
    CHECK(ds4_gpu_tensor_alloc_on(&b, 1, max_n) == 0, "stress alloc_on 1");

    int total_ok = 1;
    for (int s = 0; s < n_sizes && total_ok; s++) {
        size_t n = sizes[s];
        for (int it = 0; it < iters && total_ok; it++) {
            for (size_t k = 0; k < n; k++) {
                host_src[k] = (unsigned char)
                    ((k * 31u + (size_t)it * 17u +
                      (size_t)s * 53u + 11u) & 0xffu);
            }
            memset(host_dst, 0, n);
            int io_ok = ds4_gpu_tensor_write(&a, 0, host_src, n) &&
                        ds4_gpu_tensor_copy_xdev(&b, &a, n) &&
                        ds4_gpu_tensor_read(&b, 0, host_dst, n);
            if (!io_ok) {
                fprintf(stderr,
                    "FAIL: stress IO error size=%zu iter=%d force_bounce=%d\n",
                    n, it, force_bounce);
                total_ok = 0;
                break;
            }
            for (size_t k = 0; k < n; k++) {
                if (host_src[k] != host_dst[k]) {
                    fprintf(stderr,
                        "FAIL: stress data mismatch size=%zu iter=%d offset=%zu"
                        " src=0x%02x dst=0x%02x force_bounce=%d\n",
                        n, it, k,
                        host_src[k], host_dst[k], force_bounce);
                    total_ok = 0;
                    break;
                }
            }
        }
    }

    ds4_gpu_tensor_free_in_place(&a);
    ds4_gpu_tensor_free_in_place(&b);
    free(host_src);
    free(host_dst);
    ds4_gpu_cleanup();
    if (total_ok) {
        fprintf(stderr,
            "  stress OK (force_bounce=%d, %d iters x %d sizes, max %zu MiB)\n",
            force_bounce, iters, n_sizes, max_n / (1024u * 1024u));
    }
    return total_ok ? 0 : 1;
}

int main(void) {
    int dev_count = 0;
    (void)cudaGetDeviceCount(&dev_count);
    fprintf(stderr, "test_gpu_xdev: %d CUDA devices visible\n", dev_count);

    /* N=1 same-device path. */
    if (run_one(1, 0)) return 1;
    /* If 2+ GPUs, exercise peer-auto and forced-bounce paths. */
    if (dev_count >= 2) {
        if (run_one(2, 0)) return 1;
        if (run_one(2, 1)) return 1;
        /* Stress: catches the cudaMemcpyPeer driver-corruption pattern that
         * single-size single-iter tests can miss. Runs both peer-auto and
         * forced-bounce so we get coverage of both code paths under load. */
        if (run_stress(0)) return 1;
        if (run_stress(1)) return 1;
    } else {
        fprintf(stderr, "  skipping multi-GPU paths (need >= 2 devices)\n");
    }
    fprintf(stderr, "test_gpu_xdev PASS\n");
    return 0;
}
