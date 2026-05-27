/* Cross-device GPU tensor copy test (mgpu-device-aware-cuda).
 *
 * Exercises:
 *   - ds4_gpu_init_multi single-device and multi-device init paths.
 *   - ds4_gpu_tensor_alloc_on / ds4_gpu_tensor_free_in_place.
 *   - ds4_gpu_tensor_copy_xdev same-device fast path.
 *   - ds4_gpu_tensor_copy_xdev peer-auto and DS4_FORCE_HOST_BOUNCE paths
 *     (when 2+ GPUs are visible). */

#include "ds4_gpu.h"

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
    } else {
        fprintf(stderr, "  skipping multi-GPU paths (need >= 2 devices)\n");
    }
    fprintf(stderr, "test_gpu_xdev PASS\n");
    return 0;
}
