/* ds4_gpu_mgpu.h — multi-GPU plumbing types and APIs (v0).
 *
 * This header carries the new multi-GPU additions for the wave-1 PP work
 * (mgpu-device-aware-cuda). It is included from ds4_cuda.cu and from
 * downstream tasks that need access to g_gpu[], g_n_gpus, g_gpu_peer_ok[],
 * the ds4_gpu_config struct, and the new tensor APIs.
 *
 * Why not in ds4_gpu.h? The legacy ds4_gpu.h is included from C-only
 * callers (ds4.c, ds4_cli.c, etc.) and from the Metal build, but is NOT
 * included from ds4_cuda.cu historically. That asymmetry hid pre-existing
 * signature mismatches between the legacy header and ds4_cuda.cu. We keep
 * the legacy header opaque and put the new shared types here, so this
 * file is the single source of truth for both ds4_cuda.cu and downstream
 * multi-GPU tasks without disturbing the legacy contract.
 *
 * The struct definitions reference CUDA-specific handle types via void *
 * placeholders so the header is safe to include from C builds, Metal
 * builds, and the CUDA build (where ds4_cuda.cu casts the void * back
 * to cudaStream_t / cublasHandle_t / cudaEvent_t internally).
 */
#ifndef DS4_GPU_MGPU_H
#define DS4_GPU_MGPU_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DS4_MAX_GPUS 16

/* Complete definition of the previously-opaque ds4_gpu_tensor, plus a
 * typedef so the new API prototypes below can use the bare name
 * `ds4_gpu_tensor *` in both C and C++ without forcing callers to
 * include ds4_gpu.h first. Callers that include this header can
 * stack-allocate or struct-embed tensors and pass to
 * ds4_gpu_tensor_alloc_on. */
struct ds4_gpu_tensor {
    void    *ptr;
    uint64_t bytes;
    int      owner;
    int      device_id;   /* -1 means legacy/untagged → treat as device 0 */
};
typedef struct ds4_gpu_tensor ds4_gpu_tensor;

/* Tagged so headers (notably ds4.h) can forward-declare `struct
 * ds4_gpu_config` without dragging in this entire header. */
typedef struct ds4_gpu_config {
    int    device_indices[DS4_MAX_GPUS];   /* CUDA device IDs to use */
    size_t vram_bytes[DS4_MAX_GPUS];       /* per-device budget; 0 = unset */
    int    n_gpus;
    size_t safety_margin_bytes;            /* per-device reserve */
} ds4_gpu_config;

typedef struct {
    int    device_id;
    void  *stream;             /* cudaStream_t under CUDA */
    void  *cublas;             /* cublasHandle_t under CUDA */
    int    cublas_ready;
    void  *scratch;
    size_t scratch_bytes;
    size_t budget_bytes;
    size_t used_bytes;
    void  *boundary_event;     /* cudaEvent_t under CUDA */
} ds4_gpu_ctx;

extern ds4_gpu_ctx g_gpu[DS4_MAX_GPUS];
extern int         g_n_gpus;
extern int         g_gpu_peer_ok[DS4_MAX_GPUS][DS4_MAX_GPUS];

/* Primary multi-device init. The existing ds4_gpu_init (declared in
 * ds4_gpu.h) is a thin shim that builds a single-device config for
 * device 0 and calls this. */
int ds4_gpu_init_multi(const ds4_gpu_config *cfg);

/* Caller-supplied struct alloc on a specific device. Returns 0 on
 * success, nonzero on error. Pair with ds4_gpu_tensor_free_in_place. */
int  ds4_gpu_tensor_alloc_on(ds4_gpu_tensor *t, int device_id, uint64_t bytes);
void ds4_gpu_tensor_free_in_place(ds4_gpu_tensor *t);

/* Cross-device tensor copy. Same-device → cudaMemcpyAsync; peer-capable
 * cross-device → cudaMemcpyPeerAsync with event sync; non-peer → pinned
 * host bounce (per src→dst pair). Honors DS4_FORCE_HOST_BOUNCE=1. */
int ds4_gpu_tensor_copy_xdev(ds4_gpu_tensor *dst,
                              const ds4_gpu_tensor *src,
                              uint64_t bytes);

/* Returns the device_id recorded on the tensor; -1 if untagged. */
int ds4_gpu_tensor_device(const ds4_gpu_tensor *t);

/* Set the current CUDA device by LOGICAL tier index (0..g_n_gpus-1).
 *
 * This is the canonical shim for per-layer device routing in the
 * multi-tier execution path. The caller passes a logical tier index;
 * the shim internally indexes g_gpu[tier].device_id and calls
 * cudaSetDevice. Returns 0 on success, nonzero on error or if the
 * tier index is out of range.
 *
 * Wave-2 mgpu-graph-session-placement adds this shim but does not
 * exercise the multi-tier execution path; mgpu-graph-session-execution
 * (follow-up) is its first caller. */
int ds4_gpu_set_current_device(int logical_tier);

/* Register the mmap'd host model pointer for selective-cache lookups
 * WITHOUT triggering any device-side copy. This bypasses the
 * DS4_CUDA_COPY_MODEL environment-variable branch that
 * ds4_gpu_set_model_map normally honors, which is essential for
 * multi-tier startup: we want only per-device selective tensor caches,
 * never the whole-model copy. Returns 1 on success, 0 on error.
 *
 * Added for mgpu-graph-session-placement (wave 2) per codex
 * plan-review round 3 finding #1. */
int ds4_gpu_register_model_map_no_copy(const void *model_map, uint64_t model_size);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* DS4_GPU_MGPU_H */
