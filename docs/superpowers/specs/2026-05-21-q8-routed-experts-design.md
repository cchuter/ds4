# Q8 Routed Experts Design

Date: 2026-05-21
Branch: `feat/q8-routed-experts`

## Goal

Add runnable high-precision routed expert variants for DeepSeek V4 Flash in DS4.
The work is split into two phases:

1. Build and run a `Q8_0Experts` model using the existing GGUF `q8_0` type.
2. Build and run a true `Q8KExperts` model using GGUF `q8_K` expert weights.

Both phases must support CPU fallback and Metal execution.

## Non-Goals

- Do not modify the upstream DeepSeek HF model files.
- Do not change the existing q2/q4/imatrix download targets.
- Do not add CUDA Q8 expert support in this pass. The requested target is CPU
  plus Metal on this machine.
- Do not treat `Q8_0Experts` as the same thing as `Q8KExperts`; they are
  different GGUF tensor types and should have different filenames.

## Current State

The quantizer can already emit `q8_0` tensors via `--experts q8_0`.
The runtime currently validates routed expert tensors as only `IQ2_XXS`,
`Q2_K`, or `Q4_K`, so a `q8_0` routed-expert GGUF is rejected before inference.

Metal already exposes routed-MoE `q8_0` matmul kernels:

- `kernel_mul_mv_id_q8_0_f32`
- `kernel_mul_mm_id_q8_0_f32`
- `kernel_mul_mm_id_q8_0_f16`

The runtime also has Q8_K activation blocks and CPU dot products for existing
Q2/IQ2 expert weights. It does not currently have `q8_K` quantizer output or
runtime support for stored `q8_K` routed expert weights.

## Phase 1: Q8_0Experts

Output file:

```text
DeepSeek-V4-Flash-Q8_0Experts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2.gguf
```

Build command:

```sh
./gguf-tools/deepseek4-quantize \
  --hf /Users/cchuter/models/DeepSeek-V4-Flash \
  --template gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf \
  --out gguf/DeepSeek-V4-Flash-Q8_0Experts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2.gguf \
  --experts q8_0 \
  --threads 8
```

Runtime changes:

- Accept `DS4_TENSOR_Q8_0` as a routed expert tensor type.
- Compute routed expert row bytes correctly for `q8_0` 3D tensors.
- Add CPU routed expert projection helpers for `q8_0` gate/up/down tensors.
- Generalize routed MoE CPU dispatch so gate/up and down paths select the
  correct implementation by tensor type.
- Wire Metal routed MoE dispatch to the existing `q8_0` routed kernels.
- Update routed quant metadata so cache files report an 8-bit routed expert
  model instead of falling through to the old 2-bit default.

Expected result:

- The `Q8_0Experts` GGUF loads and runs on CPU.
- The same GGUF loads and runs through Metal.
- A short prompt produces streamed output without validation, layout, or cache
  metadata failures.

## Phase 2: Q8KExperts

Output file:

```text
DeepSeek-V4-Flash-Q8KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2.gguf
```

Quantizer changes:

- Add `q8_K` as a supported output target in `gguf-tools`.
- Reuse the existing DS4 `block_q8_K` layout, including `d`, `qs`, and
  `bsums`, so stored model weights match the runtime type definition.
- Add a focused quantizer comparison path for at least one routed expert tensor
  before writing the full GGUF.

Runtime changes:

- Accept `DS4_TENSOR_Q8_K` as a routed expert type.
- Add CPU dot support for stored `Q8_K` expert weights.
- Add Metal routed `Q8_K` kernels/templates and dispatch for decode and
  prefill paths.
- Update routed quant metadata to report an 8-bit routed expert model.

Expected result:

- The true `Q8KExperts` GGUF loads and runs on CPU.
- The same GGUF loads and runs through Metal.
- Output can be compared against Q4 and Q8_0 on a small prompt set.

## Data Flow

1. HF safetensors stay in `/Users/cchuter/models/DeepSeek-V4-Flash`.
2. `deepseek4-quantize` reads the HF tensors and a DS4 GGUF template.
3. The quantizer writes a new GGUF with the same tensor order, tokenizer, and
   metadata, but new routed expert tensor types.
4. DS4 loads the GGUF, validates tensor type/layout, and maps routed expert
   tensor bytes into CPU or Metal kernels.
5. A smoke prompt verifies end-to-end prompt rendering, routed MoE execution,
   token generation, streaming output, and cache metadata.

## Error Handling

- Model validation should fail early with tensor name, type, and expected
  routed expert type when an unsupported quantization is present.
- Metal dispatch should report unsupported routed gate/down type pairs instead
  of silently falling back.
- Quantizer full writes should require explicit output paths and should not
  overwrite existing GGUFs unless `--overwrite` is supplied.
- Cache metadata should distinguish 2-bit, 4-bit, and 8-bit routed expert
  models so strict cache reuse remains meaningful.

## Testing

Phase 1 checks:

- Build DS4 and `gguf-tools`.
- Run a dry-run quantizer plan and confirm only 129 routed expert tensors change
  from `q4_K` to `q8_0`.
- Generate the `Q8_0Experts` GGUF.
- Run CPU smoke inference with a short prompt.
- Run Metal smoke inference with the same short prompt.
- Run a cache metadata check for routed quant bits.

Phase 2 checks:

- Unit-level quantizer checks for `q8_K` row size and emitted bytes.
- Single-tensor compare or hash check for a routed expert tensor.
- Generate the `Q8KExperts` GGUF.
- Run CPU smoke inference.
- Run Metal smoke inference.
- Compare Q4, Q8_0, and Q8_K outputs on a small prompt set for basic sanity.

## Open Risks

- `Q8_0Experts` will be much larger than Q4 and may be slower despite higher
  precision because it moves more expert bytes.
- Q8_K model-weight support is not just a filename change; it requires new
  quantizer output and new runtime dot paths.
- Existing CUDA code has useful Q4/Q8_K references, but this implementation
  should remain focused on CPU plus Metal unless the scope changes.
