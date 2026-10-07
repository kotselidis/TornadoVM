# Parked: CUDA dynamic shared memory above 48 KB

Status: parked. The change works and passes its tests, but no kernel that ships today needs it.
The code is in the commit before this document on this branch.

## What it does

CUDA limits statically declared shared memory to 48 KB per block. Larger amounts, up to the
device's opt-in limit (99 KB on sm_86, 227 KB on sm_90), must be dynamic shared memory, enabled
per kernel with `cuFuncSetAttribute` and passed at launch.

With this change, a TornadoVM kernel whose local arrays (`context.allocate*LocalArray`) add up to
more than 48 KB is compiled and launched with dynamic shared memory, with no change to the Java
kernel:

- **Code generation** (`CUDADynamicSharedMemory`, run from `CUDACompilationResultBuilder.finish()`
  for host-launched kernels): the local arrays become typed pointers, at 128-byte-aligned
  offsets, into one `extern __shared__ __align__(128) unsigned char tornado_dynamic_smem[]`. The
  byte count is written into the kernel source as a marker comment. Kernels at or below 48 KB are
  not changed.
- **Runtime** (`CUDACodeCache.installSource`, for both NVRTC compiles and the on-disk cubin
  cache): the marker is read, `cuFuncSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES)`
  is called, and the byte count is passed to `cuLaunchKernel`, to
  `cuOccupancyMaxPotentialBlockSize` and therefore to CUDA graph capture.
- **Over the limit:** a kernel needing more than `MAX_SHARED_MEMORY_PER_BLOCK_OPTIN` fails with a
  `TornadoMemoryException` naming the kernel, the bytes requested and the device limit.

Tests: `TestDynamicSharedMemory` (a 64 KB array; two 32 KB arrays run twice, checking they don't
overlap; a 256 KB request rejected). Without the change the first test fails with
`ptxas error : Entry function ... uses too much shared data (0x10000 bytes, 0xc000 max)`. The
existing shared-memory, MMA and cp.async tests pass, and quickPass shows no new failures on an
NVIDIA A10.

## Why it is parked

It was built to let the jitllm int8 tensor-core GEMM stage more data per barrier. Measured on two
NVIDIA A10s with Qwen3.8-27B Q8_0 prefill, every variant that used more than 48 KB was slower:

| Kernel variant | Shared memory | Prefill vs 48 KB kernel |
|---|---|---|
| 128-k rounds instead of 64-k | 72 KB | about 11% slower |
| cp.async ring, 2, 3 or 4 stages | 72 KB | about 11% slower, whatever the stage count |
| Same ring kernel | 36 KB | on par |

The cause is the unified L1/shared-memory carveout: a 72 KB block shrinks L1, and those kernels
gathered their weights with 16-bit global loads that depend on L1. The win jitllm kept came from
repacking the weights into a 16-byte-aligned tile layout and copying them with `cp.async`, which
fits in 36 KB of static shared memory.

A side finding: at 512 threads a 72 KB kernel needing more than 128 registers fails to launch with
`CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`. TornadoVM has no per-kernel register cap
(`__launch_bounds__`), and the global `-maxrregcount=128` compiler flag breaks some applications'
warm-up, so a large-shared-memory kernel must fit in 128 registers on its own.

## When to revive it

When a kernel genuinely needs more than 48 KB per block, for example a larger GEMM tile whose
operands all arrive through `cp.async` or TMA, so that the smaller L1 does not matter, or a
kernel on sm_90 and later that benefits from the larger opt-in limit. The open pull request is
kotselidis/TornadoVM#66.
