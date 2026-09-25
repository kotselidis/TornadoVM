# M0 findings (Apple M1 Pro, macOS 27.0, MLX 0.32.1, mlx-c 0.6.0)

## 1. Zero-copy input from a TornadoVM-style MTLBuffer — works

`zerocopy_probe.m` allocates buffers the way `MetalObjects.createBuffer` does
(`newBufferWithLength`, `MTLResourceStorageModeShared`), wraps `contents()` with
`mlx_array_new_data_managed`, skips the header with `mlx_as_strided`, and runs
`y = 2x + 1` on the GPU stream.

All 16 cases pass (n = 1, 1000, 4096, 4096²; header 16 and 24; exact and
page-rounded allocations): correct results, MLX keeps the original pointer,
a host write after wrapping is visible to a later MLX op, and the deleter runs
exactly once (after `mlx_synchronize`; MLX releases from the completion handler).

**Mechanism** (MLX `MetalAllocator::make_buffer`, v0.32.1): calls
`device->newBuffer(ptr, size, options, nullptr)`, i.e.
`newBufferWithBytesNoCopy`, and registers the new MTLBuffer in MLX's residency
set. If Metal returns nil, `array(void*, …)` silently falls back to a copy.

## 2. Corrections to the plan

- **Header is 16 bytes by default, not 24.** `TornadoNativeArray.ARRAY_HEADER`
  is 16 with compressed oops (the default; the generated `tornado-argfile` sets
  no oops flag) and 24 only with `-XX:-UseCompressedOops`. The provider must use
  `ARRAY_HEADER` at runtime: `as_strided(offset = ARRAY_HEADER / elemsize)`.
  For fp16/bf16 both values divide evenly; for 8-byte types 24/8 = 3, 16/8 = 2.
- **TornadoVM buffer bases are not always page-aligned.** Metal suballocates
  small shared buffers (a 20-byte buffer landed at page offset 6400). On macOS 27
  `newBufferWithBytesNoCopy` still accepted these pointers, so wrapping worked,
  but this is outside Metal's documented contract (page-aligned pointer and
  length) and may fail on older macOS, which would silently turn into a copy.

## 3. Consequences for the design

- The provider must detect the copy fallback: a deleter call during
  `mlx_array_new_data_managed` means MLX copied. Treat that as an error in
  tests and a logged warning (plus correct-but-slower path) in production.
- Each wrap creates a new MTLBuffer and residency-set entry; that per-op cost is
  part of the boundary cost measured in M0 step 5. Caching wrappers per
  TornadoVM buffer across executions is a likely optimisation.

## 4. From Java, on real TornadoVM buffers — works (`java/`, run `java/run.sh`)

A throwaway `apple/mlx-spike` provider (module `tornado.mlx.spike`, FFM bindings in
`MlxC.java`) runs inside a TaskGraph on the Metal backend. No runtime changes: the
interpreter hands the provider `toBuffer() + ARRAY_HEADER`, which on Metal is the
MTLBuffer handle + 16, so the spike subtracts the header and calls
`MetalAPI.bufferContents` (assumes `getBufferOffset() == 0`; M1 replaces this).

- `iota` (JIT) → `affine` (MLX) → `plusOne` (JIT) is correct for n = 1, 1000, 2^20,
  with MLX reading TornadoVM's buffers in place (no copy at wrap, same pointer).
- MLX matmul 64×96·96×80 and 257×129·129×65 match a Java reference.
- Launcher notes: the spike module must be `open` (TornadoVM reflects on task
  lambdas) and needs `--enable-native-access=tornado.mlx.spike`.

### Boundary cost, y = 2x + 1 (float32, M1 Pro, median of 200 after 20 warm-up)

Per-call split inside the provider (`wrap=direct`):

| n | wrap, fresh | wrap, cached | MLX compute+eval, fresh | cached | copy-out 1 thread | 8 threads |
|---|---|---|---|---|---|---|
| 1024 | 34 µs | 12 µs | 267 µs | 260 µs | 14 µs | — |
| 2^20 | 23 µs | 5 µs | 431 µs | 390 µs | 113 µs (37 GB/s) | 108 µs |
| 2^24 | 72 µs | 16 µs | 3066 µs | 1985 µs | 1736 µs (39 GB/s) | 1069 µs (63 GB/s) |

Whole `execute()` for the same op, three ways (µs):

| n | MLX alone | MLX on TornadoVM buffers (cached wrap, 1-thread copy) | TornadoVM JIT |
|---|---|---|---|
| 1024 | 206 | 311 | 366 |
| 2^16 | 226 | 318 | 322 |
| 2^20 | 348 | 502 | 373 |
| 2^24 | 1751 | 4042 | 1063 |

(MLX runs `2x + 1` as two kernels with an intermediate; the JIT kernel is fused.
Step 5 uses the real LLM ops instead.)

What this shows:

- **`as_strided` is free.** Wrapping from the base plus a header view costs the same
  as wrapping the data pointer directly.
- **Wrappers must be cached.** A fresh no-copy MTLBuffer per call makes the GPU map
  it every time: 64 MB costs about 1.1 ms extra. Reusing the wrapper across
  executions brings compute within 15% of MLX alone. The provider needs a wrapper
  cache keyed by buffer address and shape, invalidated when TornadoVM frees or
  reuses the buffer (buffers are pooled by `getOrAllocateBufferWithSize`).
- **Output cannot be written in place through MLX's graph API.** Every op allocates
  its output (no `out=`); primitives call `set_data(malloc)` themselves, so a shim
  cannot reliably redirect them. Copy-out costs one extra memory pass: 39 GB/s
  single-threaded, 63 GB/s on 8 threads. Only dispatching MLX's kernels ourselves
  (`apple/mlx-kernels`) removes it. For decode-sized outputs (e.g. 4096 floats)
  the copy is about 15 µs and does not matter.
- **Fixed per-call overhead dominates small ops**: about 200 µs for MLX alone and
  about 330 µs for a TornadoVM `execute()`, both mostly command-buffer commit and wait.
  Step 5 decides whether that matters at decode-op sizes.

## 5. MLX's own kernel on TornadoVM's queue — works (rms_norm)

`MlxKernels.java` loads `/opt/homebrew/opt/mlx/lib/mlx.metallib` with
`MetalAPI.newLibraryWithData`, builds a pipeline for `rmsfloat32` /
`rms_loopedfloat32`, and dispatches it on TornadoVM's own command queue (reached by
reflection for now) with TornadoVM's MTLBuffers bound at offset `ARRAY_HEADER`.
Inputs and output stay in place — no wrapping, no copy-out. Dispatch geometry
mirrors MLX's `RMSNorm::eval_gpu` (N_READS 4, looped above 4096).

All three paths match a Java reference (1e-4 relative) at 1×2048, 1×4096, 1×8192,
128×4096, 512×4096 (float32).

Median µs per `execute()` (one task per graph):

| rows × dim | MLX alone | apple/mlx (mlx-c, copy-out) | mlx-kernels (metallib, GPU time) | TornadoVM JIT |
|---|---|---|---|---|
| 1 × 2048 | 193 | 229 (copy 4) | 292 (GPU 12) | 321 |
| 1 × 4096 | 213 | 280 (copy 5) | 293 (GPU 17) | 347 |
| 1 × 8192 | 220 | 259 (copy 5) | 267 (GPU 24) | 348 |
| 128 × 4096 | 249 | 392 (copy 67) | 327 (GPU 57) | 357 |
| 512 × 4096 | 272 | 600 (copy 168) | 431 (GPU 164) | 513 |

Marginal µs per task inside one TaskGraph (chain of 16 minus chain of 1, /15):

| rows × dim | apple/mlx | mlx-kernels | TornadoVM JIT |
|---|---|---|---|
| 1 × 4096 | 210 | 226 | 295 |
| 512 × 4096 | 574 | 287 | 444 |

(The JIT rms kernel is a straightforward one-threadgroup-per-row reduction, not
jitLLM's tuned kernel; M4 uses the tuned ones.)

## 6. The dominant cost is the command-buffer round trip, not MLX vs JIT

The metallib rms_norm needs 17 µs of GPU time at 1×4096, yet one dispatch costs
~240 µs: `commit` + `waitUntilCompleted` on this M1 Pro. Every path pays it once
per op — the JIT (every Metal kernel ends in commit + wait,
`MetalObjects.java:600/690/793`), mlx-c (`eval` commits and waits), and the
metallib path.

Same MLX kernel, raw Metal, 16 ops:

| rows × dim | 16 command buffers (wait each) | 1 command buffer, 16 dispatches | speed-up |
|---|---|---|---|
| 1 × 4096 | 226 µs/op | 24 µs/op | 9.3× |
| 128 × 4096 | 252 µs/op | 40 µs/op | 6.3× |
| 512 × 4096 | 285 µs/op | 72 µs/op | 4.0× |

At decode sizes ~90% of each op is synchronisation. A Llama-3.2-1B decode step
runs on the order of 150 GPU ops, so ~225 µs each caps decode near 30 tok/s from
synchronisation alone, whichever library runs the ops. The plan's M5 target
("tok/s ≥ Metal JIT") is reachable, but the "within 15% of mlx-lm" stretch goal
is not while the backend waits after every op. The plan lists command-buffer
batching as out of scope; this result says it is the largest lever for M4/M5.

## 7. Decode-shape ops (`MAIN=DecodeOps java/run.sh -Dmlx.spike.wrap=direct -Dmlx.spike.cache=true`)

fp16, M1 Pro, µs. "Marginal" = cost of one more task in a 16-task chain inside one
TaskGraph. The bound is bytes moved at 200 GB/s. All results match a Java reference
(max abs error ≤ 2.4e-3). JIT kernels are straightforward one-threadgroup-per-row
versions, not jitLLM's tuned kernels.

| op | bound | MLX alone | apple/mlx per execute() (wrap / MLX / copy) | JIT per execute() | apple/mlx marginal | JIT marginal |
|---|---|---|---|---|---|---|
| GEMV fp16 4096×4096 | 168 | 417 | 570 (17 / 430 / 17) | 1013 | 562 | 515 |
| Q4 qmv, affine g32, 4096×4096 | 52 | 265 | 421 (5 / 357 / 4) | 679 | 369 | 357 |
| rope half-split, 32×128 | ~0 | 212 | 239 (1 / 197 / 5) | 279 | 199 | 239 |

- The provider's own cost (cached wrap + copy-out) is 10–35 µs, 2–8% of the op.
- Q4: weights quantized by `mlx_quantize` and copied into TornadoVM `IntArray` /
  `HalfFloatArray`; MLX's packing (8 values per uint32, low nibble first,
  `w = scale·q + bias` per group) is confirmed by an independent Java reference.
- **Metal backend bug found**: `MetalArithmeticTool.emitUShr` lowers `>>>` to MSL's
  arithmetic `>>`. With `(word >>> 28) & 0xF` Graal drops the mask, so the Q4 JIT
  kernel was wrong until it used `>>`. Flagged as a separate fix, outside this plan.

## 8. prebuiltTask route — works

`java/kernels/mlx_rms_f32.metal` is MLX's `rms_single_row` body with only the entry
point adapted to TornadoVM's prebuilt ABI (four implicit buffers, header-relative
pointers, trailing `_global_sizes`; eps and axis_size from a params `FloatArray`,
since scalars cannot be prebuilt arguments). Correct at 1×2048, 1×4096, 128×4096,
512×4096; per `execute()` 283 / 290 / 319 / 421 µs, the same as the raw-metallib path.

## 9. No-copy evidence (stand-in for an Instruments trace)

`xctrace` needs the Xcode license accepted (`sudo xcodebuild -license`), which is
the user's call. Instead, `make trace` interposes `memcpy`/`memmove` into
`zerocopy_probe` and logs every copy ≥ 64 KiB: **none** across all 16 cases up to
64 MB. Positive control: with `MEMCPY_TRACE_MIN=16` the same run logs 69,623 small
copies from inside MLX, so the interposer does see MLX's copies. (The JDK is a
hardened binary, so the interposer cannot be injected into the Java spike; that
path relies on the pointer-identity, deleter and host-write-visibility checks.)

## M0 exit criteria

- [x] **Correct results** — every path and shape above matches a Java reference.
- [x] **Zero-copy input shown** — no buffer-sized memcpy (section 9), same pointer,
      deleter not called at wrap, host writes visible to MLX (sections 1, 4).
- [x] **Boundary and copy costs written down** — sections 4, 5, 7.
- [x] **Decisions recorded:**
  - **Shim: no.** mlx-c's `mlx_array_new_data_managed` adopts TornadoVM memory
    without a copy (step 2.1 succeeded). A shim could not make MLX write outputs in
    place either, because MLX primitives allocate their own outputs.
  - **apple/mlx-kernels: no, by the plan's criterion.** The apple/mlx provider adds
    2–8% to a decode-step op (GEMV 8%, rms_norm ~5%, Q4 3%, rope 2%), under the 10%
    threshold. Caveat for M4: for light ops at prefill shapes the copy-out is large
    (rms_norm 512×4096: mlx-c 574 µs vs metallib 287 µs marginal), and both
    mlx-kernels mechanisms are proven (raw metallib and prebuiltTask), so re-check
    with M4's prefill shapes before M5 Stage A.

## Corrections and inputs for M1–M2

1. Header offset is `ARRAY_HEADER` at runtime (16 by default), not a fixed 24.
2. Small TornadoVM buffers are not page-aligned; no-copy wrapping still works on
   macOS 27, but the provider must detect the silent copy fallback (deleter called
   during wrap) rather than assume zero-copy.
3. Cache MLX wrappers per buffer (≈1 ms per 64 MB saved per call) and invalidate
   on buffer release/reuse — a stale wrapper returned wrong data in the spike.
4. Per-op commit + wait (~225 µs on this machine) dominates decode-size ops for
   every path; command-buffer batching stays out of scope (user decision), so M5's
   "within 15% of mlx-lm" stretch goal is not expected to be met.
5. The interpreter's `toBuffer() + ARRAY_HEADER` pointer on Metal is the MTLBuffer
   handle + 16; the spike reverses it, M1 replaces it with `libraryAddress()` /
   `getNativeBuffer()` + `getNativeOffset()`.
6. `getCommandQueue` is private in `MetalDeviceContext`; the spike reaches it by
   reflection, M1 exposes it through `TornadoNativeStreamSupport`.
7. Spike module must be `open` and needs `--enable-native-access`; JDK 21 FFM is a
   preview API (`allocateArray`, not `allocateFrom`; beware its `(layout, long count)`
   overload, which bit the spike twice).

## How to reproduce

```
source setvars.sh                              # TornadoVM, JDK 21, Metal backend
make -C prototypes/mlx-m0 run trace            # C probe + memcpy trace
prototypes/mlx-m0/java/run.sh -Dmlx.spike.wrap=direct -Dmlx.spike.cache=true -Dmlx.spike.copyThreads=8
MAIN=DecodeOps prototypes/mlx-m0/java/run.sh -Dmlx.spike.wrap=direct -Dmlx.spike.cache=true
```
