# TornadoVM Hybrid API — Apple MLX Library Tasks (mlx-c)

This module lets a TornadoVM `TaskGraph` mix JIT-compiled Java tasks with
[Apple MLX](https://github.com/ml-explore/mlx) operations on the Metal backend, called through MLX's C
API, [mlx-c](https://github.com/ml-explore/mlx-c). MLX wraps each TornadoVM input buffer as an MLX array
without copying, runs the operation on its own GPU or CPU stream, and the provider copies MLX's result
into the output buffer. Every in-scope MLX operation is bound, including the ones MLX runs only on its
CPU stream.

```java
TaskGraph taskGraph = new TaskGraph("mlx")
    .transferToDevice(DataTransferMode.EVERY_EXECUTION, a, rhs)
    .task("assemble", MyClass::assemble, a)                                  // JIT-compiled kernel
    .libraryTask("chol", MlxLinearAlgebra::cholesky, a, l, batch, n, false)  // MLX, CPU stream (LAPACK)
    .libraryTask("solve", MlxLinearAlgebra::solveTriangular, l, rhs, x, batch, n, nrhs, false)
    .task("residual", MyClass::residual, a, x, rhs)                          // JIT-compiled kernel
    .transferToHost(DataTransferMode.EVERY_EXECUTION, x);

try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
    plan.execute();
}
```

## Requirements

- macOS on Apple silicon
- MLX and mlx-c: `brew install mlx mlx-c`
- JDK 21

## Build

From the repository root:

```bash
make BACKEND=metal
source setvars.sh
```

This builds the Metal backend and the `tornado-mlx` Java module, which binds straight to
`libmlxc.dylib` through `java.lang.foreign`. There is no native module to build. The provider loads
`libmlxc.dylib` from the library path or from Homebrew's prefix, is registered as a
`TornadoLibraryProvider` (`apple/mlx`), and is found through `ServiceLoader`. The FFM bindings in
`MlxC.java` are generated from mlx-c's headers by `scripts/generate_bindings.py`.

## Supported operations

All factories are static methods used as the second argument of
`taskGraph.libraryTask(id, factory, args...)`, one class per category. Each task's function name is
the mlx-c name of the operation it runs (`mlx_add`, `mlx_sum_axis`, ...).
Operations are grouped by category, as in the coverage manifest of
[TornadoMLXBenchmarks](https://github.com/kotselidis/TornadoMLXBenchmarks), which also lists the 40 deliberately excluded mlx-c
functions (autodiff transforms, custom-kernel builders, PRNG key management) with reasons.

| Category | Factory class | Operations |
|---|---|---|
| Arithmetic (51) | `MlxArithmetic` | add, subtract, multiply, divide, maximum, minimum, negative, square, sqrt, rsqrt, exp, expm1, log, log1p, log2, log10, logaddexp, power, remainder, floor_divide, divmod, reciprocal, abs, sign, ceil, floor, round, clip, where, sin, cos, tan, arcsin, arccos, arctan, arctan2, sinh, cosh, tanh, arcsinh, arccosh, arctanh, degrees, radians, erf, erfinv, sigmoid, nan_to_num, real, imag, conjugate |
| Logic (23) | `MlxLogic` | equal, not_equal, greater, greater_equal, less, less_equal, isfinite, isinf, isnan, isneginf, isposinf, bitwise_and/or/xor/invert, left_shift, right_shift, logical_and/or/not, isclose, allclose, array_equal |
| Reductions (38) | `MlxReductions` | sum, prod, max, min, mean, var, std, logsumexp, all, any (whole, `_axis`, `_axes`), median, argmin, argmax, softmax |
| Scans (5) | `MlxScans` | cumsum, cumprod, cummax, cummin, logcumsumexp |
| Sorting (10) | `MlxSorting` | sort, argsort, partition, argpartition (whole and `_axis`), topk |
| Indexing (26) | `MlxIndexing` | take, take_axis, take_along_axis, put_along_axis, gather, scatter, scatter_add/max/min/prod (points, rows, axis), masked_scatter, slice, slice_dynamic, slice_update (set, add, max, min, prod, dynamic) |
| LinearAlgebra (31) | `MlxLinearAlgebra` | matmul, addmm, einsum, tensordot, inner, outer, kron, block_masked_mm, segmented_mm, gather_mm, hadamard_transform, cross, norms, cholesky, cholesky_inv, tri_inv, inv, solve, solve_triangular, lu, lu_factor, qr, eigh, eigvalsh, svd, pinv, eig, eigvals |
| Quantization (7) | `MlxQuantization` | quantize, dequantize (affine and mxfp8), quantized_matmul, gather_qmm, qqmm, to_fp8, from_fp8 |
| NeuralNetwork (5) | `MlxNeuralNetwork` | fast_rms_norm, fast_layer_norm, fast_rope, fast_rope_dynamic, fast_scaled_dot_product_attention |
| Fft (16) | `MlxFft` | fft, ifft, rfft, irfft (1D, 2D, nD), fftshift, ifftshift, fftfreq, rfftfreq |
| Convolution (7) | `MlxConvolution` | conv1d, conv2d, conv3d, conv_general, conv_transpose1d, conv_transpose2d, conv_transpose3d |
| Creation (21) | `MlxCreation` | arange, linspace, eye, identity, tri, tril, triu, diag, diagonal, trace, full, zeros, ones (and `_like`), bartlett, blackman, hamming, hanning, meshgrid |
| Shape (37) | `MlxShape` | reshape, flatten, unflatten, squeeze, expand_dims, atleast_1d/2d/3d, transpose, swapaxes, moveaxis, broadcast_to, broadcast_arrays, as_strided, contiguous, copy, astype, view, number_of_elements, concatenate, stack, split, repeat, tile, roll, pad |
| Random (15) | `MlxRandom` | bits, uniform, normal, randint, bernoulli, truncated_normal, gumbel, laplace, categorical, multivariate_normal, permutation |

### Conventions

- **Layout.** MLX and TornadoVM are both row-major, so arrays need no transposes and dimensions are
  passed in their natural order (`matmul(a, b, c, m, k, n)` for `c[m, n] = a[m, k] @ b[k, n]`).
- **Types.** Most operations come in `FloatArray`, `HalfFloatArray` and `BFloat16Array` forms; integer,
  logic and index operations also take `IntArray`. Boolean results are written as 0 or 1 into a
  `ByteArray`. Complex values are interleaved `(re, im)` pairs in a `FloatArray`. The linear-algebra
  decompositions are float32 only.
- **Axes.** Reductions and scans view their input as `[outer, len, inner]` (one axis) or
  `[outer, len1, len2, inner]` (two adjacent axes) and reduce the middle; any `inner` is accepted.
- **Batches.** Matrices for the decompositions are batched, `[batch, n, n]`.

## Usage

### Mixing with TornadoVM tasks

MLX library tasks are scheduled like any other task. Dependencies come from the standard
`Access[]`-driven data-flow graph: each factory marks its output arguments `WRITE_ONLY` and the rest
`READ_ONLY`. The Metal backend batches JIT kernels into shared command buffers and drains the batch
before a library task takes a buffer address, so MLX always reads the writes of earlier JIT kernels.
MLX evaluates the operation and waits for it before the provider copies the result into the output
buffer, so the next JIT task sees it.

Wrapping a buffer as an MLX array maps it into an MLX-owned `MTLBuffer`, which is costly for large
arrays, so wrappers are cached per execution plan and freed when the plan closes.

### Per-call options: `MlxOptions`

Operations run on MLX's GPU stream by default. Attach `MlxOptions` with
`LibraryTaskDescriptor.withTuning(...)` to run one task on the CPU stream instead:

```java
.libraryTask("add", (FloatArray x, FloatArray y, FloatArray z) -> MlxArithmetic.add(x, y, z).withTuning(MlxOptions.cpu()), a, b, c)
```

The operations MLX implements only on its CPU stream always use it: Cholesky, triangular and general
inverses, solves, LU, QR, eigh, SVD, pseudo-inverse, eig and `random_multivariate_normal`.

### Errors and debugging

mlx-c's default error handler prints the message and exits the process. The provider installs its own
handler, so an MLX error (an operation MLX does not implement on the chosen device, or a shape it
rejects) surfaces as a `TornadoRuntimeException` with MLX's message instead.

| Flag | Effect |
|---|---|
| `--devices` | Check that the Metal device is visible |
| `--jvm "-Dtornado.unittests.device=0:0"` | Select backend:device for the unit tests |

If `libmlxc.dylib` cannot be loaded, the provider declines the device and an MLX task fails with
``Library `apple/mlx` is not supported on device``; install mlx-c (`brew install mlx-c`).

## Tests

The JUnit suite lives in `tornado-unittests` (`uk.ac.manchester.tornado.unittests.mlx`), one class
per category (`TestMlxArithmetic`, `TestMlxReductions`, `TestMlxLinearAlgebra`, ...) plus `TestMlx`,
which covers how MLX tasks behave in a task graph: mixed with JIT tasks, buffers shared across task
graphs, inputs changing between executions, input adoption, the CPU stream, and MLX errors raised as
exceptions. Each test runs one factory and checks it against a sequential Java reference.
`TestMlxMemory` checks that repeated executions and many short-lived plans do not grow MLX's live
memory. The tests report `UNSUPPORTED` when the default device is not Metal or mlx-c is missing.

```bash
tornado-test --mlx                                                      # all MLX tests
tornado-test -V uk.ac.manchester.tornado.unittests.mlx.TestMlxLinearAlgebra
```

Every bound operation has a test. [TornadoMLXBenchmarks](https://github.com/kotselidis/TornadoMLXBenchmarks) keeps the
coverage manifest that checks it.

## Benchmarks

The benchmarks live in [TornadoMLXBenchmarks](https://github.com/kotselidis/TornadoMLXBenchmarks); its
README shows how to build and run them. Reference numbers: Apple M1 Pro (16-core GPU, 200 GB/s),
MLX 0.32.1, mlx-c 0.6.0, measured at commit `c26ee040b`. Times are marginal: the cost of one more task
in a chain of 8.

**JIT vs mlx-c** (JIT time / mlx-c time, above 1 means MLX is faster):

| Regime | Cases | Median | 25th–75th percentile |
|---|---:|---:|---:|
| Decode-sized | 172 | 0.63× | 0.59–0.71× |
| Large | 171 | 0.41× | 0.33–0.54× |
| Prefill | 24 | 1.68× | 0.78–2.33× |

The copy-back makes simple memory-bound operations slower through mlx-c than as JIT kernels. MLX
leads on tuned compute-bound kernels.

Selected operations (µs):

| Operation | Shape | Device | mlx-c | JIT | Java |
|---|---|---|---:|---:|---:|
| add (f32) | 16,777,216 | GPU | 2,633 | 1,387 | 13,370 |
| rmsNorm (f32) | 512×4096 | GPU | 552 | 402 | 3,749 |
| matmul (f32) | 512×4096×4096 | GPU | 4,830 | 11,248 | 32,715,721 |
| quantizedMatmul q4 | 16×4096×4096 | GPU | 619 | 3,148 | – |
| qqmm mxfp8 | 16×4096×4096 | GPU | 1,149 | 16,475 | – |
| conv3d (f32) | 1×32×32×32×16, k3 | GPU | 696 | 7,604 | – |
| maskedScatter | 16,777,216, 30% masked | GPU | 11,051 | 124,673 | – |
| cholesky | 256 × 32×32 | CPU | 554 | 452 | – |
| svd | 256 × 32×32 | CPU | 21,602 | 9,438 | – |

These numbers predate Metal command-buffer batching (`d786a9518`), which lowers per-task cost, so a
fresh run gives lower absolute times, mostly for decode-sized cases.

## Layout

| Path | Contents |
|---|---|
| `src/main/java/.../mlx/` | `Mlx` (library name and task helpers), one factory class per category, `MlxOptions` |
| `src/main/java/.../mlx/provider/` | `MlxLibraryProvider` (the SPI provider), `MlxNativeLib` (loading, wrapping, errors, and `MlxCall`, the argument marshalling), `MlxC` (generated FFM bindings) |
| `scripts/generate_bindings.py` | Regenerates `MlxC.java` from mlx-c's headers |
| `../prototypes/mlx-m0/` | The zero-copy spike and its findings (`FINDINGS.md`) |

## Adding an operation

1. If mlx-c added the function, rerun `scripts/generate_bindings.py` to regenerate `MlxC.java`.
2. Add a factory to the category's class that calls `Mlx.task("mlx_<op>", outputIndex, args...)`, with
   the mlx-c name of the operation.
3. Add an entry for `mlx_<op>` in `MlxLibraryProvider`'s operation table: it wraps the inputs with
   `MlxCall`, calls the `MlxC` function and stores the result.
4. Add a test to the category's class in `tornado-unittests/.../unittests/mlx`.

For a new native library, see `HYBRID_API_GUIDE.md`; `MlxLibraryProvider` is a provider bound
entirely through `java.lang.foreign`.
