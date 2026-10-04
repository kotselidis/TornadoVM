# tornado-mlx: Apple MLX through mlx-c

This branch (`feature/mlx-c`) runs [Apple MLX](https://github.com/ml-explore/mlx) operations as
TornadoVM library tasks through MLX's C API, [mlx-c](https://github.com/ml-explore/mlx-c). Java calls
mlx-c through FFM bindings. MLX wraps each TornadoVM input buffer as an MLX array without copying,
then runs the operation on its own stream. MLX allocates the result, and the provider copies it into
the output buffer.

```java
TaskGraph graph = new TaskGraph("s0")
        .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b)
        .libraryTask("add", Mlx::add, a, b, c)
        .transferToHost(DataTransferMode.EVERY_EXECUTION, c);
```

## Two MLX branches

| Branch | MLX path | Operations |
|---|---|---|
| `feature/mlx-c` (this one) | mlx-c only: MLX allocates each result, then the provider copies it into the output | 292, including CPU-only linear algebra |
| [`feature/mlx-integration-7.1.0`](https://github.com/kotselidis/TornadoVM/tree/feature/mlx-integration-7.1.0) | In-place Metal kernels only: MLX's kernels write TornadoVM's buffers directly | 246 |

Commit `c3a21345f` is the last one with both paths. The benchmarks that compare them live in
[TornadoMLXBenchmarks](https://github.com/kotselidis/TornadoMLXBenchmarks). On an M1 Pro, the in-place
path costs a median 0.59× of this path at decode sizes and 0.50× at large sizes. The extra cost here
is the result allocation and copy-back.

## Requirements

- macOS on Apple silicon, TornadoVM built with `make BACKEND=metal`
- MLX and mlx-c: `brew install mlx mlx-c`. The provider loads `libmlxc.dylib` from the library path or
  from Homebrew's prefix.

## What is supported

All 292 in-scope operations of the MLX C API are bound; `coverage.json` lists them, and the 40
deliberately excluded ones with reasons. The factories are in `uk.ac.manchester.tornado.mlx`:

- `Mlx` (the operations LLM inference uses)
- `MlxMath`, `MlxLogic`, `MlxReduce`, `MlxShape` and `MlxCreate`
- `MlxIndex`, `MlxSort`, `MlxFft` and `MlxConv`
- `MlxLinalg`, `MlxProducts` and `MlxRandom`

Each factory carries `@MlxOp` with the mlx-c function it binds.

### Choosing the device

Operations run on MLX's GPU stream by default. Use `MlxOptions` to run a task on the CPU stream
instead:

```java
.libraryTask("add", (x, y, z) -> Mlx.add(x, y, z).withTuning(MlxOptions.cpu()), a, b, c)
```

Some operations exist only on MLX's CPU stream (LAPACK): Cholesky, triangular and general inverses,
solves, LU, QR, eigh, SVD, pseudo-inverse, eig and `random_multivariate_normal`. These always use
the CPU stream.

MLX errors come back as `TornadoRuntimeException`, so they don't end the JVM. That includes an
operation MLX doesn't implement on the chosen device, or a shape it rejects.

## Layout

| Path | Contents |
|---|---|
| `src/main/java/.../mlx/` | Factory classes, `MlxOp`, `MlxOptions` |
| `src/main/java/.../mlx/provider/` | `MlxLibraryProvider` (the SPI provider), `MlxCall` (argument marshalling), `MlxNativeLib` (loading, wrapping, errors), `MlxC` (generated FFM bindings) |
| `src/main/java/.../mlx/jit/` | `KernelContext` JIT counterparts of each operation, annotated `@JitBaseline` |
| `mlx-c-api.json` | Every mlx-c operation, written by `scripts/generate_bindings.py` |
| `scripts/generate_bindings.py` | Regenerates `MlxC.java` and `mlx-c-api.json` from mlx-c's headers |
| `coverage.json`, `coverage-overrides.json`, `scripts/update_coverage.py` | Coverage manifest. The build fails if a bound operation has no test or no tested JIT baseline |
| `../prototypes/mlx-m0/` | The M0 zero-copy spike and its findings (`FINDINGS.md`) |

## Tests

```bash
tornado-test --mlx
```

The command runs the MLX unit tests (`tornado-unittests/.../unittests/mlx`). Each test checks the MLX
task and its JIT counterpart against a Java reference. `TestMlxMemory` checks that MLX adopts the
input buffers rather than copying them, and that wrappers do not leak. After you add a factory or a
test, run `python3 tornado-mlx/scripts/update_coverage.py`.
