# Core ML benchmark

Times Core ML predictions run as TornadoVM library tasks against a native Swift loop that predicts
the same model, with the same compute units, on preallocated buffers with an output backing. The
difference is what the TornadoVM task adds per prediction.

The models are float16 perceptrons `Mlp_<rows>x<dim>x<layers>`: input and output `[rows, dim]`,
`layers` linear layers of `[dim, dim]` with ReLU between them.

## Run

From a shell where TornadoVM (Metal backend) is built and `setvars.sh` is sourced, with coremltools
installed (`COREMLTOOLS_PYTHON`, default `~/.venvs/coremltools/bin/python`):

```bash
tornado-coreml/bench/run.sh ~/coreml-bench
```

`WARMUP` (50), `ITERATIONS` (500), `UNITS` and `SHAPES` (`"64,256,2 256,1024,4 512,2048,8"`)
override the defaults. The script writes `native.csv`, `tornado.csv` and `summary.md`.

| Column | Meaning |
|---|---|
| native us | median latency of the Swift loop |
| tornado task us | median `execute()` of a plan with only the Core ML task, data already on the device |
| overhead us | tornado task minus native |
| tornado e2e us | the same plan copying `x` in and `y` out on every execution |
| in place | outputs Core ML wrote straight into the TornadoVM buffer / predictions |

## Results: Apple M4 Pro, macOS 27.0.1, JDK 21

50 warm-up and 500 timed iterations, one run, medians in microseconds.

| model | units | placement | native | tornado task | overhead | tornado e2e | TFLOP/s | in place |
|---|---|---|---:|---:|---:|---:|---:|---|
| Mlp_64x256x2 | CPU_ONLY | CPU | 31.6 | 78.2 | +46.6 | 49.6 | 0.21 | 551/551 |
| Mlp_64x256x2 | CPU_AND_GPU | GPU | 184.9 | 186.0 | +1.1 | 192.9 | 0.09 | 551/551 |
| Mlp_64x256x2 | CPU_AND_NEURAL_ENGINE | Neural Engine | 106.2 | 105.6 | -0.6 | 101.5 | 0.16 | 551/551 |
| Mlp_64x256x2 | ALL | Neural Engine | 99.7 | 99.2 | -0.5 | 100.1 | 0.17 | 551/551 |
| Mlp_256x1024x4 | CPU_ONLY | CPU | 465.6 | 458.8 | -6.8 | 470.5 | 4.68 | 551/551 |
| Mlp_256x1024x4 | CPU_AND_GPU | GPU | 731.8 | 953.0 | +221.2 | 831.5 | 2.25 | 551/551 |
| Mlp_256x1024x4 | CPU_AND_NEURAL_ENGINE | Neural Engine | 238.8 | 241.3 | +2.5 | 255.2 | 8.90 | 551/551 |
| Mlp_256x1024x4 | ALL | Neural Engine | 234.8 | 238.5 | +3.7 | 254.3 | 9.00 | 551/551 |
| Mlp_512x2048x8 | CPU_ONLY | CPU | 5655.2 | 5739.5 | +84.3 | 5770.9 | 5.99 | 551/551 |
| Mlp_512x2048x8 | CPU_AND_GPU | GPU | 6461.5 | 6498.5 | +37.0 | 6532.5 | 5.29 | 551/551 |
| Mlp_512x2048x8 | CPU_AND_NEURAL_ENGINE | Neural Engine | 2787.2 | 2860.0 | +72.8 | 2885.6 | 12.01 | 551/551 |
| Mlp_512x2048x8 | ALL | Neural Engine | 2791.9 | 2866.5 | +74.6 | 2888.4 | 11.99 | 551/551 |

- On the two smaller models the TornadoVM task is as fast as native Core ML on the Neural Engine
  (-0.6 to +3.7 us). On the largest model the difference (+73 us, 2.5%) changes from run to run
  (+8.5 us in an earlier run) and is not a per-call cost: timing each step of the task puts the
  provider and the TornadoVM interpreter at under 3 us per call, plus about 10 us for the rest of
  `execute()`.
- The Neural Engine is 2.0x faster than the CPU and 2.3x faster than Core ML's GPU path on the
  largest model (12 TFLOP/s). Core ML's GPU path is slower than its CPU path for these models.
- Every output, on every compute unit, was written in place.
- With `EVERY_EXECUTION` transfers the plan adds about 25 us on the largest model (2 MB in, 2 MB out).
- The first configuration of each model (`CPU_ONLY`, `tornado task`) runs right after loading and
  is slower than the `tornado e2e` row that follows; the GPU rows vary by up to 30% between runs.
