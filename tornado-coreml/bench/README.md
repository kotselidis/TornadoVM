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
| Mlp_64x256x2 | CPU_ONLY | CPU | 29.8 | 115.4 | +85.6 | 72.2 | 0.15 | 551/551 |
| Mlp_64x256x2 | CPU_AND_GPU | GPU | 172.9 | 246.4 | +73.5 | 242.3 | 0.07 | 551/551 |
| Mlp_64x256x2 | CPU_AND_NEURAL_ENGINE | Neural Engine | 106.2 | 140.8 | +34.6 | 137.4 | 0.12 | 551/551 |
| Mlp_64x256x2 | ALL | Neural Engine | 99.2 | 128.5 | +29.3 | 131.5 | 0.13 | 551/551 |
| Mlp_256x1024x4 | CPU_ONLY | CPU | 468.6 | 513.1 | +44.5 | 509.8 | 4.19 | 551/551 |
| Mlp_256x1024x4 | CPU_AND_GPU | GPU | 743.8 | 922.5 | +178.7 | 834.2 | 2.33 | 551/551 |
| Mlp_256x1024x4 | CPU_AND_NEURAL_ENGINE | Neural Engine | 252.9 | 273.4 | +20.5 | 280.7 | 7.85 | 551/551 |
| Mlp_256x1024x4 | ALL | Neural Engine | 251.5 | 276.9 | +25.4 | 282.6 | 7.76 | 551/551 |
| Mlp_512x2048x8 | CPU_ONLY | CPU | 5603.5 | 5880.8 | +277.3 | 5925.9 | 5.84 | 551/551 |
| Mlp_512x2048x8 | CPU_AND_GPU | GPU | 6502.2 | 6681.9 | +179.7 | 6727.2 | 5.14 | 551/551 |
| Mlp_512x2048x8 | CPU_AND_NEURAL_ENGINE | Neural Engine | 2832.5 | 2841.0 | +8.5 | 2928.3 | 12.09 | 551/551 |
| Mlp_512x2048x8 | ALL | Neural Engine | 2809.8 | 2903.0 | +93.2 | 2913.5 | 11.84 | 551/551 |

- On the Neural Engine the TornadoVM task costs 9–35 µs more than native Core ML: 0.3–4% for the two
  larger models.
- The Neural Engine is 2.0x faster than the CPU and 2.3x faster than Core ML's GPU path on the
  largest model (12 TFLOP/s). Core ML's GPU path is slower than its CPU path for these models.
- Every output, on every compute unit, was written in place.
- With `EVERY_EXECUTION` transfers the plan adds about 90 µs on the largest model (2 MB in, 2 MB out).
- The smallest model is noisy: its `tornado task` rows ran first after loading, so they are slower
  than the `tornado e2e` rows that follow.
