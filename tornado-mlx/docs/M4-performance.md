# M4: MLX vs TornadoVM JIT on Metal (Tier 1 operations)

Apple M1 Pro (16-core GPU, 200 GB/s), MLX 0.32.1, mlx-c 0.6.0, TornadoVM 7.0.2-dev (Metal), JDK 21.
Harness: `src/main/java/uk/ac/manchester/tornado/mlx/benchmarks/MlxBenchmarks.java`
(`tornado -m tornado.mlx/uk.ac.manchester.tornado.mlx.benchmarks.MlxBenchmarks [--params=--quick]`).

## Summary

- **Every decode-sized op costs at least ~207 µs**, whoever runs it: each op ends in a Metal
  command-buffer commit and wait. Anything moving less than ~10 MB sits on that floor.
- **The `apple/mlx` library task adds +2.8% (median) over MLX alone at decode sizes**, inside the 10%
  budget, so the M0 decision stands for decode: no `apple/mlx-kernels` layer.
- **Light prefill ops break the budget**: RMS norm at 512×4096 costs +155% and RoPE at 32×512×128
  +126%, because MLX allocates its result and the provider copies 8 MB back. Revisit
  `apple/mlx-kernels` (MLX kernels writing TornadoVM's buffers in place) before M5 Stage A. Heavy
  prefill ops (GEMM) stay under 10%.
- **Large memory-bound ops run near bandwidth**: fp16 GEMV at 151936×2048 reaches 173 GB/s (86% of
  peak) through `apple/mlx`; jitLLM's tuned JIT kernel reaches 160 GB/s (1.07× slower).
- **The JIT is more than 1.5× slower than `apple/mlx` on 17 of 23 cases.** Largest gaps: decode
  attention (up to 21× at 8k context), prefill GEMM (0.4 vs 4.0 TFLOP/s, 8–10×) and Q4 GEMV (2–3×).
  The GEMM, attention, RoPE and softmax JIT kernels are straightforward kernels written for this
  benchmark (jitLLM's are fused with its paged KV cache or use CUDA tensor cores); GEMV, Q4/Q8 GEMV,
  RMS norm and argmax use jitLLM's tuned kernels.
- **jitLLM's RMS norm is two kernels**, so it pays the floor twice (2.2–2.7× the single MLX kernel).
  With command-buffer batching out of scope, fewer kernels per layer is the JIT path's main lever.

## Results

Times are the marginal cost of one more task inside a TaskGraph (chain of 8 minus a single task,
divided by 7), which is what an op costs inside an LLM layer; for MLX alone it is the time per call.
Median of up to 100 runs after 10 warm-up runs.

| op | shape | MLX alone µs | apple/mlx µs | JIT µs | apple/mlx overhead | JIT ÷ apple/mlx | apple/mlx GB/s | JIT GB/s |
|---|---|---|---|---|---|---|---|---|
| GEMV fp16 | 2048x2048 | 285.9 | 295.7 | 488.4 | +3% | 1.65× | 28.4 | 17.2 |
| GEMV fp16 | 3072x3072 | 293.6 | 395.8 | 629.3 | +35% | 1.59× | 47.7 | 30.0 |
| GEMV fp16 | 4096x4096 | 430.3 | 536.8 | 934.7 | +25% | 1.74× | 62.5 | 35.9 |
| GEMV fp16 | 128256x2048 | 3066.7 | 3069.0 | 3292.7 | +0% | 1.07× | 171.3 | 159.6 |
| GEMV fp16 | 151936x2048 | 3676.6 | 3604.0 | 3900.3 | -2% | 1.08× | 172.8 | 159.6 |
| GEMV q4 | 2048x2048 | 264.0 | 276.3 | 563.7 | +5% | 2.04× | 9.5 | 4.2 |
| GEMV q8 | 2048x2048 | 259.0 | 279.7 | 366.4 | +8% | 1.31× | 16.9 | 12.2 |
| GEMV q4 | 3072x3072 | 285.5 | 272.0 | 836.1 | -5% | 3.07× | 21.7 | 6.4 |
| GEMV q8 | 3072x3072 | 270.6 | 279.5 | 466.5 | +3% | 1.67× | 38.0 | 21.5 |
| GEMV q4 | 4096x4096 | 337.9 | 385.7 | 589.7 | +14% | 1.53× | 27.2 | 16.1 |
| GEMV q8 | 4096x4096 | 309.4 | 314.1 | 716.4 | +2% | 2.28× | 60.1 | 24.9 |
| GEMV q4 | 128256x2048 | 1105.5 | 1571.0 | 4882.4 | +42% | 3.11× | 104.7 | 30.4 |
| GEMM fp16 | 512x2048x2048 | 1179.4 | 1275.2 | 10364.4 | +8% | 8.13× | 9.9 | 1.2 |
| GEMM fp16 | 512x4096x4096 | 4186.2 | 4335.7 | 42327.9 | +4% | 9.76× | 9.7 | 1.0 |
| RMS norm | 1x2048 | 227.4 | 221.1 | 504.6 | -3% | 2.28× | 0.1 | 0.0 |
| RMS norm | 1x3072 | 224.2 | 233.3 | 513.7 | +4% | 2.20× | 0.2 | 0.1 |
| RMS norm | 1x4096 | 206.9 | 195.8 | 532.4 | -5% | 2.72× | 0.3 | 0.1 |
| RMS norm | 512x4096 | 277.9 | 708.3 | – | +155% | – | 23.7 | – |
| RoPE | 32x1x128 | 213.7 | 217.8 | 266.6 | +2% | 1.22× | 0.2 | 0.1 |
| RoPE | 32x512x128 | 283.8 | 641.8 | – | +126% | – | 26.1 | – |
| SDPA decode | ctx 512 | 285.5 | 323.1 | 1258.1 | +13% | 3.89× | 13.1 | 3.4 |
| SDPA decode | ctx 2048 | 361.2 | 312.0 | 3831.0 | -14% | 12.28× | 53.9 | 4.4 |
| SDPA decode | ctx 8192 | 683.3 | 697.6 | 14938.3 | +2% | 21.41× | 96.2 | 4.5 |
| softmax | vocab 151936 | 274.0 | 317.0 | 429.2 | +16% | 1.35× | 3.8 | 2.8 |
| argmax | vocab 151936 | 289.8 | 292.3 | 326.2 | +1% | 1.12× | 2.1 | 1.9 |
| top-k (k=50) | vocab 151936 | 525.3 | 537.1 | – | +2% | – | 1.1 | – |

## JIT to-do list (JIT more than 1.5× slower than apple/mlx, worst first)

| op | shape | JIT marginal us | apple/mlx marginal us | JIT / MLX |
|---|---|---|---|---|
| SDPA decode | ctx 8192 | 14938.3 | 697.6 | 21.41x |
| SDPA decode | ctx 2048 | 3831.0 | 312.0 | 12.28x |
| GEMM fp16 | 512x4096x4096 | 42327.9 | 4335.7 | 9.76x |
| GEMM fp16 | 512x2048x2048 | 10364.4 | 1275.2 | 8.13x |
| SDPA decode | ctx 512 | 1258.1 | 323.1 | 3.89x |
| GEMV q4 | 128256x2048 | 4882.4 | 1571.0 | 3.11x |
| GEMV q4 | 3072x3072 | 836.1 | 272.0 | 3.07x |
| RMS norm | 1x4096 | 532.4 | 195.8 | 2.72x |
| RMS norm | 1x2048 | 504.6 | 221.1 | 2.28x |
| GEMV q8 | 4096x4096 | 716.4 | 314.1 | 2.28x |
| RMS norm | 1x3072 | 513.7 | 233.3 | 2.20x |
| GEMV q4 | 2048x2048 | 563.7 | 276.3 | 2.04x |
| GEMV fp16 | 4096x4096 | 934.7 | 536.8 | 1.74x |
| GEMV q8 | 3072x3072 | 466.5 | 279.5 | 1.67x |
| GEMV fp16 | 2048x2048 | 488.4 | 295.7 | 1.65x |
| GEMV fp16 | 3072x3072 | 629.3 | 395.8 | 1.59x |
| GEMV q4 | 4096x4096 | 589.7 | 385.7 | 1.53x |

## Full measurements

Single-task `execute()` medians with p10–p90, marginal times, bandwidth and throughput:

| op | shape | dtype | variant | execute() median us (p10-p90) | marginal us | GB/s (% of 200) | GFLOP/s | note |
|---|---|---|---|---|---|---|---|---|
| GEMV fp16 | 2048x2048 | f16 | MLX alone | 285.9 (call) | 285.9 | 29.4 (14.7%) | 29.3 |  |
| GEMV fp16 | 2048x2048 | f16 | apple/mlx | 555.6 (466.2-657.5) | 295.7 | 28.4 (14.2%) | 28.4 |  |
| GEMV fp16 | 2048x2048 | f16 | JIT | 605.8 (471.6-757.2) | 488.4 | 17.2 (8.6%) | 17.2 | jitLLM matrixVectorGenericSimd32 (fp32 output) |
| GEMV fp16 | 3072x3072 | f16 | MLX alone | 293.6 (call) | 293.6 | 64.3 (32.2%) | 64.3 |  |
| GEMV fp16 | 3072x3072 | f16 | apple/mlx | 364.8 (328.6-451.3) | 395.8 | 47.7 (23.9%) | 47.7 |  |
| GEMV fp16 | 3072x3072 | f16 | JIT | 588.2 (502.3-684.3) | 629.3 | 30.0 (15.0%) | 30.0 | jitLLM matrixVectorGenericSimd32 (fp32 output) |
| GEMV fp16 | 4096x4096 | f16 | MLX alone | 430.3 (call) | 430.3 | 78.0 (39.0%) | 78.0 |  |
| GEMV fp16 | 4096x4096 | f16 | apple/mlx | 488.0 (445.9-561.4) | 536.8 | 62.5 (31.3%) | 62.5 |  |
| GEMV fp16 | 4096x4096 | f16 | JIT | 979.8 (923.8-1192.8) | 934.7 | 35.9 (18.0%) | 35.9 | jitLLM matrixVectorGenericSimd32 (fp32 output) |
| GEMV fp16 | 128256x2048 | f16 | MLX alone | 3066.7 (call) | 3066.7 | 171.4 (85.7%) | 171.3 |  |
| GEMV fp16 | 128256x2048 | f16 | apple/mlx | 3144.2 (3108.5-3213.5) | 3069.0 | 171.3 (85.6%) | 171.2 |  |
| GEMV fp16 | 128256x2048 | f16 | JIT | 3317.0 (3265.5-3411.2) | 3292.7 | 159.6 (79.8%) | 159.5 | jitLLM matrixVectorGenericSimd32 (fp32 output) |
| GEMV fp16 | 151936x2048 | f16 | MLX alone | 3676.6 (call) | 3676.6 | 169.4 (84.7%) | 169.3 |  |
| GEMV fp16 | 151936x2048 | f16 | apple/mlx | 3624.2 (3565.0-3837.3) | 3604.0 | 172.8 (86.4%) | 172.7 |  |
| GEMV fp16 | 151936x2048 | f16 | JIT | 3909.7 (3863.3-4027.0) | 3900.3 | 159.6 (79.8%) | 159.6 | jitLLM matrixVectorGenericSimd32 (fp32 output) |
| GEMV q4 | 2048x2048 | q4 | MLX alone | 264.0 (call) | 264.0 | 10.0 (5.0%) | 31.8 |  |
| GEMV q4 | 2048x2048 | q4 | apple/mlx | 323.8 (294.5-378.5) | 276.3 | 9.5 (4.8%) | 30.4 | MLX affine, group 32, fp16 scale+bias |
| GEMV q4 | 2048x2048 | q4 | JIT | 474.6 (432.3-599.5) | 563.7 | 4.2 (2.1%) | 14.9 | jitLLM matrixVectorGenericQ4_0Simd32, GGUF Q4_0 |
| GEMV q8 | 2048x2048 | q8 | MLX alone | 259.0 (call) | 259.0 | 18.3 (9.1%) | 32.4 |  |
| GEMV q8 | 2048x2048 | q8 | apple/mlx | 294.6 (272.1-338.8) | 279.7 | 16.9 (8.4%) | 30.0 | MLX affine, group 32, fp16 scale+bias |
| GEMV q8 | 2048x2048 | q8 | JIT | 416.0 (388.1-493.2) | 366.4 | 12.2 (6.1%) | 22.9 | jitLLM matrixVectorGenericQ8Byte, GGUF Q8_0 |
| GEMV q4 | 3072x3072 | q4 | MLX alone | 285.5 (call) | 285.5 | 20.7 (10.4%) | 66.1 |  |
| GEMV q4 | 3072x3072 | q4 | apple/mlx | 282.7 (261.1-317.7) | 272.0 | 21.7 (10.9%) | 69.4 | MLX affine, group 32, fp16 scale+bias |
| GEMV q4 | 3072x3072 | q4 | JIT | 877.5 (839.9-1004.4) | 836.1 | 6.4 (3.2%) | 22.6 | jitLLM matrixVectorGenericQ4_0Simd32, GGUF Q4_0 |
| GEMV q8 | 3072x3072 | q8 | MLX alone | 270.6 (call) | 270.6 | 39.3 (19.6%) | 69.7 |  |
| GEMV q8 | 3072x3072 | q8 | apple/mlx | 301.4 (271.4-334.8) | 279.5 | 38.0 (19.0%) | 67.5 | MLX affine, group 32, fp16 scale+bias |
| GEMV q8 | 3072x3072 | q8 | JIT | 504.3 (491.5-572.7) | 466.5 | 21.5 (10.8%) | 40.5 | jitLLM matrixVectorGenericQ8Byte, GGUF Q8_0 |
| GEMV q4 | 4096x4096 | q4 | MLX alone | 337.9 (call) | 337.9 | 31.1 (15.5%) | 99.3 |  |
| GEMV q4 | 4096x4096 | q4 | apple/mlx | 332.1 (296.3-409.7) | 385.7 | 27.2 (13.6%) | 87.0 | MLX affine, group 32, fp16 scale+bias |
| GEMV q4 | 4096x4096 | q4 | JIT | 657.5 (580.6-860.0) | 589.7 | 16.1 (8.0%) | 56.9 | jitLLM matrixVectorGenericQ4_0Simd32, GGUF Q4_0 |
| GEMV q8 | 4096x4096 | q8 | MLX alone | 309.4 (call) | 309.4 | 61.1 (30.5%) | 108.5 |  |
| GEMV q8 | 4096x4096 | q8 | apple/mlx | 343.3 (316.1-390.1) | 314.1 | 60.2 (30.1%) | 106.8 | MLX affine, group 32, fp16 scale+bias |
| GEMV q8 | 4096x4096 | q8 | JIT | 677.0 (661.3-709.8) | 716.4 | 24.9 (12.5%) | 46.8 | jitLLM matrixVectorGenericQ8Byte, GGUF Q8_0 |
| GEMV q4 | 128256x2048 | q4 | MLX alone | 1105.5 (call) | 1105.5 | 148.7 (74.4%) | 475.2 |  |
| GEMV q4 | 128256x2048 | q4 | apple/mlx | 1165.2 (1129.9-1239.3) | 1571.0 | 104.7 (52.3%) | 334.4 | MLX affine, group 32, fp16 scale+bias |
| GEMV q4 | 128256x2048 | q4 | JIT | 5137.9 (5030.2-5472.0) | 4882.4 | 30.4 (15.2%) | 107.6 | jitLLM matrixVectorGenericQ4_0Simd32, GGUF Q4_0 |
| GEMM fp16 | 512x2048x2048 | f16 | MLX alone | 1179.4 (call) | 1179.4 | 10.7 (5.3%) | 3641.6 |  |
| GEMM fp16 | 512x2048x2048 | f16 | apple/mlx | 1304.4 (1280.1-1331.1) | 1275.2 | 9.9 (4.9%) | 3368.2 |  |
| GEMM fp16 | 512x2048x2048 | f16 | JIT | 10406.4 (10347.3-10486.8) | 10364.4 | 1.2 (0.6%) | 414.4 | 16x16 tiled KernelContext GEMM (not jitLLM-tuned) |
| GEMM fp16 | 512x4096x4096 | f16 | MLX alone | 4186.2 (call) | 4186.2 | 10.0 (5.0%) | 4104.0 |  |
| GEMM fp16 | 512x4096x4096 | f16 | apple/mlx | 4525.6 (4423.0-4788.0) | 4335.7 | 9.7 (4.8%) | 3962.5 |  |
| GEMM fp16 | 512x4096x4096 | f16 | JIT | 41701.8 (41594.2-41816.3) | 42327.9 | 1.0 (0.5%) | 405.9 | 16x16 tiled KernelContext GEMM (not jitLLM-tuned) |
| RMS norm | 1x2048 | f32 | MLX alone | 227.4 (call) | 227.4 | 0.1 (0.1%) | 0.0 |  |
| RMS norm | 1x2048 | f32 | apple/mlx | 265.8 (239.8-327.8) | 221.1 | 0.1 (0.1%) | 0.0 |  |
| RMS norm | 1x2048 | f32 | JIT | 546.5 (513.6-653.3) | 504.6 | 0.0 (0.0%) | 0.0 | jitLLM reductionOneBlockWithLayerSingleGroup + reductionOneBlock2WithLayer (2 kernels) |
| RMS norm | 1x3072 | f32 | MLX alone | 224.2 (call) | 224.2 | 0.2 (0.1%) | 0.1 |  |
| RMS norm | 1x3072 | f32 | apple/mlx | 249.5 (228.3-285.2) | 233.3 | 0.2 (0.1%) | 0.1 |  |
| RMS norm | 1x3072 | f32 | JIT | 549.5 (502.0-661.7) | 513.7 | 0.1 (0.0%) | 0.0 | jitLLM reductionOneBlockWithLayerSingleGroup + reductionOneBlock2WithLayer (2 kernels) |
| RMS norm | 1x4096 | f32 | MLX alone | 206.9 (call) | 206.9 | 0.2 (0.1%) | 0.1 |  |
| RMS norm | 1x4096 | f32 | apple/mlx | 245.3 (216.7-296.4) | 195.8 | 0.3 (0.1%) | 0.1 |  |
| RMS norm | 1x4096 | f32 | JIT | 475.8 (412.3-569.8) | 532.4 | 0.1 (0.0%) | 0.0 | jitLLM reductionOneBlockWithLayerSingleGroup + reductionOneBlock2WithLayer (2 kernels) |
| RMS norm | 512x4096 | f32 | MLX alone | 277.9 (call) | 277.9 | 60.4 (30.2%) | 30.2 |  |
| RMS norm | 512x4096 | f32 | apple/mlx | 696.6 (655.6-775.3) | 708.3 | 23.7 (11.9%) | 11.8 |  |
| RoPE | 32x1x128 | f32 | MLX alone | 213.7 (call) | 213.7 | 0.2 (0.1%) | 0.1 |  |
| RoPE | 32x1x128 | f32 | apple/mlx | 270.4 (239.0-324.4) | 217.8 | 0.2 (0.1%) | 0.1 |  |
| RoPE | 32x1x128 | f32 | JIT | 279.3 (253.2-375.9) | 266.6 | 0.1 (0.1%) | 0.1 | @Parallel kernel (jitLLM fuses RoPE with its KV-cache write) |
| RoPE | 32x512x128 | f32 | MLX alone | 283.8 (call) | 283.8 | 59.1 (29.6%) | 44.3 |  |
| RoPE | 32x512x128 | f32 | apple/mlx | 690.2 (651.1-728.1) | 641.8 | 26.1 (13.1%) | 19.6 |  |
| SDPA decode | ctx 512 | f32 | MLX alone | 285.5 (call) | 285.5 | 14.8 (7.4%) | 29.4 |  |
| SDPA decode | ctx 512 | f32 | apple/mlx | 296.0 (270.6-331.2) | 323.1 | 13.1 (6.5%) | 26.0 | 32 query / 8 KV heads, head dim 128 |
| SDPA decode | ctx 512 | f32 | JIT | 1440.5 (1324.5-1647.6) | 1258.1 | 3.4 (1.7%) | 6.7 | online-softmax KernelContext kernel (jitLLM's is paged-KV) |
| SDPA decode | ctx 2048 | f32 | MLX alone | 361.2 (call) | 361.2 | 46.5 (23.3%) | 92.9 |  |
| SDPA decode | ctx 2048 | f32 | apple/mlx | 372.3 (340.6-458.6) | 312.0 | 53.9 (26.9%) | 107.6 | 32 query / 8 KV heads, head dim 128 |
| SDPA decode | ctx 2048 | f32 | JIT | 3695.6 (3664.8-3744.7) | 3831.0 | 4.4 (2.2%) | 8.8 | online-softmax KernelContext kernel (jitLLM's is paged-KV) |
| SDPA decode | ctx 8192 | f32 | MLX alone | 683.3 (call) | 683.3 | 98.3 (49.1%) | 196.4 |  |
| SDPA decode | ctx 8192 | f32 | apple/mlx | 740.3 (717.9-772.7) | 697.6 | 96.2 (48.1%) | 192.4 | 32 query / 8 KV heads, head dim 128 |
| SDPA decode | ctx 8192 | f32 | JIT | 14982.6 (14914.4-15041.0) | 14938.3 | 4.5 (2.2%) | 9.0 | online-softmax KernelContext kernel (jitLLM's is paged-KV) |
| softmax | vocab 151936 | f32 | MLX alone | 274.0 (call) | 274.0 | 4.4 (2.2%) | 2.8 |  |
| softmax | vocab 151936 | f32 | apple/mlx | 337.3 (320.9-359.8) | 317.0 | 3.8 (1.9%) | 2.4 |  |
| softmax | vocab 151936 | f32 | JIT | 412.7 (395.1-429.0) | 429.2 | 2.8 (1.4%) | 1.8 | one-workgroup KernelContext kernel |
| argmax | vocab 151936 | f32 | MLX alone | 289.8 (call) | 289.8 | 2.1 (1.0%) | 0.5 |  |
| argmax | vocab 151936 | f32 | apple/mlx | 337.4 (322.5-377.6) | 292.3 | 2.1 (1.0%) | 0.5 |  |
| argmax | vocab 151936 | f32 | JIT | 324.2 (290.6-359.4) | 326.2 | 1.9 (0.9%) | 0.5 | jitLLM argmaxLogits |
| top-k (k=50) | vocab 151936 | f32 | MLX alone | 525.3 (call) | 525.3 | 1.2 (0.6%) | 0.3 |  |
| top-k (k=50) | vocab 151936 | f32 | apple/mlx | 592.2 (572.0-620.4) | 537.1 | 1.1 (0.6%) | 0.3 | no JIT top-k to compare against |
