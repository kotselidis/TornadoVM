/*
 * Copyright (c) 2026, APT Group, Department of Computer Science,
 * The University of Manchester.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */
package uk.ac.manchester.tornado.mlx.jit;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * JIT counterparts of the MLX affine group quantization operations, float32, for bit widths that
 * divide 32 (2, 4, 8). The format matches MLX: a row of {@code cols} weights is split into groups of
 * {@code groupSize}, each with {@code w = scale * q + bias}, and the {@code q} are packed into 32-bit
 * words low bits first.
 */
public final class JitQuantized {

    /** Threads per simdgroup; the matmul kernels use one simdgroup per output element. */
    public static final int SIMD_THREADS = 32;

    private static final float EPS = 1e-7f;

    private JitQuantized() {
    }

    /** Round half away from zero, as Metal's {@code round}. */
    private static float roundAway(float v) {
        return v >= 0.0f ? TornadoMath.floor(v + 0.5f) : -TornadoMath.floor(-v + 0.5f);
    }

    /**
     * Quantizes w[rows, cols], one thread per group, with MLX's scale selection: the edge of larger
     * magnitude is kept exact and the scale is adjusted so that zero is representable.
     */
    @JitBaseline("mlx_quantize")
    public static void quantize(KernelContext ctx, FloatArray w, IntArray wq, FloatArray scales, FloatArray biases, int groups, int groupSize, int bits) {
        int g = ctx.globalIdx;
        if (g < groups) {
            int base = g * groupSize;
            float wMin = w.get(base);
            float wMax = wMin;
            for (int i = 1; i < groupSize; i++) {
                float v = w.get(base + i);
                wMin = TornadoMath.min(wMin, v);
                wMax = TornadoMath.max(wMax, v);
            }
            float nBins = (float) ((1 << bits) - 1);
            float scale = TornadoMath.max((wMax - wMin) / nBins, EPS);
            boolean side = TornadoMath.abs(wMin) > TornadoMath.abs(wMax);
            scale = side ? scale : -scale;
            float edge = side ? wMin : wMax;
            float q0 = roundAway(edge / scale);
            float bias = 0.0f;
            if (q0 != 0.0f) {
                scale = edge / q0;
                bias = edge;
            }
            scales.set(g, scale);
            biases.set(g, bias);
            int perWord = 32 / bits;
            int wordBase = base / perWord;
            for (int wd = 0; wd < groupSize / perWord; wd++) {
                int word = 0;
                for (int j = 0; j < perWord; j++) {
                    float q = roundAway((w.get(base + wd * perWord + j) - bias) / scale);
                    int qi = (int) TornadoMath.min(TornadoMath.max(q, 0.0f), nBins);
                    word |= qi << (j * bits);
                }
                wq.set(wordBase + wd, word);
            }
        }
    }

    /** Dequantizes packed weights into w[rows, cols], one thread per packed word. */
    @JitBaseline("mlx_dequantize")
    public static void dequantize(KernelContext ctx, IntArray wq, FloatArray scales, FloatArray biases, FloatArray w, int words, int groupSize, int bits) {
        int wd = ctx.globalIdx;
        if (wd < words) {
            int perWord = 32 / bits;
            int mask = (1 << bits) - 1;
            int base = wd * perWord;
            int g = base / groupSize;
            float scale = scales.get(g);
            float bias = biases.get(g);
            int word = wq.get(wd);
            for (int j = 0; j < perWord; j++) {
                // Arithmetic shift is safe: the mask drops the sign bits.
                int q = (word >> (j * bits)) & mask;
                w.set(base + j, scale * q + bias);
            }
        }
    }

    /**
     * y[m, n] = x[m, k] @ dequantize(w)[n, k]^T, one simdgroup per output element (the
     * {@link JitBlas#gemv} pattern over packed words). Each word lies inside one group, so a word
     * contributes {@code scale * sum(x * q) + bias * sum(x)}.
     */
    @JitBaseline(value = "mlx_quantized_matmul", source = "tornado-examples/.../compute/MatrixVectorSimdReduction.java#matvecSimd (MLX affine weights)")
    public static void quantizedMatmul(KernelContext ctx, FloatArray x, IntArray wq, FloatArray scales, FloatArray biases, FloatArray y, int k, int n, int groupSize,
            int bits) {
        int out = ctx.groupIdx;
        int lane = ctx.localIdx;
        int row = out / n;
        int col = out % n;
        int perWord = 32 / bits;
        int mask = (1 << bits) - 1;
        int wordsPerRow = k / perWord;
        int wBase = col * wordsPerRow;
        int gBase = col * (k / groupSize);
        int xBase = row * k;
        float partial = 0.0f;
        for (int wd = lane; wd < wordsPerRow; wd += SIMD_THREADS) {
            int word = wq.get(wBase + wd);
            int kBase = wd * perWord;
            float sxq = 0.0f;
            float sx = 0.0f;
            for (int j = 0; j < perWord; j++) {
                float xv = x.get(xBase + kBase + j);
                sxq += xv * ((word >> (j * bits)) & mask);
                sx += xv;
            }
            int g = gBase + kBase / groupSize;
            partial += scales.get(g) * sxq + biases.get(g) * sx;
        }
        float sum = ctx.simdSum(partial);
        if (lane == 0) {
            y.set(out, sum);
        }
    }

    /**
     * Mixture-of-experts form of {@link #quantizedMatmul}: y[b] = x[lhs[b]] @ dequantize(w[rhs[b]])^T
     * with x[batches, m, k], w[experts, n, k] and y[batches, m, n].
     */
    @JitBaseline(value = "mlx_gather_qmm", source = "tornado-examples/.../compute/MatrixVectorSimdReduction.java#matvecSimd (MLX affine weights, gathered)")
    public static void gatherQmm(KernelContext ctx, FloatArray x, IntArray wq, FloatArray scales, FloatArray biases, IntArray lhs, IntArray rhs, FloatArray y, int m, int k,
            int n, int groupSize, int bits) {
        int out = ctx.groupIdx;
        int lane = ctx.localIdx;
        int batch = out / (m * n);
        int row = (out / n) % m;
        int col = out % n;
        int expertRow = rhs.get(batch) * n + col;
        int perWord = 32 / bits;
        int mask = (1 << bits) - 1;
        int wordsPerRow = k / perWord;
        int wBase = expertRow * wordsPerRow;
        int gBase = expertRow * (k / groupSize);
        int xBase = (lhs.get(batch) * m + row) * k;
        float partial = 0.0f;
        for (int wd = lane; wd < wordsPerRow; wd += SIMD_THREADS) {
            int word = wq.get(wBase + wd);
            int kBase = wd * perWord;
            float sxq = 0.0f;
            float sx = 0.0f;
            for (int j = 0; j < perWord; j++) {
                float xv = x.get(xBase + kBase + j);
                sxq += xv * ((word >> (j * bits)) & mask);
                sx += xv;
            }
            int g = gBase + kBase / groupSize;
            partial += scales.get(g) * sxq + biases.get(g) * sx;
        }
        float sum = ctx.simdSum(partial);
        if (lane == 0) {
            y.set(out, sum);
        }
    }
}
