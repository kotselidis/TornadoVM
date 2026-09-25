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
 * JIT counterparts of the MLX softmax, argmax and top-k operations, float32, one threadgroup per
 * row. The whole-array forms are the one-row case.
 */
public final class JitReductions {

    /** Threads per row for softmax and argmax (a power of two, at most 1024). */
    public static final int ROW_THREADS = 1024;
    /** Threads per row for top-k. */
    public static final int TOPK_THREADS = 64;
    /** Largest k the top-k kernel supports. */
    public static final int TOPK_MAX_K = 64;

    private static final int TOPK_CANDIDATES = TOPK_THREADS * TOPK_MAX_K;

    private JitReductions() {
    }

    /**
     * Softmax of each row of x[rows, cols]: max and sum of exponentials reduced through
     * threadgroup memory, then normalised.
     */
    @JitBaseline({ "mlx_softmax", "mlx_softmax_axis", "mlx_softmax_axes" })
    public static void softmaxRows(KernelContext ctx, FloatArray x, FloatArray out, int cols) {
        float[] red = ctx.allocateFloatLocalArray(ROW_THREADS);
        int row = ctx.groupIdx;
        int tid = ctx.localIdx;
        int lsz = ctx.localGroupSizeX;
        int base = row * cols;
        float max = Float.NEGATIVE_INFINITY;
        for (int i = tid; i < cols; i += lsz) {
            max = TornadoMath.max(max, x.get(base + i));
        }
        red[tid] = max;
        ctx.localBarrier();
        for (int s = lsz / 2; s > 0; s >>= 1) {
            if (tid < s) {
                red[tid] = TornadoMath.max(red[tid], red[tid + s]);
            }
            ctx.localBarrier();
        }
        float rowMax = red[0];
        ctx.localBarrier();
        float sum = 0.0f;
        for (int i = tid; i < cols; i += lsz) {
            sum += TornadoMath.exp(x.get(base + i) - rowMax);
        }
        red[tid] = sum;
        ctx.localBarrier();
        for (int s = lsz / 2; s > 0; s >>= 1) {
            if (tid < s) {
                red[tid] += red[tid + s];
            }
            ctx.localBarrier();
        }
        float inv = 1.0f / red[0];
        for (int i = tid; i < cols; i += lsz) {
            out.set(base + i, TornadoMath.exp(x.get(base + i) - rowMax) * inv);
        }
    }

    /** Index of the largest element of each row of x[rows, cols] (the first one on ties). */
    @JitBaseline({ "mlx_argmax", "mlx_argmax_axis" })
    public static void argmaxRows(KernelContext ctx, FloatArray x, IntArray out, int cols) {
        float[] vals = ctx.allocateFloatLocalArray(ROW_THREADS);
        int[] idxs = ctx.allocateIntLocalArray(ROW_THREADS);
        int row = ctx.groupIdx;
        int tid = ctx.localIdx;
        int lsz = ctx.localGroupSizeX;
        int base = row * cols;
        float best = Float.NEGATIVE_INFINITY;
        int bestIdx = cols;
        for (int i = tid; i < cols; i += lsz) {
            float v = x.get(base + i);
            if (v > best) {
                best = v;
                bestIdx = i;
            }
        }
        vals[tid] = best;
        idxs[tid] = bestIdx;
        ctx.localBarrier();
        for (int s = lsz / 2; s > 0; s >>= 1) {
            if (tid < s) {
                float other = vals[tid + s];
                int otherIdx = idxs[tid + s];
                if (other > vals[tid] || (other == vals[tid] && otherIdx < idxs[tid])) {
                    vals[tid] = other;
                    idxs[tid] = otherIdx;
                }
            }
            ctx.localBarrier();
        }
        if (tid == 0) {
            out.set(row, idxs[0]);
        }
    }

    /**
     * The {@code k} largest elements of each row of x[rows, cols], in descending order
     * ({@code k <= 64}). Each of the 64 threads keeps its own top-k of a strided slice; the 64 * 64
     * candidates are then sorted in threadgroup memory with a bitonic sort (the network of
     * {@code tornado-unittests/.../kernelcontext/sort/TestBitonicSort.java}). Launch with
     * {@link #TOPK_THREADS} threads per row.
     */
    @JitBaseline(value = { "mlx_topk", "mlx_topk_axis" }, source = "written (bitonic network as in tornado-unittests/.../kernelcontext/sort/TestBitonicSort.java)")
    public static void topkRows(KernelContext ctx, FloatArray x, FloatArray out, int cols, int k) {
        float[] cand = ctx.allocateFloatLocalArray(TOPK_CANDIDATES);
        int row = ctx.groupIdx;
        int tid = ctx.localIdx;
        int base = row * cols;
        float[] mine = new float[TOPK_MAX_K];
        for (int i = 0; i < TOPK_MAX_K; i++) {
            mine[i] = Float.NEGATIVE_INFINITY;
        }
        // Private top-k, descending, by insertion.
        for (int i = tid; i < cols; i += TOPK_THREADS) {
            float v = x.get(base + i);
            if (v > mine[k - 1]) {
                int p = k - 1;
                while (p > 0 && mine[p - 1] < v) {
                    mine[p] = mine[p - 1];
                    p--;
                }
                mine[p] = v;
            }
        }
        for (int i = 0; i < TOPK_MAX_K; i++) {
            cand[tid * TOPK_MAX_K + i] = mine[i];
        }
        ctx.localBarrier();
        // Bitonic sort of all candidates, descending.
        for (int size = 2; size <= TOPK_CANDIDATES; size <<= 1) {
            for (int stride = size >> 1; stride > 0; stride >>= 1) {
                for (int t = tid; t < TOPK_CANDIDATES; t += TOPK_THREADS) {
                    int partner = t ^ stride;
                    if (partner > t) {
                        boolean descending = (t & size) == 0;
                        float a = cand[t];
                        float b = cand[partner];
                        if (descending ? a < b : a > b) {
                            cand[t] = b;
                            cand[partner] = a;
                        }
                    }
                }
                ctx.localBarrier();
            }
        }
        for (int i = tid; i < k; i += TOPK_THREADS) {
            out.set(row * k + i, cand[i]);
        }
    }
}
