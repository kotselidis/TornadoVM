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
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * JIT counterparts of the MLX sort, argsort, partition and argpartition, float32, as bitonic
 * sorts of (key, index) pairs (the network of
 * {@code tornado-unittests/.../kernelcontext/sort/TestBitonicSort.java}):
 * <ul>
 * <li><b>along an axis</b> ({@link #sortSlices}): one threadgroup per slice {@code x[o, :, j]},
 * sorted in threadgroup memory; slices of up to {@link #BLOCK} elements;</li>
 * <li><b>a whole array</b>: {@link #pad} into a power-of-two scratch buffer (+infinity padding),
 * {@link #sortBlocks} sorts each {@link #BLOCK}-element block in threadgroup memory, then each
 * merge stage {@code k > BLOCK} runs {@link #mergeGlobal} for strides {@code j >= BLOCK} and
 * {@link #mergeBlocks} for the strides below, and {@link #unpad} writes the first {@code n}.</li>
 * </ul>
 * A sorted array also satisfies a partition, so partition and argpartition use the same kernels.
 */
public final class JitSort {

    /** Threads per threadgroup. */
    public static final int THREADS = 256;
    /** Elements sorted per threadgroup (a power of two); pairs of float and int fill 16 KB. */
    public static final int BLOCK = 2048;

    private JitSort() {
    }

    /**
     * Sorts each slice x[o, :, j] of x viewed as [outer, len, inner] ascending into values (if
     * {@code writeValues != 0}) and indices (if {@code writeIndices != 0}); {@code size} is a power
     * of two with {@code len <= size <= BLOCK}. One threadgroup per slice.
     */
    @JitBaseline(value = { "mlx_sort_axis", "mlx_argsort_axis", "mlx_partition_axis", "mlx_argpartition_axis", "mlx_sort", "mlx_argsort", "mlx_partition", "mlx_argpartition" },
            source = "written (bitonic network as in tornado-unittests/.../kernelcontext/sort/TestBitonicSort.java)")
    public static void sortSlices(KernelContext ctx, FloatArray x, FloatArray values, IntArray indices, int len, int inner, int size, int writeValues, int writeIndices) {
        float[] keys = ctx.allocateFloatLocalArray(BLOCK);
        int[] idx = ctx.allocateIntLocalArray(BLOCK);
        int slice = ctx.groupIdx;
        int tid = ctx.localIdx;
        int base = (slice / inner) * len * inner + slice % inner;
        for (int i = tid; i < size; i += THREADS) {
            keys[i] = i < len ? x.get(base + i * inner) : Float.POSITIVE_INFINITY;
            idx[i] = i;
        }
        ctx.localBarrier();
        for (int k = 2; k <= size; k <<= 1) {
            for (int j = k >> 1; j > 0; j >>= 1) {
                for (int t = tid; t < size; t += THREADS) {
                    int partner = t ^ j;
                    if (partner > t) {
                        boolean ascending = (t & k) == 0;
                        float a = keys[t];
                        float b = keys[partner];
                        if (ascending ? a > b : a < b) {
                            keys[t] = b;
                            keys[partner] = a;
                            int ti = idx[t];
                            idx[t] = idx[partner];
                            idx[partner] = ti;
                        }
                    }
                }
                ctx.localBarrier();
            }
        }
        for (int i = tid; i < len; i += THREADS) {
            if (writeValues != 0) {
                values.set(base + i * inner, keys[i]);
            }
            if (writeIndices != 0) {
                indices.set(base + i * inner, idx[i]);
            }
        }
    }

    /** keys[i] = x[i] (or +infinity past n), indices[i] = i, for i below the padded size. */
    @JitBaseline({ "mlx_sort", "mlx_argsort", "mlx_partition", "mlx_argpartition" })
    public static void pad(KernelContext ctx, FloatArray x, FloatArray keys, IntArray indices, int n, int padded) {
        int i = ctx.globalIdx;
        if (i < padded) {
            keys.set(i, i < n ? x.get(i) : Float.POSITIVE_INFINITY);
            indices.set(i, i);
        }
    }

    /** All bitonic stages up to BLOCK within each BLOCK-element block, directions by global index. */
    @JitBaseline({ "mlx_sort", "mlx_argsort", "mlx_partition", "mlx_argpartition" })
    public static void sortBlocks(KernelContext ctx, FloatArray keys, IntArray indices) {
        float[] k2 = ctx.allocateFloatLocalArray(BLOCK);
        int[] i2 = ctx.allocateIntLocalArray(BLOCK);
        int tid = ctx.localIdx;
        int base = ctx.groupIdx * BLOCK;
        for (int i = tid; i < BLOCK; i += THREADS) {
            k2[i] = keys.get(base + i);
            i2[i] = indices.get(base + i);
        }
        ctx.localBarrier();
        for (int k = 2; k <= BLOCK; k <<= 1) {
            for (int j = k >> 1; j > 0; j >>= 1) {
                for (int t = tid; t < BLOCK; t += THREADS) {
                    int partner = t ^ j;
                    if (partner > t) {
                        boolean ascending = ((base + t) & k) == 0;
                        float a = k2[t];
                        float b = k2[partner];
                        if (ascending ? a > b : a < b) {
                            k2[t] = b;
                            k2[partner] = a;
                            int ti = i2[t];
                            i2[t] = i2[partner];
                            i2[partner] = ti;
                        }
                    }
                }
                ctx.localBarrier();
            }
        }
        for (int i = tid; i < BLOCK; i += THREADS) {
            keys.set(base + i, k2[i]);
            indices.set(base + i, i2[i]);
        }
    }

    /** One compare-exchange step (stage k, stride j >= BLOCK) over the whole padded array; padded / 2 threads. */
    @JitBaseline({ "mlx_sort", "mlx_argsort", "mlx_partition", "mlx_argpartition" })
    public static void mergeGlobal(KernelContext ctx, FloatArray keys, IntArray indices, int padded, int k, int j) {
        int p = ctx.globalIdx;
        if (p < padded / 2) {
            // The p-th pair: t has bit j clear.
            int t = (p / j) * 2 * j + p % j;
            int partner = t + j;
            boolean ascending = (t & k) == 0;
            float a = keys.get(t);
            float b = keys.get(partner);
            if (ascending ? a > b : a < b) {
                keys.set(t, b);
                keys.set(partner, a);
                int ti = indices.get(t);
                indices.set(t, indices.get(partner));
                indices.set(partner, ti);
            }
        }
    }

    /** The strides below BLOCK of stage k, within each BLOCK-element block in threadgroup memory. */
    @JitBaseline({ "mlx_sort", "mlx_argsort", "mlx_partition", "mlx_argpartition" })
    public static void mergeBlocks(KernelContext ctx, FloatArray keys, IntArray indices, int k) {
        float[] k2 = ctx.allocateFloatLocalArray(BLOCK);
        int[] i2 = ctx.allocateIntLocalArray(BLOCK);
        int tid = ctx.localIdx;
        int base = ctx.groupIdx * BLOCK;
        for (int i = tid; i < BLOCK; i += THREADS) {
            k2[i] = keys.get(base + i);
            i2[i] = indices.get(base + i);
        }
        ctx.localBarrier();
        boolean ascending = (base & k) == 0;
        for (int j = BLOCK >> 1; j > 0; j >>= 1) {
            for (int t = tid; t < BLOCK; t += THREADS) {
                int partner = t ^ j;
                if (partner > t) {
                    float a = k2[t];
                    float b = k2[partner];
                    if (ascending ? a > b : a < b) {
                        k2[t] = b;
                        k2[partner] = a;
                        int ti = i2[t];
                        i2[t] = i2[partner];
                        i2[partner] = ti;
                    }
                }
            }
            ctx.localBarrier();
        }
        for (int i = tid; i < BLOCK; i += THREADS) {
            keys.set(base + i, k2[i]);
            indices.set(base + i, i2[i]);
        }
    }

    /** Writes the first n sorted keys (if writeValues != 0) and indices (if writeIndices != 0). */
    @JitBaseline({ "mlx_sort", "mlx_argsort", "mlx_partition", "mlx_argpartition" })
    public static void unpad(KernelContext ctx, FloatArray keys, IntArray indices, FloatArray values, IntArray outIndices, int n, int writeValues, int writeIndices) {
        int i = ctx.globalIdx;
        if (i < n) {
            if (writeValues != 0) {
                values.set(i, keys.get(i));
            }
            if (writeIndices != 0) {
                outIndices.set(i, indices.get(i));
            }
        }
    }
}
