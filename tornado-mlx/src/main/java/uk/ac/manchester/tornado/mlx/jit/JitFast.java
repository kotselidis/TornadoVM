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
 * JIT counterparts of the MLX {@code mlx.fast} operations, float32.
 */
public final class JitFast {

    /** Threads per row for the normalisation kernels (a multiple of 32, at most 1024). */
    public static final int NORM_THREADS = 256;
    /** Threads per query head for the attention kernel. */
    public static final int ATTENTION_THREADS = 64;

    private static final int MAX_HEAD_DIM = 128;
    private static final int KEY_BLOCK = 16;

    private JitFast() {
    }

    /** out[r, :] = x[r, :] / sqrt(mean(x[r, :]^2) + eps) * weight; one threadgroup per row. */
    @JitBaseline("mlx_fast_rms_norm")
    public static void rmsNorm(KernelContext ctx, FloatArray x, FloatArray weight, FloatArray out, int dim, float eps) {
        float[] scratch = ctx.allocateFloatLocalArray(32);
        int row = ctx.groupIdx;
        int lid = ctx.localIdx;
        int lsz = ctx.localGroupSizeX;
        int base = row * dim;
        float partial = 0.0f;
        for (int j = lid; j < dim; j += lsz) {
            float v = x.get(base + j);
            partial += v * v;
        }
        // Threadgroup sum: simd_sum within each SIMD group, then the group sums through threadgroup
        // memory. (Inlined: TornadoVM cannot pass a threadgroup array to a helper method.)
        float partialSimd = ctx.simdSum(partial);
        if (lid % 32 == 0) {
            scratch[lid / 32] = partialSimd;
        }
        ctx.localBarrier();
        if (lid < 32) {
            float t = (lid < lsz / 32) ? scratch[lid] : 0.0f;
            t = ctx.simdSum(t);
            if (lid == 0) {
                scratch[0] = t;
            }
        }
        ctx.localBarrier();
        float sumSquares = scratch[0];
        ctx.localBarrier();
        float inv = 1.0f / TornadoMath.sqrt(sumSquares / dim + eps);
        for (int j = lid; j < dim; j += lsz) {
            out.set(base + j, x.get(base + j) * inv * weight.get(j));
        }
    }

    /** out[r, :] = (x[r, :] - mean) / sqrt(var + eps) * weight + bias; one threadgroup per row. */
    @JitBaseline("mlx_fast_layer_norm")
    public static void layerNorm(KernelContext ctx, FloatArray x, FloatArray weight, FloatArray bias, FloatArray out, int dim, float eps) {
        float[] scratch = ctx.allocateFloatLocalArray(32);
        int row = ctx.groupIdx;
        int lid = ctx.localIdx;
        int lsz = ctx.localGroupSizeX;
        int base = row * dim;
        float partial = 0.0f;
        for (int j = lid; j < dim; j += lsz) {
            partial += x.get(base + j);
        }
        // Threadgroup sum: simd_sum within each SIMD group, then the group sums through threadgroup
        // memory. (Inlined: TornadoVM cannot pass a threadgroup array to a helper method.)
        float partialSimd = ctx.simdSum(partial);
        if (lid % 32 == 0) {
            scratch[lid / 32] = partialSimd;
        }
        ctx.localBarrier();
        if (lid < 32) {
            float t = (lid < lsz / 32) ? scratch[lid] : 0.0f;
            t = ctx.simdSum(t);
            if (lid == 0) {
                scratch[0] = t;
            }
        }
        ctx.localBarrier();
        float sum = scratch[0];
        ctx.localBarrier();
        float mean = sum / dim;
        float partialVar = 0.0f;
        for (int j = lid; j < dim; j += lsz) {
            float d = x.get(base + j) - mean;
            partialVar += d * d;
        }
        // Threadgroup sum: simd_sum within each SIMD group, then the group sums through threadgroup
        // memory. (Inlined: TornadoVM cannot pass a threadgroup array to a helper method.)
        float partialVarSimd = ctx.simdSum(partialVar);
        if (lid % 32 == 0) {
            scratch[lid / 32] = partialVarSimd;
        }
        ctx.localBarrier();
        if (lid < 32) {
            float tv = (lid < lsz / 32) ? scratch[lid] : 0.0f;
            tv = ctx.simdSum(tv);
            if (lid == 0) {
                scratch[0] = tv;
            }
        }
        ctx.localBarrier();
        float sumVar = scratch[0];
        ctx.localBarrier();
        float inv = 1.0f / TornadoMath.sqrt(sumVar / dim + eps);
        for (int j = lid; j < dim; j += lsz) {
            out.set(base + j, (x.get(base + j) - mean) * inv * weight.get(j) + bias.get(j));
        }
    }

    /**
     * RoPE of x[rows = batch * heads * seqLen, headDim] at positions {@code offset + (row % seqLen)}.
     * One thread per pair of features: pairs {@code (i, i + dims/2)}, or {@code (2i, 2i + 1)} if
     * {@code traditional}; features past {@code dims} are copied. Launch with
     * {@code rows * headDim / 2} threads.
     */
    @JitBaseline(value = "mlx_fast_rope", source = "tornado-unittests/.../compute/TransformerKernelsTest.java#ropeRotation (out of place, parametric base/scale/offset, both pairings)")
    public static void rope(KernelContext ctx, FloatArray x, FloatArray out, int rows, int seqLen, int headDim, int dims, int traditional, float base, float scale, int offset) {
        ropePair(ctx, x, out, rows, seqLen, headDim, dims, traditional, base, scale, offset);
    }

    /** {@link #rope} with the position offset read from a one-element array on the device. */
    @JitBaseline(value = "mlx_fast_rope_dynamic", source = "tornado-unittests/.../compute/TransformerKernelsTest.java#ropeRotation (position from device memory)")
    public static void ropeDynamic(KernelContext ctx, FloatArray x, IntArray offset, FloatArray out, int rows, int seqLen, int headDim, int dims, int traditional, float base, float scale) {
        ropePair(ctx, x, out, rows, seqLen, headDim, dims, traditional, base, scale, offset.get(0));
    }

    private static void ropePair(KernelContext ctx, FloatArray x, FloatArray out, int rows, int seqLen, int headDim, int dims, int traditional, float base, float scale, int offset) {
        int gid = ctx.globalIdx;
        int pairsPerRow = headDim / 2;
        if (gid >= rows * pairsPerRow) {
            return;
        }
        int row = gid / pairsPerRow;
        int j = gid % pairsPerRow;
        int rowBase = row * headDim;
        int half = dims / 2;
        if (j < half) {
            float position = (offset + row % seqLen) * scale;
            float theta = position * TornadoMath.pow(base, -2.0f * j / dims);
            float cosT = TornadoMath.cos(theta);
            float sinT = TornadoMath.sin(theta);
            int i1 = traditional != 0 ? 2 * j : j;
            int i2 = traditional != 0 ? 2 * j + 1 : j + half;
            float x1 = x.get(rowBase + i1);
            float x2 = x.get(rowBase + i2);
            out.set(rowBase + i1, x1 * cosT - x2 * sinT);
            out.set(rowBase + i2, x1 * sinT + x2 * cosT);
        } else {
            int f = dims + 2 * (j - half);
            out.set(rowBase + f, x.get(rowBase + f));
            out.set(rowBase + f + 1, x.get(rowBase + f + 1));
        }
    }

    /**
     * Decode attention, one query per head: {@code out[h] = softmax(q[h] . k[kh]^T * scale) v[kh]},
     * with {@code q[qHeads, headDim]}, {@code k, v[kvHeads, kvLen, headDim]} and
     * {@code kh = h / (qHeads / kvHeads)}. Tiled online softmax: key/value blocks of 16 staged in
     * threadgroup memory, scores and softmax statistics reduced across the threadgroup, each
     * thread accumulating its share of the output features. Launch with {@code qHeads} threadgroups
     * of {@link #ATTENTION_THREADS} threads; {@code headDim} at most 128.
     */
    @JitBaseline(value = "mlx_fast_scaled_dot_product_attention",
            source = "tornado-unittests/.../kernelcontext/api/TestFlashAttentionKernelContext.java#processHeadsFlashAttentionOptV2 (MLX layout, head dim 128, explicit scale)")
    public static void attentionDecode(KernelContext context, FloatArray q, FloatArray keys, FloatArray values, FloatArray out, int qHeads, int kvHeads, int kvLen, int headDim, float scale) {
        int tid = context.localIdx;
        int h = context.groupIdx;
        int localSize = context.localGroupSizeX;
        if (h >= qHeads) {
            return;
        }
        int kvHead = h / (qHeads / kvHeads);
        int kvBase = kvHead * kvLen * headDim;

        float[] qShared = context.allocateFloatLocalArray(MAX_HEAD_DIM);
        float[] kTile = context.allocateFloatLocalArray(KEY_BLOCK * MAX_HEAD_DIM);
        float[] vTile = context.allocateFloatLocalArray(KEY_BLOCK * MAX_HEAD_DIM);
        float[] sTile = context.allocateFloatLocalArray(KEY_BLOCK);
        float[] expTile = context.allocateFloatLocalArray(KEY_BLOCK);
        float[] reduction = context.allocateFloatLocalArray(ATTENTION_THREADS);
        float[] state = context.allocateFloatLocalArray(4);

        int dimsPerThread = (headDim + localSize - 1) / localSize;
        int myStartDim = tid * dimsPerThread;
        int myEndDim = Math.min(myStartDim + dimsPerThread, headDim);
        int myDimCount = myEndDim - myStartDim;
        float[] output = new float[MAX_HEAD_DIM / 8];
        for (int i = 0; i < myDimCount; i++) {
            output[i] = 0.0f;
        }
        if (tid == 0) {
            state[0] = Float.NEGATIVE_INFINITY;
            state[1] = 0.0f;
        }
        for (int i = tid; i < headDim; i += localSize) {
            qShared[i] = q.get(h * headDim + i);
        }
        context.localBarrier();

        for (int tileC = 0; tileC < kvLen; tileC += KEY_BLOCK) {
            int tileLen = Math.min(KEY_BLOCK, kvLen - tileC);
            int totalElements = tileLen * headDim;
            for (int e = tid; e < totalElements; e += localSize) {
                int seqIdx = e / headDim;
                int dimIdx = e % headDim;
                int offset = kvBase + (tileC + seqIdx) * headDim + dimIdx;
                kTile[e] = keys.get(offset);
                vTile[e] = values.get(offset);
            }
            context.localBarrier();

            for (int t = tid; t < tileLen; t += localSize) {
                float score = 0.0f;
                for (int d = 0; d < headDim; d++) {
                    score += qShared[d] * kTile[t * headDim + d];
                }
                sTile[t] = score * scale;
            }
            context.localBarrier();

            float threadMax = Float.NEGATIVE_INFINITY;
            for (int t = tid; t < tileLen; t += localSize) {
                threadMax = Math.max(threadMax, sTile[t]);
            }
            reduction[tid] = threadMax;
            context.localBarrier();
            for (int stride = localSize / 2; stride > 0; stride /= 2) {
                if (tid < stride) {
                    reduction[tid] = Math.max(reduction[tid], reduction[tid + stride]);
                }
                context.localBarrier();
            }
            float tileMax = reduction[0];

            float prevMax = state[0];
            float newMax = Math.max(prevMax, tileMax);
            float rescale = 1.0f;
            if (newMax != prevMax && prevMax != Float.NEGATIVE_INFINITY) {
                rescale = TornadoMath.exp(prevMax - newMax);
                for (int i = 0; i < myDimCount; i++) {
                    output[i] *= rescale;
                }
            }
            for (int t = tid; t < tileLen; t += localSize) {
                expTile[t] = TornadoMath.exp(sTile[t] - newMax);
            }
            context.localBarrier();

            float threadSum = 0.0f;
            for (int t = tid; t < tileLen; t += localSize) {
                threadSum += expTile[t];
            }
            reduction[tid] = threadSum;
            context.localBarrier();
            for (int stride = localSize / 2; stride > 0; stride /= 2) {
                if (tid < stride) {
                    reduction[tid] += reduction[tid + stride];
                }
                context.localBarrier();
            }
            float tileSum = reduction[0];
            if (tid == 0) {
                state[0] = newMax;
                state[1] = state[1] * rescale + tileSum;
            }
            context.localBarrier();

            for (int t = 0; t < tileLen; t++) {
                float expScore = expTile[t];
                for (int i = 0; i < myDimCount; i++) {
                    output[i] += expScore * vTile[t * headDim + myStartDim + i];
                }
            }
            context.localBarrier();
        }

        float sumExp = state[1];
        float norm = (sumExp > 0.0f) ? (1.0f / sumExp) : 0.0f;
        int outBase = h * headDim + myStartDim;
        for (int i = 0; i < myDimCount; i++) {
            out.set(outBase + i, output[i] * norm);
        }
    }
}
