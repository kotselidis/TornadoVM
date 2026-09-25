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
 * The kernels marked "jitLLM" below are adapted from beehive-lab/GPULlama3.java (jitLLM),
 * org.beehive.jitllm.backend.tornado.kernels, under the MIT License:
 *
 *   Copyright (c) 2025 Beehive lab
 *
 *   Permission is hereby granted, free of charge, to any person obtaining a copy of this software
 *   and associated documentation files (the "Software"), to deal in the Software without
 *   restriction, including without limitation the rights to use, copy, modify, merge, publish,
 *   distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
 *   Software is furnished to do so, subject to the following conditions: The above copyright notice
 *   and this permission notice shall be included in all copies or substantial portions of the
 *   Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 *   INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE
 *   AND NONINFRINGEMENT.
 *
 */
package uk.ac.manchester.tornado.mlx.benchmarks;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * TornadoVM JIT baselines for the MLX benchmarks. Kernels marked "jitLLM" are the tuned kernels
 * jitLLM's decode path runs on Metal, copied with only their names changed. The others are
 * straightforward kernels written for these benchmarks (jitLLM's rope and attention are fused with
 * its paged KV cache, and its GEMM uses CUDA tensor-core intrinsics), so they are a lower bound on
 * what a tuned JIT kernel would do.
 */
public final class JitKernels {

    private JitKernels() {
    }

    // ---------------------------------------------------------------- GEMV (jitLLM)

    /** jitLLM matrixVectorGenericSimd32: out[row] = w[row, :] . x, one 32-lane group per row. */
    public static void gemvF16Simd32(KernelContext context, HalfFloatArray x, FloatArray out, HalfFloatArray w, int n, int d) {
        int rowId = context.groupIdx;
        int localId = context.localIdx;
        if (rowId >= d) {
            return;
        }
        int rowOffset = rowId * n;
        float partialSum = 0.0f;
        for (int j = localId; j < n; j += 32) {
            partialSum += w.get(rowOffset + j).getFloat32() * x.get(j).getFloat32();
        }
        partialSum += context.simdShuffleDown(partialSum, 16);
        partialSum += context.simdShuffleDown(partialSum, 8);
        partialSum += context.simdShuffleDown(partialSum, 4);
        partialSum += context.simdShuffleDown(partialSum, 2);
        partialSum += context.simdShuffleDown(partialSum, 1);
        if (localId == 0) {
            out.set(rowId, partialSum);
        }
    }

    private static final int Q4_QK = 32;
    private static final int Q4_BLOCK_BYTES = 18;
    private static final int Q4_QS_OFFSET = 2;

    private static float decodeQ4(ByteArray w, int blockByteOffset, int withinBlock) {
        float d = w.getHalfFloat(blockByteOffset).getFloat32();
        int half = withinBlock / 16;
        int byteIndex = withinBlock - half * 16;
        int packed = w.get(blockByteOffset + Q4_QS_OFFSET + byteIndex) & 0xFF;
        int q = (half == 0) ? (packed & 0xF) : ((packed >> 4) & 0xF);
        return d * (q - 8);
    }

    /** jitLLM matrixVectorGenericQ4_0Simd32: GGUF Q4_0 weights (18-byte blocks of 32). */
    public static void gemvQ4Simd32(KernelContext context, FloatArray x, FloatArray out, ByteArray w, int n, int d) {
        int rowId = context.groupIdx;
        if (rowId >= d) {
            return;
        }
        int localId = context.localIdx;
        int blocksPerRow = (n + Q4_QK - 1) / Q4_QK;
        int rowBlockOffset = rowId * blocksPerRow;
        float partialSum = 0.0f;
        for (int j = localId; j < n; j += 32) {
            int blockIdx = j / Q4_QK;
            int withinBlock = j - blockIdx * Q4_QK;
            int blockByteOffset = (rowBlockOffset + blockIdx) * Q4_BLOCK_BYTES;
            partialSum += decodeQ4(w, blockByteOffset, withinBlock) * x.get(j);
        }
        partialSum += context.simdShuffleDown(partialSum, 16);
        partialSum += context.simdShuffleDown(partialSum, 8);
        partialSum += context.simdShuffleDown(partialSum, 4);
        partialSum += context.simdShuffleDown(partialSum, 2);
        partialSum += context.simdShuffleDown(partialSum, 1);
        if (localId == 0) {
            out.set(rowId, partialSum);
        }
    }

    /** jitLLM matrixVectorGenericQ8Byte: GGUF Q8_0 weights (34-byte blocks of 32), tree reduction. */
    public static void gemvQ8(KernelContext context, FloatArray x, FloatArray out, ByteArray q, int n, int d, int localSize) {
        int rowId = context.groupIdx;
        int localId = context.localIdx;
        if (rowId >= d) {
            return;
        }
        final int blockSize = 32;
        final int blockBytes = 34;
        float[] localSums = context.allocateFloatLocalArray(localSize);
        int blocksPerRow = (n + blockSize - 1) / blockSize;
        int rowBlockOffset = rowId * blocksPerRow;
        float partialSum1 = 0.0f;
        float partialSum2 = 0.0f;
        float partialSum3 = 0.0f;
        float partialSum4 = 0.0f;
        for (int j = localId * 4; j < n - 3; j += localSize * 4) {
            int blockIdx = j / blockSize;
            int withinBlockIdx = j % blockSize;
            int blockByteOffset = (rowBlockOffset + blockIdx) * blockBytes;
            float scale = q.getHalfFloat(blockByteOffset).getFloat32();
            int quantsOffset = blockByteOffset + 2 + withinBlockIdx;
            partialSum1 += ((float) q.get(quantsOffset) * scale) * x.get(j);
            partialSum2 += ((float) q.get(quantsOffset + 1) * scale) * x.get(j + 1);
            partialSum3 += ((float) q.get(quantsOffset + 2) * scale) * x.get(j + 2);
            partialSum4 += ((float) q.get(quantsOffset + 3) * scale) * x.get(j + 3);
        }
        localSums[localId] = partialSum1 + partialSum2 + partialSum3 + partialSum4;
        context.localBarrier();
        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSums[localId] += localSums[localId + stride];
            }
            context.localBarrier();
        }
        if (localId == 0) {
            out.set(rowId, localSums[0]);
        }
    }

    // ---------------------------------------------------------------- RMS norm (jitLLM: two kernels)

    /** jitLLM reductionOneBlockWithLayerSingleGroup: temp[0] = 1 / sqrt(mean(x^2) + eps). */
    public static void rmsReduce(KernelContext context, FloatArray temp, FloatArray x, int size, float eps, int localMemSize) {
        int lid = context.localIdx;
        int groupSize = context.localGroupSizeX;
        float[] localX = context.allocateFloatLocalArray(localMemSize);
        float partial = 0.0f;
        for (int j = lid; j < size; j += groupSize) {
            float v = x.get(j);
            partial += v * v;
        }
        localX[lid] = partial;
        for (int stride = groupSize / 2; stride > 0; stride /= 2) {
            context.localBarrier();
            if (lid < stride) {
                localX[lid] += localX[lid + stride];
            }
        }
        if (lid == 0) {
            float ss = localX[0] / size + eps;
            temp.set(0, 1.0f / TornadoMath.sqrt(ss));
        }
    }

    /** jitLLM reductionOneBlock2WithLayer: out[i] = weight[i] * (temp[0] * x[i]). */
    public static void rmsApply(KernelContext context, FloatArray out, FloatArray x, FloatArray weights, FloatArray temp) {
        int gid = context.globalIdx;
        float ss = temp.get(0);
        out.set(gid, weights.get(gid) * (ss * x.get(gid)));
    }

    // ---------------------------------------------------------------- argmax (jitLLM)

    /** jitLLM argmaxLogits: one workgroup scans the vocabulary. */
    public static void argmax(KernelContext context, FloatArray logits, IntArray out, int vocab, int localMemSize) {
        int tid = context.localIdx;
        int localSz = context.localGroupSizeX;
        float[] vals = context.allocateFloatLocalArray(localMemSize);
        int[] idxs = context.allocateIntLocalArray(localMemSize);
        float best = Float.NEGATIVE_INFINITY;
        int bestIdx = 0;
        for (int i = tid; i < vocab; i += localSz) {
            float v = logits.get(i);
            if (v > best) {
                best = v;
                bestIdx = i;
            }
        }
        vals[tid] = best;
        idxs[tid] = bestIdx;
        context.localBarrier();
        for (int s = localSz / 2; s > 0; s >>= 1) {
            if (tid < s) {
                if (vals[tid + s] > vals[tid]) {
                    vals[tid] = vals[tid + s];
                    idxs[tid] = idxs[tid + s];
                }
            }
            context.localBarrier();
        }
        if (tid == 0) {
            out.set(0, idxs[0]);
        }
    }

    // ---------------------------------------------------------------- straightforward kernels (not jitLLM)

    static final int TILE = 16;

    /** Tiled GEMM c[m, n] = a[m, k] @ w[n, k]^T with 16x16 local-memory tiles; float accumulation. */
    public static void gemmTiled(KernelContext context, HalfFloatArray a, HalfFloatArray w, FloatArray c, int m, int k, int n) {
        int row = context.globalIdy;
        int col = context.globalIdx;
        int ly = context.localIdy;
        int lx = context.localIdx;
        float[] aTile = context.allocateFloatLocalArray(TILE * TILE);
        float[] wTile = context.allocateFloatLocalArray(TILE * TILE);
        float acc = 0.0f;
        for (int t = 0; t < k; t += TILE) {
            aTile[ly * TILE + lx] = (row < m && t + lx < k) ? a.get(row * k + t + lx).getFloat32() : 0.0f;
            int wRow = context.groupIdx * TILE + ly;
            wTile[ly * TILE + lx] = (wRow < n && t + lx < k) ? w.get(wRow * k + t + lx).getFloat32() : 0.0f;
            context.localBarrier();
            for (int p = 0; p < TILE; p++) {
                acc += aTile[ly * TILE + p] * wTile[lx * TILE + p];
            }
            context.localBarrier();
        }
        if (row < m && col < n) {
            c.set(row * n + col, acc);
        }
    }

    /**
     * Decode attention, one workgroup of headDim threads per query head: for each key, a
     * threadgroup-reduced dot product, then an online-softmax update of each thread's output lane.
     * q[qHeads, headDim], k and v[kvHeads, ctx, headDim].
     */
    public static void attentionDecode(KernelContext context, FloatArray q, FloatArray k, FloatArray v, FloatArray out, int qHeads, int kvHeads, int ctx, int headDim, float scale) {
        int h = context.groupIdx;
        int d = context.localIdx;
        float[] partial = context.allocateFloatLocalArray(128);
        int kh = h / (qHeads / kvHeads);
        float qd = q.get(h * headDim + d);
        float max = Float.NEGATIVE_INFINITY;
        float sum = 0.0f;
        float acc = 0.0f;
        for (int j = 0; j < ctx; j++) {
            int kvBase = (kh * ctx + j) * headDim;
            partial[d] = qd * k.get(kvBase + d);
            for (int stride = headDim / 2; stride > 0; stride >>= 1) {
                context.localBarrier();
                if (d < stride) {
                    partial[d] += partial[d + stride];
                }
            }
            context.localBarrier();
            float score = partial[0] * scale;
            float newMax = TornadoMath.max(max, score);
            float correction = TornadoMath.exp(max - newMax);
            float weight = TornadoMath.exp(score - newMax);
            sum = sum * correction + weight;
            acc = acc * correction + weight * v.get(kvBase + d);
            max = newMax;
            context.localBarrier();
        }
        out.set(h * headDim + d, acc / sum);
    }

    /** Softmax over the whole array with one workgroup: max, sum of exponentials, normalise. */
    public static void softmax(KernelContext context, FloatArray x, FloatArray out, int n, int localMemSize) {
        int tid = context.localIdx;
        int localSz = context.localGroupSizeX;
        float[] red = context.allocateFloatLocalArray(localMemSize);
        float max = Float.NEGATIVE_INFINITY;
        for (int i = tid; i < n; i += localSz) {
            max = TornadoMath.max(max, x.get(i));
        }
        red[tid] = max;
        context.localBarrier();
        for (int s = localSz / 2; s > 0; s >>= 1) {
            if (tid < s) {
                red[tid] = TornadoMath.max(red[tid], red[tid + s]);
            }
            context.localBarrier();
        }
        float globalMax = red[0];
        context.localBarrier();
        float sum = 0.0f;
        for (int i = tid; i < n; i += localSz) {
            sum += TornadoMath.exp(x.get(i) - globalMax);
        }
        red[tid] = sum;
        context.localBarrier();
        for (int s = localSz / 2; s > 0; s >>= 1) {
            if (tid < s) {
                red[tid] += red[tid + s];
            }
            context.localBarrier();
        }
        float inv = 1.0f / red[0];
        for (int i = tid; i < n; i += localSz) {
            out.set(i, TornadoMath.exp(x.get(i) - globalMax) * inv);
        }
    }
}
