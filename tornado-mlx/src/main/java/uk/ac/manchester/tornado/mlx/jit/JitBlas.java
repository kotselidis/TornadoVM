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
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.matrix.Matrix8x8Float;

/**
 * JIT counterparts of the MLX matrix operations.
 *
 * <p>
 * GEMM kernels are TornadoVM's fastest Metal GEMM, {@code gemmTiled} from
 * {@code tornado-examples/.../compute/MatrixMultiplySimdgroup.java}: one threadgroup of 128 threads
 * (four SIMD groups) per 32x32 output tile, 32x8 / 8x32 blocks of the operands staged in
 * threadgroup memory and multiplied with 8x8 simdgroup matrices. {@code m} and {@code n} must be
 * multiples of 32 and {@code k} a multiple of 8. Launch with {@code (m / 32) * (n / 32) * 128}
 * threads, 128 per group. The variants differ only in how the operand blocks are staged.
 * </p>
 *
 * <p>
 * GEMV kernels (the {@code m = 1} decode case) are {@code matvecSimd} from
 * {@code tornado-examples/.../compute/MatrixVectorSimdReduction.java}: one 32-lane SIMD group per
 * output row, reduced with {@code simd_sum}. Launch with {@code rows * 32} threads, 32 per group.
 * </p>
 */
public final class JitBlas {

    /** Output tile edge of the GEMM kernels. */
    public static final int GEMM_BLOCK = 32;
    /** Threads per GEMM threadgroup. */
    public static final int GEMM_THREADS = 128;

    private static final int TILE = 8;
    private static final int SIMD_GROUP = 32;

    private JitBlas() {
    }

    /** c[m, n] = a[m, k] @ b[k, n]. */
    @JitBaseline(value = { "mlx_matmul", "mlx_tensordot", "mlx_tensordot_axis" }, source = "tornado-examples/.../compute/MatrixMultiplySimdgroup.java#gemmTiled")
    public static void gemm(KernelContext ctx, FloatArray a, FloatArray b, FloatArray c, int m, int n, int k) {
        float[] as = ctx.allocateFloatLocalArray(256);
        float[] bs = ctx.allocateFloatLocalArray(256);
        int tilesPerRow = n / GEMM_BLOCK;
        int rowBase = (ctx.groupIdx / tilesPerRow) * GEMM_BLOCK;
        int colBase = (ctx.groupIdx % tilesPerRow) * GEMM_BLOCK;
        int tid = ctx.localIdx;
        int sgRow = (tid / SIMD_GROUP) / 2;
        int sgCol = (tid / SIMD_GROUP) % 2;
        Matrix8x8Float acc00 = ctx.simdgroupMatrixZero();
        Matrix8x8Float acc01 = ctx.simdgroupMatrixZero();
        Matrix8x8Float acc10 = ctx.simdgroupMatrixZero();
        Matrix8x8Float acc11 = ctx.simdgroupMatrixZero();
        for (int kb = 0; kb < k; kb += TILE) {
            for (int e = tid; e < 256; e += GEMM_THREADS) {
                as[e] = a.get((rowBase + e / TILE) * k + (kb + e % TILE));
            }
            for (int e = tid; e < 256; e += GEMM_THREADS) {
                bs[e] = b.get((kb + e / GEMM_BLOCK) * n + (colBase + e % GEMM_BLOCK));
            }
            ctx.localBarrier();
            Matrix8x8Float a0 = ctx.simdgroupMatrixLoad(as, (sgRow * 2) * 64, TILE);
            Matrix8x8Float a1 = ctx.simdgroupMatrixLoad(as, (sgRow * 2 + 1) * 64, TILE);
            Matrix8x8Float b0 = ctx.simdgroupMatrixLoad(bs, (sgCol * 2) * TILE, GEMM_BLOCK);
            Matrix8x8Float b1 = ctx.simdgroupMatrixLoad(bs, (sgCol * 2 + 1) * TILE, GEMM_BLOCK);
            acc00 = ctx.simdgroupMatrixMultiplyAccumulate(a0, b0, acc00);
            acc01 = ctx.simdgroupMatrixMultiplyAccumulate(a0, b1, acc01);
            acc10 = ctx.simdgroupMatrixMultiplyAccumulate(a1, b0, acc10);
            acc11 = ctx.simdgroupMatrixMultiplyAccumulate(a1, b1, acc11);
            ctx.localBarrier();
        }
        int cr0 = rowBase + (sgRow * 2) * TILE;
        int cr1 = rowBase + (sgRow * 2 + 1) * TILE;
        int cc0 = colBase + (sgCol * 2) * TILE;
        int cc1 = colBase + (sgCol * 2 + 1) * TILE;
        ctx.simdgroupMatrixStore(acc00, c, cr0 * n + cc0, n);
        ctx.simdgroupMatrixStore(acc01, c, cr0 * n + cc1, n);
        ctx.simdgroupMatrixStore(acc10, c, cr1 * n + cc0, n);
        ctx.simdgroupMatrixStore(acc11, c, cr1 * n + cc1, n);
    }

    /** c[m, n] = a[m, k] @ w[n, k]^T (weights stored one row per output, as in LLM checkpoints). */
    @JitBaseline(value = { "mlx_matmul", "mlx_transpose" }, source = "tornado-examples/.../compute/MatrixMultiplySimdgroup.java#gemmTiled (transposed B staging)")
    public static void gemmTransposed(KernelContext ctx, FloatArray a, FloatArray w, FloatArray c, int m, int n, int k) {
        float[] as = ctx.allocateFloatLocalArray(256);
        float[] bs = ctx.allocateFloatLocalArray(256);
        int tilesPerRow = n / GEMM_BLOCK;
        int rowBase = (ctx.groupIdx / tilesPerRow) * GEMM_BLOCK;
        int colBase = (ctx.groupIdx % tilesPerRow) * GEMM_BLOCK;
        int tid = ctx.localIdx;
        int sgRow = (tid / SIMD_GROUP) / 2;
        int sgCol = (tid / SIMD_GROUP) % 2;
        Matrix8x8Float acc00 = ctx.simdgroupMatrixZero();
        Matrix8x8Float acc01 = ctx.simdgroupMatrixZero();
        Matrix8x8Float acc10 = ctx.simdgroupMatrixZero();
        Matrix8x8Float acc11 = ctx.simdgroupMatrixZero();
        for (int kb = 0; kb < k; kb += TILE) {
            for (int e = tid; e < 256; e += GEMM_THREADS) {
                as[e] = a.get((rowBase + e / TILE) * k + (kb + e % TILE));
            }
            // bs is the 8x32 block B[kb .. kb+7, colBase .. colBase+31] with B = w^T.
            for (int e = tid; e < 256; e += GEMM_THREADS) {
                bs[e] = w.get((colBase + e % GEMM_BLOCK) * k + (kb + e / GEMM_BLOCK));
            }
            ctx.localBarrier();
            Matrix8x8Float a0 = ctx.simdgroupMatrixLoad(as, (sgRow * 2) * 64, TILE);
            Matrix8x8Float a1 = ctx.simdgroupMatrixLoad(as, (sgRow * 2 + 1) * 64, TILE);
            Matrix8x8Float b0 = ctx.simdgroupMatrixLoad(bs, (sgCol * 2) * TILE, GEMM_BLOCK);
            Matrix8x8Float b1 = ctx.simdgroupMatrixLoad(bs, (sgCol * 2 + 1) * TILE, GEMM_BLOCK);
            acc00 = ctx.simdgroupMatrixMultiplyAccumulate(a0, b0, acc00);
            acc01 = ctx.simdgroupMatrixMultiplyAccumulate(a0, b1, acc01);
            acc10 = ctx.simdgroupMatrixMultiplyAccumulate(a1, b0, acc10);
            acc11 = ctx.simdgroupMatrixMultiplyAccumulate(a1, b1, acc11);
            ctx.localBarrier();
        }
        int cr0 = rowBase + (sgRow * 2) * TILE;
        int cr1 = rowBase + (sgRow * 2 + 1) * TILE;
        int cc0 = colBase + (sgCol * 2) * TILE;
        int cc1 = colBase + (sgCol * 2 + 1) * TILE;
        ctx.simdgroupMatrixStore(acc00, c, cr0 * n + cc0, n);
        ctx.simdgroupMatrixStore(acc01, c, cr0 * n + cc1, n);
        ctx.simdgroupMatrixStore(acc10, c, cr1 * n + cc0, n);
        ctx.simdgroupMatrixStore(acc11, c, cr1 * n + cc1, n);
    }

    /**
     * {@link #gemmTransposed} with float16 operands, converted to float while staging; float
     * accumulation and output (the simdgroup-matrix API is float).
     */
    @JitBaseline(value = { "mlx_matmul", "mlx_transpose" }, source = "tornado-examples/.../compute/MatrixMultiplySimdgroup.java#gemmTiled (float16 operands, transposed B staging)")
    public static void gemmTransposedF16(KernelContext ctx, HalfFloatArray a, HalfFloatArray w, FloatArray c, int m, int n, int k) {
        float[] as = ctx.allocateFloatLocalArray(256);
        float[] bs = ctx.allocateFloatLocalArray(256);
        int tilesPerRow = n / GEMM_BLOCK;
        int rowBase = (ctx.groupIdx / tilesPerRow) * GEMM_BLOCK;
        int colBase = (ctx.groupIdx % tilesPerRow) * GEMM_BLOCK;
        int tid = ctx.localIdx;
        int sgRow = (tid / SIMD_GROUP) / 2;
        int sgCol = (tid / SIMD_GROUP) % 2;
        Matrix8x8Float acc00 = ctx.simdgroupMatrixZero();
        Matrix8x8Float acc01 = ctx.simdgroupMatrixZero();
        Matrix8x8Float acc10 = ctx.simdgroupMatrixZero();
        Matrix8x8Float acc11 = ctx.simdgroupMatrixZero();
        for (int kb = 0; kb < k; kb += TILE) {
            for (int e = tid; e < 256; e += GEMM_THREADS) {
                as[e] = a.get((rowBase + e / TILE) * k + (kb + e % TILE)).getFloat32();
            }
            for (int e = tid; e < 256; e += GEMM_THREADS) {
                bs[e] = w.get((colBase + e % GEMM_BLOCK) * k + (kb + e / GEMM_BLOCK)).getFloat32();
            }
            ctx.localBarrier();
            Matrix8x8Float a0 = ctx.simdgroupMatrixLoad(as, (sgRow * 2) * 64, TILE);
            Matrix8x8Float a1 = ctx.simdgroupMatrixLoad(as, (sgRow * 2 + 1) * 64, TILE);
            Matrix8x8Float b0 = ctx.simdgroupMatrixLoad(bs, (sgCol * 2) * TILE, GEMM_BLOCK);
            Matrix8x8Float b1 = ctx.simdgroupMatrixLoad(bs, (sgCol * 2 + 1) * TILE, GEMM_BLOCK);
            acc00 = ctx.simdgroupMatrixMultiplyAccumulate(a0, b0, acc00);
            acc01 = ctx.simdgroupMatrixMultiplyAccumulate(a0, b1, acc01);
            acc10 = ctx.simdgroupMatrixMultiplyAccumulate(a1, b0, acc10);
            acc11 = ctx.simdgroupMatrixMultiplyAccumulate(a1, b1, acc11);
            ctx.localBarrier();
        }
        int cr0 = rowBase + (sgRow * 2) * TILE;
        int cr1 = rowBase + (sgRow * 2 + 1) * TILE;
        int cc0 = colBase + (sgCol * 2) * TILE;
        int cc1 = colBase + (sgCol * 2 + 1) * TILE;
        ctx.simdgroupMatrixStore(acc00, c, cr0 * n + cc0, n);
        ctx.simdgroupMatrixStore(acc01, c, cr0 * n + cc1, n);
        ctx.simdgroupMatrixStore(acc10, c, cr1 * n + cc0, n);
        ctx.simdgroupMatrixStore(acc11, c, cr1 * n + cc1, n);
    }

    /**
     * out[m, n] = alpha * a[m, k] @ b[k, n] + beta * cIn[m, n]: {@link #gemm}, then an epilogue in
     * which each threadgroup rescales its own 32x32 tile after a device-memory barrier.
     */
    @JitBaseline(value = "mlx_addmm", source = "tornado-examples/.../compute/MatrixMultiplySimdgroup.java#gemmTiled (alpha/beta epilogue)")
    public static void addmm(KernelContext ctx, FloatArray cIn, FloatArray a, FloatArray b, FloatArray out, int m, int n, int k, float alpha, float beta) {
        float[] as = ctx.allocateFloatLocalArray(256);
        float[] bs = ctx.allocateFloatLocalArray(256);
        int tilesPerRow = n / GEMM_BLOCK;
        int rowBase = (ctx.groupIdx / tilesPerRow) * GEMM_BLOCK;
        int colBase = (ctx.groupIdx % tilesPerRow) * GEMM_BLOCK;
        int tid = ctx.localIdx;
        int sgRow = (tid / SIMD_GROUP) / 2;
        int sgCol = (tid / SIMD_GROUP) % 2;
        Matrix8x8Float acc00 = ctx.simdgroupMatrixZero();
        Matrix8x8Float acc01 = ctx.simdgroupMatrixZero();
        Matrix8x8Float acc10 = ctx.simdgroupMatrixZero();
        Matrix8x8Float acc11 = ctx.simdgroupMatrixZero();
        for (int kb = 0; kb < k; kb += TILE) {
            for (int e = tid; e < 256; e += GEMM_THREADS) {
                as[e] = a.get((rowBase + e / TILE) * k + (kb + e % TILE));
            }
            for (int e = tid; e < 256; e += GEMM_THREADS) {
                bs[e] = b.get((kb + e / GEMM_BLOCK) * n + (colBase + e % GEMM_BLOCK));
            }
            ctx.localBarrier();
            Matrix8x8Float a0 = ctx.simdgroupMatrixLoad(as, (sgRow * 2) * 64, TILE);
            Matrix8x8Float a1 = ctx.simdgroupMatrixLoad(as, (sgRow * 2 + 1) * 64, TILE);
            Matrix8x8Float b0 = ctx.simdgroupMatrixLoad(bs, (sgCol * 2) * TILE, GEMM_BLOCK);
            Matrix8x8Float b1 = ctx.simdgroupMatrixLoad(bs, (sgCol * 2 + 1) * TILE, GEMM_BLOCK);
            acc00 = ctx.simdgroupMatrixMultiplyAccumulate(a0, b0, acc00);
            acc01 = ctx.simdgroupMatrixMultiplyAccumulate(a0, b1, acc01);
            acc10 = ctx.simdgroupMatrixMultiplyAccumulate(a1, b0, acc10);
            acc11 = ctx.simdgroupMatrixMultiplyAccumulate(a1, b1, acc11);
            ctx.localBarrier();
        }
        int cr0 = rowBase + (sgRow * 2) * TILE;
        int cr1 = rowBase + (sgRow * 2 + 1) * TILE;
        int cc0 = colBase + (sgCol * 2) * TILE;
        int cc1 = colBase + (sgCol * 2 + 1) * TILE;
        ctx.simdgroupMatrixStore(acc00, out, cr0 * n + cc0, n);
        ctx.simdgroupMatrixStore(acc01, out, cr0 * n + cc1, n);
        ctx.simdgroupMatrixStore(acc10, out, cr1 * n + cc0, n);
        ctx.simdgroupMatrixStore(acc11, out, cr1 * n + cc1, n);
        ctx.globalBarrier();
        for (int e = tid; e < GEMM_BLOCK * GEMM_BLOCK; e += GEMM_THREADS) {
            int idx = (rowBase + e / GEMM_BLOCK) * n + colBase + e % GEMM_BLOCK;
            out.set(idx, alpha * out.get(idx) + beta * cIn.get(idx));
        }
    }

    /** out[row] = w[row, :] . x (float32), one SIMD group per row. */
    @JitBaseline(value = "mlx_matmul", source = "tornado-examples/.../compute/MatrixVectorSimdReduction.java#matvecSimd")
    public static void gemv(KernelContext ctx, FloatArray x, FloatArray w, FloatArray out, int n) {
        int rowId = ctx.groupIdx;
        int localId = ctx.localIdx;
        int localSize = ctx.localGroupSizeX;
        float partial = 0.0f;
        int rowOffset = rowId * n;
        for (int j = localId; j < n; j += localSize) {
            partial += w.get(rowOffset + j) * x.get(j);
        }
        float rowSum = ctx.simdSum(partial);
        if (localId == 0) {
            out.set(rowId, rowSum);
        }
    }

    /** {@link #gemv} with float16 weights and activations, float accumulation and output. */
    @JitBaseline(value = { "mlx_matmul", "mlx_transpose" }, source = "tornado-examples/.../compute/MatrixVectorSimdReduction.java#matvecSimd (float16 operands)")
    public static void gemvF16(KernelContext ctx, HalfFloatArray x, HalfFloatArray w, FloatArray out, int n) {
        int rowId = ctx.groupIdx;
        int localId = ctx.localIdx;
        int localSize = ctx.localGroupSizeX;
        float partial = 0.0f;
        int rowOffset = rowId * n;
        for (int j = localId; j < n; j += localSize) {
            partial += w.get(rowOffset + j).getFloat32() * x.get(j).getFloat32();
        }
        float rowSum = ctx.simdSum(partial);
        if (localId == 0) {
            out.set(rowId, rowSum);
        }
    }

    /** out[cols, rows] = in[rows, cols]^T through 32x32 threadgroup-memory tiles (padded to avoid bank conflicts). */
    @JitBaseline("mlx_transpose")
    public static void transpose(KernelContext ctx, FloatArray in, FloatArray out, int rows, int cols) {
        float[] tile = ctx.allocateFloatLocalArray(32 * 33);
        int tx = ctx.localIdx;
        int ty = ctx.localIdy;
        int c = ctx.groupIdx * 32 + tx;
        int r = ctx.groupIdy * 32 + ty;
        if (r < rows && c < cols) {
            tile[ty * 33 + tx] = in.get(r * cols + c);
        }
        ctx.localBarrier();
        int oc = ctx.groupIdy * 32 + tx;
        int or = ctx.groupIdx * 32 + ty;
        if (or < cols && oc < rows) {
            out.set(or * rows + oc, tile[tx * 33 + ty]);
        }
    }
}
