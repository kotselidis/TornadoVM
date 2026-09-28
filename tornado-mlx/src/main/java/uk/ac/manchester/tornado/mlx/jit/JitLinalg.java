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
 * JIT counterparts of the MLX linear-algebra operations ({@link uk.ac.manchester.tornado.mlx.MlxLinalg}),
 * float32. MLX runs most of these on its CPU stream (LAPACK); the JIT kernels run on the GPU over
 * batches of small matrices: one threadgroup of {@link #THREADS} threads per matrix, the matrix
 * held in threadgroup memory, for {@code n <= } {@link #MAX_N}. The vector operations (cross,
 * norms) are one thread or one threadgroup per output.
 */
public final class JitLinalg {

    /** Threads per threadgroup. */
    public static final int THREADS = 256;
    /** Largest matrix order the per-matrix kernels hold in threadgroup memory. */
    public static final int MAX_N = 32;

    private static final int MAX_NN = MAX_N * MAX_N;
    private static final int MAX_AUG = 2 * MAX_NN;

    private JitLinalg() {
    }

    // ---------------------------------------------------------------- vectors and norms

    /** out[i] = a[i] x b[i] for count 3-vectors; one thread per vector. */
    @JitBaseline("mlx_linalg_cross")
    public static void crossProduct(KernelContext ctx, FloatArray a, FloatArray b, FloatArray out, int count) {
        int i = ctx.globalIdx;
        if (i < count) {
            int p = 3 * i;
            float a0 = a.get(p);
            float a1 = a.get(p + 1);
            float a2 = a.get(p + 2);
            float b0 = b.get(p);
            float b1 = b.get(p + 1);
            float b2 = b.get(p + 2);
            out.set(p, a1 * b2 - a2 * b1);
            out.set(p + 1, a2 * b0 - a0 * b2);
            out.set(p + 2, a0 * b1 - a1 * b0);
        }
    }

    /**
     * out[row] = the {@code ord}-norm of x[row, :]: sum |x|^ord ^ (1/ord), with ord = +/-infinity
     * giving max/min |x| and ord = 0 the count of non-zeros. One threadgroup per row; used with
     * len = rows * cols for the Frobenius norm of a matrix.
     */
    @JitBaseline({ "mlx_linalg_norm", "mlx_linalg_norm_l2", "mlx_linalg_norm_matrix" })
    public static void normRows(KernelContext ctx, FloatArray x, FloatArray out, int len, float ord) {
        float[] red = ctx.allocateFloatLocalArray(THREADS);
        int row = ctx.groupIdx;
        int tid = ctx.localIdx;
        int base = row * len;
        boolean isMax = ord > 3.0e38f;
        boolean isMin = ord < -3.0e38f;
        float acc = isMin ? Float.MAX_VALUE : 0.0f;
        for (int i = tid; i < len; i += THREADS) {
            float v = TornadoMath.abs(x.get(base + i));
            if (isMax) {
                acc = TornadoMath.max(acc, v);
            } else if (isMin) {
                acc = TornadoMath.min(acc, v);
            } else if (ord == 0.0f) {
                acc += v != 0.0f ? 1.0f : 0.0f;
            } else if (ord == 1.0f) {
                acc += v;
            } else if (ord == 2.0f) {
                acc += v * v;
            } else if (v != 0.0f) {
                acc += TornadoMath.exp(ord * TornadoMath.log(v));
            }
        }
        red[tid] = acc;
        ctx.localBarrier();
        for (int s = THREADS / 2; s > 0; s >>= 1) {
            if (tid < s) {
                if (isMax) {
                    red[tid] = TornadoMath.max(red[tid], red[tid + s]);
                } else if (isMin) {
                    red[tid] = TornadoMath.min(red[tid], red[tid + s]);
                } else {
                    red[tid] += red[tid + s];
                }
            }
            ctx.localBarrier();
        }
        if (tid == 0) {
            float r = red[0];
            if (ord == 2.0f) {
                r = TornadoMath.sqrt(r);
            } else if (!isMax && !isMin && ord != 0.0f && ord != 1.0f && r != 0.0f) {
                r = TornadoMath.exp(TornadoMath.log(r) / ord);
            }
            out.set(row, r);
        }
    }

    // ---------------------------------------------------------------- Cholesky and triangular inverses

    /** Cholesky factor of a[m] ([batch, n, n], symmetric positive definite), lower (or upper = L^T). */
    @JitBaseline("mlx_linalg_cholesky")
    public static void cholesky(KernelContext ctx, FloatArray a, FloatArray out, int n, int upper) {
        float[] l = ctx.allocateFloatLocalArray(MAX_NN);
        int tid = ctx.localIdx;
        int base = ctx.groupIdx * n * n;
        for (int e = tid; e < n * n; e += THREADS) {
            l[e] = a.get(base + e);
        }
        ctx.localBarrier();
        for (int j = 0; j < n; j++) {
            if (tid == 0) {
                float d = l[j * n + j];
                for (int k = 0; k < j; k++) {
                    d -= l[j * n + k] * l[j * n + k];
                }
                l[j * n + j] = TornadoMath.sqrt(d);
            }
            ctx.localBarrier();
            float djj = l[j * n + j];
            for (int i = j + 1 + tid; i < n; i += THREADS) {
                float s = l[i * n + j];
                for (int k = 0; k < j; k++) {
                    s -= l[i * n + k] * l[j * n + k];
                }
                l[i * n + j] = s / djj;
            }
            ctx.localBarrier();
        }
        for (int e = tid; e < n * n; e += THREADS) {
            int i = e / n;
            int j = e % n;
            float v = upper != 0 ? (i <= j ? l[j * n + i] : 0.0f) : (j <= i ? l[e] : 0.0f);
            out.set(base + e, v);
        }
    }

    /** Inverse of each triangular a[m] ([batch, n, n]; lower, or upper). */
    @JitBaseline("mlx_linalg_tri_inv")
    public static void triInv(KernelContext ctx, FloatArray a, FloatArray out, int n, int upper) {
        float[] t = ctx.allocateFloatLocalArray(MAX_NN);
        float[] inv = ctx.allocateFloatLocalArray(MAX_NN);
        int tid = ctx.localIdx;
        int base = ctx.groupIdx * n * n;
        for (int e = tid; e < n * n; e += THREADS) {
            t[e] = a.get(base + e);
        }
        ctx.localBarrier();
        // Triangular inverse, one column per thread by substitution.

        for (int c = tid; c < n; c += THREADS) {
            if (upper == 0) {
                for (int i = 0; i < n; i++) {
                    float s = i == c ? 1.0f : 0.0f;
                    for (int k = c; k < i; k++) {
                        s -= t[i * n + k] * inv[k * n + c];
                    }
                    inv[i * n + c] = i < c ? 0.0f : s / t[i * n + i];
                }
            } else {
                for (int i = n - 1; i >= 0; i--) {
                    float s = i == c ? 1.0f : 0.0f;
                    for (int k = i + 1; k <= c; k++) {
                        s -= t[i * n + k] * inv[k * n + c];
                    }
                    inv[i * n + c] = i > c ? 0.0f : s / t[i * n + i];
                }
            }
        }
        ctx.localBarrier();
        for (int e = tid; e < n * n; e += THREADS) {
            out.set(base + e, inv[e]);
        }
    }

    /**
     * (L L^T)^-1 from each Cholesky factor L = a[m] (or (U^T U)^-1 from an upper factor): the
     * triangular inverse, then its Gram product.
     */
    @JitBaseline("mlx_linalg_cholesky_inv")
    public static void choleskyInv(KernelContext ctx, FloatArray a, FloatArray out, int n, int upper) {
        float[] t = ctx.allocateFloatLocalArray(MAX_NN);
        float[] inv = ctx.allocateFloatLocalArray(MAX_NN);
        int tid = ctx.localIdx;
        int base = ctx.groupIdx * n * n;
        for (int e = tid; e < n * n; e += THREADS) {
            t[e] = a.get(base + e);
        }
        ctx.localBarrier();
        // Triangular inverse, one column per thread by substitution.

        for (int c = tid; c < n; c += THREADS) {
            if (upper == 0) {
                for (int i = 0; i < n; i++) {
                    float s = i == c ? 1.0f : 0.0f;
                    for (int k = c; k < i; k++) {
                        s -= t[i * n + k] * inv[k * n + c];
                    }
                    inv[i * n + c] = i < c ? 0.0f : s / t[i * n + i];
                }
            } else {
                for (int i = n - 1; i >= 0; i--) {
                    float s = i == c ? 1.0f : 0.0f;
                    for (int k = i + 1; k <= c; k++) {
                        s -= t[i * n + k] * inv[k * n + c];
                    }
                    inv[i * n + c] = i > c ? 0.0f : s / t[i * n + i];
                }
            }
        }
        ctx.localBarrier();
        for (int e = tid; e < n * n; e += THREADS) {
            int i = e / n;
            int j = e % n;
            float s = 0.0f;
            for (int k = 0; k < n; k++) {
                // Lower: (L L^T)^-1 = L^-T L^-1; upper: (U^T U)^-1 = U^-1 U^-T.
                s += upper != 0 ? inv[i * n + k] * inv[j * n + k] : inv[k * n + i] * inv[k * n + j];
            }
            out.set(base + e, s);
        }
    }

    // ---------------------------------------------------------------- inverse and solves

    /** Inverse of each a[m] ([batch, n, n]) by Gauss-Jordan elimination on [A | I]. */
    @JitBaseline("mlx_linalg_inv")
    public static void inv(KernelContext ctx, FloatArray a, FloatArray out, int n) {
        float[] m = ctx.allocateFloatLocalArray(MAX_AUG);
        int[] pivot = ctx.allocateIntLocalArray(1);
        int tid = ctx.localIdx;
        int base = ctx.groupIdx * n * n;
        int w = 2 * n;
        for (int e = tid; e < n * w; e += THREADS) {
            int r = e / w;
            int c = e % w;
            m[e] = c < n ? a.get(base + r * n + c) : (c - n == r ? 1.0f : 0.0f);
        }
        ctx.localBarrier();
        // Gauss-Jordan elimination with partial pivoting of the n x w matrix m.

        for (int col = 0; col < n; col++) {
            if (tid == 0) {
                int p = col;
                float best = TornadoMath.abs(m[col * w + col]);
                for (int r = col + 1; r < n; r++) {
                    float v = TornadoMath.abs(m[r * w + col]);
                    if (v > best) {
                        best = v;
                        p = r;
                    }
                }
                pivot[0] = p;
            }
            ctx.localBarrier();
            int p = pivot[0];
            if (p != col) {
                for (int c = tid; c < w; c += THREADS) {
                    float tmp = m[col * w + c];
                    m[col * w + c] = m[p * w + c];
                    m[p * w + c] = tmp;
                }
            }
            ctx.localBarrier();
            float inv = 1.0f / m[col * w + col];
            ctx.localBarrier();
            for (int c = tid; c < w; c += THREADS) {
                m[col * w + c] *= inv;
            }
            ctx.localBarrier();
            for (int e = tid; e < n * w; e += THREADS) {
                int r = e / w;
                int c = e % w;
                if (r != col && c != col) {
                    m[e] -= m[r * w + col] * m[col * w + c];
                }
            }
            ctx.localBarrier();
            for (int r = tid; r < n; r += THREADS) {
                if (r != col) {
                    m[r * w + col] = 0.0f;
                }
            }
            ctx.localBarrier();
        }
        for (int e = tid; e < n * n; e += THREADS) {
            out.set(base + e, m[(e / n) * w + n + e % n]);
        }
    }

    /** x[m] = a[m]^-1 b[m] for a [batch, n, n] and b, x [batch, n, nrhs], by Gauss-Jordan on [A | B]. */
    @JitBaseline("mlx_linalg_solve")
    public static void solve(KernelContext ctx, FloatArray a, FloatArray b, FloatArray out, int n, int nrhs) {
        float[] m = ctx.allocateFloatLocalArray(MAX_AUG);
        int[] pivot = ctx.allocateIntLocalArray(1);
        int tid = ctx.localIdx;
        int aBase = ctx.groupIdx * n * n;
        int bBase = ctx.groupIdx * n * nrhs;
        int w = n + nrhs;
        for (int e = tid; e < n * w; e += THREADS) {
            int r = e / w;
            int c = e % w;
            m[e] = c < n ? a.get(aBase + r * n + c) : b.get(bBase + r * nrhs + c - n);
        }
        ctx.localBarrier();
        // Gauss-Jordan elimination with partial pivoting of the n x w matrix m.

        for (int col = 0; col < n; col++) {
            if (tid == 0) {
                int p = col;
                float best = TornadoMath.abs(m[col * w + col]);
                for (int r = col + 1; r < n; r++) {
                    float v = TornadoMath.abs(m[r * w + col]);
                    if (v > best) {
                        best = v;
                        p = r;
                    }
                }
                pivot[0] = p;
            }
            ctx.localBarrier();
            int p = pivot[0];
            if (p != col) {
                for (int c = tid; c < w; c += THREADS) {
                    float tmp = m[col * w + c];
                    m[col * w + c] = m[p * w + c];
                    m[p * w + c] = tmp;
                }
            }
            ctx.localBarrier();
            float inv = 1.0f / m[col * w + col];
            ctx.localBarrier();
            for (int c = tid; c < w; c += THREADS) {
                m[col * w + c] *= inv;
            }
            ctx.localBarrier();
            for (int e = tid; e < n * w; e += THREADS) {
                int r = e / w;
                int c = e % w;
                if (r != col && c != col) {
                    m[e] -= m[r * w + col] * m[col * w + c];
                }
            }
            ctx.localBarrier();
            for (int r = tid; r < n; r += THREADS) {
                if (r != col) {
                    m[r * w + col] = 0.0f;
                }
            }
            ctx.localBarrier();
        }
        for (int e = tid; e < n * nrhs; e += THREADS) {
            out.set(bBase + e, m[(e / nrhs) * w + n + e % nrhs]);
        }
    }

    /** x[m] = a[m]^-1 b[m] for triangular a (lower, or upper), one right-hand side per thread. */
    @JitBaseline("mlx_linalg_solve_triangular")
    public static void solveTriangular(KernelContext ctx, FloatArray a, FloatArray b, FloatArray out, int n, int nrhs, int upper) {
        float[] t = ctx.allocateFloatLocalArray(MAX_NN);
        int tid = ctx.localIdx;
        int aBase = ctx.groupIdx * n * n;
        int bBase = ctx.groupIdx * n * nrhs;
        for (int e = tid; e < n * n; e += THREADS) {
            t[e] = a.get(aBase + e);
        }
        ctx.localBarrier();
        for (int c = tid; c < nrhs; c += THREADS) {
            if (upper == 0) {
                for (int i = 0; i < n; i++) {
                    float s = b.get(bBase + i * nrhs + c);
                    for (int k = 0; k < i; k++) {
                        s -= t[i * n + k] * out.get(bBase + k * nrhs + c);
                    }
                    out.set(bBase + i * nrhs + c, s / t[i * n + i]);
                }
            } else {
                for (int i = n - 1; i >= 0; i--) {
                    float s = b.get(bBase + i * nrhs + c);
                    for (int k = i + 1; k < n; k++) {
                        s -= t[i * n + k] * out.get(bBase + k * nrhs + c);
                    }
                    out.set(bBase + i * nrhs + c, s / t[i * n + i]);
                }
            }
        }
    }

    // ---------------------------------------------------------------- LU and QR

    /**
     * LU factorisation with partial pivoting of each a[m] ([batch, n, n]): {@code lu} holds L (unit
     * lower, below the diagonal) and U; pivots[k] is the row swapped with row k at step k (LAPACK
     * getrf, 0-based); perm[i] is the row of L U that row i of a becomes. Either output may be a
     * one-element dummy when {@code writePerm} or {@code writePivots} is 0.
     */
    @JitBaseline({ "mlx_linalg_lu", "mlx_linalg_lu_factor" })
    public static void lu(KernelContext ctx, FloatArray a, FloatArray lu, IntArray pivots, IntArray perm, int n, int writePivots, int writePerm) {
        float[] m = ctx.allocateFloatLocalArray(MAX_NN);
        int[] rows = ctx.allocateIntLocalArray(MAX_N);
        int[] piv = ctx.allocateIntLocalArray(MAX_N);
        int tid = ctx.localIdx;
        int base = ctx.groupIdx * n * n;
        for (int e = tid; e < n * n; e += THREADS) {
            m[e] = a.get(base + e);
        }
        for (int i = tid; i < n; i += THREADS) {
            rows[i] = i;
        }
        ctx.localBarrier();
        for (int k = 0; k < n; k++) {
            if (tid == 0) {
                int p = k;
                float best = TornadoMath.abs(m[k * n + k]);
                for (int r = k + 1; r < n; r++) {
                    float v = TornadoMath.abs(m[r * n + k]);
                    if (v > best) {
                        best = v;
                        p = r;
                    }
                }
                piv[k] = p;
                int t = rows[k];
                rows[k] = rows[p];
                rows[p] = t;
            }
            ctx.localBarrier();
            int p = piv[k];
            if (p != k) {
                for (int c = tid; c < n; c += THREADS) {
                    float tmp = m[k * n + c];
                    m[k * n + c] = m[p * n + c];
                    m[p * n + c] = tmp;
                }
            }
            ctx.localBarrier();
            float d = m[k * n + k];
            for (int r = k + 1 + tid; r < n; r += THREADS) {
                m[r * n + k] /= d;
            }
            ctx.localBarrier();
            int rest = n - k - 1;
            for (int e = tid; e < rest * rest; e += THREADS) {
                int r = k + 1 + e / rest;
                int c = k + 1 + e % rest;
                m[r * n + c] -= m[r * n + k] * m[k * n + c];
            }
            ctx.localBarrier();
        }
        for (int e = tid; e < n * n; e += THREADS) {
            lu.set(base + e, m[e]);
        }
        for (int i = tid; i < n; i += THREADS) {
            if (writePivots != 0) {
                pivots.set(ctx.groupIdx * n + i, piv[i]);
            }
            if (writePerm != 0) {
                // MLX's convention: row i of a is row perm[i] of l u.
                perm.set(ctx.groupIdx * n + rows[i], i);
            }
        }
    }

    /** Splits packed LU factors into a unit lower L and an upper U. */
    @JitBaseline("mlx_linalg_lu")
    public static void splitLu(KernelContext ctx, FloatArray lu, FloatArray l, FloatArray u, int total, int n) {
        int e = ctx.globalIdx;
        if (e < total) {
            int i = (e / n) % n;
            int j = e % n;
            float v = lu.get(e);
            l.set(e, i > j ? v : (i == j ? 1.0f : 0.0f));
            u.set(e, i <= j ? v : 0.0f);
        }
    }

    /**
     * Householder QR of each a[m] ([batch, n, n]): q orthogonal, r upper triangular, a = q r. The
     * reflector for column k zeroes it below the diagonal; each thread updates one column of R and
     * one row of Q.
     */
    @JitBaseline("mlx_linalg_qr")
    public static void qr(KernelContext ctx, FloatArray a, FloatArray q, FloatArray r, int n) {
        float[] rm = ctx.allocateFloatLocalArray(MAX_NN);
        float[] qm = ctx.allocateFloatLocalArray(MAX_NN);
        float[] v = ctx.allocateFloatLocalArray(MAX_N);
        float[] scal = ctx.allocateFloatLocalArray(1);
        int tid = ctx.localIdx;
        int base = ctx.groupIdx * n * n;
        for (int e = tid; e < n * n; e += THREADS) {
            rm[e] = a.get(base + e);
            qm[e] = (e / n) == (e % n) ? 1.0f : 0.0f;
        }
        ctx.localBarrier();
        for (int k = 0; k < n - 1; k++) {
            if (tid == 0) {
                float norm = 0.0f;
                for (int i = k; i < n; i++) {
                    norm += rm[i * n + k] * rm[i * n + k];
                }
                norm = TornadoMath.sqrt(norm);
                float x0 = rm[k * n + k];
                float alpha = x0 >= 0.0f ? -norm : norm;
                float vv = 0.0f;
                for (int i = k; i < n; i++) {
                    float vi = rm[i * n + k] - (i == k ? alpha : 0.0f);
                    v[i] = vi;
                    vv += vi * vi;
                }
                scal[0] = vv > 0.0f ? 2.0f / vv : 0.0f;
            }
            ctx.localBarrier();
            float beta = scal[0];
            // R = H R, one column per thread.
            for (int j = tid; j < n; j += THREADS) {
                float dot = 0.0f;
                for (int i = k; i < n; i++) {
                    dot += v[i] * rm[i * n + j];
                }
                dot *= beta;
                for (int i = k; i < n; i++) {
                    rm[i * n + j] -= dot * v[i];
                }
            }
            // Q = Q H, one row per thread.
            for (int i = tid; i < n; i += THREADS) {
                float dot = 0.0f;
                for (int j = k; j < n; j++) {
                    dot += qm[i * n + j] * v[j];
                }
                dot *= beta;
                for (int j = k; j < n; j++) {
                    qm[i * n + j] -= dot * v[j];
                }
            }
            ctx.localBarrier();
        }
        for (int e = tid; e < n * n; e += THREADS) {
            q.set(base + e, qm[e]);
            r.set(base + e, (e / n) <= (e % n) ? rm[e] : 0.0f);
        }
    }

    // ---------------------------------------------------------------- symmetric eigenproblem, SVD, pseudo-inverse

    /** Jacobi sweeps before the eigen and singular value kernels stop regardless of convergence. */
    public static final int MAX_SWEEPS = 20;

    /**
     * Player at round-robin position {@code i} in round {@code r} of a tournament of {@code m}
     * players (m even): position 0 is fixed, the others rotate. Pairs are positions (i, m - 1 - i).
     */
    private static int player(int i, int r, int m) {
        return i == 0 ? 0 : ((i - 1 + r) % (m - 1)) + 1;
    }

    /**
     * Eigenvalues (ascending) and, if {@code writeVectors != 0}, eigenvectors (columns) of each
     * symmetric a[m] ([batch, n, n], read from its lower or upper triangle) by the parallel cyclic
     * Jacobi method: each round applies n / 2 disjoint rotations (Brent-Luk ordering), one per
     * thread for the angles and in parallel for the row, column and vector updates.
     */
    @JitBaseline({ "mlx_linalg_eigh", "mlx_linalg_eigvalsh" })
    public static void eigh(KernelContext ctx, FloatArray a, FloatArray values, FloatArray vectors, int n, int upper, int writeVectors) {
        float[] m = ctx.allocateFloatLocalArray(MAX_NN);
        float[] v = ctx.allocateFloatLocalArray(MAX_NN);
        float[] cs = ctx.allocateFloatLocalArray(MAX_N);
        float[] sn = ctx.allocateFloatLocalArray(MAX_N);
        int[] pp = ctx.allocateIntLocalArray(MAX_N);
        int[] qq = ctx.allocateIntLocalArray(MAX_N);
        float[] off = ctx.allocateFloatLocalArray(1);
        int tid = ctx.localIdx;
        int base = ctx.groupIdx * n * n;
        for (int e = tid; e < n * n; e += THREADS) {
            int i = e / n;
            int j = e % n;
            boolean stored = upper != 0 ? i <= j : i >= j;
            m[e] = stored ? a.get(base + e) : a.get(base + j * n + i);
            v[e] = i == j ? 1.0f : 0.0f;
        }
        ctx.localBarrier();
        int players = (n & 1) == 0 ? n : n + 1;
        int pairs = players / 2;
        for (int sweep = 0; sweep < MAX_SWEEPS; sweep++) {
            if (tid == 0) {
                float o = 0.0f;
                float d = 0.0f;
                for (int e = 0; e < n * n; e++) {
                    if (e / n != e % n) {
                        o += m[e] * m[e];
                    } else {
                        d += m[e] * m[e];
                    }
                }
                off[0] = o <= 1.0e-14f * d ? 1.0f : 0.0f;
            }
            ctx.localBarrier();
            float converged = off[0];
            ctx.localBarrier();
            if (converged != 0.0f) {
                break;
            }
            for (int r = 0; r < players - 1; r++) {
                for (int k = tid; k < pairs; k += THREADS) {
                    int p = player(k, r, players);
                    int q = player(players - 1 - k, r, players);
                    if (p > q) {
                        int t = p;
                        p = q;
                        q = t;
                    }
                    float c = 1.0f;
                    float s = 0.0f;
                    if (q < n) {
                        float apq = m[p * n + q];
                        if (apq != 0.0f) {
                            float theta = (m[q * n + q] - m[p * n + p]) / (2.0f * apq);
                            float t = (theta >= 0.0f ? 1.0f : -1.0f) / (TornadoMath.abs(theta) + TornadoMath.sqrt(theta * theta + 1.0f));
                            c = 1.0f / TornadoMath.sqrt(t * t + 1.0f);
                            s = t * c;
                        }
                    }
                    pp[k] = p;
                    qq[k] = q;
                    cs[k] = c;
                    sn[k] = s;
                }
                ctx.localBarrier();
                // Columns p and q of m and of v.
                for (int e = tid; e < pairs * n; e += THREADS) {
                    int k = e / n;
                    int i = e % n;
                    int q = qq[k];
                    if (q < n) {
                        int p = pp[k];
                        float c = cs[k];
                        float s = sn[k];
                        float mip = m[i * n + p];
                        float miq = m[i * n + q];
                        m[i * n + p] = c * mip - s * miq;
                        m[i * n + q] = s * mip + c * miq;
                        float vip = v[i * n + p];
                        float viq = v[i * n + q];
                        v[i * n + p] = c * vip - s * viq;
                        v[i * n + q] = s * vip + c * viq;
                    }
                }
                ctx.localBarrier();
                // Rows p and q of m.
                for (int e = tid; e < pairs * n; e += THREADS) {
                    int k = e / n;
                    int j = e % n;
                    int q = qq[k];
                    if (q < n) {
                        int p = pp[k];
                        float c = cs[k];
                        float s = sn[k];
                        float mpj = m[p * n + j];
                        float mqj = m[q * n + j];
                        m[p * n + j] = c * mpj - s * mqj;
                        m[q * n + j] = s * mpj + c * mqj;
                    }
                }
                ctx.localBarrier();
            }
        }
        // Ascending order: each eigenvalue's rank among the diagonal.
        int vbase = ctx.groupIdx * n;
        for (int i = tid; i < n; i += THREADS) {
            float di = m[i * n + i];
            int rank = 0;
            for (int j = 0; j < n; j++) {
                float dj = m[j * n + j];
                if (dj < di || (dj == di && j < i)) {
                    rank++;
                }
            }
            values.set(vbase + rank, di);
            if (writeVectors != 0) {
                for (int row = 0; row < n; row++) {
                    vectors.set(base + row * n + rank, v[row * n + i]);
                }
            }
        }
    }

    /**
     * Singular value decomposition a = u diag(s) vt of each square a[m] ([batch, n, n]) by one-sided
     * (Hestenes) Jacobi: column pairs of a working copy are rotated until orthogonal, accumulating
     * the rotations in v; s are the column norms (descending), u the normalised columns. With
     * {@code writeVectors == 0} only s is written.
     */
    @JitBaseline({ "mlx_linalg_svd", "mlx_linalg_pinv" })
    public static void svd(KernelContext ctx, FloatArray a, FloatArray u, FloatArray sv, FloatArray vt, int n, int writeVectors) {
        float[] w = ctx.allocateFloatLocalArray(MAX_NN);
        float[] v = ctx.allocateFloatLocalArray(MAX_NN);
        float[] cs = ctx.allocateFloatLocalArray(MAX_N);
        float[] sn = ctx.allocateFloatLocalArray(MAX_N);
        int[] pp = ctx.allocateIntLocalArray(MAX_N);
        int[] qq = ctx.allocateIntLocalArray(MAX_N);
        float[] norms = ctx.allocateFloatLocalArray(MAX_N);
        int[] rotated = ctx.allocateIntLocalArray(1);
        int tid = ctx.localIdx;
        int base = ctx.groupIdx * n * n;
        for (int e = tid; e < n * n; e += THREADS) {
            w[e] = a.get(base + e);
            v[e] = (e / n) == (e % n) ? 1.0f : 0.0f;
        }
        ctx.localBarrier();
        int players = (n & 1) == 0 ? n : n + 1;
        int pairs = players / 2;
        for (int sweep = 0; sweep < MAX_SWEEPS; sweep++) {
            if (tid == 0) {
                rotated[0] = 0;
            }
            ctx.localBarrier();
            for (int r = 0; r < players - 1; r++) {
                for (int k = tid; k < pairs; k += THREADS) {
                    int p = player(k, r, players);
                    int q = player(players - 1 - k, r, players);
                    if (p > q) {
                        int t = p;
                        p = q;
                        q = t;
                    }
                    float c = 1.0f;
                    float s = 0.0f;
                    if (q < n) {
                        float alpha = 0.0f;
                        float beta = 0.0f;
                        float gamma = 0.0f;
                        for (int i = 0; i < n; i++) {
                            float wp = w[i * n + p];
                            float wq = w[i * n + q];
                            alpha += wp * wp;
                            beta += wq * wq;
                            gamma += wp * wq;
                        }
                        if (TornadoMath.abs(gamma) > 1.0e-7f * TornadoMath.sqrt(alpha * beta)) {
                            float zeta = (beta - alpha) / (2.0f * gamma);
                            float t = (zeta >= 0.0f ? 1.0f : -1.0f) / (TornadoMath.abs(zeta) + TornadoMath.sqrt(1.0f + zeta * zeta));
                            c = 1.0f / TornadoMath.sqrt(1.0f + t * t);
                            s = c * t;
                            rotated[0] = 1;
                        }
                    }
                    pp[k] = p;
                    qq[k] = q;
                    cs[k] = c;
                    sn[k] = s;
                }
                ctx.localBarrier();
                for (int e = tid; e < pairs * n; e += THREADS) {
                    int k = e / n;
                    int i = e % n;
                    int q = qq[k];
                    if (q < n) {
                        int p = pp[k];
                        float c = cs[k];
                        float s = sn[k];
                        float wp = w[i * n + p];
                        float wq = w[i * n + q];
                        w[i * n + p] = c * wp - s * wq;
                        w[i * n + q] = s * wp + c * wq;
                        float vp = v[i * n + p];
                        float vq = v[i * n + q];
                        v[i * n + p] = c * vp - s * vq;
                        v[i * n + q] = s * vp + c * vq;
                    }
                }
                ctx.localBarrier();
            }
            // Every thread reads the flag before thread 0 may clear it for the next sweep.
            int again = rotated[0];
            ctx.localBarrier();
            if (again == 0) {
                break;
            }
        }
        for (int j = tid; j < n; j += THREADS) {
            float s2 = 0.0f;
            for (int i = 0; i < n; i++) {
                s2 += w[i * n + j] * w[i * n + j];
            }
            norms[j] = TornadoMath.sqrt(s2);
        }
        ctx.localBarrier();
        int sbase = ctx.groupIdx * n;
        for (int j = tid; j < n; j += THREADS) {
            float sj = norms[j];
            int rank = 0;
            for (int k = 0; k < n; k++) {
                if (norms[k] > sj || (norms[k] == sj && k < j)) {
                    rank++;
                }
            }
            sv.set(sbase + rank, sj);
            if (writeVectors != 0) {
                float inv = sj > 0.0f ? 1.0f / sj : 0.0f;
                for (int i = 0; i < n; i++) {
                    u.set(base + i * n + rank, w[i * n + j] * inv);
                    vt.set(base + rank * n + i, v[i * n + j]);
                }
            }
        }
    }

    /**
     * Pseudo-inverse from an SVD ({@link #svd}): out = vt^T diag(1/s) u^T, with singular values
     * below {@code rcond * s_max} treated as zero; one thread per output element.
     */
    @JitBaseline("mlx_linalg_pinv")
    public static void pinvFromSvd(KernelContext ctx, FloatArray u, FloatArray sv, FloatArray vt, FloatArray out, int total, int n, float rcond) {
        int e = ctx.globalIdx;
        if (e < total) {
            int b = e / (n * n);
            int i = (e / n) % n;
            int j = e % n;
            int base = b * n * n;
            float cutoff = rcond * sv.get(b * n);
            float acc = 0.0f;
            for (int k = 0; k < n; k++) {
                float s = sv.get(b * n + k);
                if (s > cutoff) {
                    acc += vt.get(base + k * n + i) * u.get(base + j * n + k) / s;
                }
            }
            out.set(e, acc);
        }
    }

    // ---------------------------------------------------------------- general eigenproblem

    /** Largest QR iterations per eigenvalue in {@link #eig} before it gives up on that block. */
    private static final int MAX_QR_ITERATIONS = 60;
    private static final float EPS = 1.2e-7f;

    /**
     * Eigenvalues and, if {@code writeVectors != 0}, right eigenvectors of each real a[m]
     * ([batch, n, n]); complex results are interleaved (real, imaginary) pairs: values
     * {@code [batch, n, 2]}, vectors {@code [batch, n, n, 2]} with eigenvector k in column k, of
     * unit norm with its largest component real and positive (LAPACK geev also makes it real but
     * leaves its sign free, so the two agree up to sign). The matrix is reduced
     * to Hessenberg form with Householder reflectors (in parallel), thread 0 runs the Francis
     * double-shift QR iteration of Numerical Recipes' hqr (with explicit tolerances in place of its
     * exact floating-point deflation tests, which fast math does not preserve), and each
     * eigenvector is found by two steps of complex inverse iteration, a Gauss-Jordan solve of
     * (A - lambda I) x = b in threadgroup memory. Eigenvalues are in the order the QR iteration
     * deflates them.
     */
    @JitBaseline({ "mlx_linalg_eig", "mlx_linalg_eigvals" })
    public static void eig(KernelContext ctx, FloatArray a, FloatArray values, FloatArray vectors, int n, int writeVectors) {
        float[] a0 = ctx.allocateFloatLocalArray(MAX_NN);
        float[] h = ctx.allocateFloatLocalArray(MAX_NN + MAX_N);
        float[] mi = ctx.allocateFloatLocalArray(MAX_NN + MAX_N);
        float[] wr = ctx.allocateFloatLocalArray(MAX_N);
        float[] wi = ctx.allocateFloatLocalArray(MAX_N);
        float[] v = ctx.allocateFloatLocalArray(MAX_N);
        float[] scal = ctx.allocateFloatLocalArray(4);
        int[] pivot = ctx.allocateIntLocalArray(1);
        int tid = ctx.localIdx;
        int base = ctx.groupIdx * n * n;
        for (int e = tid; e < n * n; e += THREADS) {
            float x = a.get(base + e);
            a0[e] = x;
            h[e] = x;
        }
        ctx.localBarrier();

        // Hessenberg reduction: reflector k zeroes column k below the subdiagonal; H = P H P.
        for (int k = 0; k < n - 2; k++) {
            if (tid == 0) {
                float norm = 0.0f;
                for (int i = k + 1; i < n; i++) {
                    norm += h[i * n + k] * h[i * n + k];
                }
                norm = TornadoMath.sqrt(norm);
                float x0 = h[(k + 1) * n + k];
                float alpha = x0 >= 0.0f ? -norm : norm;
                float vv = 0.0f;
                for (int i = k + 1; i < n; i++) {
                    float vi = h[i * n + k] - (i == k + 1 ? alpha : 0.0f);
                    v[i] = vi;
                    vv += vi * vi;
                }
                scal[0] = vv > 0.0f ? 2.0f / vv : 0.0f;
            }
            ctx.localBarrier();
            float beta = scal[0];
            for (int j = tid; j < n; j += THREADS) {
                float dot = 0.0f;
                for (int i = k + 1; i < n; i++) {
                    dot += v[i] * h[i * n + j];
                }
                dot *= beta;
                for (int i = k + 1; i < n; i++) {
                    h[i * n + j] -= dot * v[i];
                }
            }
            ctx.localBarrier();
            for (int i = tid; i < n; i += THREADS) {
                float dot = 0.0f;
                for (int j = k + 1; j < n; j++) {
                    dot += h[i * n + j] * v[j];
                }
                dot *= beta;
                for (int j = k + 1; j < n; j++) {
                    h[i * n + j] -= dot * v[j];
                }
            }
            ctx.localBarrier();
        }

        // Francis double-shift QR (hqr), 1-based indices: element (i, j) is h[o + i * n + j].
        if (tid == 0) {
            int o = -n - 1;
            float anorm = 0.0f;
            for (int i = 1; i <= n; i++) {
                for (int j = i > 1 ? i - 1 : 1; j <= n; j++) {
                    anorm += TornadoMath.abs(h[o + i * n + j]);
                }
            }
            int nn = n;
            float t = 0.0f;
            while (nn >= 1) {
                int its = 0;
                int l;
                do {
                    for (l = nn; l >= 2; l--) {
                        float s = TornadoMath.abs(h[o + (l - 1) * n + l - 1]) + TornadoMath.abs(h[o + l * n + l]);
                        if (s == 0.0f) {
                            s = anorm;
                        }
                        if (TornadoMath.abs(h[o + l * n + l - 1]) <= EPS * s) {
                            h[o + l * n + l - 1] = 0.0f;
                            break;
                        }
                    }
                    float x = h[o + nn * n + nn];
                    if (l == nn) {
                        wr[nn - 1] = x + t;
                        wi[nn - 1] = 0.0f;
                        nn--;
                    } else {
                        float y = h[o + (nn - 1) * n + nn - 1];
                        float w = h[o + nn * n + nn - 1] * h[o + (nn - 1) * n + nn];
                        if (l == nn - 1) {
                            float p = 0.5f * (y - x);
                            float q = p * p + w;
                            float z = TornadoMath.sqrt(TornadoMath.abs(q));
                            x += t;
                            if (q >= 0.0f) {
                                z = p + (p >= 0.0f ? z : -z);
                                wr[nn - 2] = x + z;
                                wr[nn - 1] = z != 0.0f ? x - w / z : x + z;
                                wi[nn - 2] = 0.0f;
                                wi[nn - 1] = 0.0f;
                            } else {
                                wr[nn - 2] = x + p;
                                wr[nn - 1] = x + p;
                                wi[nn - 2] = -z;
                                wi[nn - 1] = z;
                            }
                            nn -= 2;
                        } else {
                            if (its == MAX_QR_ITERATIONS) {
                                // No convergence: report the block's diagonal and move on.
                                for (int i = l; i <= nn; i++) {
                                    wr[i - 1] = h[o + i * n + i] + t;
                                    wi[i - 1] = 0.0f;
                                }
                                nn = l - 1;
                                l = nn + 2;
                            } else {
                                if (its == 10 || its == 20 || its == 40) {
                                    t += x;
                                    for (int i = 1; i <= nn; i++) {
                                        h[o + i * n + i] -= x;
                                    }
                                    float s = TornadoMath.abs(h[o + nn * n + nn - 1]) + TornadoMath.abs(h[o + (nn - 1) * n + nn - 2]);
                                    x = 0.75f * s;
                                    y = x;
                                    w = -0.4375f * s * s;
                                }
                                its++;
                                int m;
                                float p = 0.0f;
                                float q = 0.0f;
                                float r = 0.0f;
                                float z;
                                for (m = nn - 2; m >= l; m--) {
                                    z = h[o + m * n + m];
                                    r = x - z;
                                    float s = y - z;
                                    p = (r * s - w) / h[o + (m + 1) * n + m] + h[o + m * n + m + 1];
                                    q = h[o + (m + 1) * n + m + 1] - z - r - s;
                                    r = h[o + (m + 2) * n + m + 1];
                                    s = TornadoMath.abs(p) + TornadoMath.abs(q) + TornadoMath.abs(r);
                                    p /= s;
                                    q /= s;
                                    r /= s;
                                    if (m == l) {
                                        break;
                                    }
                                    float u = TornadoMath.abs(h[o + m * n + m - 1]) * (TornadoMath.abs(q) + TornadoMath.abs(r));
                                    float vv = TornadoMath.abs(p) * (TornadoMath.abs(h[o + (m - 1) * n + m - 1]) + TornadoMath.abs(z) + TornadoMath.abs(h[o + (m + 1) * n + m + 1]));
                                    if (u <= EPS * vv) {
                                        break;
                                    }
                                }
                                for (int i = m + 2; i <= nn; i++) {
                                    h[o + i * n + i - 2] = 0.0f;
                                    if (i != m + 2) {
                                        h[o + i * n + i - 3] = 0.0f;
                                    }
                                }
                                for (int k = m; k <= nn - 1; k++) {
                                    if (k != m) {
                                        p = h[o + k * n + k - 1];
                                        q = h[o + (k + 1) * n + k - 1];
                                        r = 0.0f;
                                        if (k != nn - 1) {
                                            r = h[o + (k + 2) * n + k - 1];
                                        }
                                        x = TornadoMath.abs(p) + TornadoMath.abs(q) + TornadoMath.abs(r);
                                        if (x != 0.0f) {
                                            p /= x;
                                            q /= x;
                                            r /= x;
                                        }
                                    }
                                    float s = TornadoMath.sqrt(p * p + q * q + r * r);
                                    s = p >= 0.0f ? s : -s;
                                    if (s != 0.0f) {
                                        if (k == m) {
                                            if (l != m) {
                                                h[o + k * n + k - 1] = -h[o + k * n + k - 1];
                                            }
                                        } else {
                                            h[o + k * n + k - 1] = -s * x;
                                        }
                                        p += s;
                                        x = p / s;
                                        y = q / s;
                                        z = r / s;
                                        q /= p;
                                        r /= p;
                                        for (int j = k; j <= nn; j++) {
                                            p = h[o + k * n + j] + q * h[o + (k + 1) * n + j];
                                            if (k != nn - 1) {
                                                p += r * h[o + (k + 2) * n + j];
                                                h[o + (k + 2) * n + j] -= p * z;
                                            }
                                            h[o + (k + 1) * n + j] -= p * y;
                                            h[o + k * n + j] -= p * x;
                                        }
                                        int mmin = nn < k + 3 ? nn : k + 3;
                                        for (int i = l; i <= mmin; i++) {
                                            p = x * h[o + i * n + k] + y * h[o + i * n + k + 1];
                                            if (k != nn - 1) {
                                                p += z * h[o + i * n + k + 2];
                                                h[o + i * n + k + 2] -= p * r;
                                            }
                                            h[o + i * n + k + 1] -= p * q;
                                            h[o + i * n + k] -= p;
                                        }
                                    }
                                }
                            }
                        }
                    }
                } while (l < nn - 1);
            }
        }
        ctx.localBarrier();
        int vbase = ctx.groupIdx * n;
        for (int k = tid; k < n; k += THREADS) {
            values.set(2 * (vbase + k), wr[k]);
            values.set(2 * (vbase + k) + 1, wi[k]);
        }
        if (writeVectors != 0) {
            // Inverse iteration on the original matrix; h now holds the real part and mi the
            // imaginary part of the augmented system [A - mu I | b], row stride n + 1.
            int w1 = n + 1;
            for (int k = 0; k < n; k++) {
                float scale = TornadoMath.abs(wr[k]) + TornadoMath.abs(wi[k]) + 1.0f;
                float muR = wr[k] + 1.0e-4f * scale;
                float muI = wi[k];
                for (int i = tid; i < n; i += THREADS) {
                    h[i * w1 + n] = 1.0f;
                    mi[i * w1 + n] = 0.0f;
                }
                for (int step = 0; step < 2; step++) {
                    for (int e = tid; e < n * n; e += THREADS) {
                        int i = e / n;
                        int j = e % n;
                        h[i * w1 + j] = a0[e] - (i == j ? muR : 0.0f);
                        mi[i * w1 + j] = i == j ? -muI : 0.0f;
                    }
                    ctx.localBarrier();
                    for (int col = 0; col < n; col++) {
                        if (tid == 0) {
                            int p = col;
                            float best = h[col * w1 + col] * h[col * w1 + col] + mi[col * w1 + col] * mi[col * w1 + col];
                            for (int r = col + 1; r < n; r++) {
                                float mag = h[r * w1 + col] * h[r * w1 + col] + mi[r * w1 + col] * mi[r * w1 + col];
                                if (mag > best) {
                                    best = mag;
                                    p = r;
                                }
                            }
                            pivot[0] = p;
                        }
                        ctx.localBarrier();
                        int p = pivot[0];
                        if (p != col) {
                            for (int c = tid; c < w1; c += THREADS) {
                                float tr = h[col * w1 + c];
                                float ti = mi[col * w1 + c];
                                h[col * w1 + c] = h[p * w1 + c];
                                mi[col * w1 + c] = mi[p * w1 + c];
                                h[p * w1 + c] = tr;
                                mi[p * w1 + c] = ti;
                            }
                        }
                        ctx.localBarrier();
                        float dr = h[col * w1 + col];
                        float di = mi[col * w1 + col];
                        float dd = dr * dr + di * di;
                        float ir = dd > 0.0f ? dr / dd : 0.0f;
                        float ii = dd > 0.0f ? -di / dd : 0.0f;
                        ctx.localBarrier();
                        for (int c = tid; c < w1; c += THREADS) {
                            float xr = h[col * w1 + c];
                            float xi = mi[col * w1 + c];
                            h[col * w1 + c] = xr * ir - xi * ii;
                            mi[col * w1 + c] = xr * ii + xi * ir;
                        }
                        ctx.localBarrier();
                        for (int e = tid; e < n * w1; e += THREADS) {
                            int r = e / w1;
                            int c = e % w1;
                            if (r != col && c != col) {
                                float fr = h[r * w1 + col];
                                float fi = mi[r * w1 + col];
                                float pr = h[col * w1 + c];
                                float pi = mi[col * w1 + c];
                                h[e] -= fr * pr - fi * pi;
                                mi[e] -= fr * pi + fi * pr;
                            }
                        }
                        ctx.localBarrier();
                    }
                    // Normalise x (the last column): unit norm, largest component real and positive.
                    if (tid == 0) {
                        int big = 0;
                        float bestMag = -1.0f;
                        float norm = 0.0f;
                        for (int i = 0; i < n; i++) {
                            float mag = h[i * w1 + n] * h[i * w1 + n] + mi[i * w1 + n] * mi[i * w1 + n];
                            norm += mag;
                            if (mag > bestMag) {
                                bestMag = mag;
                                big = i;
                            }
                        }
                        float br = h[big * w1 + n];
                        float bi = mi[big * w1 + n];
                        float bm = TornadoMath.sqrt(bestMag);
                        float inv = 1.0f / (TornadoMath.sqrt(norm) * bm);
                        // Multiply by conj(x_big) / (|x_big| * |x|).
                        scal[0] = br * inv;
                        scal[1] = -bi * inv;
                    }
                    ctx.localBarrier();
                    float sr = scal[0];
                    float si = scal[1];
                    for (int i = tid; i < n; i += THREADS) {
                        float xr = h[i * w1 + n];
                        float xi = mi[i * w1 + n];
                        mi[i * w1 + n] = xr * si + xi * sr;
                        h[i * w1 + n] = xr * sr - xi * si;
                    }
                    ctx.localBarrier();
                }
                for (int i = tid; i < n; i += THREADS) {
                    int at = 2 * (base + i * n + k);
                    vectors.set(at, h[i * w1 + n]);
                    vectors.set(at + 1, mi[i * w1 + n]);
                }
                ctx.localBarrier();
            }
        }
    }
}
