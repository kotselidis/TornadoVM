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
}
