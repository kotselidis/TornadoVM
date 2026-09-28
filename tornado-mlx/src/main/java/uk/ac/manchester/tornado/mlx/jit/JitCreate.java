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
 * JIT counterparts of the MLX array-construction and matrix-structure operations
 * ({@link uk.ac.manchester.tornado.mlx.MlxCreate}), float32, one thread per output element except
 * {@link #trace} (one threadgroup).
 */
public final class JitCreate {

    public static final int BARTLETT = 0;
    public static final int BLACKMAN = 1;
    public static final int HAMMING = 2;
    public static final int HANNING = 3;

    private static final int THREADS = 256;
    private static final float TWO_PI = 6.283185307179586f;

    private JitCreate() {
    }

    @JitBaseline({ "mlx_full", "mlx_full_like", "mlx_zeros", "mlx_zeros_like", "mlx_ones", "mlx_ones_like" })
    public static void fill(KernelContext ctx, FloatArray out, int n, float value) {
        int i = ctx.globalIdx;
        if (i < n) {
            out.set(i, value);
        }
    }

    /** out[i] = start + i * step. */
    @JitBaseline("mlx_arange")
    public static void arange(KernelContext ctx, FloatArray out, int n, float start, float step) {
        int i = ctx.globalIdx;
        if (i < n) {
            out.set(i, start + i * step);
        }
    }

    /** n points from start to stop inclusive. */
    @JitBaseline("mlx_linspace")
    public static void linspace(KernelContext ctx, FloatArray out, int n, float start, float stop) {
        int i = ctx.globalIdx;
        if (i < n) {
            float step = n > 1 ? (stop - start) / (n - 1) : 0.0f;
            out.set(i, start + i * step);
        }
    }

    /** out[r, c] of an n x m matrix: 1 on diagonal k ({@code tri == 0}), or on and below it ({@code tri != 0}). */
    @JitBaseline({ "mlx_eye", "mlx_identity", "mlx_tri" })
    public static void eye(KernelContext ctx, FloatArray out, int n, int m, int k, int tri) {
        int t = ctx.globalIdx;
        if (t < n * m) {
            int d = t % m - t / m;
            boolean one = tri != 0 ? d <= k : d == k;
            out.set(t, one ? 1.0f : 0.0f);
        }
    }

    /** Window functions of length m, as NumPy defines them. */
    @JitBaseline({ "mlx_bartlett", "mlx_blackman", "mlx_hamming", "mlx_hanning" })
    public static void window(KernelContext ctx, FloatArray out, int m, int kind) {
        int i = ctx.globalIdx;
        if (i < m) {
            float r;
            if (m == 1) {
                r = 1.0f;
            } else {
                float x = (float) i / (m - 1);
                if (kind == BARTLETT) {
                    r = 1.0f - TornadoMath.abs(2.0f * x - 1.0f);
                } else if (kind == BLACKMAN) {
                    r = 0.42f - 0.5f * TornadoMath.cos(TWO_PI * x) + 0.08f * TornadoMath.cos(2.0f * TWO_PI * x);
                } else if (kind == HAMMING) {
                    r = 0.54f - 0.46f * TornadoMath.cos(TWO_PI * x);
                } else {
                    r = 0.5f - 0.5f * TornadoMath.cos(TWO_PI * x);
                }
            }
            out.set(i, r);
        }
    }

    /**
     * Coordinate grids from x (nx) and y (ny): with {@code ij == 0} (Cartesian), outX and outY are
     * [ny, nx] with outX[r, c] = x[c] and outY[r, c] = y[r]; with {@code ij != 0} (matrix indexing)
     * they are [nx, ny] with outX[r, c] = x[r] and outY[r, c] = y[c].
     */
    @JitBaseline("mlx_meshgrid")
    public static void meshgrid(KernelContext ctx, FloatArray x, FloatArray y, FloatArray outX, FloatArray outY, int nx, int ny, int ij) {
        int t = ctx.globalIdx;
        if (t < nx * ny) {
            if (ij == 0) {
                outX.set(t, x.get(t % nx));
                outY.set(t, y.get(t / nx));
            } else {
                outX.set(t, x.get(t / ny));
                outY.set(t, y.get(t % ny));
            }
        }
    }

    /** The (n + |k|)-square matrix with v (length n) on diagonal k and zeros elsewhere. */
    @JitBaseline("mlx_diag")
    public static void diag(KernelContext ctx, FloatArray v, FloatArray out, int n, int k) {
        int s = n + (k >= 0 ? k : -k);
        int t = ctx.globalIdx;
        if (t < s * s) {
            int r = t / s;
            int c = t % s;
            float value = 0.0f;
            if (c - r == k) {
                value = v.get(k >= 0 ? r : c);
            }
            out.set(t, value);
        }
    }

    /** out[t] = a[t + max(-offset, 0), t + max(offset, 0)] for a [rows, cols]; len elements. */
    @JitBaseline("mlx_diagonal")
    public static void diagonal(KernelContext ctx, FloatArray a, FloatArray out, int cols, int offset, int len) {
        int t = ctx.globalIdx;
        if (t < len) {
            int r = t + (offset < 0 ? -offset : 0);
            int c = t + (offset > 0 ? offset : 0);
            out.set(t, a.get(r * cols + c));
        }
    }

    /** out[0] = the sum of diagonal {@code offset} of a [rows, cols] (len elements); one threadgroup. */
    @JitBaseline("mlx_trace")
    public static void trace(KernelContext ctx, FloatArray a, FloatArray out, int cols, int offset, int len) {
        float[] red = ctx.allocateFloatLocalArray(THREADS);
        int tid = ctx.localIdx;
        int r0 = offset < 0 ? -offset : 0;
        int c0 = offset > 0 ? offset : 0;
        float sum = 0.0f;
        for (int t = tid; t < len; t += THREADS) {
            sum += a.get((r0 + t) * cols + c0 + t);
        }
        red[tid] = sum;
        ctx.localBarrier();
        for (int s = THREADS / 2; s > 0; s >>= 1) {
            if (tid < s) {
                red[tid] += red[tid + s];
            }
            ctx.localBarrier();
        }
        if (tid == 0) {
            out.set(0, red[0]);
        }
    }

    /** a [rows, cols] with the elements above ({@code upper == 0}) or below diagonal k zeroed. */
    @JitBaseline({ "mlx_tril", "mlx_triu" })
    public static void triangle(KernelContext ctx, FloatArray a, FloatArray out, int rows, int cols, int k, int upper) {
        int t = ctx.globalIdx;
        if (t < rows * cols) {
            int d = t % cols - t / cols;
            boolean keep = upper != 0 ? d >= k : d <= k;
            out.set(t, keep ? a.get(t) : 0.0f);
        }
    }
}
