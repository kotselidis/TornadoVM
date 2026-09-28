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
 * JIT counterparts of the MLX FFTs ({@link uk.ac.manchester.tornado.mlx.MlxFft}), float32, complex
 * values as interleaved (real, imaginary) pairs. A transform runs along the middle axis of data
 * viewed as {@code [outer, len, inner]}, so one kernel serves every axis of a multi-dimensional
 * transform:
 * <ul>
 * <li>{@link #fft}: one threadgroup per transformed line, an iterative radix-2 Cooley-Tukey FFT in
 * threadgroup memory, for power-of-two lengths up to {@link #MAX_N};</li>
 * <li>{@link #dft}: one thread per output coefficient, a direct O(n^2) transform, for other
 * lengths.</li>
 * </ul>
 * Both read a line of {@code inLen} values (complex, real, or the {@code n / 2 + 1} coefficients of a
 * real signal's spectrum), zero-padded or cropped to the transform length {@code n}, and write
 * {@code outLen} values (complex, or the real parts), multiplied by {@code scale}.
 */
public final class JitFft {

    /** Threads per threadgroup. */
    public static final int THREADS = 256;
    /** Largest power-of-two length {@link #fft} holds in threadgroup memory. */
    public static final int MAX_N = 2048;

    /** Input kinds. */
    public static final int COMPLEX = 0;
    public static final int REAL = 1;
    /** The first n / 2 + 1 coefficients of a real signal's spectrum (the rest by conjugate symmetry). */
    public static final int HERMITIAN = 2;

    private static final float TWO_PI = 6.283185307179586f;

    private JitFft() {
    }

    /**
     * Radix-2 FFT of each line: in [outer, inLen, inner] (complex pairs unless {@code inKind == REAL}),
     * out [outer, outLen, inner] (complex pairs unless {@code realOut != 0}); {@code n} is a power of
     * two. {@code inverse != 0} uses exp(+i ...).
     */
    @JitBaseline({ "mlx_fft_fft", "mlx_fft_ifft", "mlx_fft_rfft", "mlx_fft_irfft", "mlx_fft_fft2", "mlx_fft_ifft2", "mlx_fft_rfft2", "mlx_fft_irfft2", "mlx_fft_fftn", "mlx_fft_ifftn",
            "mlx_fft_rfftn", "mlx_fft_irfftn" })
    public static void fft(KernelContext ctx, FloatArray in, FloatArray out, int n, int logN, int inLen, int outLen, int inner, int inKind, int realOut, int inverse, float scale) {
        float[] re = ctx.allocateFloatLocalArray(MAX_N);
        float[] im = ctx.allocateFloatLocalArray(MAX_N);
        int line = ctx.groupIdx;
        int tid = ctx.localIdx;
        int o = line / inner;
        int j = line % inner;
        int inBase = o * inLen * inner + j;
        int outBase = o * outLen * inner + j;
        // Load in bit-reversed order.
        for (int i = tid; i < n; i += THREADS) {
            int rev = 0;
            int x = i;
            for (int b = 0; b < logN; b++) {
                rev = (rev << 1) | (x & 1);
                x >>= 1;
            }
            float vr = 0.0f;
            float vi = 0.0f;
            if (inKind == HERMITIAN) {
                int k = i <= n / 2 ? i : n - i;
                if (k < inLen) {
                    vr = in.get(2 * (inBase + k * inner));
                    vi = in.get(2 * (inBase + k * inner) + 1);
                    if (i > n / 2) {
                        vi = -vi;
                    }
                }
            } else if (i < inLen) {
                if (inKind == REAL) {
                    vr = in.get(inBase + i * inner);
                } else {
                    vr = in.get(2 * (inBase + i * inner));
                    vi = in.get(2 * (inBase + i * inner) + 1);
                }
            }
            re[rev] = vr;
            im[rev] = vi;
        }
        ctx.localBarrier();
        float sign = inverse != 0 ? 1.0f : -1.0f;
        for (int half = 1; half < n; half <<= 1) {
            for (int b = tid; b < n / 2; b += THREADS) {
                int group = b / half;
                int pos = b % half;
                int top = group * 2 * half + pos;
                int bottom = top + half;
                float angle = sign * TWO_PI * pos / (2 * half);
                float wr = TornadoMath.cos(angle);
                float wi = TornadoMath.sin(angle);
                float br = re[bottom] * wr - im[bottom] * wi;
                float bi = re[bottom] * wi + im[bottom] * wr;
                float tr = re[top];
                float ti = im[top];
                re[top] = tr + br;
                im[top] = ti + bi;
                re[bottom] = tr - br;
                im[bottom] = ti - bi;
            }
            ctx.localBarrier();
        }
        for (int k = tid; k < outLen; k += THREADS) {
            if (realOut != 0) {
                out.set(outBase + k * inner, re[k] * scale);
            } else {
                out.set(2 * (outBase + k * inner), re[k] * scale);
                out.set(2 * (outBase + k * inner) + 1, im[k] * scale);
            }
        }
    }

    /** Direct DFT of each line, for any n: one thread per output coefficient of every line; same layouts as {@link #fft}. */
    @JitBaseline({ "mlx_fft_fft", "mlx_fft_ifft", "mlx_fft_rfft", "mlx_fft_irfft", "mlx_fft_fft2", "mlx_fft_ifft2", "mlx_fft_rfft2", "mlx_fft_irfft2", "mlx_fft_fftn", "mlx_fft_ifftn",
            "mlx_fft_rfftn", "mlx_fft_irfftn" })
    public static void dft(KernelContext ctx, FloatArray in, FloatArray out, int lines, int n, int inLen, int outLen, int inner, int inKind, int realOut, int inverse, float scale) {
        int t = ctx.globalIdx;
        if (t < lines * outLen) {
            int line = t / outLen;
            int k = t % outLen;
            int o = line / inner;
            int j = line % inner;
            int inBase = o * inLen * inner + j;
            float sign = inverse != 0 ? 1.0f : -1.0f;
            float accR = 0.0f;
            float accI = 0.0f;
            for (int i = 0; i < n; i++) {
                float vr = 0.0f;
                float vi = 0.0f;
                if (inKind == HERMITIAN) {
                    int m = i <= n / 2 ? i : n - i;
                    if (m < inLen) {
                        vr = in.get(2 * (inBase + m * inner));
                        vi = in.get(2 * (inBase + m * inner) + 1);
                        if (i > n / 2) {
                            vi = -vi;
                        }
                    }
                } else if (i < inLen) {
                    if (inKind == REAL) {
                        vr = in.get(inBase + i * inner);
                    } else {
                        vr = in.get(2 * (inBase + i * inner));
                        vi = in.get(2 * (inBase + i * inner) + 1);
                    }
                }
                // (i * k) mod n keeps the angle small and accurate.
                int ik = (int) (((long) i * k) % n);
                float angle = sign * TWO_PI * ik / n;
                float wr = TornadoMath.cos(angle);
                float wi = TornadoMath.sin(angle);
                accR += vr * wr - vi * wi;
                accI += vr * wi + vi * wr;
            }
            int at = o * outLen * inner + k * inner + j;
            if (realOut != 0) {
                out.set(at, accR * scale);
            } else {
                out.set(2 * at, accR * scale);
                out.set(2 * at + 1, accI * scale);
            }
        }
    }

    /**
     * out = x rolled along its last axis by {@code shift} (n / 2 for fftshift, -(n / 2) for
     * ifftshift): out[r, (i + shift) mod len] = x[r, i]; one thread per element.
     */
    @JitBaseline({ "mlx_fft_fftshift", "mlx_fft_ifftshift" })
    public static void roll(KernelContext ctx, FloatArray x, FloatArray out, int total, int len, int shift) {
        int t = ctx.globalIdx;
        if (t < total) {
            int r = t / len;
            int i = t % len;
            int dst = ((i + shift) % len + len) % len;
            out.set(r * len + dst, x.get(t));
        }
    }

    /**
     * Sample frequencies: fftfreq ({@code real == 0}), out[i] = i / (d n) for i &lt; (n + 1) / 2 and
     * (i - n) / (d n) after; rfftfreq, out[i] = i / (d n) for i &lt;= n / 2.
     */
    @JitBaseline({ "mlx_fft_fftfreq", "mlx_fft_rfftfreq" })
    public static void frequencies(KernelContext ctx, FloatArray out, int count, int n, float d, int real) {
        int i = ctx.globalIdx;
        if (i < count) {
            int k = real == 0 && i >= (n + 1) / 2 ? i - n : i;
            out.set(i, k / (d * n));
        }
    }
}
