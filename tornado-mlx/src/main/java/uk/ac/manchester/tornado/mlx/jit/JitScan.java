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
 * JIT counterparts of the MLX scans (cumsum, cumprod, cummax, cummin, logcumsumexp), float32,
 * along the middle axis of an input viewed as {@code [outer, len, inner]}:
 * <ul>
 * <li><b>rows</b> ({@code inner == 1}): one threadgroup of {@link #THREADS} threads per row, which
 * scans the row in chunks of {@link #THREADS} with a Hillis-Steele scan in threadgroup memory and
 * carries each chunk's total into the next;</li>
 * <li><b>columns</b> ({@code inner > 1}): one thread per {@code (o, j)}, scanning sequentially
 * with stride {@code inner}, so neighbouring threads read neighbouring addresses.</li>
 * </ul>
 * {@code reverse} scans from the end; an exclusive scan ({@code inclusive == 0}) shifts the result
 * by one and starts from the operation's identity.
 */
public final class JitScan {

    /** Threads per threadgroup for row scans (a power of two). */
    public static final int THREADS = 256;

    public static final int SUM = 0;
    public static final int PROD = 1;
    public static final int MAX = 2;
    public static final int MIN = 3;
    /** log(exp(a) + exp(b)). */
    public static final int LOGADDEXP = 4;

    /**
     * The identity of LOGADDEXP. Metal compiles with fast math, which assumes there are no
     * infinities, so log(0) is represented by -FLT_MAX inside the kernels and written as -inf.
     */
    private static final float LOG_ZERO = -Float.MAX_VALUE;

    private JitScan() {
    }

    private static float identity(int op) {
        float id = 0.0f;
        if (op == PROD) {
            id = 1.0f;
        } else if (op == MAX) {
            id = Float.NEGATIVE_INFINITY;
        } else if (op == LOGADDEXP) {
            id = LOG_ZERO;
        } else if (op == MIN) {
            id = Float.POSITIVE_INFINITY;
        }
        return id;
    }

    private static float combine(float a, float b, int op) {
        float r;
        if (op == SUM) {
            r = a + b;
        } else if (op == PROD) {
            r = a * b;
        } else if (op == MAX) {
            r = TornadoMath.max(a, b);
        } else if (op == MIN) {
            r = TornadoMath.min(a, b);
        } else if (a == LOG_ZERO) {
            r = b;
        } else if (b == LOG_ZERO) {
            r = a;
        } else {
            // max + log1p(exp(-|a - b|)), through Kahan's log1p.
            float m = TornadoMath.max(a, b);
            float e = TornadoMath.exp(-TornadoMath.abs(a - b));
            float u = 1.0f + e;
            r = m + (u == 1.0f ? e : TornadoMath.log(u) * e / (u - 1.0f));
        }
        return r;
    }

    /** Maps the in-kernel LOGADDEXP identity back to -infinity. */
    private static float finish(float v, int op) {
        return op == LOGADDEXP && v == LOG_ZERO ? Float.NEGATIVE_INFINITY : v;
    }

    /** Scan of each row of x[rows, len], one threadgroup per row. */
    @JitBaseline({ "mlx_cumsum", "mlx_cumprod", "mlx_cummax", "mlx_cummin", "mlx_logcumsumexp" })
    public static void scanRows(KernelContext ctx, FloatArray x, FloatArray out, int len, int op, int reverse, int inclusive) {
        float[] buf = ctx.allocateFloatLocalArray(THREADS);
        int row = ctx.groupIdx;
        int tid = ctx.localIdx;
        int base = row * len;
        float carry = identity(op);
        for (int start = 0; start < len; start += THREADS) {
            int i = start + tid;
            int pos = reverse != 0 ? len - 1 - i : i;
            buf[tid] = i < len ? x.get(base + pos) : identity(op);
            ctx.localBarrier();
            for (int offset = 1; offset < THREADS; offset <<= 1) {
                float v = buf[tid];
                if (tid >= offset) {
                    v = combine(buf[tid - offset], v, op);
                }
                ctx.localBarrier();
                buf[tid] = v;
                ctx.localBarrier();
            }
            if (i < len) {
                float value;
                if (inclusive != 0) {
                    value = combine(carry, buf[tid], op);
                } else {
                    value = tid == 0 ? carry : combine(carry, buf[tid - 1], op);
                }
                out.set(base + pos, finish(value, op));
            }
            carry = combine(carry, buf[THREADS - 1], op);
            ctx.localBarrier();
        }
    }

    /** Scan along the middle axis of x[outer, len, inner], one thread per (o, j). */
    @JitBaseline({ "mlx_cumsum", "mlx_cumprod", "mlx_cummax", "mlx_cummin", "mlx_logcumsumexp" })
    public static void scanColumns(KernelContext ctx, FloatArray x, FloatArray out, int outputs, int len, int inner, int op, int reverse, int inclusive) {
        int idx = ctx.globalIdx;
        if (idx < outputs) {
            int base = (idx / inner) * len * inner + idx % inner;
            float acc = identity(op);
            for (int k = 0; k < len; k++) {
                int i = reverse != 0 ? len - 1 - k : k;
                float v = x.get(base + i * inner);
                if (inclusive != 0) {
                    acc = combine(acc, v, op);
                    out.set(base + i * inner, finish(acc, op));
                } else {
                    out.set(base + i * inner, finish(acc, op));
                    acc = combine(acc, v, op);
                }
            }
        }
    }
}
