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
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * JIT counterparts of the MLX comparisons, logical, bitwise and complex-part operations
 * ({@link uk.ac.manchester.tornado.mlx.MlxLogic}), one thread per element. Metal compiles with fast
 * math, which may assume there are no NaNs or infinities, so NaN and infinity checks inspect the
 * float's bits instead of comparing values.
 */
public final class JitLogic {

    public static final int EQUAL = 0;
    public static final int NOT_EQUAL = 1;
    public static final int GREATER = 2;
    public static final int GREATER_EQUAL = 3;
    public static final int LESS = 4;
    public static final int LESS_EQUAL = 5;

    public static final int FINITE = 0;
    public static final int INF = 1;
    public static final int NAN = 2;
    public static final int NEG_INF = 3;
    public static final int POS_INF = 4;

    public static final int AND = 0;
    public static final int OR = 1;
    public static final int XOR = 2;
    public static final int LEFT_SHIFT = 3;
    public static final int RIGHT_SHIFT = 4;

    private JitLogic() {
    }

    private static boolean isNan(float v) {
        int bits = Float.floatToRawIntBits(v);
        return (bits & 0x7f800000) == 0x7f800000 && (bits & 0x007fffff) != 0;
    }

    private static boolean isInf(float v) {
        return (Float.floatToRawIntBits(v) & 0x7fffffff) == 0x7f800000;
    }

    @JitBaseline({ "mlx_equal", "mlx_not_equal", "mlx_greater", "mlx_greater_equal", "mlx_less", "mlx_less_equal" })
    public static void compare(KernelContext context, FloatArray a, FloatArray b, ByteArray out, int n, int op) {
        int i = context.globalIdx;
        if (i < n) {
            float x = a.get(i);
            float y = b.get(i);
            boolean r;
            if (isNan(x) || isNan(y)) {
                // Any comparison with NaN is false except !=, which fast math would not guarantee.
                r = op == NOT_EQUAL;
            } else if (op == EQUAL) {
                r = x == y;
            } else if (op == NOT_EQUAL) {
                r = x != y;
            } else if (op == GREATER) {
                r = x > y;
            } else if (op == GREATER_EQUAL) {
                r = x >= y;
            } else if (op == LESS) {
                r = x < y;
            } else {
                r = x <= y;
            }
            out.set(i, (byte) (r ? 1 : 0));
        }
    }

    /** out[i] = |a - b| <= atol + rtol * |b| (NaNs equal each other if {@code equalNan != 0}). */
    @JitBaseline("mlx_isclose")
    public static void isclose(KernelContext context, FloatArray a, FloatArray b, ByteArray out, int n, float rtol, float atol, int equalNan) {
        int i = context.globalIdx;
        if (i < n) {
            float x = a.get(i);
            float y = b.get(i);
            boolean r;
            if (isNan(x) || isNan(y)) {
                r = equalNan != 0 && isNan(x) && isNan(y);
            } else if (isInf(x) || isInf(y)) {
                r = x == y;
            } else {
                r = TornadoMath.abs(x - y) <= atol + rtol * TornadoMath.abs(y);
            }
            out.set(i, (byte) (r ? 1 : 0));
        }
    }

    /** out[0] = value: the first step of allclose and array_equal, which only ever clear it. */
    @JitBaseline({ "mlx_allclose", "mlx_array_equal" })
    public static void setFlag(KernelContext context, ByteArray out, int value) {
        if (context.globalIdx == 0) {
            out.set(0, (byte) value);
        }
    }

    /** Clears out[0] if any pair is not close; grid-strided over n. */
    @JitBaseline("mlx_allclose")
    public static void allcloseCheck(KernelContext context, FloatArray a, FloatArray b, ByteArray out, int n, float rtol, float atol, int equalNan) {
        for (int i = context.globalIdx; i < n; i += context.globalGroupSizeX) {
            float x = a.get(i);
            float y = b.get(i);
            boolean close;
            if (isNan(x) || isNan(y)) {
                close = equalNan != 0 && isNan(x) && isNan(y);
            } else if (isInf(x) || isInf(y)) {
                close = x == y;
            } else {
                close = TornadoMath.abs(x - y) <= atol + rtol * TornadoMath.abs(y);
            }
            if (!close) {
                out.set(0, (byte) 0);
            }
        }
    }

    /** Clears out[0] if any pair differs; grid-strided over n. */
    @JitBaseline("mlx_array_equal")
    public static void arrayEqualCheck(KernelContext context, FloatArray a, FloatArray b, ByteArray out, int n, int equalNan) {
        for (int i = context.globalIdx; i < n; i += context.globalGroupSizeX) {
            float x = a.get(i);
            float y = b.get(i);
            boolean same = (equalNan != 0 && isNan(x) && isNan(y)) || (!isNan(x) && x == y);
            if (!same) {
                out.set(0, (byte) 0);
            }
        }
    }

    @JitBaseline({ "mlx_isfinite", "mlx_isinf", "mlx_isnan", "mlx_isneginf", "mlx_isposinf" })
    public static void classify(KernelContext context, FloatArray a, ByteArray out, int n, int kind) {
        int i = context.globalIdx;
        if (i < n) {
            int bits = Float.floatToRawIntBits(a.get(i));
            boolean special = (bits & 0x7f800000) == 0x7f800000;
            boolean mantissa = (bits & 0x007fffff) != 0;
            boolean r;
            if (kind == FINITE) {
                r = !special;
            } else if (kind == NAN) {
                r = special && mantissa;
            } else if (kind == INF) {
                r = special && !mantissa;
            } else if (kind == NEG_INF) {
                r = special && !mantissa && bits < 0;
            } else {
                r = special && !mantissa && bits > 0;
            }
            out.set(i, (byte) (r ? 1 : 0));
        }
    }

    @JitBaseline({ "mlx_logical_and", "mlx_logical_or" })
    public static void logical(KernelContext context, ByteArray a, ByteArray b, ByteArray out, int n, int op) {
        int i = context.globalIdx;
        if (i < n) {
            boolean x = a.get(i) != 0;
            boolean y = b.get(i) != 0;
            out.set(i, (byte) ((op == AND ? x && y : x || y) ? 1 : 0));
        }
    }

    @JitBaseline("mlx_logical_not")
    public static void logicalNot(KernelContext context, ByteArray a, ByteArray out, int n) {
        int i = context.globalIdx;
        if (i < n) {
            out.set(i, (byte) (a.get(i) != 0 ? 0 : 1));
        }
    }

    @JitBaseline({ "mlx_bitwise_and", "mlx_bitwise_or", "mlx_bitwise_xor", "mlx_left_shift", "mlx_right_shift" })
    public static void bitwise(KernelContext context, IntArray a, IntArray b, IntArray out, int n, int op) {
        int i = context.globalIdx;
        if (i < n) {
            int x = a.get(i);
            int y = b.get(i);
            int r;
            if (op == AND) {
                r = x & y;
            } else if (op == OR) {
                r = x | y;
            } else if (op == XOR) {
                r = x ^ y;
            } else if (op == LEFT_SHIFT) {
                r = x << y;
            } else {
                r = x >> y;
            }
            out.set(i, r);
        }
    }

    @JitBaseline("mlx_bitwise_invert")
    public static void invert(KernelContext context, IntArray a, IntArray out, int n) {
        int i = context.globalIdx;
        if (i < n) {
            out.set(i, ~a.get(i));
        }
    }

    /** NaN to {@code nan}, +infinity to {@code posinf}, -infinity to {@code neginf}; other values unchanged. */
    @JitBaseline("mlx_nan_to_num")
    public static void nanToNum(KernelContext context, FloatArray a, FloatArray out, int n, float nan, float posinf, float neginf) {
        int i = context.globalIdx;
        if (i < n) {
            float v = a.get(i);
            int bits = Float.floatToRawIntBits(v);
            float r = v;
            if ((bits & 0x7f800000) == 0x7f800000) {
                if ((bits & 0x007fffff) != 0) {
                    r = nan;
                } else {
                    r = bits < 0 ? neginf : posinf;
                }
            }
            out.set(i, r);
        }
    }

    /** out[i] = the real ({@code part == 0}) or imaginary part of complex z[i] (interleaved pairs). */
    @JitBaseline({ "mlx_real", "mlx_imag" })
    public static void complexPart(KernelContext context, FloatArray z, FloatArray out, int n, int part) {
        int i = context.globalIdx;
        if (i < n) {
            out.set(i, z.get(2 * i + part));
        }
    }

    @JitBaseline("mlx_conjugate")
    public static void conjugate(KernelContext context, FloatArray z, FloatArray out, int n) {
        int i = context.globalIdx;
        if (i < n) {
            out.set(2 * i, z.get(2 * i));
            out.set(2 * i + 1, -z.get(2 * i + 1));
        }
    }
}
