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

/**
 * JIT counterparts of the MLX element-wise math operations ({@link uk.ac.manchester.tornado.mlx.MlxMath}),
 * float32, one thread per element; launch with a 1D grid of at least {@code n} threads. Functions
 * TornadoMath lacks (sinh, cosh, expm1, log1p, erfinv, round-half-even) are built from exp, log and
 * floor.
 */
public final class JitMath {

    private JitMath() {
    }

    @JitBaseline("mlx_abs")
    public static void abs(KernelContext context, FloatArray a, FloatArray out, int n) {
        int i = context.globalIdx;
        if (i < n) {
            float x = a.get(i);
            out.set(i, TornadoMath.abs(x));
        }
    }

    @JitBaseline("mlx_arccos")
    public static void arccos(KernelContext context, FloatArray a, FloatArray out, int n) {
        int i = context.globalIdx;
        if (i < n) {
            float x = a.get(i);
            out.set(i, TornadoMath.acos(x));
        }
    }

    @JitBaseline("mlx_arccosh")
    public static void arccosh(KernelContext context, FloatArray a, FloatArray out, int n) {
        int i = context.globalIdx;
        if (i < n) {
            float x = a.get(i);
            out.set(i, TornadoMath.acosh(x));
        }
    }

    @JitBaseline("mlx_arcsin")
    public static void arcsin(KernelContext context, FloatArray a, FloatArray out, int n) {
        int i = context.globalIdx;
        if (i < n) {
            float x = a.get(i);
            out.set(i, TornadoMath.asin(x));
        }
    }

    @JitBaseline("mlx_arcsinh")
    public static void arcsinh(KernelContext context, FloatArray a, FloatArray out, int n) {
        int i = context.globalIdx;
        if (i < n) {
            float x = a.get(i);
            out.set(i, TornadoMath.asinh(x));
        }
    }

    @JitBaseline("mlx_arctan")
    public static void arctan(KernelContext context, FloatArray a, FloatArray out, int n) {
        int i = context.globalIdx;
        if (i < n) {
            float x = a.get(i);
            out.set(i, TornadoMath.atan(x));
        }
    }

    @JitBaseline("mlx_arctanh")
    public static void arctanh(KernelContext context, FloatArray a, FloatArray out, int n) {
        int i = context.globalIdx;
        if (i < n) {
            float x = a.get(i);
            out.set(i, 0.5f * TornadoMath.log((1.0f + x) / (1.0f - x)));
        }
    }

    @JitBaseline("mlx_ceil")
    public static void ceil(KernelContext context, FloatArray a, FloatArray out, int n) {
        int i = context.globalIdx;
        if (i < n) {
            float x = a.get(i);
            out.set(i, TornadoMath.ceil(x));
        }
    }

    @JitBaseline("mlx_cos")
    public static void cos(KernelContext context, FloatArray a, FloatArray out, int n) {
        int i = context.globalIdx;
        if (i < n) {
            float x = a.get(i);
            out.set(i, TornadoMath.cos(x));
        }
    }

    @JitBaseline("mlx_cosh")
    public static void cosh(KernelContext context, FloatArray a, FloatArray out, int n) {
        int i = context.globalIdx;
        if (i < n) {
            float x = a.get(i);
            float e = TornadoMath.exp(x);
            out.set(i, 0.5f * (e + 1.0f / e));
        }
    }

    @JitBaseline("mlx_degrees")
    public static void degrees(KernelContext context, FloatArray a, FloatArray out, int n) {
        int i = context.globalIdx;
        if (i < n) {
            float x = a.get(i);
            out.set(i, x * 57.29577951308232f);
        }
    }

    @JitBaseline("mlx_erfinv")
    public static void erfinv(KernelContext context, FloatArray a, FloatArray out, int n) {
        int i = context.globalIdx;
        if (i < n) {
            float x = a.get(i);
            // Giles, "Approximating the erfinv function" (GPU Computing Gems, 2011), single precision.
            float w = -TornadoMath.log((1.0f - x) * (1.0f + x));
            float p;
            if (w < 5.0f) {
                w = w - 2.5f;
                p = 2.81022636e-08f;
                p = 3.43273939e-07f + p * w;
                p = -3.5233877e-06f + p * w;
                p = -4.39150654e-06f + p * w;
                p = 0.00021858087f + p * w;
                p = -0.00125372503f + p * w;
                p = -0.00417768164f + p * w;
                p = 0.246640727f + p * w;
                p = 1.50140941f + p * w;
            } else {
                w = TornadoMath.sqrt(w) - 3.0f;
                p = -0.000200214257f;
                p = 0.000100950558f + p * w;
                p = 0.00134934322f + p * w;
                p = -0.00367342844f + p * w;
                p = 0.00573950773f + p * w;
                p = -0.0076224613f + p * w;
                p = 0.00943887047f + p * w;
                p = 1.00167406f + p * w;
                p = 2.83297682f + p * w;
            }
            out.set(i, p * x);
        }
    }

    @JitBaseline("mlx_expm1")
    public static void expm1(KernelContext context, FloatArray a, FloatArray out, int n) {
        int i = context.globalIdx;
        if (i < n) {
            float x = a.get(i);
            // Kahan's formulation: exact where exp(x) rounds to 1.
            float u = TornadoMath.exp(x);
            float r;
            if (u == 1.0f) {
                r = x;
            } else if (u - 1.0f == -1.0f) {
                r = -1.0f;
            } else {
                r = (u - 1.0f) * x / TornadoMath.log(u);
            }
            out.set(i, r);
        }
    }

    @JitBaseline("mlx_floor")
    public static void floor(KernelContext context, FloatArray a, FloatArray out, int n) {
        int i = context.globalIdx;
        if (i < n) {
            float x = a.get(i);
            out.set(i, TornadoMath.floor(x));
        }
    }

    @JitBaseline("mlx_log")
    public static void log(KernelContext context, FloatArray a, FloatArray out, int n) {
        int i = context.globalIdx;
        if (i < n) {
            float x = a.get(i);
            out.set(i, TornadoMath.log(x));
        }
    }

    @JitBaseline("mlx_log10")
    public static void log10(KernelContext context, FloatArray a, FloatArray out, int n) {
        int i = context.globalIdx;
        if (i < n) {
            float x = a.get(i);
            out.set(i, TornadoMath.log(x) * 0.4342944819032518f);
        }
    }

    @JitBaseline("mlx_log1p")
    public static void log1p(KernelContext context, FloatArray a, FloatArray out, int n) {
        int i = context.globalIdx;
        if (i < n) {
            float x = a.get(i);
            // Kahan's formulation: exact where 1 + x rounds to 1.
            float u = 1.0f + x;
            out.set(i, u == 1.0f ? x : TornadoMath.log(u) * x / (u - 1.0f));
        }
    }

    @JitBaseline("mlx_log2")
    public static void log2(KernelContext context, FloatArray a, FloatArray out, int n) {
        int i = context.globalIdx;
        if (i < n) {
            float x = a.get(i);
            out.set(i, TornadoMath.log2(x));
        }
    }

    @JitBaseline("mlx_radians")
    public static void radians(KernelContext context, FloatArray a, FloatArray out, int n) {
        int i = context.globalIdx;
        if (i < n) {
            float x = a.get(i);
            out.set(i, x * 0.017453292519943295f);
        }
    }

    @JitBaseline("mlx_reciprocal")
    public static void reciprocal(KernelContext context, FloatArray a, FloatArray out, int n) {
        int i = context.globalIdx;
        if (i < n) {
            float x = a.get(i);
            out.set(i, 1.0f / x);
        }
    }

    @JitBaseline("mlx_sign")
    public static void sign(KernelContext context, FloatArray a, FloatArray out, int n) {
        int i = context.globalIdx;
        if (i < n) {
            float x = a.get(i);
            out.set(i, TornadoMath.signum(x));
        }
    }

    @JitBaseline("mlx_sin")
    public static void sin(KernelContext context, FloatArray a, FloatArray out, int n) {
        int i = context.globalIdx;
        if (i < n) {
            float x = a.get(i);
            out.set(i, TornadoMath.sin(x));
        }
    }

    @JitBaseline("mlx_sinh")
    public static void sinh(KernelContext context, FloatArray a, FloatArray out, int n) {
        int i = context.globalIdx;
        if (i < n) {
            float x = a.get(i);
            float r;
            if (TornadoMath.abs(x) < 0.5f) {
                // Series near zero, where exp(x) - exp(-x) cancels.
                float x2 = x * x;
                r = x * (1.0f + x2 * (1.0f / 6.0f + x2 * (1.0f / 120.0f + x2 * (1.0f / 5040.0f))));
            } else {
                float e = TornadoMath.exp(x);
                r = 0.5f * (e - 1.0f / e);
            }
            out.set(i, r);
        }
    }

    @JitBaseline("mlx_tan")
    public static void tan(KernelContext context, FloatArray a, FloatArray out, int n) {
        int i = context.globalIdx;
        if (i < n) {
            float x = a.get(i);
            out.set(i, TornadoMath.tan(x));
        }
    }

    /** Round to {@code decimals} places, ties to even (MLX and NumPy semantics). */
    @JitBaseline("mlx_round")
    public static void round(KernelContext context, FloatArray a, FloatArray out, int n, int decimals) {
        int i = context.globalIdx;
        if (i < n) {
            float scale = 1.0f;
            for (int k = 0; k < decimals; k++) {
                scale *= 10.0f;
            }
            float y = decimals == 0 ? a.get(i) : a.get(i) * scale;
            float r = TornadoMath.floor(y);
            float d = y - r;
            if (d > 0.5f || (d == 0.5f && r - 2.0f * TornadoMath.floor(0.5f * r) != 0.0f)) {
                r += 1.0f;
            }
            out.set(i, decimals == 0 ? r : r / scale);
        }
    }

    @JitBaseline("mlx_arctan2")
    public static void arctan2(KernelContext context, FloatArray a, FloatArray b, FloatArray out, int n) {
        int i = context.globalIdx;
        if (i < n) {
            float x = a.get(i);
            float y = b.get(i);
            out.set(i, TornadoMath.atan2(x, y));
        }
    }

    @JitBaseline("mlx_floor_divide")
    public static void floorDivide(KernelContext context, FloatArray a, FloatArray b, FloatArray out, int n) {
        int i = context.globalIdx;
        if (i < n) {
            float x = a.get(i);
            float y = b.get(i);
            out.set(i, TornadoMath.floor(x / y));
        }
    }

    @JitBaseline("mlx_logaddexp")
    public static void logaddexp(KernelContext context, FloatArray a, FloatArray b, FloatArray out, int n) {
        int i = context.globalIdx;
        if (i < n) {
            float x = a.get(i);
            float y = b.get(i);
            float m = TornadoMath.max(x, y);
            float d = TornadoMath.abs(x - y);
            // log1p(exp(-d)) through Kahan's formulation.
            float e = TornadoMath.exp(-d);
            float u = 1.0f + e;
            out.set(i, m + (u == 1.0f ? e : TornadoMath.log(u) * e / (u - 1.0f)));
        }
    }

    @JitBaseline("mlx_power")
    public static void power(KernelContext context, FloatArray a, FloatArray b, FloatArray out, int n) {
        int i = context.globalIdx;
        if (i < n) {
            float x = a.get(i);
            float y = b.get(i);
            out.set(i, TornadoMath.pow(x, y));
        }
    }

    @JitBaseline("mlx_remainder")
    public static void remainder(KernelContext context, FloatArray a, FloatArray b, FloatArray out, int n) {
        int i = context.globalIdx;
        if (i < n) {
            float x = a.get(i);
            float y = b.get(i);
            // fmod, moved to the sign of the divisor as in Python and MLX.
            float r = x - y * (float) (int) (x / y);
            if (r != 0.0f && (r < 0.0f) != (y < 0.0f)) {
                r += y;
            }
            out.set(i, r);
        }
    }

    @JitBaseline("mlx_divmod")
    public static void divmod(KernelContext context, FloatArray a, FloatArray b, FloatArray quotient, FloatArray remainder, int n) {
        int i = context.globalIdx;
        if (i < n) {
            float x = a.get(i);
            float y = b.get(i);
            // As MLX's divmod: quotient rounded toward zero, remainder with the sign of the divisor.
            float q = (float) (int) (x / y);
            float r = x - y * q;
            if (r != 0.0f && (r < 0.0f) != (y < 0.0f)) {
                r += y;
            }
            quotient.set(i, q);
            remainder.set(i, r);
        }
    }

    @JitBaseline("mlx_clip")
    public static void clip(KernelContext context, FloatArray a, FloatArray out, int n, float lo, float hi) {
        int i = context.globalIdx;
        if (i < n) {
            out.set(i, TornadoMath.min(TornadoMath.max(a.get(i), lo), hi));
        }
    }

    @JitBaseline("mlx_where")
    public static void where(KernelContext context, ByteArray condition, FloatArray x, FloatArray y, FloatArray out, int n) {
        int i = context.globalIdx;
        if (i < n) {
            out.set(i, condition.get(i) != 0 ? x.get(i) : y.get(i));
        }
    }
}
