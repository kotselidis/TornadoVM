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
 * JIT counterparts of the MLX element-wise operations, float32. One thread per element; launch
 * with a 1D grid of at least {@code n} threads (e.g. rounded up to a multiple of 256). Follows the
 * shape of {@code TestVectorAdditionKernelContext}.
 */
public final class JitElementwise {

    private JitElementwise() {
    }

    @JitBaseline(value = "mlx_add", source = "tornado-unittests/.../kernelcontext/api/TestVectorAdditionKernelContext.java (pattern)")
    public static void add(KernelContext context, FloatArray a, FloatArray b, FloatArray c, int n) {
        int i = context.globalIdx;
        if (i < n) {
            c.set(i, a.get(i) + b.get(i));
        }
    }

    @JitBaseline("mlx_subtract")
    public static void subtract(KernelContext context, FloatArray a, FloatArray b, FloatArray c, int n) {
        int i = context.globalIdx;
        if (i < n) {
            c.set(i, a.get(i) - b.get(i));
        }
    }

    @JitBaseline("mlx_multiply")
    public static void multiply(KernelContext context, FloatArray a, FloatArray b, FloatArray c, int n) {
        int i = context.globalIdx;
        if (i < n) {
            c.set(i, a.get(i) * b.get(i));
        }
    }

    @JitBaseline("mlx_divide")
    public static void divide(KernelContext context, FloatArray a, FloatArray b, FloatArray c, int n) {
        int i = context.globalIdx;
        if (i < n) {
            c.set(i, a.get(i) / b.get(i));
        }
    }

    @JitBaseline("mlx_maximum")
    public static void maximum(KernelContext context, FloatArray a, FloatArray b, FloatArray c, int n) {
        int i = context.globalIdx;
        if (i < n) {
            c.set(i, TornadoMath.max(a.get(i), b.get(i)));
        }
    }

    @JitBaseline("mlx_minimum")
    public static void minimum(KernelContext context, FloatArray a, FloatArray b, FloatArray c, int n) {
        int i = context.globalIdx;
        if (i < n) {
            c.set(i, TornadoMath.min(a.get(i), b.get(i)));
        }
    }

    @JitBaseline("mlx_negative")
    public static void negative(KernelContext context, FloatArray a, FloatArray out, int n) {
        int i = context.globalIdx;
        if (i < n) {
            out.set(i, -a.get(i));
        }
    }

    @JitBaseline("mlx_exp")
    public static void exp(KernelContext context, FloatArray a, FloatArray out, int n) {
        int i = context.globalIdx;
        if (i < n) {
            out.set(i, TornadoMath.exp(a.get(i)));
        }
    }

    @JitBaseline("mlx_tanh")
    public static void tanh(KernelContext context, FloatArray a, FloatArray out, int n) {
        int i = context.globalIdx;
        if (i < n) {
            out.set(i, TornadoMath.tanh(a.get(i)));
        }
    }

    /**
     * erf through the Numerical Recipes erfc Chebyshev fit (|error| &lt; 1.2e-7); TornadoMath has
     * no erf.
     */
    @JitBaseline("mlx_erf")
    public static void erf(KernelContext context, FloatArray a, FloatArray out, int n) {
        int i = context.globalIdx;
        if (i < n) {
            float x = a.get(i);
            float t = 1.0f / (1.0f + 0.5f * TornadoMath.abs(x));
            float poly = -x * x - 1.26551223f + t * (1.00002368f + t * (0.37409196f + t * (0.09678418f + t * (-0.18628806f + t * (0.27886807f + t * (-1.13520398f + t * (1.48851587f
                    + t * (-0.82215223f + t * 0.17087277f))))))));
            float y = t * TornadoMath.exp(poly);
            out.set(i, x >= 0 ? 1.0f - y : y - 1.0f);
        }
    }

    @JitBaseline("mlx_sigmoid")
    public static void sigmoid(KernelContext context, FloatArray a, FloatArray out, int n) {
        int i = context.globalIdx;
        if (i < n) {
            out.set(i, 1.0f / (1.0f + TornadoMath.exp(-a.get(i))));
        }
    }

    @JitBaseline("mlx_sqrt")
    public static void sqrt(KernelContext context, FloatArray a, FloatArray out, int n) {
        int i = context.globalIdx;
        if (i < n) {
            out.set(i, TornadoMath.sqrt(a.get(i)));
        }
    }

    @JitBaseline("mlx_rsqrt")
    public static void rsqrt(KernelContext context, FloatArray a, FloatArray out, int n) {
        int i = context.globalIdx;
        if (i < n) {
            out.set(i, 1.0f / TornadoMath.sqrt(a.get(i)));
        }
    }

    @JitBaseline("mlx_square")
    public static void square(KernelContext context, FloatArray a, FloatArray out, int n) {
        int i = context.globalIdx;
        if (i < n) {
            float v = a.get(i);
            out.set(i, v * v);
        }
    }
}
