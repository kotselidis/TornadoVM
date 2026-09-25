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
package uk.ac.manchester.tornado.unittests.mlx;

import static org.junit.Assert.assertEquals;

import java.util.Arrays;
import java.util.function.DoubleBinaryOperator;

import org.junit.Test;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.common.TornadoFunctions.LibraryTask7;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.mlx.MlxReduce;
import uk.ac.manchester.tornado.mlx.jit.JitScan;

/**
 * The MLX scans and their KernelContext JIT counterparts, run in one graph on the same input and
 * checked against a sequential Java scan: rows (one threadgroup per row, several chunks per row),
 * a long single row, and strided columns, each forward, reversed and exclusive. Also checks the
 * int32 and float16 forms of the MLX tasks.
 *
 * <p>
 * How to run?
 * </p>
 * <code>
 * tornado-test -V uk.ac.manchester.tornado.unittests.mlx.TestJitScan
 * </code>
 */
public class TestJitScan extends MlxTestBase {

    /** {outer, len, inner}. */
    private static final int[][] SHAPES = { { 9, 700, 1 }, { 1, 5000, 1 }, { 5, 40, 13 } };
    /** {reverse, inclusive}. */
    private static final boolean[][] MODES = { { false, true }, { true, true }, { false, false }, { true, false } };

    private static double[] reference(float[] x, int outer, int len, int inner, boolean reverse, boolean inclusive, double identity, DoubleBinaryOperator op) {
        double[] out = new double[x.length];
        for (int o = 0; o < outer; o++) {
            for (int j = 0; j < inner; j++) {
                double acc = identity;
                for (int k = 0; k < len; k++) {
                    int i = reverse ? len - 1 - k : k;
                    int at = (o * len + i) * inner + j;
                    if (inclusive) {
                        acc = op.applyAsDouble(acc, x[at]);
                        out[at] = acc;
                    } else {
                        out[at] = acc;
                        acc = op.applyAsDouble(acc, x[at]);
                    }
                }
            }
        }
        return out;
    }

    private static double logaddexp(double a, double b) {
        double m = Math.max(a, b);
        return m == Double.NEGATIVE_INFINITY ? m : m + Math.log1p(Math.exp(-Math.abs(a - b)));
    }

    private static void check(String name, LibraryTask7<FloatArray, FloatArray, Integer, Integer, Integer, Boolean, Boolean> mlx, int op, double identity, DoubleBinaryOperator ref,
            float lo, float hi, double relTol, double absTol) throws TornadoExecutionPlanException {
        for (int[] shape : SHAPES) {
            int outer = shape[0];
            int len = shape[1];
            int inner = shape[2];
            for (boolean[] mode : MODES) {
                boolean reverse = mode[0];
                boolean inclusive = mode[1];
                float[] xv = values(outer * len * inner, lo, hi, 81L + len);
                FloatArray x = FloatArray.fromArray(xv);
                FloatArray outMlx = new FloatArray(xv.length);
                FloatArray outJit = new FloatArray(xv.length);
                TaskGraph g = new TaskGraph("sc").transferToDevice(DataTransferMode.FIRST_EXECUTION, x) //
                        .libraryTask("mlx", mlx, x, outMlx, outer, len, inner, reverse, inclusive);
                GridScheduler gs = new GridScheduler();
                if (inner == 1) {
                    g.task("j", JitScan::scanRows, new KernelContext(), x, outJit, len, op, reverse ? 1 : 0, inclusive ? 1 : 0);
                    gs.addWorkerGrid("sc.j", TestJitReduce.groups(outer, JitScan.THREADS));
                } else {
                    g.task("j", JitScan::scanColumns, new KernelContext(), x, outJit, outer * inner, len, inner, op, reverse ? 1 : 0, inclusive ? 1 : 0);
                    gs.addWorkerGrid("sc.j", TestJitElementwise.grid1D(outer * inner));
                }
                g.transferToHost(DataTransferMode.EVERY_EXECUTION, outMlx, outJit);
                try (TornadoExecutionPlan plan = new TornadoExecutionPlan(g.snapshot())) {
                    plan.withGridScheduler(gs).execute();
                }
                double[] expected = reference(xv, outer, len, inner, reverse, inclusive, identity, ref);
                String what = name + " " + Arrays.toString(shape) + " reverse=" + reverse + " inclusive=" + inclusive;
                assertAllClose(what + " JIT", expected, outJit, relTol, absTol);
                assertAllClose(what + " MLX", expected, outMlx, relTol, absTol);
            }
        }
    }

    @Test
    public void testCumsum() throws TornadoExecutionPlanException {
        check("cumsum", MlxReduce::cumsum, JitScan.SUM, 0, Double::sum, -1, 1, 1e-4, 2e-4);
    }

    @Test
    public void testCumprod() throws TornadoExecutionPlanException {
        check("cumprod", MlxReduce::cumprod, JitScan.PROD, 1, (a, b) -> a * b, 0.999f, 1.001f, 1e-4, 1e-6);
    }

    @Test
    public void testCummax() throws TornadoExecutionPlanException {
        check("cummax", MlxReduce::cummax, JitScan.MAX, Double.NEGATIVE_INFINITY, Math::max, -10, 10, 0, 0);
    }

    @Test
    public void testCummin() throws TornadoExecutionPlanException {
        check("cummin", MlxReduce::cummin, JitScan.MIN, Double.POSITIVE_INFINITY, Math::min, -10, 10, 0, 0);
    }

    @Test
    public void testLogcumsumexp() throws TornadoExecutionPlanException {
        check("logcumsumexp", MlxReduce::logcumsumexp, JitScan.LOGADDEXP, Double.NEGATIVE_INFINITY, TestJitScan::logaddexp, -3, 3, 1e-5, 1e-5);
    }

    @Test
    public void testIntCumsumAndHalfCummax() throws TornadoExecutionPlanException {
        final int outer = 4;
        final int len = 300;
        int[] iv = new int[outer * len];
        for (int i = 0; i < iv.length; i++) {
            iv[i] = (i * 7919) % 23 - 11;
        }
        IntArray xi = IntArray.fromArray(iv);
        IntArray sum = new IntArray(iv.length);
        float[] hv = values(outer * len, -5, 5, 91);
        HalfFloatArray xh = half(hv);
        HalfFloatArray max = new HalfFloatArray(hv.length);
        run(new TaskGraph("ih").transferToDevice(DataTransferMode.FIRST_EXECUTION, xi, xh) //
                .libraryTask("sum", MlxReduce::cumsum, xi, sum, outer, len, 1, false, true) //
                .libraryTask("max", MlxReduce::cummax, xh, max, outer, len, 1, true, true) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, sum, max));
        float[] h = widen(xh);
        for (int o = 0; o < outer; o++) {
            int acc = 0;
            for (int i = 0; i < len; i++) {
                acc += iv[o * len + i];
                assertEquals("cumsum int32 " + o + "," + i, acc, sum.get(o * len + i));
            }
            float m = Float.NEGATIVE_INFINITY;
            for (int i = len - 1; i >= 0; i--) {
                m = Math.max(m, h[o * len + i]);
                assertEquals("reverse cummax float16 " + o + "," + i, m, max.get(o * len + i).getFloat32(), 0f);
            }
        }
    }
}
