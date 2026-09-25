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

import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

import org.junit.Test;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.common.TornadoFunctions.LibraryTask2;
import uk.ac.manchester.tornado.api.common.TornadoFunctions.LibraryTask3;
import uk.ac.manchester.tornado.api.common.TornadoFunctions.Task4;
import uk.ac.manchester.tornado.api.common.TornadoFunctions.Task5;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.mlx.MlxMath;
import uk.ac.manchester.tornado.mlx.jit.JitMath;

/**
 * The KernelContext JIT counterparts of the MLX element-wise math operations: each runs in the
 * same graph as the MLX library task on the same inputs, and both are checked against a Java
 * reference and against each other.
 *
 * <p>
 * How to run?
 * </p>
 * <code>
 * tornado-test -V uk.ac.manchester.tornado.unittests.mlx.TestJitMath
 * </code>
 */
public class TestJitMath extends MlxTestBase {

    private static final int[] SIZES = { 1, 1027, 1 << 20 };

    private static void unary(String name, LibraryTask2<FloatArray, FloatArray> mlx, Task4<KernelContext, FloatArray, FloatArray, Integer> jit, DoubleUnaryOperator ref, float lo,
            float hi, double relTol, double absTol) throws TornadoExecutionPlanException {
        for (int n : SIZES) {
            float[] av = values(n, lo, hi, 3);
            FloatArray a = FloatArray.fromArray(av);
            FloatArray outMlx = new FloatArray(n);
            FloatArray outJit = new FloatArray(n);
            TaskGraph graph = new TaskGraph("m1") //
                    .transferToDevice(DataTransferMode.FIRST_EXECUTION, a) //
                    .libraryTask("mlx", mlx, a, outMlx) //
                    .task("jit", jit, new KernelContext(), a, outJit, n) //
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, outMlx, outJit);
            try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
                plan.withGridScheduler(new GridScheduler("m1.jit", TestJitElementwise.grid1D(n))).execute();
            }
            for (int i = 0; i < n; i++) {
                double expected = ref.applyAsDouble(av[i]);
                assertClose(name + " JIT n=" + n + " x=" + av[i], i, expected, outJit.get(i), relTol, absTol);
                assertClose(name + " MLX n=" + n + " x=" + av[i], i, expected, outMlx.get(i), relTol, absTol);
            }
        }
    }

    private static void binary(String name, LibraryTask3<FloatArray, FloatArray, FloatArray> mlx, Task5<KernelContext, FloatArray, FloatArray, FloatArray, Integer> jit,
            DoubleBinaryOperator ref, float aLo, float aHi, float bLo, float bHi, double relTol, double absTol) throws TornadoExecutionPlanException {
        for (int n : SIZES) {
            float[] av = values(n, aLo, aHi, 1);
            float[] bv = values(n, bLo, bHi, 2);
            FloatArray a = FloatArray.fromArray(av);
            FloatArray b = FloatArray.fromArray(bv);
            FloatArray cMlx = new FloatArray(n);
            FloatArray cJit = new FloatArray(n);
            TaskGraph graph = new TaskGraph("m2") //
                    .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b) //
                    .libraryTask("mlx", mlx, a, b, cMlx) //
                    .task("jit", jit, new KernelContext(), a, b, cJit, n) //
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, cMlx, cJit);
            try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
                plan.withGridScheduler(new GridScheduler("m2.jit", TestJitElementwise.grid1D(n))).execute();
            }
            for (int i = 0; i < n; i++) {
                // The reference sees the float32 quotient, as both kernels do.
                double expected = ref.applyAsDouble(av[i], bv[i]);
                assertClose(name + " JIT n=" + n, i, expected, cJit.get(i), relTol, absTol);
                assertClose(name + " MLX n=" + n, i, expected, cMlx.get(i), relTol, absTol);
            }
        }
    }

    /** erf^-1 in double: Newton's method on {@link #erf}, from a crude start. */
    private static double erfinv(double y) {
        double x = 0;
        for (int k = 0; k < 60; k++) {
            x -= (erf(x) - y) / (2 / Math.sqrt(Math.PI) * Math.exp(-x * x));
        }
        return x;
    }

    /** erf with |error| &lt; 1.2e-7 (Numerical Recipes erfc Chebyshev fit). */
    private static double erf(double x) {
        double t = 1.0 / (1.0 + 0.5 * Math.abs(x));
        double y = t * Math.exp(-x * x - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223
                + t * 0.17087277)))))))));
        return x >= 0 ? 1.0 - y : y - 1.0;
    }

    @Test
    public void testRound() throws TornadoExecutionPlanException {
        for (int decimals : new int[] { 0, 2 }) {
            final int n = 1027;
            float[] av = values(n, -100, 100, 7);
            float[] ties = { 0.5f, 1.5f, 2.5f, -0.5f, -1.5f, -2.5f, 3.5f, 4.5f };
            if (decimals == 0) {
                System.arraycopy(ties, 0, av, 0, ties.length);
            }
            FloatArray a = FloatArray.fromArray(av);
            FloatArray outMlx = new FloatArray(n);
            FloatArray outJit = new FloatArray(n);
            TaskGraph graph = new TaskGraph("rd") //
                    .transferToDevice(DataTransferMode.FIRST_EXECUTION, a) //
                    .libraryTask("mlx", MlxMath::round, a, outMlx, decimals) //
                    .task("jit", JitMath::round, new KernelContext(), a, outJit, n, decimals) //
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, outMlx, outJit);
            try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
                plan.withGridScheduler(new GridScheduler("rd.jit", TestJitElementwise.grid1D(n))).execute();
            }
            float scale = (float) Math.pow(10, decimals);
            for (int i = 0; i < n; i++) {
                double expected = decimals == 0 ? Math.rint(av[i]) : (float) Math.rint(av[i] * scale) / scale;
                assertClose("round(" + decimals + ") JIT x=" + av[i], i, expected, outJit.get(i), 1e-6, 1e-6);
                assertClose("round(" + decimals + ") MLX x=" + av[i], i, expected, outMlx.get(i), 1e-6, 1e-6);
            }
        }
    }

    @Test
    public void testDivmod() throws TornadoExecutionPlanException {
        final int n = 1027;
        float[] av = values(n, -10, 10, 8);
        float[] bv = values(n, 0.5f, 10, 9);
        FloatArray a = FloatArray.fromArray(av);
        FloatArray b = FloatArray.fromArray(bv);
        FloatArray qMlx = new FloatArray(n);
        FloatArray rMlx = new FloatArray(n);
        FloatArray qJit = new FloatArray(n);
        FloatArray rJit = new FloatArray(n);
        TaskGraph graph = new TaskGraph("dm") //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b) //
                .libraryTask("mlx", MlxMath::divmod, a, b, qMlx, rMlx) //
                .task("jit", JitMath::divmod, new KernelContext(), a, b, qJit, rJit, n) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, qMlx, rMlx, qJit, rJit);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(new GridScheduler("dm.jit", TestJitElementwise.grid1D(n))).execute();
        }
        for (int i = 0; i < n; i++) {
            double q = (int) (av[i] / bv[i]);
            double r = av[i] - bv[i] * Math.floor(av[i] / bv[i]);
            assertClose("divmod quotient JIT", i, q, qJit.get(i), 0, 0);
            assertClose("divmod quotient MLX", i, q, qMlx.get(i), 0, 0);
            assertClose("divmod remainder JIT", i, r, rJit.get(i), 1e-5, 1e-5);
            assertClose("divmod remainder MLX", i, r, rMlx.get(i), 1e-5, 1e-5);
        }
    }

    @Test
    public void testClip() throws TornadoExecutionPlanException {
        final int n = 1027;
        float[] av = values(n, -10, 10, 10);
        FloatArray a = FloatArray.fromArray(av);
        FloatArray outMlx = new FloatArray(n);
        FloatArray outJit = new FloatArray(n);
        TaskGraph graph = new TaskGraph("cl") //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, a) //
                .libraryTask("mlx", MlxMath::clip, a, outMlx, -2.5f, 4f) //
                .task("jit", JitMath::clip, new KernelContext(), a, outJit, n, -2.5f, 4f) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, outMlx, outJit);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(new GridScheduler("cl.jit", TestJitElementwise.grid1D(n))).execute();
        }
        for (int i = 0; i < n; i++) {
            float expected = Math.min(Math.max(av[i], -2.5f), 4f);
            assertEquals("clip JIT " + i, expected, outJit.get(i), 0f);
            assertEquals("clip MLX " + i, expected, outMlx.get(i), 0f);
        }
    }

    @Test
    public void testWhere() throws TornadoExecutionPlanException {
        final int n = 1027;
        float[] xv = values(n, -10, 10, 11);
        float[] yv = values(n, -10, 10, 12);
        ByteArray cond = new ByteArray(n);
        for (int i = 0; i < n; i++) {
            cond.set(i, (byte) (i % 3 == 0 ? 1 : 0));
        }
        FloatArray x = FloatArray.fromArray(xv);
        FloatArray y = FloatArray.fromArray(yv);
        FloatArray outMlx = new FloatArray(n);
        FloatArray outJit = new FloatArray(n);
        TaskGraph graph = new TaskGraph("wh") //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, cond, x, y) //
                .libraryTask("mlx", MlxMath::where, cond, x, y, outMlx) //
                .task("jit", JitMath::where, new KernelContext(), cond, x, y, outJit, n) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, outMlx, outJit);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(new GridScheduler("wh.jit", TestJitElementwise.grid1D(n))).execute();
        }
        for (int i = 0; i < n; i++) {
            float expected = i % 3 == 0 ? xv[i] : yv[i];
            assertEquals("where JIT " + i, expected, outJit.get(i), 0f);
            assertEquals("where MLX " + i, expected, outMlx.get(i), 0f);
        }
    }

    @Test
    public void testAbs() throws TornadoExecutionPlanException {
        unary("abs", MlxMath::abs, JitMath::abs, x -> Math.abs(x), -10.0f, 10.0f, 1e-06, 1e-07);
    }

    @Test
    public void testArccos() throws TornadoExecutionPlanException {
        unary("arccos", MlxMath::arccos, JitMath::arccos, x -> Math.acos(x), -1.0f, 1.0f, 2e-06, 1e-06);
    }

    @Test
    public void testArccosh() throws TornadoExecutionPlanException {
        unary("arccosh", MlxMath::arccosh, JitMath::arccosh, x -> Math.log(x + Math.sqrt(x * x - 1)), 1.0f, 10.0f, 2e-06, 1e-06);
    }

    @Test
    public void testArcsin() throws TornadoExecutionPlanException {
        unary("arcsin", MlxMath::arcsin, JitMath::arcsin, x -> Math.asin(x), -1.0f, 1.0f, 2e-06, 1e-06);
    }

    @Test
    public void testArcsinh() throws TornadoExecutionPlanException {
        unary("arcsinh", MlxMath::arcsinh, JitMath::arcsinh, x -> Math.log(x + Math.sqrt(x * x + 1)), -10.0f, 10.0f, 1e-05, 1e-06);
    }

    @Test
    public void testArctan() throws TornadoExecutionPlanException {
        unary("arctan", MlxMath::arctan, JitMath::arctan, x -> Math.atan(x), -10.0f, 10.0f, 2e-06, 1e-06);
    }

    @Test
    public void testArctanh() throws TornadoExecutionPlanException {
        unary("arctanh", MlxMath::arctanh, JitMath::arctanh, x -> 0.5 * Math.log((1 + x) / (1 - x)), -0.99f, 0.99f, 1e-05, 1e-06);
    }

    @Test
    public void testCeil() throws TornadoExecutionPlanException {
        unary("ceil", MlxMath::ceil, JitMath::ceil, x -> Math.ceil(x), -10.0f, 10.0f, 0, 0);
    }

    @Test
    public void testCos() throws TornadoExecutionPlanException {
        unary("cos", MlxMath::cos, JitMath::cos, x -> Math.cos(x), -10.0f, 10.0f, 2e-06, 1e-06);
    }

    @Test
    public void testCosh() throws TornadoExecutionPlanException {
        unary("cosh", MlxMath::cosh, JitMath::cosh, x -> Math.cosh(x), -5.0f, 5.0f, 2e-06, 1e-06);
    }

    @Test
    public void testDegrees() throws TornadoExecutionPlanException {
        unary("degrees", MlxMath::degrees, JitMath::degrees, x -> Math.toDegrees(x), -10.0f, 10.0f, 1e-06, 1e-06);
    }

    @Test
    public void testErfinv() throws TornadoExecutionPlanException {
        unary("erfinv", MlxMath::erfinv, JitMath::erfinv, x -> erfinv(x), -0.99f, 0.99f, 1e-05, 1e-06);
    }

    @Test
    public void testExpm1() throws TornadoExecutionPlanException {
        unary("expm1", MlxMath::expm1, JitMath::expm1, x -> Math.expm1(x), -5.0f, 5.0f, 3e-05, 1e-06);
    }

    @Test
    public void testFloor() throws TornadoExecutionPlanException {
        unary("floor", MlxMath::floor, JitMath::floor, x -> Math.floor(x), -10.0f, 10.0f, 0, 0);
    }

    @Test
    public void testLog() throws TornadoExecutionPlanException {
        unary("log", MlxMath::log, JitMath::log, x -> Math.log(x), 0.01f, 10.0f, 2e-06, 1e-06);
    }

    @Test
    public void testLog10() throws TornadoExecutionPlanException {
        unary("log10", MlxMath::log10, JitMath::log10, x -> Math.log10(x), 0.01f, 10.0f, 2e-06, 1e-06);
    }

    @Test
    public void testLog1p() throws TornadoExecutionPlanException {
        unary("log1p", MlxMath::log1p, JitMath::log1p, x -> Math.log1p(x), -0.9f, 10.0f, 1e-05, 1e-06);
    }

    @Test
    public void testLog2() throws TornadoExecutionPlanException {
        unary("log2", MlxMath::log2, JitMath::log2, x -> Math.log(x) / Math.log(2), 0.01f, 10.0f, 2e-06, 1e-06);
    }

    @Test
    public void testRadians() throws TornadoExecutionPlanException {
        unary("radians", MlxMath::radians, JitMath::radians, x -> Math.toRadians(x), -10.0f, 10.0f, 1e-06, 1e-07);
    }

    @Test
    public void testReciprocal() throws TornadoExecutionPlanException {
        unary("reciprocal", MlxMath::reciprocal, JitMath::reciprocal, x -> 1.0 / x, 0.5f, 10.0f, 1e-06, 1e-07);
    }

    @Test
    public void testSign() throws TornadoExecutionPlanException {
        unary("sign", MlxMath::sign, JitMath::sign, x -> Math.signum(x), -10.0f, 10.0f, 0, 0);
    }

    @Test
    public void testSin() throws TornadoExecutionPlanException {
        unary("sin", MlxMath::sin, JitMath::sin, x -> Math.sin(x), -10.0f, 10.0f, 2e-06, 1e-06);
    }

    @Test
    public void testSinh() throws TornadoExecutionPlanException {
        unary("sinh", MlxMath::sinh, JitMath::sinh, x -> Math.sinh(x), -5.0f, 5.0f, 1e-05, 1e-06);
    }

    @Test
    public void testTan() throws TornadoExecutionPlanException {
        unary("tan", MlxMath::tan, JitMath::tan, x -> Math.tan(x), -1.5f, 1.5f, 1e-05, 1e-06);
    }

    @Test
    public void testArctan2() throws TornadoExecutionPlanException {
        binary("arctan2", MlxMath::arctan2, JitMath::arctan2, (x, y) -> Math.atan2(x, y), -10.0f, 10.0f, -10.0f, 10.0f, 2e-06, 1e-06);
    }

    @Test
    public void testFloorDivide() throws TornadoExecutionPlanException {
        binary("floor_divide", MlxMath::floorDivide, JitMath::floorDivide, (x, y) -> Math.floor(x / y), -10.0f, 10.0f, 0.5f, 10.0f, 0, 0);
    }

    @Test
    public void testLogaddexp() throws TornadoExecutionPlanException {
        binary("logaddexp", MlxMath::logaddexp, JitMath::logaddexp, (x, y) -> Math.max(x, y) + Math.log1p(Math.exp(-Math.abs(x - y))), -10.0f, 10.0f, -10.0f, 10.0f, 2e-06, 1e-06);
    }

    @Test
    public void testPower() throws TornadoExecutionPlanException {
        binary("power", MlxMath::power, JitMath::power, (x, y) -> Math.pow(x, y), 0.1f, 5.0f, -3.0f, 3.0f, 1e-05, 1e-06);
    }

    @Test
    public void testRemainder() throws TornadoExecutionPlanException {
        binary("remainder", MlxMath::remainder, JitMath::remainder, (x, y) -> x - y * Math.floor(x / y), -10.0f, 10.0f, 0.5f, 10.0f, 1e-05, 1e-05);
    }
}
