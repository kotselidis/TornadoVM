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

import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

import org.junit.Test;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.common.TornadoFunctions.LibraryTask2;
import uk.ac.manchester.tornado.api.common.TornadoFunctions.LibraryTask3;
import uk.ac.manchester.tornado.api.common.TornadoFunctions.Task4;
import uk.ac.manchester.tornado.api.common.TornadoFunctions.Task5;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.mlx.Mlx;
import uk.ac.manchester.tornado.mlx.jit.JitElementwise;

/**
 * The KernelContext JIT counterparts of the MLX element-wise operations: each runs in the same
 * graph as the MLX library task on the same inputs, and both are checked against a Java reference
 * and against each other.
 *
 * <p>
 * How to run?
 * </p>
 * <code>
 * tornado-test -V uk.ac.manchester.tornado.unittests.mlx.TestJitElementwise
 * </code>
 */
public class TestJitElementwise extends MlxTestBase {

    private static final int LOCAL = 256;

    static WorkerGrid1D grid1D(int n) {
        WorkerGrid1D grid = new WorkerGrid1D((n + LOCAL - 1) / LOCAL * LOCAL);
        grid.setLocalWork(LOCAL, 1, 1);
        return grid;
    }

    private static void binary(String name, LibraryTask3<FloatArray, FloatArray, FloatArray> mlx, Task5<KernelContext, FloatArray, FloatArray, FloatArray, Integer> jit,
            DoubleBinaryOperator ref, float lo, float hi) throws TornadoExecutionPlanException {
        for (int n : new int[] { 1, 1027, 1 << 20 }) {
            float[] av = values(n, lo, hi, 1);
            float[] bv = values(n, lo, hi, 2);
            FloatArray a = FloatArray.fromArray(av);
            FloatArray b = FloatArray.fromArray(bv);
            FloatArray cMlx = new FloatArray(n);
            FloatArray cJit = new FloatArray(n);
            TaskGraph graph = new TaskGraph("ew") //
                    .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b) //
                    .libraryTask("mlx", mlx, a, b, cMlx) //
                    .task("jit", jit, new KernelContext(), a, b, cJit, n) //
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, cMlx, cJit);
            try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
                plan.withGridScheduler(new GridScheduler("ew.jit", grid1D(n))).execute();
            }
            for (int i = 0; i < n; i++) {
                double expected = (float) ref.applyAsDouble(av[i], bv[i]);
                assertClose(name + " JIT n=" + n, i, expected, cJit.get(i), 1e-6, 1e-7);
                assertClose(name + " JIT vs MLX n=" + n, i, cMlx.get(i), cJit.get(i), 1e-6, 1e-7);
            }
        }
    }

    private static void unary(String name, LibraryTask2<FloatArray, FloatArray> mlx, Task4<KernelContext, FloatArray, FloatArray, Integer> jit, DoubleUnaryOperator ref, float lo,
            float hi) throws TornadoExecutionPlanException {
        for (int n : new int[] { 1, 1027, 1 << 20 }) {
            float[] av = values(n, lo, hi, 3);
            FloatArray a = FloatArray.fromArray(av);
            FloatArray outMlx = new FloatArray(n);
            FloatArray outJit = new FloatArray(n);
            TaskGraph graph = new TaskGraph("ew") //
                    .transferToDevice(DataTransferMode.FIRST_EXECUTION, a) //
                    .libraryTask("mlx", mlx, a, outMlx) //
                    .task("jit", jit, new KernelContext(), a, outJit, n) //
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, outMlx, outJit);
            try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
                plan.withGridScheduler(new GridScheduler("ew.jit", grid1D(n))).execute();
            }
            for (int i = 0; i < n; i++) {
                assertClose(name + " JIT n=" + n, i, ref.applyAsDouble(av[i]), outJit.get(i), 2e-6, 1e-6);
                assertClose(name + " JIT vs MLX n=" + n, i, outMlx.get(i), outJit.get(i), 4e-6, 2e-6);
            }
        }
    }

    /** erf reference with |error| &lt; 1.2e-7. */
    private static double erf(double x) {
        double t = 1.0 / (1.0 + 0.5 * Math.abs(x));
        double y = t * Math.exp(-x * x - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223
                + t * 0.17087277)))))))));
        return x >= 0 ? 1.0 - y : y - 1.0;
    }

    @Test
    public void testAdd() throws TornadoExecutionPlanException {
        binary("add", Mlx::add, JitElementwise::add, (x, y) -> x + y, -10, 10);
    }

    @Test
    public void testSubtract() throws TornadoExecutionPlanException {
        binary("subtract", Mlx::subtract, JitElementwise::subtract, (x, y) -> x - y, -10, 10);
    }

    @Test
    public void testMultiply() throws TornadoExecutionPlanException {
        binary("multiply", Mlx::multiply, JitElementwise::multiply, (x, y) -> x * y, -10, 10);
    }

    @Test
    public void testDivide() throws TornadoExecutionPlanException {
        binary("divide", Mlx::divide, JitElementwise::divide, (x, y) -> x / y, 0.5f, 10);
    }

    @Test
    public void testMaximum() throws TornadoExecutionPlanException {
        binary("maximum", Mlx::maximum, JitElementwise::maximum, Math::max, -10, 10);
    }

    @Test
    public void testMinimum() throws TornadoExecutionPlanException {
        binary("minimum", Mlx::minimum, JitElementwise::minimum, Math::min, -10, 10);
    }

    @Test
    public void testNegative() throws TornadoExecutionPlanException {
        unary("negative", Mlx::negative, JitElementwise::negative, x -> -x, -10, 10);
    }

    @Test
    public void testExp() throws TornadoExecutionPlanException {
        unary("exp", Mlx::exp, JitElementwise::exp, Math::exp, -8, 8);
    }

    @Test
    public void testTanh() throws TornadoExecutionPlanException {
        unary("tanh", Mlx::tanh, JitElementwise::tanh, Math::tanh, -5, 5);
    }

    @Test
    public void testErf() throws TornadoExecutionPlanException {
        unary("erf", Mlx::erf, JitElementwise::erf, TestJitElementwise::erf, -3, 3);
    }

    @Test
    public void testSigmoid() throws TornadoExecutionPlanException {
        unary("sigmoid", Mlx::sigmoid, JitElementwise::sigmoid, x -> 1.0 / (1.0 + Math.exp(-x)), -10, 10);
    }

    @Test
    public void testSqrt() throws TornadoExecutionPlanException {
        unary("sqrt", Mlx::sqrt, JitElementwise::sqrt, Math::sqrt, 0, 100);
    }

    @Test
    public void testRsqrt() throws TornadoExecutionPlanException {
        unary("rsqrt", Mlx::rsqrt, JitElementwise::rsqrt, x -> 1.0 / Math.sqrt(x), 0.01f, 100);
    }

    @Test
    public void testSquare() throws TornadoExecutionPlanException {
        unary("square", Mlx::square, JitElementwise::square, x -> x * x, -10, 10);
    }
}
