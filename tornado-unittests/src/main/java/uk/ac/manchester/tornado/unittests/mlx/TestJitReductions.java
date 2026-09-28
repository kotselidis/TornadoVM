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

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import java.util.Arrays;

import org.junit.Test;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.mlx.Mlx;
import uk.ac.manchester.tornado.mlx.jit.JitReductions;

/**
 * The KernelContext JIT counterparts of the MLX softmax, argmax and top-k operations, run in the
 * same graph as the MLX task and checked against a Java reference and against MLX.
 *
 * <p>
 * How to run?
 * </p>
 * <code>
 * tornado-test -V uk.ac.manchester.tornado.unittests.mlx.TestJitReductions
 * </code>
 */
public class TestJitReductions extends MlxTestBase {

    private static GridScheduler perRow(String task, int rows, int threads) {
        WorkerGrid1D grid = new WorkerGrid1D(rows * threads);
        grid.setLocalWork(threads, 1, 1);
        return new GridScheduler(task, grid);
    }

    private static double[] softmaxRef(float[] x, int group) {
        double[] out = new double[x.length];
        for (int start = 0; start < x.length; start += group) {
            double max = Double.NEGATIVE_INFINITY;
            for (int i = start; i < start + group; i++) {
                max = Math.max(max, x[i]);
            }
            double sum = 0;
            for (int i = start; i < start + group; i++) {
                out[i] = Math.exp(x[i] - max);
                sum += out[i];
            }
            for (int i = start; i < start + group; i++) {
                out[i] /= sum;
            }
        }
        return out;
    }

    private static void softmaxCase(String name, int rows, int cols, boolean whole, int[] axes3) throws TornadoExecutionPlanException {
        float[] xv = values(rows * cols, -8, 8, rows + cols);
        FloatArray x = FloatArray.fromArray(xv);
        FloatArray outMlx = new FloatArray(xv.length);
        FloatArray outJit = new FloatArray(xv.length);
        TaskGraph graph = new TaskGraph("sm").transferToDevice(DataTransferMode.FIRST_EXECUTION, x);
        if (whole) {
            graph.libraryTask("mlx", Mlx::softmax, x, outMlx);
        } else if (axes3 != null) {
            graph.libraryTask("mlx", Mlx::softmaxLastTwoAxes, x, outMlx, axes3[0], axes3[1], axes3[2]);
        } else {
            graph.libraryTask("mlx", Mlx::softmaxRows, x, outMlx, rows, cols);
        }
        graph.task("jit", JitReductions::softmaxRows, new KernelContext(), x, outJit, cols).transferToHost(DataTransferMode.EVERY_EXECUTION, outMlx, outJit);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(perRow("sm.jit", rows, JitReductions.ROW_THREADS)).execute();
        }
        double[] expected = softmaxRef(xv, cols);
        assertAllClose(name + " JIT", expected, outJit, 1e-5, 1e-8);
        for (int i = 0; i < expected.length; i++) {
            assertClose(name + " JIT vs MLX", i, outMlx.get(i), outJit.get(i), 1e-5, 1e-8);
        }
    }

    @Test
    public void testSoftmax() throws TornadoExecutionPlanException {
        softmaxCase("softmax whole", 1, 151936, true, null);
        softmaxCase("softmaxRows", 5, 333, false, null);
        softmaxCase("softmaxLastTwoAxes", 3, 4 * 50, false, new int[] { 3, 4, 50 });
    }

    private static int argmaxRef(float[] x, int from, int to) {
        int best = from;
        for (int i = from + 1; i < to; i++) {
            if (x[i] > x[best]) {
                best = i;
            }
        }
        return best - from;
    }

    @Test
    public void testArgmax() throws TornadoExecutionPlanException {
        for (int[] s : new int[][] { { 1, 151936 }, { 6, 1001 } }) {
            int rows = s[0], cols = s[1];
            float[] xv = values(rows * cols, -10, 10, cols);
            FloatArray x = FloatArray.fromArray(xv);
            IntArray outMlx = new IntArray(rows);
            IntArray outJit = new IntArray(rows);
            TaskGraph graph = new TaskGraph("am").transferToDevice(DataTransferMode.FIRST_EXECUTION, x);
            if (rows == 1) {
                graph.libraryTask("mlx", Mlx::argmax, x, outMlx);
            } else {
                graph.libraryTask("mlx", Mlx::argmaxRows, x, outMlx, rows, cols);
            }
            graph.task("jit", JitReductions::argmaxRows, new KernelContext(), x, outJit, cols).transferToHost(DataTransferMode.EVERY_EXECUTION, outMlx, outJit);
            try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
                plan.withGridScheduler(perRow("am.jit", rows, JitReductions.ROW_THREADS)).execute();
            }
            for (int r = 0; r < rows; r++) {
                int expected = argmaxRef(xv, r * cols, (r + 1) * cols);
                assertEquals("argmax JIT row " + r, expected, outJit.get(r));
                assertEquals("argmax MLX row " + r, expected, outMlx.get(r));
            }
        }
    }

    private static float[] topkRef(float[] x, int from, int to, int k) {
        float[] sorted = Arrays.copyOfRange(x, from, to);
        Arrays.sort(sorted);
        return Arrays.copyOfRange(sorted, sorted.length - k, sorted.length);
    }

    @Test
    public void testTopk() throws TornadoExecutionPlanException {
        for (int[] s : new int[][] { { 1, 151936, 50 }, { 4, 777, 5 }, { 3, 10000, 64 } }) {
            int rows = s[0], cols = s[1], k = s[2];
            float[] xv = values(rows * cols, -10, 10, 100L + k);
            FloatArray x = FloatArray.fromArray(xv);
            FloatArray outMlx = new FloatArray(rows * k);
            FloatArray outJit = new FloatArray(rows * k);
            TaskGraph graph = new TaskGraph("tk").transferToDevice(DataTransferMode.FIRST_EXECUTION, x);
            if (rows == 1) {
                graph.libraryTask("mlx", Mlx::topk, x, outMlx, k);
            } else {
                graph.libraryTask("mlx", Mlx::topkRows, x, outMlx, rows, cols, k);
            }
            graph.task("jit", JitReductions::topkRows, new KernelContext(), x, outJit, cols, k).transferToHost(DataTransferMode.EVERY_EXECUTION, outMlx, outJit);
            try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
                plan.withGridScheduler(perRow("tk.jit", rows, JitReductions.TOPK_THREADS)).execute();
            }
            for (int r = 0; r < rows; r++) {
                float[] expected = topkRef(xv, r * cols, (r + 1) * cols, k);
                float[] jit = Arrays.copyOfRange(outJit.toHeapArray(), r * k, (r + 1) * k);
                float[] mlx = Arrays.copyOfRange(outMlx.toHeapArray(), r * k, (r + 1) * k);
                Arrays.sort(jit);
                Arrays.sort(mlx);
                assertArrayEquals("topk JIT row " + r + " (k=" + k + ")", expected, jit, 0.0f);
                assertArrayEquals("topk MLX row " + r + " (k=" + k + ")", expected, mlx, 0.0f);
            }
        }
    }
}
