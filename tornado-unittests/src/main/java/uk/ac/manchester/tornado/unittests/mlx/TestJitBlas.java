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

import org.junit.Test;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.mlx.Mlx;
import uk.ac.manchester.tornado.mlx.jit.JitBlas;

/**
 * The KernelContext JIT counterparts of the MLX matrix operations, run in the same graph as the MLX
 * task and checked against a Java reference and against MLX.
 *
 * <p>
 * How to run?
 * </p>
 * <code>
 * tornado-test -V uk.ac.manchester.tornado.unittests.mlx.TestJitBlas
 * </code>
 */
public class TestJitBlas extends MlxTestBase {

    private static double[] reference(float[] a, float[] b, int m, int k, int n, boolean bTransposed) {
        double[] c = new double[m * n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0;
                for (int p = 0; p < k; p++) {
                    sum += (double) a[i * k + p] * (bTransposed ? b[j * k + p] : b[p * n + j]);
                }
                c[i * n + j] = sum;
            }
        }
        return c;
    }

    private static WorkerGrid1D gemmGrid(int m, int n) {
        WorkerGrid1D grid = new WorkerGrid1D((m / JitBlas.GEMM_BLOCK) * (n / JitBlas.GEMM_BLOCK) * JitBlas.GEMM_THREADS);
        grid.setLocalWork(JitBlas.GEMM_THREADS, 1, 1);
        return grid;
    }

    private static WorkerGrid1D gemvGrid(int rows) {
        WorkerGrid1D grid = new WorkerGrid1D(rows * 32);
        grid.setLocalWork(32, 1, 1);
        return grid;
    }

    private static void check(String name, double[] expected, FloatArray jit, FloatArray mlx, double rel, double abs) {
        assertAllClose(name + " JIT", expected, jit, rel, abs);
        for (int i = 0; i < expected.length; i++) {
            assertClose(name + " JIT vs MLX", i, mlx.get(i), jit.get(i), rel, abs);
        }
    }

    @Test
    public void testGemm() throws TornadoExecutionPlanException {
        for (int[] s : new int[][] { { 32, 8, 32 }, { 64, 96, 128 }, { 256, 512, 128 } }) {
            int m = s[0], k = s[1], n = s[2];
            float[] av = values(m * k, -1, 1, 1);
            float[] bv = values(k * n, -1, 1, 2);
            FloatArray a = FloatArray.fromArray(av);
            FloatArray b = FloatArray.fromArray(bv);
            FloatArray cMlx = new FloatArray(m * n);
            FloatArray cJit = new FloatArray(m * n);
            TaskGraph graph = new TaskGraph("mm").transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b) //
                    .libraryTask("mlx", Mlx::matmul, a, b, cMlx, m, k, n) //
                    .task("jit", JitBlas::gemm, new KernelContext(), a, b, cJit, m, n, k) //
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, cMlx, cJit);
            try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
                plan.withGridScheduler(new GridScheduler("mm.jit", gemmGrid(m, n))).execute();
            }
            check("gemm " + m + "x" + k + "x" + n, reference(av, bv, m, k, n, false), cJit, cMlx, 1e-4, 1e-4);
        }
    }

    @Test
    public void testGemmTransposed() throws TornadoExecutionPlanException {
        final int m = 64, k = 256, n = 96;
        float[] av = values(m * k, -1, 1, 3);
        float[] wv = values(n * k, -1, 1, 4);
        FloatArray a = FloatArray.fromArray(av);
        FloatArray w = FloatArray.fromArray(wv);
        FloatArray cMlx = new FloatArray(m * n);
        FloatArray cJit = new FloatArray(m * n);
        TaskGraph graph = new TaskGraph("mmt").transferToDevice(DataTransferMode.FIRST_EXECUTION, a, w) //
                .libraryTask("mlx", Mlx::matmulTransposed, a, w, cMlx, m, k, n) //
                .task("jit", JitBlas::gemmTransposed, new KernelContext(), a, w, cJit, m, n, k) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, cMlx, cJit);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(new GridScheduler("mmt.jit", gemmGrid(m, n))).execute();
        }
        check("gemmTransposed", reference(av, wv, m, k, n, true), cJit, cMlx, 1e-4, 1e-4);
    }

    @Test
    public void testGemmTransposedHalf() throws TornadoExecutionPlanException {
        final int m = 32, k = 512, n = 64;
        HalfFloatArray a = half(values(m * k, -1, 1, 5));
        HalfFloatArray w = half(values(n * k, -1, 1, 6));
        HalfFloatArray cMlx = new HalfFloatArray(m * n);
        FloatArray cJit = new FloatArray(m * n);
        TaskGraph graph = new TaskGraph("mmt16").transferToDevice(DataTransferMode.FIRST_EXECUTION, a, w) //
                .libraryTask("mlx", Mlx::matmulTransposed, a, w, cMlx, m, k, n) //
                .task("jit", JitBlas::gemmTransposedF16, new KernelContext(), a, w, cJit, m, n, k) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, cMlx, cJit);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(new GridScheduler("mmt16.jit", gemmGrid(m, n))).execute();
        }
        double[] expected = reference(widen(a), widen(w), m, k, n, true);
        assertAllClose("gemmTransposedF16 JIT", expected, cJit, 1e-4, 1e-3);
        assertAllClose("gemmTransposedF16 MLX", expected, cMlx, 2e-3, 3e-2);
    }

    @Test
    public void testAddmm() throws TornadoExecutionPlanException {
        final int m = 64, k = 72, n = 96;
        final float alpha = 0.5f, beta = 2.0f;
        float[] cv = values(m * n, -1, 1, 7);
        float[] av = values(m * k, -1, 1, 8);
        float[] bv = values(k * n, -1, 1, 9);
        FloatArray cIn = FloatArray.fromArray(cv);
        FloatArray a = FloatArray.fromArray(av);
        FloatArray b = FloatArray.fromArray(bv);
        FloatArray outMlx = new FloatArray(m * n);
        FloatArray outJit = new FloatArray(m * n);
        TaskGraph graph = new TaskGraph("addmm").transferToDevice(DataTransferMode.FIRST_EXECUTION, cIn, a, b) //
                .libraryTask("mlx", Mlx::addmm, cIn, a, b, outMlx, m, k, n, alpha, beta) //
                .task("jit", JitBlas::addmm, new KernelContext(), cIn, a, b, outJit, m, n, k, alpha, beta) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, outMlx, outJit);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(new GridScheduler("addmm.jit", gemmGrid(m, n))).execute();
        }
        double[] ab = reference(av, bv, m, k, n, false);
        double[] expected = new double[m * n];
        for (int i = 0; i < expected.length; i++) {
            expected[i] = alpha * ab[i] + beta * cv[i];
        }
        check("addmm", expected, outJit, outMlx, 1e-4, 1e-4);
    }

    @Test
    public void testGemv() throws TornadoExecutionPlanException {
        final int rows = 1000, cols = 2048;
        float[] wv = values(rows * cols, -0.05f, 0.05f, 10);
        float[] xv = values(cols, -1, 1, 11);
        FloatArray w = FloatArray.fromArray(wv);
        FloatArray x = FloatArray.fromArray(xv);
        FloatArray yMlx = new FloatArray(rows);
        FloatArray yJit = new FloatArray(rows);
        TaskGraph graph = new TaskGraph("gemv").transferToDevice(DataTransferMode.FIRST_EXECUTION, w, x) //
                .libraryTask("mlx", Mlx::matmulTransposed, x, w, yMlx, 1, cols, rows) //
                .task("jit", JitBlas::gemv, new KernelContext(), x, w, yJit, cols) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, yMlx, yJit);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(new GridScheduler("gemv.jit", gemvGrid(rows))).execute();
        }
        check("gemv", reference(xv, wv, 1, cols, rows, true), yJit, yMlx, 1e-4, 1e-5);
    }

    @Test
    public void testGemvHalf() throws TornadoExecutionPlanException {
        final int rows = 512, cols = 4096;
        HalfFloatArray w = half(values(rows * cols, -0.05f, 0.05f, 12));
        HalfFloatArray x = half(values(cols, -1, 1, 13));
        HalfFloatArray yMlx = new HalfFloatArray(rows);
        FloatArray yJit = new FloatArray(rows);
        TaskGraph graph = new TaskGraph("gemv16").transferToDevice(DataTransferMode.FIRST_EXECUTION, w, x) //
                .libraryTask("mlx", Mlx::matmulTransposed, x, w, yMlx, 1, cols, rows) //
                .task("jit", JitBlas::gemvF16, new KernelContext(), x, w, yJit, cols) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, yMlx, yJit);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(new GridScheduler("gemv16.jit", gemvGrid(rows))).execute();
        }
        double[] expected = reference(widen(x), widen(w), 1, cols, rows, true);
        assertAllClose("gemvF16 JIT", expected, yJit, 1e-4, 1e-4);
        assertAllClose("gemvF16 MLX", expected, yMlx, 2e-3, 2e-3);
    }

    @Test
    public void testTranspose() throws TornadoExecutionPlanException {
        final int rows = 100, cols = 70;
        float[] v = values(rows * cols, -1, 1, 14);
        FloatArray in = FloatArray.fromArray(v);
        FloatArray out = new FloatArray(rows * cols);
        // MLX's transpose is exercised through matmulTransposed; here: the JIT kernel against Java.
        TaskGraph graph = new TaskGraph("tr").transferToDevice(DataTransferMode.FIRST_EXECUTION, in) //
                .task("jit", JitBlas::transpose, new KernelContext(), in, out, rows, cols) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, out);
        WorkerGrid2D grid = new WorkerGrid2D((cols + 31) / 32 * 32, (rows + 31) / 32 * 32);
        grid.setLocalWork(32, 32, 1);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(new GridScheduler("tr.jit", grid)).execute();
        }
        double[] expected = new double[rows * cols];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                expected[c * rows + r] = v[r * cols + c];
            }
        }
        assertAllClose("transpose JIT", expected, out, 0, 0);
    }
}
