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

import static org.junit.Assert.assertTrue;

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
import uk.ac.manchester.tornado.mlx.jit.JitQuantized;

/**
 * The KernelContext JIT counterparts of the MLX affine quantization operations, run in the same
 * graph as the MLX tasks and checked against a Java decoder of the packed format and against MLX.
 *
 * <p>
 * How to run?
 * </p>
 * <code>
 * tornado-test -V uk.ac.manchester.tornado.unittests.mlx.TestJitQuantized
 * </code>
 */
public class TestJitQuantized extends MlxTestBase {

    private static WorkerGrid1D grid1D(int n) {
        return TestJitElementwise.grid1D(n);
    }

    private static WorkerGrid1D perOutput(int outputs) {
        WorkerGrid1D grid = new WorkerGrid1D(outputs * JitQuantized.SIMD_THREADS);
        grid.setLocalWork(JitQuantized.SIMD_THREADS, 1, 1);
        return grid;
    }

    /** Decodes packed weights into w[rows, cols]. */
    private static double[] decode(IntArray wq, FloatArray scales, FloatArray biases, int rows, int cols, int groupSize, int bits) {
        int perWord = 32 / bits;
        int mask = (1 << bits) - 1;
        double[] w = new double[rows * cols];
        for (int i = 0; i < rows * cols; i++) {
            int q = (wq.get(i / perWord) >>> (bits * (i % perWord))) & mask;
            int g = i / groupSize;
            w[i] = scales.get(g) * q + biases.get(g);
        }
        return w;
    }

    /** y[m, n] = x[m, k] @ w[n, k]^T. */
    private static double[] matmulTransposed(float[] x, int xOff, double[] w, int wOff, int m, int k, int n) {
        double[] y = new double[m * n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0;
                for (int p = 0; p < k; p++) {
                    sum += x[xOff + i * k + p] * w[wOff + j * k + p];
                }
                y[i * n + j] = sum;
            }
        }
        return y;
    }

    private static void quantizeCase(int rows, int cols, int groupSize, int bits) throws TornadoExecutionPlanException {
        float[] wv = values(rows * cols, -1, 1, 17L * bits + groupSize);
        int groups = rows * cols / groupSize;
        int words = rows * cols * bits / 32;
        FloatArray w = FloatArray.fromArray(wv);
        IntArray wqMlx = new IntArray(words);
        FloatArray scalesMlx = new FloatArray(groups);
        FloatArray biasesMlx = new FloatArray(groups);
        IntArray wqJit = new IntArray(words);
        FloatArray scalesJit = new FloatArray(groups);
        FloatArray biasesJit = new FloatArray(groups);
        FloatArray backMlx = new FloatArray(rows * cols);
        FloatArray backJit = new FloatArray(rows * cols);
        TaskGraph graph = new TaskGraph("q").transferToDevice(DataTransferMode.FIRST_EXECUTION, w) //
                .libraryTask("mlxQ", Mlx::quantize, w, wqMlx, scalesMlx, biasesMlx, rows, cols, groupSize, bits) //
                .libraryTask("mlxDq", Mlx::dequantize, wqMlx, scalesMlx, biasesMlx, backMlx, rows, cols, groupSize, bits) //
                .task("jitQ", JitQuantized::quantize, new KernelContext(), w, wqJit, scalesJit, biasesJit, groups, groupSize, bits) //
                // The JIT dequantize decodes MLX's packing, so it is checked against MLX's dequantize.
                .task("jitDq", JitQuantized::dequantize, new KernelContext(), wqMlx, scalesMlx, biasesMlx, backJit, words, groupSize, bits) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, wqMlx, scalesMlx, biasesMlx, wqJit, scalesJit, biasesJit, backMlx, backJit);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            GridScheduler grids = new GridScheduler();
            grids.addWorkerGrid("q.jitQ", grid1D(groups));
            grids.addWorkerGrid("q.jitDq", grid1D(words));
            plan.withGridScheduler(grids).execute();
        }
        String what = "bits=" + bits + " group=" + groupSize;
        double[] decodedMlx = decode(wqMlx, scalesMlx, biasesMlx, rows, cols, groupSize, bits);
        assertAllClose("JIT dequantize " + what, decodedMlx, backJit, 1e-6, 1e-6);
        assertAllClose("MLX dequantize " + what, decodedMlx, backMlx, 1e-6, 1e-6);

        // The JIT quantize picks the same scales and biases as MLX...
        double[] scalesRef = new double[groups];
        double[] biasesRef = new double[groups];
        for (int g = 0; g < groups; g++) {
            scalesRef[g] = scalesMlx.get(g);
            biasesRef[g] = biasesMlx.get(g);
        }
        assertAllClose("JIT quantize scales " + what, scalesRef, scalesJit, 1e-5, 1e-7);
        assertAllClose("JIT quantize biases " + what, biasesRef, biasesJit, 1e-5, 1e-7);
        // ...and its codes decode to within one step of the input, like MLX's (rounding ties may differ).
        double[] decodedJit = decode(wqJit, scalesJit, biasesJit, rows, cols, groupSize, bits);
        int sameWords = 0;
        for (int i = 0; i < words; i++) {
            sameWords += wqJit.get(i) == wqMlx.get(i) ? 1 : 0;
        }
        for (int i = 0; i < wv.length; i++) {
            double step = Math.abs(scalesJit.get(i / groupSize));
            assertTrue("JIT quantize round trip " + what + " element " + i, Math.abs(wv[i] - decodedJit[i]) <= step + 1e-5);
        }
        assertTrue("JIT quantize codes match MLX in " + sameWords + "/" + words + " words (" + what + ")", sameWords >= words * 0.99);
    }

    @Test
    public void testQuantizeDequantize() throws TornadoExecutionPlanException {
        quantizeCase(64, 256, 32, 4);
        quantizeCase(16, 512, 64, 8);
        quantizeCase(8, 256, 128, 2);
    }

    @Test
    public void testQuantizedMatmul() throws TornadoExecutionPlanException {
        for (int[] s : new int[][] { { 1, 2048, 256, 64, 4 }, { 3, 256, 40, 32, 4 }, { 2, 512, 64, 128, 8 } }) {
            int m = s[0], k = s[1], n = s[2], groupSize = s[3], bits = s[4];
            float[] xv = values(m * k, -1, 1, 61L + k);
            FloatArray x = FloatArray.fromArray(xv);
            FloatArray w = FloatArray.fromArray(values(n * k, -1, 1, 62L + n));
            IntArray wq = new IntArray(n * k * bits / 32);
            FloatArray scales = new FloatArray(n * k / groupSize);
            FloatArray biases = new FloatArray(n * k / groupSize);
            FloatArray yMlx = new FloatArray(m * n);
            FloatArray yJit = new FloatArray(m * n);
            TaskGraph graph = new TaskGraph("qmm").transferToDevice(DataTransferMode.FIRST_EXECUTION, x, w) //
                    .libraryTask("quantize", Mlx::quantize, w, wq, scales, biases, n, k, groupSize, bits) //
                    .libraryTask("mlx", Mlx::quantizedMatmul, x, wq, scales, biases, yMlx, m, k, n, groupSize, bits) //
                    .task("jit", JitQuantized::quantizedMatmul, new KernelContext(), x, wq, scales, biases, yJit, k, n, groupSize, bits) //
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, wq, scales, biases, yMlx, yJit);
            try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
                plan.withGridScheduler(new GridScheduler("qmm.jit", perOutput(m * n))).execute();
            }
            String what = "m=" + m + " k=" + k + " n=" + n + " bits=" + bits;
            double[] expected = matmulTransposed(xv, 0, decode(wq, scales, biases, n, k, groupSize, bits), 0, m, k, n);
            assertAllClose("JIT quantizedMatmul " + what, expected, yJit, 1e-4, 1e-4);
            assertAllClose("MLX quantizedMatmul " + what, expected, yMlx, 1e-4, 1e-4);
        }
    }

    @Test
    public void testGatherQmm() throws TornadoExecutionPlanException {
        final int batches = 4, experts = 3, m = 2, k = 256, n = 24, groupSize = 32, bits = 4;
        float[] xv = values(batches * m * k, -1, 1, 71);
        FloatArray x = FloatArray.fromArray(xv);
        FloatArray w = FloatArray.fromArray(values(experts * n * k, -1, 1, 72));
        IntArray wq = new IntArray(experts * n * k * bits / 32);
        FloatArray scales = new FloatArray(experts * n * k / groupSize);
        FloatArray biases = new FloatArray(experts * n * k / groupSize);
        IntArray lhs = IntArray.fromElements(3, 1, 2, 0);
        IntArray rhs = IntArray.fromElements(2, 0, 1, 2);
        FloatArray yMlx = new FloatArray(batches * m * n);
        FloatArray yJit = new FloatArray(batches * m * n);
        TaskGraph graph = new TaskGraph("gqmm").transferToDevice(DataTransferMode.FIRST_EXECUTION, x, w, lhs, rhs) //
                .libraryTask("quantize", Mlx::quantize, w, wq, scales, biases, experts * n, k, groupSize, bits) //
                .libraryTask("mlx", Mlx::gatherQmm, x, wq, scales, biases, lhs, rhs, yMlx, batches, experts, m, k, n, groupSize, bits) //
                .task("jit", JitQuantized::gatherQmm, new KernelContext(), x, wq, scales, biases, lhs, rhs, yJit, m, k, n, groupSize, bits) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, wq, scales, biases, yMlx, yJit);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(new GridScheduler("gqmm.jit", perOutput(batches * m * n))).execute();
        }
        double[] wDeq = decode(wq, scales, biases, experts * n, k, groupSize, bits);
        double[] expected = new double[batches * m * n];
        for (int b = 0; b < batches; b++) {
            double[] yb = matmulTransposed(xv, lhs.get(b) * m * k, wDeq, rhs.get(b) * n * k, m, k, n);
            System.arraycopy(yb, 0, expected, b * m * n, m * n);
        }
        assertAllClose("JIT gatherQmm", expected, yJit, 1e-4, 1e-4);
        assertAllClose("MLX gatherQmm", expected, yMlx, 1e-4, 1e-4);
        assertTrue(Arrays.stream(expected).anyMatch(v -> v != 0));
    }
}
