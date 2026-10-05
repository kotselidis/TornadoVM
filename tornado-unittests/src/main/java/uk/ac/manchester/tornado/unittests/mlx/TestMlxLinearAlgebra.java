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
import static org.junit.Assert.assertTrue;

import java.util.Arrays;

import org.junit.Test;

import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.BFloat16Array;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.mlx.MlxLinearAlgebra;
import java.util.Random;

/**
 * Unit tests for the MLX linear-algebra library tasks: matmul, transposed matmul, addmm, einsum,
 * tensordot, inner, outer and Kronecker products, segmented, gathered and block-masked matmuls, the
 * Hadamard transform, cross products, norms, and the decompositions, inverses and solves MLX runs on
 * its CPU stream. Arrays are row-major, as in MLX. Each test checks the task against a sequential Java
 * reference. Skipped unless the default device is on the Metal backend and mlx.metallib is available.
 *
 * <p>
 * How to run?
 * </p>
 * <code>
 * tornado-test -V uk.ac.manchester.tornado.unittests.mlx.TestMlxLinearAlgebra
 * </code>
 */
public class TestMlxLinearAlgebra extends MlxTestBase {

    /** c[m, n] = a[m, k] @ b[k, n] (b at offset bOff), with b given as [n, k] if transposed. */
    private static double[] matmulJava(float[] a, int aOff, float[] b, int bOff, int m, int k, int n, boolean bTransposed) {
        double[] c = new double[m * n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0;
                for (int p = 0; p < k; p++) {
                    sum += (double) a[aOff + i * k + p] * (bTransposed ? b[bOff + j * k + p] : b[bOff + p * n + j]);
                }
                c[i * n + j] = sum;
            }
        }
        return c;
    }

    private static double[] matmulJava(float[] a, float[] b, int m, int k, int n, boolean bTransposed) {
        return matmulJava(a, 0, b, 0, m, k, n, bTransposed);
    }

    @Test
    public void testMatmul() throws TornadoExecutionPlanException {
        final int m = 37;
        final int k = 65;
        final int n = 29;
        float[] av = values(m * k, -1, 1, 1);
        float[] bv = values(k * n, -1, 1, 2);
        FloatArray a = FloatArray.fromArray(av);
        FloatArray b = FloatArray.fromArray(bv);
        FloatArray c = new FloatArray(m * n);

        TaskGraph taskGraph = new TaskGraph("g") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, a, b) //
                .libraryTask("matmul", MlxLinearAlgebra::matmul, a, b, c, m, k, n) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, c);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
            plan.execute();
        }

        assertAllClose("matmul", matmulJava(av, bv, m, k, n, false), c, 1e-4, 1e-4);
    }

    @Test
    public void testMatmulTransposed() throws TornadoExecutionPlanException {
        final int m = 64;
        final int k = 256;
        final int n = 96;
        float[] av = values(m * k, -1, 1, 3);
        float[] wv = values(n * k, -1, 1, 4);
        FloatArray a = FloatArray.fromArray(av);
        FloatArray w = FloatArray.fromArray(wv);
        FloatArray c = new FloatArray(m * n);

        TaskGraph taskGraph = new TaskGraph("g") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, a, w) //
                .libraryTask("matmul", MlxLinearAlgebra::matmulTransposed, a, w, c, m, k, n) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, c);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
            plan.execute();
        }

        assertAllClose("matmulTransposed", matmulJava(av, wv, m, k, n, true), c, 1e-4, 1e-4);
    }

    @Test
    public void testAddmm() throws TornadoExecutionPlanException {
        final int m = 17;
        final int k = 40;
        final int n = 23;
        final float alpha = 0.5f;
        final float beta = 2.0f;
        float[] cv = values(m * n, -1, 1, 5);
        float[] av = values(m * k, -1, 1, 6);
        float[] bv = values(k * n, -1, 1, 7);
        FloatArray cIn = FloatArray.fromArray(cv);
        FloatArray a = FloatArray.fromArray(av);
        FloatArray b = FloatArray.fromArray(bv);
        FloatArray output = new FloatArray(m * n);

        TaskGraph taskGraph = new TaskGraph("g") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, cIn, a, b) //
                .libraryTask("addmm", MlxLinearAlgebra::addmm, cIn, a, b, output, m, k, n, alpha, beta) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
            plan.execute();
        }

        double[] ab = matmulJava(av, bv, m, k, n, false);
        double[] expected = new double[m * n];
        for (int i = 0; i < expected.length; i++) {
            expected[i] = alpha * ab[i] + beta * cv[i];
        }
        assertAllClose("addmm", expected, output, 1e-4, 1e-4);
    }

    @Test
    public void testEinsumBatchedMatmul() throws TornadoExecutionPlanException {
        final int batch = 3;
        final int m = 64;
        final int k = 32;
        final int n = 96;
        float[] av = values(batch * m * k, -1, 1, 8);
        float[] bv = values(batch * k * n, -1, 1, 9);
        FloatArray a = FloatArray.fromArray(av);
        FloatArray b = FloatArray.fromArray(bv);
        FloatArray output = new FloatArray(batch * m * n);

        TaskGraph taskGraph = new TaskGraph("g") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, a, b) //
                .libraryTask("einsum", MlxLinearAlgebra::einsumBatchedMatmul, a, b, output, batch, m, k, n) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
            plan.execute();
        }

        double[] expected = new double[batch * m * n];
        for (int bb = 0; bb < batch; bb++) {
            System.arraycopy(matmulJava(av, bb * m * k, bv, bb * k * n, m, k, n, false), 0, expected, bb * m * n, m * n);
        }
        assertAllClose("einsum bij,bjk->bik", expected, output, 1e-4, 1e-4);
    }

    @Test
    public void testTensordot() throws TornadoExecutionPlanException {
        // a[m, 4, 8] and b[4, 8, n], contracted over both middle axes: a[m, 32] @ b[32, n].
        final int m = 64;
        final int n = 96;
        float[] av = values(m * 32, -1, 1, 10);
        float[] bv = values(32 * n, -1, 1, 11);
        FloatArray a = FloatArray.fromArray(av);
        FloatArray b = FloatArray.fromArray(bv);
        FloatArray output = new FloatArray(m * n);

        TaskGraph taskGraph = new TaskGraph("g") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, a, b) //
                .libraryTask("tensordot", MlxLinearAlgebra::tensordot, a, b, output, m, 4, 8, n) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
            plan.execute();
        }

        assertAllClose("tensordot", matmulJava(av, bv, m, 32, n, false), output, 1e-4, 1e-4);
    }

    @Test
    public void testTensordotAxis() throws TornadoExecutionPlanException {
        final int m = 64;
        final int k = 32;
        final int n = 96;
        float[] av = values(m * k, -1, 1, 12);
        float[] bv = values(k * n, -1, 1, 13);
        FloatArray a = FloatArray.fromArray(av);
        FloatArray b = FloatArray.fromArray(bv);
        FloatArray output = new FloatArray(m * n);

        TaskGraph taskGraph = new TaskGraph("g") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, a, b) //
                .libraryTask("tensordot", MlxLinearAlgebra::tensordotAxis, a, b, output, m, k, n) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
            plan.execute();
        }

        assertAllClose("tensordotAxis", matmulJava(av, bv, m, k, n, false), output, 1e-4, 1e-4);
    }

    @Test
    public void testInner() throws TornadoExecutionPlanException {
        final int n = 100_000;
        float[] av = values(n, -1, 1, 14);
        float[] bv = values(n, -1, 1, 15);
        FloatArray a = FloatArray.fromArray(av);
        FloatArray b = FloatArray.fromArray(bv);
        FloatArray output = new FloatArray(1);

        TaskGraph taskGraph = new TaskGraph("g") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, a, b) //
                .libraryTask("inner", MlxLinearAlgebra::inner, a, b, output) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
            plan.execute();
        }

        double dot = 0;
        for (int i = 0; i < n; i++) {
            dot += (double) av[i] * bv[i];
        }
        assertAllClose("inner", new double[] { dot }, output, 1e-3, 1e-3);
    }

    @Test
    public void testOuter() throws TornadoExecutionPlanException {
        final int m = 37;
        final int n = 53;
        float[] av = values(m, -2, 2, 16);
        float[] bv = values(n, -2, 2, 17);
        FloatArray a = FloatArray.fromArray(av);
        FloatArray b = FloatArray.fromArray(bv);
        FloatArray output = new FloatArray(m * n);

        TaskGraph taskGraph = new TaskGraph("g") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, a, b) //
                .libraryTask("outer", MlxLinearAlgebra::outer, a, b, output) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
            plan.execute();
        }

        double[] expected = new double[m * n];
        for (int t = 0; t < expected.length; t++) {
            expected[t] = (double) av[t / n] * bv[t % n];
        }
        assertAllClose("outer", expected, output, 1e-6, 1e-6);
    }

    @Test
    public void testKron() throws TornadoExecutionPlanException {
        // a[5, 4] (x) b[3, 6] = out[15, 24]
        float[] av = values(5 * 4, -2, 2, 18);
        float[] bv = values(3 * 6, -2, 2, 19);
        FloatArray a = FloatArray.fromArray(av);
        FloatArray b = FloatArray.fromArray(bv);
        FloatArray output = new FloatArray(15 * 24);

        TaskGraph taskGraph = new TaskGraph("g") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, a, b) //
                .libraryTask("kron", MlxLinearAlgebra::kron, a, b, output, 5, 4, 3, 6) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
            plan.execute();
        }

        double[] expected = new double[15 * 24];
        for (int t = 0; t < expected.length; t++) {
            int r = t / 24;
            int c = t % 24;
            expected[t] = (double) av[(r / 3) * 4 + c / 6] * bv[(r % 3) * 6 + c % 6];
        }
        assertAllClose("kron", expected, output, 1e-6, 1e-6);
    }

    @Test
    public void testSegmentedMm() throws TornadoExecutionPlanException {
        // Four segments of the k axis: [0, 10), [10, 64), [5, 5) (empty) and [20, 40).
        final int m = 16;
        final int k = 64;
        final int n = 24;
        int[] segs = { 0, 10, 10, 64, 5, 5, 20, 40 };
        float[] av = values(m * k, -1, 1, 20);
        float[] bv = values(k * n, -1, 1, 21);
        IntArray segments = IntArray.fromArray(segs);
        FloatArray a = FloatArray.fromArray(av);
        FloatArray b = FloatArray.fromArray(bv);
        FloatArray output = new FloatArray(4 * m * n);

        TaskGraph taskGraph = new TaskGraph("g") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, segments, a, b) //
                .libraryTask("segmentedMm", MlxLinearAlgebra::segmentedMm, a, b, segments, output, m, k, n) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
            plan.execute();
        }

        double[] expected = new double[4 * m * n];
        for (int s = 0; s < 4; s++) {
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    double acc = 0;
                    for (int p = segs[2 * s]; p < segs[2 * s + 1]; p++) {
                        acc += (double) av[i * k + p] * bv[p * n + j];
                    }
                    expected[(s * m + i) * n + j] = acc;
                }
            }
        }
        assertAllClose("segmentedMm", expected, output, 1e-4, 1e-4);
    }

    @Test
    public void testGatherMm() throws TornadoExecutionPlanException {
        // out[i] = a[lhs[i]] @ b[rhs[i]] for five (lhs, rhs) pairs over 4 a and 3 b matrices.
        final int batchesA = 4;
        final int batchesB = 3;
        final int m = 64;
        final int k = 32;
        final int n = 96;
        int[] lhs = { 3, 0, 1, 2, 3 };
        int[] rhs = { 2, 0, 1, 1, 2 };
        float[] av = values(batchesA * m * k, -1, 1, 22);
        float[] bv = values(batchesB * k * n, -1, 1, 23);
        FloatArray a = FloatArray.fromArray(av);
        FloatArray b = FloatArray.fromArray(bv);
        IntArray lhsIndices = IntArray.fromArray(lhs);
        IntArray rhsIndices = IntArray.fromArray(rhs);
        FloatArray output = new FloatArray(lhs.length * m * n);

        TaskGraph taskGraph = new TaskGraph("g") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, a, b, lhsIndices, rhsIndices) //
                .libraryTask("gatherMm", MlxLinearAlgebra::gatherMm, a, b, lhsIndices, rhsIndices, output, batchesA, batchesB, m, k, n) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
            plan.execute();
        }

        double[] expected = new double[lhs.length * m * n];
        for (int i = 0; i < lhs.length; i++) {
            System.arraycopy(matmulJava(av, lhs[i] * m * k, bv, rhs[i] * k * n, m, k, n, false), 0, expected, i * m * n, m * n);
        }
        assertAllClose("gatherMm", expected, output, 1e-4, 1e-4);
    }

    @Test
    public void testCross() throws TornadoExecutionPlanException {
        final int count = 1000;
        float[] av = values(count * 3, -3, 3, 24);
        float[] bv = values(count * 3, -3, 3, 25);
        FloatArray a = FloatArray.fromArray(av);
        FloatArray b = FloatArray.fromArray(bv);
        FloatArray output = new FloatArray(count * 3);

        TaskGraph taskGraph = new TaskGraph("g") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, a, b) //
                .libraryTask("cross", MlxLinearAlgebra::cross, a, b, output, count) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
            plan.execute();
        }

        double[] expected = new double[count * 3];
        for (int i = 0; i < count; i++) {
            int p = 3 * i;
            expected[p] = (double) av[p + 1] * bv[p + 2] - (double) av[p + 2] * bv[p + 1];
            expected[p + 1] = (double) av[p + 2] * bv[p] - (double) av[p] * bv[p + 2];
            expected[p + 2] = (double) av[p] * bv[p + 1] - (double) av[p + 1] * bv[p];
        }
        assertAllClose("cross", expected, output, 1e-5, 1e-5);
    }

    @Test
    public void testNorm() throws TornadoExecutionPlanException {
        final int rows = 37;
        final int cols = 300;
        float[] xv = values(rows * cols, -2, 2, 26);
        FloatArray x = FloatArray.fromArray(xv);
        for (float ord : new float[] { 1f, 2f, 3f, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY }) {
            FloatArray output = new FloatArray(rows);

            TaskGraph taskGraph = new TaskGraph("g") //
                    .transferToDevice(DataTransferMode.EVERY_EXECUTION, x) //
                    .libraryTask("norm", MlxLinearAlgebra::norm, x, output, rows, cols, ord) //
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

            try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
                plan.execute();
            }

            double[] expected = new double[rows];
            for (int r = 0; r < rows; r++) {
                double acc = ord == Float.NEGATIVE_INFINITY ? Double.MAX_VALUE : 0;
                for (int c = 0; c < cols; c++) {
                    double v = Math.abs(xv[r * cols + c]);
                    acc = ord == Float.POSITIVE_INFINITY ? Math.max(acc, v) : ord == Float.NEGATIVE_INFINITY ? Math.min(acc, v) : acc + Math.pow(v, ord);
                }
                expected[r] = Double.isInfinite(ord) ? acc : Math.pow(acc, 1.0 / ord);
            }
            assertAllClose("norm ord=" + ord, expected, output, 1e-4, 1e-5);
        }
    }

    @Test
    public void testL2Norm() throws TornadoExecutionPlanException {
        final int rows = 37;
        final int cols = 300;
        float[] xv = values(rows * cols, -2, 2, 27);
        FloatArray x = FloatArray.fromArray(xv);
        FloatArray output = new FloatArray(rows);

        TaskGraph taskGraph = new TaskGraph("g") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, x) //
                .libraryTask("l2Norm", MlxLinearAlgebra::l2Norm, x, output, rows, cols) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
            plan.execute();
        }

        double[] expected = new double[rows];
        for (int r = 0; r < rows; r++) {
            double s = 0;
            for (int c = 0; c < cols; c++) {
                s += (double) xv[r * cols + c] * xv[r * cols + c];
            }
            expected[r] = Math.sqrt(s);
        }
        assertAllClose("l2Norm", expected, output, 1e-5, 1e-5);
    }

    @Test
    public void testFrobeniusNorm() throws TornadoExecutionPlanException {
        final int batch = 6;
        final int rows = 37;
        final int cols = 50;
        float[] xv = values(batch * rows * cols, -2, 2, 28);
        FloatArray x = FloatArray.fromArray(xv);
        FloatArray output = new FloatArray(batch);

        TaskGraph taskGraph = new TaskGraph("g") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, x) //
                .libraryTask("frobenius", MlxLinearAlgebra::frobeniusNorm, x, output, batch, rows, cols) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
            plan.execute();
        }

        double[] expected = new double[batch];
        for (int b = 0; b < batch; b++) {
            double s = 0;
            for (int i = 0; i < rows * cols; i++) {
                s += (double) xv[b * rows * cols + i] * xv[b * rows * cols + i];
            }
            expected[b] = Math.sqrt(s);
        }
        assertAllClose("frobeniusNorm", expected, output, 1e-5, 1e-5);
    }

    // ---------------------------------------------------------------- other shapes and types

    @Test
    public void testMatmulTransposedAsGemv() throws TornadoExecutionPlanException {
        // m = 1: y[1, rows] = x[1, cols] @ w[rows, cols]^T, the decode-time matrix-vector product.
        final int rows = 1000;
        final int cols = 2048;
        float[] wv = values(rows * cols, -0.05f, 0.05f, 29);
        float[] xv = values(cols, -1, 1, 30);
        FloatArray w = FloatArray.fromArray(wv);
        FloatArray x = FloatArray.fromArray(xv);
        FloatArray y = new FloatArray(rows);

        TaskGraph taskGraph = new TaskGraph("g") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, w, x) //
                .libraryTask("gemv", MlxLinearAlgebra::matmulTransposed, x, w, y, 1, cols, rows) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, y);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
            plan.execute();
        }

        assertAllClose("gemv", matmulJava(xv, wv, 1, cols, rows, true), y, 1e-4, 1e-4);
    }

    @Test
    public void testMatmulHalf() throws TornadoExecutionPlanException {
        final int m = 64;
        final int k = 96;
        final int n = 80;
        HalfFloatArray a = half(values(m * k, -1, 1, 31));
        HalfFloatArray b = half(values(k * n, -1, 1, 32));
        HalfFloatArray c = new HalfFloatArray(m * n);

        TaskGraph taskGraph = new TaskGraph("g") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, a, b) //
                .libraryTask("matmul", MlxLinearAlgebra::matmul, a, b, c, m, k, n) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, c);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
            plan.execute();
        }

        assertAllClose("matmul float16", matmulJava(widen(a), widen(b), m, k, n, false), c, 2e-3, 2e-2);
    }

    @Test
    public void testMatmulBFloat16() throws TornadoExecutionPlanException {
        final int m = 64;
        final int k = 96;
        final int n = 80;
        BFloat16Array a = bf16(values(m * k, -1, 1, 33));
        BFloat16Array b = bf16(values(k * n, -1, 1, 34));
        BFloat16Array c = new BFloat16Array(m * n);

        TaskGraph taskGraph = new TaskGraph("g") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, a, b) //
                .libraryTask("matmul", MlxLinearAlgebra::matmul, a, b, c, m, k, n) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, c);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
            plan.execute();
        }

        assertAllClose("matmul bfloat16", matmulJava(widen(a), widen(b), m, k, n, false), c, 1e-2, 1e-1);
    }

    @Test
    public void testMatmulTransposedHalfAsGemv() throws TornadoExecutionPlanException {
        final int rows = 512;
        final int cols = 4096;
        HalfFloatArray w = half(values(rows * cols, -0.05f, 0.05f, 35));
        HalfFloatArray x = half(values(cols, -1, 1, 36));
        HalfFloatArray y = new HalfFloatArray(rows);

        TaskGraph taskGraph = new TaskGraph("g") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, w, x) //
                .libraryTask("gemv", MlxLinearAlgebra::matmulTransposed, x, w, y, 1, cols, rows) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, y);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
            plan.execute();
        }

        assertAllClose("gemv float16", matmulJava(widen(x), widen(w), 1, cols, rows, true), y, 2e-3, 2e-3);
    }

    @Test
    public void testCrossHalf() throws TornadoExecutionPlanException {
        final int count = 100;
        HalfFloatArray a = half(values(count * 3, -2, 2, 37));
        HalfFloatArray b = half(values(count * 3, -2, 2, 38));
        HalfFloatArray output = new HalfFloatArray(count * 3);

        TaskGraph taskGraph = new TaskGraph("g") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, a, b) //
                .libraryTask("cross", MlxLinearAlgebra::cross, a, b, output, count) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
            plan.execute();
        }

        float[] x = widen(a);
        float[] y = widen(b);
        double[] expected = new double[count * 3];
        for (int i = 0; i < count; i++) {
            int p = 3 * i;
            expected[p] = x[p + 1] * y[p + 2] - x[p + 2] * y[p + 1];
            expected[p + 1] = x[p + 2] * y[p] - x[p] * y[p + 2];
            expected[p + 2] = x[p] * y[p + 1] - x[p + 1] * y[p];
        }
        assertAllClose("cross float16", expected, output, 5e-3, 5e-3);
    }

    @Test
    public void testMatmulOneByOne() throws TornadoExecutionPlanException {
        float[] av = values(8, -1, 1, 39);
        float[] bv = values(8, -1, 1, 40);
        FloatArray a = FloatArray.fromArray(av);
        FloatArray b = FloatArray.fromArray(bv);
        FloatArray c = new FloatArray(1);

        TaskGraph taskGraph = new TaskGraph("g") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, a, b) //
                .libraryTask("matmul", MlxLinearAlgebra::matmul, a, b, c, 1, 8, 1) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, c);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
            plan.execute();
        }

        assertAllClose("matmul 1x8x1", matmulJava(av, bv, 1, 8, 1, false), c, 1e-5, 1e-5);
    }

    @Test
    public void testBlockMaskedMm() throws TornadoExecutionPlanException {
        // 128x128 by 128x128 in 32x32 blocks; masks zero out output blocks and drop operand blocks.
        final int m = 128;
        final int k = 128;
        final int n = 128;
        final int bs = 32;
        float[] av = values(m * k, -1, 1, 41);
        float[] bv = values(k * n, -1, 1, 42);
        Random random = new Random(43);
        byte[] mo = new byte[16];
        byte[] ml = new byte[16];
        byte[] mr = new byte[16];
        for (int i = 0; i < 16; i++) {
            mo[i] = (byte) (random.nextInt(4) == 0 ? 0 : 1);
            ml[i] = (byte) (random.nextInt(3) == 0 ? 0 : 1);
            mr[i] = (byte) (random.nextInt(3) == 0 ? 0 : 1);
        }
        FloatArray a = FloatArray.fromArray(av);
        FloatArray b = FloatArray.fromArray(bv);
        ByteArray maskOut = ByteArray.fromArray(mo);
        ByteArray maskLhs = ByteArray.fromArray(ml);
        ByteArray maskRhs = ByteArray.fromArray(mr);
        FloatArray output = new FloatArray(m * n);

        TaskGraph taskGraph = new TaskGraph("g") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, a, b, maskOut, maskLhs, maskRhs) //
                .libraryTask("blockMaskedMm", MlxLinearAlgebra::blockMaskedMm, a, b, maskOut, maskLhs, maskRhs, output, m, k, n, bs) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
            plan.execute();
        }

        double[] expected = new double[m * n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (mo[(i / bs) * 4 + j / bs] == 0) {
                    continue;
                }
                double s = 0;
                for (int p = 0; p < k; p++) {
                    if (ml[(i / bs) * 4 + p / bs] != 0 && mr[(p / bs) * 4 + j / bs] != 0) {
                        s += (double) av[i * k + p] * bv[p * n + j];
                    }
                }
                expected[i * n + j] = s;
            }
        }
        assertAllClose("blockMaskedMm", expected, output, 1e-4, 1e-4);
    }

    @Test
    public void testHadamardTransform() throws TornadoExecutionPlanException {
        final int rows = 5;
        final int n = 1024;
        final float scale = 1.0f / 32;
        float[] xv = values(rows * n, -1, 1, 44);
        FloatArray x = FloatArray.fromArray(xv);
        FloatArray output = new FloatArray(rows * n);

        TaskGraph taskGraph = new TaskGraph("g") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, x) //
                .libraryTask("hadamard", MlxLinearAlgebra::hadamardTransform, x, output, rows, n, scale) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
            plan.execute();
        }

        // H[i, j] = (-1)^popcount(i & j)
        double[] expected = new double[rows * n];
        for (int r = 0; r < rows; r++) {
            for (int i = 0; i < n; i++) {
                double s = 0;
                for (int j = 0; j < n; j++) {
                    s += (Integer.bitCount(i & j) % 2 == 0 ? 1 : -1) * xv[r * n + j];
                }
                expected[r * n + i] = s * scale;
            }
        }
        assertAllClose("hadamardTransform", expected, output, 1e-4, 1e-4);
    }

    // ---------------------------------------------------------------- decompositions, inverses and solves (MLX's CPU stream)

    private static final int BATCH = 5;
    private static final int ORDER = 16;

    /** Random batch of matrices with entries in [-1, 1) plus {@code diagonal} on the diagonal. */
    private static double[] matrices(int batch, int n, double diagonal, long seed) {
        Random r = new Random(seed);
        double[] m = new double[batch * n * n];
        for (int b = 0; b < batch; b++) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    m[(b * n + i) * n + j] = 2 * r.nextDouble() - 1 + (i == j ? diagonal : 0);
                }
            }
        }
        return m;
    }

    /** Symmetric positive definite: M M^T + n I. */
    private static double[] spd(int batch, int n, long seed) {
        double[] m = matrices(batch, n, 0, seed);
        double[] a = new double[m.length];
        for (int b = 0; b < batch; b++) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    double s = i == j ? n : 0;
                    for (int k = 0; k < n; k++) {
                        s += m[(b * n + i) * n + k] * m[(b * n + j) * n + k];
                    }
                    a[(b * n + i) * n + j] = s;
                }
            }
        }
        return a;
    }

    /** Symmetric: (M + M^T) / 2. */
    private static double[] symmetric(int batch, int n, long seed) {
        double[] m = matrices(batch, n, 0, seed);
        double[] a = new double[m.length];
        for (int b = 0; b < batch; b++) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    a[(b * n + i) * n + j] = 0.5 * (m[(b * n + i) * n + j] + m[(b * n + j) * n + i]);
                }
            }
        }
        return a;
    }

    /** Lower (or upper) triangular with diagonal in [1, 2). */
    private static double[] triangular(int batch, int n, boolean upper, long seed) {
        double[] m = matrices(batch, n, 0, seed);
        for (int b = 0; b < batch; b++) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    int e = (b * n + i) * n + j;
                    if (i == j) {
                        m[e] = 1.5 + 0.5 * m[e];
                    } else if (upper ? j < i : j > i) {
                        m[e] = 0;
                    }
                }
            }
        }
        return m;
    }

    private static double[] randomDoubles(int n, long seed) {
        Random r = new Random(seed);
        double[] v = new double[n];
        for (int i = 0; i < n; i++) {
            v[i] = 2 * r.nextDouble() - 1;
        }
        return v;
    }

    private static float[] toFloat(double[] v) {
        float[] f = new float[v.length];
        for (int i = 0; i < v.length; i++) {
            f[i] = (float) v[i];
        }
        return f;
    }

    private static double[] doubles(FloatArray f) {
        double[] d = new double[f.getSize()];
        for (int i = 0; i < d.length; i++) {
            d[i] = f.get(i);
        }
        return d;
    }

    /** a[b] @ c[b] for [batch, n, k] and [batch, k, m]. */
    private static double[] batchedMatmulJava(double[] a, double[] c, int batch, int n, int k, int m) {
        double[] out = new double[batch * n * m];
        for (int b = 0; b < batch; b++) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    double s = 0;
                    for (int p = 0; p < k; p++) {
                        s += a[(b * n + i) * k + p] * c[(b * k + p) * m + j];
                    }
                    out[(b * n + i) * m + j] = s;
                }
            }
        }
        return out;
    }

    private static double[] transposeJava(double[] a, int batch, int n) {
        double[] t = new double[a.length];
        for (int b = 0; b < batch; b++) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    t[(b * n + j) * n + i] = a[(b * n + i) * n + j];
                }
            }
        }
        return t;
    }

    /** Inverse of each matrix by Gauss-Jordan with partial pivoting, in double precision. */
    private static double[] inverseJava(double[] a, int batch, int n) {
        double[] out = new double[a.length];
        for (int b = 0; b < batch; b++) {
            double[][] m = new double[n][2 * n];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    m[i][j] = a[(b * n + i) * n + j];
                }
                m[i][n + i] = 1;
            }
            for (int c = 0; c < n; c++) {
                int p = c;
                for (int r = c + 1; r < n; r++) {
                    if (Math.abs(m[r][c]) > Math.abs(m[p][c])) {
                        p = r;
                    }
                }
                double[] t = m[c];
                m[c] = m[p];
                m[p] = t;
                double d = m[c][c];
                for (int j = 0; j < 2 * n; j++) {
                    m[c][j] /= d;
                }
                for (int r = 0; r < n; r++) {
                    if (r != c) {
                        double f = m[r][c];
                        for (int j = 0; j < 2 * n; j++) {
                            m[r][j] -= f * m[c][j];
                        }
                    }
                }
            }
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    out[(b * n + i) * n + j] = m[i][n + j];
                }
            }
        }
        return out;
    }

    /** Lower Cholesky factor of each SPD matrix, in double precision. */
    private static double[] choleskyJava(double[] a, int batch, int n) {
        double[] l = new double[a.length];
        for (int b = 0; b < batch; b++) {
            for (int j = 0; j < n; j++) {
                double d = a[(b * n + j) * n + j];
                for (int k = 0; k < j; k++) {
                    d -= l[(b * n + j) * n + k] * l[(b * n + j) * n + k];
                }
                l[(b * n + j) * n + j] = Math.sqrt(d);
                for (int i = j + 1; i < n; i++) {
                    double s = a[(b * n + i) * n + j];
                    for (int k = 0; k < j; k++) {
                        s -= l[(b * n + i) * n + k] * l[(b * n + j) * n + k];
                    }
                    l[(b * n + i) * n + j] = s / l[(b * n + j) * n + j];
                }
            }
        }
        return l;
    }

    /** Eigenvalues (ascending) of each symmetric matrix by cyclic Jacobi, in double precision. */
    private static double[] eigenvaluesJava(double[] a, int batch, int n) {
        double[] out = new double[batch * n];
        for (int b = 0; b < batch; b++) {
            double[][] m = new double[n][n];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    m[i][j] = a[(b * n + i) * n + j];
                }
            }
            for (int sweep = 0; sweep < 100; sweep++) {
                double off = 0;
                for (int p = 0; p < n; p++) {
                    for (int q = p + 1; q < n; q++) {
                        off += m[p][q] * m[p][q];
                    }
                }
                if (off < 1e-30) {
                    break;
                }
                for (int p = 0; p < n; p++) {
                    for (int q = p + 1; q < n; q++) {
                        if (m[p][q] == 0) {
                            continue;
                        }
                        double theta = (m[q][q] - m[p][p]) / (2 * m[p][q]);
                        double t = Math.signum(theta == 0 ? 1 : theta) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
                        double c = 1 / Math.sqrt(t * t + 1);
                        double sn = t * c;
                        for (int i = 0; i < n; i++) {
                            double mip = m[i][p];
                            double miq = m[i][q];
                            m[i][p] = c * mip - sn * miq;
                            m[i][q] = sn * mip + c * miq;
                        }
                        for (int j = 0; j < n; j++) {
                            double mpj = m[p][j];
                            double mqj = m[q][j];
                            m[p][j] = c * mpj - sn * mqj;
                            m[q][j] = sn * mpj + c * mqj;
                        }
                    }
                }
            }
            double[] d = new double[n];
            for (int i = 0; i < n; i++) {
                d[i] = m[i][i];
            }
            Arrays.sort(d);
            System.arraycopy(d, 0, out, b * n, n);
        }
        return out;
    }

    private static double[] identityJava(int batch, int n) {
        double[] id = new double[batch * n * n];
        for (int b = 0; b < batch; b++) {
            for (int i = 0; i < n; i++) {
                id[(b * n + i) * n + i] = 1;
            }
        }
        return id;
    }

    @Test
    public void testCholesky() throws TornadoExecutionPlanException {
        double[] a = spd(BATCH, ORDER, 45);
        double[] l = choleskyJava(a, BATCH, ORDER);
        FloatArray input = FloatArray.fromArray(toFloat(a));
        for (boolean upper : new boolean[] { false, true }) {
            FloatArray output = new FloatArray(a.length);

            TaskGraph taskGraph = new TaskGraph("g") //
                    .transferToDevice(DataTransferMode.EVERY_EXECUTION, input) //
                    .libraryTask("cholesky", MlxLinearAlgebra::cholesky, input, output, BATCH, ORDER, upper) //
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

            try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
                plan.execute();
            }

            // The factor is unique, so it is compared directly.
            assertAllClose("cholesky upper=" + upper, upper ? transposeJava(l, BATCH, ORDER) : l, output, 1e-4, 1e-4);
        }
    }

    @Test
    public void testCholeskyInv() throws TornadoExecutionPlanException {
        // (L L^T)^-1 from a lower Cholesky factor L.
        double[] l = triangular(BATCH, ORDER, false, 46);
        FloatArray factor = FloatArray.fromArray(toFloat(l));
        FloatArray output = new FloatArray(l.length);

        TaskGraph taskGraph = new TaskGraph("g") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, factor) //
                .libraryTask("choleskyInv", MlxLinearAlgebra::choleskyInv, factor, output, BATCH, ORDER, false) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
            plan.execute();
        }

        double[] llt = batchedMatmulJava(l, transposeJava(l, BATCH, ORDER), BATCH, ORDER, ORDER, ORDER);
        assertAllClose("choleskyInv", inverseJava(llt, BATCH, ORDER), output, 2e-3, 1e-4);
    }

    @Test
    public void testTriInv() throws TornadoExecutionPlanException {
        for (boolean upper : new boolean[] { false, true }) {
            double[] t = triangular(BATCH, ORDER, upper, 47);
            FloatArray input = FloatArray.fromArray(toFloat(t));
            FloatArray output = new FloatArray(t.length);

            TaskGraph taskGraph = new TaskGraph("g") //
                    .transferToDevice(DataTransferMode.EVERY_EXECUTION, input) //
                    .libraryTask("triInv", MlxLinearAlgebra::triInv, input, output, BATCH, ORDER, upper) //
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

            try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
                plan.execute();
            }

            assertAllClose("triInv upper=" + upper, inverseJava(t, BATCH, ORDER), output, 1e-3, 1e-4);
        }
    }

    @Test
    public void testInv() throws TornadoExecutionPlanException {
        double[] a = matrices(BATCH, ORDER, ORDER / 2.0 + 2, 48);
        FloatArray input = FloatArray.fromArray(toFloat(a));
        FloatArray output = new FloatArray(a.length);

        TaskGraph taskGraph = new TaskGraph("g") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, input) //
                .libraryTask("inv", MlxLinearAlgebra::inv, input, output, BATCH, ORDER) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
            plan.execute();
        }

        assertAllClose("inv", inverseJava(a, BATCH, ORDER), output, 1e-3, 1e-4);
    }

    @Test
    public void testSolve() throws TornadoExecutionPlanException {
        final int nrhs = 5;
        double[] a = matrices(BATCH, ORDER, ORDER / 2.0 + 2, 49);
        double[] rhs = randomDoubles(BATCH * ORDER * nrhs, 50);
        FloatArray input = FloatArray.fromArray(toFloat(a));
        FloatArray b = FloatArray.fromArray(toFloat(rhs));
        FloatArray x = new FloatArray(rhs.length);

        TaskGraph taskGraph = new TaskGraph("g") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, input, b) //
                .libraryTask("solve", MlxLinearAlgebra::solve, input, b, x, BATCH, ORDER, nrhs) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, x);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
            plan.execute();
        }

        assertAllClose("solve", batchedMatmulJava(inverseJava(a, BATCH, ORDER), rhs, BATCH, ORDER, ORDER, nrhs), x, 1e-3, 1e-4);
    }

    @Test
    public void testSolveTriangular() throws TornadoExecutionPlanException {
        final int nrhs = 5;
        double[] t = triangular(BATCH, ORDER, true, 51);
        double[] rhs = randomDoubles(BATCH * ORDER * nrhs, 52);
        FloatArray input = FloatArray.fromArray(toFloat(t));
        FloatArray b = FloatArray.fromArray(toFloat(rhs));
        FloatArray x = new FloatArray(rhs.length);

        TaskGraph taskGraph = new TaskGraph("g") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, input, b) //
                .libraryTask("solveTriangular", MlxLinearAlgebra::solveTriangular, input, b, x, BATCH, ORDER, nrhs, true) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, x);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
            plan.execute();
        }

        assertAllClose("solveTriangular", batchedMatmulJava(inverseJava(t, BATCH, ORDER), rhs, BATCH, ORDER, ORDER, nrhs), x, 1e-3, 1e-4);
    }

    @Test
    public void testLu() throws TornadoExecutionPlanException {
        // a[i, :] = (l u)[perm[i], :], with l unit lower and u upper triangular.
        double[] a = matrices(BATCH, ORDER, 0, 53);
        FloatArray input = FloatArray.fromArray(toFloat(a));
        IntArray perm = new IntArray(BATCH * ORDER);
        FloatArray l = new FloatArray(a.length);
        FloatArray u = new FloatArray(a.length);

        TaskGraph taskGraph = new TaskGraph("g") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, input) //
                .libraryTask("lu", MlxLinearAlgebra::lu, input, perm, l, u, BATCH, ORDER) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, perm, l, u);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
            plan.execute();
        }

        double[] lv = doubles(l);
        double[] uv = doubles(u);
        double[] permuted = new double[a.length];
        for (int b = 0; b < BATCH; b++) {
            for (int i = 0; i < ORDER; i++) {
                int dst = perm.get(b * ORDER + i);
                for (int j = 0; j < ORDER; j++) {
                    permuted[(b * ORDER + dst) * ORDER + j] = a[(b * ORDER + i) * ORDER + j];
                    int e = (b * ORDER + i) * ORDER + j;
                    assertTrue("l not unit lower at " + e, j < i || (j == i ? Math.abs(lv[e] - 1) < 1e-6 : lv[e] == 0));
                    assertTrue("u not upper at " + e, j >= i || uv[e] == 0);
                }
            }
        }
        assertAllClose("a = (l u)[perm]", permuted, FloatArray.fromArray(toFloat(batchedMatmulJava(lv, uv, BATCH, ORDER, ORDER, ORDER))), 1e-3, 1e-4);
    }

    @Test
    public void testLuFactor() throws TornadoExecutionPlanException {
        // Packed LU with LAPACK getrf pivots (0-based): applying the row swaps to a gives l u.
        double[] a = matrices(BATCH, ORDER, 0, 54);
        FloatArray input = FloatArray.fromArray(toFloat(a));
        FloatArray lu = new FloatArray(a.length);
        IntArray pivots = new IntArray(BATCH * ORDER);

        TaskGraph taskGraph = new TaskGraph("g") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, input) //
                .libraryTask("luFactor", MlxLinearAlgebra::luFactor, input, lu, pivots, BATCH, ORDER) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, lu, pivots);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
            plan.execute();
        }

        double[] packed = doubles(lu);
        double[] l = new double[a.length];
        double[] u = new double[a.length];
        double[] swapped = a.clone();
        for (int b = 0; b < BATCH; b++) {
            for (int i = 0; i < ORDER; i++) {
                for (int j = 0; j < ORDER; j++) {
                    int e = (b * ORDER + i) * ORDER + j;
                    l[e] = j < i ? packed[e] : j == i ? 1 : 0;
                    u[e] = j >= i ? packed[e] : 0;
                }
            }
            for (int k = 0; k < ORDER; k++) {
                int p = pivots.get(b * ORDER + k);
                for (int j = 0; j < ORDER; j++) {
                    double t = swapped[(b * ORDER + k) * ORDER + j];
                    swapped[(b * ORDER + k) * ORDER + j] = swapped[(b * ORDER + p) * ORDER + j];
                    swapped[(b * ORDER + p) * ORDER + j] = t;
                }
            }
        }
        assertAllClose("P a = l u", swapped, FloatArray.fromArray(toFloat(batchedMatmulJava(l, u, BATCH, ORDER, ORDER, ORDER))), 1e-3, 1e-4);
    }

    @Test
    public void testQr() throws TornadoExecutionPlanException {
        double[] a = matrices(BATCH, ORDER, 0, 55);
        FloatArray input = FloatArray.fromArray(toFloat(a));
        FloatArray q = new FloatArray(a.length);
        FloatArray r = new FloatArray(a.length);

        TaskGraph taskGraph = new TaskGraph("g") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, input) //
                .libraryTask("qr", MlxLinearAlgebra::qr, input, q, r, BATCH, ORDER) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, q, r);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
            plan.execute();
        }

        double[] qv = doubles(q);
        double[] rv = doubles(r);
        assertAllClose("q r = a", a, FloatArray.fromArray(toFloat(batchedMatmulJava(qv, rv, BATCH, ORDER, ORDER, ORDER))), 1e-3, 1e-4);
        assertAllClose("q^T q = I", identityJava(BATCH, ORDER), FloatArray.fromArray(toFloat(batchedMatmulJava(transposeJava(qv, BATCH, ORDER), qv, BATCH, ORDER, ORDER, ORDER))), 1e-4,
                1e-4);
        for (int e = 0; e < a.length; e++) {
            int i = (e / ORDER) % ORDER;
            assertTrue("r not upper at " + e, e % ORDER >= i || rv[e] == 0);
        }
    }

    @Test
    public void testEigh() throws TornadoExecutionPlanException {
        double[] a = symmetric(BATCH, ORDER, 56);
        double[] w = eigenvaluesJava(a, BATCH, ORDER);
        FloatArray input = FloatArray.fromArray(toFloat(a));
        FloatArray values = new FloatArray(BATCH * ORDER);
        FloatArray vectors = new FloatArray(a.length);

        TaskGraph taskGraph = new TaskGraph("g") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, input) //
                .libraryTask("eigh", MlxLinearAlgebra::eigh, input, values, vectors, BATCH, ORDER, false) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, values, vectors);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
            plan.execute();
        }

        assertAllClose("eigh values", w, values, 1e-4, 2e-4);
        // A v = lambda v for every column v.
        double[] v = doubles(vectors);
        double[] av = batchedMatmulJava(a, v, BATCH, ORDER, ORDER, ORDER);
        double[] lv = new double[a.length];
        for (int b = 0; b < BATCH; b++) {
            for (int i = 0; i < ORDER; i++) {
                for (int k = 0; k < ORDER; k++) {
                    lv[(b * ORDER + i) * ORDER + k] = w[b * ORDER + k] * v[(b * ORDER + i) * ORDER + k];
                }
            }
        }
        assertAllClose("A v = w v", lv, FloatArray.fromArray(toFloat(av)), 1e-3, 1e-3);
    }

    @Test
    public void testEigvalsh() throws TornadoExecutionPlanException {
        double[] a = symmetric(BATCH, ORDER, 57);
        FloatArray input = FloatArray.fromArray(toFloat(a));
        FloatArray values = new FloatArray(BATCH * ORDER);

        TaskGraph taskGraph = new TaskGraph("g") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, input) //
                .libraryTask("eigvalsh", MlxLinearAlgebra::eigvalsh, input, values, BATCH, ORDER, true) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, values);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
            plan.execute();
        }

        assertAllClose("eigvalsh", eigenvaluesJava(a, BATCH, ORDER), values, 1e-4, 2e-4);
    }

    /** Singular values (descending) as square roots of the eigenvalues of A^T A. */
    private static double[] singularValuesJava(double[] a, int batch, int n) {
        double[] ev = eigenvaluesJava(batchedMatmulJava(transposeJava(a, batch, n), a, batch, n, n, n), batch, n);
        double[] sv = new double[batch * n];
        for (int b = 0; b < batch; b++) {
            for (int k = 0; k < n; k++) {
                sv[b * n + k] = Math.sqrt(Math.max(ev[b * n + n - 1 - k], 0));
            }
        }
        return sv;
    }

    @Test
    public void testSvd() throws TornadoExecutionPlanException {
        double[] a = matrices(BATCH, ORDER, 0, 58);
        FloatArray input = FloatArray.fromArray(toFloat(a));
        FloatArray u = new FloatArray(a.length);
        FloatArray s = new FloatArray(BATCH * ORDER);
        FloatArray vt = new FloatArray(a.length);

        TaskGraph taskGraph = new TaskGraph("g") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, input) //
                .libraryTask("svd", MlxLinearAlgebra::svd, input, u, s, vt, BATCH, ORDER) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, u, s, vt);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
            plan.execute();
        }

        assertAllClose("svd s", singularValuesJava(a, BATCH, ORDER), s, 1e-3, 1e-4);
        double[] uv = doubles(u);
        double[] us = new double[a.length];
        for (int b = 0; b < BATCH; b++) {
            for (int i = 0; i < ORDER; i++) {
                for (int k = 0; k < ORDER; k++) {
                    us[(b * ORDER + i) * ORDER + k] = uv[(b * ORDER + i) * ORDER + k] * s.get(b * ORDER + k);
                }
            }
        }
        assertAllClose("u s vt = a", a, FloatArray.fromArray(toFloat(batchedMatmulJava(us, doubles(vt), BATCH, ORDER, ORDER, ORDER))), 1e-3, 1e-4);
    }

    @Test
    public void testSingularValues() throws TornadoExecutionPlanException {
        double[] a = matrices(BATCH, ORDER, 0, 59);
        FloatArray input = FloatArray.fromArray(toFloat(a));
        FloatArray s = new FloatArray(BATCH * ORDER);

        TaskGraph taskGraph = new TaskGraph("g") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, input) //
                .libraryTask("singularValues", MlxLinearAlgebra::singularValues, input, s, BATCH, ORDER) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, s);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
            plan.execute();
        }

        assertAllClose("singularValues", singularValuesJava(a, BATCH, ORDER), s, 1e-3, 1e-4);
    }

    @Test
    public void testPinv() throws TornadoExecutionPlanException {
        // For invertible matrices the pseudo-inverse is the inverse.
        double[] a = matrices(BATCH, ORDER, ORDER / 2.0 + 2, 60);
        FloatArray input = FloatArray.fromArray(toFloat(a));
        FloatArray output = new FloatArray(a.length);

        TaskGraph taskGraph = new TaskGraph("g") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, input) //
                .libraryTask("pinv", MlxLinearAlgebra::pinv, input, output, BATCH, ORDER) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
            plan.execute();
        }

        assertAllClose("pinv", inverseJava(a, BATCH, ORDER), output, 5e-3, 5e-4);
    }

    /** max_k |A v_k - lambda_k v_k| over the complex eigenpairs of matrix b. */
    private static double residualJava(double[] a, FloatArray values, FloatArray vectors, int b, int n) {
        double worst = 0;
        for (int k = 0; k < n; k++) {
            double lr = values.get(2 * (b * n + k));
            double li = values.get(2 * (b * n + k) + 1);
            for (int i = 0; i < n; i++) {
                double sr = 0;
                double si = 0;
                for (int j = 0; j < n; j++) {
                    double aij = a[(b * n + i) * n + j];
                    sr += aij * vectors.get(2 * ((b * n + j) * n + k));
                    si += aij * vectors.get(2 * ((b * n + j) * n + k) + 1);
                }
                double vr = vectors.get(2 * ((b * n + i) * n + k));
                double vi = vectors.get(2 * ((b * n + i) * n + k) + 1);
                sr -= lr * vr - li * vi;
                si -= lr * vi + li * vr;
                worst = Math.max(worst, Math.hypot(sr, si));
            }
        }
        return worst;
    }

    @Test
    public void testEig() throws TornadoExecutionPlanException {
        // Complex eigenvalues and eigenvectors, interleaved (re, im): A v = lambda v, and the
        // eigenvalues sum to the trace.
        double[] a = matrices(BATCH, ORDER, 0, 61);
        FloatArray input = FloatArray.fromArray(toFloat(a));
        FloatArray values = new FloatArray(2 * BATCH * ORDER);
        FloatArray vectors = new FloatArray(2 * a.length);

        TaskGraph taskGraph = new TaskGraph("g") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, input) //
                .libraryTask("eig", MlxLinearAlgebra::eig, input, values, vectors, BATCH, ORDER) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, values, vectors);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
            plan.execute();
        }

        for (int b = 0; b < BATCH; b++) {
            double trace = 0;
            double sum = 0;
            for (int i = 0; i < ORDER; i++) {
                trace += a[(b * ORDER + i) * ORDER + i];
                sum += values.get(2 * (b * ORDER + i));
            }
            assertClose("matrix " + b + " sum of eigenvalues", 0, trace, sum, 1e-3, 1e-3);
            double residual = residualJava(a, values, vectors, b, ORDER);
            assertTrue("matrix " + b + " residual " + residual, residual < 1e-3 * ORDER);
        }
    }

    @Test
    public void testEigvals() throws TornadoExecutionPlanException {
        double[] a = matrices(BATCH, ORDER, 0, 62);
        FloatArray input = FloatArray.fromArray(toFloat(a));
        FloatArray values = new FloatArray(2 * BATCH * ORDER);

        TaskGraph taskGraph = new TaskGraph("g") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, input) //
                .libraryTask("eigvals", MlxLinearAlgebra::eigvals, input, values, BATCH, ORDER) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, values);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
            plan.execute();
        }

        for (int b = 0; b < BATCH; b++) {
            double trace = 0;
            double sum = 0;
            double imag = 0;
            for (int i = 0; i < ORDER; i++) {
                trace += a[(b * ORDER + i) * ORDER + i];
                sum += values.get(2 * (b * ORDER + i));
                imag += values.get(2 * (b * ORDER + i) + 1);
            }
            assertClose("matrix " + b + " sum of eigenvalues", 0, trace, sum, 1e-3, 1e-3);
            // A real matrix's eigenvalues come in conjugate pairs.
            assertEquals("matrix " + b + " sum of imaginary parts", 0.0, imag, 1e-3);
        }
    }
}
