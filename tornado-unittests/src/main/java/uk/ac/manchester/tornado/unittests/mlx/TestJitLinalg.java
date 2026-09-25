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

import java.util.Random;

import org.junit.Test;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.mlx.MlxLinalg;
import uk.ac.manchester.tornado.mlx.jit.JitLinalg;

/**
 * The MLX linear-algebra operations and their KernelContext JIT counterparts, run in one graph
 * on the same batched inputs and checked against double-precision Java references: cross
 * products and norms (MLX on the GPU), and Cholesky factorisation, triangular and Cholesky
 * inverses, inverse and linear solves (MLX on its CPU stream).
 *
 * <p>
 * How to run?
 * </p>
 * <code>
 * tornado-test -V uk.ac.manchester.tornado.unittests.mlx.TestJitLinalg
 * </code>
 */
public class TestJitLinalg extends MlxTestBase {

    private static final int BATCH = 5;
    private static final int[] ORDERS = { 7, 16, 32 };

    static GridScheduler perMatrix(String task, int batch) {
        return new GridScheduler(task, TestJitReduce.groups(batch, JitLinalg.THREADS));
    }

    private static void execute(TaskGraph g, GridScheduler gs) throws TornadoExecutionPlanException {
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(g.snapshot())) {
            plan.withGridScheduler(gs).execute();
        }
    }

    /** Random batch of matrices with entries in [-1, 1) plus {@code diagonal} on the diagonal. */
    static double[] matrices(int batch, int n, double diagonal, long seed) {
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
    static double[] spd(int batch, int n, long seed) {
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

    /** Lower (or upper) triangular with diagonal in [1, 2). */
    static double[] triangular(int batch, int n, boolean upper, long seed) {
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

    static float[] toFloat(double[] v) {
        float[] f = new float[v.length];
        for (int i = 0; i < v.length; i++) {
            f[i] = (float) v[i];
        }
        return f;
    }

    /** a[b] @ c[b] for [batch, n, k] and [batch, k, m]. */
    static double[] matmul(double[] a, double[] c, int batch, int n, int k, int m) {
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

    /** Inverse of each matrix by Gauss-Jordan in double precision. */
    static double[] inverse(double[] a, int batch, int n) {
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

    static double[] transpose(double[] a, int batch, int n) {
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

    private static void both(String what, double[] expected, FloatArray mlx, FloatArray jit, double relTol, double absTol) {
        assertAllClose(what + " JIT", expected, jit, relTol, absTol);
        assertAllClose(what + " MLX", expected, mlx, relTol, absTol);
    }

    @Test
    public void testCrossAndNorms() throws TornadoExecutionPlanException {
        final int count = 1000;
        final int rows = 37;
        final int cols = 300;
        final int batch = 6;
        float[] av = values(count * 3, -3, 3, 251);
        float[] bv = values(count * 3, -3, 3, 252);
        float[] xv = values(rows * cols, -2, 2, 253);
        FloatArray a = FloatArray.fromArray(av);
        FloatArray b = FloatArray.fromArray(bv);
        FloatArray x = FloatArray.fromArray(xv);
        FloatArray crossMlx = new FloatArray(count * 3);
        FloatArray crossJit = new FloatArray(count * 3);
        float[] ords = { 1f, 2f, 3f, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY };
        TaskGraph g = new TaskGraph("cn").transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b, x) //
                .libraryTask("m", MlxLinalg::cross, a, b, crossMlx, count) //
                .task("j", JitLinalg::crossProduct, new KernelContext(), a, b, crossJit, count) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, crossMlx, crossJit);
        execute(g, new GridScheduler("cn.j", TestJitElementwise.grid1D(count)));
        double[] eCross = new double[count * 3];
        for (int i = 0; i < count; i++) {
            int p = 3 * i;
            eCross[p] = (double) av[p + 1] * bv[p + 2] - (double) av[p + 2] * bv[p + 1];
            eCross[p + 1] = (double) av[p + 2] * bv[p] - (double) av[p] * bv[p + 2];
            eCross[p + 2] = (double) av[p] * bv[p + 1] - (double) av[p + 1] * bv[p];
        }
        both("cross", eCross, crossMlx, crossJit, 1e-5, 1e-5);

        for (float ord : ords) {
            FloatArray nMlx = new FloatArray(rows);
            FloatArray nJit = new FloatArray(rows);
            TaskGraph gn = new TaskGraph("nr").transferToDevice(DataTransferMode.FIRST_EXECUTION, x) //
                    .libraryTask("m", MlxLinalg::norm, x, nMlx, rows, cols, ord) //
                    .task("j", JitLinalg::normRows, new KernelContext(), x, nJit, cols, ord) //
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, nMlx, nJit);
            execute(gn, perMatrix("nr.j", rows));
            double[] e = new double[rows];
            for (int r = 0; r < rows; r++) {
                double acc = ord == Float.NEGATIVE_INFINITY ? Double.MAX_VALUE : 0;
                for (int c = 0; c < cols; c++) {
                    double v = Math.abs(xv[r * cols + c]);
                    acc = ord == Float.POSITIVE_INFINITY ? Math.max(acc, v) : ord == Float.NEGATIVE_INFINITY ? Math.min(acc, v) : acc + Math.pow(v, ord);
                }
                e[r] = Double.isInfinite(ord) ? acc : Math.pow(acc, 1.0 / ord);
            }
            both("norm ord=" + ord, e, nMlx, nJit, 1e-4, 1e-5);
        }

        FloatArray l2Mlx = new FloatArray(rows);
        FloatArray l2Jit = new FloatArray(rows);
        FloatArray froMlx = new FloatArray(batch);
        FloatArray froJit = new FloatArray(batch);
        int mr = rows * cols / batch / 50;
        TaskGraph g2 = new TaskGraph("l2").transferToDevice(DataTransferMode.FIRST_EXECUTION, x) //
                .libraryTask("m1", MlxLinalg::l2Norm, x, l2Mlx, rows, cols) //
                .libraryTask("m2", MlxLinalg::frobeniusNorm, x, froMlx, batch, mr, 50) //
                .task("j1", JitLinalg::normRows, new KernelContext(), x, l2Jit, cols, 2f) //
                .task("j2", JitLinalg::normRows, new KernelContext(), x, froJit, mr * 50, 2f) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, l2Mlx, l2Jit, froMlx, froJit);
        GridScheduler gs = new GridScheduler();
        gs.addWorkerGrid("l2.j1", TestJitReduce.groups(rows, JitLinalg.THREADS));
        gs.addWorkerGrid("l2.j2", TestJitReduce.groups(batch, JitLinalg.THREADS));
        execute(g2, gs);
        double[] eL2 = new double[rows];
        for (int r = 0; r < rows; r++) {
            double s = 0;
            for (int c = 0; c < cols; c++) {
                s += (double) xv[r * cols + c] * xv[r * cols + c];
            }
            eL2[r] = Math.sqrt(s);
        }
        double[] eFro = new double[batch];
        for (int bb = 0; bb < batch; bb++) {
            double s = 0;
            for (int i = 0; i < mr * 50; i++) {
                s += (double) xv[bb * mr * 50 + i] * xv[bb * mr * 50 + i];
            }
            eFro[bb] = Math.sqrt(s);
        }
        both("l2Norm", eL2, l2Mlx, l2Jit, 1e-5, 1e-5);
        both("frobeniusNorm", eFro, froMlx, froJit, 1e-5, 1e-5);
    }

    @Test
    public void testCholesky() throws TornadoExecutionPlanException {
        for (int n : ORDERS) {
            for (boolean upper : new boolean[] { false, true }) {
                double[] a = spd(BATCH, n, 261L + n);
                FloatArray fa = FloatArray.fromArray(toFloat(a));
                FloatArray outMlx = new FloatArray(a.length);
                FloatArray outJit = new FloatArray(a.length);
                TaskGraph g = new TaskGraph("ch").transferToDevice(DataTransferMode.FIRST_EXECUTION, fa) //
                        .libraryTask("m", MlxLinalg::cholesky, fa, outMlx, BATCH, n, upper) //
                        .task("j", JitLinalg::cholesky, new KernelContext(), fa, outJit, n, upper ? 1 : 0) //
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, outMlx, outJit);
                execute(g, perMatrix("ch.j", BATCH));
                // The factor is unique: compare against a double-precision Cholesky.
                double[] l = new double[a.length];
                for (int b = 0; b < BATCH; b++) {
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
                both("cholesky n=" + n + " upper=" + upper, upper ? transpose(l, BATCH, n) : l, outMlx, outJit, 1e-4, 1e-4);
            }
        }
    }

    @Test
    public void testTriangularInverses() throws TornadoExecutionPlanException {
        for (int n : ORDERS) {
            for (boolean upper : new boolean[] { false, true }) {
                double[] t = triangular(BATCH, n, upper, 271L + n);
                FloatArray ft = FloatArray.fromArray(toFloat(t));
                FloatArray triMlx = new FloatArray(t.length);
                FloatArray triJit = new FloatArray(t.length);
                FloatArray chiMlx = new FloatArray(t.length);
                FloatArray chiJit = new FloatArray(t.length);
                int u = upper ? 1 : 0;
                TaskGraph g = new TaskGraph("ti").transferToDevice(DataTransferMode.FIRST_EXECUTION, ft) //
                        .libraryTask("m1", MlxLinalg::triInv, ft, triMlx, BATCH, n, upper) //
                        .libraryTask("m2", MlxLinalg::choleskyInv, ft, chiMlx, BATCH, n, upper) //
                        .task("j1", JitLinalg::triInv, new KernelContext(), ft, triJit, n, u) //
                        .task("j2", JitLinalg::choleskyInv, new KernelContext(), ft, chiJit, n, u) //
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, triMlx, triJit, chiMlx, chiJit);
                GridScheduler gs = new GridScheduler();
                gs.addWorkerGrid("ti.j1", TestJitReduce.groups(BATCH, JitLinalg.THREADS));
                gs.addWorkerGrid("ti.j2", TestJitReduce.groups(BATCH, JitLinalg.THREADS));
                execute(g, gs);
                double[] tInv = inverse(t, BATCH, n);
                // (L L^T)^-1 for lower, (U^T U)^-1 for upper.
                double[] chol = upper ? matmul(transpose(t, BATCH, n), t, BATCH, n, n, n) : matmul(t, transpose(t, BATCH, n), BATCH, n, n, n);
                both("triInv n=" + n + " upper=" + upper, tInv, triMlx, triJit, 1e-3, 1e-4);
                both("choleskyInv n=" + n + " upper=" + upper, inverse(chol, BATCH, n), chiMlx, chiJit, 2e-3, 1e-4);
            }
        }
    }

    @Test
    public void testInverseAndSolves() throws TornadoExecutionPlanException {
        for (int n : ORDERS) {
            final int nrhs = 5;
            double[] a = matrices(BATCH, n, n / 2.0 + 2, 281L + n);
            double[] rhs = new double[BATCH * n * nrhs];
            Random r = new Random(282L + n);
            for (int i = 0; i < rhs.length; i++) {
                rhs[i] = 2 * r.nextDouble() - 1;
            }
            double[] t = triangular(BATCH, n, true, 283L + n);
            FloatArray fa = FloatArray.fromArray(toFloat(a));
            FloatArray fb = FloatArray.fromArray(toFloat(rhs));
            FloatArray ft = FloatArray.fromArray(toFloat(t));
            FloatArray invMlx = new FloatArray(a.length);
            FloatArray invJit = new FloatArray(a.length);
            FloatArray solMlx = new FloatArray(rhs.length);
            FloatArray solJit = new FloatArray(rhs.length);
            FloatArray triMlx = new FloatArray(rhs.length);
            FloatArray triJit = new FloatArray(rhs.length);
            TaskGraph g = new TaskGraph("is").transferToDevice(DataTransferMode.FIRST_EXECUTION, fa, fb, ft) //
                    .libraryTask("m1", MlxLinalg::inv, fa, invMlx, BATCH, n) //
                    .libraryTask("m2", MlxLinalg::solve, fa, fb, solMlx, BATCH, n, nrhs) //
                    .libraryTask("m3", MlxLinalg::solveTriangular, ft, fb, triMlx, BATCH, n, nrhs, true) //
                    .task("j1", JitLinalg::inv, new KernelContext(), fa, invJit, n) //
                    .task("j2", JitLinalg::solve, new KernelContext(), fa, fb, solJit, n, nrhs) //
                    .task("j3", JitLinalg::solveTriangular, new KernelContext(), ft, fb, triJit, n, nrhs, 1) //
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, invMlx, invJit, solMlx, solJit, triMlx, triJit);
            GridScheduler gs = new GridScheduler();
            gs.addWorkerGrid("is.j1", TestJitReduce.groups(BATCH, JitLinalg.THREADS));
            gs.addWorkerGrid("is.j2", TestJitReduce.groups(BATCH, JitLinalg.THREADS));
            gs.addWorkerGrid("is.j3", TestJitReduce.groups(BATCH, JitLinalg.THREADS));
            execute(g, gs);
            double[] aInv = inverse(a, BATCH, n);
            both("inv n=" + n, aInv, invMlx, invJit, 1e-3, 1e-4);
            both("solve n=" + n, matmul(aInv, rhs, BATCH, n, n, nrhs), solMlx, solJit, 1e-3, 1e-4);
            both("solveTriangular n=" + n, matmul(inverse(t, BATCH, n), rhs, BATCH, n, n, nrhs), triMlx, triJit, 1e-3, 1e-4);
        }
    }

    @Test
    public void testHalfCrossAndNorm() throws TornadoExecutionPlanException {
        final int count = 100;
        float[] av = values(count * 3, -2, 2, 291);
        float[] bv = values(count * 3, -2, 2, 292);
        HalfFloatArray a = half(av);
        HalfFloatArray b = half(bv);
        HalfFloatArray cross = new HalfFloatArray(count * 3);
        HalfFloatArray l2 = new HalfFloatArray(count);
        run(new TaskGraph("hc").transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b) //
                .libraryTask("c", MlxLinalg::cross, a, b, cross, count) //
                .libraryTask("n", MlxLinalg::l2Norm, a, l2, count, 3) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, cross, l2));
        float[] x = widen(a);
        float[] y = widen(b);
        double[] eCross = new double[count * 3];
        double[] eL2 = new double[count];
        for (int i = 0; i < count; i++) {
            int p = 3 * i;
            eCross[p] = x[p + 1] * y[p + 2] - x[p + 2] * y[p + 1];
            eCross[p + 1] = x[p + 2] * y[p] - x[p] * y[p + 2];
            eCross[p + 2] = x[p] * y[p + 1] - x[p + 1] * y[p];
            eL2[i] = Math.sqrt(x[p] * x[p] + x[p + 1] * x[p + 1] + x[p + 2] * x[p + 2]);
        }
        assertAllClose("cross float16", eCross, cross, 5e-3, 5e-3);
        assertAllClose("l2Norm float16", eL2, l2, 2e-3, 2e-3);
    }
}
