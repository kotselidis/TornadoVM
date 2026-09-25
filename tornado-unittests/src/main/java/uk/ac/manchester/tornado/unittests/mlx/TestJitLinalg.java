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
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
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
    public void testLuFactor() throws TornadoExecutionPlanException {
        for (int n : ORDERS) {
            double[] a = matrices(BATCH, n, 0, 301L + n);
            FloatArray fa = FloatArray.fromArray(toFloat(a));
            FloatArray luMlx = new FloatArray(a.length);
            FloatArray luJit = new FloatArray(a.length);
            IntArray pivMlx = new IntArray(BATCH * n);
            IntArray pivJit = new IntArray(BATCH * n);
            IntArray unused = new IntArray(1);
            TaskGraph g = new TaskGraph("lf").transferToDevice(DataTransferMode.FIRST_EXECUTION, fa) //
                    .libraryTask("m", MlxLinalg::luFactor, fa, luMlx, pivMlx, BATCH, n) //
                    .task("j", JitLinalg::lu, new KernelContext(), fa, luJit, pivJit, unused, n, 1, 0) //
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, luMlx, luJit, pivMlx, pivJit);
            execute(g, perMatrix("lf.j", BATCH));
            for (int i = 0; i < BATCH * n; i++) {
                assertEquals("luFactor pivots n=" + n + " " + i, pivMlx.get(i), pivJit.get(i));
            }
            double[] e = new double[a.length];
            for (int i = 0; i < e.length; i++) {
                e[i] = luMlx.get(i);
            }
            assertAllClose("luFactor n=" + n + " JIT vs MLX", e, luJit, 1e-3, 1e-4);
        }
    }

    @Test
    public void testLu() throws TornadoExecutionPlanException {
        for (int n : ORDERS) {
            double[] a = matrices(BATCH, n, 0, 311L + n);
            FloatArray fa = FloatArray.fromArray(toFloat(a));
            IntArray permMlx = new IntArray(BATCH * n);
            IntArray permJit = new IntArray(BATCH * n);
            IntArray unused = new IntArray(1);
            FloatArray lMlx = new FloatArray(a.length);
            FloatArray uMlx = new FloatArray(a.length);
            FloatArray packed = new FloatArray(a.length);
            FloatArray lJit = new FloatArray(a.length);
            FloatArray uJit = new FloatArray(a.length);
            TaskGraph g = new TaskGraph("lu").transferToDevice(DataTransferMode.FIRST_EXECUTION, fa) //
                    .libraryTask("m", MlxLinalg::lu, fa, permMlx, lMlx, uMlx, BATCH, n) //
                    .task("j", JitLinalg::lu, new KernelContext(), fa, packed, unused, permJit, n, 0, 1) //
                    .task("s", JitLinalg::splitLu, new KernelContext(), packed, lJit, uJit, a.length, n) //
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, permMlx, lMlx, uMlx, permJit, lJit, uJit);
            GridScheduler gs = new GridScheduler();
            gs.addWorkerGrid("lu.j", TestJitReduce.groups(BATCH, JitLinalg.THREADS));
            gs.addWorkerGrid("lu.s", TestJitElementwise.grid1D(a.length));
            execute(g, gs);
            for (String who : new String[] { "MLX", "JIT" }) {
                IntArray perm = who.equals("MLX") ? permMlx : permJit;
                FloatArray l = who.equals("MLX") ? lMlx : lJit;
                FloatArray u = who.equals("MLX") ? uMlx : uJit;
                double[] lv = new double[a.length];
                double[] uv = new double[a.length];
                for (int i = 0; i < a.length; i++) {
                    lv[i] = l.get(i);
                    uv[i] = u.get(i);
                }
                double[] prod = matmul(lv, uv, BATCH, n, n, n);
                double[] permuted = new double[a.length];
                for (int b = 0; b < BATCH; b++) {
                    for (int i = 0; i < n; i++) {
                        // a[i, :] = (l u)[perm[i], :]
                        int dst = perm.get(b * n + i);
                        for (int j = 0; j < n; j++) {
                            permuted[(b * n + dst) * n + j] = a[(b * n + i) * n + j];
                        }
                        for (int j = 0; j < n; j++) {
                            int e = (b * n + i) * n + j;
                            assertTrue("lu " + who + " L not unit lower", j < i || (j == i ? Math.abs(lv[e] - 1) < 1e-6 : lv[e] == 0));
                            assertTrue("lu " + who + " U not upper", j >= i || uv[e] == 0);
                        }
                    }
                }
                FloatArray pf = FloatArray.fromArray(toFloat(prod));
                assertAllClose("lu n=" + n + " " + who + " a = (l u)[perm]", permuted, pf, 1e-3, 1e-4);
            }
        }
    }

    /** Flips signs so that diag(r) >= 0 (row k of r and column k of q together). */
    private static void normaliseQr(double[] q, double[] r, int batch, int n) {
        for (int b = 0; b < batch; b++) {
            for (int k = 0; k < n; k++) {
                if (r[(b * n + k) * n + k] < 0) {
                    for (int j = 0; j < n; j++) {
                        r[(b * n + k) * n + j] = -r[(b * n + k) * n + j];
                        q[(b * n + j) * n + k] = -q[(b * n + j) * n + k];
                    }
                }
            }
        }
    }

    @Test
    public void testQr() throws TornadoExecutionPlanException {
        for (int n : ORDERS) {
            double[] a = matrices(BATCH, n, 0, 321L + n);
            FloatArray fa = FloatArray.fromArray(toFloat(a));
            FloatArray qMlx = new FloatArray(a.length);
            FloatArray rMlx = new FloatArray(a.length);
            FloatArray qJit = new FloatArray(a.length);
            FloatArray rJit = new FloatArray(a.length);
            TaskGraph g = new TaskGraph("qr").transferToDevice(DataTransferMode.FIRST_EXECUTION, fa) //
                    .libraryTask("m", MlxLinalg::qr, fa, qMlx, rMlx, BATCH, n) //
                    .task("j", JitLinalg::qr, new KernelContext(), fa, qJit, rJit, n) //
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, qMlx, rMlx, qJit, rJit);
            execute(g, perMatrix("qr.j", BATCH));
            double[][] q = new double[2][a.length];
            double[][] r = new double[2][a.length];
            for (int i = 0; i < a.length; i++) {
                q[0][i] = qMlx.get(i);
                r[0][i] = rMlx.get(i);
                q[1][i] = qJit.get(i);
                r[1][i] = rJit.get(i);
            }
            for (int w = 0; w < 2; w++) {
                String who = w == 0 ? "MLX" : "JIT";
                FloatArray recon = FloatArray.fromArray(toFloat(matmul(q[w], r[w], BATCH, n, n, n)));
                assertAllClose("qr n=" + n + " " + who + " q r = a", a, recon, 1e-3, 1e-4);
                double[] identity = new double[a.length];
                for (int b = 0; b < BATCH; b++) {
                    for (int i = 0; i < n; i++) {
                        identity[(b * n + i) * n + i] = 1;
                    }
                }
                FloatArray qtq = FloatArray.fromArray(toFloat(matmul(transpose(q[w], BATCH, n), q[w], BATCH, n, n, n)));
                assertAllClose("qr n=" + n + " " + who + " q^T q = I", identity, qtq, 1e-4, 1e-4);
                normaliseQr(q[w], r[w], BATCH, n);
            }
            assertAllClose("qr n=" + n + " R JIT vs MLX", r[0], FloatArray.fromArray(toFloat(r[1])), 1e-3, 1e-4);
            assertAllClose("qr n=" + n + " Q JIT vs MLX", q[0], FloatArray.fromArray(toFloat(q[1])), 1e-3, 1e-4);
        }
    }

    /** Symmetric: (M + M^T) / 2. */
    static double[] symmetric(int batch, int n, long seed) {
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

    /** Eigenvalues (ascending) of each symmetric matrix by cyclic Jacobi in double precision. */
    static double[] eigenvalues(double[] a, int batch, int n) {
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
            java.util.Arrays.sort(d);
            System.arraycopy(d, 0, out, b * n, n);
        }
        return out;
    }

    /** Flips each column k of {@code cols} (and row k of {@code rows}, if given) so that its largest-magnitude entry is positive. */
    static void normaliseColumns(double[] cols, double[] rows, int batch, int n) {
        for (int b = 0; b < batch; b++) {
            for (int k = 0; k < n; k++) {
                int best = 0;
                for (int i = 1; i < n; i++) {
                    if (Math.abs(cols[(b * n + i) * n + k]) > Math.abs(cols[(b * n + best) * n + k])) {
                        best = i;
                    }
                }
                if (cols[(b * n + best) * n + k] < 0) {
                    for (int i = 0; i < n; i++) {
                        cols[(b * n + i) * n + k] = -cols[(b * n + i) * n + k];
                        if (rows != null) {
                            rows[(b * n + k) * n + i] = -rows[(b * n + k) * n + i];
                        }
                    }
                }
            }
        }
    }

    static double[] doubles(FloatArray f) {
        double[] d = new double[f.getSize()];
        for (int i = 0; i < d.length; i++) {
            d[i] = f.get(i);
        }
        return d;
    }

    @Test
    public void testEigh() throws TornadoExecutionPlanException {
        for (int n : ORDERS) {
            for (boolean upper : new boolean[] { false, true }) {
                double[] a = symmetric(BATCH, n, 331L + n);
                FloatArray fa = FloatArray.fromArray(toFloat(a));
                FloatArray wMlx = new FloatArray(BATCH * n);
                FloatArray wJit = new FloatArray(BATCH * n);
                FloatArray vMlx = new FloatArray(a.length);
                FloatArray vJit = new FloatArray(a.length);
                FloatArray w2Mlx = new FloatArray(BATCH * n);
                FloatArray w2Jit = new FloatArray(BATCH * n);
                FloatArray unused = new FloatArray(1);
                int u = upper ? 1 : 0;
                TaskGraph g = new TaskGraph("eh").transferToDevice(DataTransferMode.FIRST_EXECUTION, fa) //
                        .libraryTask("m1", MlxLinalg::eigh, fa, wMlx, vMlx, BATCH, n, upper) //
                        .libraryTask("m2", MlxLinalg::eigvalsh, fa, w2Mlx, BATCH, n, upper) //
                        .task("j1", JitLinalg::eigh, new KernelContext(), fa, wJit, vJit, n, u, 1) //
                        .task("j2", JitLinalg::eigh, new KernelContext(), fa, w2Jit, unused, n, u, 0) //
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, wMlx, wJit, vMlx, vJit, w2Mlx, w2Jit);
                GridScheduler gs = new GridScheduler();
                gs.addWorkerGrid("eh.j1", TestJitReduce.groups(BATCH, JitLinalg.THREADS));
                gs.addWorkerGrid("eh.j2", TestJitReduce.groups(BATCH, JitLinalg.THREADS));
                execute(g, gs);
                double[] w = eigenvalues(a, BATCH, n);
                String what = "n=" + n + " upper=" + upper;
                both("eigh values " + what, w, wMlx, wJit, 1e-4, 2e-4);
                both("eigvalsh " + what, w, w2Mlx, w2Jit, 1e-4, 2e-4);
                double[] vm = doubles(vMlx);
                double[] vj = doubles(vJit);
                // A v = lambda v for every column.
                for (double[] vec : new double[][] { vm, vj }) {
                    double[] av = matmul(a, vec, BATCH, n, n, n);
                    double[] lv = new double[a.length];
                    for (int b = 0; b < BATCH; b++) {
                        for (int i = 0; i < n; i++) {
                            for (int k = 0; k < n; k++) {
                                lv[(b * n + i) * n + k] = w[b * n + k] * vec[(b * n + i) * n + k];
                            }
                        }
                    }
                    assertAllClose("eigh A v = w v " + what + (vec == vm ? " MLX" : " JIT"), lv, FloatArray.fromArray(toFloat(av)), 1e-3, 1e-3);
                }
                normaliseColumns(vm, null, BATCH, n);
                normaliseColumns(vj, null, BATCH, n);
                assertAllClose("eigh vectors JIT vs MLX " + what, vm, FloatArray.fromArray(toFloat(vj)), 1e-2, 2e-3);
            }
        }
    }

    @Test
    public void testSvdAndPinv() throws TornadoExecutionPlanException {
        for (int n : ORDERS) {
            double[] a = matrices(BATCH, n, 0, 341L + n);
            FloatArray fa = FloatArray.fromArray(toFloat(a));
            FloatArray uMlx = new FloatArray(a.length);
            FloatArray sMlx = new FloatArray(BATCH * n);
            FloatArray vtMlx = new FloatArray(a.length);
            FloatArray uJit = new FloatArray(a.length);
            FloatArray sJit = new FloatArray(BATCH * n);
            FloatArray vtJit = new FloatArray(a.length);
            FloatArray s2Mlx = new FloatArray(BATCH * n);
            FloatArray s2Jit = new FloatArray(BATCH * n);
            FloatArray pMlx = new FloatArray(a.length);
            FloatArray pJit = new FloatArray(a.length);
            // Distinct dummies: the same array passed twice as an output hangs the TornadoVM graph scheduler.
            FloatArray unusedU = new FloatArray(1);
            FloatArray unusedVt = new FloatArray(1);
            TaskGraph g = new TaskGraph("sv").transferToDevice(DataTransferMode.FIRST_EXECUTION, fa) //
                    .libraryTask("m1", MlxLinalg::svd, fa, uMlx, sMlx, vtMlx, BATCH, n) //
                    .libraryTask("m2", MlxLinalg::singularValues, fa, s2Mlx, BATCH, n) //
                    .libraryTask("m3", MlxLinalg::pinv, fa, pMlx, BATCH, n) //
                    .task("j1", JitLinalg::svd, new KernelContext(), fa, uJit, sJit, vtJit, n, 1) //
                    .task("j2", JitLinalg::svd, new KernelContext(), fa, unusedU, s2Jit, unusedVt, n, 0) //
                    .task("j3", JitLinalg::pinvFromSvd, new KernelContext(), uJit, sJit, vtJit, pJit, a.length, n, 1e-6f) //
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, uMlx, sMlx, vtMlx, uJit, sJit, vtJit, s2Mlx, s2Jit, pMlx, pJit);
            GridScheduler gs = new GridScheduler();
            gs.addWorkerGrid("sv.j1", TestJitReduce.groups(BATCH, JitLinalg.THREADS));
            gs.addWorkerGrid("sv.j2", TestJitReduce.groups(BATCH, JitLinalg.THREADS));
            gs.addWorkerGrid("sv.j3", TestJitElementwise.grid1D(a.length));
            execute(g, gs);
            // Singular values: square roots of the eigenvalues of A^T A, descending.
            double[] ev = eigenvalues(matmul(transpose(a, BATCH, n), a, BATCH, n, n, n), BATCH, n);
            double[] sv = new double[BATCH * n];
            for (int b = 0; b < BATCH; b++) {
                for (int k = 0; k < n; k++) {
                    sv[b * n + k] = Math.sqrt(Math.max(ev[b * n + n - 1 - k], 0));
                }
            }
            String what = "n=" + n;
            both("svd s " + what, sv, sMlx, sJit, 1e-3, 1e-4);
            both("singularValues " + what, sv, s2Mlx, s2Jit, 1e-3, 1e-4);
            both("pinv " + what, inverse(a, BATCH, n), pMlx, pJit, 5e-3, 5e-4);
            for (int w = 0; w < 2; w++) {
                double[] u = doubles(w == 0 ? uMlx : uJit);
                double[] s2 = doubles(w == 0 ? sMlx : sJit);
                double[] vt = doubles(w == 0 ? vtMlx : vtJit);
                double[] us = new double[a.length];
                for (int b = 0; b < BATCH; b++) {
                    for (int i = 0; i < n; i++) {
                        for (int k = 0; k < n; k++) {
                            us[(b * n + i) * n + k] = u[(b * n + i) * n + k] * s2[b * n + k];
                        }
                    }
                }
                assertAllClose("svd u s vt = a " + what + (w == 0 ? " MLX" : " JIT"), a, FloatArray.fromArray(toFloat(matmul(us, vt, BATCH, n, n, n))), 1e-3, 1e-4);
            }
            double[] um = doubles(uMlx);
            double[] vtm = doubles(vtMlx);
            double[] uj = doubles(uJit);
            double[] vtj = doubles(vtJit);
            normaliseColumns(um, vtm, BATCH, n);
            normaliseColumns(uj, vtj, BATCH, n);
            assertAllClose("svd u JIT vs MLX " + what, um, FloatArray.fromArray(toFloat(uj)), 1e-2, 2e-3);
            assertAllClose("svd vt JIT vs MLX " + what, vtm, FloatArray.fromArray(toFloat(vtj)), 1e-2, 2e-3);
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
