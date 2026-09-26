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
import uk.ac.manchester.tornado.mlx.MlxCreate;
import uk.ac.manchester.tornado.mlx.jit.JitCreate;

/**
 * The MLX construction and matrix-structure operations and their KernelContext JIT counterparts,
 * run in one graph and checked against Java.
 *
 * <p>
 * How to run?
 * </p>
 * <code>
 * tornado-test -V uk.ac.manchester.tornado.unittests.mlx.TestJitCreate
 * </code>
 */
public class TestJitCreate extends MlxTestBase {

    private static void execute(TaskGraph g, int threads, String... jitTasks) throws TornadoExecutionPlanException {
        GridScheduler gs = new GridScheduler();
        for (String t : jitTasks) {
            gs.addWorkerGrid(g.getTaskGraphName() + "." + t, TestJitElementwise.grid1D(threads));
        }
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(g.snapshot())) {
            plan.withGridScheduler(gs).execute();
        }
    }

    private static void both(String what, double[] expected, FloatArray mlx, FloatArray jit, double tol) {
        assertAllClose(what + " MLX", expected, mlx, tol, tol);
        assertAllClose(what + " JIT", expected, jit, tol, tol);
    }

    @Test
    public void testRanges() throws TornadoExecutionPlanException {
        final int n = 1000;
        FloatArray arM = new FloatArray(n);
        FloatArray arJ = new FloatArray(n);
        FloatArray lsM = new FloatArray(n);
        FloatArray lsJ = new FloatArray(n);
        TaskGraph g = new TaskGraph("rg").transferToDevice(DataTransferMode.FIRST_EXECUTION, arM, lsM) //
                .libraryTask("m1", MlxCreate::arange, arM, -2.0f, -2.0f + 0.25f * n, 0.25f) //
                .libraryTask("m2", MlxCreate::linspace, lsM, -1.0f, 3.0f) //
                .task("j1", JitCreate::arange, new KernelContext(), arJ, n, -2.0f, 0.25f) //
                .task("j2", JitCreate::linspace, new KernelContext(), lsJ, n, -1.0f, 3.0f) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, arM, arJ, lsM, lsJ);
        execute(g, n, "j1", "j2");
        double[] ar = new double[n];
        double[] ls = new double[n];
        for (int i = 0; i < n; i++) {
            ar[i] = -2.0 + 0.25 * i;
            ls[i] = -1.0 + 4.0 * i / (n - 1);
        }
        both("arange", ar, arM, arJ, 1e-5);
        both("linspace", ls, lsM, lsJ, 1e-5);
    }

    @Test
    public void testConstants() throws TornadoExecutionPlanException {
        final int n = 777;
        FloatArray like = new FloatArray(n);
        FloatArray[] m = new FloatArray[6];
        FloatArray[] j = new FloatArray[6];
        for (int k = 0; k < 6; k++) {
            m[k] = new FloatArray(n);
            j[k] = new FloatArray(n);
        }
        TaskGraph g = new TaskGraph("ct").transferToDevice(DataTransferMode.FIRST_EXECUTION, like, m[0], m[2], m[4]) //
                .libraryTask("m1", MlxCreate::full, m[0], 2.5f) //
                .libraryTask("m2", MlxCreate::fullLike, like, m[1], -1.5f) //
                .libraryTask("m3", MlxCreate::zeros, m[2]) //
                .libraryTask("m4", MlxCreate::zerosLike, like, m[3]) //
                .libraryTask("m5", MlxCreate::ones, m[4]) //
                .libraryTask("m6", MlxCreate::onesLike, like, m[5]) //
                .task("j1", JitCreate::fill, new KernelContext(), j[0], n, 2.5f) //
                .task("j2", JitCreate::fill, new KernelContext(), j[1], n, -1.5f) //
                .task("j3", JitCreate::fill, new KernelContext(), j[2], n, 0.0f) //
                .task("j4", JitCreate::fill, new KernelContext(), j[3], n, 0.0f) //
                .task("j5", JitCreate::fill, new KernelContext(), j[4], n, 1.0f) //
                .task("j6", JitCreate::fill, new KernelContext(), j[5], n, 1.0f) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, m[0], m[1], m[2], m[3], m[4], m[5], j[0], j[1], j[2], j[3], j[4], j[5]);
        execute(g, n, "j1", "j2", "j3", "j4", "j5", "j6");
        float[] values = { 2.5f, -1.5f, 0f, 0f, 1f, 1f };
        String[] names = { "full", "fullLike", "zeros", "zerosLike", "ones", "onesLike" };
        for (int k = 0; k < 6; k++) {
            double[] e = new double[n];
            java.util.Arrays.fill(e, values[k]);
            both(names[k], e, m[k], j[k], 0);
        }
    }

    @Test
    public void testIdentityShapes() throws TornadoExecutionPlanException {
        final int n = 37;
        final int mm = 53;
        FloatArray eyeM = new FloatArray(n * mm);
        FloatArray eyeJ = new FloatArray(n * mm);
        FloatArray idM = new FloatArray(n * n);
        FloatArray idJ = new FloatArray(n * n);
        FloatArray triM = new FloatArray(n * mm);
        FloatArray triJ = new FloatArray(n * mm);
        TaskGraph g = new TaskGraph("id").transferToDevice(DataTransferMode.FIRST_EXECUTION, eyeM, idM, triM) //
                .libraryTask("m1", MlxCreate::eye, eyeM, n, mm, 3) //
                .libraryTask("m2", MlxCreate::identity, idM, n) //
                .libraryTask("m3", MlxCreate::tri, triM, n, mm, -2) //
                .task("j1", JitCreate::eye, new KernelContext(), eyeJ, n, mm, 3, 0) //
                .task("j2", JitCreate::eye, new KernelContext(), idJ, n, n, 0, 0) //
                .task("j3", JitCreate::eye, new KernelContext(), triJ, n, mm, -2, 1) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, eyeM, eyeJ, idM, idJ, triM, triJ);
        execute(g, n * mm, "j1", "j2", "j3");
        double[] eEye = new double[n * mm];
        double[] eId = new double[n * n];
        double[] eTri = new double[n * mm];
        for (int r = 0; r < n; r++) {
            for (int c = 0; c < mm; c++) {
                eEye[r * mm + c] = c - r == 3 ? 1 : 0;
                eTri[r * mm + c] = c - r <= -2 ? 1 : 0;
            }
            for (int c = 0; c < n; c++) {
                eId[r * n + c] = r == c ? 1 : 0;
            }
        }
        both("eye", eEye, eyeM, eyeJ, 0);
        both("identity", eId, idM, idJ, 0);
        both("tri", eTri, triM, triJ, 0);
    }

    @Test
    public void testWindows() throws TornadoExecutionPlanException {
        for (int m : new int[] { 1, 64, 257 }) {
            FloatArray[] mw = { new FloatArray(m), new FloatArray(m), new FloatArray(m), new FloatArray(m) };
            FloatArray[] jw = { new FloatArray(m), new FloatArray(m), new FloatArray(m), new FloatArray(m) };
            TaskGraph g = new TaskGraph("wn").transferToDevice(DataTransferMode.FIRST_EXECUTION, mw[0], mw[1], mw[2], mw[3]) //
                    .libraryTask("m1", MlxCreate::bartlett, mw[0]) //
                    .libraryTask("m2", MlxCreate::blackman, mw[1]) //
                    .libraryTask("m3", MlxCreate::hamming, mw[2]) //
                    .libraryTask("m4", MlxCreate::hanning, mw[3]) //
                    .task("j1", JitCreate::window, new KernelContext(), jw[0], m, JitCreate.BARTLETT) //
                    .task("j2", JitCreate::window, new KernelContext(), jw[1], m, JitCreate.BLACKMAN) //
                    .task("j3", JitCreate::window, new KernelContext(), jw[2], m, JitCreate.HAMMING) //
                    .task("j4", JitCreate::window, new KernelContext(), jw[3], m, JitCreate.HANNING) //
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, mw[0], mw[1], mw[2], mw[3], jw[0], jw[1], jw[2], jw[3]);
            execute(g, m, "j1", "j2", "j3", "j4");
            String[] names = { "bartlett", "blackman", "hamming", "hanning" };
            for (int k = 0; k < 4; k++) {
                double[] e = new double[m];
                for (int i = 0; i < m; i++) {
                    double x = m == 1 ? 0 : (double) i / (m - 1);
                    e[i] = m == 1 ? 1 : switch (k) {
                        case 0 -> 1 - Math.abs(2 * x - 1);
                        case 1 -> 0.42 - 0.5 * Math.cos(2 * Math.PI * x) + 0.08 * Math.cos(4 * Math.PI * x);
                        case 2 -> 0.54 - 0.46 * Math.cos(2 * Math.PI * x);
                        default -> 0.5 - 0.5 * Math.cos(2 * Math.PI * x);
                    };
                }
                both(names[k] + " m=" + m, e, mw[k], jw[k], 1e-5);
            }
        }
    }

    @Test
    public void testMeshgrid() throws TornadoExecutionPlanException {
        final int nx = 13;
        final int ny = 7;
        float[] xv = values(nx, -1, 1, 11);
        float[] yv = values(ny, -1, 1, 12);
        for (boolean ij : new boolean[] { false, true }) {
            FloatArray x = FloatArray.fromArray(xv);
            FloatArray y = FloatArray.fromArray(yv);
            FloatArray gxM = new FloatArray(nx * ny);
            FloatArray gyM = new FloatArray(nx * ny);
            FloatArray gxJ = new FloatArray(nx * ny);
            FloatArray gyJ = new FloatArray(nx * ny);
            TaskGraph g = new TaskGraph("mg").transferToDevice(DataTransferMode.FIRST_EXECUTION, x, y) //
                    .libraryTask("m", MlxCreate::meshgrid, x, y, gxM, gyM, ij) //
                    .task("j", JitCreate::meshgrid, new KernelContext(), x, y, gxJ, gyJ, nx, ny, ij ? 1 : 0) //
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, gxM, gyM, gxJ, gyJ);
            execute(g, nx * ny, "j");
            double[] ex = new double[nx * ny];
            double[] ey = new double[nx * ny];
            for (int t = 0; t < nx * ny; t++) {
                ex[t] = ij ? xv[t / ny] : xv[t % nx];
                ey[t] = ij ? yv[t % ny] : yv[t / nx];
            }
            both("meshgrid x ij=" + ij, ex, gxM, gxJ, 0);
            both("meshgrid y ij=" + ij, ey, gyM, gyJ, 0);
        }
    }

    @Test
    public void testMatrixStructure() throws TornadoExecutionPlanException {
        final int rows = 23;
        final int cols = 31;
        float[] av = values(rows * cols, -5, 5, 13);
        FloatArray a = FloatArray.fromArray(av);
        for (int k : new int[] { 0, 4, -3 }) {
            int len = Math.min(rows + Math.min(k, 0), cols - Math.max(k, 0));
            float[] vv = values(9, -2, 2, 14);
            FloatArray v = FloatArray.fromArray(vv);
            int s = 9 + Math.abs(k);
            FloatArray dgM = new FloatArray(s * s);
            FloatArray dgJ = new FloatArray(s * s);
            FloatArray dnM = new FloatArray(len);
            FloatArray dnJ = new FloatArray(len);
            FloatArray trM = new FloatArray(1);
            FloatArray trJ = new FloatArray(1);
            FloatArray loM = new FloatArray(rows * cols);
            FloatArray loJ = new FloatArray(rows * cols);
            FloatArray upM = new FloatArray(rows * cols);
            FloatArray upJ = new FloatArray(rows * cols);
            TaskGraph g = new TaskGraph("ms").transferToDevice(DataTransferMode.FIRST_EXECUTION, a, v) //
                    .libraryTask("m1", MlxCreate::diag, v, dgM, k) //
                    .libraryTask("m2", MlxCreate::diagonal, a, dnM, rows, cols, k) //
                    .libraryTask("m3", MlxCreate::trace, a, trM, rows, cols, k) //
                    .libraryTask("m4", MlxCreate::tril, a, loM, rows, cols, k) //
                    .libraryTask("m5", MlxCreate::triu, a, upM, rows, cols, k) //
                    .task("j1", JitCreate::diag, new KernelContext(), v, dgJ, 9, k) //
                    .task("j2", JitCreate::diagonal, new KernelContext(), a, dnJ, cols, k, len) //
                    .task("j3", JitCreate::trace, new KernelContext(), a, trJ, cols, k, len) //
                    .task("j4", JitCreate::triangle, new KernelContext(), a, loJ, rows, cols, k, 0) //
                    .task("j5", JitCreate::triangle, new KernelContext(), a, upJ, rows, cols, k, 1) //
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, dgM, dgJ, dnM, dnJ, trM, trJ, loM, loJ, upM, upJ);
            GridScheduler gs = new GridScheduler();
            gs.addWorkerGrid("ms.j1", TestJitElementwise.grid1D(s * s));
            gs.addWorkerGrid("ms.j2", TestJitElementwise.grid1D(len));
            gs.addWorkerGrid("ms.j3", TestJitReduce.groups(1, 256));
            gs.addWorkerGrid("ms.j4", TestJitElementwise.grid1D(rows * cols));
            gs.addWorkerGrid("ms.j5", TestJitElementwise.grid1D(rows * cols));
            try (TornadoExecutionPlan plan = new TornadoExecutionPlan(g.snapshot())) {
                plan.withGridScheduler(gs).execute();
            }
            double[] eDg = new double[s * s];
            for (int i = 0; i < 9; i++) {
                int r = k >= 0 ? i : i - k;
                int c = k >= 0 ? i + k : i;
                eDg[r * s + c] = vv[i];
            }
            double[] eDn = new double[len];
            double tr = 0;
            for (int t = 0; t < len; t++) {
                eDn[t] = av[(t + Math.max(-k, 0)) * cols + t + Math.max(k, 0)];
                tr += eDn[t];
            }
            double[] eLo = new double[rows * cols];
            double[] eUp = new double[rows * cols];
            for (int t = 0; t < rows * cols; t++) {
                int d = t % cols - t / cols;
                eLo[t] = d <= k ? av[t] : 0;
                eUp[t] = d >= k ? av[t] : 0;
            }
            String what = " k=" + k;
            both("diag" + what, eDg, dgM, dgJ, 0);
            both("diagonal" + what, eDn, dnM, dnJ, 0);
            both("trace" + what, new double[] { tr }, trM, trJ, 1e-4);
            both("tril" + what, eLo, loM, loJ, 0);
            both("triu" + what, eUp, upM, upJ, 0);
        }
    }

    @Test
    public void testIntAndHalfForms() throws TornadoExecutionPlanException {
        final int n = 100;
        IntArray ar = new IntArray(n);
        IntArray eye = new IntArray(n * n);
        HalfFloatArray ones = new HalfFloatArray(n);
        run(new TaskGraph("ih").transferToDevice(DataTransferMode.FIRST_EXECUTION, ar, eye, ones) //
                .libraryTask("a", MlxCreate::arange, ar, 5f, 5f + 3f * n, 3f) //
                .libraryTask("e", MlxCreate::identity, eye, n) //
                .libraryTask("o", MlxCreate::ones, ones) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, ar, eye, ones));
        for (int i = 0; i < n; i++) {
            assertEquals("arange int32 " + i, 5 + 3 * i, ar.get(i));
            assertEquals("ones float16 " + i, 1.0f, ones.get(i).getFloat32(), 0f);
            for (int c = 0; c < n; c++) {
                assertEquals("identity int32", i == c ? 1 : 0, eye.get(i * n + c));
            }
        }
    }
}
