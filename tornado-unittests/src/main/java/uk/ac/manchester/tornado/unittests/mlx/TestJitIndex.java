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

import java.util.Random;

import org.junit.Test;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.mlx.MlxIndex;
import uk.ac.manchester.tornado.mlx.jit.JitIndex;

/**
 * The MLX indexing operations and their KernelContext JIT counterparts, run in one graph on the
 * same inputs and checked against Java references. Set, max, min and prod scatters use distinct
 * target positions (the JIT kernels have no float atomics for them); additive scatters repeat
 * positions.
 *
 * <p>
 * How to run?
 * </p>
 * <code>
 * tornado-test -V uk.ac.manchester.tornado.unittests.mlx.TestJitIndex
 * </code>
 */
public class TestJitIndex extends MlxTestBase {

    private static final int ROWS = 40;
    private static final int COLS = 24;

    private static int[] randomInts(int n, int bound, long seed) {
        Random r = new Random(seed);
        int[] v = new int[n];
        for (int i = 0; i < n; i++) {
            v[i] = r.nextInt(bound);
        }
        return v;
    }

    /** The first n of a random permutation of [0, bound). */
    private static int[] distinct(int n, int bound, long seed) {
        int[] p = new int[bound];
        for (int i = 0; i < bound; i++) {
            p[i] = i;
        }
        Random r = new Random(seed);
        for (int i = bound - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            int t = p[i];
            p[i] = p[j];
            p[j] = t;
        }
        int[] out = new int[n];
        System.arraycopy(p, 0, out, 0, n);
        return out;
    }

    private static void execute(TaskGraph g, GridScheduler gs) throws TornadoExecutionPlanException {
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(g.snapshot())) {
            plan.withGridScheduler(gs).execute();
        }
    }

    private static void assertExact(String what, float[] expected, FloatArray mlx, FloatArray jit, double relTol) {
        for (int i = 0; i < expected.length; i++) {
            assertClose(what + " JIT", i, expected[i], jit.get(i), relTol, relTol);
            assertClose(what + " MLX", i, expected[i], mlx.get(i), relTol, relTol);
        }
    }

    private static float applyRef(float old, float u, int op) {
        return switch (op) {
            case JitIndex.SET -> u;
            case JitIndex.ADD -> old + u;
            case JitIndex.MAX -> Math.max(old, u);
            case JitIndex.MIN -> Math.min(old, u);
            default -> old * u;
        };
    }

    @Test
    public void testTakeFamily() throws TornadoExecutionPlanException {
        final int outer = 3;
        final int len = 40;
        final int inner = 5;
        final int count = 17;
        final int m = 6;
        float[] xv = values(outer * len * inner, -10, 10, 121);
        FloatArray x = FloatArray.fromArray(xv);
        int[] flatIdx = randomInts(100, xv.length, 122);
        int[] axisIdx = randomInts(count, len, 123);
        int[] alongIdx = randomInts(outer * m * inner, len, 124);
        IntArray fi = IntArray.fromArray(flatIdx);
        IntArray ai = IntArray.fromArray(axisIdx);
        IntArray li = IntArray.fromArray(alongIdx);
        FloatArray takeMlx = new FloatArray(100);
        FloatArray takeJit = new FloatArray(100);
        FloatArray axisMlx = new FloatArray(outer * count * inner);
        FloatArray axisJit = new FloatArray(outer * count * inner);
        FloatArray alongMlx = new FloatArray(outer * m * inner);
        FloatArray alongJit = new FloatArray(outer * m * inner);
        TaskGraph g = new TaskGraph("tk").transferToDevice(DataTransferMode.FIRST_EXECUTION, x, fi, ai, li) //
                .libraryTask("m1", MlxIndex::take, x, fi, takeMlx) //
                .libraryTask("m2", MlxIndex::takeAxis, x, ai, axisMlx, outer, len, inner) //
                .libraryTask("m3", MlxIndex::takeAlongAxis, x, li, alongMlx, outer, len, m, inner) //
                .task("j1", JitIndex::take, new KernelContext(), x, fi, takeJit, 100) //
                .task("j2", JitIndex::takeAxis, new KernelContext(), x, ai, axisJit, outer, len, count, inner) //
                .task("j3", JitIndex::takeAlongAxis, new KernelContext(), x, li, alongJit, outer, len, m, inner) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, takeMlx, takeJit, axisMlx, axisJit, alongMlx, alongJit);
        GridScheduler gs = new GridScheduler();
        gs.addWorkerGrid("tk.j1", TestJitElementwise.grid1D(100));
        gs.addWorkerGrid("tk.j2", TestJitElementwise.grid1D(outer * count * inner));
        gs.addWorkerGrid("tk.j3", TestJitElementwise.grid1D(outer * m * inner));
        execute(g, gs);
        float[] eTake = new float[100];
        for (int i = 0; i < 100; i++) {
            eTake[i] = xv[flatIdx[i]];
        }
        float[] eAxis = new float[outer * count * inner];
        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < count; i++) {
                for (int j = 0; j < inner; j++) {
                    eAxis[(o * count + i) * inner + j] = xv[(o * len + axisIdx[i]) * inner + j];
                }
            }
        }
        float[] eAlong = new float[outer * m * inner];
        for (int t = 0; t < eAlong.length; t++) {
            int o = t / (m * inner);
            int j = t % inner;
            eAlong[t] = xv[(o * len + alongIdx[t]) * inner + j];
        }
        assertExact("take", eTake, takeMlx, takeJit, 0);
        assertExact("takeAxis", eAxis, axisMlx, axisJit, 0);
        assertExact("takeAlongAxis", eAlong, alongMlx, alongJit, 0);
    }

    @Test
    public void testPutAndScatterAddAlongAxis() throws TornadoExecutionPlanException {
        final int outer = 3;
        final int len = 40;
        final int inner = 5;
        final int m = 6;
        for (int op : new int[] { JitIndex.SET, JitIndex.ADD }) {
            float[] xv = values(outer * len * inner, -10, 10, 131);
            float[] vv = values(outer * m * inner, -10, 10, 132);
            int[] idx = new int[outer * m * inner];
            for (int o = 0; o < outer; o++) {
                for (int j = 0; j < inner; j++) {
                    int[] rows = op == JitIndex.SET ? distinct(m, len, 133L + o * inner + j) : randomInts(m, 4, 133L + o * inner + j);
                    for (int i = 0; i < m; i++) {
                        idx[(o * m + i) * inner + j] = rows[i];
                    }
                }
            }
            FloatArray x = FloatArray.fromArray(xv);
            FloatArray values = FloatArray.fromArray(vv);
            IntArray indices = IntArray.fromArray(idx);
            FloatArray outMlx = new FloatArray(xv.length);
            FloatArray outJit = new FloatArray(xv.length);
            TaskGraph g = new TaskGraph("pa").transferToDevice(DataTransferMode.FIRST_EXECUTION, x, values, indices);
            if (op == JitIndex.SET) {
                g.libraryTask("m", MlxIndex::putAlongAxis, x, indices, values, outMlx, outer, len, m, inner);
            } else {
                g.libraryTask("m", MlxIndex::scatterAddAxis, x, indices, values, outMlx, outer, len, m, inner);
            }
            g.task("c", JitIndex::copy, new KernelContext(), x, outJit, xv.length) //
                    .task("s", JitIndex::putAlongAxis, new KernelContext(), indices, values, outJit, outer, len, m, inner, op) //
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, outMlx, outJit);
            GridScheduler gs = new GridScheduler();
            gs.addWorkerGrid("pa.c", TestJitElementwise.grid1D(xv.length));
            gs.addWorkerGrid("pa.s", TestJitElementwise.grid1D(idx.length));
            execute(g, gs);
            float[] expected = xv.clone();
            for (int t = 0; t < idx.length; t++) {
                int o = t / (m * inner);
                int j = t % inner;
                int p = (o * len + idx[t]) * inner + j;
                expected[p] = applyRef(expected[p], vv[t], op);
            }
            assertExact(op == JitIndex.SET ? "putAlongAxis" : "scatterAddAxis", expected, outMlx, outJit, 1e-5);
        }
    }

    @Test
    public void testGather() throws TornadoExecutionPlanException {
        final int count = 50;
        final int windows = 9;
        final int sliceRows = 3;
        float[] xv = values(ROWS * COLS, -10, 10, 141);
        FloatArray x = FloatArray.fromArray(xv);
        int[] rows = randomInts(count, ROWS, 142);
        int[] cols = randomInts(count, COLS, 143);
        int[] starts = randomInts(windows, ROWS - sliceRows + 1, 144);
        IntArray ri = IntArray.fromArray(rows);
        IntArray ci = IntArray.fromArray(cols);
        IntArray si = IntArray.fromArray(starts);
        FloatArray ptMlx = new FloatArray(count);
        FloatArray ptJit = new FloatArray(count);
        FloatArray rwMlx = new FloatArray(windows * sliceRows * COLS);
        FloatArray rwJit = new FloatArray(windows * sliceRows * COLS);
        TaskGraph g = new TaskGraph("ga").transferToDevice(DataTransferMode.FIRST_EXECUTION, x, ri, ci, si) //
                .libraryTask("m1", MlxIndex::gather, x, ri, ci, ptMlx, ROWS, COLS) //
                .libraryTask("m2", MlxIndex::gatherRows, x, si, rwMlx, ROWS, COLS, sliceRows) //
                .task("j1", JitIndex::gatherPoints, new KernelContext(), x, ri, ci, ptJit, COLS, count) //
                .task("j2", JitIndex::gatherRows, new KernelContext(), x, si, rwJit, COLS, windows, sliceRows) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, ptMlx, ptJit, rwMlx, rwJit);
        GridScheduler gs = new GridScheduler();
        gs.addWorkerGrid("ga.j1", TestJitElementwise.grid1D(count));
        gs.addWorkerGrid("ga.j2", TestJitElementwise.grid1D(windows * sliceRows * COLS));
        execute(g, gs);
        float[] ePt = new float[count];
        for (int i = 0; i < count; i++) {
            ePt[i] = xv[rows[i] * COLS + cols[i]];
        }
        float[] eRw = new float[windows * sliceRows * COLS];
        for (int t = 0; t < eRw.length; t++) {
            int i = t / (sliceRows * COLS);
            int r = (t / COLS) % sliceRows;
            eRw[t] = xv[(starts[i] + r) * COLS + t % COLS];
        }
        assertExact("gather", ePt, ptMlx, ptJit, 0);
        assertExact("gatherRows", eRw, rwMlx, rwJit, 0);
    }

    @Test
    public void testScatterPoints() throws TornadoExecutionPlanException {
        final int count = 60;
        for (int op : new int[] { JitIndex.SET, JitIndex.ADD, JitIndex.MAX, JitIndex.MIN, JitIndex.PROD }) {
            float[] xv = values(ROWS * COLS, 1, 2, 151);
            float[] uv = values(count, op == JitIndex.PROD ? 0.5f : -3, op == JitIndex.PROD ? 1.5f : 3, 152);
            int[] rows;
            int[] cols;
            if (op == JitIndex.ADD) {
                rows = randomInts(count, 5, 153);
                cols = randomInts(count, 4, 154);
            } else {
                int[] flat = distinct(count, ROWS * COLS, 155);
                rows = new int[count];
                cols = new int[count];
                for (int i = 0; i < count; i++) {
                    rows[i] = flat[i] / COLS;
                    cols[i] = flat[i] % COLS;
                }
            }
            FloatArray x = FloatArray.fromArray(xv);
            FloatArray u = FloatArray.fromArray(uv);
            IntArray ri = IntArray.fromArray(rows);
            IntArray ci = IntArray.fromArray(cols);
            FloatArray outMlx = new FloatArray(xv.length);
            FloatArray outJit = new FloatArray(xv.length);
            TaskGraph g = new TaskGraph("sp").transferToDevice(DataTransferMode.FIRST_EXECUTION, x, u, ri, ci);
            switch (op) {
                case JitIndex.SET -> g.libraryTask("m", MlxIndex::scatter, x, ri, ci, u, outMlx, ROWS, COLS);
                case JitIndex.ADD -> g.libraryTask("m", MlxIndex::scatterAdd, x, ri, ci, u, outMlx, ROWS, COLS);
                case JitIndex.MAX -> g.libraryTask("m", MlxIndex::scatterMax, x, ri, ci, u, outMlx, ROWS, COLS);
                case JitIndex.MIN -> g.libraryTask("m", MlxIndex::scatterMin, x, ri, ci, u, outMlx, ROWS, COLS);
                default -> g.libraryTask("m", MlxIndex::scatterProd, x, ri, ci, u, outMlx, ROWS, COLS);
            }
            g.task("c", JitIndex::copy, new KernelContext(), x, outJit, xv.length) //
                    .task("s", JitIndex::scatterPoints, new KernelContext(), ri, ci, u, outJit, COLS, count, op) //
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, outMlx, outJit);
            GridScheduler gs = new GridScheduler();
            gs.addWorkerGrid("sp.c", TestJitElementwise.grid1D(xv.length));
            gs.addWorkerGrid("sp.s", TestJitElementwise.grid1D(count));
            execute(g, gs);
            float[] expected = xv.clone();
            for (int i = 0; i < count; i++) {
                int p = rows[i] * COLS + cols[i];
                expected[p] = applyRef(expected[p], uv[i], op);
            }
            assertExact("scatter op=" + op, expected, outMlx, outJit, 1e-5);
        }
    }

    @Test
    public void testScatterRows() throws TornadoExecutionPlanException {
        final int count = 10;
        for (int op : new int[] { JitIndex.SET, JitIndex.ADD, JitIndex.MAX, JitIndex.MIN, JitIndex.PROD }) {
            float[] xv = values(ROWS * COLS, 1, 2, 161);
            float[] uv = values(count * COLS, op == JitIndex.PROD ? 0.5f : -3, op == JitIndex.PROD ? 1.5f : 3, 162);
            int[] idx = op == JitIndex.ADD ? randomInts(count, 4, 163) : distinct(count, ROWS, 164);
            FloatArray x = FloatArray.fromArray(xv);
            FloatArray u = FloatArray.fromArray(uv);
            IntArray ii = IntArray.fromArray(idx);
            FloatArray outMlx = new FloatArray(xv.length);
            FloatArray outJit = new FloatArray(xv.length);
            TaskGraph g = new TaskGraph("sr").transferToDevice(DataTransferMode.FIRST_EXECUTION, x, u, ii);
            switch (op) {
                case JitIndex.SET -> g.libraryTask("m", MlxIndex::scatterRows, x, ii, u, outMlx, ROWS, COLS);
                case JitIndex.ADD -> g.libraryTask("m", MlxIndex::scatterAddRows, x, ii, u, outMlx, ROWS, COLS);
                case JitIndex.MAX -> g.libraryTask("m", MlxIndex::scatterMaxRows, x, ii, u, outMlx, ROWS, COLS);
                case JitIndex.MIN -> g.libraryTask("m", MlxIndex::scatterMinRows, x, ii, u, outMlx, ROWS, COLS);
                default -> g.libraryTask("m", MlxIndex::scatterProdRows, x, ii, u, outMlx, ROWS, COLS);
            }
            g.task("c", JitIndex::copy, new KernelContext(), x, outJit, xv.length) //
                    .task("s", JitIndex::scatterRows, new KernelContext(), ii, u, outJit, COLS, count, op) //
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, outMlx, outJit);
            GridScheduler gs = new GridScheduler();
            gs.addWorkerGrid("sr.c", TestJitElementwise.grid1D(xv.length));
            gs.addWorkerGrid("sr.s", TestJitElementwise.grid1D(count * COLS));
            execute(g, gs);
            float[] expected = xv.clone();
            for (int t = 0; t < count * COLS; t++) {
                int p = idx[t / COLS] * COLS + t % COLS;
                expected[p] = applyRef(expected[p], uv[t], op);
            }
            assertExact("scatterRows op=" + op, expected, outMlx, outJit, 1e-5);
        }
    }

    @Test
    public void testSlices() throws TornadoExecutionPlanException {
        final int r0 = 3;
        final int r1 = 37;
        final int rs = 2;
        final int c0 = 1;
        final int c1 = 23;
        final int cs = 3;
        final int outRows = (r1 - r0 + rs - 1) / rs;
        final int outCols = (c1 - c0 + cs - 1) / cs;
        final int sliceRows = 7;
        float[] xv = values(ROWS * COLS, -10, 10, 171);
        FloatArray x = FloatArray.fromArray(xv);
        IntArray start = IntArray.fromElements(11);
        FloatArray slMlx = new FloatArray(outRows * outCols);
        FloatArray slJit = new FloatArray(outRows * outCols);
        FloatArray dyMlx = new FloatArray(sliceRows * COLS);
        FloatArray dyJit = new FloatArray(sliceRows * COLS);
        TaskGraph g = new TaskGraph("sl").transferToDevice(DataTransferMode.FIRST_EXECUTION, x, start) //
                .libraryTask("m1", MlxIndex::slice, x, slMlx, ROWS, COLS, r0, r1, rs, c0, c1, cs) //
                .libraryTask("m2", MlxIndex::sliceRowsAt, x, start, dyMlx, ROWS, COLS, sliceRows) //
                .task("j1", JitIndex::slice, new KernelContext(), x, slJit, COLS, r0, rs, c0, cs, outRows, outCols) //
                .task("j2", JitIndex::sliceRowsAt, new KernelContext(), x, start, dyJit, COLS, sliceRows) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, slMlx, slJit, dyMlx, dyJit);
        GridScheduler gs = new GridScheduler();
        gs.addWorkerGrid("sl.j1", TestJitElementwise.grid1D(outRows * outCols));
        gs.addWorkerGrid("sl.j2", TestJitElementwise.grid1D(sliceRows * COLS));
        execute(g, gs);
        float[] eSl = new float[outRows * outCols];
        for (int r = 0; r < outRows; r++) {
            for (int c = 0; c < outCols; c++) {
                eSl[r * outCols + c] = xv[(r0 + r * rs) * COLS + c0 + c * cs];
            }
        }
        float[] eDy = new float[sliceRows * COLS];
        System.arraycopy(xv, 11 * COLS, eDy, 0, eDy.length);
        assertExact("slice", eSl, slMlx, slJit, 0);
        assertExact("sliceRowsAt", eDy, dyMlx, dyJit, 0);
    }

    @Test
    public void testSliceUpdates() throws TornadoExecutionPlanException {
        final int r0 = 5;
        final int c0 = 4;
        final int ur = 9;
        final int uc = 10;
        for (int op : new int[] { JitIndex.SET, JitIndex.ADD, JitIndex.MAX, JitIndex.MIN, JitIndex.PROD }) {
            float[] xv = values(ROWS * COLS, 1, 2, 181);
            float[] uv = values(ur * uc, 0.5f, 2.5f, 182);
            FloatArray x = FloatArray.fromArray(xv);
            FloatArray u = FloatArray.fromArray(uv);
            FloatArray outMlx = new FloatArray(xv.length);
            FloatArray outJit = new FloatArray(xv.length);
            TaskGraph g = new TaskGraph("su").transferToDevice(DataTransferMode.FIRST_EXECUTION, x, u);
            switch (op) {
                case JitIndex.SET -> g.libraryTask("m", MlxIndex::sliceUpdate, x, u, outMlx, ROWS, COLS, r0, c0, ur, uc);
                case JitIndex.ADD -> g.libraryTask("m", MlxIndex::sliceUpdateAdd, x, u, outMlx, ROWS, COLS, r0, c0, ur, uc);
                case JitIndex.MAX -> g.libraryTask("m", MlxIndex::sliceUpdateMax, x, u, outMlx, ROWS, COLS, r0, c0, ur, uc);
                case JitIndex.MIN -> g.libraryTask("m", MlxIndex::sliceUpdateMin, x, u, outMlx, ROWS, COLS, r0, c0, ur, uc);
                default -> g.libraryTask("m", MlxIndex::sliceUpdateProd, x, u, outMlx, ROWS, COLS, r0, c0, ur, uc);
            }
            g.task("c", JitIndex::copy, new KernelContext(), x, outJit, xv.length) //
                    .task("s", JitIndex::sliceUpdate, new KernelContext(), u, outJit, COLS, r0, c0, ur, uc, op) //
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, outMlx, outJit);
            GridScheduler gs = new GridScheduler();
            gs.addWorkerGrid("su.c", TestJitElementwise.grid1D(xv.length));
            gs.addWorkerGrid("su.s", TestJitElementwise.grid1D(ur * uc));
            execute(g, gs);
            float[] expected = xv.clone();
            for (int r = 0; r < ur; r++) {
                for (int c = 0; c < uc; c++) {
                    int p = (r0 + r) * COLS + c0 + c;
                    expected[p] = applyRef(expected[p], uv[r * uc + c], op);
                }
            }
            assertExact("sliceUpdate op=" + op, expected, outMlx, outJit, 1e-6);
        }
    }

    @Test
    public void testSliceUpdateRowsAt() throws TornadoExecutionPlanException {
        final int ur = 6;
        float[] xv = values(ROWS * COLS, -10, 10, 191);
        float[] uv = values(ur * COLS, -10, 10, 192);
        FloatArray x = FloatArray.fromArray(xv);
        FloatArray u = FloatArray.fromArray(uv);
        IntArray start = IntArray.fromElements(13);
        FloatArray outMlx = new FloatArray(xv.length);
        FloatArray outJit = new FloatArray(xv.length);
        TaskGraph g = new TaskGraph("sd").transferToDevice(DataTransferMode.FIRST_EXECUTION, x, u, start) //
                .libraryTask("m", MlxIndex::sliceUpdateRowsAt, x, u, start, outMlx, ROWS, COLS, ur) //
                .task("c", JitIndex::copy, new KernelContext(), x, outJit, xv.length) //
                .task("s", JitIndex::sliceUpdateRowsAt, new KernelContext(), u, start, outJit, COLS, ur) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, outMlx, outJit);
        GridScheduler gs = new GridScheduler();
        gs.addWorkerGrid("sd.c", TestJitElementwise.grid1D(xv.length));
        gs.addWorkerGrid("sd.s", TestJitElementwise.grid1D(ur * COLS));
        execute(g, gs);
        float[] expected = xv.clone();
        System.arraycopy(uv, 0, expected, 13 * COLS, uv.length);
        assertExact("sliceUpdateRowsAt", expected, outMlx, outJit, 0);
    }

    @Test
    public void testMaskedScatter() throws TornadoExecutionPlanException {
        final int n = 1000;
        float[] xv = values(n, -10, 10, 201);
        float[] sv = values(n, 100, 200, 202);
        ByteArray mask = new ByteArray(n);
        Random r = new Random(203);
        boolean[] mv = new boolean[n];
        for (int i = 0; i < n; i++) {
            mv[i] = r.nextInt(10) < 3;
            mask.set(i, (byte) (mv[i] ? 1 : 0));
        }
        FloatArray x = FloatArray.fromArray(xv);
        FloatArray src = FloatArray.fromArray(sv);
        IntArray positions = new IntArray(n);
        FloatArray outMlx = new FloatArray(n);
        FloatArray outJit = new FloatArray(n);
        TaskGraph g = new TaskGraph("ms").transferToDevice(DataTransferMode.FIRST_EXECUTION, x, mask, src) //
                .libraryTask("m", MlxIndex::maskedScatter, x, mask, src, outMlx) //
                .task("p", JitIndex::maskPositions, new KernelContext(), mask, positions, n) //
                .task("s", JitIndex::maskedScatter, new KernelContext(), x, mask, src, positions, outJit, n) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, outMlx, outJit);
        GridScheduler gs = new GridScheduler();
        gs.addWorkerGrid("ms.p", TestJitReduce.groups(1, 256));
        gs.addWorkerGrid("ms.s", TestJitElementwise.grid1D(n));
        execute(g, gs);
        float[] expected = new float[n];
        int k = 0;
        for (int i = 0; i < n; i++) {
            expected[i] = mv[i] ? sv[k++] : xv[i];
        }
        assertExact("maskedScatter", expected, outMlx, outJit, 0);
    }

    @Test
    public void testGatherMm() throws TornadoExecutionPlanException {
        final int batchesA = 4;
        final int batchesB = 3;
        final int m = 64;
        final int k = 32;
        final int n = 96;
        int[] lhs = { 3, 0, 1, 2, 3 };
        int[] rhs = { 2, 0, 1, 1, 2 };
        int count = lhs.length;
        float[] av = values(batchesA * m * k, -1, 1, 211);
        float[] bv = values(batchesB * k * n, -1, 1, 212);
        FloatArray a = FloatArray.fromArray(av);
        FloatArray b = FloatArray.fromArray(bv);
        IntArray li = IntArray.fromArray(lhs);
        IntArray ri = IntArray.fromArray(rhs);
        FloatArray outMlx = new FloatArray(count * m * n);
        FloatArray outJit = new FloatArray(count * m * n);
        TaskGraph g = new TaskGraph("gm").transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b, li, ri) //
                .libraryTask("m", MlxIndex::gatherMm, a, b, li, ri, outMlx, batchesA, batchesB, m, k, n) //
                .task("j", JitIndex::gatherMm, new KernelContext(), a, b, li, ri, outJit, m, n, k) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, outMlx, outJit);
        execute(g, new GridScheduler("gm.j", TestJitReduce.groups(count * (m / 32) * (n / 32), 128)));
        float[] expected = new float[count * m * n];
        for (int i = 0; i < count; i++) {
            for (int r = 0; r < m; r++) {
                for (int c = 0; c < n; c++) {
                    double s = 0;
                    for (int p = 0; p < k; p++) {
                        s += av[(lhs[i] * m + r) * k + p] * bv[(rhs[i] * k + p) * n + c];
                    }
                    expected[(i * m + r) * n + c] = (float) s;
                }
            }
        }
        assertExact("gatherMm", expected, outMlx, outJit, 1e-4);
    }

    @Test
    public void testHalfAndIntForms() throws TornadoExecutionPlanException {
        final int count = 7;
        float[] hv = values(ROWS * COLS, -10, 10, 221);
        HalfFloatArray x = half(hv);
        int[] idx = randomInts(count, ROWS, 222);
        IntArray ii = IntArray.fromArray(idx);
        HalfFloatArray rows = new HalfFloatArray(count * COLS);
        int[] base = new int[ROWS * COLS];
        IntArray xi = IntArray.fromArray(base);
        IntArray ups = new IntArray(count * COLS);
        ups.init(1);
        IntArray added = new IntArray(ROWS * COLS);
        run(new TaskGraph("hi").transferToDevice(DataTransferMode.FIRST_EXECUTION, x, ii, xi, ups) //
                .libraryTask("h", MlxIndex::takeAxis, x, ii, rows, 1, ROWS, COLS) //
                .libraryTask("i", MlxIndex::scatterAddRows, xi, ii, ups, added, ROWS, COLS) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, rows, added));
        float[] w = widen(x);
        int[] hits = new int[ROWS];
        for (int i = 0; i < count; i++) {
            hits[idx[i]]++;
            for (int c = 0; c < COLS; c++) {
                assertEquals("takeAxis float16", w[idx[i] * COLS + c], rows.get(i * COLS + c).getFloat32(), 0f);
            }
        }
        for (int p = 0; p < ROWS * COLS; p++) {
            assertEquals("scatterAddRows int32 " + p, hits[p / COLS], added.get(p));
        }
    }
}
