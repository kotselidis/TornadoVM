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
import uk.ac.manchester.tornado.mlx.MlxShape;
import uk.ac.manchester.tornado.mlx.jit.JitShape;

/**
 * The MLX shape and layout operations and their KernelContext JIT counterparts, run in one graph
 * and checked against Java index arithmetic.
 *
 * <p>
 * How to run?
 * </p>
 * <code>
 * tornado-test -V uk.ac.manchester.tornado.unittests.mlx.TestJitShape
 * </code>
 */
public class TestJitShape extends MlxTestBase {

    interface MlxTask {
        void add(TaskGraph g, FloatArray x, FloatArray out);
    }

    interface JitTask {
        void add(TaskGraph g, FloatArray x, FloatArray out);
    }

    private static FloatArray ramp(int n) {
        FloatArray a = new FloatArray(n);
        for (int i = 0; i < n; i++) {
            a.set(i, i * 0.5f - 7f);
        }
        return a;
    }

    /** Runs an MLX task and a JIT task (launched with {@code threads} threads) on x and checks both outputs against expected. */
    private static void check(String name, FloatArray x, float[] expected, MlxTask mlx, JitTask jit, int threads) throws TornadoExecutionPlanException {
        FloatArray outM = new FloatArray(expected.length);
        FloatArray outJ = new FloatArray(expected.length);
        TaskGraph g = new TaskGraph("sh").transferToDevice(DataTransferMode.FIRST_EXECUTION, x);
        mlx.add(g, x, outM);
        jit.add(g, x, outJ);
        g.transferToHost(DataTransferMode.EVERY_EXECUTION, outM, outJ);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(g.snapshot())) {
            plan.withGridScheduler(new GridScheduler("sh.j", TestJitElementwise.grid1D(threads))).execute();
        }
        for (int i = 0; i < expected.length; i++) {
            assertEquals(name + " MLX element " + i, expected[i], outM.get(i), 0f);
            assertEquals(name + " JIT element " + i, expected[i], outJ.get(i), 0f);
        }
    }

    @Test
    public void testLayoutPreserving() throws TornadoExecutionPlanException {
        final int n = 4 * 6 * 5;
        FloatArray x = ramp(n);
        float[] same = x.toHeapArray();
        JitTask copy = (g, in, out) -> g.task("j", JitShape::copy, new KernelContext(), in, out, n);
        check("reshape", x, same, (g, in, out) -> g.libraryTask("m", MlxShape::reshape, in, out, 12, 10), copy, n);
        check("flatten", x, same, (g, in, out) -> g.libraryTask("m", MlxShape::flatten, in, out, 4, 6, 5), copy, n);
        check("unflatten", x, same, (g, in, out) -> g.libraryTask("m", MlxShape::unflatten, in, out, 8, 15), copy, n);
        check("squeeze", x, same, (g, in, out) -> g.libraryTask("m", MlxShape::squeeze, in, out, 24, 5), copy, n);
        check("squeezeAxis", x, same, (g, in, out) -> g.libraryTask("m", MlxShape::squeezeAxis, in, out, 24, 5), copy, n);
        check("squeezeAxes", x, same, (g, in, out) -> g.libraryTask("m", MlxShape::squeezeAxes, in, out, 24, 5), copy, n);
        check("expandDims", x, same, (g, in, out) -> g.libraryTask("m", MlxShape::expandDims, in, out, 24, 5), copy, n);
        check("expandDimsAxes", x, same, (g, in, out) -> g.libraryTask("m", MlxShape::expandDimsAxes, in, out, 24, 5), copy, n);
        check("atleast1d", x, same, (g, in, out) -> g.libraryTask("m", MlxShape::atleast1d, in, out), copy, n);
        check("atleast2d", x, same, (g, in, out) -> g.libraryTask("m", MlxShape::atleast2d, in, out), copy, n);
        check("atleast3d", x, same, (g, in, out) -> g.libraryTask("m", MlxShape::atleast3d, in, out), copy, n);
        check("contiguous", x, same, (g, in, out) -> g.libraryTask("m", MlxShape::contiguous, in, out), copy, n);
        check("copy", x, same, (g, in, out) -> g.libraryTask("m", MlxShape::copy, in, out), copy, n);
    }

    /** x [d0, d1, d2] with output axis k taken from input axis p[k]. */
    private static float[] permute(float[] x, int[] d, int[] p) {
        int[] e = { d[p[0]], d[p[1]], d[p[2]] };
        float[] out = new float[x.length];
        for (int t = 0; t < x.length; t++) {
            int[] o = { t / (e[1] * e[2]), (t / e[2]) % e[1], t % e[2] };
            int[] in = new int[3];
            for (int k = 0; k < 3; k++) {
                in[p[k]] = o[k];
            }
            out[t] = x[(in[0] * d[1] + in[1]) * d[2] + in[2]];
        }
        return out;
    }

    @Test
    public void testPermutations() throws TornadoExecutionPlanException {
        final int[] d = { 4, 5, 6 };
        final int n = 120;
        FloatArray x = ramp(n);
        float[] xv = x.toHeapArray();
        check("transposeAxes", x, permute(xv, d, new int[] { 2, 0, 1 }), (g, in, out) -> g.libraryTask("m", MlxShape::transposeAxes, in, out, 4, 5, 6, 2, 0, 1), //
                (g, in, out) -> g.task("j", JitShape::permute3, new KernelContext(), in, out, 4, 5, 6, 2, 0, 1), n);
        check("swapaxes", x, permute(xv, d, new int[] { 2, 1, 0 }), (g, in, out) -> g.libraryTask("m", MlxShape::swapaxes, in, out, 4, 5, 6, 0, 2), //
                (g, in, out) -> g.task("j", JitShape::permute3, new KernelContext(), in, out, 4, 5, 6, 2, 1, 0), n);
        check("moveaxis", x, permute(xv, d, new int[] { 1, 2, 0 }), (g, in, out) -> g.libraryTask("m", MlxShape::moveaxis, in, out, 4, 5, 6, 0, 2), //
                (g, in, out) -> g.task("j", JitShape::permute3, new KernelContext(), in, out, 4, 5, 6, 1, 2, 0), n);
    }

    @Test
    public void testBroadcastAndStrides() throws TornadoExecutionPlanException {
        final int rows = 7;
        final int cols = 9;
        FloatArray row = ramp(cols);
        float[] eb = new float[rows * cols];
        for (int t = 0; t < eb.length; t++) {
            eb[t] = row.get(t % cols);
        }
        check("broadcastTo", row, eb, (g, in, out) -> g.libraryTask("m", MlxShape::broadcastTo, in, out, rows, cols), //
                (g, in, out) -> g.task("j", JitShape::broadcastRows, new KernelContext(), in, out, rows, cols), rows * cols);
        FloatArray x = ramp(64);
        float[] es = new float[5 * 4];
        for (int t = 0; t < es.length; t++) {
            es[t] = x.get(3 + (t / 4) * 7 + (t % 4) * 2);
        }
        check("asStrided", x, es, (g, in, out) -> g.libraryTask("m", MlxShape::asStrided, in, out, 5, 4, 7, 2, 3), //
                (g, in, out) -> g.task("j", JitShape::asStrided, new KernelContext(), in, out, 5, 4, 7, 2, 3), 20);

        FloatArray col = ramp(rows);
        FloatArray aM = new FloatArray(rows * cols);
        FloatArray bM = new FloatArray(rows * cols);
        FloatArray aJ = new FloatArray(rows * cols);
        FloatArray bJ = new FloatArray(rows * cols);
        TaskGraph g = new TaskGraph("ba").transferToDevice(DataTransferMode.FIRST_EXECUTION, row, col) //
                .libraryTask("m", MlxShape::broadcastArrays, row, col, aM, bM, rows, cols) //
                .task("j", JitShape::broadcastPair, new KernelContext(), row, col, aJ, bJ, rows, cols) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, aM, bM, aJ, bJ);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(g.snapshot())) {
            plan.withGridScheduler(new GridScheduler("ba.j", TestJitElementwise.grid1D(rows * cols))).execute();
        }
        for (int t = 0; t < rows * cols; t++) {
            assertEquals("broadcastArrays a MLX", row.get(t % cols), aM.get(t), 0f);
            assertEquals("broadcastArrays b MLX", col.get(t / cols), bM.get(t), 0f);
            assertEquals("broadcastArrays a JIT", row.get(t % cols), aJ.get(t), 0f);
            assertEquals("broadcastArrays b JIT", col.get(t / cols), bJ.get(t), 0f);
        }
    }

    @Test
    public void testTypes() throws TornadoExecutionPlanException {
        final int n = 257;
        float[] xv = values(n, -100, 100, 21);
        FloatArray x = FloatArray.fromArray(xv);
        IntArray bitsM = new IntArray(n);
        IntArray bitsJ = new IntArray(n);
        HalfFloatArray halfM = new HalfFloatArray(n);
        HalfFloatArray halfJ = new HalfFloatArray(n);
        IntArray ints = new IntArray(n);
        FloatArray back = new FloatArray(n);
        IntArray count = new IntArray(1);
        IntArray countJ = new IntArray(1);
        FloatArray cube = ramp(3 * 4 * 5);
        TaskGraph g = new TaskGraph("ty").transferToDevice(DataTransferMode.FIRST_EXECUTION, x, cube) //
                .libraryTask("m1", MlxShape::view, x, bitsM) //
                .libraryTask("m2", MlxShape::astype, x, halfM) //
                .libraryTask("m3", MlxShape::astype, x, ints) //
                .libraryTask("m4", MlxShape::view, bitsM, back) //
                .libraryTask("m5", MlxShape::numberOfElements, cube, count, 3, 4, 5) //
                .task("j1", JitShape::viewAsInt, new KernelContext(), x, bitsJ, n) //
                .task("j2", JitShape::toHalf, new KernelContext(), x, halfJ, n) //
                .task("j3", JitShape::writeInt, new KernelContext(), countJ, 4 * 5) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, bitsM, bitsJ, halfM, halfJ, ints, back, count, countJ);
        GridScheduler gs = new GridScheduler();
        gs.addWorkerGrid("ty.j1", TestJitElementwise.grid1D(n));
        gs.addWorkerGrid("ty.j2", TestJitElementwise.grid1D(n));
        gs.addWorkerGrid("ty.j3", TestJitElementwise.grid1D(1));
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(g.snapshot())) {
            plan.withGridScheduler(gs).execute();
        }
        for (int i = 0; i < n; i++) {
            assertEquals("view MLX", Float.floatToRawIntBits(xv[i]), bitsM.get(i));
            assertEquals("view JIT", Float.floatToRawIntBits(xv[i]), bitsJ.get(i));
            float h = new uk.ac.manchester.tornado.api.types.HalfFloat(xv[i]).getFloat32();
            assertEquals("astype float16 MLX", h, halfM.get(i).getFloat32(), Math.abs(h) * 1e-3f);
            assertEquals("astype float16 JIT", h, halfJ.get(i).getFloat32(), Math.abs(h) * 1e-3f);
            assertEquals("astype int32 MLX", (int) xv[i], ints.get(i));
            assertEquals("view back MLX", xv[i], back.get(i), 0f);
        }
        assertEquals("numberOfElements MLX", 20, count.get(0));
        assertEquals("numberOfElements JIT", 20, countJ.get(0));
    }

    private static void pairCheck(String name, FloatArray a, FloatArray b, float[] expected, MlxPair mlx, JitPair jit, int threads) throws TornadoExecutionPlanException {
        FloatArray outM = new FloatArray(expected.length);
        FloatArray outJ = new FloatArray(expected.length);
        TaskGraph g = new TaskGraph("pr").transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b);
        mlx.add(g, outM);
        jit.add(g, outJ);
        g.transferToHost(DataTransferMode.EVERY_EXECUTION, outM, outJ);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(g.snapshot())) {
            plan.withGridScheduler(new GridScheduler("pr.j", TestJitElementwise.grid1D(threads))).execute();
        }
        for (int i = 0; i < expected.length; i++) {
            assertEquals(name + " MLX element " + i, expected[i], outM.get(i), 0f);
            assertEquals(name + " JIT element " + i, expected[i], outJ.get(i), 0f);
        }
    }

    interface MlxPair {
        void add(TaskGraph g, FloatArray out);
    }

    interface JitPair {
        void add(TaskGraph g, FloatArray out);
    }

    @Test
    public void testJoinAndSplit() throws TornadoExecutionPlanException {
        final int rows = 6;
        final int colsA = 5;
        final int colsB = 3;
        FloatArray a = ramp(rows * colsA);
        FloatArray b = FloatArray.fromArray(values(rows * colsB, 100, 200, 22));
        int na = a.getSize();
        int nb = b.getSize();
        float[] cat = new float[na + nb];
        for (int i = 0; i < na; i++) {
            cat[i] = a.get(i);
        }
        for (int i = 0; i < nb; i++) {
            cat[na + i] = b.get(i);
        }
        pairCheck("concatenate", a, b, cat, (g, out) -> g.libraryTask("m", MlxShape::concatenate, a, b, out), //
                (g, out) -> g.task("j", JitShape::concat, new KernelContext(), a, b, out, 1, na, nb), na + nb);
        float[] catAxis = new float[rows * (colsA + colsB)];
        for (int t = 0; t < catAxis.length; t++) {
            int r = t / (colsA + colsB);
            int c = t % (colsA + colsB);
            catAxis[t] = c < colsA ? a.get(r * colsA + c) : b.get(r * colsB + c - colsA);
        }
        pairCheck("concatenateAxis", a, b, catAxis, (g, out) -> g.libraryTask("m", MlxShape::concatenateAxis, a, b, out, rows, colsA, colsB), //
                (g, out) -> g.task("j", JitShape::concat, new KernelContext(), a, b, out, rows, colsA, colsB), catAxis.length);
        FloatArray a2 = ramp(nb);
        float[] stacked = new float[2 * nb];
        float[] interleaved = new float[2 * nb];
        for (int i = 0; i < nb; i++) {
            stacked[i] = a2.get(i);
            stacked[nb + i] = b.get(i);
            interleaved[2 * i] = a2.get(i);
            interleaved[2 * i + 1] = b.get(i);
        }
        pairCheck("stack", a2, b, stacked, (g, out) -> g.libraryTask("m", MlxShape::stack, a2, b, out), //
                (g, out) -> g.task("j", JitShape::concat, new KernelContext(), a2, b, out, 1, nb, nb), 2 * nb);
        pairCheck("stackAxis", a2, b, interleaved, (g, out) -> g.libraryTask("m", MlxShape::stackAxis, a2, b, out), //
                (g, out) -> g.task("j", JitShape::interleave, new KernelContext(), a2, b, out, nb), nb);

        final int cols = 8;
        FloatArray x = ramp(rows * cols);
        for (int index : new int[] { cols / 2, 3 }) {
            boolean halves = index == cols / 2;
            FloatArray f1 = new FloatArray(rows * index);
            FloatArray f2 = new FloatArray(rows * (cols - index));
            FloatArray g1 = new FloatArray(rows * index);
            FloatArray g2 = new FloatArray(rows * (cols - index));
            TaskGraph g = new TaskGraph("sp").transferToDevice(DataTransferMode.FIRST_EXECUTION, x);
            if (halves) {
                g.libraryTask("m", MlxShape::split, x, f1, f2, rows, cols);
            } else {
                g.libraryTask("m", MlxShape::splitSections, x, f1, f2, rows, cols, index);
            }
            g.task("j", JitShape::split, new KernelContext(), x, g1, g2, rows, cols, index).transferToHost(DataTransferMode.EVERY_EXECUTION, f1, f2, g1, g2);
            try (TornadoExecutionPlan plan = new TornadoExecutionPlan(g.snapshot())) {
                plan.withGridScheduler(new GridScheduler("sp.j", TestJitElementwise.grid1D(rows * cols))).execute();
            }
            for (int t = 0; t < rows * cols; t++) {
                int r = t / cols;
                int c = t % cols;
                String what = (halves ? "split" : "splitSections") + " element " + t;
                if (c < index) {
                    assertEquals(what + " MLX", x.get(t), f1.get(r * index + c), 0f);
                    assertEquals(what + " JIT", x.get(t), g1.get(r * index + c), 0f);
                } else {
                    assertEquals(what + " MLX", x.get(t), f2.get(r * (cols - index) + c - index), 0f);
                    assertEquals(what + " JIT", x.get(t), g2.get(r * (cols - index) + c - index), 0f);
                }
            }
        }
    }

    @Test
    public void testRepeatTileRollPad() throws TornadoExecutionPlanException {
        final int rows = 5;
        final int cols = 7;
        final int n = rows * cols;
        FloatArray x = ramp(n);
        float[] xv = x.toHeapArray();
        float[] rep = new float[3 * n];
        for (int t = 0; t < rep.length; t++) {
            rep[t] = xv[t / 3];
        }
        check("repeat", x, rep, (g, in, out) -> g.libraryTask("m", MlxShape::repeat, in, out, 3), //
                (g, in, out) -> g.task("j", JitShape::repeatRows, new KernelContext(), in, out, n, 1, 3), rep.length);
        float[] repAxis = new float[2 * n];
        for (int t = 0; t < repAxis.length; t++) {
            repAxis[t] = xv[(t / (2 * cols)) * cols + t % cols];
        }
        check("repeatAxis", x, repAxis, (g, in, out) -> g.libraryTask("m", MlxShape::repeatAxis, in, out, rows, cols, 2), //
                (g, in, out) -> g.task("j", JitShape::repeatRows, new KernelContext(), in, out, rows, cols, 2), repAxis.length);
        float[] tile = new float[2 * rows * 3 * cols];
        for (int t = 0; t < tile.length; t++) {
            tile[t] = xv[((t / (3 * cols)) % rows) * cols + (t % (3 * cols)) % cols];
        }
        check("tile", x, tile, (g, in, out) -> g.libraryTask("m", MlxShape::tile, in, out, rows, cols, 2, 3), //
                (g, in, out) -> g.task("j", JitShape::tile, new KernelContext(), in, out, rows, cols, 2, 3), tile.length);
        float[] roll = new float[n];
        float[] rollAxis = new float[n];
        float[] rollAxes = new float[n];
        for (int t = 0; t < n; t++) {
            roll[t] = xv[Math.floorMod(t - 4, n)];
            int r = t / cols;
            int c = t % cols;
            rollAxis[t] = xv[r * cols + Math.floorMod(c + 2, cols)];
            rollAxes[t] = xv[Math.floorMod(r - 1, rows) * cols + Math.floorMod(c - 3, cols)];
        }
        check("roll", x, roll, (g, in, out) -> g.libraryTask("m", MlxShape::roll, in, out, 4), //
                (g, in, out) -> g.task("j", JitShape::roll, new KernelContext(), in, out, 1, n, 0, 4), n);
        check("rollAxis", x, rollAxis, (g, in, out) -> g.libraryTask("m", MlxShape::rollAxis, in, out, rows, cols, -2), //
                (g, in, out) -> g.task("j", JitShape::roll, new KernelContext(), in, out, rows, cols, 0, -2), n);
        check("rollAxes", x, rollAxes, (g, in, out) -> g.libraryTask("m", MlxShape::rollAxes, in, out, rows, cols, 1, 3), //
                (g, in, out) -> g.task("j", JitShape::roll, new KernelContext(), in, out, rows, cols, 1, 3), n);
        int outRows = rows + 1 + 2;
        int outCols = cols + 3 + 0;
        float[] pad = new float[outRows * outCols];
        for (int t = 0; t < pad.length; t++) {
            int r = t / outCols - 1;
            int c = t % outCols - 3;
            pad[t] = r >= 0 && r < rows && c >= 0 && c < cols ? xv[r * cols + c] : -9f;
        }
        check("pad", x, pad, (g, in, out) -> g.libraryTask("m", MlxShape::pad, in, out, rows, cols, 1, 2, 3, 0, -9f), //
                (g, in, out) -> g.task("j", JitShape::pad, new KernelContext(), in, out, rows, cols, 1, 3, outRows, outCols, -9f), pad.length);
        int w = 2;
        float[] padS = new float[(rows + 2 * w) * (cols + 2 * w)];
        for (int t = 0; t < padS.length; t++) {
            int r = t / (cols + 2 * w) - w;
            int c = t % (cols + 2 * w) - w;
            padS[t] = r >= 0 && r < rows && c >= 0 && c < cols ? xv[r * cols + c] : 0.5f;
        }
        check("padSymmetric", x, padS, (g, in, out) -> g.libraryTask("m", MlxShape::padSymmetric, in, out, rows, cols, w, 0.5f), //
                (g, in, out) -> g.task("j", JitShape::pad, new KernelContext(), in, out, rows, cols, w, w, rows + 2 * w, cols + 2 * w, 0.5f), padS.length);
    }

    @Test
    public void testIntForms() throws TornadoExecutionPlanException {
        final int rows = 4;
        final int cols = 6;
        int[] xv = new int[rows * cols];
        for (int i = 0; i < xv.length; i++) {
            xv[i] = i * 3 - 10;
        }
        IntArray x = IntArray.fromArray(xv);
        IntArray t = new IntArray(xv.length);
        IntArray r = new IntArray(xv.length);
        FloatArray f = new FloatArray(xv.length);
        run(new TaskGraph("it").transferToDevice(DataTransferMode.FIRST_EXECUTION, x) //
                .libraryTask("t", MlxShape::transposeAxes, x, t, 1, rows, cols, 0, 2, 1) //
                .libraryTask("r", MlxShape::rollAxis, x, r, rows, cols, 1) //
                .libraryTask("f", MlxShape::astype, x, f) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, t, r, f));
        for (int i = 0; i < xv.length; i++) {
            int rr = i / rows;
            int cc = i % rows;
            assertEquals("transposeAxes int32", xv[cc * cols + rr], t.get(i));
            assertEquals("rollAxis int32", xv[(i / cols) * cols + Math.floorMod(i % cols - 1, cols)], r.get(i));
            assertEquals("astype int32 to float32", xv[i], f.get(i), 0f);
        }
    }
}
