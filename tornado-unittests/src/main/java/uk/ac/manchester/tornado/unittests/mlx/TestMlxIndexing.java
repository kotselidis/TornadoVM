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

import java.util.function.DoubleBinaryOperator;

import org.junit.Test;

import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.common.LibraryTaskDescriptor;
import uk.ac.manchester.tornado.api.common.TornadoFunctions.LibraryTask9;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.mlx.MlxIndex;
import java.util.Arrays;
import java.util.Random;

/**
 * Unit tests for the MLX indexing library tasks: take, take along an axis, put along an axis,
 * gather, scatter (set, add, max, min, multiply; points and rows), masked scatter, strided and dynamic
 * slices, and slice updates (set, add, max, min, multiply; fixed and dynamic). An update writes the modified copy to {@code out} and leaves {@code x} unchanged. Each
 * test checks the task against a sequential Java reference. Skipped unless the default device is on
 * the Metal backend and mlx.metallib is available.
 *
 * <p>
 * How to run?
 * </p>
 * <code>
 * tornado-test -V uk.ac.manchester.tornado.unittests.mlx.TestMlxIndexing
 * </code>
 */
public class TestMlxIndexing extends MlxTestBase {

    private static final int ROWS = 40;
    private static final int COLS = 24;
    private static final int R0 = 5;
    private static final int C0 = 4;
    private static final int UPDATE_ROWS = 9;
    private static final int UPDATE_COLS = 10;

    /** x[rows, cols] with the update[ur, uc] block at (r0, c0) combined by {@code op}. */
    private static float[] sliceUpdateJava(float[] x, float[] update, DoubleBinaryOperator op) {
        float[] out = x.clone();
        for (int r = 0; r < UPDATE_ROWS; r++) {
            for (int c = 0; c < UPDATE_COLS; c++) {
                int p = (R0 + r) * COLS + C0 + c;
                out[p] = (float) op.applyAsDouble(out[p], update[r * UPDATE_COLS + c]);
            }
        }
        return out;
    }

    private static void sliceUpdate(LibraryTask9<FloatArray, FloatArray, FloatArray, Integer, Integer, Integer, Integer, Integer, Integer> mlx, DoubleBinaryOperator op)
            throws TornadoExecutionPlanException {
        float[] xv = values(ROWS * COLS, 1, 2, 1);
        float[] uv = values(UPDATE_ROWS * UPDATE_COLS, 0.5f, 2.5f, 2);
        FloatArray x = FloatArray.fromArray(xv);
        FloatArray update = FloatArray.fromArray(uv);
        FloatArray output = new FloatArray(xv.length);

        TaskGraph taskGraph = new TaskGraph("g") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, x, update) //
                .libraryTask("update", mlx, x, update, output, ROWS, COLS, R0, C0, UPDATE_ROWS, UPDATE_COLS) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, output, x);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
            plan.execute();
        }

        float[] expected = sliceUpdateJava(xv, uv, op);
        for (int i = 0; i < expected.length; i++) {
            assertEquals("element " + i, expected[i], output.get(i), 1e-6f * Math.abs(expected[i]));
            assertEquals("x must not change, element " + i, xv[i], x.get(i), 0f);
        }
    }

    @Test
    public void testSlice() throws TornadoExecutionPlanException {
        // x[3:37:2, 1:23:3]
        final int r0 = 3;
        final int r1 = 37;
        final int rowStep = 2;
        final int c0 = 1;
        final int c1 = 23;
        final int colStep = 3;
        final int outRows = (r1 - r0 + rowStep - 1) / rowStep;
        final int outCols = (c1 - c0 + colStep - 1) / colStep;
        float[] xv = values(ROWS * COLS, -10, 10, 3);
        FloatArray x = FloatArray.fromArray(xv);
        FloatArray output = new FloatArray(outRows * outCols);

        TaskGraph taskGraph = new TaskGraph("g") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, x) //
                .libraryTask("slice", MlxIndex::slice, x, output, ROWS, COLS, r0, r1, rowStep, c0, c1, colStep) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
            plan.execute();
        }

        for (int r = 0; r < outRows; r++) {
            for (int c = 0; c < outCols; c++) {
                assertEquals(r + "," + c, xv[(r0 + r * rowStep) * COLS + c0 + c * colStep], output.get(r * outCols + c), 0f);
            }
        }
    }

    @Test
    public void testSliceUpdate() throws TornadoExecutionPlanException {
        sliceUpdate(MlxIndex::sliceUpdate, (old, u) -> u);
    }

    @Test
    public void testSliceUpdateAdd() throws TornadoExecutionPlanException {
        sliceUpdate(MlxIndex::sliceUpdateAdd, Double::sum);
    }

    @Test
    public void testSliceUpdateProd() throws TornadoExecutionPlanException {
        sliceUpdate(MlxIndex::sliceUpdateProd, (old, u) -> old * u);
    }

    // ---------------------------------------------------------------- other types

    @Test
    public void testSliceInt() throws TornadoExecutionPlanException {
        int[] xv = new int[ROWS * COLS];
        for (int i = 0; i < xv.length; i++) {
            xv[i] = i;
        }
        IntArray x = IntArray.fromArray(xv);
        // x[0:40:4, 2:24:5]
        final int outRows = 10;
        final int outCols = 5;
        IntArray output = new IntArray(outRows * outCols);

        TaskGraph taskGraph = new TaskGraph("g") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, x) //
                .libraryTask("slice", MlxIndex::slice, x, output, ROWS, COLS, 0, ROWS, 4, 2, COLS, 5) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
            plan.execute();
        }

        for (int r = 0; r < outRows; r++) {
            for (int c = 0; c < outCols; c++) {
                assertEquals(r + "," + c, (r * 4) * COLS + 2 + c * 5, output.get(r * outCols + c));
            }
        }
    }

    @Test
    public void testSliceUpdateAddHalf() throws TornadoExecutionPlanException {
        float[] xv = values(ROWS * COLS, 1, 2, 4);
        float[] uv = values(UPDATE_ROWS * UPDATE_COLS, 0.5f, 2.5f, 5);
        HalfFloatArray x = half(xv);
        HalfFloatArray update = half(uv);
        HalfFloatArray output = new HalfFloatArray(xv.length);

        TaskGraph taskGraph = new TaskGraph("g") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, x, update) //
                .libraryTask("update", MlxIndex::sliceUpdateAdd, x, update, output, ROWS, COLS, R0, C0, UPDATE_ROWS, UPDATE_COLS) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
            plan.execute();
        }

        float[] expected = sliceUpdateJava(widen(x), widen(update), Double::sum);
        double[] e = new double[expected.length];
        for (int i = 0; i < e.length; i++) {
            e[i] = expected[i];
        }
        assertAllClose("sliceUpdateAdd float16", e, output, 1e-3, 1e-3);
    }

    // ---------------------------------------------------------------- take, gather and scatter

    private static int[] randomInts(int n, int bound, long seed) {
        Random random = new Random(seed);
        int[] v = new int[n];
        for (int i = 0; i < n; i++) {
            v[i] = random.nextInt(bound);
        }
        return v;
    }

    /** The first n of a random permutation of [0, bound): distinct indices. */
    private static int[] distinct(int n, int bound, long seed) {
        int[] p = new int[bound];
        for (int i = 0; i < bound; i++) {
            p[i] = i;
        }
        Random random = new Random(seed);
        for (int i = bound - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int t = p[i];
            p[i] = p[j];
            p[j] = t;
        }
        return Arrays.copyOf(p, n);
    }

    private static void assertEach(float[] expected, FloatArray output, double relTol) {
        for (int i = 0; i < expected.length; i++) {
            assertClose("element", i, expected[i], output.get(i), relTol, relTol);
        }
    }

    @Test
    public void testTake() throws TornadoExecutionPlanException {
        float[] xv = values(ROWS * COLS, -10, 10, 10);
        int[] idx = randomInts(100, xv.length, 11);
        FloatArray x = FloatArray.fromArray(xv);
        IntArray indices = IntArray.fromArray(idx);
        FloatArray output = new FloatArray(idx.length);

        TaskGraph taskGraph = new TaskGraph("g") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, x, indices) //
                .libraryTask("take", MlxIndex::take, x, indices, output) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
            plan.execute();
        }

        for (int i = 0; i < idx.length; i++) {
            assertEquals("element " + i, xv[idx[i]], output.get(i), 0f);
        }
    }

    @Test
    public void testTakeAxis() throws TornadoExecutionPlanException {
        // 17 positions of the middle axis of x[3, 40, 5].
        final int outer = 3;
        final int len = 40;
        final int inner = 5;
        final int count = 17;
        float[] xv = values(outer * len * inner, -10, 10, 12);
        int[] idx = randomInts(count, len, 13);
        FloatArray x = FloatArray.fromArray(xv);
        IntArray indices = IntArray.fromArray(idx);
        FloatArray output = new FloatArray(outer * count * inner);

        TaskGraph taskGraph = new TaskGraph("g") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, x, indices) //
                .libraryTask("takeAxis", MlxIndex::takeAxis, x, indices, output, outer, len, inner) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
            plan.execute();
        }

        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < count; i++) {
                for (int j = 0; j < inner; j++) {
                    assertEquals(o + "," + i + "," + j, xv[(o * len + idx[i]) * inner + j], output.get((o * count + i) * inner + j), 0f);
                }
            }
        }
    }

    @Test
    public void testTakeAlongAxis() throws TornadoExecutionPlanException {
        final int outer = 3;
        final int len = 40;
        final int inner = 5;
        final int m = 6;
        float[] xv = values(outer * len * inner, -10, 10, 14);
        int[] idx = randomInts(outer * m * inner, len, 15);
        FloatArray x = FloatArray.fromArray(xv);
        IntArray indices = IntArray.fromArray(idx);
        FloatArray output = new FloatArray(outer * m * inner);

        TaskGraph taskGraph = new TaskGraph("g") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, x, indices) //
                .libraryTask("takeAlongAxis", MlxIndex::takeAlongAxis, x, indices, output, outer, len, m, inner) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
            plan.execute();
        }

        for (int t = 0; t < idx.length; t++) {
            assertEquals("element " + t, xv[((t / (m * inner)) * len + idx[t]) * inner + t % inner], output.get(t), 0f);
        }
    }

    private static void alongAxisUpdate(boolean add) throws TornadoExecutionPlanException {
        final int outer = 3;
        final int len = 40;
        final int inner = 5;
        final int m = 6;
        float[] xv = values(outer * len * inner, -10, 10, 16);
        float[] vv = values(outer * m * inner, -10, 10, 17);
        int[] idx = new int[outer * m * inner];
        for (int o = 0; o < outer; o++) {
            for (int j = 0; j < inner; j++) {
                // put needs distinct indices per slice; add accumulates repeated ones.
                int[] rows = add ? randomInts(m, 4, 18L + o * inner + j) : distinct(m, len, 18L + o * inner + j);
                for (int i = 0; i < m; i++) {
                    idx[(o * m + i) * inner + j] = rows[i];
                }
            }
        }
        FloatArray x = FloatArray.fromArray(xv);
        FloatArray values = FloatArray.fromArray(vv);
        IntArray indices = IntArray.fromArray(idx);
        FloatArray output = new FloatArray(xv.length);

        TaskGraph taskGraph = new TaskGraph("g").transferToDevice(DataTransferMode.EVERY_EXECUTION, x, values, indices);
        if (add) {
            taskGraph.libraryTask("scatterAddAxis", MlxIndex::scatterAddAxis, x, indices, values, output, outer, len, m, inner);
        } else {
            taskGraph.libraryTask("putAlongAxis", MlxIndex::putAlongAxis, x, indices, values, output, outer, len, m, inner);
        }
        taskGraph.transferToHost(DataTransferMode.EVERY_EXECUTION, output);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
            plan.execute();
        }

        float[] expected = xv.clone();
        for (int t = 0; t < idx.length; t++) {
            int p = ((t / (m * inner)) * len + idx[t]) * inner + t % inner;
            expected[p] = add ? expected[p] + vv[t] : vv[t];
        }
        assertEach(expected, output, 1e-5);
    }

    @Test
    public void testPutAlongAxis() throws TornadoExecutionPlanException {
        alongAxisUpdate(false);
    }

    @Test
    public void testScatterAddAxis() throws TornadoExecutionPlanException {
        alongAxisUpdate(true);
    }

    @Test
    public void testGather() throws TornadoExecutionPlanException {
        // out[i] = x[rows[i], cols[i]]
        final int count = 50;
        float[] xv = values(ROWS * COLS, -10, 10, 19);
        int[] rows = randomInts(count, ROWS, 20);
        int[] cols = randomInts(count, COLS, 21);
        FloatArray x = FloatArray.fromArray(xv);
        IntArray rowIndices = IntArray.fromArray(rows);
        IntArray colIndices = IntArray.fromArray(cols);
        FloatArray output = new FloatArray(count);

        TaskGraph taskGraph = new TaskGraph("g") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, x, rowIndices, colIndices) //
                .libraryTask("gather", MlxIndex::gather, x, rowIndices, colIndices, output, ROWS, COLS) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
            plan.execute();
        }

        for (int i = 0; i < count; i++) {
            assertEquals("element " + i, xv[rows[i] * COLS + cols[i]], output.get(i), 0f);
        }
    }

    @Test
    public void testGatherRows() throws TornadoExecutionPlanException {
        // Windows of 3 rows starting at each start index.
        final int windows = 9;
        final int sliceRows = 3;
        float[] xv = values(ROWS * COLS, -10, 10, 22);
        int[] starts = randomInts(windows, ROWS - sliceRows + 1, 23);
        FloatArray x = FloatArray.fromArray(xv);
        IntArray startIndices = IntArray.fromArray(starts);
        FloatArray output = new FloatArray(windows * sliceRows * COLS);

        TaskGraph taskGraph = new TaskGraph("g") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, x, startIndices) //
                .libraryTask("gatherRows", MlxIndex::gatherRows, x, startIndices, output, ROWS, COLS, sliceRows) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
            plan.execute();
        }

        for (int t = 0; t < output.getSize(); t++) {
            int w = t / (sliceRows * COLS);
            int r = (t / COLS) % sliceRows;
            assertEquals("element " + t, xv[(starts[w] + r) * COLS + t % COLS], output.get(t), 0f);
        }
    }

    interface PointScatter {
        LibraryTaskDescriptor scatter(FloatArray x, IntArray rows, IntArray cols, FloatArray updates, FloatArray out, int r, int c);
    }

    interface RowScatter {
        LibraryTaskDescriptor scatter(FloatArray x, IntArray indices, FloatArray updates, FloatArray out, int r, int c);
    }

    /** Scatters 60 point updates into x[ROWS, COLS], combined with {@code op}. */
    private static void scatterPoints(PointScatter mlx, DoubleBinaryOperator op, boolean repeated, float lo, float hi) throws TornadoExecutionPlanException {
        final int count = 60;
        float[] xv = values(ROWS * COLS, 1, 2, 24);
        float[] uv = values(count, lo, hi, 25);
        int[] rows;
        int[] cols;
        if (repeated) {
            rows = randomInts(count, 5, 26);
            cols = randomInts(count, 4, 27);
        } else {
            int[] flat = distinct(count, ROWS * COLS, 28);
            rows = new int[count];
            cols = new int[count];
            for (int i = 0; i < count; i++) {
                rows[i] = flat[i] / COLS;
                cols[i] = flat[i] % COLS;
            }
        }
        FloatArray x = FloatArray.fromArray(xv);
        FloatArray updates = FloatArray.fromArray(uv);
        IntArray rowIndices = IntArray.fromArray(rows);
        IntArray colIndices = IntArray.fromArray(cols);
        FloatArray output = new FloatArray(xv.length);

        TaskGraph taskGraph = new TaskGraph("g") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, x, updates, rowIndices, colIndices) //
                .libraryTask("scatter", (FloatArray a, IntArray r, IntArray c, FloatArray u, FloatArray o) -> mlx.scatter(a, r, c, u, o, ROWS, COLS), x, rowIndices, colIndices,
                        updates, output) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
            plan.execute();
        }

        float[] expected = xv.clone();
        for (int i = 0; i < count; i++) {
            int p = rows[i] * COLS + cols[i];
            expected[p] = (float) op.applyAsDouble(expected[p], uv[i]);
        }
        assertEach(expected, output, 1e-5);
    }

    /** Scatters 10 rows of updates into x[ROWS, COLS], combined with {@code op}. */
    private static void scatterRows(RowScatter mlx, DoubleBinaryOperator op, boolean repeated, float lo, float hi) throws TornadoExecutionPlanException {
        final int count = 10;
        float[] xv = values(ROWS * COLS, 1, 2, 29);
        float[] uv = values(count * COLS, lo, hi, 30);
        int[] idx = repeated ? randomInts(count, 4, 31) : distinct(count, ROWS, 32);
        FloatArray x = FloatArray.fromArray(xv);
        FloatArray updates = FloatArray.fromArray(uv);
        IntArray indices = IntArray.fromArray(idx);
        FloatArray output = new FloatArray(xv.length);

        TaskGraph taskGraph = new TaskGraph("g") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, x, updates, indices) //
                .libraryTask("scatter", (FloatArray a, IntArray i, FloatArray u, FloatArray o) -> mlx.scatter(a, i, u, o, ROWS, COLS), x, indices, updates, output) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
            plan.execute();
        }

        float[] expected = xv.clone();
        for (int t = 0; t < count * COLS; t++) {
            int p = idx[t / COLS] * COLS + t % COLS;
            expected[p] = (float) op.applyAsDouble(expected[p], uv[t]);
        }
        assertEach(expected, output, 1e-5);
    }

    @Test
    public void testScatter() throws TornadoExecutionPlanException {
        scatterPoints(MlxIndex::scatter, (old, u) -> u, false, -3, 3);
    }

    @Test
    public void testScatterAdd() throws TornadoExecutionPlanException {
        // Repeated indices accumulate.
        scatterPoints(MlxIndex::scatterAdd, Double::sum, true, -3, 3);
    }

    @Test
    public void testScatterMax() throws TornadoExecutionPlanException {
        scatterPoints(MlxIndex::scatterMax, Math::max, false, -3, 3);
    }

    @Test
    public void testScatterMin() throws TornadoExecutionPlanException {
        scatterPoints(MlxIndex::scatterMin, Math::min, false, -3, 3);
    }

    @Test
    public void testScatterProd() throws TornadoExecutionPlanException {
        scatterPoints(MlxIndex::scatterProd, (old, u) -> old * u, false, 0.5f, 1.5f);
    }

    @Test
    public void testScatterRows() throws TornadoExecutionPlanException {
        scatterRows(MlxIndex::scatterRows, (old, u) -> u, false, -3, 3);
    }

    @Test
    public void testScatterAddRows() throws TornadoExecutionPlanException {
        scatterRows(MlxIndex::scatterAddRows, Double::sum, true, -3, 3);
    }

    @Test
    public void testScatterMaxRows() throws TornadoExecutionPlanException {
        scatterRows(MlxIndex::scatterMaxRows, Math::max, false, -3, 3);
    }

    @Test
    public void testScatterMinRows() throws TornadoExecutionPlanException {
        scatterRows(MlxIndex::scatterMinRows, Math::min, false, -3, 3);
    }

    @Test
    public void testScatterProdRows() throws TornadoExecutionPlanException {
        scatterRows(MlxIndex::scatterProdRows, (old, u) -> old * u, false, 0.5f, 1.5f);
    }

    @Test
    public void testMaskedScatter() throws TornadoExecutionPlanException {
        // The masked positions of x, in order, take consecutive values of src.
        final int n = 1000;
        float[] xv = values(n, -10, 10, 33);
        float[] sv = values(n, 100, 200, 34);
        Random random = new Random(35);
        boolean[] mv = new boolean[n];
        ByteArray mask = new ByteArray(n);
        for (int i = 0; i < n; i++) {
            mv[i] = random.nextInt(10) < 3;
            mask.set(i, (byte) (mv[i] ? 1 : 0));
        }
        FloatArray x = FloatArray.fromArray(xv);
        FloatArray src = FloatArray.fromArray(sv);
        FloatArray output = new FloatArray(n);

        TaskGraph taskGraph = new TaskGraph("g") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, x, mask, src) //
                .libraryTask("maskedScatter", MlxIndex::maskedScatter, x, mask, src, output) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
            plan.execute();
        }

        int k = 0;
        for (int i = 0; i < n; i++) {
            assertEquals("element " + i, mv[i] ? sv[k++] : xv[i], output.get(i), 0f);
        }
    }

    // ---------------------------------------------------------------- dynamic slices and min/max updates

    @Test
    public void testSliceRowsAt() throws TornadoExecutionPlanException {
        // 7 rows from the row held in a device array, so one graph serves any start.
        final int sliceRows = 7;
        float[] xv = values(ROWS * COLS, -10, 10, 36);
        FloatArray x = FloatArray.fromArray(xv);
        IntArray start = IntArray.fromElements(11);
        FloatArray output = new FloatArray(sliceRows * COLS);

        TaskGraph taskGraph = new TaskGraph("g") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, x, start) //
                .libraryTask("sliceRowsAt", MlxIndex::sliceRowsAt, x, start, output, ROWS, COLS, sliceRows) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
            plan.execute();
        }

        for (int i = 0; i < sliceRows * COLS; i++) {
            assertEquals("element " + i, xv[11 * COLS + i], output.get(i), 0f);
        }
    }

    @Test
    public void testSliceUpdateRowsAt() throws TornadoExecutionPlanException {
        final int updateRows = 6;
        float[] xv = values(ROWS * COLS, -10, 10, 37);
        float[] uv = values(updateRows * COLS, -10, 10, 38);
        FloatArray x = FloatArray.fromArray(xv);
        FloatArray update = FloatArray.fromArray(uv);
        IntArray start = IntArray.fromElements(13);
        FloatArray output = new FloatArray(xv.length);

        TaskGraph taskGraph = new TaskGraph("g") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, x, update, start) //
                .libraryTask("sliceUpdateRowsAt", MlxIndex::sliceUpdateRowsAt, x, update, start, output, ROWS, COLS, updateRows) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
            plan.execute();
        }

        float[] expected = xv.clone();
        System.arraycopy(uv, 0, expected, 13 * COLS, uv.length);
        assertEach(expected, output, 0);
    }

    @Test
    public void testSliceUpdateMax() throws TornadoExecutionPlanException {
        sliceUpdate(MlxIndex::sliceUpdateMax, Math::max);
    }

    @Test
    public void testSliceUpdateMin() throws TornadoExecutionPlanException {
        sliceUpdate(MlxIndex::sliceUpdateMin, Math::min);
    }
}
