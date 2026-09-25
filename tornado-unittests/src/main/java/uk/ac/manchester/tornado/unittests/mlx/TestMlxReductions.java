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

import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.mlx.Mlx;

/**
 * MLX softmax, argmax and top-k library tasks, checked against a Java reference.
 *
 * <p>
 * How to run?
 * </p>
 * <code>
 * tornado-test -V uk.ac.manchester.tornado.unittests.mlx.TestMlxReductions
 * </code>
 */
public class TestMlxReductions extends MlxTestBase {

    /** Softmax of each consecutive group of {@code group} values. */
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

    @Test
    public void testSoftmax() throws TornadoExecutionPlanException {
        float[] xv = values(1000, -5, 5, 101);
        FloatArray x = FloatArray.fromArray(xv);
        FloatArray out = new FloatArray(xv.length);
        run(new TaskGraph("sm").transferToDevice(DataTransferMode.FIRST_EXECUTION, x).libraryTask("t", Mlx::softmax, x, out).transferToHost(DataTransferMode.EVERY_EXECUTION, out));
        assertAllClose("softmax float32", softmaxRef(xv, xv.length), out, 1e-5, 1e-8);

        HalfFloatArray hx = half(values(1000, -5, 5, 102));
        HalfFloatArray hout = new HalfFloatArray(1000);
        run(new TaskGraph("sm16").transferToDevice(DataTransferMode.FIRST_EXECUTION, hx).libraryTask("t", Mlx::softmax, hx, hout).transferToHost(DataTransferMode.EVERY_EXECUTION, hout));
        assertAllClose("softmax float16", softmaxRef(widen(hx), 1000), hout, 2e-3, 1e-6);
    }

    @Test
    public void testSoftmaxRows() throws TornadoExecutionPlanException {
        final int rows = 5, cols = 333;
        float[] xv = values(rows * cols, -8, 8, 103);
        FloatArray x = FloatArray.fromArray(xv);
        FloatArray out = new FloatArray(xv.length);
        run(new TaskGraph("smr").transferToDevice(DataTransferMode.FIRST_EXECUTION, x).libraryTask("t", Mlx::softmaxRows, x, out, rows, cols)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, out));
        assertAllClose("softmaxRows float32", softmaxRef(xv, cols), out, 1e-5, 1e-8);
    }

    @Test
    public void testSoftmaxLastTwoAxes() throws TornadoExecutionPlanException {
        final int d0 = 3, d1 = 4, d2 = 50;
        float[] xv = values(d0 * d1 * d2, -4, 4, 104);
        FloatArray x = FloatArray.fromArray(xv);
        FloatArray out = new FloatArray(xv.length);
        run(new TaskGraph("sma").transferToDevice(DataTransferMode.FIRST_EXECUTION, x).libraryTask("t", Mlx::softmaxLastTwoAxes, x, out, d0, d1, d2)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, out));
        assertAllClose("softmaxLastTwoAxes float32", softmaxRef(xv, d1 * d2), out, 1e-5, 1e-8);
    }

    private static int argmax(float[] x, int from, int to) {
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
        float[] xv = values(151936, -10, 10, 105);
        xv[98765] = 42.0f;
        FloatArray x = FloatArray.fromArray(xv);
        IntArray out = new IntArray(1);
        run(new TaskGraph("am").transferToDevice(DataTransferMode.FIRST_EXECUTION, x).libraryTask("t", Mlx::argmax, x, out).transferToHost(DataTransferMode.EVERY_EXECUTION, out));
        assertEquals("argmax over a vocabulary-sized array", 98765, out.get(0));
    }

    @Test
    public void testArgmaxRows() throws TornadoExecutionPlanException {
        final int rows = 6, cols = 1001;
        float[] xv = values(rows * cols, -10, 10, 106);
        FloatArray x = FloatArray.fromArray(xv);
        IntArray out = new IntArray(rows);
        run(new TaskGraph("amr").transferToDevice(DataTransferMode.FIRST_EXECUTION, x).libraryTask("t", Mlx::argmaxRows, x, out, rows, cols)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, out));
        for (int r = 0; r < rows; r++) {
            assertEquals("argmaxRows row " + r, argmax(xv, r * cols, (r + 1) * cols), out.get(r));
        }
    }

    /** The k largest values of x[from, to), ascending. */
    private static float[] topk(float[] x, int from, int to, int k) {
        float[] sorted = Arrays.copyOfRange(x, from, to);
        Arrays.sort(sorted);
        return Arrays.copyOfRange(sorted, sorted.length - k, sorted.length);
    }

    @Test
    public void testTopk() throws TornadoExecutionPlanException {
        final int k = 10;
        float[] xv = values(5000, -10, 10, 107);
        FloatArray x = FloatArray.fromArray(xv);
        FloatArray out = new FloatArray(k);
        run(new TaskGraph("tk").transferToDevice(DataTransferMode.FIRST_EXECUTION, x).libraryTask("t", Mlx::topk, x, out, k).transferToHost(DataTransferMode.EVERY_EXECUTION, out));
        float[] got = out.toHeapArray();
        Arrays.sort(got);
        assertArrayEquals("topk values", topk(xv, 0, xv.length, k), got, 0.0f);
    }

    @Test
    public void testTopkRows() throws TornadoExecutionPlanException {
        final int rows = 4, cols = 777, k = 5;
        float[] xv = values(rows * cols, -10, 10, 108);
        FloatArray x = FloatArray.fromArray(xv);
        FloatArray out = new FloatArray(rows * k);
        run(new TaskGraph("tkr").transferToDevice(DataTransferMode.FIRST_EXECUTION, x).libraryTask("t", Mlx::topkRows, x, out, rows, cols, k)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, out));
        for (int r = 0; r < rows; r++) {
            float[] got = Arrays.copyOfRange(out.toHeapArray(), r * k, (r + 1) * k);
            Arrays.sort(got);
            assertArrayEquals("topkRows row " + r, topk(xv, r * cols, (r + 1) * cols, k), got, 0.0f);
        }
    }
}
