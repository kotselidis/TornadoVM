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

import org.junit.Test;

import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.mlx.Mlx;

/**
 * MLX affine group quantization: quantize/dequantize round trips, quantized matmul and the
 * mixture-of-experts gather variant. The reference decodes MLX's packed format independently
 * (values low bits first in uint32 words, {@code w = scale * q + bias} per group).
 *
 * <p>
 * How to run?
 * </p>
 * <code>
 * tornado-test -V uk.ac.manchester.tornado.unittests.mlx.TestMlxQuantized
 * </code>
 */
public class TestMlxQuantized extends MlxTestBase {

    /** Decodes packed weights into w[rows, cols] (bits must divide 32). */
    private static double[] decode(IntArray wq, float[] scales, float[] biases, int rows, int cols, int groupSize, int bits) {
        int perWord = 32 / bits;
        int mask = (1 << bits) - 1;
        int wordsPerRow = cols / perWord;
        double[] w = new double[rows * cols];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                int word = wq.get(r * wordsPerRow + c / perWord);
                int q = (word >>> (bits * (c % perWord))) & mask;
                int g = r * (cols / groupSize) + c / groupSize;
                w[r * cols + c] = scales[g] * q + biases[g];
            }
        }
        return w;
    }

    private static float[] toFloats(FloatArray a) {
        return a.toHeapArray();
    }

    private static void roundTrip(int rows, int cols, int groupSize, int bits) throws TornadoExecutionPlanException {
        float[] wv = values(rows * cols, -1, 1, 31L * bits + groupSize);
        FloatArray w = FloatArray.fromArray(wv);
        IntArray wq = new IntArray(rows * cols * bits / 32);
        FloatArray scales = new FloatArray(rows * cols / groupSize);
        FloatArray biases = new FloatArray(rows * cols / groupSize);
        FloatArray back = new FloatArray(rows * cols);
        run(new TaskGraph("q").transferToDevice(DataTransferMode.FIRST_EXECUTION, w) //
                .libraryTask("quantize", Mlx::quantize, w, wq, scales, biases, rows, cols, groupSize, bits) //
                .libraryTask("dequantize", Mlx::dequantize, wq, scales, biases, back, rows, cols, groupSize, bits) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, wq, scales, biases, back));

        String what = "bits=" + bits + " group=" + groupSize;
        double[] decoded = decode(wq, toFloats(scales), toFloats(biases), rows, cols, groupSize, bits);
        // MLX's dequantize agrees with an independent decoder of its packed format...
        assertAllClose("dequantize vs decoder " + what, decoded, back, 1e-6, 1e-6);
        // ...and quantization error stays within one quantization step. (Not half a step: MLX adjusts
        // each group's scale so that zero is exactly representable, which can push the far edge of
        // the group up to one step away.)
        for (int i = 0; i < wv.length; i++) {
            float step = Math.abs(scales.get((i / cols) * (cols / groupSize) + (i % cols) / groupSize));
            assertTrue("round trip " + what + " element " + i + ": " + wv[i] + " vs " + back.get(i), Math.abs(wv[i] - back.get(i)) <= step + 1e-5);
        }
    }

    @Test
    public void testQuantizeRoundTrip4Bit() throws TornadoExecutionPlanException {
        roundTrip(64, 256, 32, 4);
        roundTrip(3, 128, 64, 4);
    }

    @Test
    public void testQuantizeRoundTrip8Bit() throws TornadoExecutionPlanException {
        roundTrip(16, 256, 32, 8);
        roundTrip(5, 256, 128, 8);
    }

    @Test
    public void testQuantizeRoundTrip2Bit() throws TornadoExecutionPlanException {
        roundTrip(8, 128, 64, 2);
    }

    /** y[m, n] = x[m, k] @ w[n, k]^T. */
    private static double[] matmulTransposed(float[] x, double[] w, int m, int k, int n) {
        double[] y = new double[m * n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0;
                for (int p = 0; p < k; p++) {
                    sum += x[i * k + p] * w[j * k + p];
                }
                y[i * n + j] = sum;
            }
        }
        return y;
    }

    @Test
    public void testQuantizedMatmulFloat() throws TornadoExecutionPlanException {
        final int m = 3, k = 256, n = 40, groupSize = 32, bits = 4;
        float[] xv = values(m * k, -1, 1, 41);
        FloatArray x = FloatArray.fromArray(xv);
        FloatArray w = FloatArray.fromArray(values(n * k, -1, 1, 42));
        IntArray wq = new IntArray(n * k * bits / 32);
        FloatArray scales = new FloatArray(n * k / groupSize);
        FloatArray biases = new FloatArray(n * k / groupSize);
        FloatArray y = new FloatArray(m * n);
        run(new TaskGraph("qmm").transferToDevice(DataTransferMode.FIRST_EXECUTION, x, w) //
                .libraryTask("quantize", Mlx::quantize, w, wq, scales, biases, n, k, groupSize, bits) //
                .libraryTask("qmm", Mlx::quantizedMatmul, x, wq, scales, biases, y, m, k, n, groupSize, bits) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, wq, scales, biases, y));
        double[] wDeq = decode(wq, toFloats(scales), toFloats(biases), n, k, groupSize, bits);
        assertAllClose("quantizedMatmul float32", matmulTransposed(xv, wDeq, m, k, n), y, 1e-4, 1e-4);
    }

    @Test
    public void testQuantizedMatmulHalf() throws TornadoExecutionPlanException {
        final int m = 1, k = 512, n = 64, groupSize = 64, bits = 8;
        HalfFloatArray x = half(values(m * k, -1, 1, 43));
        HalfFloatArray w = half(values(n * k, -1, 1, 44));
        IntArray wq = new IntArray(n * k * bits / 32);
        HalfFloatArray scales = new HalfFloatArray(n * k / groupSize);
        HalfFloatArray biases = new HalfFloatArray(n * k / groupSize);
        HalfFloatArray y = new HalfFloatArray(m * n);
        run(new TaskGraph("qmm16").transferToDevice(DataTransferMode.FIRST_EXECUTION, x, w) //
                .libraryTask("quantize", Mlx::quantize, w, wq, scales, biases, n, k, groupSize, bits) //
                .libraryTask("qmm", Mlx::quantizedMatmul, x, wq, scales, biases, y, m, k, n, groupSize, bits) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, wq, scales, biases, y));
        double[] wDeq = decode(wq, widen(scales), widen(biases), n, k, groupSize, bits);
        assertAllClose("quantizedMatmul float16", matmulTransposed(widen(x), wDeq, m, k, n), y, 3e-3, 3e-2);
    }

    @Test
    public void testGatherQmm() throws TornadoExecutionPlanException {
        final int batches = 4, experts = 3, m = 2, k = 128, n = 16, groupSize = 32, bits = 4;
        float[] xv = values(batches * m * k, -1, 1, 51);
        FloatArray x = FloatArray.fromArray(xv);
        // w[experts, n, k] has the same layout as w[experts * n, k], so one quantize call covers all experts.
        FloatArray w = FloatArray.fromArray(values(experts * n * k, -1, 1, 52));
        IntArray wq = new IntArray(experts * n * k * bits / 32);
        FloatArray scales = new FloatArray(experts * n * k / groupSize);
        FloatArray biases = new FloatArray(experts * n * k / groupSize);
        IntArray lhs = IntArray.fromElements(0, 1, 2, 3);
        IntArray rhs = IntArray.fromElements(2, 0, 1, 2);
        FloatArray y = new FloatArray(batches * m * n);
        run(new TaskGraph("gqmm").transferToDevice(DataTransferMode.FIRST_EXECUTION, x, w, lhs, rhs) //
                .libraryTask("quantize", Mlx::quantize, w, wq, scales, biases, experts * n, k, groupSize, bits) //
                .libraryTask("gqmm", Mlx::gatherQmm, x, wq, scales, biases, lhs, rhs, y, batches, experts, m, k, n, groupSize, bits) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, wq, scales, biases, y));
        double[] wDeq = decode(wq, toFloats(scales), toFloats(biases), experts * n, k, groupSize, bits);
        double[] expected = new double[batches * m * n];
        for (int b = 0; b < batches; b++) {
            float[] xb = java.util.Arrays.copyOfRange(xv, lhs.get(b) * m * k, (lhs.get(b) + 1) * m * k);
            double[] we = java.util.Arrays.copyOfRange(wDeq, rhs.get(b) * n * k, (rhs.get(b) + 1) * n * k);
            System.arraycopy(matmulTransposed(xb, we, m, k, n), 0, expected, b * m * n, m * n);
        }
        assertAllClose("gatherQmm float32", expected, y, 1e-4, 1e-4);
    }
}
