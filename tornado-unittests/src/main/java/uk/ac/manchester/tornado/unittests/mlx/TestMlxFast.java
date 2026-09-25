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

import org.junit.Test;

import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.BFloat16Array;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.mlx.Mlx;

/**
 * MLX {@code mlx.fast} library tasks (normalisation, RoPE, attention), checked against a Java
 * reference.
 *
 * <p>
 * How to run?
 * </p>
 * <code>
 * tornado-test -V uk.ac.manchester.tornado.unittests.mlx.TestMlxFast
 * </code>
 */
public class TestMlxFast extends MlxTestBase {

    // ---------------------------------------------------------------- normalisation

    private static double[] rmsNormRef(float[] x, float[] w, int rows, int dim, float eps) {
        double[] out = new double[rows * dim];
        for (int r = 0; r < rows; r++) {
            double ss = 0;
            for (int i = 0; i < dim; i++) {
                ss += (double) x[r * dim + i] * x[r * dim + i];
            }
            double inv = 1.0 / Math.sqrt(ss / dim + eps);
            for (int i = 0; i < dim; i++) {
                out[r * dim + i] = x[r * dim + i] * inv * w[i];
            }
        }
        return out;
    }

    @Test
    public void testRmsNorm() throws TornadoExecutionPlanException {
        for (int[] shape : new int[][] { { 1, 4096 }, { 7, 1000 }, { 3, 8192 } }) {
            int rows = shape[0], dim = shape[1];
            float[] xv = values(rows * dim, -2, 2, rows);
            float[] wv = values(dim, 0.5f, 1.5f, dim);
            FloatArray x = FloatArray.fromArray(xv);
            FloatArray w = FloatArray.fromArray(wv);
            FloatArray out = new FloatArray(rows * dim);
            run(new TaskGraph("rms").transferToDevice(DataTransferMode.FIRST_EXECUTION, x, w).libraryTask("t", Mlx::rmsNorm, x, w, out, rows, dim, 1e-5f)
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, out));
            assertAllClose("rmsNorm float32 " + rows + "x" + dim, rmsNormRef(xv, wv, rows, dim, 1e-5f), out, 1e-5, 1e-6);
        }
        final int rows = 4, dim = 2048;
        HalfFloatArray hx = half(values(rows * dim, -2, 2, 71));
        HalfFloatArray hw = half(values(dim, 0.5f, 1.5f, 72));
        HalfFloatArray hout = new HalfFloatArray(rows * dim);
        run(new TaskGraph("rms16").transferToDevice(DataTransferMode.FIRST_EXECUTION, hx, hw).libraryTask("t", Mlx::rmsNorm, hx, hw, hout, rows, dim, 1e-5f)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, hout));
        assertAllClose("rmsNorm float16", rmsNormRef(widen(hx), widen(hw), rows, dim, 1e-5f), hout, 3e-3, 3e-3);
    }

    @Test
    public void testLayerNorm() throws TornadoExecutionPlanException {
        final int rows = 5, dim = 777;
        final float eps = 1e-5f;
        float[] xv = values(rows * dim, -3, 3, 73);
        float[] wv = values(dim, 0.5f, 1.5f, 74);
        float[] bv = values(dim, -0.5f, 0.5f, 75);
        FloatArray x = FloatArray.fromArray(xv);
        FloatArray w = FloatArray.fromArray(wv);
        FloatArray b = FloatArray.fromArray(bv);
        FloatArray out = new FloatArray(rows * dim);
        run(new TaskGraph("ln").transferToDevice(DataTransferMode.FIRST_EXECUTION, x, w, b).libraryTask("t", Mlx::layerNorm, x, w, b, out, rows, dim, eps)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, out));
        double[] expected = new double[rows * dim];
        for (int r = 0; r < rows; r++) {
            double mean = 0;
            for (int i = 0; i < dim; i++) {
                mean += xv[r * dim + i];
            }
            mean /= dim;
            double var = 0;
            for (int i = 0; i < dim; i++) {
                double d = xv[r * dim + i] - mean;
                var += d * d;
            }
            var /= dim;
            for (int i = 0; i < dim; i++) {
                expected[r * dim + i] = (xv[r * dim + i] - mean) / Math.sqrt(var + eps) * wv[i] + bv[i];
            }
        }
        assertAllClose("layerNorm float32", expected, out, 1e-4, 1e-5);
    }

    // ---------------------------------------------------------------- RoPE

    /** RoPE of x[batch, heads, seqLen, headDim] at positions offset + l. */
    private static double[] ropeRef(float[] x, int batch, int heads, int seqLen, int headDim, int dims, boolean traditional, float base, float scale, int offset) {
        double[] out = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            out[i] = x[i];
        }
        int half = dims / 2;
        for (int b = 0; b < batch; b++) {
            for (int h = 0; h < heads; h++) {
                for (int l = 0; l < seqLen; l++) {
                    int row = ((b * heads + h) * seqLen + l) * headDim;
                    double pos = (offset + l) * (double) scale;
                    for (int i = 0; i < half; i++) {
                        double theta = pos * Math.pow(base, -2.0 * i / dims);
                        int i1 = traditional ? 2 * i : i;
                        int i2 = traditional ? 2 * i + 1 : i + half;
                        double x1 = x[row + i1], x2 = x[row + i2];
                        out[row + i1] = x1 * Math.cos(theta) - x2 * Math.sin(theta);
                        out[row + i2] = x1 * Math.sin(theta) + x2 * Math.cos(theta);
                    }
                }
            }
        }
        return out;
    }

    private static void rope(boolean traditional, int dims) throws TornadoExecutionPlanException {
        final int batch = 1, heads = 4, seqLen = 3, headDim = 64, offset = 5;
        final float base = 10000f, scale = 1f;
        float[] xv = values(batch * heads * seqLen * headDim, -1, 1, dims + (traditional ? 1 : 0));
        FloatArray x = FloatArray.fromArray(xv);
        FloatArray out = new FloatArray(xv.length);
        run(new TaskGraph("rope").transferToDevice(DataTransferMode.FIRST_EXECUTION, x) //
                .libraryTask("t", Mlx::rope, x, out, batch, heads, seqLen, headDim, dims, traditional, base, scale, offset) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, out));
        assertAllClose("rope traditional=" + traditional + " dims=" + dims, ropeRef(xv, batch, heads, seqLen, headDim, dims, traditional, base, scale, offset), out, 1e-4, 1e-5);
    }

    @Test
    public void testRope() throws TornadoExecutionPlanException {
        rope(false, 64);
        rope(true, 64);
        rope(false, 32);
    }

    @Test
    public void testRopeBFloat16() throws TornadoExecutionPlanException {
        final int batch = 1, heads = 8, seqLen = 1, headDim = 128, dims = 128, offset = 100;
        BFloat16Array x = bf16(values(batch * heads * seqLen * headDim, -1, 1, 81));
        BFloat16Array out = new BFloat16Array(x.getSize());
        run(new TaskGraph("rope16").transferToDevice(DataTransferMode.FIRST_EXECUTION, x) //
                .libraryTask("t", Mlx::rope, x, out, batch, heads, seqLen, headDim, dims, false, 500000f, 1f, offset) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, out));
        assertAllClose("rope bfloat16", ropeRef(widen(x), batch, heads, seqLen, headDim, dims, false, 500000f, 1f, offset), out, 1.6e-2, 1.6e-2);
    }

    /** The position comes from a device IntArray, so one graph serves every decode step. */
    @Test
    public void testRopeDynamic() throws TornadoExecutionPlanException {
        final int batch = 1, heads = 2, seqLen = 1, headDim = 64, dims = 64;
        final float base = 10000f;
        float[] xv = values(batch * heads * seqLen * headDim, -1, 1, 82);
        FloatArray x = FloatArray.fromArray(xv);
        IntArray offset = new IntArray(1);
        FloatArray out = new FloatArray(xv.length);
        TaskGraph graph = new TaskGraph("ropeDyn").transferToDevice(DataTransferMode.FIRST_EXECUTION, x) //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, offset) //
                .libraryTask("t", Mlx::ropeDynamic, x, offset, out, batch, heads, seqLen, headDim, dims, false, base, 1f) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, out);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            for (int position : new int[] { 0, 7, 123 }) {
                offset.set(0, position);
                plan.execute();
                assertAllClose("ropeDynamic position " + position, ropeRef(xv, batch, heads, seqLen, headDim, dims, false, base, 1f, position), out, 1e-4, 1e-5);
            }
        }
    }

    // ---------------------------------------------------------------- attention

    /** softmax(q k^T * scale) v per head, with grouped-query heads and an optional causal mask aligned to the end. */
    private static double[] attentionRef(float[] q, float[] k, float[] v, int batch, int qHeads, int kvHeads, int qLen, int kvLen, int headDim, float scale, boolean causal) {
        double[] out = new double[batch * qHeads * qLen * headDim];
        int group = qHeads / kvHeads;
        for (int b = 0; b < batch; b++) {
            for (int h = 0; h < qHeads; h++) {
                int kh = h / group;
                for (int i = 0; i < qLen; i++) {
                    double[] scores = new double[kvLen];
                    double max = Double.NEGATIVE_INFINITY;
                    for (int j = 0; j < kvLen; j++) {
                        if (causal && j > i + (kvLen - qLen)) {
                            scores[j] = Double.NEGATIVE_INFINITY;
                            continue;
                        }
                        double dot = 0;
                        for (int d = 0; d < headDim; d++) {
                            dot += (double) q[((b * qHeads + h) * qLen + i) * headDim + d] * k[((b * kvHeads + kh) * kvLen + j) * headDim + d];
                        }
                        scores[j] = dot * scale;
                        max = Math.max(max, scores[j]);
                    }
                    double sum = 0;
                    for (int j = 0; j < kvLen; j++) {
                        scores[j] = Double.isInfinite(scores[j]) ? 0 : Math.exp(scores[j] - max);
                        sum += scores[j];
                    }
                    for (int d = 0; d < headDim; d++) {
                        double acc = 0;
                        for (int j = 0; j < kvLen; j++) {
                            acc += scores[j] / sum * v[((b * kvHeads + kh) * kvLen + j) * headDim + d];
                        }
                        out[((b * qHeads + h) * qLen + i) * headDim + d] = acc;
                    }
                }
            }
        }
        return out;
    }

    private static void attention(int qLen, boolean causal) throws TornadoExecutionPlanException {
        final int batch = 1, qHeads = 8, kvHeads = 2, kvLen = 17, headDim = 64;
        final float scale = (float) (1.0 / Math.sqrt(headDim));
        float[] qv = values(batch * qHeads * qLen * headDim, -1, 1, 91L + qLen);
        float[] kv = values(batch * kvHeads * kvLen * headDim, -1, 1, 92);
        float[] vv = values(batch * kvHeads * kvLen * headDim, -1, 1, 93);
        FloatArray q = FloatArray.fromArray(qv);
        FloatArray k = FloatArray.fromArray(kv);
        FloatArray v = FloatArray.fromArray(vv);
        FloatArray out = new FloatArray(qv.length);
        run(new TaskGraph("sdpa").transferToDevice(DataTransferMode.FIRST_EXECUTION, q, k, v) //
                .libraryTask("t", Mlx::scaledDotProductAttention, q, k, v, out, batch, qHeads, kvHeads, qLen, kvLen, headDim, scale, causal) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, out));
        assertAllClose("sdpa qLen=" + qLen + " causal=" + causal, attentionRef(qv, kv, vv, batch, qHeads, kvHeads, qLen, kvLen, headDim, scale, causal), out, 1e-4, 1e-5);
    }

    @Test
    public void testScaledDotProductAttentionDecode() throws TornadoExecutionPlanException {
        attention(1, false);
    }

    @Test
    public void testScaledDotProductAttentionPrefill() throws TornadoExecutionPlanException {
        attention(5, false);
        attention(5, true);
    }
}
