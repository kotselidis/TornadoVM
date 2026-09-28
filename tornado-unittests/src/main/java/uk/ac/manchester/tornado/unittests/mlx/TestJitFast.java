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

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.mlx.Mlx;
import uk.ac.manchester.tornado.mlx.jit.JitFast;

/**
 * The KernelContext JIT counterparts of the MLX {@code mlx.fast} operations, run in the same graph
 * as the MLX task and checked against a Java reference and against MLX.
 *
 * <p>
 * How to run?
 * </p>
 * <code>
 * tornado-test -V uk.ac.manchester.tornado.unittests.mlx.TestJitFast
 * </code>
 */
public class TestJitFast extends MlxTestBase {

    private static WorkerGrid1D perRow(int rows, int threads) {
        WorkerGrid1D grid = new WorkerGrid1D(rows * threads);
        grid.setLocalWork(threads, 1, 1);
        return grid;
    }

    private static void both(String name, double[] expected, FloatArray jit, FloatArray mlx, double rel, double abs) {
        assertAllClose(name + " JIT", expected, jit, rel, abs);
        for (int i = 0; i < expected.length; i++) {
            assertClose(name + " JIT vs MLX", i, mlx.get(i), jit.get(i), rel, abs);
        }
    }

    @Test
    public void testRmsNorm() throws TornadoExecutionPlanException {
        for (int[] s : new int[][] { { 1, 4096 }, { 7, 1000 }, { 3, 8192 }, { 64, 2048 } }) {
            int rows = s[0], dim = s[1];
            float[] xv = values(rows * dim, -2, 2, rows);
            float[] wv = values(dim, 0.5f, 1.5f, dim);
            FloatArray x = FloatArray.fromArray(xv);
            FloatArray w = FloatArray.fromArray(wv);
            FloatArray outMlx = new FloatArray(rows * dim);
            FloatArray outJit = new FloatArray(rows * dim);
            TaskGraph graph = new TaskGraph("rms").transferToDevice(DataTransferMode.FIRST_EXECUTION, x, w) //
                    .libraryTask("mlx", Mlx::rmsNorm, x, w, outMlx, rows, dim, 1e-5f) //
                    .task("jit", JitFast::rmsNorm, new KernelContext(), x, w, outJit, dim, 1e-5f) //
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, outMlx, outJit);
            try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
                plan.withGridScheduler(new GridScheduler("rms.jit", perRow(rows, JitFast.NORM_THREADS))).execute();
            }
            double[] expected = new double[rows * dim];
            for (int r = 0; r < rows; r++) {
                double ss = 0;
                for (int i = 0; i < dim; i++) {
                    ss += (double) xv[r * dim + i] * xv[r * dim + i];
                }
                double inv = 1.0 / Math.sqrt(ss / dim + 1e-5);
                for (int i = 0; i < dim; i++) {
                    expected[r * dim + i] = xv[r * dim + i] * inv * wv[i];
                }
            }
            both("rmsNorm " + rows + "x" + dim, expected, outJit, outMlx, 1e-5, 1e-6);
        }
    }

    @Test
    public void testLayerNorm() throws TornadoExecutionPlanException {
        final int rows = 5, dim = 777;
        float[] xv = values(rows * dim, -3, 3, 1);
        float[] wv = values(dim, 0.5f, 1.5f, 2);
        float[] bv = values(dim, -0.5f, 0.5f, 3);
        FloatArray x = FloatArray.fromArray(xv);
        FloatArray w = FloatArray.fromArray(wv);
        FloatArray b = FloatArray.fromArray(bv);
        FloatArray outMlx = new FloatArray(rows * dim);
        FloatArray outJit = new FloatArray(rows * dim);
        TaskGraph graph = new TaskGraph("ln").transferToDevice(DataTransferMode.FIRST_EXECUTION, x, w, b) //
                .libraryTask("mlx", Mlx::layerNorm, x, w, b, outMlx, rows, dim, 1e-5f) //
                .task("jit", JitFast::layerNorm, new KernelContext(), x, w, b, outJit, dim, 1e-5f) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, outMlx, outJit);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(new GridScheduler("ln.jit", perRow(rows, JitFast.NORM_THREADS))).execute();
        }
        double[] expected = new double[rows * dim];
        for (int r = 0; r < rows; r++) {
            double mean = 0;
            for (int i = 0; i < dim; i++) {
                mean += xv[r * dim + i];
            }
            mean /= dim;
            double var = 0;
            for (int i = 0; i < dim; i++) {
                var += (xv[r * dim + i] - mean) * (xv[r * dim + i] - mean);
            }
            double inv = 1.0 / Math.sqrt(var / dim + 1e-5);
            for (int i = 0; i < dim; i++) {
                expected[r * dim + i] = (xv[r * dim + i] - mean) * inv * wv[i] + bv[i];
            }
        }
        both("layerNorm", expected, outJit, outMlx, 1e-4, 1e-5);
    }

    private static double[] ropeRef(float[] x, int heads, int seqLen, int headDim, int dims, boolean traditional, float base, float scale, int offset) {
        double[] out = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            out[i] = x[i];
        }
        int half = dims / 2;
        for (int row = 0; row < heads * seqLen; row++) {
            double pos = (offset + row % seqLen) * (double) scale;
            for (int i = 0; i < half; i++) {
                double theta = pos * Math.pow(base, -2.0 * i / dims);
                int i1 = row * headDim + (traditional ? 2 * i : i);
                int i2 = row * headDim + (traditional ? 2 * i + 1 : i + half);
                double x1 = x[i1], x2 = x[i2];
                out[i1] = x1 * Math.cos(theta) - x2 * Math.sin(theta);
                out[i2] = x1 * Math.sin(theta) + x2 * Math.cos(theta);
            }
        }
        return out;
    }

    private static void rope(boolean traditional, int dims) throws TornadoExecutionPlanException {
        final int heads = 4, seqLen = 3, headDim = 64, offset = 5;
        final float base = 10000f, scale = 1f;
        final int rows = heads * seqLen;
        float[] xv = values(rows * headDim, -1, 1, dims + (traditional ? 1 : 0));
        FloatArray x = FloatArray.fromArray(xv);
        FloatArray outMlx = new FloatArray(xv.length);
        FloatArray outJit = new FloatArray(xv.length);
        TaskGraph graph = new TaskGraph("rope").transferToDevice(DataTransferMode.FIRST_EXECUTION, x) //
                .libraryTask("mlx", Mlx::rope, x, outMlx, 1, heads, seqLen, headDim, dims, traditional, base, scale, offset) //
                .task("jit", JitFast::rope, new KernelContext(), x, outJit, rows, seqLen, headDim, dims, traditional ? 1 : 0, base, scale, offset) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, outMlx, outJit);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(new GridScheduler("rope.jit", perRow(rows, headDim / 2))).execute();
        }
        both("rope traditional=" + traditional + " dims=" + dims, ropeRef(xv, heads, seqLen, headDim, dims, traditional, base, scale, offset), outJit, outMlx, 1e-4, 1e-5);
    }

    @Test
    public void testRope() throws TornadoExecutionPlanException {
        rope(false, 64);
        rope(true, 64);
        rope(false, 32);
    }

    @Test
    public void testRopeDynamic() throws TornadoExecutionPlanException {
        final int heads = 8, seqLen = 1, headDim = 128, dims = 128;
        final float base = 500000f;
        float[] xv = values(heads * headDim, -1, 1, 7);
        FloatArray x = FloatArray.fromArray(xv);
        IntArray offset = new IntArray(1);
        FloatArray outMlx = new FloatArray(xv.length);
        FloatArray outJit = new FloatArray(xv.length);
        TaskGraph graph = new TaskGraph("ropeDyn").transferToDevice(DataTransferMode.FIRST_EXECUTION, x) //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, offset) //
                .libraryTask("mlx", Mlx::ropeDynamic, x, offset, outMlx, 1, heads, seqLen, headDim, dims, false, base, 1f) //
                .task("jit", JitFast::ropeDynamic, new KernelContext(), x, offset, outJit, heads, seqLen, headDim, dims, 0, base, 1f) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, outMlx, outJit);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(new GridScheduler("ropeDyn.jit", perRow(heads, headDim / 2)));
            for (int position : new int[] { 0, 7, 123 }) {
                offset.set(0, position);
                plan.execute();
                both("ropeDynamic position " + position, ropeRef(xv, heads, seqLen, headDim, dims, false, base, 1f, position), outJit, outMlx, 1e-4, 1e-5);
            }
        }
    }

    private static double[] attentionRef(float[] q, float[] k, float[] v, int qHeads, int kvHeads, int kvLen, int headDim, float scale) {
        double[] out = new double[qHeads * headDim];
        int group = qHeads / kvHeads;
        for (int h = 0; h < qHeads; h++) {
            int kh = h / group;
            double[] s = new double[kvLen];
            double max = Double.NEGATIVE_INFINITY;
            for (int j = 0; j < kvLen; j++) {
                double dot = 0;
                for (int d = 0; d < headDim; d++) {
                    dot += (double) q[h * headDim + d] * k[(kh * kvLen + j) * headDim + d];
                }
                s[j] = dot * scale;
                max = Math.max(max, s[j]);
            }
            double sum = 0;
            for (int j = 0; j < kvLen; j++) {
                s[j] = Math.exp(s[j] - max);
                sum += s[j];
            }
            for (int d = 0; d < headDim; d++) {
                double acc = 0;
                for (int j = 0; j < kvLen; j++) {
                    acc += s[j] / sum * v[(kh * kvLen + j) * headDim + d];
                }
                out[h * headDim + d] = acc;
            }
        }
        return out;
    }

    private static void attention(int qHeads, int kvHeads, int kvLen, int headDim) throws TornadoExecutionPlanException {
        final float scale = (float) (1.0 / Math.sqrt(headDim));
        float[] qv = values(qHeads * headDim, -1, 1, 11);
        float[] kv = values(kvHeads * kvLen * headDim, -1, 1, 12);
        float[] vv = values(kvHeads * kvLen * headDim, -1, 1, 13);
        FloatArray q = FloatArray.fromArray(qv);
        FloatArray k = FloatArray.fromArray(kv);
        FloatArray v = FloatArray.fromArray(vv);
        FloatArray outMlx = new FloatArray(qHeads * headDim);
        FloatArray outJit = new FloatArray(qHeads * headDim);
        TaskGraph graph = new TaskGraph("sdpa").transferToDevice(DataTransferMode.FIRST_EXECUTION, q, k, v) //
                .libraryTask("mlx", Mlx::scaledDotProductAttention, q, k, v, outMlx, 1, qHeads, kvHeads, 1, kvLen, headDim, scale, false) //
                .task("jit", JitFast::attentionDecode, new KernelContext(), q, k, v, outJit, qHeads, kvHeads, kvLen, headDim, scale) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, outMlx, outJit);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(new GridScheduler("sdpa.jit", perRow(qHeads, JitFast.ATTENTION_THREADS))).execute();
        }
        both("attention " + qHeads + "/" + kvHeads + " heads, ctx " + kvLen + ", dim " + headDim, attentionRef(qv, kv, vv, qHeads, kvHeads, kvLen, headDim, scale), outJit, outMlx, 1e-4,
                1e-5);
    }

    @Test
    public void testAttentionDecode() throws TornadoExecutionPlanException {
        attention(8, 2, 17, 64);
        attention(32, 8, 1000, 128);
    }
}
