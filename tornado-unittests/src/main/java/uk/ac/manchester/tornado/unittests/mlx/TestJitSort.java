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
import static org.junit.Assert.assertTrue;

import java.util.Arrays;

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
import uk.ac.manchester.tornado.mlx.MlxSort;
import uk.ac.manchester.tornado.mlx.jit.JitSort;

/**
 * The MLX sort, argsort, partition and argpartition, and their KernelContext JIT counterparts
 * (bitonic sorts), run in one graph on the same input: over a whole array that fits one
 * threadgroup, over a whole array that needs the global merge passes, and along an axis (rows and
 * strided slices). Sorted outputs must match Java's sort; partitions must place the kth element and
 * split the rest around it.
 *
 * <p>
 * How to run?
 * </p>
 * <code>
 * tornado-test -V uk.ac.manchester.tornado.unittests.mlx.TestJitSort
 * </code>
 */
public class TestJitSort extends MlxTestBase {

    /** {whole (1) or axis (0), outer, len, inner}. */
    private static final int[][] CASES = { { 1, 1, 1000, 1 }, { 1, 1, 100_000, 1 }, { 0, 7, 1500, 1 }, { 0, 4, 300, 9 } };

    /**
     * Adds the JIT tasks that sort x (viewed as [outer, len, inner] along the middle axis, or as a
     * whole array) into values and/or indices.
     */
    static void jitSort(TaskGraph g, GridScheduler gs, FloatArray x, FloatArray values, IntArray indices, int outer, int len, int inner, boolean whole, boolean writeValues,
            boolean writeIndices) {
        String name = g.getTaskGraphName();
        int wv = writeValues ? 1 : 0;
        int wi = writeIndices ? 1 : 0;
        if (!whole || len <= JitSort.BLOCK) {
            g.task("s", JitSort::sortSlices, new KernelContext(), x, values, indices, len, inner, TestJitReduce.nextPowerOfTwo(len), wv, wi);
            gs.addWorkerGrid(name + ".s", TestJitReduce.groups(outer * inner, JitSort.THREADS));
            return;
        }
        int padded = TestJitReduce.nextPowerOfTwo(len);
        FloatArray keys = new FloatArray(padded);
        IntArray idx = new IntArray(padded);
        g.task("pad", JitSort::pad, new KernelContext(), x, keys, idx, len, padded);
        gs.addWorkerGrid(name + ".pad", TestJitElementwise.grid1D(padded));
        g.task("blocks", JitSort::sortBlocks, new KernelContext(), keys, idx);
        gs.addWorkerGrid(name + ".blocks", TestJitReduce.groups(padded / JitSort.BLOCK, JitSort.THREADS));
        for (int k = 2 * JitSort.BLOCK; k <= padded; k <<= 1) {
            for (int j = k >> 1; j >= JitSort.BLOCK; j >>= 1) {
                String t = "g" + k + "_" + j;
                g.task(t, JitSort::mergeGlobal, new KernelContext(), keys, idx, padded, k, j);
                gs.addWorkerGrid(name + "." + t, TestJitElementwise.grid1D(padded / 2));
            }
            String t = "b" + k;
            g.task(t, JitSort::mergeBlocks, new KernelContext(), keys, idx, k);
            gs.addWorkerGrid(name + "." + t, TestJitReduce.groups(padded / JitSort.BLOCK, JitSort.THREADS));
        }
        g.task("unpad", JitSort::unpad, new KernelContext(), keys, idx, values, indices, len, wv, wi);
        gs.addWorkerGrid(name + ".unpad", TestJitElementwise.grid1D(len));
    }

    private static float[] slice(float[] v, int o, int j, int len, int inner) {
        float[] s = new float[len];
        for (int i = 0; i < len; i++) {
            s[i] = v[(o * len + i) * inner + j];
        }
        return s;
    }

    private static float[] slice(FloatArray v, int o, int j, int len, int inner) {
        return slice(v.toHeapArray(), o, j, len, inner);
    }

    /** x[o, idx[i], j] for the slice's indices. */
    private static float[] gather(float[] x, IntArray idx, int o, int j, int len, int inner) {
        float[] s = new float[len];
        boolean[] seen = new boolean[len];
        for (int i = 0; i < len; i++) {
            int k = idx.get((o * len + i) * inner + j);
            assertTrue("index " + k + " out of range", k >= 0 && k < len);
            assertTrue("index " + k + " repeated", !seen[k]);
            seen[k] = true;
            s[i] = x[(o * len + k) * inner + j];
        }
        return s;
    }

    private static void assertPartitioned(String what, float[] original, float[] got, int kth) {
        float[] sorted = original.clone();
        Arrays.sort(sorted);
        assertEquals(what + " kth element", sorted[kth], got[kth], 0f);
        for (int i = 0; i < got.length; i++) {
            assertTrue(what + " element " + i + " on the wrong side of kth", i < kth ? got[i] <= got[kth] : got[i] >= got[kth]);
        }
        float[] g = got.clone();
        Arrays.sort(g);
        assertArrayEquals(what + " is not a permutation", sorted, g, 0f);
    }

    private enum Kind {
        SORT, ARGSORT, PARTITION, ARGPARTITION
    }

    private static void check(Kind kind) throws TornadoExecutionPlanException {
        for (int[] c : CASES) {
            boolean whole = c[0] == 1;
            int outer = c[1];
            int len = c[2];
            int inner = c[3];
            int n = outer * len * inner;
            int kth = len / 3;
            float[] xv = values(n, -100, 100, 101L + n);
            FloatArray x = FloatArray.fromArray(xv);
            FloatArray valMlx = new FloatArray(n);
            FloatArray valJit = new FloatArray(n);
            IntArray idxMlx = new IntArray(n);
            IntArray idxJit = new IntArray(n);
            TaskGraph g = new TaskGraph("so").transferToDevice(DataTransferMode.FIRST_EXECUTION, x);
            switch (kind) {
                case SORT -> {
                    if (whole) {
                        g.libraryTask("mlx", MlxSort::sort, x, valMlx);
                    } else {
                        g.libraryTask("mlx", MlxSort::sortAxis, x, valMlx, outer, len, inner);
                    }
                }
                case ARGSORT -> {
                    if (whole) {
                        g.libraryTask("mlx", MlxSort::argsort, x, idxMlx);
                    } else {
                        g.libraryTask("mlx", MlxSort::argsortAxis, x, idxMlx, outer, len, inner);
                    }
                }
                case PARTITION -> {
                    if (whole) {
                        g.libraryTask("mlx", MlxSort::partition, x, valMlx, kth);
                    } else {
                        g.libraryTask("mlx", MlxSort::partitionAxis, x, valMlx, outer, len, inner, kth);
                    }
                }
                default -> {
                    if (whole) {
                        g.libraryTask("mlx", MlxSort::argpartition, x, idxMlx, kth);
                    } else {
                        g.libraryTask("mlx", MlxSort::argpartitionAxis, x, idxMlx, outer, len, inner, kth);
                    }
                }
            }
            boolean indices = kind == Kind.ARGSORT || kind == Kind.ARGPARTITION;
            GridScheduler gs = new GridScheduler();
            jitSort(g, gs, x, valJit, idxJit, outer, len, inner, whole, !indices, indices);
            g.transferToHost(DataTransferMode.EVERY_EXECUTION, valMlx, valJit, idxMlx, idxJit);
            try (TornadoExecutionPlan plan = new TornadoExecutionPlan(g.snapshot())) {
                plan.withGridScheduler(gs).execute();
            }
            for (int o = 0; o < outer; o++) {
                for (int j = 0; j < inner; j++) {
                    String what = kind + " " + Arrays.toString(c) + " slice " + o + "," + j;
                    float[] original = slice(xv, o, j, len, inner);
                    float[] sorted = original.clone();
                    Arrays.sort(sorted);
                    float[] mlx = indices ? gather(xv, idxMlx, o, j, len, inner) : slice(valMlx, o, j, len, inner);
                    float[] jit = indices ? gather(xv, idxJit, o, j, len, inner) : slice(valJit, o, j, len, inner);
                    if (kind == Kind.SORT || kind == Kind.ARGSORT) {
                        assertArrayEquals(what + " MLX", sorted, mlx, 0f);
                        assertArrayEquals(what + " JIT", sorted, jit, 0f);
                    } else {
                        assertPartitioned(what + " MLX", original, mlx, kth);
                        assertPartitioned(what + " JIT", original, jit, kth);
                    }
                }
            }
        }
    }

    @Test
    public void testSort() throws TornadoExecutionPlanException {
        check(Kind.SORT);
    }

    @Test
    public void testArgsort() throws TornadoExecutionPlanException {
        check(Kind.ARGSORT);
    }

    @Test
    public void testPartition() throws TornadoExecutionPlanException {
        check(Kind.PARTITION);
    }

    @Test
    public void testArgpartition() throws TornadoExecutionPlanException {
        check(Kind.ARGPARTITION);
    }

    @Test
    public void testHalfAndIntSort() throws TornadoExecutionPlanException {
        final int n = 777;
        HalfFloatArray h = half(values(n, -10, 10, 111));
        HalfFloatArray hs = new HalfFloatArray(n);
        int[] iv = new int[n];
        for (int i = 0; i < n; i++) {
            iv[i] = (i * 7919) % 1000 - 500;
        }
        IntArray xi = IntArray.fromArray(iv);
        IntArray is = new IntArray(n);
        run(new TaskGraph("hi").transferToDevice(DataTransferMode.FIRST_EXECUTION, h, xi) //
                .libraryTask("h", MlxSort::sort, h, hs) //
                .libraryTask("i", MlxSort::sort, xi, is) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, hs, is));
        float[] hv = widen(h);
        Arrays.sort(hv);
        int[] sortedInts = iv.clone();
        Arrays.sort(sortedInts);
        for (int i = 0; i < n; i++) {
            assertEquals("sort float16 " + i, hv[i], hs.get(i).getFloat32(), 0f);
            assertEquals("sort int32 " + i, sortedInts[i], is.get(i));
        }
    }
}
