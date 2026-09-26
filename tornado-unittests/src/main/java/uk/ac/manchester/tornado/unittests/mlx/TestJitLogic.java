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
import java.util.function.IntBinaryOperator;

import org.junit.Test;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.common.TornadoFunctions.LibraryTask2;
import uk.ac.manchester.tornado.api.common.TornadoFunctions.LibraryTask3;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.mlx.MlxLogic;
import uk.ac.manchester.tornado.mlx.jit.JitLogic;

/**
 * The MLX comparisons, classification, logical, bitwise and complex-part operations and their
 * KernelContext JIT counterparts, run in one graph on the same inputs (which include NaNs,
 * infinities and repeated values) and checked against Java.
 *
 * <p>
 * How to run?
 * </p>
 * <code>
 * tornado-test -V uk.ac.manchester.tornado.unittests.mlx.TestJitLogic
 * </code>
 */
public class TestJitLogic extends MlxTestBase {

    private static final int N = 1027;

    interface FloatPredicate {
        boolean test(float x, float y);
    }

    private static void execute(TaskGraph g, String... jitTasks) throws TornadoExecutionPlanException {
        GridScheduler gs = new GridScheduler();
        for (String t : jitTasks) {
            gs.addWorkerGrid(g.getTaskGraphName() + "." + t, TestJitElementwise.grid1D(N));
        }
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(g.snapshot())) {
            plan.withGridScheduler(gs).execute();
        }
    }

    /** Values in [-5, 5) with NaNs, infinities and copies of the partner array mixed in. */
    private static float[] specials(long seed, float[] partner) {
        Random r = new Random(seed);
        float[] v = new float[N];
        for (int i = 0; i < N; i++) {
            v[i] = switch (i % 11) {
                case 3 -> Float.NaN;
                case 5 -> Float.POSITIVE_INFINITY;
                case 7 -> Float.NEGATIVE_INFINITY;
                case 9 -> partner != null ? partner[i] : 1.0f;
                default -> 10 * r.nextFloat() - 5;
            };
        }
        return v;
    }

    private static void assertMask(String what, boolean[] expected, ByteArray mlx, ByteArray jit) {
        for (int i = 0; i < expected.length; i++) {
            assertEquals(what + " MLX element " + i, expected[i] ? 1 : 0, mlx.get(i));
            assertEquals(what + " JIT element " + i, expected[i] ? 1 : 0, jit.get(i));
        }
    }

    private static void comparison(String name, LibraryTask3<FloatArray, FloatArray, ByteArray> mlx, int op, FloatPredicate ref) throws TornadoExecutionPlanException {
        float[] bv = specials(1, null);
        float[] av = specials(2, bv);
        FloatArray a = FloatArray.fromArray(av);
        FloatArray b = FloatArray.fromArray(bv);
        ByteArray outMlx = new ByteArray(N);
        ByteArray outJit = new ByteArray(N);
        TaskGraph g = new TaskGraph("cmp").transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b) //
                .libraryTask("m", mlx, a, b, outMlx) //
                .task("j", JitLogic::compare, new KernelContext(), a, b, outJit, N, op) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, outMlx, outJit);
        execute(g, "j");
        boolean[] e = new boolean[N];
        for (int i = 0; i < N; i++) {
            e[i] = ref.test(av[i], bv[i]);
        }
        assertMask(name, e, outMlx, outJit);
    }

    @Test
    public void testComparisons() throws TornadoExecutionPlanException {
        comparison("equal", MlxLogic::equal, JitLogic.EQUAL, (x, y) -> x == y);
        comparison("notEqual", MlxLogic::notEqual, JitLogic.NOT_EQUAL, (x, y) -> x != y);
        comparison("greater", MlxLogic::greater, JitLogic.GREATER, (x, y) -> x > y);
        comparison("greaterEqual", MlxLogic::greaterEqual, JitLogic.GREATER_EQUAL, (x, y) -> x >= y);
        comparison("less", MlxLogic::less, JitLogic.LESS, (x, y) -> x < y);
        comparison("lessEqual", MlxLogic::lessEqual, JitLogic.LESS_EQUAL, (x, y) -> x <= y);
    }

    private static void classification(String name, LibraryTask2<FloatArray, ByteArray> mlx, int kind, java.util.function.Predicate<Float> ref) throws TornadoExecutionPlanException {
        float[] av = specials(3, null);
        FloatArray a = FloatArray.fromArray(av);
        ByteArray outMlx = new ByteArray(N);
        ByteArray outJit = new ByteArray(N);
        TaskGraph g = new TaskGraph("cls").transferToDevice(DataTransferMode.FIRST_EXECUTION, a) //
                .libraryTask("m", mlx, a, outMlx) //
                .task("j", JitLogic::classify, new KernelContext(), a, outJit, N, kind) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, outMlx, outJit);
        execute(g, "j");
        boolean[] e = new boolean[N];
        for (int i = 0; i < N; i++) {
            e[i] = ref.test(av[i]);
        }
        assertMask(name, e, outMlx, outJit);
    }

    @Test
    public void testClassification() throws TornadoExecutionPlanException {
        classification("isfinite", MlxLogic::isfinite, JitLogic.FINITE, x -> Float.isFinite(x));
        classification("isinf", MlxLogic::isinf, JitLogic.INF, x -> Float.isInfinite(x));
        classification("isnan", MlxLogic::isnan, JitLogic.NAN, x -> Float.isNaN(x));
        classification("isneginf", MlxLogic::isneginf, JitLogic.NEG_INF, x -> x == Float.NEGATIVE_INFINITY);
        classification("isposinf", MlxLogic::isposinf, JitLogic.POS_INF, x -> x == Float.POSITIVE_INFINITY);
    }

    private static boolean close(float x, float y, float rtol, float atol, boolean equalNan) {
        if (Float.isNaN(x) || Float.isNaN(y)) {
            return equalNan && Float.isNaN(x) && Float.isNaN(y);
        }
        if (Float.isInfinite(x) || Float.isInfinite(y)) {
            return x == y;
        }
        return Math.abs(x - y) <= atol + rtol * Math.abs(y);
    }

    @Test
    public void testIscloseAllcloseArrayEqual() throws TornadoExecutionPlanException {
        final float rtol = 1e-3f;
        final float atol = 1e-5f;
        float[] bv = specials(4, null);
        float[] av = bv.clone();
        for (int i = 0; i < N; i += 4) {
            if (Float.isFinite(av[i])) {
                av[i] += (i % 8 == 0 ? 1e-6f : 1e-1f) * (1 + Math.abs(av[i]));
            }
        }
        FloatArray a = FloatArray.fromArray(av);
        FloatArray b = FloatArray.fromArray(bv);
        FloatArray same = FloatArray.fromArray(bv.clone());
        for (boolean equalNan : new boolean[] { false, true }) {
            int en = equalNan ? 1 : 0;
            ByteArray icMlx = new ByteArray(N);
            ByteArray icJit = new ByteArray(N);
            ByteArray[] flags = new ByteArray[8];
            for (int k = 0; k < flags.length; k++) {
                flags[k] = new ByteArray(1);
            }
            TaskGraph g = new TaskGraph("ic").transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b, same) //
                    .libraryTask("m1", MlxLogic::isclose, a, b, icMlx, rtol, atol, equalNan) //
                    .libraryTask("m2", MlxLogic::allclose, a, b, flags[0], rtol, atol, equalNan) //
                    .libraryTask("m3", MlxLogic::allclose, same, b, flags[1], rtol, atol, equalNan) //
                    .libraryTask("m4", MlxLogic::arrayEqual, a, b, flags[2], equalNan) //
                    .libraryTask("m5", MlxLogic::arrayEqual, same, b, flags[3], equalNan) //
                    .task("j1", JitLogic::isclose, new KernelContext(), a, b, icJit, N, rtol, atol, en) //
                    .task("s1", JitLogic::setFlag, new KernelContext(), flags[4], 1) //
                    .task("j2", JitLogic::allcloseCheck, new KernelContext(), a, b, flags[4], N, rtol, atol, en) //
                    .task("s2", JitLogic::setFlag, new KernelContext(), flags[5], 1) //
                    .task("j3", JitLogic::allcloseCheck, new KernelContext(), same, b, flags[5], N, rtol, atol, en) //
                    .task("s3", JitLogic::setFlag, new KernelContext(), flags[6], 1) //
                    .task("j4", JitLogic::arrayEqualCheck, new KernelContext(), a, b, flags[6], N, en) //
                    .task("s4", JitLogic::setFlag, new KernelContext(), flags[7], 1) //
                    .task("j5", JitLogic::arrayEqualCheck, new KernelContext(), same, b, flags[7], N, en) //
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, icMlx, icJit, flags[0], flags[1], flags[2], flags[3], flags[4], flags[5], flags[6], flags[7]);
            execute(g, "j1", "s1", "j2", "s2", "j3", "s3", "j4", "s4", "j5");
            boolean[] e = new boolean[N];
            boolean allAB = true;
            for (int i = 0; i < N; i++) {
                e[i] = close(av[i], bv[i], rtol, atol, equalNan);
                allAB &= e[i];
            }
            String what = " equalNan=" + equalNan;
            assertMask("isclose" + what, e, icMlx, icJit);
            // same == b element for element, so only NaN handling decides allclose and array_equal for it.
            int expectSame = equalNan ? 1 : 0;
            int[] expected = { allAB ? 1 : 0, expectSame, 0, expectSame };
            String[] names = { "allclose(a, b)", "allclose(b, b)", "arrayEqual(a, b)", "arrayEqual(b, b)" };
            for (int k = 0; k < 4; k++) {
                assertEquals(names[k] + what + " MLX", expected[k], flags[k].get(0));
                assertEquals(names[k] + what + " JIT", expected[k], flags[k + 4].get(0));
            }
        }
    }

    @Test
    public void testLogical() throws TornadoExecutionPlanException {
        Random r = new Random(5);
        byte[] av = new byte[N];
        byte[] bv = new byte[N];
        for (int i = 0; i < N; i++) {
            av[i] = (byte) (r.nextBoolean() ? r.nextInt(5) + 1 : 0);
            bv[i] = (byte) (r.nextBoolean() ? 1 : 0);
        }
        ByteArray a = ByteArray.fromArray(av);
        ByteArray b = ByteArray.fromArray(bv);
        ByteArray andM = new ByteArray(N);
        ByteArray orM = new ByteArray(N);
        ByteArray notM = new ByteArray(N);
        ByteArray andJ = new ByteArray(N);
        ByteArray orJ = new ByteArray(N);
        ByteArray notJ = new ByteArray(N);
        TaskGraph g = new TaskGraph("lg").transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b) //
                .libraryTask("m1", MlxLogic::logicalAnd, a, b, andM) //
                .libraryTask("m2", MlxLogic::logicalOr, a, b, orM) //
                .libraryTask("m3", MlxLogic::logicalNot, a, notM) //
                .task("j1", JitLogic::logical, new KernelContext(), a, b, andJ, N, JitLogic.AND) //
                .task("j2", JitLogic::logical, new KernelContext(), a, b, orJ, N, JitLogic.OR) //
                .task("j3", JitLogic::logicalNot, new KernelContext(), a, notJ, N) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, andM, orM, notM, andJ, orJ, notJ);
        execute(g, "j1", "j2", "j3");
        boolean[] eAnd = new boolean[N];
        boolean[] eOr = new boolean[N];
        boolean[] eNot = new boolean[N];
        for (int i = 0; i < N; i++) {
            eAnd[i] = av[i] != 0 && bv[i] != 0;
            eOr[i] = av[i] != 0 || bv[i] != 0;
            eNot[i] = av[i] == 0;
        }
        assertMask("logicalAnd", eAnd, andM, andJ);
        assertMask("logicalOr", eOr, orM, orJ);
        assertMask("logicalNot", eNot, notM, notJ);
    }

    private static void bitwise(String name, LibraryTask3<IntArray, IntArray, IntArray> mlx, int op, IntBinaryOperator ref, boolean shift) throws TornadoExecutionPlanException {
        Random r = new Random(6);
        int[] av = new int[N];
        int[] bv = new int[N];
        for (int i = 0; i < N; i++) {
            av[i] = r.nextInt();
            bv[i] = shift ? r.nextInt(31) : r.nextInt();
        }
        IntArray a = IntArray.fromArray(av);
        IntArray b = IntArray.fromArray(bv);
        IntArray outM = new IntArray(N);
        IntArray outJ = new IntArray(N);
        TaskGraph g = new TaskGraph("bw").transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b) //
                .libraryTask("m", mlx, a, b, outM) //
                .task("j", JitLogic::bitwise, new KernelContext(), a, b, outJ, N, op) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, outM, outJ);
        execute(g, "j");
        for (int i = 0; i < N; i++) {
            int e = ref.applyAsInt(av[i], bv[i]);
            assertEquals(name + " MLX " + av[i] + ", " + bv[i], e, outM.get(i));
            assertEquals(name + " JIT " + av[i] + ", " + bv[i], e, outJ.get(i));
        }
    }

    @Test
    public void testBitwise() throws TornadoExecutionPlanException {
        bitwise("bitwiseAnd", MlxLogic::bitwiseAnd, JitLogic.AND, (x, y) -> x & y, false);
        bitwise("bitwiseOr", MlxLogic::bitwiseOr, JitLogic.OR, (x, y) -> x | y, false);
        bitwise("bitwiseXor", MlxLogic::bitwiseXor, JitLogic.XOR, (x, y) -> x ^ y, false);
        bitwise("leftShift", MlxLogic::leftShift, JitLogic.LEFT_SHIFT, (x, y) -> x << y, true);
        bitwise("rightShift", MlxLogic::rightShift, JitLogic.RIGHT_SHIFT, (x, y) -> x >> y, true);
        int[] av = new int[N];
        Random r = new Random(7);
        for (int i = 0; i < N; i++) {
            av[i] = r.nextInt();
        }
        IntArray a = IntArray.fromArray(av);
        IntArray outM = new IntArray(N);
        IntArray outJ = new IntArray(N);
        TaskGraph g = new TaskGraph("inv").transferToDevice(DataTransferMode.FIRST_EXECUTION, a) //
                .libraryTask("m", MlxLogic::bitwiseInvert, a, outM) //
                .task("j", JitLogic::invert, new KernelContext(), a, outJ, N) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, outM, outJ);
        execute(g, "j");
        for (int i = 0; i < N; i++) {
            assertEquals("bitwiseInvert MLX", ~av[i], outM.get(i));
            assertEquals("bitwiseInvert JIT", ~av[i], outJ.get(i));
        }
    }

    @Test
    public void testNanToNum() throws TornadoExecutionPlanException {
        float[] av = specials(8, null);
        FloatArray a = FloatArray.fromArray(av);
        FloatArray outM = new FloatArray(N);
        FloatArray outJ = new FloatArray(N);
        TaskGraph g = new TaskGraph("nn").transferToDevice(DataTransferMode.FIRST_EXECUTION, a) //
                .libraryTask("m", MlxLogic::nanToNum, a, outM, 0.5f, 1e30f, -1e30f) //
                .task("j", JitLogic::nanToNum, new KernelContext(), a, outJ, N, 0.5f, 1e30f, -1e30f) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, outM, outJ);
        execute(g, "j");
        for (int i = 0; i < N; i++) {
            float e = Float.isNaN(av[i]) ? 0.5f : av[i] == Float.POSITIVE_INFINITY ? 1e30f : av[i] == Float.NEGATIVE_INFINITY ? -1e30f : av[i];
            assertEquals("nanToNum MLX " + i, e, outM.get(i), 0f);
            assertEquals("nanToNum JIT " + i, e, outJ.get(i), 0f);
        }
    }

    @Test
    public void testComplexParts() throws TornadoExecutionPlanException {
        float[] zv = values(2 * N, -3, 3, 9);
        FloatArray z = FloatArray.fromArray(zv);
        FloatArray reM = new FloatArray(N);
        FloatArray imM = new FloatArray(N);
        FloatArray cjM = new FloatArray(2 * N);
        FloatArray reJ = new FloatArray(N);
        FloatArray imJ = new FloatArray(N);
        FloatArray cjJ = new FloatArray(2 * N);
        TaskGraph g = new TaskGraph("cx").transferToDevice(DataTransferMode.FIRST_EXECUTION, z) //
                .libraryTask("m1", MlxLogic::real, z, reM) //
                .libraryTask("m2", MlxLogic::imag, z, imM) //
                .libraryTask("m3", MlxLogic::conjugate, z, cjM) //
                .task("j1", JitLogic::complexPart, new KernelContext(), z, reJ, N, 0) //
                .task("j2", JitLogic::complexPart, new KernelContext(), z, imJ, N, 1) //
                .task("j3", JitLogic::conjugate, new KernelContext(), z, cjJ, N) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, reM, imM, cjM, reJ, imJ, cjJ);
        execute(g, "j1", "j2", "j3");
        for (int i = 0; i < N; i++) {
            assertEquals("real MLX", zv[2 * i], reM.get(i), 0f);
            assertEquals("real JIT", zv[2 * i], reJ.get(i), 0f);
            assertEquals("imag MLX", zv[2 * i + 1], imM.get(i), 0f);
            assertEquals("imag JIT", zv[2 * i + 1], imJ.get(i), 0f);
            assertEquals("conjugate MLX re", zv[2 * i], cjM.get(2 * i), 0f);
            assertEquals("conjugate MLX im", -zv[2 * i + 1], cjM.get(2 * i + 1), 0f);
            assertEquals("conjugate JIT re", zv[2 * i], cjJ.get(2 * i), 0f);
            assertEquals("conjugate JIT im", -zv[2 * i + 1], cjJ.get(2 * i + 1), 0f);
        }
    }

    @Test
    public void testHalfAndIntForms() throws TornadoExecutionPlanException {
        float[] av = values(N, -2, 2, 10);
        float[] bv = values(N, -2, 2, 11);
        HalfFloatArray a = half(av);
        HalfFloatArray b = half(bv);
        int[] iv = new int[N];
        int[] jv = new int[N];
        for (int i = 0; i < N; i++) {
            iv[i] = i % 7;
            jv[i] = i % 5;
        }
        IntArray ia = IntArray.fromArray(iv);
        IntArray ib = IntArray.fromArray(jv);
        ByteArray lessH = new ByteArray(N);
        ByteArray eqI = new ByteArray(N);
        ByteArray allI = new ByteArray(1);
        run(new TaskGraph("hi").transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b, ia, ib) //
                .libraryTask("h", MlxLogic::less, a, b, lessH) //
                .libraryTask("i", MlxLogic::equal, ia, ib, eqI) //
                .libraryTask("e", MlxLogic::arrayEqual, ia, ia, allI, false) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, lessH, eqI, allI));
        float[] x = widen(a);
        float[] y = widen(b);
        for (int i = 0; i < N; i++) {
            assertEquals("less float16 " + i, x[i] < y[i] ? 1 : 0, lessH.get(i));
            assertEquals("equal int32 " + i, iv[i] == jv[i] ? 1 : 0, eqI.get(i));
        }
        assertEquals("arrayEqual int32", 1, allI.get(0));
    }
}
