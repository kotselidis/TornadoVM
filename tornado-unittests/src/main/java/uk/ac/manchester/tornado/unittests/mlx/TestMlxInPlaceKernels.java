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
import static org.junit.Assert.assertTrue;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.function.Function;

import org.junit.Test;

import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.BFloat16Array;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.api.types.arrays.TornadoNativeArray;
import uk.ac.manchester.tornado.mlx.Mlx;
import uk.ac.manchester.tornado.mlx.MlxLogic;
import uk.ac.manchester.tornado.mlx.MlxMath;
import uk.ac.manchester.tornado.mlx.MlxOptions;
import uk.ac.manchester.tornado.mlx.provider.MlxLibraryProvider;

/**
 * Element-wise MLX tasks that run MLX's own Metal kernels directly on TornadoVM's buffers, with no
 * MLX result array and no copy back. Covers both kernel forms (one element per thread below 65,536
 * elements, several above, with an uneven tail), float16 and bfloat16, and a comparison writing
 * bytes, and checks that the in-place route was taken. Every routed operation is also run through
 * MLX's C API in the same graph ({@code MlxOptions.inPlaceKernels(false)}), on inputs with NaNs,
 * infinities and signed zeros, and the two results must be byte-identical.
 *
 * <p>
 * How to run?
 * </p>
 * <code>
 * tornado-test -V uk.ac.manchester.tornado.unittests.mlx.TestMlxInPlaceKernels
 * </code>
 */
public class TestMlxInPlaceKernels extends MlxTestBase {

    private static final int[] SIZES = { 1, 1027, 65535, 65536, 65537, 1 << 20 };

    private static void assertInPlace(long before) {
        assertTrue("the task did not run as an in-place MLX kernel (run without -Dtornado.mlx.kernels=False)", MlxLibraryProvider.kernelDispatches() > before);
    }

    @Test
    public void testAddFloat() throws TornadoExecutionPlanException {
        for (int n : SIZES) {
            FloatArray a = FloatArray.fromArray(values(n, -5, 5, 1));
            FloatArray b = FloatArray.fromArray(values(n, -5, 5, 2));
            FloatArray c = new FloatArray(n);
            long before = MlxLibraryProvider.kernelDispatches();
            run(new TaskGraph("add").transferToDevice(DataTransferMode.EVERY_EXECUTION, a, b) //
                    .libraryTask("t", Mlx::add, a, b, c) //
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, c));
            assertInPlace(before);
            for (int i = 0; i < n; i++) {
                assertEquals("add n=" + n + " element " + i, a.get(i) + b.get(i), c.get(i), 0.0f);
            }
        }
    }

    @Test
    public void testExpHalf() throws TornadoExecutionPlanException {
        for (int n : SIZES) {
            float[] v = values(n, -3, 3, 3);
            HalfFloatArray a = new HalfFloatArray(n);
            for (int i = 0; i < n; i++) {
                a.set(i, new HalfFloat(v[i]));
            }
            HalfFloatArray out = new HalfFloatArray(n);
            long before = MlxLibraryProvider.kernelDispatches();
            run(new TaskGraph("exp").transferToDevice(DataTransferMode.EVERY_EXECUTION, a) //
                    .libraryTask("t", Mlx::exp, a, out) //
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, out));
            assertInPlace(before);
            for (int i = 0; i < n; i++) {
                float expected = (float) Math.exp(a.get(i).getFloat32());
                assertEquals("exp n=" + n + " element " + i, expected, out.get(i).getFloat32(), 2e-3f * Math.max(1.0f, expected));
            }
        }
    }

    @Test
    public void testMultiplyBFloat16() throws TornadoExecutionPlanException {
        int n = 70001;
        float[] va = values(n, -4, 4, 4);
        float[] vb = values(n, -4, 4, 5);
        BFloat16Array a = new BFloat16Array(n);
        BFloat16Array b = new BFloat16Array(n);
        for (int i = 0; i < n; i++) {
            a.setFloat(i, va[i]);
            b.setFloat(i, vb[i]);
        }
        BFloat16Array c = new BFloat16Array(n);
        long before = MlxLibraryProvider.kernelDispatches();
        run(new TaskGraph("mul").transferToDevice(DataTransferMode.EVERY_EXECUTION, a, b) //
                .libraryTask("t", Mlx::multiply, a, b, c) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, c));
        assertInPlace(before);
        for (int i = 0; i < n; i++) {
            float expected = a.getFloat(i) * b.getFloat(i);
            assertEquals("multiply element " + i, expected, c.getFloat(i), 1e-2f * Math.max(1.0f, Math.abs(expected)));
        }
    }

    @Test
    public void testLessWithNaNs() throws TornadoExecutionPlanException {
        int n = 100003;
        float[] va = values(n, -1, 1, 6);
        float[] vb = values(n, -1, 1, 7);
        for (int i = 0; i < n; i += 13) {
            va[i] = Float.NaN;
        }
        FloatArray a = FloatArray.fromArray(va);
        FloatArray b = FloatArray.fromArray(vb);
        ByteArray out = new ByteArray(n);
        long before = MlxLibraryProvider.kernelDispatches();
        run(new TaskGraph("less").transferToDevice(DataTransferMode.EVERY_EXECUTION, a, b) //
                .libraryTask("t", MlxLogic::less, a, b, out) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, out));
        assertInPlace(before);
        for (int i = 0; i < n; i++) {
            assertEquals("less element " + i, va[i] < vb[i] ? 1 : 0, out.get(i));
        }
    }

    /** A chain of in-place kernels whose outputs feed the next task, as in a real graph. */
    @Test
    public void testChain() throws TornadoExecutionPlanException {
        int n = 1 << 18;
        FloatArray a = FloatArray.fromArray(values(n, 0.5f, 2, 8));
        FloatArray b = new FloatArray(n);
        FloatArray c = new FloatArray(n);
        FloatArray d = new FloatArray(n);
        run(new TaskGraph("chain").transferToDevice(DataTransferMode.EVERY_EXECUTION, a) //
                .libraryTask("t1", Mlx::sqrt, a, b) //
                .libraryTask("t2", Mlx::multiply, b, b, c) //
                .libraryTask("t3", Mlx::subtract, c, a, d) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, d));
        for (int i = 0; i < n; i++) {
            assertEquals("sqrt(a)^2 - a element " + i, 0.0f, d.get(i), 1e-5f * a.get(i));
        }
    }

    private static final int N = 70001;
    private static final MlxOptions C_API = MlxOptions.gpu().inPlaceKernels(false);

    /** Values in [-4, 4) with NaNs, infinities, signed zeros and exact halves mixed in. */
    private static float[] specials(long seed) {
        float[] v = values(N, -4, 4, seed);
        float[] special = { Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, 0.0f, -0.0f, 2.5f, -1.5f, 1.0f };
        for (int i = 0; i < N; i += 7) {
            v[i] = special[(i / 7) % special.length];
        }
        return v;
    }

    private static FloatArray floats(long seed) {
        return FloatArray.fromArray(specials(seed));
    }

    private static HalfFloatArray halves(long seed) {
        float[] v = specials(seed);
        HalfFloatArray a = new HalfFloatArray(N);
        for (int i = 0; i < N; i++) {
            a.set(i, new HalfFloat(v[i]));
        }
        return a;
    }

    private static BFloat16Array bfloats(long seed) {
        float[] v = specials(seed);
        BFloat16Array a = new BFloat16Array(N);
        for (int i = 0; i < N; i++) {
            a.setFloat(i, v[i]);
        }
        return a;
    }

    private static IntArray ints(long seed, int lo, int hi) {
        java.util.Random r = new java.util.Random(seed);
        IntArray a = new IntArray(N);
        for (int i = 0; i < N; i++) {
            a.set(i, lo + r.nextInt(hi - lo));
        }
        return a;
    }

    private static ByteArray bytes(long seed) {
        java.util.Random r = new java.util.Random(seed);
        ByteArray a = new ByteArray(N);
        for (int i = 0; i < N; i++) {
            a.set(i, (byte) (r.nextInt(3) == 0 ? 0 : r.nextInt(5)));
        }
        return a;
    }

    private static void assertSameBytes(String what, TornadoNativeArray expected, TornadoNativeArray actual) {
        MemorySegment e = expected.getSegment();
        MemorySegment a = actual.getSegment();
        long n = expected.getNumBytesOfSegment();
        assertEquals(what + " size", n, actual.getNumBytesOfSegment());
        for (long i = 0; i < n; i++) {
            byte x = e.get(ValueLayout.JAVA_BYTE, i);
            byte y = a.get(ValueLayout.JAVA_BYTE, i);
            if (x != y) {
                throw new AssertionError(what + ": byte " + i + " (element " + i / expected.getElementSize() + ") differs: C API " + x + ", in place " + y);
            }
        }
    }

    /**
     * Adds the in-place task ({@code inPlace}) and the C API task ({@code cApi}) to one graph, runs
     * it, and compares the outputs byte for byte; exactly one call must have run in place.
     */
    private static void compare(String name, Object[] inputs, TornadoNativeArray[] inPlaceOut, TornadoNativeArray[] cApiOut, Function<TaskGraph, TaskGraph> tasks)
            throws TornadoExecutionPlanException {
        TaskGraph g = new TaskGraph("cmp").transferToDevice(DataTransferMode.EVERY_EXECUTION, inputs);
        g = tasks.apply(g);
        Object[] outs = new Object[inPlaceOut.length + cApiOut.length];
        System.arraycopy(inPlaceOut, 0, outs, 0, inPlaceOut.length);
        System.arraycopy(cApiOut, 0, outs, inPlaceOut.length, cApiOut.length);
        g.transferToHost(DataTransferMode.EVERY_EXECUTION, outs);
        long before = MlxLibraryProvider.kernelDispatches();
        run(g);
        assertEquals(name + ": in-place calls", before + 1, MlxLibraryProvider.kernelDispatches());
        for (int i = 0; i < inPlaceOut.length; i++) {
            assertSameBytes(name, cApiOut[i], inPlaceOut[i]);
        }
    }

    @Test
    public void testScaledAndReciprocal() throws TornadoExecutionPlanException {
        FloatArray a = floats(11);
        HalfFloatArray h = halves(12);
        BFloat16Array b = bfloats(13);
        FloatArray o1 = new FloatArray(N);
        FloatArray o2 = new FloatArray(N);
        compare("degrees f32", new Object[] { a }, new TornadoNativeArray[] { o1 }, new TornadoNativeArray[] { o2 }, g -> g //
                .libraryTask("i", MlxMath::degrees, a, o1).libraryTask("c", (x, y) -> MlxMath.degrees(x, y).withTuning(C_API), a, o2));
        HalfFloatArray h1 = new HalfFloatArray(N);
        HalfFloatArray h2 = new HalfFloatArray(N);
        compare("radians f16", new Object[] { h }, new TornadoNativeArray[] { h1 }, new TornadoNativeArray[] { h2 }, g -> g //
                .libraryTask("i", MlxMath::radians, h, h1).libraryTask("c", (x, y) -> MlxMath.radians(x, y).withTuning(C_API), h, h2));
        BFloat16Array b1 = new BFloat16Array(N);
        BFloat16Array b2 = new BFloat16Array(N);
        compare("reciprocal bf16", new Object[] { b }, new TornadoNativeArray[] { b1 }, new TornadoNativeArray[] { b2 }, g -> g //
                .libraryTask("i", MlxMath::reciprocal, b, b1).libraryTask("c", (x, y) -> MlxMath.reciprocal(x, y).withTuning(C_API), b, b2));
    }

    @Test
    public void testDivisionAndRounding() throws TornadoExecutionPlanException {
        FloatArray a = floats(21);
        FloatArray b = floats(22);
        FloatArray o1 = new FloatArray(N);
        FloatArray o2 = new FloatArray(N);
        compare("floor_divide f32", new Object[] { a, b }, new TornadoNativeArray[] { o1 }, new TornadoNativeArray[] { o2 }, g -> g //
                .libraryTask("i", MlxMath::floorDivide, a, b, o1).libraryTask("c", (x, y, z) -> MlxMath.floorDivide(x, y, z).withTuning(C_API), a, b, o2));
        IntArray ia = ints(23, -1000, 1000);
        IntArray ib = ints(24, 1, 50);
        IntArray i1 = new IntArray(N);
        IntArray i2 = new IntArray(N);
        compare("floor_divide i32", new Object[] { ia, ib }, new TornadoNativeArray[] { i1 }, new TornadoNativeArray[] { i2 }, g -> g //
                .libraryTask("i", MlxMath::floorDivide, ia, ib, i1).libraryTask("c", (x, y, z) -> MlxMath.floorDivide(x, y, z).withTuning(C_API), ia, ib, i2));
        HalfFloatArray h = halves(25);
        HalfFloatArray h1 = new HalfFloatArray(N);
        HalfFloatArray h2 = new HalfFloatArray(N);
        compare("round f16, 2 decimals", new Object[] { h }, new TornadoNativeArray[] { h1 }, new TornadoNativeArray[] { h2 }, g -> g //
                .libraryTask("i", MlxMath::round, h, h1, 2).libraryTask("c", (x, y, d) -> MlxMath.round(x, y, d).withTuning(C_API), h, h2, 2));
        FloatArray r1 = new FloatArray(N);
        FloatArray r2 = new FloatArray(N);
        compare("round f32", new Object[] { a }, new TornadoNativeArray[] { r1 }, new TornadoNativeArray[] { r2 }, g -> g //
                .libraryTask("i", MlxMath::round, a, r1, 0).libraryTask("c", (x, y, d) -> MlxMath.round(x, y, d).withTuning(C_API), a, r2, 0));
        FloatArray q1 = new FloatArray(N);
        FloatArray m1 = new FloatArray(N);
        FloatArray q2 = new FloatArray(N);
        FloatArray m2 = new FloatArray(N);
        compare("divmod f32", new Object[] { a, b }, new TornadoNativeArray[] { q1, m1 }, new TornadoNativeArray[] { q2, m2 }, g -> g //
                .libraryTask("i", MlxMath::divmod, a, b, q1, m1).libraryTask("c", (x, y, q, m) -> MlxMath.divmod(x, y, q, m).withTuning(C_API), a, b, q2, m2));
    }

    @Test
    public void testClipAndWhere() throws TornadoExecutionPlanException {
        HalfFloatArray h = halves(31);
        HalfFloatArray h1 = new HalfFloatArray(N);
        HalfFloatArray h2 = new HalfFloatArray(N);
        compare("clip f16", new Object[] { h }, new TornadoNativeArray[] { h1 }, new TornadoNativeArray[] { h2 }, g -> g //
                .libraryTask("i", MlxMath::clip, h, h1, -1.0004f, 2.0006f).libraryTask("c", (x, y, lo, hi) -> MlxMath.clip(x, y, lo, hi).withTuning(C_API), h, h2, -1.0004f, 2.0006f));
        IntArray ia = ints(32, -100, 100);
        IntArray i1 = new IntArray(N);
        IntArray i2 = new IntArray(N);
        compare("clip i32", new Object[] { ia }, new TornadoNativeArray[] { i1 }, new TornadoNativeArray[] { i2 }, g -> g //
                .libraryTask("i", MlxMath::clip, ia, i1, -10, 20).libraryTask("c", (x, y, lo, hi) -> MlxMath.clip(x, y, lo, hi).withTuning(C_API), ia, i2, -10, 20));
        ByteArray cond = bytes(33);
        FloatArray x = floats(34);
        FloatArray y = floats(35);
        FloatArray w1 = new FloatArray(N);
        FloatArray w2 = new FloatArray(N);
        compare("where f32", new Object[] { cond, x, y }, new TornadoNativeArray[] { w1 }, new TornadoNativeArray[] { w2 }, g -> g //
                .libraryTask("i", MlxMath::where, cond, x, y, w1).libraryTask("c", (c, p, q, o) -> MlxMath.where(c, p, q, o).withTuning(C_API), cond, x, y, w2));
    }

    @Test
    public void testPredicates() throws TornadoExecutionPlanException {
        FloatArray a = floats(41);
        BFloat16Array b = bfloats(42);
        String[] names = { "isnan", "isinf", "isfinite", "isposinf", "isneginf" };
        for (String name : names) {
            ByteArray o1 = new ByteArray(N);
            ByteArray o2 = new ByteArray(N);
            ByteArray p1 = new ByteArray(N);
            ByteArray p2 = new ByteArray(N);
            compare(name + " f32", new Object[] { a }, new TornadoNativeArray[] { o1 }, new TornadoNativeArray[] { o2 }, g -> g //
                    .libraryTask("i", (x, y) -> predicate(name, x, y), a, o1).libraryTask("c", (x, y) -> predicate(name, x, y).withTuning(C_API), a, o2));
            compare(name + " bf16", new Object[] { b }, new TornadoNativeArray[] { p1 }, new TornadoNativeArray[] { p2 }, g -> g //
                    .libraryTask("i", (x, y) -> predicate(name, x, y), b, p1).libraryTask("c", (x, y) -> predicate(name, x, y).withTuning(C_API), b, p2));
        }
    }

    private static uk.ac.manchester.tornado.api.common.LibraryTaskDescriptor predicate(String name, FloatArray a, ByteArray out) {
        return switch (name) {
            case "isnan" -> MlxLogic.isnan(a, out);
            case "isinf" -> MlxLogic.isinf(a, out);
            case "isfinite" -> MlxLogic.isfinite(a, out);
            case "isposinf" -> MlxLogic.isposinf(a, out);
            default -> MlxLogic.isneginf(a, out);
        };
    }

    private static uk.ac.manchester.tornado.api.common.LibraryTaskDescriptor predicate(String name, BFloat16Array a, ByteArray out) {
        return switch (name) {
            case "isnan" -> MlxLogic.isnan(a, out);
            case "isinf" -> MlxLogic.isinf(a, out);
            case "isfinite" -> MlxLogic.isfinite(a, out);
            case "isposinf" -> MlxLogic.isposinf(a, out);
            default -> MlxLogic.isneginf(a, out);
        };
    }

    @Test
    public void testBitwiseAndLogical() throws TornadoExecutionPlanException {
        IntArray a = ints(51, Integer.MIN_VALUE / 2, Integer.MAX_VALUE / 2);
        IntArray s = ints(52, 0, 31);
        IntArray o1 = new IntArray(N);
        IntArray o2 = new IntArray(N);
        compare("bitwise_and", new Object[] { a, s }, new TornadoNativeArray[] { o1 }, new TornadoNativeArray[] { o2 }, g -> g //
                .libraryTask("i", MlxLogic::bitwiseAnd, a, s, o1).libraryTask("c", (x, y, z) -> MlxLogic.bitwiseAnd(x, y, z).withTuning(C_API), a, s, o2));
        IntArray l1 = new IntArray(N);
        IntArray l2 = new IntArray(N);
        compare("right_shift", new Object[] { a, s }, new TornadoNativeArray[] { l1 }, new TornadoNativeArray[] { l2 }, g -> g //
                .libraryTask("i", MlxLogic::rightShift, a, s, l1).libraryTask("c", (x, y, z) -> MlxLogic.rightShift(x, y, z).withTuning(C_API), a, s, l2));
        IntArray n1 = new IntArray(N);
        IntArray n2 = new IntArray(N);
        compare("bitwise_invert", new Object[] { a }, new TornadoNativeArray[] { n1 }, new TornadoNativeArray[] { n2 }, g -> g //
                .libraryTask("i", MlxLogic::bitwiseInvert, a, n1).libraryTask("c", (x, y) -> MlxLogic.bitwiseInvert(x, y).withTuning(C_API), a, n2));
        ByteArray p = bytes(53);
        ByteArray q = bytes(54);
        ByteArray b1 = new ByteArray(N);
        ByteArray b2 = new ByteArray(N);
        compare("logical_and", new Object[] { p, q }, new TornadoNativeArray[] { b1 }, new TornadoNativeArray[] { b2 }, g -> g //
                .libraryTask("i", MlxLogic::logicalAnd, p, q, b1).libraryTask("c", (x, y, z) -> MlxLogic.logicalAnd(x, y, z).withTuning(C_API), p, q, b2));
        ByteArray c1 = new ByteArray(N);
        ByteArray c2 = new ByteArray(N);
        compare("logical_not", new Object[] { p }, new TornadoNativeArray[] { c1 }, new TornadoNativeArray[] { c2 }, g -> g //
                .libraryTask("i", MlxLogic::logicalNot, p, c1).libraryTask("c", (x, y) -> MlxLogic.logicalNot(x, y).withTuning(C_API), p, c2));
    }

    @Test
    public void testNanToNumIscloseAndComplex() throws TornadoExecutionPlanException {
        HalfFloatArray h = halves(61);
        HalfFloatArray h1 = new HalfFloatArray(N);
        HalfFloatArray h2 = new HalfFloatArray(N);
        compare("nan_to_num f16", new Object[] { h }, new TornadoNativeArray[] { h1 }, new TornadoNativeArray[] { h2 }, g -> g //
                .libraryTask("i", MlxLogic::nanToNum, h, h1, 7.0f, 1000.0f, -1000.0f)
                .libraryTask("c", (x, y, a, b, c) -> MlxLogic.nanToNum(x, y, a, b, c).withTuning(C_API), h, h2, 7.0f, 1000.0f, -1000.0f));
        FloatArray a = floats(62);
        FloatArray b = floats(63);
        for (int i = 0; i < N; i += 3) {
            b.set(i, a.get(i) * (1 + 1e-6f * (i % 5)));
        }
        for (boolean equalNan : new boolean[] { false, true }) {
            ByteArray o1 = new ByteArray(N);
            ByteArray o2 = new ByteArray(N);
            compare("isclose equal_nan=" + equalNan, new Object[] { a, b }, new TornadoNativeArray[] { o1 }, new TornadoNativeArray[] { o2 }, g -> g //
                    .libraryTask("i", MlxLogic::isclose, a, b, o1, 1e-5f, 1e-8f, equalNan)
                    .libraryTask("c", (x, y, z, r, t, e) -> MlxLogic.isclose(x, y, z, r, t, e).withTuning(C_API), a, b, o2, 1e-5f, 1e-8f, equalNan));
        }
        FloatArray z = floats(64);
        FloatArray zc = new FloatArray(N - 1);
        for (int i = 0; i < N - 1; i++) {
            zc.set(i, z.get(i));
        }
        FloatArray reA = new FloatArray((N - 1) / 2);
        FloatArray reB = new FloatArray((N - 1) / 2);
        compare("real", new Object[] { zc }, new TornadoNativeArray[] { reA }, new TornadoNativeArray[] { reB }, g -> g //
                .libraryTask("i", MlxLogic::real, zc, reA).libraryTask("c", (x, y) -> MlxLogic.real(x, y).withTuning(C_API), zc, reB));
        FloatArray cj1 = new FloatArray(N - 1);
        FloatArray cj2 = new FloatArray(N - 1);
        compare("conjugate", new Object[] { zc }, new TornadoNativeArray[] { cj1 }, new TornadoNativeArray[] { cj2 }, g -> g //
                .libraryTask("i", MlxLogic::conjugate, zc, cj1).libraryTask("c", (x, y) -> MlxLogic.conjugate(x, y).withTuning(C_API), zc, cj2));
    }
}
