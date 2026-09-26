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
import java.util.function.Supplier;

import org.junit.Test;

import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.common.LibraryTaskDescriptor;
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
import uk.ac.manchester.tornado.mlx.MlxLinalg;
import uk.ac.manchester.tornado.mlx.MlxLogic;
import uk.ac.manchester.tornado.mlx.MlxMath;
import uk.ac.manchester.tornado.mlx.MlxConv;
import uk.ac.manchester.tornado.mlx.MlxCreate;
import uk.ac.manchester.tornado.mlx.MlxFft;
import uk.ac.manchester.tornado.mlx.MlxIndex;
import uk.ac.manchester.tornado.mlx.MlxOptions;
import uk.ac.manchester.tornado.mlx.MlxProducts;
import uk.ac.manchester.tornado.mlx.MlxRandom;
import uk.ac.manchester.tornado.mlx.MlxReduce;
import uk.ac.manchester.tornado.mlx.MlxSort;
import uk.ac.manchester.tornado.mlx.MlxShape;
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
        TaskGraph g = new TaskGraph("cmp");
        if (inputs.length > 0) {
            g = g.transferToDevice(DataTransferMode.EVERY_EXECUTION, inputs);
        }
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

    private static LibraryTaskDescriptor predicate(String name, FloatArray a, ByteArray out) {
        return switch (name) {
            case "isnan" -> MlxLogic.isnan(a, out);
            case "isinf" -> MlxLogic.isinf(a, out);
            case "isfinite" -> MlxLogic.isfinite(a, out);
            case "isposinf" -> MlxLogic.isposinf(a, out);
            default -> MlxLogic.isneginf(a, out);
        };
    }

    private static LibraryTaskDescriptor predicate(String name, BFloat16Array a, ByteArray out) {
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

    // ---------------------------------------------------------------- shape and construction

    /** Adds one task, in place or through the C API, writing {@code outs}. */
    interface Builder {
        TaskGraph add(TaskGraph g, String id, boolean cApi, TornadoNativeArray[] outs);
    }

    private static LibraryTaskDescriptor tune(LibraryTaskDescriptor task, boolean cApi) {
        return cApi ? task.withTuning(C_API) : task;
    }

    /** Runs the task both ways in one graph and compares every output byte for byte. */
    private static void same(String name, Object[] inputs, Supplier<TornadoNativeArray[]> outputs, Builder builder) throws TornadoExecutionPlanException {
        TornadoNativeArray[] inPlace = outputs.get();
        TornadoNativeArray[] cApi = outputs.get();
        compare(name, inputs, inPlace, cApi, g -> builder.add(builder.add(g, "i", false, inPlace), "c", true, cApi));
    }

    private static Supplier<TornadoNativeArray[]> floatsOut(int... sizes) {
        return () -> {
            TornadoNativeArray[] outs = new TornadoNativeArray[sizes.length];
            for (int i = 0; i < sizes.length; i++) {
                outs[i] = new FloatArray(sizes[i]);
            }
            return outs;
        };
    }

    @Test
    public void testReshapeFamily() throws TornadoExecutionPlanException {
        int d0 = 5;
        int d1 = 6;
        int d2 = 7;
        int n = d0 * d1 * d2;
        FloatArray x = FloatArray.fromArray(values(n, -3, 3, 71));
        Object[] in = { x };
        same("reshape", in, floatsOut(n), (g, id, c, o) -> g.libraryTask(id, (a, b) -> tune(MlxShape.reshape(a, b, d0 * d1, d2), c), x, (FloatArray) o[0]));
        same("flatten", in, floatsOut(n), (g, id, c, o) -> g.libraryTask(id, (a, b) -> tune(MlxShape.flatten(a, b, d0, d1, d2), c), x, (FloatArray) o[0]));
        same("squeeze", in, floatsOut(n), (g, id, c, o) -> g.libraryTask(id, (a, b) -> tune(MlxShape.squeeze(a, b, d0 * d1, d2), c), x, (FloatArray) o[0]));
        same("expandDims", in, floatsOut(n), (g, id, c, o) -> g.libraryTask(id, (a, b) -> tune(MlxShape.expandDims(a, b, d0 * d1, d2), c), x, (FloatArray) o[0]));
        same("atleast3d", in, floatsOut(n), (g, id, c, o) -> g.libraryTask(id, (a, b) -> tune(MlxShape.atleast3d(a, b), c), x, (FloatArray) o[0]));
        same("copy", in, floatsOut(n), (g, id, c, o) -> g.libraryTask(id, (a, b) -> tune(MlxShape.copy(a, b), c), x, (FloatArray) o[0]));
        same("astype f32->f16", in, () -> new TornadoNativeArray[] { new HalfFloatArray(n) },
                (g, id, c, o) -> g.libraryTask(id, (a, b) -> tune(MlxShape.astype(a, b), c), x, (HalfFloatArray) o[0]));
        same("astype f32->i32", in, () -> new TornadoNativeArray[] { new IntArray(n) }, (g, id, c, o) -> g.libraryTask(id, (a, b) -> tune(MlxShape.astype(a, b), c), x, (IntArray) o[0]));
        same("view f32->i32", in, () -> new TornadoNativeArray[] { new IntArray(n) }, (g, id, c, o) -> g.libraryTask(id, (a, b) -> tune(MlxShape.view(a, b), c), x, (IntArray) o[0]));
        same("numberOfElements", in, () -> new TornadoNativeArray[] { new IntArray(1) },
                (g, id, c, o) -> g.libraryTask(id, (a, b) -> tune(MlxShape.numberOfElements(a, b, d0, d1, d2), c), x, (IntArray) o[0]));
    }

    @Test
    public void testPermutations() throws TornadoExecutionPlanException {
        int d0 = 33;
        int d1 = 17;
        int d2 = 65;
        int n = d0 * d1 * d2;
        FloatArray x = FloatArray.fromArray(values(n, -3, 3, 72));
        Object[] in = { x };
        int[][] perms = { { 2, 0, 1 }, { 1, 2, 0 }, { 0, 2, 1 }, { 2, 1, 0 } };
        for (int[] q : perms) {
            same("transpose " + java.util.Arrays.toString(q), in, floatsOut(n),
                    (g, id, c, o) -> g.libraryTask(id, (a, b) -> tune(MlxShape.transposeAxes(a, b, d0, d1, d2, q[0], q[1], q[2]), c), x, (FloatArray) o[0]));
        }
        same("swapaxes 0,2", in, floatsOut(n), (g, id, c, o) -> g.libraryTask(id, (a, b) -> tune(MlxShape.swapaxes(a, b, d0, d1, d2, 0, 2), c), x, (FloatArray) o[0]));
        same("moveaxis 2->0", in, floatsOut(n), (g, id, c, o) -> g.libraryTask(id, (a, b) -> tune(MlxShape.moveaxis(a, b, d0, d1, d2, 2, 0), c), x, (FloatArray) o[0]));
        same("moveaxis 0->2", in, floatsOut(n), (g, id, c, o) -> g.libraryTask(id, (a, b) -> tune(MlxShape.moveaxis(a, b, d0, d1, d2, 0, 2), c), x, (FloatArray) o[0]));
    }

    @Test
    public void testBroadcastsAndStrides() throws TornadoExecutionPlanException {
        int rows = 129;
        int cols = 67;
        FloatArray row = FloatArray.fromArray(values(cols, -3, 3, 73));
        FloatArray col = FloatArray.fromArray(values(rows, -3, 3, 74));
        same("broadcastTo", new Object[] { row }, floatsOut(rows * cols),
                (g, id, c, o) -> g.libraryTask(id, (a, b) -> tune(MlxShape.broadcastTo(a, b, rows, cols), c), row, (FloatArray) o[0]));
        same("broadcastArrays", new Object[] { row, col }, floatsOut(rows * cols, rows * cols), (g, id, c, o) -> g.libraryTask(id,
                (a, b, p, q) -> tune(MlxShape.broadcastArrays(a, b, p, q, rows, cols), c), row, col, (FloatArray) o[0], (FloatArray) o[1]));
        FloatArray x = FloatArray.fromArray(values(rows * cols, -3, 3, 75));
        same("asStrided", new Object[] { x }, floatsOut(40 * 30),
                (g, id, c, o) -> g.libraryTask(id, (a, b) -> tune(MlxShape.asStrided(a, b, 40, 30, 2 * cols, 2, 5), c), x, (FloatArray) o[0]));
        same("repeat", new Object[] { row }, floatsOut(cols * 3), (g, id, c, o) -> g.libraryTask(id, (a, b) -> tune(MlxShape.repeat(a, b, 3), c), row, (FloatArray) o[0]));
        same("repeatAxis", new Object[] { x }, floatsOut(rows * cols * 2),
                (g, id, c, o) -> g.libraryTask(id, (a, b) -> tune(MlxShape.repeatAxis(a, b, rows, cols, 2), c), x, (FloatArray) o[0]));
        same("tile", new Object[] { x }, floatsOut(rows * cols * 6), (g, id, c, o) -> g.libraryTask(id, (a, b) -> tune(MlxShape.tile(a, b, rows, cols, 2, 3), c), x, (FloatArray) o[0]));
        same("diagonal", new Object[] { x }, floatsOut(cols - 5), (g, id, c, o) -> g.libraryTask(id, (a, b) -> tune(MlxCreate.diagonal(a, b, rows, cols, 5), c), x, (FloatArray) o[0]));
        same("meshgrid xy", new Object[] { row, col }, floatsOut(rows * cols, rows * cols), (g, id, c, o) -> g.libraryTask(id,
                (a, b, p, q) -> tune(MlxCreate.meshgrid(a, b, p, q, false), c), row, col, (FloatArray) o[0], (FloatArray) o[1]));
    }

    @Test
    public void testJoinsSplitsRollsAndPads() throws TornadoExecutionPlanException {
        int rows = 37;
        int ca = 19;
        int cb = 23;
        FloatArray a = FloatArray.fromArray(values(rows * ca, -3, 3, 76));
        FloatArray b = FloatArray.fromArray(values(rows * cb, -3, 3, 77));
        FloatArray b2 = FloatArray.fromArray(values(rows * ca, -3, 3, 78));
        same("concatenate", new Object[] { a, b }, floatsOut(rows * (ca + cb)), (g, id, c, o) -> g.libraryTask(id, (p, q, r) -> tune(MlxShape.concatenate(p, q, r), c), a, b, (FloatArray) o[0]));
        same("concatenateAxis", new Object[] { a, b }, floatsOut(rows * (ca + cb)),
                (g, id, c, o) -> g.libraryTask(id, (p, q, r) -> tune(MlxShape.concatenateAxis(p, q, r, rows, ca, cb), c), a, b, (FloatArray) o[0]));
        same("stack", new Object[] { a, b2 }, floatsOut(2 * rows * ca), (g, id, c, o) -> g.libraryTask(id, (p, q, r) -> tune(MlxShape.stack(p, q, r), c), a, b2, (FloatArray) o[0]));
        same("stackAxis", new Object[] { a, b2 }, floatsOut(2 * rows * ca), (g, id, c, o) -> g.libraryTask(id, (p, q, r) -> tune(MlxShape.stackAxis(p, q, r), c), a, b2, (FloatArray) o[0]));
        FloatArray x = FloatArray.fromArray(values(rows * 2 * ca, -3, 3, 79));
        same("split", new Object[] { x }, floatsOut(rows * ca, rows * ca),
                (g, id, c, o) -> g.libraryTask(id, (p, q, r) -> tune(MlxShape.split(p, q, r, rows, 2 * ca), c), x, (FloatArray) o[0], (FloatArray) o[1]));
        same("splitSections", new Object[] { x }, floatsOut(rows * 7, rows * (2 * ca - 7)),
                (g, id, c, o) -> g.libraryTask(id, (p, q, r) -> tune(MlxShape.splitSections(p, q, r, rows, 2 * ca, 7), c), x, (FloatArray) o[0], (FloatArray) o[1]));
        same("roll", new Object[] { a }, floatsOut(rows * ca), (g, id, c, o) -> g.libraryTask(id, (p, q) -> tune(MlxShape.roll(p, q, -45), c), a, (FloatArray) o[0]));
        same("rollAxis", new Object[] { a }, floatsOut(rows * ca), (g, id, c, o) -> g.libraryTask(id, (p, q) -> tune(MlxShape.rollAxis(p, q, rows, ca, 5), c), a, (FloatArray) o[0]));
        same("rollAxes", new Object[] { a }, floatsOut(rows * ca), (g, id, c, o) -> g.libraryTask(id, (p, q) -> tune(MlxShape.rollAxes(p, q, rows, ca, 40, -3), c), a, (FloatArray) o[0]));
        same("pad", new Object[] { a }, floatsOut((rows + 3) * (ca + 5)),
                (g, id, c, o) -> g.libraryTask(id, (p, q) -> tune(MlxShape.pad(p, q, rows, ca, 1, 2, 3, 2, 1.5f), c), a, (FloatArray) o[0]));
        same("padSymmetric", new Object[] { a }, floatsOut((rows + 4) * (ca + 4)),
                (g, id, c, o) -> g.libraryTask(id, (p, q) -> tune(MlxShape.padSymmetric(p, q, rows, ca, 2, -7.0f), c), a, (FloatArray) o[0]));
    }

    @Test
    public void testConstruction() throws TornadoExecutionPlanException {
        Object[] none = {};
        same("full", none, floatsOut(1001), (g, id, c, o) -> g.libraryTask(id, (p, v) -> tune(MlxCreate.full(p, v), c), (FloatArray) o[0], 3.25f));
        same("full i32", none, () -> new TornadoNativeArray[] { new IntArray(1001) }, (g, id, c, o) -> g.libraryTask(id, (p, v) -> tune(MlxCreate.full(p, v), c), (IntArray) o[0], -7.9f));
        same("zeros", none, floatsOut(70001), (g, id, c, o) -> g.libraryTask(id, p -> tune(MlxCreate.zeros(p), c), (FloatArray) o[0]));
        same("ones", none, floatsOut(70001), (g, id, c, o) -> g.libraryTask(id, p -> tune(MlxCreate.ones(p), c), (FloatArray) o[0]));
        same("arange", none, floatsOut(494), (g, id, c, o) -> g.libraryTask(id, (p, s0, s1, st) -> tune(MlxCreate.arange(p, s0, s1, st), c), (FloatArray) o[0], -3.0f, 120.5f, 0.25f));
        same("arange i32", none, () -> new TornadoNativeArray[] { new IntArray(40) },
                (g, id, c, o) -> g.libraryTask(id, (p, s0, s1, st) -> tune(MlxCreate.arange(p, s0, s1, st), c), (IntArray) o[0], 5f, 125f, 3f));
        same("linspace", none, floatsOut(1001), (g, id, c, o) -> g.libraryTask(id, (p, s0, s1) -> tune(MlxCreate.linspace(p, s0, s1), c), (FloatArray) o[0], -2.5f, 7.75f));
        same("eye", none, floatsOut(37 * 53), (g, id, c, o) -> g.libraryTask(id, (p, n, m, k) -> tune(MlxCreate.eye(p, n, m, k), c), (FloatArray) o[0], 37, 53, -4));
        same("identity", none, floatsOut(64 * 64), (g, id, c, o) -> g.libraryTask(id, (p, n) -> tune(MlxCreate.identity(p, n), c), (FloatArray) o[0], 64));
        same("tri", none, floatsOut(37 * 53), (g, id, c, o) -> g.libraryTask(id, (p, n, m, k) -> tune(MlxCreate.tri(p, n, m, k), c), (FloatArray) o[0], 37, 53, 3));
        for (String w : new String[] { "hanning", "hamming", "blackman", "bartlett" }) {
            same(w, none, floatsOut(4097), (g, id, c, o) -> g.libraryTask(id, p -> tune(window(w, p), c), (FloatArray) o[0]));
        }
        FloatArray v = FloatArray.fromArray(values(50, -3, 3, 81));
        same("diag", new Object[] { v }, floatsOut(53 * 53), (g, id, c, o) -> g.libraryTask(id, (p, q, k) -> tune(MlxCreate.diag(p, q, k), c), v, (FloatArray) o[0], -3));
        FloatArray x = FloatArray.fromArray(values(37 * 53, -3, 3, 82));
        same("tril", new Object[] { x }, floatsOut(37 * 53), (g, id, c, o) -> g.libraryTask(id, (p, q) -> tune(MlxCreate.tril(p, q, 37, 53, 2), c), x, (FloatArray) o[0]));
        same("triu", new Object[] { x }, floatsOut(37 * 53), (g, id, c, o) -> g.libraryTask(id, (p, q) -> tune(MlxCreate.triu(p, q, 37, 53, -1), c), x, (FloatArray) o[0]));
        same("fullLike", new Object[] { x }, floatsOut(37 * 53), (g, id, c, o) -> g.libraryTask(id, (p, q, f) -> tune(MlxCreate.fullLike(p, q, f), c), x, (FloatArray) o[0], 0.3f));
        same("onesLike", new Object[] { x }, floatsOut(37 * 53), (g, id, c, o) -> g.libraryTask(id, (p, q) -> tune(MlxCreate.onesLike(p, q), c), x, (FloatArray) o[0]));
    }

    private static LibraryTaskDescriptor window(String name, FloatArray out) {
        return switch (name) {
            case "hanning" -> MlxCreate.hanning(out);
            case "hamming" -> MlxCreate.hamming(out);
            case "blackman" -> MlxCreate.blackman(out);
            default -> MlxCreate.bartlett(out);
        };
    }

    // ---------------------------------------------------------------- row kernels

    private static Supplier<TornadoNativeArray[]> halvesOut(int... sizes) {
        return () -> {
            TornadoNativeArray[] outs = new TornadoNativeArray[sizes.length];
            for (int i = 0; i < sizes.length; i++) {
                outs[i] = new HalfFloatArray(sizes[i]);
            }
            return outs;
        };
    }

    private static HalfFloatArray halfValues(int n, float lo, float hi, long seed) {
        float[] v = values(n, lo, hi, seed);
        HalfFloatArray a = new HalfFloatArray(n);
        for (int i = 0; i < n; i++) {
            a.set(i, new HalfFloat(v[i]));
        }
        return a;
    }

    @Test
    public void testSoftmaxAndNorms() throws TornadoExecutionPlanException {
        for (int n : new int[] { 1000, 4096, 151936 }) {
            FloatArray x = FloatArray.fromArray(values(n, -8, 8, 90 + n));
            same("softmax " + n, new Object[] { x }, floatsOut(n), (g, id, c, o) -> g.libraryTask(id, (a, b) -> tune(Mlx.softmax(a, b), c), x, (FloatArray) o[0]));
        }
        int rows = 33;
        for (int cols : new int[] { 100, 4096, 8192 }) {
            HalfFloatArray x = halfValues(rows * cols, -6, 6, 91 + cols);
            same("softmaxRows f16 " + cols, new Object[] { x }, halvesOut(rows * cols),
                    (g, id, c, o) -> g.libraryTask(id, (a, b) -> tune(Mlx.softmaxRows(a, b, rows, cols), c), x, (HalfFloatArray) o[0]));
        }
        for (int dim : new int[] { 2048, 4096, 8192 }) {
            FloatArray x = FloatArray.fromArray(values(rows * dim, -3, 3, 92 + dim));
            FloatArray w = FloatArray.fromArray(values(dim, 0.5f, 1.5f, 93 + dim));
            FloatArray bias = FloatArray.fromArray(values(dim, -0.5f, 0.5f, 94 + dim));
            same("rmsNorm " + dim, new Object[] { x, w }, floatsOut(rows * dim),
                    (g, id, c, o) -> g.libraryTask(id, (a, b, q) -> tune(Mlx.rmsNorm(a, b, q, rows, dim, 1e-5f), c), x, w, (FloatArray) o[0]));
            same("layerNorm " + dim, new Object[] { x, w, bias }, floatsOut(rows * dim),
                    (g, id, c, o) -> g.libraryTask(id, (a, b, q, r) -> tune(Mlx.layerNorm(a, b, q, r, rows, dim, 1e-5f), c), x, w, bias, (FloatArray) o[0]));
        }
        HalfFloatArray xh = halfValues(rows * 4096, -3, 3, 95);
        HalfFloatArray wh = halfValues(4096, 0.5f, 1.5f, 96);
        same("rmsNorm f16", new Object[] { xh, wh }, halvesOut(rows * 4096),
                (g, id, c, o) -> g.libraryTask(id, (a, b, q) -> tune(Mlx.rmsNorm(a, b, q, rows, 4096, 1e-6f), c), xh, wh, (HalfFloatArray) o[0]));
    }

    @Test
    public void testRope() throws TornadoExecutionPlanException {
        int[][] shapes = { { 1, 32, 1, 128, 128 }, { 2, 8, 5, 64, 64 }, { 1, 4, 7, 128, 64 } };
        for (int[] sh : shapes) {
            int n = sh[0] * sh[1] * sh[2] * sh[3];
            for (boolean traditional : new boolean[] { false, true }) {
                FloatArray x = FloatArray.fromArray(values(n, -2, 2, 97 + n));
                same("rope " + java.util.Arrays.toString(sh) + " traditional=" + traditional, new Object[] { x }, floatsOut(n), (g, id, c, o) -> g.libraryTask(id,
                        (a, b) -> tune(Mlx.rope(a, b, sh[0], sh[1], sh[2], sh[3], sh[4], traditional, 10000f, 1.0f, 17), c), x, (FloatArray) o[0]));
                HalfFloatArray xh = halfValues(n, -2, 2, 98 + n);
                IntArray offset = IntArray.fromElements(42);
                same("ropeDynamic f16 " + java.util.Arrays.toString(sh) + " traditional=" + traditional, new Object[] { xh, offset }, halvesOut(n), (g, id, c, o) -> g.libraryTask(id,
                        (a, q, b) -> tune(Mlx.ropeDynamic(a, q, b, sh[0], sh[1], sh[2], sh[3], sh[4], traditional, 500000f, 0.5f), c), xh, offset, (HalfFloatArray) o[0]));
            }
        }
    }

    // ---------------------------------------------------------------- reductions

    @Test
    public void testReductions() throws TornadoExecutionPlanException {
        String[] names = { "sum", "prod", "max", "min", "mean", "var", "std" };
        int[] wholes = { 1000, 4096, 4097, 70001, 1 << 22 };
        for (int n : wholes) {
            FloatArray x = FloatArray.fromArray(values(n, n > 5000 ? 0.999f : -2, n > 5000 ? 1.001f : 2, 100 + n));
            for (String name : names) {
                same(name + " whole " + n, new Object[] { x }, floatsOut(1), (g, id, c, o) -> g.libraryTask(id, (a, b) -> tune(reduceWhole(name, a, b), c), x, (FloatArray) o[0]));
            }
            same("logsumexp " + n, new Object[] { x }, floatsOut(1), (g, id, c, o) -> g.libraryTask(id, (a, b) -> tune(MlxReduce.logsumexp(a, b), c), x, (FloatArray) o[0]));
            same("argmax " + n, new Object[] { x }, () -> new TornadoNativeArray[] { new IntArray(1) }, (g, id, c, o) -> g.libraryTask(id, (a, b) -> tune(Mlx.argmax(a, b), c), x, (IntArray) o[0]));
        }
        int[][] rowShapes = { { 7, 50 }, { 40, 60 }, { 20, 3000 }, { 64, 4096 }, { 3, 100000 } };
        for (int[] rs : rowShapes) {
            int rows = rs[0];
            int len = rs[1];
            FloatArray x = FloatArray.fromArray(values(rows * len, -2, 2, 101 + len));
            for (String name : names) {
                same(name + " axis " + rows + "x" + len, new Object[] { x }, floatsOut(rows),
                        (g, id, c, o) -> g.libraryTask(id, (a, b) -> tune(reduceAxis(name, a, b, rows, len), c), x, (FloatArray) o[0]));
            }
            same("argmax rows " + rows + "x" + len, new Object[] { x }, () -> new TornadoNativeArray[] { new IntArray(rows) },
                    (g, id, c, o) -> g.libraryTask(id, (a, b) -> tune(Mlx.argmaxRows(a, b, rows, len), c), x, (IntArray) o[0]));
            same("argmin axis " + rows + "x" + len, new Object[] { x }, () -> new TornadoNativeArray[] { new IntArray(rows) },
                    (g, id, c, o) -> g.libraryTask(id, (a, b) -> tune(MlxReduce.argminAxis(a, b, rows, len, 1), c), x, (IntArray) o[0]));
        }
        FloatArray x = FloatArray.fromArray(values(12 * 30 * 5, -2, 2, 102));
        same("argmin inner axis", new Object[] { x }, () -> new TornadoNativeArray[] { new IntArray(12 * 5) },
                (g, id, c, o) -> g.libraryTask(id, (a, b) -> tune(MlxReduce.argminAxis(a, b, 12, 30, 5), c), x, (IntArray) o[0]));
        same("sum axes", new Object[] { x }, floatsOut(12), (g, id, c, o) -> g.libraryTask(id, (a, b) -> tune(MlxReduce.sumAxes(a, b, 12, 30, 5, 1), c), x, (FloatArray) o[0]));
        IntArray ints = ints(103, -1000, 1000);
        same("sum i32", new Object[] { ints }, () -> new TornadoNativeArray[] { new IntArray(1) }, (g, id, c, o) -> g.libraryTask(id, (a, b) -> tune(MlxReduce.sum(a, b), c), ints, (IntArray) o[0]));
        FloatArray mostlyNonZero = floats(104);
        same("all", new Object[] { mostlyNonZero }, () -> new TornadoNativeArray[] { new ByteArray(1) },
                (g, id, c, o) -> g.libraryTask(id, (a, b) -> tune(MlxReduce.all(a, b), c), mostlyNonZero, (ByteArray) o[0]));
        same("any", new Object[] { mostlyNonZero }, () -> new TornadoNativeArray[] { new ByteArray(1) },
                (g, id, c, o) -> g.libraryTask(id, (a, b) -> tune(MlxReduce.any(a, b), c), mostlyNonZero, (ByteArray) o[0]));
        same("all rows", new Object[] { x }, () -> new TornadoNativeArray[] { new ByteArray(12) },
                (g, id, c, o) -> g.libraryTask(id, (a, b) -> tune(MlxReduce.allAxes(a, b, 12, 30, 5, 1), c), x, (ByteArray) o[0]));
        HalfFloatArray h = halfValues(33 * 4096, -2, 2, 105);
        same("mean f16 rows", new Object[] { h }, halvesOut(33), (g, id, c, o) -> g.libraryTask(id, (a, b) -> tune(MlxReduce.meanAxis(a, b, 33, 4096, 1), c), h, (HalfFloatArray) o[0]));
        same("var ddof=1", new Object[] { x }, floatsOut(1), (g, id, c, o) -> g.libraryTask(id, (a, b, d) -> tune(MlxReduce.var(a, b, d), c), x, (FloatArray) o[0], 1));
    }

    private static LibraryTaskDescriptor reduceWhole(String name, FloatArray a, FloatArray b) {
        return switch (name) {
            case "sum" -> MlxReduce.sum(a, b);
            case "prod" -> MlxReduce.prod(a, b);
            case "max" -> MlxReduce.max(a, b);
            case "min" -> MlxReduce.min(a, b);
            case "mean" -> MlxReduce.mean(a, b);
            case "var" -> MlxReduce.var(a, b, 0);
            default -> MlxReduce.std(a, b, 0);
        };
    }

    private static LibraryTaskDescriptor reduceAxis(String name, FloatArray a, FloatArray b, int rows, int len) {
        return switch (name) {
            case "sum" -> MlxReduce.sumAxis(a, b, rows, len, 1);
            case "prod" -> MlxReduce.prodAxis(a, b, rows, len, 1);
            case "max" -> MlxReduce.maxAxis(a, b, rows, len, 1);
            case "min" -> MlxReduce.minAxis(a, b, rows, len, 1);
            case "mean" -> MlxReduce.meanAxis(a, b, rows, len, 1);
            case "var" -> MlxReduce.varAxis(a, b, rows, len, 1, 0);
            default -> MlxReduce.stdAxis(a, b, rows, len, 1, 2);
        };
    }

    // ---------------------------------------------------------------- slices and composites

    @Test
    public void testSlicesAndComposites() throws TornadoExecutionPlanException {
        int rows = 57;
        int cols = 83;
        FloatArray x = FloatArray.fromArray(values(rows * cols, -3, 3, 110));
        FloatArray upd = FloatArray.fromArray(values(20 * 30, -3, 3, 111));
        same("slice", new Object[] { x }, floatsOut(((50 - 3 + 2) / 3) * ((80 - 5 + 1) / 2)),
                (g, id, c, o) -> g.libraryTask(id, (a, b) -> tune(MlxIndex.slice(a, b, rows, cols, 3, 50, 3, 5, 80, 2), c), x, (FloatArray) o[0]));
        same("sliceUpdate", new Object[] { x, upd }, floatsOut(rows * cols),
                (g, id, c, o) -> g.libraryTask(id, (a, u, b) -> tune(MlxIndex.sliceUpdate(a, u, b, rows, cols, 7, 11, 20, 30), c), x, upd, (FloatArray) o[0]));
        same("sliceUpdateAdd", new Object[] { x, upd }, floatsOut(rows * cols),
                (g, id, c, o) -> g.libraryTask(id, (a, u, b) -> tune(MlxIndex.sliceUpdateAdd(a, u, b, rows, cols, 7, 11, 20, 30), c), x, upd, (FloatArray) o[0]));
        same("sliceUpdateProd", new Object[] { x, upd }, floatsOut(rows * cols),
                (g, id, c, o) -> g.libraryTask(id, (a, u, b) -> tune(MlxIndex.sliceUpdateProd(a, u, b, rows, cols, 0, 53, 20, 30), c), x, upd, (FloatArray) o[0]));
        same("trace", new Object[] { x }, floatsOut(1), (g, id, c, o) -> g.libraryTask(id, (a, b) -> tune(MlxCreate.trace(a, b, rows, cols, 4), c), x, (FloatArray) o[0]));
        FloatArray a = floats(112);
        FloatArray b = floats(113);
        for (int i = 0; i < N; i++) {
            if (i % 5 != 0) {
                b.set(i, a.get(i));
            }
        }
        FloatArray a2 = new FloatArray(N);
        for (int i = 0; i < N; i++) {
            a2.set(i, a.get(i));
        }
        for (boolean equalNan : new boolean[] { false, true }) {
            same("allclose equal_nan=" + equalNan, new Object[] { a, b }, () -> new TornadoNativeArray[] { new ByteArray(1) }, (g, id, c, o) -> g.libraryTask(id,
                    (p, q, r, rt, at, e) -> tune(MlxLogic.allclose(p, q, r, rt, at, e), c), a, b, (ByteArray) o[0], 1e-5f, 1e-8f, equalNan));
            same("arrayEqual equal_nan=" + equalNan, new Object[] { a, a2 }, () -> new TornadoNativeArray[] { new ByteArray(1) },
                    (g, id, c, o) -> g.libraryTask(id, (p, q, r, e) -> tune(MlxLogic.arrayEqual(p, q, r, e), c), a, a2, (ByteArray) o[0], equalNan));
        }
        int d0 = 12;
        int d1 = 30;
        int d2 = 70;
        FloatArray y = FloatArray.fromArray(values(d0 * d1 * d2, -5, 5, 114));
        same("softmaxLastTwoAxes", new Object[] { y }, floatsOut(d0 * d1 * d2),
                (g, id, c, o) -> g.libraryTask(id, (p, q) -> tune(Mlx.softmaxLastTwoAxes(p, q, d0, d1, d2), c), y, (FloatArray) o[0]));
        HalfFloatArray yh = halfValues(d0 * d1 * d2, -5, 5, 115);
        same("softmaxLastTwoAxes f16", new Object[] { yh }, halvesOut(d0 * d1 * d2),
                (g, id, c, o) -> g.libraryTask(id, (p, q) -> tune(Mlx.softmaxLastTwoAxes(p, q, d0, d1, d2), c), yh, (HalfFloatArray) o[0]));
        FloatArray z = FloatArray.fromArray(values(70000, -5, 5, 116));
        same("logsumexpAxis", new Object[] { z }, floatsOut(7), (g, id, c, o) -> g.libraryTask(id, (p, q) -> tune(MlxReduce.logsumexpAxis(p, q, 7, 10000, 1), c), z, (FloatArray) o[0]));
        same("logsumexpAxes", new Object[] { y }, floatsOut(d0),
                (g, id, c, o) -> g.libraryTask(id, (p, q) -> tune(MlxReduce.logsumexpAxes(p, q, d0, d1, d2, 1), c), y, (FloatArray) o[0]));
    }

    // ---------------------------------------------------------------- matmul

    @Test
    public void testMatmul() throws TornadoExecutionPlanException {
        // {m, k, n}: decode GEMV both ways, a column GEMV, split-K, and regular GEMMs, aligned and not.
        int[][] shapes = { { 1, 4096, 4096 }, { 1, 2048, 311 }, { 1, 64, 1000 }, { 300, 512, 1 }, { 32, 4096, 32 }, { 64, 2048, 96 }, { 128, 512, 256 }, { 100, 300, 70 },
                { 512, 1024, 1024 } };
        for (int[] sh : shapes) {
            int m = sh[0];
            int k = sh[1];
            int n = sh[2];
            FloatArray a = FloatArray.fromArray(values(m * k, -1, 1, 120 + m + k));
            FloatArray b = FloatArray.fromArray(values(k * n, -1, 1, 121 + k + n));
            same("matmul " + m + "x" + k + "x" + n, new Object[] { a, b }, floatsOut(m * n),
                    (g, id, c, o) -> g.libraryTask(id, (p, q, r) -> tune(Mlx.matmul(p, q, r, m, k, n), c), a, b, (FloatArray) o[0]));
            same("matmulTransposed " + m + "x" + k + "x" + n, new Object[] { a, b }, floatsOut(m * n),
                    (g, id, c, o) -> g.libraryTask(id, (p, q, r) -> tune(Mlx.matmulTransposed(p, q, r, m, k, n), c), a, b, (FloatArray) o[0]));
            HalfFloatArray ah = halfValues(m * k, -1, 1, 122 + m);
            HalfFloatArray bh = halfValues(k * n, -1, 1, 123 + n);
            same("matmul f16 " + m + "x" + k + "x" + n, new Object[] { ah, bh }, halvesOut(m * n),
                    (g, id, c, o) -> g.libraryTask(id, (p, q, r) -> tune(Mlx.matmul(p, q, r, m, k, n), c), ah, bh, (HalfFloatArray) o[0]));
        }
    }

    // ---------------------------------------------------------------- quantized

    @Test
    public void testQuantized() throws TornadoExecutionPlanException {
        // {m, k, n, group, bits}: qmv_fast, qmv (unaligned k), qmv_quad (k = 128), a small batch, 8-bit.
        int[][] shapes = { { 1, 4096, 4096, 64, 4 }, { 1, 576, 1000, 64, 4 }, { 1, 128, 512, 64, 4 }, { 5, 2048, 1024, 32, 4 }, { 1, 2048, 2048, 64, 8 },
                { 128, 2048, 2048, 64, 4 }, { 64, 4096, 1000, 64, 4 }, { 512, 1024, 2048, 64, 4 }, { 32, 4096, 128, 32, 8 } };
        for (int[] sh : shapes) {
            int m = sh[0];
            int k = sh[1];
            int n = sh[2];
            int gs = sh[3];
            int bits = sh[4];
            FloatArray x = FloatArray.fromArray(values(m * k, -1, 1, 130 + k));
            IntArray wq = ints(131 + n, -1000000000, 1000000000);
            IntArray w = new IntArray(n * k * bits / 32);
            for (int i = 0; i < w.getSize(); i++) {
                w.set(i, wq.get(i % wq.getSize()) * 31 + i);
            }
            FloatArray scales = FloatArray.fromArray(values(n * (k / gs), 0.001f, 0.02f, 132 + n));
            FloatArray biases = FloatArray.fromArray(values(n * (k / gs), -0.1f, 0.1f, 133 + n));
            same("quantizedMatmul " + java.util.Arrays.toString(sh), new Object[] { x, w, scales, biases }, floatsOut(m * n), (g, id, c, o) -> g.libraryTask(id,
                    (p, q, r, t, y) -> tune(Mlx.quantizedMatmul(p, q, r, t, y, m, k, n, gs, bits), c), x, w, scales, biases, (FloatArray) o[0]));
            HalfFloatArray xh = halfValues(m * k, -1, 1, 134 + k);
            HalfFloatArray sh16 = halfValues(n * (k / gs), 0.001f, 0.02f, 135 + n);
            HalfFloatArray bh16 = halfValues(n * (k / gs), -0.1f, 0.1f, 136 + n);
            same("quantizedMatmul f16 " + java.util.Arrays.toString(sh), new Object[] { xh, w, sh16, bh16 }, halvesOut(m * n), (g, id, c, o) -> g.libraryTask(id,
                    (p, q, r, t, y) -> tune(Mlx.quantizedMatmul(p, q, r, t, y, m, k, n, gs, bits), c), xh, w, sh16, bh16, (HalfFloatArray) o[0]));
        }
        int rows = 96;
        int cols = 1024;
        FloatArray weights = FloatArray.fromArray(values(rows * cols, -1, 1, 137));
        for (int bits : new int[] { 4, 8 }) {
            int gs = 64;
            same("quantize " + bits, new Object[] { weights },
                    () -> new TornadoNativeArray[] { new IntArray(rows * cols * bits / 32), new FloatArray(rows * cols / gs), new FloatArray(rows * cols / gs) },
                    (g, id, c, o) -> g.libraryTask(id, (p, q, r, t) -> tune(Mlx.quantize(p, q, r, t, rows, cols, gs, bits), c), weights, (IntArray) o[0], (FloatArray) o[1],
                            (FloatArray) o[2]));
            IntArray packed = ints(138 + bits, -1000000000, 1000000000);
            IntArray pw = new IntArray(rows * cols * bits / 32);
            for (int i = 0; i < pw.getSize(); i++) {
                pw.set(i, packed.get(i % packed.getSize()) ^ (i * 2654435761L > 0 ? i : -i));
            }
            FloatArray s1 = FloatArray.fromArray(values(rows * cols / gs, 0.001f, 0.02f, 139));
            FloatArray b1 = FloatArray.fromArray(values(rows * cols / gs, -0.1f, 0.1f, 140));
            same("dequantize " + bits, new Object[] { pw, s1, b1 }, floatsOut(rows * cols),
                    (g, id, c, o) -> g.libraryTask(id, (p, q, r, t) -> tune(Mlx.dequantize(p, q, r, t, rows, cols, gs, bits), c), pw, s1, b1, (FloatArray) o[0]));
        }
    }

    // ---------------------------------------------------------------- attention

    @Test
    public void testAttention() throws TornadoExecutionPlanException {
        // {batch, qHeads, kvHeads, qLen, kvLen, headDim, causal}: one-pass and two-pass decode, grouped queries, a causal query block.
        int[][] shapes = { { 1, 32, 8, 1, 512, 128, 0 }, { 1, 32, 8, 1, 2048, 128, 0 }, { 2, 8, 8, 1, 300, 64, 0 }, { 1, 8, 2, 4, 100, 128, 1 }, { 1, 16, 16, 1, 9000, 64, 0 },
                { 1, 8, 2, 128, 128, 128, 1 }, { 2, 4, 4, 100, 150, 64, 0 }, { 1, 8, 8, 77, 77, 64, 1 }, { 1, 4, 1, 33, 200, 128, 1 } };
        for (int[] sh : shapes) {
            int b = sh[0];
            int hq = sh[1];
            int hkv = sh[2];
            int lq = sh[3];
            int lk = sh[4];
            int d = sh[5];
            boolean causal = sh[6] == 1;
            float scale = (float) (1.0 / Math.sqrt(d));
            FloatArray q = FloatArray.fromArray(values(b * hq * lq * d, -1, 1, 150 + lk));
            FloatArray k = FloatArray.fromArray(values(b * hkv * lk * d, -1, 1, 151 + lk));
            FloatArray vv = FloatArray.fromArray(values(b * hkv * lk * d, -1, 1, 152 + lk));
            same("sdpa " + java.util.Arrays.toString(sh), new Object[] { q, k, vv }, floatsOut(b * hq * lq * d), (g, id, c, o) -> g.libraryTask(id,
                    (x1, x2, x3, x4) -> tune(Mlx.scaledDotProductAttention(x1, x2, x3, x4, b, hq, hkv, lq, lk, d, scale, causal), c), q, k, vv, (FloatArray) o[0]));
            HalfFloatArray qh = halfValues(b * hq * lq * d, -1, 1, 153 + lk);
            HalfFloatArray kh = halfValues(b * hkv * lk * d, -1, 1, 154 + lk);
            HalfFloatArray vh = halfValues(b * hkv * lk * d, -1, 1, 155 + lk);
            same("sdpa f16 " + java.util.Arrays.toString(sh), new Object[] { qh, kh, vh }, halvesOut(b * hq * lq * d), (g, id, c, o) -> g.libraryTask(id,
                    (x1, x2, x3, x4) -> tune(Mlx.scaledDotProductAttention(x1, x2, x3, x4, b, hq, hkv, lq, lk, d, scale, causal), c), qh, kh, vh, (HalfFloatArray) o[0]));
        }
    }

    // ---------------------------------------------------------------- scans

    @Test
    public void testScans() throws TornadoExecutionPlanException {
        // {outer, len, inner}: short and long rows, a single long row, and strided scans.
        int[][] shapes = { { 64, 100, 1 }, { 33, 4096, 1 }, { 8, 7000, 1 }, { 1, 100000, 1 }, { 16, 300, 64 }, { 4, 50, 7 } };
        String[] names = { "cumsum", "cumprod", "cummax", "cummin", "logcumsumexp" };
        for (int[] sh : shapes) {
            int n = sh[0] * sh[1] * sh[2];
            FloatArray x = FloatArray.fromArray(values(n, 0.99f, 1.01f, 160 + n));
            for (String name : names) {
                for (boolean reverse : new boolean[] { false, true }) {
                    boolean inclusive = !reverse || sh[1] % 2 == 0;
                    same(name + " " + java.util.Arrays.toString(sh) + " reverse=" + reverse + " inclusive=" + inclusive, new Object[] { x }, floatsOut(n), (g, id, c, o) -> g
                            .libraryTask(id, (a, b) -> tune(scanTask(name, a, b, sh[0], sh[1], sh[2], reverse, inclusive), c), x, (FloatArray) o[0]));
                }
            }
        }
    }

    private static LibraryTaskDescriptor scanTask(String name, FloatArray a, FloatArray b, int outer, int len, int inner, boolean reverse, boolean inclusive) {
        return switch (name) {
            case "cumsum" -> MlxReduce.cumsum(a, b, outer, len, inner, reverse, inclusive);
            case "cumprod" -> MlxReduce.cumprod(a, b, outer, len, inner, reverse, inclusive);
            case "cummax" -> MlxReduce.cummax(a, b, outer, len, inner, reverse, inclusive);
            case "cummin" -> MlxReduce.cummin(a, b, outer, len, inner, reverse, inclusive);
            default -> MlxReduce.logcumsumexp(a, b, outer, len, inner, reverse, inclusive);
        };
    }

    // ---------------------------------------------------------------- sorting

    @Test
    public void testSorting() throws TornadoExecutionPlanException {
        for (int n : new int[] { 100, 1500, 2048, 5000, 100000 }) {
            float[] raw = values(n, -50, 50, 170 + n);
            for (int i = 0; i < n; i += 9) {
                raw[i] = Math.round(raw[i]);
            }
            FloatArray x = FloatArray.fromArray(raw);
            same("sort " + n, new Object[] { x }, floatsOut(n), (g, id, c, o) -> g.libraryTask(id, (a, b) -> tune(MlxSort.sort(a, b), c), x, (FloatArray) o[0]));
            same("argsort " + n, new Object[] { x }, () -> new TornadoNativeArray[] { new IntArray(n) },
                    (g, id, c, o) -> g.libraryTask(id, (a, b) -> tune(MlxSort.argsort(a, b), c), x, (IntArray) o[0]));
            same("partition " + n, new Object[] { x }, floatsOut(n), (g, id, c, o) -> g.libraryTask(id, (a, b, k) -> tune(MlxSort.partition(a, b, k), c), x, (FloatArray) o[0], n / 3));
            same("topk " + n, new Object[] { x }, floatsOut(Math.min(50, n)), (g, id, c, o) -> g.libraryTask(id, (a, b, k) -> tune(Mlx.topk(a, b, k), c), x, (FloatArray) o[0], Math.min(50, n)));
        }
        // {outer, len, inner}: rows, strided axes, and multi-block rows and strided axes.
        int[][] shapes = { { 64, 300, 1 }, { 8, 200, 5 }, { 3, 6000, 1 }, { 2, 4100, 3 } };
        for (int[] sh : shapes) {
            int n = sh[0] * sh[1] * sh[2];
            FloatArray x = FloatArray.fromArray(values(n, -50, 50, 171 + n));
            same("sortAxis " + java.util.Arrays.toString(sh), new Object[] { x }, floatsOut(n),
                    (g, id, c, o) -> g.libraryTask(id, (a, b) -> tune(MlxSort.sortAxis(a, b, sh[0], sh[1], sh[2]), c), x, (FloatArray) o[0]));
            same("argsortAxis " + java.util.Arrays.toString(sh), new Object[] { x }, () -> new TornadoNativeArray[] { new IntArray(n) },
                    (g, id, c, o) -> g.libraryTask(id, (a, b) -> tune(MlxSort.argsortAxis(a, b, sh[0], sh[1], sh[2]), c), x, (IntArray) o[0]));
        }
        FloatArray rows = FloatArray.fromArray(values(32 * 32000, -10, 10, 172));
        same("topkRows", new Object[] { rows }, floatsOut(32 * 40), (g, id, c, o) -> g.libraryTask(id, (a, b) -> tune(Mlx.topkRows(a, b, 32, 32000, 40), c), rows, (FloatArray) o[0]));
    }

    // ---------------------------------------------------------------- products

    @Test
    public void testProducts() throws TornadoExecutionPlanException {
        int[][] shapes = { { 1, 2048, 1000 }, { 32, 4096, 32 }, { 100, 300, 70 }, { 128, 512, 256 } };
        float[][] ab = { { 1.0f, 0.0f }, { 0.5f, 2.0f }, { 1.0f, 1.0f }, { 0.0f, 3.0f } };
        for (int[] sh : shapes) {
            int m = sh[0];
            int k = sh[1];
            int n = sh[2];
            FloatArray a = FloatArray.fromArray(values(m * k, -1, 1, 180 + m));
            FloatArray b = FloatArray.fromArray(values(k * n, -1, 1, 181 + n));
            FloatArray cIn = FloatArray.fromArray(values(m * n, -1, 1, 182 + k));
            for (float[] coefficients : ab) {
                same("addmm " + java.util.Arrays.toString(sh) + " alpha=" + coefficients[0] + " beta=" + coefficients[1], new Object[] { cIn, a, b }, floatsOut(m * n),
                        (g, id, c, o) -> g.libraryTask(id, (x1, x2, x3, x4) -> tune(Mlx.addmm(x1, x2, x3, x4, m, k, n, coefficients[0], coefficients[1]), c), cIn, a, b,
                                (FloatArray) o[0]));
            }
            same("tensordotAxis " + java.util.Arrays.toString(sh), new Object[] { a, b }, floatsOut(m * n),
                    (g, id, c, o) -> g.libraryTask(id, (x1, x2, x3) -> tune(MlxProducts.tensordotAxis(x1, x2, x3, m, k, n), c), a, b, (FloatArray) o[0]));
        }
        FloatArray t1 = FloatArray.fromArray(values(20 * 16 * 32, -1, 1, 183));
        FloatArray t2 = FloatArray.fromArray(values(16 * 32 * 50, -1, 1, 184));
        same("tensordot", new Object[] { t1, t2 }, floatsOut(20 * 50),
                (g, id, c, o) -> g.libraryTask(id, (x1, x2, x3) -> tune(MlxProducts.tensordot(x1, x2, x3, 20, 16, 32, 50), c), t1, t2, (FloatArray) o[0]));
        for (int[] bs : new int[][] { { 8, 64, 128, 96 }, { 4, 1, 256, 300 }, { 3, 50, 70, 1 } }) {
            FloatArray e1 = FloatArray.fromArray(values(bs[0] * bs[1] * bs[2], -1, 1, 185 + bs[1]));
            FloatArray e2 = FloatArray.fromArray(values(bs[0] * bs[2] * bs[3], -1, 1, 186 + bs[3]));
            same("einsumBmm " + java.util.Arrays.toString(bs), new Object[] { e1, e2 }, floatsOut(bs[0] * bs[1] * bs[3]), (g, id, c, o) -> g.libraryTask(id,
                    (x1, x2, x3) -> tune(MlxProducts.einsumBatchedMatmul(x1, x2, x3, bs[0], bs[1], bs[2], bs[3]), c), e1, e2, (FloatArray) o[0]));
        }
        FloatArray u = FloatArray.fromArray(values(333, -1, 1, 187));
        FloatArray w = FloatArray.fromArray(values(257, -1, 1, 188));
        same("outer", new Object[] { u, w }, floatsOut(333 * 257), (g, id, c, o) -> g.libraryTask(id, (x1, x2, x3) -> tune(MlxProducts.outer(x1, x2, x3), c), u, w, (FloatArray) o[0]));
        FloatArray w2 = FloatArray.fromArray(values(333, -1, 1, 189));
        FloatArray big = FloatArray.fromArray(values(100000, -1, 1, 190));
        FloatArray big2 = FloatArray.fromArray(values(100000, -1, 1, 191));
        same("inner", new Object[] { u, w2 }, floatsOut(1), (g, id, c, o) -> g.libraryTask(id, (x1, x2, x3) -> tune(MlxProducts.inner(x1, x2, x3), c), u, w2, (FloatArray) o[0]));
        same("inner large", new Object[] { big, big2 }, floatsOut(1), (g, id, c, o) -> g.libraryTask(id, (x1, x2, x3) -> tune(MlxProducts.inner(x1, x2, x3), c), big, big2, (FloatArray) o[0]));
        FloatArray ka = FloatArray.fromArray(values(9 * 7, -1, 1, 192));
        FloatArray kb = FloatArray.fromArray(values(13 * 5, -1, 1, 193));
        same("kron", new Object[] { ka, kb }, floatsOut(9 * 7 * 13 * 5),
                (g, id, c, o) -> g.libraryTask(id, (x1, x2, x3) -> tune(MlxProducts.kron(x1, x2, x3, 9, 7, 13, 5), c), ka, kb, (FloatArray) o[0]));
    }

    // ---------------------------------------------------------------- random sampling

    @Test
    public void testRandom() throws TornadoExecutionPlanException {
        Object[] none = {};
        for (int seed : new int[] { 7, -12345 }) {
            for (int n : new int[] { 1, 1001, 70000 }) {
                String tag = " n=" + n + " seed=" + seed;
                same("bits" + tag, none, () -> new TornadoNativeArray[] { new IntArray(n) }, (g, id, c, o) -> g.libraryTask(id, (a, sd) -> tune(MlxRandom.bits(a, sd), c), (IntArray) o[0], seed));
                same("uniform" + tag, none, floatsOut(n), (g, id, c, o) -> g.libraryTask(id, (a, lo, hi, sd) -> tune(MlxRandom.uniform(a, lo, hi, sd), c), (FloatArray) o[0], -2.5f, 3.75f, seed));
                same("normal" + tag, none, floatsOut(n), (g, id, c, o) -> g.libraryTask(id, (a, l, sc, sd) -> tune(MlxRandom.normal(a, l, sc, sd), c), (FloatArray) o[0], 1.5f, 0.3f, seed));
                same("normal std" + tag, none, floatsOut(n), (g, id, c, o) -> g.libraryTask(id, (a, l, sc, sd) -> tune(MlxRandom.normal(a, l, sc, sd), c), (FloatArray) o[0], 0.0f, 1.0f, seed));
                same("randint" + tag, none, () -> new TornadoNativeArray[] { new IntArray(n) },
                        (g, id, c, o) -> g.libraryTask(id, (a, lo, hi, sd) -> tune(MlxRandom.randint(a, lo, hi, sd), c), (IntArray) o[0], -7, 100, seed));
                same("truncatedNormal" + tag, none, floatsOut(n),
                        (g, id, c, o) -> g.libraryTask(id, (a, lo, hi, sd) -> tune(MlxRandom.truncatedNormal(a, lo, hi, sd), c), (FloatArray) o[0], -0.5f, 1.25f, seed));
                same("gumbel" + tag, none, floatsOut(n), (g, id, c, o) -> g.libraryTask(id, (a, sd) -> tune(MlxRandom.gumbel(a, sd), c), (FloatArray) o[0], seed));
                same("laplace" + tag, none, floatsOut(n), (g, id, c, o) -> g.libraryTask(id, (a, l, sc, sd) -> tune(MlxRandom.laplace(a, l, sc, sd), c), (FloatArray) o[0], 0.5f, 2.0f, seed));
                same("permutationArange" + tag, none, () -> new TornadoNativeArray[] { new IntArray(n) },
                        (g, id, c, o) -> g.libraryTask(id, (a, sd) -> tune(MlxRandom.permutationArange(a, sd), c), (IntArray) o[0], seed));
                FloatArray probabilities = FloatArray.fromArray(values(n, 0, 1, 200 + n));
                same("bernoulli" + tag, new Object[] { probabilities }, () -> new TornadoNativeArray[] { new ByteArray(n) },
                        (g, id, c, o) -> g.libraryTask(id, (a, b, sd) -> tune(MlxRandom.bernoulli(a, b, sd), c), probabilities, (ByteArray) o[0], seed));
                FloatArray loc = FloatArray.fromArray(values(n, -1, 1, 201 + n));
                FloatArray scale = FloatArray.fromArray(values(n, 0.1f, 2, 202 + n));
                same("normalBroadcast" + tag, new Object[] { loc, scale }, floatsOut(n),
                        (g, id, c, o) -> g.libraryTask(id, (a, b, d, sd) -> tune(MlxRandom.normalBroadcast(a, b, d, sd), c), loc, scale, (FloatArray) o[0], seed));
            }
            FloatArray logits = FloatArray.fromArray(values(16 * 1000, -3, 3, 203));
            same("categorical seed=" + seed, new Object[] { logits }, () -> new TornadoNativeArray[] { new IntArray(16) },
                    (g, id, c, o) -> g.libraryTask(id, (a, b, r, k, sd) -> tune(MlxRandom.categorical(a, b, r, k, sd), c), logits, (IntArray) o[0], 16, 1000, seed));
            same("categoricalSamples seed=" + seed, new Object[] { logits }, () -> new TornadoNativeArray[] { new IntArray(16 * 5) }, (g, id, c, o) -> g.libraryTask(id,
                    (a, b, r, k, m, sd) -> tune(MlxRandom.categoricalSamples(a, b, r, k, m, sd), c), logits, (IntArray) o[0], 16, 1000, 5, seed));
            same("categoricalShape seed=" + seed, new Object[] { logits }, () -> new TornadoNativeArray[] { new IntArray(16 * 3) }, (g, id, c, o) -> g.libraryTask(id,
                    (a, b, r, k, m, sd) -> tune(MlxRandom.categoricalShape(a, b, r, k, m, sd), c), logits, (IntArray) o[0], 16, 1000, 3, seed));
        }
    }

    // ---------------------------------------------------------------- fft helpers, norms, cross

    @Test
    public void testSmallComposites() throws TornadoExecutionPlanException {
        int rows = 13;
        for (int cols : new int[] { 64, 101 }) {
            FloatArray x = FloatArray.fromArray(values(rows * cols, -3, 3, 210 + cols));
            same("fftshift " + cols, new Object[] { x }, floatsOut(rows * cols),
                    (g, id, c, o) -> g.libraryTask(id, (a, b) -> tune(MlxFft.fftshift(a, b, rows, cols), c), x, (FloatArray) o[0]));
            same("ifftshift " + cols, new Object[] { x }, floatsOut(rows * cols),
                    (g, id, c, o) -> g.libraryTask(id, (a, b) -> tune(MlxFft.ifftshift(a, b, rows, cols), c), x, (FloatArray) o[0]));
        }
        Object[] none = {};
        for (int n : new int[] { 1, 8, 1001 }) {
            same("fftfreq " + n, none, floatsOut(n), (g, id, c, o) -> g.libraryTask(id, (a, m, d) -> tune(MlxFft.fftfreq(a, m, d), c), (FloatArray) o[0], n, 0.37f));
            same("rfftfreq " + n, none, floatsOut(n / 2 + 1), (g, id, c, o) -> g.libraryTask(id, (a, m, d) -> tune(MlxFft.rfftfreq(a, m, d), c), (FloatArray) o[0], n, 0.37f));
        }
        int r = 7;
        int c = 10000;
        FloatArray mat = FloatArray.fromArray(values(r * c, -3, 3, 212));
        for (int i = 0; i < r * c; i += 11) {
            mat.set(i, 0.0f);
        }
        for (float ord : new float[] { 0, 1, 2, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, 3.5f }) {
            same("norm ord=" + ord, new Object[] { mat }, floatsOut(r), (g, id, cc, o) -> g.libraryTask(id, (a, b, q, w, e) -> tune(MlxLinalg.norm(a, b, q, w, e), cc), mat,
                    (FloatArray) o[0], r, c, ord));
        }
        same("l2Norm", new Object[] { mat }, floatsOut(r), (g, id, cc, o) -> g.libraryTask(id, (a, b) -> tune(MlxLinalg.l2Norm(a, b, r, c), cc), mat, (FloatArray) o[0]));
        FloatArray batch = FloatArray.fromArray(values(6 * 40 * 50, -3, 3, 213));
        same("frobeniusNorm", new Object[] { batch }, floatsOut(6),
                (g, id, cc, o) -> g.libraryTask(id, (a, b) -> tune(MlxLinalg.frobeniusNorm(a, b, 6, 40, 50), cc), batch, (FloatArray) o[0]));
        FloatArray va = FloatArray.fromArray(values(3 * 5001, -3, 3, 214));
        FloatArray vb = FloatArray.fromArray(values(3 * 5001, -3, 3, 215));
        same("cross", new Object[] { va, vb }, floatsOut(3 * 5001), (g, id, cc, o) -> g.libraryTask(id, (a, b, q) -> tune(MlxLinalg.cross(a, b, q, 5001), cc), va, vb, (FloatArray) o[0]));
    }

    // ---------------------------------------------------------------- FFT

    @Test
    public void testFft() throws TornadoExecutionPlanException {
        int rows = 9;
        for (int n : new int[] { 8, 64, 100, 1000, 1024, 2187, 4096, 7 }) {
            FloatArray z = FloatArray.fromArray(values(rows * n * 2, -1, 1, 220 + n));
            FloatArray real = FloatArray.fromArray(values(rows * n, -1, 1, 221 + n));
            FloatArray half = FloatArray.fromArray(values(rows * (n / 2 + 1) * 2, -1, 1, 222 + n));
            for (int norm = 0; norm < 3; norm++) {
                int nm = norm;
                String tag = " n=" + n + " norm=" + nm;
                same("fft" + tag, new Object[] { z }, floatsOut(rows * n * 2), (g, id, c, o) -> g.libraryTask(id, (a, b) -> tune(MlxFft.fft(a, b, rows, n, n, nm), c), z, (FloatArray) o[0]));
                same("ifft" + tag, new Object[] { z }, floatsOut(rows * n * 2), (g, id, c, o) -> g.libraryTask(id, (a, b) -> tune(MlxFft.ifft(a, b, rows, n, n, nm), c), z, (FloatArray) o[0]));
                same("rfft" + tag, new Object[] { real }, floatsOut(rows * (n / 2 + 1) * 2),
                        (g, id, c, o) -> g.libraryTask(id, (a, b) -> tune(MlxFft.rfft(a, b, rows, n, n, nm), c), real, (FloatArray) o[0]));
                same("irfft" + tag, new Object[] { half }, floatsOut(rows * n),
                        (g, id, c, o) -> g.libraryTask(id, (a, b) -> tune(MlxFft.irfft(a, b, rows, n / 2 + 1, n, nm), c), half, (FloatArray) o[0]));
            }
        }
    }

    @Test
    public void testFftNd() throws TornadoExecutionPlanException {
        int[][] shapes2 = { { 3, 16, 32 }, { 2, 60, 100 }, { 1, 256, 8 } };
        for (int[] sh : shapes2) {
            int b = sh[0];
            int h = sh[1];
            int w = sh[2];
            FloatArray z = FloatArray.fromArray(values(b * h * w * 2, -1, 1, 230 + h));
            FloatArray real = FloatArray.fromArray(values(b * h * w, -1, 1, 231 + h));
            FloatArray half = FloatArray.fromArray(values(b * h * (w / 2 + 1) * 2, -1, 1, 232 + h));
            for (int norm : new int[] { 0, 1 }) {
                String tag = " " + java.util.Arrays.toString(sh) + " norm=" + norm;
                same("fft2" + tag, new Object[] { z }, floatsOut(b * h * w * 2), (g, id, c, o) -> g.libraryTask(id, (x, y) -> tune(MlxFft.fft2(x, y, b, h, w, norm), c), z, (FloatArray) o[0]));
                same("ifft2" + tag, new Object[] { z }, floatsOut(b * h * w * 2), (g, id, c, o) -> g.libraryTask(id, (x, y) -> tune(MlxFft.ifft2(x, y, b, h, w, norm), c), z, (FloatArray) o[0]));
                same("rfft2" + tag, new Object[] { real }, floatsOut(b * h * (w / 2 + 1) * 2),
                        (g, id, c, o) -> g.libraryTask(id, (x, y) -> tune(MlxFft.rfft2(x, y, b, h, w, norm), c), real, (FloatArray) o[0]));
                same("irfft2" + tag, new Object[] { half }, floatsOut(b * h * w),
                        (g, id, c, o) -> g.libraryTask(id, (x, y) -> tune(MlxFft.irfft2(x, y, b, h, w, norm), c), half, (FloatArray) o[0]));
            }
        }
        int b = 2;
        int d = 8;
        int h = 12;
        int w = 16;
        FloatArray z = FloatArray.fromArray(values(b * d * h * w * 2, -1, 1, 233));
        FloatArray real = FloatArray.fromArray(values(b * d * h * w, -1, 1, 234));
        FloatArray half = FloatArray.fromArray(values(b * d * h * (w / 2 + 1) * 2, -1, 1, 235));
        same("fftn", new Object[] { z }, floatsOut(b * d * h * w * 2), (g, id, c, o) -> g.libraryTask(id, (x, y) -> tune(MlxFft.fftn(x, y, b, d, h, w, 0), c), z, (FloatArray) o[0]));
        same("ifftn", new Object[] { z }, floatsOut(b * d * h * w * 2), (g, id, c, o) -> g.libraryTask(id, (x, y) -> tune(MlxFft.ifftn(x, y, b, d, h, w, 2), c), z, (FloatArray) o[0]));
        same("rfftn", new Object[] { real }, floatsOut(b * d * h * (w / 2 + 1) * 2),
                (g, id, c, o) -> g.libraryTask(id, (x, y) -> tune(MlxFft.rfftn(x, y, b, d, h, w, 0), c), real, (FloatArray) o[0]));
        same("irfftn", new Object[] { half }, floatsOut(b * d * h * w),
                (g, id, c, o) -> g.libraryTask(id, (x, y) -> tune(MlxFft.irfftn(x, y, b, d, h, w, 0), c), half, (FloatArray) o[0]));
    }

    // ---------------------------------------------------------------- convolutions

    private static int convLength(int in, int k, int stride, int lo, int hi, int kd, int id) {
        return ((id * (in - 1) + 1) + lo + hi - (kd * (k - 1) + 1)) / stride + 1;
    }

    @Test
    public void testConvolutions() throws TornadoExecutionPlanException {
        // conv1d {n, len, cin, cout, k, stride, pad, dil, groups}: implicit GEMM, depthwise, grouped, small channels, strided and dilated.
        int[][] c1 = { { 8, 512, 64, 64, 5, 1, 2, 1, 1 }, { 2, 300, 32, 32, 3, 1, 0, 1, 32 }, { 2, 200, 64, 64, 3, 1, 1, 1, 4 }, { 3, 100, 3, 16, 7, 2, 2, 2, 1 },
                { 1, 64, 16, 8, 3, 1, 1, 1, 1 } };
        for (int[] q : c1) {
            int outLen = convLength(q[1], q[4], q[5], q[6], q[6], q[7], 1);
            FloatArray x = FloatArray.fromArray(values(q[0] * q[1] * q[2], -1, 1, 240 + q[1]));
            FloatArray w = FloatArray.fromArray(values(q[3] * q[4] * (q[2] / q[8]), -1, 1, 241 + q[3]));
            same("conv1d " + java.util.Arrays.toString(q), new Object[] { x, w }, floatsOut(q[0] * outLen * q[3]), (g, id, c, o) -> g.libraryTask(id,
                    (a, b, r, i3, i4, i5, i6, i7, i8, i9, i10, i11) -> tune(MlxConv.conv1d(a, b, r, i3, i4, i5, i6, i7, i8, i9, i10, i11), c), x, w, (FloatArray) o[0], q[0], q[1], q[2],
                    q[3], q[4], q[5], q[6], q[7], q[8]));
        }
        // conv2d {n, h, w, cin, cout, kh, kw, stride, pad, dil, groups}.
        int[][] c2 = { { 1, 32, 32, 16, 32, 3, 3, 1, 1, 1, 1 }, { 2, 17, 19, 3, 8, 5, 5, 1, 2, 1, 1 }, { 1, 40, 40, 32, 64, 3, 3, 2, 1, 1, 2 }, { 1, 24, 24, 64, 16, 1, 1, 1, 0, 1, 1 } };
        for (int[] q : c2) {
            int oh = convLength(q[1], q[5], q[7], q[8], q[8], q[9], 1);
            int ow = convLength(q[2], q[6], q[7], q[8], q[8], q[9], 1);
            FloatArray x = FloatArray.fromArray(values(q[0] * q[1] * q[2] * q[3], -1, 1, 242 + q[1]));
            FloatArray w = FloatArray.fromArray(values(q[4] * q[5] * q[6] * (q[3] / q[10]), -1, 1, 243 + q[4]));
            same("conv2d " + java.util.Arrays.toString(q), new Object[] { x, w }, floatsOut(q[0] * oh * ow * q[4]), (g, id, c, o) -> g.libraryTask(id,
                    (a, b, r, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13) -> tune(MlxConv.conv2d(a, b, r, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13), c), x, w,
                    (FloatArray) o[0], q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10]));
        }
        // conv_general with flipped kernels, asymmetric padding and kernel dilation.
        int n = 1;
        int h = 20;
        int w = 22;
        int cin = 16;
        int cout = 16;
        int oh = convLength(h, 3, 1, 1, 2, 2, 1);
        int ow = convLength(w, 3, 1, 1, 2, 2, 1);
        FloatArray x = FloatArray.fromArray(values(n * h * w * cin, -1, 1, 244));
        FloatArray wt = FloatArray.fromArray(values(cout * 3 * 3 * cin, -1, 1, 245));
        same("convGeneral2d flip", new Object[] { x, wt }, floatsOut(n * oh * ow * cout), (g, id, c, o) -> g.libraryTask(id,
                (a, b, r, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, i16) -> tune(MlxConv.convGeneral2d(a, b, r, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14,
                        i15, i16), c), x, wt, (FloatArray) o[0], n, h, w, cin, cout, 3, 3, 1, 1, 2, 2, 1, 1, true));
    }

    // ---------------------------------------------------------------- gather and segmented matmul

    @Test
    public void testGatherAndSegmentedMm() throws TornadoExecutionPlanException {
        int ba = 3;
        int bb = 4;
        for (int[] sh : new int[][] { { 64, 128, 96 }, { 50, 70, 33 } }) {
            int m = sh[0];
            int k = sh[1];
            int n = sh[2];
            FloatArray a = FloatArray.fromArray(values(ba * m * k, -1, 1, 250 + m));
            FloatArray b = FloatArray.fromArray(values(bb * k * n, -1, 1, 251 + n));
            IntArray lhs = IntArray.fromElements(0, 2, 1, 2, 0);
            IntArray rhs = IntArray.fromElements(3, 0, 1, 1, 2);
            same("gatherMm " + java.util.Arrays.toString(sh), new Object[] { a, b, lhs, rhs }, floatsOut(5 * m * n), (g, id, c, o) -> g.libraryTask(id,
                    (x1, x2, x3, x4, x5, i5, i6, i7, i8, i9) -> tune(MlxIndex.gatherMm(x1, x2, x3, x4, x5, i5, i6, i7, i8, i9), c), a, b, lhs, rhs, (FloatArray) o[0], ba, bb, m, k,
                    n));
            FloatArray a2 = FloatArray.fromArray(values(m * k, -1, 1, 252 + m));
            FloatArray b2 = FloatArray.fromArray(values(k * n, -1, 1, 253 + n));
            IntArray segments = IntArray.fromElements(0, k / 4, k / 4, k / 2, k / 2, k, 0, k);
            same("segmentedMm " + java.util.Arrays.toString(sh), new Object[] { a2, b2, segments }, floatsOut(4 * m * n), (g, id, c, o) -> g.libraryTask(id,
                    (x1, x2, x3, x4, i4, i5, i6) -> tune(MlxProducts.segmentedMm(x1, x2, x3, x4, i4, i5, i6), c), a2, b2, segments, (FloatArray) o[0], m, k, n));
        }
    }

    @Test
    public void testGatherQmm() throws TornadoExecutionPlanException {
        int batches = 6;
        int experts = 4;
        int gs = 64;
        int bits = 4;
        for (int[] sh : new int[][] { { 1, 2048, 512 }, { 1, 576, 200 }, { 32, 1024, 256 } }) {
            int m = sh[0];
            int k = sh[1];
            int n = sh[2];
            FloatArray x = FloatArray.fromArray(values(batches * m * k, -1, 1, 260 + m + k));
            IntArray seed = ints(261 + n, -1000000000, 1000000000);
            IntArray w = new IntArray(experts * n * k * bits / 32);
            for (int i = 0; i < w.getSize(); i++) {
                w.set(i, seed.get(i % seed.getSize()) * 31 + i);
            }
            FloatArray scales = FloatArray.fromArray(values(experts * n * (k / gs), 0.001f, 0.02f, 262 + n));
            FloatArray biases = FloatArray.fromArray(values(experts * n * (k / gs), -0.1f, 0.1f, 263 + n));
            IntArray lhs = IntArray.fromElements(0, 1, 2, 3, 4, 5);
            IntArray rhs = IntArray.fromElements(3, 0, 1, 1, 2, 3);
            same("gatherQmm " + java.util.Arrays.toString(sh), new Object[] { x, w, scales, biases, lhs, rhs }, floatsOut(batches * m * n), (g, id, c, o) -> g.libraryTask(id,
                    (x0, x1, x2, x3, x4, x5, x6, i7, i8, i9, i10, i11, i12, i13) -> tune(Mlx.gatherQmm(x0, x1, x2, x3, x4, x5, x6, i7, i8, i9, i10, i11, i12, i13), c), x, w, scales,
                    biases, lhs, rhs, (FloatArray) o[0], batches, experts, m, k, n, gs, bits));
        }
    }
}
