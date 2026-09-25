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
package uk.ac.manchester.tornado.unittests.reductions;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.annotations.Reduce;
import uk.ac.manchester.tornado.api.common.TornadoFunctions;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.unittests.common.TornadoTestBase;

/**
 * Reductions whose parallel loop does not start at 0. The GPU reduction needs one thread per
 * iteration; launching one per index up to the loop bound left the first {@code start} threads'
 * worth of work-group slots unwritten, so results picked up uninitialised local memory and were
 * wrong intermittently. Sizes cover a power-of-two number of iterations, a non-power-of-two number
 * (device + host split), and sizes below and above one work-group. Each case runs several times
 * because the failure was intermittent.
 *
 * <p>
 * How to test?
 * </p>
 * <code>
 * tornado-test -V uk.ac.manchester.tornado.unittests.reductions.TestReductionsLoopStart
 * </code>
 */
public class TestReductionsLoopStart extends TornadoTestBase {
    // CHECKSTYLE:OFF

    private static final int REPETITIONS = 5;

    public static void sumFrom1(FloatArray input, @Reduce FloatArray result) {
        result.set(0, 0.0f);
        for (@Parallel int i = 1; i < input.getSize(); i++) {
            result.set(0, result.get(0) + input.get(i));
        }
    }

    public static void sumFrom5(FloatArray input, @Reduce FloatArray result) {
        result.set(0, 0.0f);
        for (@Parallel int i = 5; i < input.getSize(); i++) {
            result.set(0, result.get(0) + input.get(i));
        }
    }

    public static void sumIntsFrom3(IntArray input, @Reduce IntArray result) {
        result.set(0, 0);
        for (@Parallel int i = 3; i < input.getSize(); i++) {
            result.set(0, result.get(0) + input.get(i));
        }
    }

    public static void maxFrom2(FloatArray input, @Reduce FloatArray result) {
        result.set(0, -1.0f);
        for (@Parallel int i = 2; i < input.getSize(); i++) {
            result.set(0, TornadoMath.max(result.get(0), input.get(i)));
        }
    }

    /** Values i % 7 + 1: small integers, so float sums are exact and every element matters. */
    private static FloatArray values(int n) {
        FloatArray a = new FloatArray(n);
        for (int i = 0; i < n; i++) {
            a.set(i, i % 7 + 1);
        }
        return a;
    }

    private static float expectedSum(FloatArray a, int start) {
        float sum = 0;
        for (int i = start; i < a.getSize(); i++) {
            sum += a.get(i);
        }
        return sum;
    }

    private static void checkFloatSum(TornadoFunctions.Task2<FloatArray, FloatArray> kernel, int start, int n) throws TornadoExecutionPlanException {
        FloatArray input = values(n);
        float expected = expectedSum(input, start);
        for (int rep = 0; rep < REPETITIONS; rep++) {
            FloatArray result = new FloatArray(1);
            TaskGraph taskGraph = new TaskGraph("s0") //
                    .transferToDevice(DataTransferMode.FIRST_EXECUTION, input) //
                    .task("t0", kernel, input, result) //
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, result);
            try (TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(taskGraph.snapshot())) {
                executionPlan.execute();
            }
            assertEquals("start " + start + ", n " + n + ", run " + rep, expected, result.get(0), 0.0f);
        }
    }

    @Test
    public void testSumFrom1PowerOfTwoBound() throws TornadoExecutionPlanException {
        // 32767 iterations: the testComputePi shape.
        checkFloatSum(TestReductionsLoopStart::sumFrom1, 1, 32768);
    }

    @Test
    public void testSumFrom1PowerOfTwoIterations() throws TornadoExecutionPlanException {
        checkFloatSum(TestReductionsLoopStart::sumFrom1, 1, 32769);
    }

    @Test
    public void testSumFrom1Small() throws TornadoExecutionPlanException {
        checkFloatSum(TestReductionsLoopStart::sumFrom1, 1, 1000);
    }

    @Test
    public void testSumFrom5() throws TornadoExecutionPlanException {
        checkFloatSum(TestReductionsLoopStart::sumFrom5, 5, 32768);
    }

    @Test
    public void testSumFrom5Small() throws TornadoExecutionPlanException {
        checkFloatSum(TestReductionsLoopStart::sumFrom5, 5, 4096);
    }

    @Test
    public void testSumIntsFrom3() throws TornadoExecutionPlanException {
        final int n = 65536;
        IntArray input = new IntArray(n);
        for (int i = 0; i < n; i++) {
            input.set(i, i % 11 - 5);
        }
        int expected = 0;
        for (int i = 3; i < n; i++) {
            expected += input.get(i);
        }
        for (int rep = 0; rep < REPETITIONS; rep++) {
            IntArray result = new IntArray(1);
            TaskGraph taskGraph = new TaskGraph("s0") //
                    .transferToDevice(DataTransferMode.FIRST_EXECUTION, input) //
                    .task("t0", TestReductionsLoopStart::sumIntsFrom3, input, result) //
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, result);
            try (TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(taskGraph.snapshot())) {
                executionPlan.execute();
            }
            assertEquals("run " + rep, expected, result.get(0));
        }
    }

    @Test
    public void testMaxFrom2() throws TornadoExecutionPlanException {
        final int n = 16384;
        FloatArray input = values(n);
        // The largest values sit before the loop start and must be ignored.
        input.set(0, 1000.0f);
        input.set(1, 1000.0f);
        for (int rep = 0; rep < REPETITIONS; rep++) {
            FloatArray result = new FloatArray(1);
            TaskGraph taskGraph = new TaskGraph("s0") //
                    .transferToDevice(DataTransferMode.FIRST_EXECUTION, input) //
                    .task("t0", TestReductionsLoopStart::maxFrom2, input, result) //
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, result);
            try (TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(taskGraph.snapshot())) {
                executionPlan.execute();
            }
            assertEquals("run " + rep, 7.0f, result.get(0), 0.0f);
        }
    }
    // CHECKSTYLE:ON
}
