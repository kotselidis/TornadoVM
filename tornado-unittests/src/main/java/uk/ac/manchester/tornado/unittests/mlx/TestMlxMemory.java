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

import org.junit.Test;

import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.mlx.Mlx;
import uk.ac.manchester.tornado.mlx.provider.MlxLibraryProvider;

/**
 * Leak checks for MLX library tasks: repeated executions of one plan, and many short-lived plans,
 * must not grow MLX's live memory. Results allocated by MLX are freed after each call; wrappers of
 * TornadoVM buffers live as long as their execution plan.
 *
 * <p>
 * How to run?
 * </p>
 * <code>
 * tornado-test -V uk.ac.manchester.tornado.unittests.mlx.TestMlxMemory
 * </code>
 */
public class TestMlxMemory extends MlxTestBase {

    /** Allowed drift in MLX's live memory, well below one array of the sizes used here. */
    private static final long TOLERANCE_BYTES = 64 * 1024;

    public static void scale(FloatArray in, FloatArray out) {
        for (@Parallel int i = 0; i < in.getSize(); i++) {
            out.set(i, in.get(i) * 0.5f);
        }
    }

    /** JIT -> rmsNorm -> matmulTransposed -> add (JIT and MLX, MLX feeding MLX). */
    private static TaskGraph layerLikeGraph(String name, int m, int k, FloatArray x, FloatArray norm, FloatArray weight, FloatArray xs, FloatArray h, FloatArray y, FloatArray out) {
        return new TaskGraph(name) //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, x, norm, weight) //
                .task("scale", TestMlxMemory::scale, x, xs) //
                .libraryTask("norm", Mlx::rmsNorm, xs, norm, h, m, k, 1e-5f) //
                .libraryTask("proj", Mlx::matmulTransposed, h, weight, y, m, k, k) //
                .libraryTask("residual", Mlx::add, y, xs, out) //
                .transferToHost(DataTransferMode.UNDER_DEMAND, out);
    }

    @Test
    public void testRepeatedExecutionsDoNotGrowMemory() throws TornadoExecutionPlanException {
        final int m = 4, k = 512;
        FloatArray x = FloatArray.fromArray(values(m * k, -1, 1, 1));
        FloatArray norm = FloatArray.fromArray(values(k, 0.5f, 1.5f, 2));
        FloatArray weight = FloatArray.fromArray(values(k * k, -0.05f, 0.05f, 3));
        FloatArray xs = new FloatArray(m * k);
        FloatArray h = new FloatArray(m * k);
        FloatArray y = new FloatArray(m * k);
        FloatArray out = new FloatArray(m * k);
        long fallbacks = MlxLibraryProvider.copyFallbacks();
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(layerLikeGraph("leak", m, k, x, norm, weight, xs, h, y, out).snapshot())) {
            for (int i = 0; i < 50; i++) {
                plan.execute();
            }
            long before = MlxLibraryProvider.activeMemoryBytes();
            for (int i = 0; i < 1000; i++) {
                plan.execute();
            }
            long after = MlxLibraryProvider.activeMemoryBytes();
            assertTrue("MLX live memory grew by " + (after - before) + " bytes over 1000 executions", after - before <= TOLERANCE_BYTES);
        }
        assertEquals("MLX copied an input instead of adopting TornadoVM's buffer", fallbacks, MlxLibraryProvider.copyFallbacks());
    }

    @Test
    public void testClosedPlansReleaseMemory() throws TornadoExecutionPlanException {
        final int m = 2, k = 1024;
        // One plan first, so MLX's own one-off allocations are not counted.
        runShortLivedPlan(m, k, -1);
        long before = MlxLibraryProvider.activeMemoryBytes();
        for (int p = 0; p < 100; p++) {
            runShortLivedPlan(m, k, p);
        }
        long after = MlxLibraryProvider.activeMemoryBytes();
        assertTrue("MLX live memory grew by " + (after - before) + " bytes over 100 closed plans", after - before <= TOLERANCE_BYTES);
    }

    private static void runShortLivedPlan(int m, int k, int seed) throws TornadoExecutionPlanException {
        FloatArray x = FloatArray.fromArray(values(m * k, -1, 1, seed));
        FloatArray norm = FloatArray.fromArray(values(k, 0.5f, 1.5f, seed + 1000));
        FloatArray weight = FloatArray.fromArray(values(k * k, -0.05f, 0.05f, seed + 2000));
        FloatArray xs = new FloatArray(m * k);
        FloatArray h = new FloatArray(m * k);
        FloatArray y = new FloatArray(m * k);
        FloatArray out = new FloatArray(m * k);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(layerLikeGraph("plan" + seed, m, k, x, norm, weight, xs, h, y, out).snapshot())) {
            plan.execute();
        }
    }
}
