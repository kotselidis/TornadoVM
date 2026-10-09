/*
 * This file is part of Tornado: A heterogeneous programming framework:
 * https://github.com/beehive-lab/tornadovm
 *
 * Copyright (c) 2026, APT Group, Department of Computer Science,
 * School of Engineering, The University of Manchester. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 */
package uk.ac.manchester.tornado.unittests.kernelcontext.api;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.enums.TornadoVMBackendType;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.unittests.common.TornadoTestBase;

/**
 * Tests for {@link KernelContext#gridBarrier()}.
 *
 * <p>How to run:
 *
 * <code>
 * tornado-test -V uk.ac.manchester.tornado.unittests.kernelcontext.api.TestGridBarrier
 * </code>
 */
public class TestGridBarrier extends TornadoTestBase {

    private static final int BLOCK = 128;

    /**
     * Each phase reads the element one block ahead, which another block wrote in the previous
     * phase. Without a grid-wide barrier between phases a block reads values that are one or more
     * phases stale.
     */
    public static void shiftRounds(KernelContext context, IntArray a, IntArray b, int rounds) {
        int i = context.globalIdx;
        int n = a.getSize();
        int neighbour = (i + BLOCK) % n;
        for (int r = 0; r < rounds; r++) {
            b.set(i, a.get(neighbour) + 1);
            context.gridBarrier();
            a.set(i, b.get(neighbour) + 1);
            context.gridBarrier();
        }
    }

    /** One partial sum per block, then the first thread of the grid adds the partials. */
    public static void gridReduce(KernelContext context, IntArray input, IntArray partials, IntArray result) {
        int[] local = context.allocateIntLocalArray(BLOCK);
        int lid = context.localIdx;
        local[lid] = input.get(context.globalIdx);
        for (int stride = BLOCK / 2; stride > 0; stride /= 2) {
            context.localBarrier();
            if (lid < stride) {
                local[lid] += local[lid + stride];
            }
        }
        if (lid == 0) {
            partials.set(context.groupIdx, local[0]);
        }
        context.gridBarrier();
        if (context.globalIdx == 0) {
            int sum = 0;
            for (int g = 0; g < partials.getSize(); g++) {
                sum += partials.get(g);
            }
            result.set(0, sum);
        }
    }

    /** One block per multiprocessor: always resident at once, so the cooperative launch accepts it. */
    private static int residentBlocks() {
        return Math.max(2, getTornadoRuntime().getDefaultDevice().getPhysicalDevice().getDeviceMaxComputeUnits());
    }

    private static GridScheduler grid(String taskId, int blocks) {
        WorkerGrid worker = new WorkerGrid1D(blocks * BLOCK);
        worker.setLocalWork(BLOCK, 1, 1);
        return new GridScheduler(taskId, worker);
    }

    @Test
    public void testPhasesSeeOtherBlocksWrites() throws TornadoExecutionPlanException {
        assertNotBackend(TornadoVMBackendType.OPENCL);
        assertNotBackend(TornadoVMBackendType.METAL);

        final int blocks = residentBlocks();
        final int n = blocks * BLOCK;
        final int rounds = 50;
        IntArray a = new IntArray(n);
        IntArray b = new IntArray(n);
        for (int i = 0; i < n; i++) {
            a.set(i, i);
        }

        TaskGraph taskGraph = new TaskGraph("gb") //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b) //
                .task("shift", TestGridBarrier::shiftRounds, new KernelContext(), a, b, rounds) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, a);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
            plan.withGridScheduler(grid("gb.shift", blocks)).execute();
        }

        // After r rounds a[i] = a0[(i + 2 * r * BLOCK) % n] + 2 * r.
        for (int i = 0; i < n; i++) {
            assertEquals((i + 2 * rounds * BLOCK) % n + 2 * rounds, a.get(i));
        }
    }

    /**
     * A grid far larger than the device keeps resident must be refused by the cooperative launch.
     * Launched non-cooperatively it would hang at the first barrier instead; that is what a kernel
     * loaded from the module cache did before the cooperative flag survived the cache, so a second
     * run of this test, which hits the cache, guards that path too.
     */
    @Test
    public void testNonResidentGridIsRefused() {
        assertNotBackend(TornadoVMBackendType.OPENCL);
        assertNotBackend(TornadoVMBackendType.METAL);

        final int blocks = 64 * residentBlocks();
        final int n = blocks * BLOCK;
        IntArray a = new IntArray(n);
        IntArray b = new IntArray(n);

        TaskGraph taskGraph = new TaskGraph("gn") //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b) //
                .task("shift", TestGridBarrier::shiftRounds, new KernelContext(), a, b, 1) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, a);

        boolean refused = false;
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
            plan.withGridScheduler(grid("gn.shift", blocks)).execute();
        } catch (Exception | Error e) {
            refused = true;
        }
        assertTrue("a grid of " + blocks + " blocks cannot be resident at once and must be refused", refused);
    }

    @Test
    public void testGridReductionAcrossLaunches() throws TornadoExecutionPlanException {
        assertNotBackend(TornadoVMBackendType.OPENCL);
        assertNotBackend(TornadoVMBackendType.METAL);

        final int blocks = residentBlocks();
        final int n = blocks * BLOCK;
        IntArray input = new IntArray(n);
        IntArray partials = new IntArray(blocks);
        IntArray result = new IntArray(1);

        TaskGraph taskGraph = new TaskGraph("gr") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, input) //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, partials) //
                .task("reduce", TestGridBarrier::gridReduce, new KernelContext(), input, partials, result) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, result);

        // Several launches of one module: the barrier's state has to be reusable after each.
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
            for (int launch = 1; launch <= 5; launch++) {
                long expected = 0;
                for (int i = 0; i < n; i++) {
                    input.set(i, (i % 7) * launch);
                    expected += (i % 7) * launch;
                }
                plan.withGridScheduler(grid("gr.reduce", blocks)).execute();
                assertEquals(expected, result.get(0));
            }
        }
    }
}
