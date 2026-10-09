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

import org.junit.Test;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.unittests.common.TornadoTestBase;

/**
 * Tests for {@link KernelContext#prefetchToL2}: a hint, so a kernel that prefetches must compute
 * exactly what it computes without the prefetch, on every backend.
 *
 * <p>How to run:
 *
 * <code>
 * tornado-test -V uk.ac.manchester.tornado.unittests.kernelcontext.api.TestPrefetchL2
 * </code>
 */
public class TestPrefetchL2 extends TornadoTestBase {

    private static final int BLOCK = 128;
    private static final int N = 64 * BLOCK;

    /** Prefetches the lines the next block will read, then sums the three inputs. */
    public static void sumWithPrefetch(KernelContext context, HalfFloatArray h, FloatArray f, ByteArray b, FloatArray out) {
        int i = context.globalIdx;
        int ahead = i + BLOCK;
        if (ahead < out.getSize()) {
            context.prefetchToL2(h, ahead);
            context.prefetchToL2(f, ahead);
            context.prefetchToL2(b, ahead);
        }
        out.set(i, h.get(i).getFloat32() + f.get(i) + b.get(i));
    }

    @Test
    public void testPrefetchDoesNotChangeResults() throws TornadoExecutionPlanException {
        HalfFloatArray h = new HalfFloatArray(N);
        FloatArray f = new FloatArray(N);
        ByteArray b = new ByteArray(N);
        FloatArray out = new FloatArray(N);
        for (int i = 0; i < N; i++) {
            h.set(i, new HalfFloat(i % 64));
            f.set(i, 0.5f * i);
            b.set(i, (byte) (i % 100));
        }

        TaskGraph taskGraph = new TaskGraph("pf") //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, h, f, b) //
                .task("sum", TestPrefetchL2::sumWithPrefetch, new KernelContext(), h, f, b, out) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, out);

        WorkerGrid worker = new WorkerGrid1D(N);
        worker.setLocalWork(BLOCK, 1, 1);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
            plan.withGridScheduler(new GridScheduler("pf.sum", worker)).execute();
        }

        for (int i = 0; i < N; i++) {
            assertEquals((i % 64) + 0.5f * i + (i % 100), out.get(i), 0.0f);
        }
    }
}
