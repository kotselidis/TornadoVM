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
package uk.ac.manchester.tornado.unittests.kernelcontext.local.memory;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.enums.TornadoVMBackendType;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.unittests.common.TornadoTestBase;

/**
 * Local arrays larger than the 48 KB of static shared memory a CUDA block can declare. The CUDA
 * backend places them in dynamic shared memory and opts the kernel in to the larger limit.
 *
 * <p>
 * How to run?
 * </p>
 *
 * <p>
 * <code>
 * tornado-test -V uk.ac.manchester.tornado.unittests.kernelcontext.local.memory.TestDynamicSharedMemory
 * </code>
 * </p>
 */
public class TestDynamicSharedMemory extends TornadoTestBase {

    private static final int LOCAL_SIZE = 256;
    private static final int GROUPS = 8;
    private static final int SIZE = LOCAL_SIZE * GROUPS;

    /** 16384 ints: 64 KB of shared memory. */
    private static final int INT_ELEMENTS = 16384;

    /** 8192 floats and 8192 ints: 32 KB each, 64 KB together. */
    private static final int HALF_ELEMENTS = 8192;

    public static void fillAndReadInt64KB(KernelContext context, IntArray input, IntArray output) {
        int globalIdx = context.globalIdx;
        int localIdx = context.localIdx;
        int groupIdx = context.groupIdx;

        int[] tile = context.allocateIntLocalArray(16384);
        for (int i = localIdx; i < 16384; i += context.localGroupSizeX) {
            tile[i] = i * 3 + groupIdx;
        }
        context.localBarrier();

        if (globalIdx < input.getSize()) {
            // Reads elements written by other threads, from both ends of the 64 KB array.
            output.set(globalIdx, tile[16383 - localIdx * 64] + tile[localIdx * 64] + input.get(globalIdx));
        }
    }

    public static void fillAndReadTwoArrays(KernelContext context, IntArray input, FloatArray output) {
        int globalIdx = context.globalIdx;
        int localIdx = context.localIdx;

        float[] floats = context.allocateFloatLocalArray(8192);
        int[] ints = context.allocateIntLocalArray(8192);
        for (int i = localIdx; i < 8192; i += context.localGroupSizeX) {
            floats[i] = i * 0.5f;
            ints[i] = 8191 - i;
        }
        context.localBarrier();

        if (globalIdx < input.getSize()) {
            // An overlap of the two arrays would overwrite one of them with the other's values.
            int k = localIdx * 32;
            output.set(globalIdx, floats[k] + ints[k] + floats[8191 - k] + input.get(globalIdx));
        }
    }

    public static void fillAndRead256KB(KernelContext context, IntArray input, IntArray output) {
        int globalIdx = context.globalIdx;
        int localIdx = context.localIdx;

        int[] tile = context.allocateIntLocalArray(65536);
        for (int i = localIdx; i < 65536; i += context.localGroupSizeX) {
            tile[i] = i;
        }
        context.localBarrier();

        if (globalIdx < input.getSize()) {
            output.set(globalIdx, tile[65535 - localIdx] + input.get(globalIdx));
        }
    }

    private static GridScheduler gridScheduler() {
        WorkerGrid worker = new WorkerGrid1D(SIZE);
        worker.setGlobalWork(SIZE, 1, 1);
        worker.setLocalWork(LOCAL_SIZE, 1, 1);
        return new GridScheduler("s0.t0", worker);
    }

    @Test
    public void testIntLocalArray64KB() throws TornadoExecutionPlanException {
        assertNotBackend(TornadoVMBackendType.OPENCL);
        assertNotBackend(TornadoVMBackendType.METAL);

        IntArray input = new IntArray(SIZE);
        IntArray output = new IntArray(SIZE);
        for (int i = 0; i < SIZE; i++) {
            input.set(i, i);
        }

        KernelContext context = new KernelContext();
        TaskGraph taskGraph = new TaskGraph("s0") //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, input) //
                .task("t0", TestDynamicSharedMemory::fillAndReadInt64KB, context, input, output) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

        ImmutableTaskGraph immutableTaskGraph = taskGraph.snapshot();
        try (TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(immutableTaskGraph)) {
            executionPlan.withGridScheduler(gridScheduler()).execute();
        }

        for (int i = 0; i < SIZE; i++) {
            int group = i / LOCAL_SIZE;
            int local = i % LOCAL_SIZE;
            int far = INT_ELEMENTS - 1 - local * 64;
            int near = local * 64;
            assertEquals("index " + i, (far * 3 + group) + (near * 3 + group) + i, output.get(i));
        }
    }

    @Test
    public void testTwoLocalArrays64KB() throws TornadoExecutionPlanException {
        assertNotBackend(TornadoVMBackendType.OPENCL);
        assertNotBackend(TornadoVMBackendType.METAL);

        IntArray input = new IntArray(SIZE);
        FloatArray output = new FloatArray(SIZE);
        for (int i = 0; i < SIZE; i++) {
            input.set(i, i % 7);
        }

        KernelContext context = new KernelContext();
        TaskGraph taskGraph = new TaskGraph("s0") //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, input) //
                .task("t0", TestDynamicSharedMemory::fillAndReadTwoArrays, context, input, output) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

        ImmutableTaskGraph immutableTaskGraph = taskGraph.snapshot();
        try (TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(immutableTaskGraph)) {
            executionPlan.withGridScheduler(gridScheduler()).execute();
            // A second run reuses the installed kernel, which must keep its shared memory size.
            executionPlan.withGridScheduler(gridScheduler()).execute();
        }

        for (int i = 0; i < SIZE; i++) {
            int k = (i % LOCAL_SIZE) * 32;
            float expected = k * 0.5f + (HALF_ELEMENTS - 1 - k) + (HALF_ELEMENTS - 1 - k) * 0.5f + i % 7;
            assertEquals("index " + i, expected, output.get(i), 0.0f);
        }
    }

    /**
     * 256 KB is above the opt-in limit of every CUDA GPU (227 KB on sm_90 and sm_100): the launch
     * must be refused with an error naming the shared memory, not fail later or return garbage.
     */
    @Test
    public void testAboveDeviceLimitIsRejected() {
        assertNotBackend(TornadoVMBackendType.OPENCL);
        assertNotBackend(TornadoVMBackendType.METAL);

        IntArray input = new IntArray(SIZE);
        IntArray output = new IntArray(SIZE);

        KernelContext context = new KernelContext();
        TaskGraph taskGraph = new TaskGraph("s0") //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, input) //
                .task("t0", TestDynamicSharedMemory::fillAndRead256KB, context, input, output) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

        Throwable failure = null;
        try (TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(taskGraph.snapshot())) {
            executionPlan.withGridScheduler(gridScheduler()).execute();
        } catch (Throwable t) {
            failure = t;
        }
        assertNotNull("a kernel above the shared memory limit must not run", failure);
        boolean namesSharedMemory = false;
        for (Throwable t = failure; t != null; t = t.getCause()) {
            namesSharedMemory |= t.getMessage() != null && t.getMessage().contains("bytes of shared memory per block");
        }
        assertTrue("unexpected error: " + failure, namesSharedMemory);
    }
}
