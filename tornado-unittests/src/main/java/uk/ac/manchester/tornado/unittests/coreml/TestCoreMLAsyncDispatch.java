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
package uk.ac.manchester.tornado.unittests.coreml;

import org.junit.jupiter.api.Test;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.coreml.CoreML;
import uk.ac.manchester.tornado.coreml.CoreMLModel;

/**
 * Core ML tasks with the Metal backend's asynchronous library dispatch
 * ({@code -Dtornado.mlx.async=true}), where the command queue is only flushed before a library task:
 * the Core ML task must still read what the kernel before it wrote.
 *
 * <p>
 * How to run?
 * </p>
 * <code>
 * tornado-test -V -J"-Dtornado.mlx.async=true" uk.ac.manchester.tornado.unittests.coreml.TestCoreMLAsyncDispatch
 * </code>
 */
public class TestCoreMLAsyncDispatch extends CoreMLTestBase {

    @Test
    public void testKernelThenCoreML() throws TornadoExecutionPlanException {
        FloatArray x = new FloatArray(N);
        FloatArray y = new FloatArray(N);
        FloatArray prod = new FloatArray(N);
        FloatArray sum = new FloatArray(N);
        FloatArray out = new FloatArray(N);
        KernelContext context = new KernelContext();

        WorkerGrid1D rampsGrid = new WorkerGrid1D(N);
        rampsGrid.setLocalWork(N, 1, 1);
        WorkerGrid1D combineGrid = new WorkerGrid1D(N);
        combineGrid.setLocalWork(N, 1, 1);
        GridScheduler gridScheduler = new GridScheduler();
        gridScheduler.addWorkerGrid("g.ramps", rampsGrid);
        gridScheduler.addWorkerGrid("g.combine", combineGrid);

        try (CoreMLModel model = CoreML.load(addMulModel())) {
            TaskGraph taskGraph = new TaskGraph("g") //
                    .transferToDevice(DataTransferMode.FIRST_EXECUTION, x, y, prod, sum, out) //
                    .task("ramps", CoreMLTestBase::ramps, context, x, y) //
                    .libraryTask("addmul", CoreML::predict, model, x, y, prod, sum) //
                    .task("combine", CoreMLTestBase::combine, context, prod, sum, out) //
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, out);

            try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
                for (int run = 0; run < 4; run++) {
                    plan.withGridScheduler(gridScheduler).execute();
                }
            }
        }
        for (int i = 0; i < N; i++) {
            double xi = i / 8.0 - 1.0;
            double yi = 1.0 - i / 16.0;
            assertClose("out", i, xi * yi + xi + yi, out.get(i));
        }
    }
}
