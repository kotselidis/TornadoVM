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

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.coreml.ComputeUnits;
import uk.ac.manchester.tornado.coreml.CoreML;
import uk.ac.manchester.tornado.coreml.CoreMLDataType;
import uk.ac.manchester.tornado.coreml.CoreMLModel;
import uk.ac.manchester.tornado.coreml.CoreMLOptions;
import uk.ac.manchester.tornado.coreml.provider.CoreMLLibraryProvider;

/**
 * Unit tests for Core ML library tasks: loading a model and its description, predictions that read
 * and write TornadoVM buffers in place, every compute-unit setting, Core ML between JIT-compiled
 * kernels in one graph, new inputs on every execution, feature order and argument checks. Skipped
 * unless the default device is on the Metal backend and Core ML is available.
 *
 * <p>
 * How to run?
 * </p>
 * <code>
 * tornado-test -V uk.ac.manchester.tornado.unittests.coreml.TestCoreML
 * </code>
 */
public class TestCoreML extends CoreMLTestBase {

    @Test
    public void testLoadDescribesFeatures() {
        try (CoreMLModel model = CoreML.load(addMulModel())) {
            assertEquals(List.of("x", "y"), model.getInputs().stream().map(CoreMLModel.Feature::name).toList());
            assertEquals(List.of("prod", "sum"), model.getOutputs().stream().map(CoreMLModel.Feature::name).toList());
            for (int i = 0; i < model.getNumFeatures(); i++) {
                assertArrayEquals(new long[] { N }, model.getFeature(i).shape());
                assertEquals(CoreMLDataType.FLOAT32, model.getFeature(i).dataType());
            }
        }
    }

    @Test
    public void testPredict() throws TornadoExecutionPlanException {
        FloatArray x = values(N, -1, 1, 1);
        FloatArray y = values(N, -1, 1, 2);
        FloatArray prod = new FloatArray(N);
        FloatArray sum = new FloatArray(N);

        try (CoreMLModel model = CoreML.load(addMulModel())) {
            TaskGraph taskGraph = new TaskGraph("g") //
                    .transferToDevice(DataTransferMode.EVERY_EXECUTION, x, y) //
                    .libraryTask("addmul", CoreML::predict, model, x, y, prod, sum) //
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, prod, sum);

            long before = CoreMLLibraryProvider.predictions();
            try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
                plan.execute();
            }
            assertEquals(before + 1, CoreMLLibraryProvider.predictions(), "predictions");
        }
        assertAddMul(x, y, prod, sum);
    }

    @Test
    public void testEveryComputeUnits() throws TornadoExecutionPlanException {
        FloatArray x = values(N, -1, 1, 3);
        FloatArray y = values(N, -1, 1, 4);
        for (ComputeUnits units : ComputeUnits.values()) {
            FloatArray prod = new FloatArray(N);
            FloatArray sum = new FloatArray(N);
            try (CoreMLModel model = CoreML.load(addMulModel(), CoreMLOptions.computeUnits(units))) {
                TaskGraph taskGraph = new TaskGraph("g") //
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, x, y) //
                        .libraryTask("addmul", CoreML::predict, model, x, y, prod, sum) //
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, prod, sum);
                try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
                    plan.execute();
                }
            }
            assertAddMul(x, y, prod, sum);
        }
    }

    @Test
    public void testOutputsWrittenInPlace() throws TornadoExecutionPlanException {
        // On the CPU, Core ML writes both outputs straight into the TornadoVM buffers through the
        // output backings: nothing is copied after the prediction.
        FloatArray x = values(N, -1, 1, 7);
        FloatArray y = values(N, -1, 1, 8);
        FloatArray prod = new FloatArray(N);
        FloatArray sum = new FloatArray(N);

        try (CoreMLModel model = CoreML.load(addMulModel(), CoreMLOptions.computeUnits(ComputeUnits.CPU_ONLY))) {
            TaskGraph taskGraph = new TaskGraph("g") //
                    .transferToDevice(DataTransferMode.EVERY_EXECUTION, x, y) //
                    .libraryTask("addmul", CoreML::predict, model, x, y, prod, sum) //
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, prod, sum);

            long inPlace = CoreMLLibraryProvider.inPlaceOutputs();
            long copied = CoreMLLibraryProvider.copiedOutputs();
            try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
                plan.execute();
                plan.execute();
            }
            assertEquals(inPlace + 4, CoreMLLibraryProvider.inPlaceOutputs(), "outputs written in place");
            assertEquals(copied, CoreMLLibraryProvider.copiedOutputs(), "outputs copied");
        }
        assertAddMul(x, y, prod, sum);
    }

    @Test
    public void testEveryOutputAccountedFor() throws TornadoExecutionPlanException {
        // On every compute unit each output is either written in place or copied, once per prediction.
        FloatArray x = values(N, -1, 1, 9);
        FloatArray y = values(N, -1, 1, 10);
        for (ComputeUnits units : ComputeUnits.values()) {
            FloatArray prod = new FloatArray(N);
            FloatArray sum = new FloatArray(N);
            try (CoreMLModel model = CoreML.load(addMulModel(), CoreMLOptions.computeUnits(units))) {
                TaskGraph taskGraph = new TaskGraph("g") //
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, x, y) //
                        .libraryTask("addmul", CoreML::predict, model, x, y, prod, sum) //
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, prod, sum);
                long inPlace = CoreMLLibraryProvider.inPlaceOutputs();
                long copied = CoreMLLibraryProvider.copiedOutputs();
                try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
                    plan.execute();
                }
                long newInPlace = CoreMLLibraryProvider.inPlaceOutputs() - inPlace;
                long newCopied = CoreMLLibraryProvider.copiedOutputs() - copied;
                if (Boolean.getBoolean("tornado.unittests.verbose")) {
                    System.out.println("\t" + units + ": " + newInPlace + " outputs in place, " + newCopied + " copied");
                }
                assertEquals(2, newInPlace + newCopied, units + ": outputs accounted for");
            }
            assertAddMul(x, y, prod, sum);
        }
    }

    @Test
    public void testBetweenKernels() throws TornadoExecutionPlanException {
        // kernel -> Core ML -> kernel with no host transfers in between: Core ML reads what the first
        // kernel wrote, the second kernel reads what Core ML wrote.
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
                plan.withGridScheduler(gridScheduler).execute();
            }
        }
        for (int i = 0; i < N; i++) {
            double xi = i / 8.0 - 1.0;
            double yi = 1.0 - i / 16.0;
            assertClose("out", i, xi * yi + xi + yi, out.get(i));
        }
    }

    @Test
    public void testNewInputsEveryExecution() throws TornadoExecutionPlanException {
        FloatArray x = new FloatArray(N);
        FloatArray y = new FloatArray(N);
        FloatArray prod = new FloatArray(N);
        FloatArray sum = new FloatArray(N);

        try (CoreMLModel model = CoreML.load(addMulModel())) {
            TaskGraph taskGraph = new TaskGraph("g") //
                    .transferToDevice(DataTransferMode.EVERY_EXECUTION, x, y) //
                    .libraryTask("addmul", CoreML::predict, model, x, y, prod, sum) //
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, prod, sum);

            try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
                for (int run = 0; run < 3; run++) {
                    for (int i = 0; i < N; i++) {
                        x.set(i, run + i * 0.25f);
                        y.set(i, 2.0f - run * 0.5f);
                    }
                    plan.execute();
                    assertAddMul(x, y, prod, sum);
                }
            }
        }
    }

    @Test
    public void testFeatureOrder() throws TornadoExecutionPlanException {
        FloatArray x = values(N, -1, 1, 5);
        FloatArray y = values(N, -1, 1, 6);
        FloatArray prod = new FloatArray(N);
        FloatArray sum = new FloatArray(N);

        try (CoreMLModel model = CoreML.load(addMulModel(), CoreMLOptions.defaults().inputs("y", "x").outputs("sum", "prod"))) {
            assertEquals("y", model.getFeature(0).name());
            assertEquals("prod", model.getFeature(3).name());
            TaskGraph taskGraph = new TaskGraph("g") //
                    .transferToDevice(DataTransferMode.EVERY_EXECUTION, x, y) //
                    .libraryTask("addmul", CoreML::predict, model, y, x, sum, prod) //
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, prod, sum);
            try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
                plan.execute();
            }
        }
        assertAddMul(x, y, prod, sum);
    }

    @Test
    public void testUnknownFeatureOrder() {
        RuntimeException e = assertThrows(RuntimeException.class, () -> CoreML.load(addMulModel(), CoreMLOptions.defaults().inputs("x", "z")));
        assertTrue(e.getMessage().contains("[x, y]"), "message: " + e.getMessage());
    }

    @Test
    public void testWrongNumberOfArrays() {
        FloatArray x = new FloatArray(N);
        FloatArray y = new FloatArray(N);
        FloatArray prod = new FloatArray(N);
        try (CoreMLModel model = CoreML.load(addMulModel())) {
            assertThrows(IllegalArgumentException.class, () -> new TaskGraph("g").libraryTask("addmul", CoreML::predict, model, x, y, prod));
        }
    }

    @Test
    public void testWrongArraySize() {
        FloatArray x = new FloatArray(N);
        FloatArray y = new FloatArray(N / 2);
        FloatArray prod = new FloatArray(N);
        FloatArray sum = new FloatArray(N);
        try (CoreMLModel model = CoreML.load(addMulModel())) {
            TaskGraph taskGraph = new TaskGraph("g") //
                    .transferToDevice(DataTransferMode.EVERY_EXECUTION, x, y) //
                    .libraryTask("addmul", CoreML::predict, model, x, y, prod, sum) //
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, prod, sum);
            RuntimeException e = assertThrows(RuntimeException.class, () -> {
                try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
                    plan.execute();
                }
            });
            Throwable cause = e;
            while (cause.getCause() != null && cause.getCause() != cause) {
                cause = cause.getCause();
            }
            assertTrue(cause.getMessage().contains("has 16 elements; its array has 8"), "message: " + cause.getMessage());
        }
    }
}
