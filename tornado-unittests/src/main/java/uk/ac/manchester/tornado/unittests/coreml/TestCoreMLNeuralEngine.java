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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

import java.util.List;
import java.util.Random;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.exceptions.TornadoLibraryTaskNotSupported;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.coreml.ComputeDevice;
import uk.ac.manchester.tornado.coreml.ComputeUnits;
import uk.ac.manchester.tornado.coreml.CoreML;
import uk.ac.manchester.tornado.coreml.CoreMLComputePlan;
import uk.ac.manchester.tornado.coreml.CoreMLDataType;
import uk.ac.manchester.tornado.coreml.CoreMLModel;
import uk.ac.manchester.tornado.coreml.CoreMLOptions;
import uk.ac.manchester.tornado.coreml.provider.CoreMLLibraryProvider;

/**
 * Unit tests for a Core ML model that runs on the Neural Engine: a float16 two-layer perceptron that
 * Core ML places entirely on the Neural Engine, whose float16 input and output are TornadoVM
 * {@code HalfFloatArray}s. Reported as unsupported on hosts without a Neural Engine or older than
 * macOS 14.4.
 *
 * <p>
 * How to run?
 * </p>
 * <code>
 * tornado-test -V uk.ac.manchester.tornado.unittests.coreml.TestCoreMLNeuralEngine
 * </code>
 */
public class TestCoreMLNeuralEngine extends CoreMLTestBase {

    @BeforeEach
    public void neuralEngineMustBeAvailable() {
        if (!CoreML.availableComputeDevices().contains(ComputeDevice.NEURAL_ENGINE)) {
            throw new TornadoLibraryTaskNotSupported("this host has no Neural Engine");
        }
    }

    /** {@code y = relu(x @ W1^T) @ W2^T} in double precision, over the values {@code x} holds. */
    private static double[] mlp(HalfFloatArray x) {
        double[] y = new double[MLP_ROWS * MLP_DIM];
        double[] h = new double[MLP_DIM];
        for (int r = 0; r < MLP_ROWS; r++) {
            for (int o = 0; o < MLP_DIM; o++) {
                double sum = 0;
                for (int i = 0; i < MLP_DIM; i++) {
                    sum += mlpWeight1(o, i) * x.get(r * MLP_DIM + i).getFloat32();
                }
                h[o] = Math.max(0, sum);
            }
            for (int o = 0; o < MLP_DIM; o++) {
                double sum = 0;
                for (int i = 0; i < MLP_DIM; i++) {
                    sum += mlpWeight2(o, i) * h[i];
                }
                y[r * MLP_DIM + o] = sum;
            }
        }
        return y;
    }

    private static HalfFloatArray input(long seed) {
        Random random = new Random(seed);
        HalfFloatArray x = new HalfFloatArray(MLP_ROWS * MLP_DIM);
        for (int i = 0; i < x.getSize(); i++) {
            x.set(i, new HalfFloat(2 * random.nextFloat() - 1));
        }
        return x;
    }

    /** Within float16 rounding of a 256-term dot product, twice. */
    private static void assertMlp(String what, double[] expected, HalfFloatArray y) {
        for (int i = 0; i < expected.length; i++) {
            double actual = y.get(i).getFloat32();
            if (Math.abs(expected[i] - actual) > 3e-2 + 3e-2 * Math.abs(expected[i])) {
                fail(what + " element " + i + ": expected " + expected[i] + " but was " + actual);
            }
        }
    }

    @Test
    public void testDescription() {
        try (CoreMLModel model = CoreML.load(mlpModel())) {
            assertEquals(List.of("x"), model.getInputs().stream().map(CoreMLModel.Feature::name).toList());
            assertEquals(List.of("y"), model.getOutputs().stream().map(CoreMLModel.Feature::name).toList());
            for (int i = 0; i < model.getNumFeatures(); i++) {
                assertEquals(CoreMLDataType.FLOAT16, model.getFeature(i).dataType());
                assertEquals((long) MLP_ROWS * MLP_DIM, model.getFeature(i).elements());
            }
        }
    }

    @Test
    public void testPlacedOnNeuralEngine() {
        for (ComputeUnits units : new ComputeUnits[] { ComputeUnits.CPU_AND_NEURAL_ENGINE, ComputeUnits.ALL }) {
            try (CoreMLModel model = CoreML.load(mlpModel(), CoreMLOptions.computeUnits(units))) {
                CoreMLComputePlan plan = model.computePlan();
                if (Boolean.getBoolean("tornado.unittests.verbose")) {
                    System.out.println("\t" + units + ": " + plan);
                }
                assertEquals(List.of("fc1", "relu", "y"), plan.getOperations().stream().map(CoreMLComputePlan.Operation::name).toList());
                assertTrue(plan.runsEntirelyOn(ComputeDevice.NEURAL_ENGINE), units + ": " + plan.getOperations());
            }
        }
    }

    @Test
    public void testPredictOnEveryComputeUnits() throws TornadoExecutionPlanException {
        HalfFloatArray x = input(11);
        double[] expected = mlp(x);
        for (ComputeUnits units : ComputeUnits.values()) {
            HalfFloatArray y = new HalfFloatArray(MLP_ROWS * MLP_DIM);
            try (CoreMLModel model = CoreML.load(mlpModel(), CoreMLOptions.computeUnits(units))) {
                TaskGraph taskGraph = new TaskGraph("g") //
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, x) //
                        .libraryTask("mlp", CoreML::predict, model, x, y) //
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, y);
                long inPlace = CoreMLLibraryProvider.inPlaceOutputs();
                long copied = CoreMLLibraryProvider.copiedOutputs();
                try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
                    plan.execute();
                }
                long newInPlace = CoreMLLibraryProvider.inPlaceOutputs() - inPlace;
                long newCopied = CoreMLLibraryProvider.copiedOutputs() - copied;
                if (Boolean.getBoolean("tornado.unittests.verbose")) {
                    System.out.println("\t" + units + ": " + newInPlace + " output in place, " + newCopied + " copied");
                }
                assertEquals(1, newInPlace + newCopied, units + ": outputs accounted for");
                if (units == ComputeUnits.CPU_AND_NEURAL_ENGINE) {
                    // The Neural Engine writes its float16 output straight into the TornadoVM buffer.
                    assertEquals(1, newInPlace, units + ": output written in place");
                }
            }
            assertMlp(units.name(), expected, y);
        }
    }

    @Test
    public void testNewInputsEveryExecution() throws TornadoExecutionPlanException {
        HalfFloatArray x = new HalfFloatArray(MLP_ROWS * MLP_DIM);
        HalfFloatArray y = new HalfFloatArray(MLP_ROWS * MLP_DIM);
        try (CoreMLModel model = CoreML.load(mlpModel(), CoreMLOptions.computeUnits(ComputeUnits.CPU_AND_NEURAL_ENGINE))) {
            TaskGraph taskGraph = new TaskGraph("g") //
                    .transferToDevice(DataTransferMode.EVERY_EXECUTION, x) //
                    .libraryTask("mlp", CoreML::predict, model, x, y) //
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, y);
            try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
                for (int run = 0; run < 3; run++) {
                    HalfFloatArray next = input(20 + run);
                    for (int i = 0; i < x.getSize(); i++) {
                        x.set(i, next.get(i));
                    }
                    plan.execute();
                    assertMlp("run " + run, mlp(x), y);
                }
            }
        }
    }
}
