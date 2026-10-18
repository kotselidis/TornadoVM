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
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.EnumSet;
import java.util.List;
import java.util.Set;

import org.junit.jupiter.api.Test;

import uk.ac.manchester.tornado.coreml.ComputeDevice;
import uk.ac.manchester.tornado.coreml.ComputeUnits;
import uk.ac.manchester.tornado.coreml.CoreML;
import uk.ac.manchester.tornado.coreml.CoreMLComputePlan;
import uk.ac.manchester.tornado.coreml.CoreMLModel;
import uk.ac.manchester.tornado.coreml.CoreMLOptions;

/**
 * Unit tests for Core ML compute plans: the devices Core ML can use, and where it places each
 * operation of a model under each compute-unit setting. Needs macOS 14.4 or newer; reported as
 * unsupported otherwise.
 *
 * <p>
 * How to run?
 * </p>
 * <code>
 * tornado-test -V uk.ac.manchester.tornado.unittests.coreml.TestCoreMLComputePlan
 * </code>
 */
public class TestCoreMLComputePlan extends CoreMLTestBase {

    /** The devices each compute-unit setting allows. */
    private static Set<ComputeDevice> allowed(ComputeUnits units) {
        return switch (units) {
            case CPU_ONLY -> EnumSet.of(ComputeDevice.CPU);
            case CPU_AND_GPU -> EnumSet.of(ComputeDevice.CPU, ComputeDevice.GPU);
            case CPU_AND_NEURAL_ENGINE -> EnumSet.of(ComputeDevice.CPU, ComputeDevice.NEURAL_ENGINE);
            case ALL -> EnumSet.allOf(ComputeDevice.class);
        };
    }

    @Test
    public void testAvailableComputeDevices() {
        assertTrue(CoreML.availableComputeDevices().contains(ComputeDevice.CPU), "devices: " + CoreML.availableComputeDevices());
    }

    @Test
    public void testCpuOnlyPlan() {
        try (CoreMLModel model = CoreML.load(addMulModel(), CoreMLOptions.computeUnits(ComputeUnits.CPU_ONLY))) {
            CoreMLComputePlan plan = model.computePlan();
            assertEquals(List.of("mul", "add"), plan.getOperations().stream().map(CoreMLComputePlan.Operation::name).toList());
            assertTrue(plan.runsEntirelyOn(ComputeDevice.CPU), "plan: " + plan.getOperations());
        }
    }

    @Test
    public void testPlanRespectsComputeUnits() {
        for (ComputeUnits units : ComputeUnits.values()) {
            try (CoreMLModel model = CoreML.load(addMulModel(), CoreMLOptions.computeUnits(units))) {
                CoreMLComputePlan plan = model.computePlan();
                if (Boolean.getBoolean("tornado.unittests.verbose")) {
                    System.out.println("\t" + units + ": " + plan.getOperations());
                }
                assertEquals(2, plan.getOperations().size(), units + ": operations");
                for (CoreMLComputePlan.Operation operation : plan.getOperations()) {
                    assertTrue(allowed(units).contains(operation.preferred()), units + ": " + operation);
                    assertTrue(operation.supported().contains(operation.preferred()), units + ": " + operation);
                }
            }
        }
    }

    @Test
    public void testClosedModel() {
        CoreMLModel model = CoreML.load(addMulModel());
        model.close();
        assertThrows(IllegalStateException.class, model::computePlan);
    }
}
