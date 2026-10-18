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
package uk.ac.manchester.tornado.coreml;

import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Where Core ML places each operation of a model under the compute units it was loaded with, from
 * {@code MLComputePlan} (macOS 14.4 or newer): the device it prefers, the devices that could run the
 * operation, and for ML programs Core ML's estimate of the operation's share of the model's cost.
 * Obtained with {@link CoreMLModel#computePlan()}.
 */
public final class CoreMLComputePlan {

    /**
     * An operation: a layer of a neural network ({@code name} is the layer's name) or an operation of
     * an ML program ({@code name} is its first output). {@code estimatedCost} is the operation's share
     * of the cost of the program, or NaN when Core ML gives none (neural networks).
     */
    public record Operation(String name, String type, ComputeDevice preferred, Set<ComputeDevice> supported, double estimatedCost) {
    }

    private final List<Operation> operations;

    public CoreMLComputePlan(List<Operation> operations) {
        this.operations = List.copyOf(operations);
    }

    /** The operations Core ML runs, in program order. */
    public List<Operation> getOperations() {
        return operations;
    }

    /** How many operations Core ML prefers to run on each device. */
    public Map<ComputeDevice, Integer> countByDevice() {
        Map<ComputeDevice, Integer> counts = new EnumMap<>(ComputeDevice.class);
        for (Operation operation : operations) {
            counts.merge(operation.preferred(), 1, Integer::sum);
        }
        return counts;
    }

    /** Whether Core ML prefers {@code device} for every operation. */
    public boolean runsEntirelyOn(ComputeDevice device) {
        return !operations.isEmpty() && operations.stream().allMatch(operation -> operation.preferred() == device);
    }

    @Override
    public String toString() {
        return "CoreMLComputePlan[" + operations.size() + " operations, " + countByDevice() + "]";
    }
}
