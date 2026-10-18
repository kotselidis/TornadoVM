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

import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * How {@link CoreML#load(String, CoreMLOptions)} loads a model: the compute units, the function of a
 * multi-function model, the order in which a task passes the model's features and the shape of
 * features whose shape is flexible. Immutable; every method returns a new instance.
 *
 * <pre>
 * CoreMLModel model = CoreML.load("Model.mlpackage", CoreMLOptions.computeUnits(ComputeUnits.CPU_AND_NEURAL_ENGINE) //
 *         .inputs("x", "mask") //
 *         .outputs("logits"));
 * </pre>
 */
public final class CoreMLOptions {

    private static final CoreMLOptions DEFAULTS = new CoreMLOptions(ComputeUnits.ALL, null, List.of(), List.of(), Map.of());

    private final ComputeUnits computeUnits;
    private final String function;
    private final List<String> inputs;
    private final List<String> outputs;
    private final Map<String, long[]> shapes;

    private CoreMLOptions(ComputeUnits computeUnits, String function, List<String> inputs, List<String> outputs, Map<String, long[]> shapes) {
        this.computeUnits = computeUnits;
        this.function = function;
        this.inputs = inputs;
        this.outputs = outputs;
        this.shapes = shapes;
    }

    /** Core ML's defaults: all compute units, the default function, features in name order. */
    public static CoreMLOptions defaults() {
        return DEFAULTS;
    }

    /** The defaults, restricted to the given compute units. */
    public static CoreMLOptions computeUnits(ComputeUnits units) {
        return DEFAULTS.withComputeUnits(units);
    }

    /** These options, restricted to the given compute units. */
    public CoreMLOptions withComputeUnits(ComputeUnits units) {
        return new CoreMLOptions(units, function, inputs, outputs, shapes);
    }

    /** These options, loading the named function of a multi-function model (macOS 15 or newer). */
    public CoreMLOptions function(String name) {
        return new CoreMLOptions(computeUnits, name, inputs, outputs, shapes);
    }

    /**
     * These options, with the model's inputs passed to a task in this order. Without it, the inputs
     * are passed in the order of their names.
     */
    public CoreMLOptions inputs(String... names) {
        return new CoreMLOptions(computeUnits, function, List.of(names), outputs, shapes);
    }

    /**
     * These options, with the model's outputs passed to a task in this order, after the inputs.
     * Without it, the outputs are passed in the order of their names.
     */
    public CoreMLOptions outputs(String... names) {
        return new CoreMLOptions(computeUnits, function, inputs, List.of(names), shapes);
    }

    /**
     * These options, with the named feature given this shape instead of the default shape of its
     * description, for a model whose feature has a flexible shape.
     */
    public CoreMLOptions shape(String feature, long... shape) {
        Map<String, long[]> copy = new LinkedHashMap<>(shapes);
        copy.put(feature, shape.clone());
        return new CoreMLOptions(computeUnits, function, inputs, outputs, Collections.unmodifiableMap(copy));
    }

    public ComputeUnits getComputeUnits() {
        return computeUnits;
    }

    /** The function of a multi-function model, or null for the default one. */
    public String getFunction() {
        return function;
    }

    /** The order of the inputs, or an empty list for name order. */
    public List<String> getInputs() {
        return inputs;
    }

    /** The order of the outputs, or an empty list for name order. */
    public List<String> getOutputs() {
        return outputs;
    }

    /** The shape set for a feature with {@link #shape(String, long...)}, or null. */
    public long[] getShape(String feature) {
        long[] shape = shapes.get(feature);
        return shape == null ? null : shape.clone();
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("CoreMLOptions[computeUnits=").append(computeUnits);
        if (function != null) {
            sb.append(", function=").append(function);
        }
        if (!inputs.isEmpty()) {
            sb.append(", inputs=").append(inputs);
        }
        if (!outputs.isEmpty()) {
            sb.append(", outputs=").append(outputs);
        }
        shapes.forEach((name, shape) -> sb.append(", shape(").append(name).append(")=").append(Arrays.toString(shape)));
        return sb.append(']').toString();
    }
}
