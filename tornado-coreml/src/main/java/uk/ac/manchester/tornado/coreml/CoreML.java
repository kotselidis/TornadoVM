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

import java.util.Set;

import uk.ac.manchester.tornado.api.common.Access;
import uk.ac.manchester.tornado.api.common.LibraryTaskDescriptor;
import uk.ac.manchester.tornado.api.types.arrays.TornadoNativeArray;
import uk.ac.manchester.tornado.coreml.provider.CoreMLLibraryProvider;
import uk.ac.manchester.tornado.coreml.provider.CoreMLRuntime;

/**
 * Apple Core ML models as library tasks on the Metal backend. A model is loaded once with
 * {@link #load(String, CoreMLOptions)}; each {@code CoreML::predict} task runs one prediction of it,
 * with the model's inputs and then its outputs as TornadoVM arrays:
 *
 * <pre>
 * CoreMLModel model = CoreML.load("Classifier.mlpackage", CoreMLOptions.computeUnits(ComputeUnits.ALL));
 * taskGraph.task("preprocess", MyKernels::normalise, context, image, pixels) //
 *         .libraryTask("classify", CoreML::predict, model, pixels, logits) //
 *         .task("softmax", MyKernels::softmax, context, logits, probabilities);
 * </pre>
 *
 * <p>
 * Core ML reads the inputs and writes the outputs in place, in TornadoVM's buffers. It decides by
 * itself which parts of the model run on the CPU, the GPU or the Neural Engine, within the compute
 * units of the model's options. Inputs are passed in the order of their names and outputs after them,
 * also by name, unless {@link CoreMLOptions#inputs(String...)} and
 * {@link CoreMLOptions#outputs(String...)} set the order; {@link CoreMLModel#getInputs()} and
 * {@link CoreMLModel#getOutputs()} list it.
 * </p>
 */
public final class CoreML {

    public static final String LIBRARY_NAME = "apple/coreml";

    /** The function name of every Core ML task. */
    public static final String PREDICT = "coreml_predict";

    private CoreML() {
    }

    /** Whether Core ML can run library tasks on this host. */
    public static boolean isAvailable() {
        return CoreMLLibraryProvider.isAvailable();
    }

    /** The devices Core ML can use on this host (macOS 14 or newer). */
    public static Set<ComputeDevice> availableComputeDevices() {
        return CoreMLRuntime.availableComputeDevices();
    }

    /** Loads a model ({@code .mlpackage}, {@code .mlmodel} or compiled {@code .mlmodelc}) with Core ML's defaults. */
    public static CoreMLModel load(String path) {
        return load(path, CoreMLOptions.defaults());
    }

    /**
     * Loads a model: an {@code .mlpackage} or {@code .mlmodel} is compiled first, a compiled
     * {@code .mlmodelc} is loaded as it is. Every input and output must be a multi-array.
     */
    public static CoreMLModel load(String path, CoreMLOptions options) {
        return CoreMLRuntime.load(path, options);
    }

    /** One prediction of {@code model}; its inputs come first in {@code arrays}, then its outputs. */
    private static LibraryTaskDescriptor task(CoreMLModel model, TornadoNativeArray... arrays) {
        if (arrays.length != model.getNumFeatures()) {
            throw new IllegalArgumentException("Core ML model " + model.getPath() + " takes " + model.getInputs() + " then " + model.getOutputs() + ", not " + arrays.length + " arrays");
        }
        Access[] accesses = new Access[arrays.length];
        for (int i = 0; i < arrays.length; i++) {
            accesses[i] = i < model.getInputs().size() ? Access.READ_ONLY : Access.WRITE_ONLY;
        }
        return new LibraryTaskDescriptor() //
                .withLibrary(LIBRARY_NAME) //
                .withFunction(PREDICT) //
                .withParameters(arrays) //
                .withAccess(accesses) //
                .withTuning(model);
    }

    /** A prediction of a model with one input and one output, or no inputs and one output. */
    public static LibraryTaskDescriptor predict(CoreMLModel model, TornadoNativeArray a0) {
        return task(model, a0);
    }

    /** A prediction of a model with two features: its inputs, then its outputs. */
    public static LibraryTaskDescriptor predict(CoreMLModel model, TornadoNativeArray a0, TornadoNativeArray a1) {
        return task(model, a0, a1);
    }

    /** A prediction of a model with three features: its inputs, then its outputs. */
    public static LibraryTaskDescriptor predict(CoreMLModel model, TornadoNativeArray a0, TornadoNativeArray a1, TornadoNativeArray a2) {
        return task(model, a0, a1, a2);
    }

    /** A prediction of a model with four features: its inputs, then its outputs. */
    public static LibraryTaskDescriptor predict(CoreMLModel model, TornadoNativeArray a0, TornadoNativeArray a1, TornadoNativeArray a2, TornadoNativeArray a3) {
        return task(model, a0, a1, a2, a3);
    }

    /** A prediction of a model with five features: its inputs, then its outputs. */
    public static LibraryTaskDescriptor predict(CoreMLModel model, TornadoNativeArray a0, TornadoNativeArray a1, TornadoNativeArray a2, TornadoNativeArray a3, TornadoNativeArray a4) {
        return task(model, a0, a1, a2, a3, a4);
    }

    /** A prediction of a model with six features: its inputs, then its outputs. */
    public static LibraryTaskDescriptor predict(CoreMLModel model, TornadoNativeArray a0, TornadoNativeArray a1, TornadoNativeArray a2, TornadoNativeArray a3, TornadoNativeArray a4,
            TornadoNativeArray a5) {
        return task(model, a0, a1, a2, a3, a4, a5);
    }

    /** A prediction of a model with seven features: its inputs, then its outputs. */
    public static LibraryTaskDescriptor predict(CoreMLModel model, TornadoNativeArray a0, TornadoNativeArray a1, TornadoNativeArray a2, TornadoNativeArray a3, TornadoNativeArray a4,
            TornadoNativeArray a5, TornadoNativeArray a6) {
        return task(model, a0, a1, a2, a3, a4, a5, a6);
    }

    /** A prediction of a model with eight features: its inputs, then its outputs. */
    public static LibraryTaskDescriptor predict(CoreMLModel model, TornadoNativeArray a0, TornadoNativeArray a1, TornadoNativeArray a2, TornadoNativeArray a3, TornadoNativeArray a4,
            TornadoNativeArray a5, TornadoNativeArray a6, TornadoNativeArray a7) {
        return task(model, a0, a1, a2, a3, a4, a5, a6, a7);
    }
}
