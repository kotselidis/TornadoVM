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
import java.util.List;

import uk.ac.manchester.tornado.coreml.provider.CoreMLRuntime;

/**
 * A Core ML model loaded by {@link CoreML#load(String, CoreMLOptions)}, with the multi-array inputs
 * and outputs a task passes to it, in task order. A model can be used by any number of tasks and
 * execution plans; {@link #close()} it after the last plan that uses it is closed.
 */
public final class CoreMLModel implements AutoCloseable {

    /** A multi-array input or output of a model. */
    public record Feature(String name, long[] shape, CoreMLDataType dataType) {

        /** The number of elements of the feature. */
        public long elements() {
            long n = 1;
            for (long d : shape) {
                n *= d;
            }
            return n;
        }

        @Override
        public String toString() {
            return name + ":" + dataType.name().toLowerCase() + Arrays.toString(shape);
        }
    }

    private final String path;
    private final String compiledPath;
    private final CoreMLOptions options;
    private final List<Feature> inputs;
    private final List<Feature> outputs;
    private volatile long handle;

    /** A loaded model, compiled at {@code compiledPath}; {@code handle} is the retained {@code MLModel}. */
    public CoreMLModel(String path, String compiledPath, CoreMLOptions options, List<Feature> inputs, List<Feature> outputs, long handle) {
        this.path = path;
        this.compiledPath = compiledPath;
        this.options = options;
        this.inputs = List.copyOf(inputs);
        this.outputs = List.copyOf(outputs);
        this.handle = handle;
    }

    /** The path the model was loaded from. */
    public String getPath() {
        return path;
    }

    /** The compiled model ({@code .mlmodelc}) Core ML loaded. */
    public String getCompiledPath() {
        return compiledPath;
    }

    public CoreMLOptions getOptions() {
        return options;
    }

    /** The inputs, in the order a task passes them. */
    public List<Feature> getInputs() {
        return inputs;
    }

    /** The outputs, in the order a task passes them, after the inputs. */
    public List<Feature> getOutputs() {
        return outputs;
    }

    /** The number of arrays a task passes: the inputs, then the outputs. */
    public int getNumFeatures() {
        return inputs.size() + outputs.size();
    }

    /** The feature at a task argument position. */
    public Feature getFeature(int position) {
        return position < inputs.size() ? inputs.get(position) : outputs.get(position - inputs.size());
    }

    /**
     * Where Core ML places each operation of this model under its compute units (macOS 14.4 or
     * newer). Each call asks Core ML again.
     */
    public CoreMLComputePlan computePlan() {
        nativeHandle();
        return CoreMLRuntime.computePlan(this);
    }

    /** The {@code MLModel}; fails once the model is closed. */
    public long nativeHandle() {
        long h = handle;
        if (h == 0) {
            throw new IllegalStateException("Core ML model " + path + " is closed");
        }
        return h;
    }

    public boolean isClosed() {
        return handle == 0;
    }

    /** Releases the {@code MLModel}. */
    @Override
    public synchronized void close() {
        if (handle != 0) {
            CoreMLRuntime.release(handle);
            handle = 0;
        }
    }

    @Override
    public String toString() {
        return "CoreMLModel[" + path + ", inputs=" + inputs + ", outputs=" + outputs + ", " + options.getComputeUnits() + "]";
    }
}
