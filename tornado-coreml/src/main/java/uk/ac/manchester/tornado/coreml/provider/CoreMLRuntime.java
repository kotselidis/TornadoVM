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
package uk.ac.manchester.tornado.coreml.provider;

import static uk.ac.manchester.tornado.coreml.provider.ObjC.count;
import static uk.ac.manchester.tornado.coreml.provider.ObjC.errorMessage;
import static uk.ac.manchester.tornado.coreml.provider.ObjC.longs;
import static uk.ac.manchester.tornado.coreml.provider.ObjC.newObject;
import static uk.ac.manchester.tornado.coreml.provider.ObjC.nsDictionary;
import static uk.ac.manchester.tornado.coreml.provider.ObjC.nsNumbers;
import static uk.ac.manchester.tornado.coreml.provider.ObjC.nsString;
import static uk.ac.manchester.tornado.coreml.provider.ObjC.objcClass;
import static uk.ac.manchester.tornado.coreml.provider.ObjC.objectAt;
import static uk.ac.manchester.tornado.coreml.provider.ObjC.poolPop;
import static uk.ac.manchester.tornado.coreml.provider.ObjC.poolPush;
import static uk.ac.manchester.tornado.coreml.provider.ObjC.send;
import static uk.ac.manchester.tornado.coreml.provider.ObjC.string;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;

import uk.ac.manchester.tornado.api.exceptions.TornadoLibraryTaskNotSupported;
import uk.ac.manchester.tornado.api.exceptions.TornadoRuntimeException;
import uk.ac.manchester.tornado.api.types.arrays.DoubleArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.coreml.CoreMLDataType;
import uk.ac.manchester.tornado.coreml.CoreMLModel;
import uk.ac.manchester.tornado.coreml.CoreMLModel.Feature;
import uk.ac.manchester.tornado.coreml.CoreMLOptions;
import uk.ac.manchester.tornado.runtime.ffm.FFMSupport;

/**
 * Core ML through the Objective-C runtime with {@code java.lang.foreign}: loading a model and its
 * description, and running a prediction whose inputs and outputs are TornadoVM buffers.
 *
 * <p>
 * On the Metal backend every TornadoVM buffer is a shared-storage {@code MTLBuffer}, so its contents
 * have a CPU address. A prediction wraps each input at that address in an {@code MLMultiArray}
 * without copying it, and hands Core ML the outputs the same way as output backings, so Core ML
 * writes its results straight into the TornadoVM buffers. An output Core ML still returns elsewhere
 * (a backing it cannot use, a different element type or padded strides) is copied in afterwards.
 * </p>
 */
public final class CoreMLRuntime {

    private static final String COREML_FRAMEWORK = "/System/Library/Frameworks/CoreML.framework/CoreML";
    private static final SymbolLookup COREML = ObjC.isAvailable() ? FFMSupport.loadLibrary(COREML_FRAMEWORK) : null;

    /** {@code MLFeatureTypeMultiArray}. */
    private static final long ML_FEATURE_TYPE_MULTI_ARRAY = 5;

    private static final boolean AVAILABLE = COREML != null && objcClass("MLModel") != 0 && supportsOutputBackings();

    private CoreMLRuntime() {
    }

    /** Output backings arrived in macOS 13; without them no output could be written in place. */
    private static boolean supportsOutputBackings() {
        long options = objcClass("MLPredictionOptions");
        return options != 0 && ObjC.sendBoolean(options, "instancesRespondToSelector:", ObjC.sel("setOutputBackings:"));
    }

    /** Whether Core ML (macOS 13 or newer) can be used on this host. */
    public static boolean isAvailable() {
        return AVAILABLE;
    }

    private static void checkAvailable() {
        if (!AVAILABLE) {
            throw new TornadoLibraryTaskNotSupported("Core ML (macOS 13 or newer) is not available on this host");
        }
    }

    /** The Core ML element type of a TornadoVM array; other arrays are not supported. */
    public static CoreMLDataType dataTypeOf(Object array) {
        return switch (array) {
            case FloatArray ignored -> CoreMLDataType.FLOAT32;
            case HalfFloatArray ignored -> CoreMLDataType.FLOAT16;
            case DoubleArray ignored -> CoreMLDataType.FLOAT64;
            case IntArray ignored -> CoreMLDataType.INT32;
            default -> throw new TornadoLibraryTaskNotSupported("Core ML takes FloatArray, HalfFloatArray, DoubleArray and IntArray arguments, not " + array.getClass().getSimpleName());
        };
    }

    // ---------------------------------------------------------------- Loading

    /**
     * Loads a model: a compiled {@code .mlmodelc} as it is, an {@code .mlpackage} or {@code .mlmodel}
     * after compiling it. The {@code MLModel} of the result is retained until the model is closed.
     */
    public static CoreMLModel load(String path, CoreMLOptions options) {
        checkAvailable();
        if (!Files.exists(Path.of(path))) {
            throw new TornadoRuntimeException("[ERROR] Core ML model not found: " + path);
        }
        long pool = poolPush();
        long model = 0;
        try {
            MemorySegment error = FFMSupport.scratchPointer();
            long url = send(objcClass("NSURL"), "fileURLWithPath:", nsString(path));
            if (!path.endsWith(".mlmodelc") && !path.endsWith(".mlmodelc/")) {
                // Compiled into a temporary directory that Core ML loads the model from.
                url = send(objcClass("MLModel"), "compileModelAtURL:error:", url, error.address());
                if (url == 0) {
                    throw new TornadoRuntimeException("[ERROR] Cannot compile Core ML model " + path + ": " + errorMessage(error));
                }
            }
            long configuration = newObject("MLModelConfiguration");
            try {
                send(configuration, "setComputeUnits:", options.getComputeUnits().value());
                if (options.getFunction() != null) {
                    if (!ObjC.sendBoolean(configuration, "respondsToSelector:", ObjC.sel("setFunctionName:"))) {
                        throw new TornadoLibraryTaskNotSupported("Core ML model functions need macOS 15 or newer");
                    }
                    send(configuration, "setFunctionName:", nsString(options.getFunction()));
                }
                error = FFMSupport.scratchPointer();
                model = send(objcClass("MLModel"), "modelWithContentsOfURL:configuration:error:", url, configuration, error.address());
            } finally {
                ObjC.release(configuration);
            }
            if (model == 0) {
                throw new TornadoRuntimeException("[ERROR] Cannot load Core ML model " + path + ": " + errorMessage(error));
            }
            ObjC.retain(model);
            long description = send(model, "modelDescription");
            List<Feature> inputs = features(path, send(description, "inputDescriptionsByName"), options.getInputs(), options, true);
            List<Feature> outputs = features(path, send(description, "outputDescriptionsByName"), options.getOutputs(), options, false);
            CoreMLModel loaded = new CoreMLModel(path, options, inputs, outputs, model);
            model = 0;
            return loaded;
        } finally {
            ObjC.release(model);
            poolPop(pool);
        }
    }

    /** The multi-array features of a description dictionary, in the requested order or by name. */
    private static List<Feature> features(String path, long byName, List<String> order, CoreMLOptions options, boolean inputs) {
        String kind = inputs ? "input" : "output";
        long keys = send(byName, "allKeys");
        TreeSet<String> names = new TreeSet<>();
        for (long i = 0; i < count(keys); i++) {
            names.add(string(objectAt(keys, i)));
        }
        List<String> ordered = order.isEmpty() ? new ArrayList<>(names) : order;
        if (ordered.size() != names.size() || !new HashSet<>(ordered).equals(names)) {
            throw new TornadoRuntimeException("[ERROR] Core ML model " + path + " has the " + kind + "s " + names + ", not " + ordered);
        }
        List<Feature> features = new ArrayList<>();
        for (String name : ordered) {
            long description = send(byName, "objectForKey:", nsString(name));
            long type = send(description, "type");
            if (type != ML_FEATURE_TYPE_MULTI_ARRAY) {
                throw new TornadoLibraryTaskNotSupported("Core ML " + kind + " " + name + " of " + path + " is not a multi-array (MLFeatureType " + type + ")");
            }
            long constraint = send(description, "multiArrayConstraint");
            long rawType = send(constraint, "dataType");
            CoreMLDataType dataType = CoreMLDataType.of(rawType);
            if (dataType == null) {
                throw new TornadoLibraryTaskNotSupported("Core ML " + kind + " " + name + " of " + path + " has an unsupported element type (MLMultiArrayDataType " + rawType + ")");
            }
            long[] shape = options.getShape(name);
            if (shape == null) {
                shape = longs(send(constraint, "shape"));
            }
            for (long d : shape) {
                if (d <= 0) {
                    throw new TornadoRuntimeException("[ERROR] Core ML " + kind + " " + name + " of " + path + " has no fixed shape " + Arrays.toString(shape) + "; set one with CoreMLOptions.shape");
                }
            }
            if (inputs && shape.length == 0) {
                throw new TornadoRuntimeException("[ERROR] Core ML input " + name + " of " + path + " has no shape; set one with CoreMLOptions.shape");
            }
            features.add(new Feature(name, shape, dataType));
        }
        return features;
    }

    /** Releases an object retained by {@link #load}. */
    public static void release(long object) {
        ObjC.release(object);
    }

    // ---------------------------------------------------------------- Prediction

    /** Row-major strides, in elements. */
    private static long[] contiguousStrides(long[] shape) {
        long[] strides = new long[shape.length];
        long stride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }

    /** {@code MLMultiArray} over memory it does not own (no deallocator); owned by the caller. */
    private static long wrap(long address, long[] shape, CoreMLDataType type) {
        MemorySegment error = FFMSupport.scratchPointer();
        long array = send(objcClass("MLMultiArray"), "alloc");
        array = send(array, "initWithDataPointer:shape:dataType:strides:deallocator:error:", address, nsNumbers(shape), type.value(), nsNumbers(contiguousStrides(shape)), 0L, error.address());
        if (array == 0) {
            throw new TornadoRuntimeException("[ERROR] Cannot wrap a TornadoVM buffer as a Core ML multi-array: " + errorMessage(error));
        }
        return array;
    }

    /**
     * The Core ML objects for one set of buffers of one model, built on first use and reused by every
     * later prediction on the same buffers: the input feature provider and the prediction options
     * that carry the output backings.
     */
    static final class Binding {
        private final long features;
        private final long backedOptions;
        private final long plainOptions;
        private final long[] outputNames;
        private boolean useBackings = true;

        private Binding(long features, long backedOptions, long plainOptions, long[] outputNames) {
            this.features = features;
            this.backedOptions = backedOptions;
            this.plainOptions = plainOptions;
            this.outputNames = outputNames;
        }

        void release() {
            ObjC.release(features);
            ObjC.release(backedOptions);
            ObjC.release(plainOptions);
            for (long name : outputNames) {
                ObjC.release(name);
            }
        }
    }

    /** The key of a binding: the model, and the address, type and size of every argument. */
    record BindingKey(CoreMLModel model, long[] addresses, CoreMLDataType[] types, long[] sizes) {
        @Override
        public boolean equals(Object o) {
            return o instanceof BindingKey k && model == k.model && Arrays.equals(addresses, k.addresses) && Arrays.equals(types, k.types) && Arrays.equals(sizes, k.sizes);
        }

        @Override
        public int hashCode() {
            return 31 * System.identityHashCode(model) + Arrays.hashCode(addresses);
        }
    }

    /** Builds the binding of a model's arguments; called inside an autorelease pool. */
    static Binding bind(CoreMLModel model, CoreMLDataType[] types, long[] addresses) {
        List<Feature> inputs = model.getInputs();
        List<Feature> outputs = model.getOutputs();
        long[] inputNames = new long[inputs.size()];
        long[] inputValues = new long[inputs.size()];
        for (int i = 0; i < inputs.size(); i++) {
            Feature feature = inputs.get(i);
            long array = wrap(addresses[i], feature.shape(), types[i]);
            inputNames[i] = nsString(feature.name());
            inputValues[i] = send(objcClass("MLFeatureValue"), "featureValueWithMultiArray:", array);
            ObjC.release(array);
        }
        MemorySegment error = FFMSupport.scratchPointer();
        long features = send(send(objcClass("MLDictionaryFeatureProvider"), "alloc"), "initWithDictionary:error:", nsDictionary(inputNames, inputValues), error.address());
        if (features == 0) {
            throw new TornadoRuntimeException("[ERROR] Cannot build the Core ML inputs of " + model.getPath() + ": " + errorMessage(error));
        }

        Map<Long, Long> backings = new LinkedHashMap<>();
        long[] outputNames = new long[outputs.size()];
        for (int o = 0; o < outputs.size(); o++) {
            Feature feature = outputs.get(o);
            int position = inputs.size() + o;
            outputNames[o] = nsString(feature.name());
            ObjC.retain(outputNames[o]);
            // An output whose shape the description leaves open cannot have a backing: Core ML requires
            // the backing's shape to be the output's.
            if (feature.shape().length > 0) {
                long array = wrap(addresses[position], feature.shape(), types[position]);
                backings.put(outputNames[o], array);
            }
        }
        long backedOptions = newObject("MLPredictionOptions");
        if (!backings.isEmpty()) {
            long[] keys = backings.keySet().stream().mapToLong(Long::longValue).toArray();
            long[] values = backings.values().stream().mapToLong(Long::longValue).toArray();
            send(backedOptions, "setOutputBackings:", nsDictionary(keys, values));
            for (long array : values) {
                ObjC.release(array);
            }
        }
        return new Binding(features, backedOptions, newObject("MLPredictionOptions"), outputNames);
    }

    /**
     * Runs one prediction of {@code model} on a binding of its arguments and makes sure every output
     * is in its TornadoVM buffer. Called inside an autorelease pool.
     */
    static void predict(CoreMLModel model, Binding binding, CoreMLDataType[] types, long[] addresses, long[] sizes) {
        long handle = model.nativeHandle();
        MemorySegment error = FFMSupport.scratchPointer();
        long result = 0;
        if (binding.useBackings) {
            result = send(handle, "predictionFromFeatures:options:error:", binding.features, binding.backedOptions, error.address());
            if (result == 0) {
                // A backing Core ML refuses (alignment, type) fails the whole prediction; from now on
                // this binding predicts without backings and copies its outputs instead.
                binding.useBackings = false;
            }
        }
        if (result == 0) {
            error = FFMSupport.scratchPointer();
            result = send(handle, "predictionFromFeatures:options:error:", binding.features, binding.plainOptions, error.address());
        }
        if (result == 0) {
            throw new TornadoRuntimeException("[ERROR] Core ML prediction of " + model.getPath() + " failed: " + errorMessage(error));
        }
        int numInputs = model.getInputs().size();
        for (int o = 0; o < binding.outputNames.length; o++) {
            int position = numInputs + o;
            long value = send(result, "featureValueForName:", binding.outputNames[o]);
            long array = value == 0 ? 0 : send(value, "multiArrayValue");
            if (array == 0) {
                throw new TornadoRuntimeException("[ERROR] Core ML model " + model.getPath() + " returned no multi-array " + model.getOutputs().get(o).name());
            }
            if (send(array, "dataPointer") != addresses[position]) {
                copyOut(model, model.getOutputs().get(o), array, addresses[position], types[position], sizes[position]);
            }
        }
    }

    /** Copies a multi-array Core ML returned into a contiguous TornadoVM buffer, converting its elements. */
    private static void copyOut(CoreMLModel model, Feature feature, long array, long destination, CoreMLDataType destinationType, long destinationSize) {
        CoreMLDataType sourceType = CoreMLDataType.of(send(array, "dataType"));
        long[] shape = longs(send(array, "shape"));
        long[] strides = longs(send(array, "strides"));
        long elements = 1;
        for (long d : shape) {
            elements *= d;
        }
        if (sourceType == null) {
            throw new TornadoLibraryTaskNotSupported("Core ML output " + feature.name() + " of " + model.getPath() + " has an unsupported element type");
        }
        if (elements != destinationSize) {
            throw new TornadoRuntimeException("[ERROR] Core ML output " + feature.name() + " of " + model.getPath() + " has " + elements + " elements " + Arrays.toString(shape)
                    + "; its TornadoVM array has " + destinationSize);
        }
        long extent = 1;
        for (int d = 0; d < shape.length; d++) {
            extent += (shape[d] - 1) * strides[d];
        }
        MemorySegment src = FFMSupport.asSegment(send(array, "dataPointer"), extent * sourceType.byteSize());
        MemorySegment dst = FFMSupport.asSegment(destination, elements * destinationType.byteSize());
        if (sourceType == destinationType && Arrays.equals(strides, contiguousStrides(shape))) {
            MemorySegment.copy(src, 0, dst, 0, elements * sourceType.byteSize());
            return;
        }
        long[] index = new long[shape.length];
        for (long i = 0; i < elements; i++) {
            long offset = 0;
            for (int d = 0; d < shape.length; d++) {
                offset += index[d] * strides[d];
            }
            write(dst, destinationType, i, read(src, sourceType, offset));
            for (int d = shape.length - 1; d >= 0; d--) {
                if (++index[d] < shape[d]) {
                    break;
                }
                index[d] = 0;
            }
        }
    }

    private static double read(MemorySegment segment, CoreMLDataType type, long index) {
        return switch (type) {
            case FLOAT32 -> segment.getAtIndex(ValueLayout.JAVA_FLOAT, index);
            case FLOAT16 -> Float.float16ToFloat(segment.getAtIndex(ValueLayout.JAVA_SHORT, index));
            case FLOAT64 -> segment.getAtIndex(ValueLayout.JAVA_DOUBLE, index);
            case INT32 -> segment.getAtIndex(ValueLayout.JAVA_INT, index);
        };
    }

    private static void write(MemorySegment segment, CoreMLDataType type, long index, double value) {
        switch (type) {
            case FLOAT32 -> segment.setAtIndex(ValueLayout.JAVA_FLOAT, index, (float) value);
            case FLOAT16 -> segment.setAtIndex(ValueLayout.JAVA_SHORT, index, Float.floatToFloat16((float) value));
            case FLOAT64 -> segment.setAtIndex(ValueLayout.JAVA_DOUBLE, index, value);
            case INT32 -> segment.setAtIndex(ValueLayout.JAVA_INT, index, (int) Math.rint(value));
        }
    }
}
