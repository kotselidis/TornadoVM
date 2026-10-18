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

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

import uk.ac.manchester.tornado.api.common.LibraryTaskDescriptor;
import uk.ac.manchester.tornado.api.enums.TornadoVMBackendType;
import uk.ac.manchester.tornado.api.exceptions.TornadoLibraryTaskNotSupported;
import uk.ac.manchester.tornado.api.exceptions.TornadoRuntimeException;
import uk.ac.manchester.tornado.api.types.arrays.TornadoNativeArray;
import uk.ac.manchester.tornado.coreml.CoreML;
import uk.ac.manchester.tornado.coreml.CoreMLDataType;
import uk.ac.manchester.tornado.coreml.CoreMLModel;
import uk.ac.manchester.tornado.coreml.CoreMLModel.Feature;
import uk.ac.manchester.tornado.runtime.common.TornadoXPUDevice;
import uk.ac.manchester.tornado.runtime.library.spi.LibraryContext;
import uk.ac.manchester.tornado.runtime.library.spi.LibraryInvocation;
import uk.ac.manchester.tornado.runtime.library.spi.TornadoLibraryProvider;
import uk.ac.manchester.tornado.runtime.library.spi.TornadoNativeStreamSupport;

/**
 * {@link TornadoLibraryProvider} for Apple Core ML. Discovered by the TornadoVM runtime through
 * {@link java.util.ServiceLoader}.
 *
 * <p>
 * A Core ML task is one prediction of a whole model, which Core ML schedules on the CPU, the GPU and
 * the Neural Engine itself. The task's arrays are the model's inputs and outputs, read and written in
 * place in TornadoVM's shared-storage Metal buffers ({@link CoreMLRuntime}). Core ML runs on the host
 * and returns when the prediction is done, so the kernels queued before the task are finished first
 * and the kernels after it see its outputs.
 * </p>
 */
public final class CoreMLLibraryProvider implements TornadoLibraryProvider {

    /** Bindings kept per plan before they are all dropped; one per distinct set of buffers of a model. */
    private static final int MAX_BINDINGS = 64;

    /** Set by the Metal backend's {@code -Dtornado.mlx.async=true}: the queue is then flushed, not finished, before a library task. */
    private static final boolean ASYNC_LIBRARY_DISPATCH = Boolean.getBoolean("tornado.mlx.async");

    private static final AtomicLong PREDICTIONS = new AtomicLong();

    /** The Core ML objects of one execution plan on one device. */
    static final class Session implements LibraryContext {
        private final Map<CoreMLRuntime.BindingKey, CoreMLRuntime.Binding> bindings = new HashMap<>();

        synchronized void predict(CoreMLModel model, CoreMLDataType[] types, long[] addresses, long[] sizes) {
            long pool = ObjC.poolPush();
            try {
                CoreMLRuntime.BindingKey key = new CoreMLRuntime.BindingKey(model, addresses, types, sizes);
                CoreMLRuntime.Binding binding = bindings.get(key);
                if (binding == null) {
                    if (bindings.size() >= MAX_BINDINGS) {
                        clear();
                    }
                    binding = CoreMLRuntime.bind(model, types, addresses);
                    bindings.put(key, binding);
                }
                CoreMLRuntime.predict(model, binding, types, addresses, sizes);
            } finally {
                ObjC.poolPop(pool);
            }
        }

        synchronized void clear() {
            bindings.values().forEach(CoreMLRuntime.Binding::release);
            bindings.clear();
        }
    }

    /** Whether Core ML can be used on this host. */
    public static boolean isAvailable() {
        return CoreMLRuntime.isAvailable();
    }

    /** How many Core ML predictions ran as library tasks, since start-up. */
    public static long predictions() {
        return PREDICTIONS.get();
    }

    @Override
    public String libraryName() {
        return CoreML.LIBRARY_NAME;
    }

    @Override
    public boolean canHandle(TornadoXPUDevice device) {
        return device.getTornadoVMBackend() == TornadoVMBackendType.METAL && isAvailable();
    }

    @Override
    public LibraryContext createContext(TornadoXPUDevice device, long executionPlanId) {
        if (!isAvailable()) {
            throw new TornadoLibraryTaskNotSupported("Core ML (macOS 13 or newer) is not available on this host");
        }
        return new Session();
    }

    @Override
    public void destroyContext(LibraryContext context) {
        if (context instanceof Session session) {
            session.clear();
        }
    }

    @Override
    public void prepare(LibraryTaskDescriptor descriptor, LibraryContext context) {
        CoreMLModel model = modelOf(descriptor.getTuning());
        Object[] parameters = descriptor.getParameters();
        checkArguments(model, parameters.length, parameters);
    }

    @Override
    public void dispatch(String functionName, LibraryInvocation invocation) {
        if (!CoreML.PREDICT.equals(functionName)) {
            throw new TornadoRuntimeException("[ERROR] Unknown Core ML function: " + functionName);
        }
        if (invocation.isCapturing()) {
            throw new TornadoLibraryTaskNotSupported("Core ML predictions run on the host and cannot be captured");
        }
        CoreMLModel model = modelOf(invocation.getTuning());
        int numArgs = invocation.getNumArgs();
        Object[] arrays = new Object[numArgs];
        for (int i = 0; i < numArgs; i++) {
            if (!invocation.isReference(i)) {
                throw new TornadoRuntimeException("[ERROR] Core ML " + model.getPath() + ": argument " + i + " is not an array");
            }
            arrays[i] = invocation.getArg(i);
        }
        CoreMLDataType[] types = checkArguments(model, numArgs, arrays);
        long[] addresses = new long[numArgs];
        long[] sizes = new long[numArgs];
        for (int i = 0; i < numArgs; i++) {
            addresses[i] = invocation.getDevicePointer(i);
            sizes[i] = ((TornadoNativeArray) arrays[i]).getSizeLong();
        }

        // Core ML reads and writes the buffers on the host: the kernels before this task must be done.
        if (invocation.getDevice() instanceof TornadoNativeStreamSupport streams) {
            long queue = streams.getNativeStream(invocation.getExecutionPlanId());
            if (ASYNC_LIBRARY_DISPATCH) {
                waitForQueue(queue);
            }
        }
        ((Session) invocation.getContext()).predict(model, types, addresses, sizes);
        PREDICTIONS.incrementAndGet();
    }

    /** Waits for everything committed to a command queue, behind an empty command buffer. */
    private static void waitForQueue(long queue) {
        long pool = ObjC.poolPush();
        try {
            long commandBuffer = ObjC.send(queue, "commandBuffer");
            ObjC.send(commandBuffer, "commit");
            ObjC.send(commandBuffer, "waitUntilCompleted");
        } finally {
            ObjC.poolPop(pool);
        }
    }

    private static CoreMLModel modelOf(Object tuning) {
        if (!(tuning instanceof CoreMLModel model)) {
            throw new TornadoRuntimeException("[ERROR] A Core ML task carries its CoreMLModel; build it with CoreML::predict");
        }
        if (model.isClosed()) {
            throw new TornadoRuntimeException("[ERROR] Core ML model " + model.getPath() + " is closed");
        }
        return model;
    }

    /** Checks that the arrays are the model's inputs then outputs, by type and size; returns their types. */
    static CoreMLDataType[] checkArguments(CoreMLModel model, int numArgs, Object[] arrays) {
        if (numArgs != model.getNumFeatures()) {
            throw new TornadoRuntimeException("[ERROR] Core ML model " + model.getPath() + " takes " + model.getInputs().size() + " inputs " + model.getInputs() + " and " + model.getOutputs().size()
                    + " outputs " + model.getOutputs() + ", not " + numArgs + " arrays");
        }
        CoreMLDataType[] types = new CoreMLDataType[numArgs];
        for (int i = 0; i < numArgs; i++) {
            Feature feature = model.getFeature(i);
            if (!(arrays[i] instanceof TornadoNativeArray array)) {
                throw new TornadoRuntimeException("[ERROR] Core ML " + feature + " of " + model.getPath() + " must be a TornadoVM array, not " + arrays[i]);
            }
            types[i] = CoreMLRuntime.dataTypeOf(array);
            if (feature.shape().length > 0 && array.getSizeLong() != feature.elements()) {
                throw new TornadoRuntimeException("[ERROR] Core ML " + feature + " of " + model.getPath() + " has " + feature.elements() + " elements; its array has " + array.getSizeLong());
            }
        }
        return types;
    }
}
