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
package uk.ac.manchester.tornado.mlx.provider;

import static java.util.Map.entry;

import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Consumer;

import uk.ac.manchester.tornado.api.enums.TornadoVMBackendType;
import uk.ac.manchester.tornado.api.exceptions.TornadoRuntimeException;
import uk.ac.manchester.tornado.mlx.Mlx;
import uk.ac.manchester.tornado.mlx.MlxOptions;
import uk.ac.manchester.tornado.runtime.common.TornadoXPUDevice;
import uk.ac.manchester.tornado.runtime.library.spi.LibraryContext;
import uk.ac.manchester.tornado.runtime.library.spi.LibraryInvocation;
import uk.ac.manchester.tornado.runtime.library.spi.TornadoLibraryProvider;

/**
 * {@link TornadoLibraryProvider} for Apple MLX, reached through mlx-c. Discovered by the
 * TornadoVM runtime through {@link java.util.ServiceLoader}.
 *
 * <p>
 * Inputs are adopted, not copied: each TornadoVM buffer is wrapped as an MLX array over the same
 * shared-storage memory. MLX runs the operation on its own command queue and waits for it; the
 * Metal backend waits after every kernel too, so the two queues never overlap. MLX operations
 * allocate their own result, which is copied into the output buffer.
 * </p>
 *
 * <p>
 * Wrapping a buffer makes MLX create a no-copy MTLBuffer the GPU has to map, which is costly for
 * large arrays, so wrappers are cached per execution plan. The cache key is the data address, so
 * the provider retains each TornadoVM MTLBuffer it has wrapped: while it is retained, no other
 * allocation can take that address, and a cached wrapper can never outlive its memory. Wrappers
 * and retains are dropped when the execution plan's context is destroyed.
 * </p>
 */
public final class MlxLibraryProvider implements TornadoLibraryProvider {

    private static final AtomicLong COPY_FALLBACKS = new AtomicLong();

    /** Operations MLX only implements on the CPU stream. */
    private static final Set<String> CPU_ONLY = Set.of();

    private interface Unary {
        int apply(MemorySegment res, MemorySegment a, MemorySegment stream);
    }

    private interface Binary {
        int apply(MemorySegment res, MemorySegment a, MemorySegment b, MemorySegment stream);
    }

    private static final Map<String, Consumer<MlxCall>> OPERATIONS = Map.ofEntries(
            // Element-wise binary: (a, b, out), all the same length.
            entry("add", c -> binary(c, "mlx_add", MlxC::mlx_add)), //
            entry("subtract", c -> binary(c, "mlx_subtract", MlxC::mlx_subtract)), //
            entry("multiply", c -> binary(c, "mlx_multiply", MlxC::mlx_multiply)), //
            entry("divide", c -> binary(c, "mlx_divide", MlxC::mlx_divide)), //
            entry("maximum", c -> binary(c, "mlx_maximum", MlxC::mlx_maximum)), //
            entry("minimum", c -> binary(c, "mlx_minimum", MlxC::mlx_minimum)), //
            // Element-wise unary: (a, out).
            entry("negative", c -> unary(c, "mlx_negative", MlxC::mlx_negative)), //
            entry("exp", c -> unary(c, "mlx_exp", MlxC::mlx_exp)), //
            entry("tanh", c -> unary(c, "mlx_tanh", MlxC::mlx_tanh)), //
            entry("erf", c -> unary(c, "mlx_erf", MlxC::mlx_erf)), //
            entry("sigmoid", c -> unary(c, "mlx_sigmoid", MlxC::mlx_sigmoid)), //
            entry("sqrt", c -> unary(c, "mlx_sqrt", MlxC::mlx_sqrt)), //
            entry("rsqrt", c -> unary(c, "mlx_rsqrt", MlxC::mlx_rsqrt)), //
            entry("square", c -> unary(c, "mlx_square", MlxC::mlx_square)));

    /** Whether libmlxc can be loaded on this host. */
    public static boolean isAvailable() {
        try {
            MlxNativeLib.load();
            return true;
        } catch (RuntimeException e) {
            return false;
        }
    }

    /** MLX version string, e.g. "0.32.1". */
    public static String mlxVersion() {
        return MlxNativeLib.version();
    }

    /**
     * Number of inputs MLX copied instead of adopting, since start-up. MLX falls back to a copy,
     * silently, when Metal refuses to wrap the memory; results stay correct but the zero-copy
     * path is lost. Tests assert this stays zero.
     */
    public static long copyFallbacks() {
        return COPY_FALLBACKS.get();
    }

    /** Names of the operations this provider dispatches, e.g. "add". */
    public static Set<String> operations() {
        return OPERATIONS.keySet();
    }

    private record WrapKey(long address, List<Integer> shape, int dtype) {
    }

    /** Per execution plan: MLX streams and the wrappers of TornadoVM buffers. */
    static final class MlxContext implements LibraryContext {
        private final MemorySegment gpuStream = MlxC.mlx_default_gpu_stream_new();
        private MemorySegment cpuStream;
        private final Map<WrapKey, MemorySegment> wrappers = new HashMap<>();
        private final Map<Long, Integer> retainedBuffers = new HashMap<>();

        MemorySegment stream(MlxOptions.Device device) {
            if (device == MlxOptions.Device.GPU) {
                return gpuStream;
            }
            if (cpuStream == null) {
                cpuStream = MlxC.mlx_default_cpu_stream_new();
            }
            return cpuStream;
        }

        /** The MLX array over a TornadoVM buffer, wrapped once per plan and shape. */
        MemorySegment wrapper(long address, long nativeBuffer, int[] shape, int dtype) {
            WrapKey key = new WrapKey(address, Arrays.stream(shape).boxed().toList(), dtype);
            MemorySegment wrapper = wrappers.get(key);
            if (wrapper != null) {
                return wrapper;
            }
            MlxNativeLib.retain(nativeBuffer);
            retainedBuffers.merge(nativeBuffer, 1, Integer::sum);

            long releasesBefore = MlxNativeLib.releases();
            wrapper = MlxNativeLib.wrap(address, shape, dtype);
            MlxNativeLib.eval(wrapper);
            if (MlxNativeLib.releases() != releasesBefore || MlxNativeLib.dataAddress(wrapper) != address) {
                COPY_FALLBACKS.incrementAndGet();
            }
            wrappers.put(key, wrapper);
            return wrapper;
        }

        void destroy() {
            wrappers.values().forEach(MlxNativeLib::free);
            wrappers.clear();
            retainedBuffers.forEach((buffer, count) -> {
                for (int i = 0; i < count; i++) {
                    MlxNativeLib.release(buffer);
                }
            });
            retainedBuffers.clear();
            MlxC.mlx_stream_free(gpuStream);
            if (cpuStream != null) {
                MlxC.mlx_stream_free(cpuStream);
            }
        }
    }

    @Override
    public String libraryName() {
        return Mlx.LIBRARY_NAME;
    }

    @Override
    public boolean canHandle(TornadoXPUDevice device) {
        return device.getTornadoVMBackend() == TornadoVMBackendType.METAL && isAvailable();
    }

    @Override
    public LibraryContext createContext(TornadoXPUDevice device, long executionPlanId) {
        MlxNativeLib.load();
        return new MlxContext();
    }

    @Override
    public void destroyContext(LibraryContext context) {
        MlxContext ctx = (MlxContext) context;
        synchronized (ctx) {
            ctx.destroy();
        }
    }

    @Override
    public void dispatch(String functionName, LibraryInvocation invocation) {
        Consumer<MlxCall> operation = OPERATIONS.get(functionName);
        if (operation == null) {
            throw new TornadoRuntimeException("[ERROR] Unknown MLX function: " + functionName);
        }
        MlxContext ctx = (MlxContext) invocation.getContext();
        MlxOptions.Device device = CPU_ONLY.contains(functionName) ? MlxOptions.Device.CPU
                : (invocation.getTuning() instanceof MlxOptions options) ? options.getDevice() : MlxOptions.Device.GPU;
        synchronized (ctx) {
            try (MlxCall call = new MlxCall(ctx, invocation, ctx.stream(device))) {
                operation.accept(call);
            }
        }
    }

    // ---------------------------------------------------------------- operation families

    private static void binary(MlxCall c, String name, Binary op) {
        int n = c.length(2);
        if (c.length(0) != n || c.length(1) != n) {
            throw new TornadoRuntimeException("[ERROR] MLX element-wise operands must have the same length");
        }
        MemorySegment a = c.input(0, n);
        MemorySegment b = c.input(1, n);
        c.store(c.op(name, res -> op.apply(res, a, b, c.stream())), 2);
    }

    private static void unary(MlxCall c, String name, Unary op) {
        int n = c.length(1);
        if (c.length(0) != n) {
            throw new TornadoRuntimeException("[ERROR] MLX element-wise input and output must have the same length");
        }
        MemorySegment a = c.input(0, n);
        c.store(c.op(name, res -> op.apply(res, a, c.stream())), 1);
    }
}
