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

import java.lang.foreign.MemorySegment;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

import uk.ac.manchester.tornado.api.enums.TornadoVMBackendType;
import uk.ac.manchester.tornado.api.exceptions.TornadoRuntimeException;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.api.types.arrays.TornadoNativeArray;
import uk.ac.manchester.tornado.mlx.Mlx;
import uk.ac.manchester.tornado.runtime.common.TornadoXPUDevice;
import uk.ac.manchester.tornado.runtime.ffm.FFMSupport;
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

    private record WrapKey(long address, int elements, int dtype) {
    }

    private static final class MlxContext implements LibraryContext {
        final MemorySegment stream = MlxNativeLib.gpuStream();
        final Map<WrapKey, MemorySegment> wrappers = new HashMap<>();
        final Map<Long, Integer> retainedBuffers = new HashMap<>();
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
            ctx.wrappers.values().forEach(MlxNativeLib::free);
            ctx.wrappers.clear();
            ctx.retainedBuffers.forEach((buffer, count) -> {
                for (int i = 0; i < count; i++) {
                    MlxNativeLib.release(buffer);
                }
            });
            ctx.retainedBuffers.clear();
            MlxNativeLib.freeStream(ctx.stream);
        }
    }

    @Override
    public void dispatch(String functionName, LibraryInvocation invocation) {
        MlxContext ctx = (MlxContext) invocation.getContext();
        synchronized (ctx) {
            switch (functionName) {
                case "add" -> binary(ctx, invocation, MlxNativeLib::add);
                default -> throw new TornadoRuntimeException("[ERROR] Unknown MLX function: " + functionName);
            }
        }
    }

    private interface BinaryOp {
        MemorySegment apply(MemorySegment a, MemorySegment b, MemorySegment stream);
    }

    private static void binary(MlxContext ctx, LibraryInvocation invocation, BinaryOp op) {
        int n = elements(invocation, 2);
        if (elements(invocation, 0) != n || elements(invocation, 1) != n) {
            throw new TornadoRuntimeException("[ERROR] MLX element-wise operands must have the same length");
        }
        int dtype = dtype(invocation, 2);
        MemorySegment a = input(ctx, invocation, 0, n, dtype);
        MemorySegment b = input(ctx, invocation, 1, n, dtype);
        MemorySegment c = op.apply(a, b, ctx.stream);
        try {
            MlxNativeLib.eval(c);
            copyOut(c, invocation, 2);
        } finally {
            MlxNativeLib.free(c);
        }
    }

    private static int elements(LibraryInvocation invocation, int index) {
        return ((TornadoNativeArray) invocation.getArg(index)).getSize();
    }

    private static int dtype(LibraryInvocation invocation, int index) {
        Object array = invocation.getArg(index);
        if (array instanceof FloatArray) {
            return MlxNativeLib.MLX_FLOAT32;
        } else if (array instanceof HalfFloatArray) {
            return MlxNativeLib.MLX_FLOAT16;
        } else if (array instanceof IntArray) {
            return MlxNativeLib.MLX_INT32;
        }
        throw new TornadoRuntimeException("[ERROR] MLX does not support arguments of type " + array.getClass().getSimpleName());
    }

    /** The MLX array over argument {@code index}'s TornadoVM buffer, wrapped once per plan. */
    private static MemorySegment input(MlxContext ctx, LibraryInvocation invocation, int index, int elements, int dtype) {
        long address = invocation.getDevicePointer(index);
        WrapKey key = new WrapKey(address, elements, dtype);
        MemorySegment wrapper = ctx.wrappers.get(key);
        if (wrapper != null) {
            return wrapper;
        }
        long nativeBuffer = invocation.getNativeBuffer(index);
        MlxNativeLib.retain(nativeBuffer);
        ctx.retainedBuffers.merge(nativeBuffer, 1, Integer::sum);

        long releasesBefore = MlxNativeLib.releases();
        wrapper = MlxNativeLib.wrap(address, new int[] { elements }, dtype);
        MlxNativeLib.eval(wrapper);
        if (MlxNativeLib.releases() != releasesBefore || MlxNativeLib.dataAddress(wrapper) != address) {
            COPY_FALLBACKS.incrementAndGet();
        }
        ctx.wrappers.put(key, wrapper);
        return wrapper;
    }

    private static void copyOut(MemorySegment result, LibraryInvocation invocation, int index) {
        long bytes = MlxNativeLib.nbytes(result);
        long expected = (long) elements(invocation, index) * ((TornadoNativeArray) invocation.getArg(index)).getElementSize();
        if (bytes != expected) {
            throw new TornadoRuntimeException("[ERROR] MLX result is " + bytes + " bytes, the output array holds " + expected);
        }
        MemorySegment source = FFMSupport.asSegment(MlxNativeLib.dataAddress(result), bytes);
        FFMSupport.asSegment(invocation.getDevicePointer(index), bytes).copyFrom(source);
    }
}
