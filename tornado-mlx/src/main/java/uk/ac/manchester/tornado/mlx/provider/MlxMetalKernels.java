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

import static uk.ac.manchester.tornado.runtime.ffm.FFMSupport.C_LONG;
import static uk.ac.manchester.tornado.runtime.ffm.FFMSupport.C_POINTER;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import uk.ac.manchester.tornado.api.exceptions.TornadoRuntimeException;
import uk.ac.manchester.tornado.runtime.ffm.FFMSupport;

/**
 * Runs MLX's own Metal kernels, from its {@code mlx.metallib}, on TornadoVM's command queue with
 * TornadoVM's buffers bound in place, the way JIT-compiled kernels run. MLX's C API always allocates
 * a new array for a result, which then has to be copied into the TornadoVM buffer; here the kernel
 * writes the TornadoVM buffer directly, so there is no MLX array and no copy.
 *
 * <p>
 * Only kernels whose launch is simple enough to reproduce exactly are used: the contiguous
 * element-wise ones ({@code v_}/{@code vn_} unary, {@code vv_}/{@code vvn_} binary). Their names,
 * arguments, work per thread and grid follow MLX's {@code unary.cpp} and {@code binary.cpp}.
 */
final class MlxMetalKernels {

    /** Element-wise kernels process this many elements per thread from this size up (MLX's {@code get_work_per_thread}). */
    private static final int WORK_PER_THREAD_THRESHOLD = 1 << 16;

    private static final long SPIN_BUDGET_NS = 1_000_000;
    private static final int STATUS_POLL_INTERVAL = 64;
    private static final long MTL_COMMAND_BUFFER_STATUS_COMPLETED = 4;

    private static final MemoryLayout MTL_SIZE = MemoryLayout.structLayout(C_LONG.withName("width"), C_LONG.withName("height"), C_LONG.withName("depth"));

    private static final SymbolLookup LIBOBJC = FFMSupport.loadLibrary("/usr/lib/libobjc.A.dylib");
    private static final SymbolLookup FOUNDATION = FFMSupport.loadLibrary("/System/Library/Frameworks/Foundation.framework/Foundation");
    private static final MethodHandle SEL_REGISTER_NAME = downcall(FunctionDescriptor.of(C_LONG, C_POINTER), "sel_registerName");
    private static final MethodHandle OBJC_GET_CLASS = downcall(FunctionDescriptor.of(C_LONG, C_POINTER), "objc_getClass");
    private static final MethodHandle POOL_PUSH = downcall(FunctionDescriptor.of(C_LONG), "objc_autoreleasePoolPush");
    private static final MethodHandle POOL_POP = downcall(FunctionDescriptor.ofVoid(C_LONG), "objc_autoreleasePoolPop");

    private static final Map<FunctionDescriptor, MethodHandle> SENDS = new ConcurrentHashMap<>();
    private static final Map<String, Long> SELECTORS = new ConcurrentHashMap<>();

    /** Pipelines by device and kernel name; the library is loaded once per device. */
    private static final Map<Long, Long> LIBRARIES = new ConcurrentHashMap<>();
    private static final Map<String, long[]> PIPELINES = new ConcurrentHashMap<>();

    /** One shared event per command queue, signalled at the end of each command buffer. */
    private static final Map<Long, long[]> QUEUE_EVENTS = new ConcurrentHashMap<>();

    private static final String METALLIB = System.getProperty("tornado.mlx.metallib", MlxNativeLib.metallibPath());

    private MlxMetalKernels() {
    }

    /** Whether MLX's kernel library can be found. */
    static boolean isAvailable() {
        return LIBOBJC != null && FOUNDATION != null && METALLIB != null && Files.isReadable(Path.of(METALLIB));
    }

    /** MLX's type suffix for a kernel name, or null for a type the element-wise route does not take. */
    static String typeName(int dtype) {
        return switch (dtype) {
            case MlxNativeLib.MLX_FLOAT32 -> "float32";
            case MlxNativeLib.MLX_FLOAT16 -> "float16";
            case MlxNativeLib.MLX_BFLOAT16 -> "bfloat16";
            default -> null;
        };
    }

    static int itemSize(int dtype) {
        return dtype == MlxNativeLib.MLX_FLOAT32 ? 4 : 2;
    }

    /**
     * Runs MLX's contiguous element-wise kernel {@code op} over {@code size} elements: {@code out =
     * op(in...)}. {@code buffers} and {@code offsets} hold the inputs then the output, as MTLBuffers
     * and the byte offsets of their first elements.
     */
    static void elementwise(long queue, String op, boolean binary, int dtype, boolean boolResult, int size, long[] buffers, long[] offsets) {
        String type = typeName(dtype);
        int workPerThread = size < WORK_PER_THREAD_THRESHOLD ? 1 : Math.max(1, 8 / itemSize(dtype));
        String prefix = (binary ? "vv" : "v") + (workPerThread > 1 ? "n_" : "_");
        // Unary kernels name the input and output types; binary ones only the input type.
        String name = prefix + op + type + (binary ? "" : (boolResult ? "bool_" : type));
        long pool = poolPush();
        try (Arena arena = Arena.ofConfined()) {
            long device = send(queue, "device");
            long[] pipeline = pipeline(device, name);
            long commandBuffer = send(queue, "commandBuffer");
            long encoder = send(commandBuffer, "computeCommandEncoder");
            sendVoid(encoder, "setComputePipelineState:", pipeline[0]);
            for (int i = 0; i < buffers.length; i++) {
                sendVoid(encoder, "setBuffer:offset:atIndex:", buffers[i], offsets[i], i);
            }
            MemorySegment sizeArg = arena.allocate(Integer.BYTES);
            sizeArg.set(FFMSupport.C_INT, 0, size);
            sendVoid(encoder, "setBytes:length:atIndex:", sizeArg.address(), Integer.BYTES, buffers.length);
            long threads = (size + workPerThread - 1) / workPerThread;
            long group = Math.min(threads, pipeline[1]);
            dispatchThreads(arena, encoder, threads, group);
            sendVoid(encoder, "endEncoding");
            commitAndWait(queue, device, commandBuffer);
        } finally {
            poolPop(pool);
        }
    }

    private static long[] pipeline(long device, String name) {
        return PIPELINES.computeIfAbsent(device + ":" + name, key -> {
            long library = LIBRARIES.computeIfAbsent(device, MlxMetalKernels::loadLibrary);
            long function = send(library, "newFunctionWithName:", nsString(name));
            if (function == 0) {
                throw new TornadoRuntimeException("[ERROR] mlx.metallib has no kernel " + name);
            }
            MemorySegment error = FFMSupport.scratchPointer();
            error.set(C_POINTER, 0, MemorySegment.NULL);
            long pso = send(device, "newComputePipelineStateWithFunction:error:", function, error.address());
            if (pso == 0) {
                throw new TornadoRuntimeException("[ERROR] Cannot build a pipeline for MLX kernel " + name);
            }
            return new long[] { pso, send(pso, "maxTotalThreadsPerThreadgroup") };
        });
    }

    private static long loadLibrary(long device) {
        long url = send(objcClass("NSURL"), "fileURLWithPath:", nsString(METALLIB));
        MemorySegment error = FFMSupport.scratchPointer();
        error.set(C_POINTER, 0, MemorySegment.NULL);
        long library = send(device, "newLibraryWithURL:error:", url, error.address());
        if (library == 0) {
            throw new TornadoRuntimeException("[ERROR] Cannot load " + METALLIB);
        }
        return library;
    }

    /**
     * Commits and waits by polling a shared event the command buffer signals when its work is done,
     * as the Metal backend does, falling back to {@code waitUntilCompleted} after a short spin.
     */
    private static void commitAndWait(long queue, long device, long commandBuffer) {
        long[] queueEvent = QUEUE_EVENTS.computeIfAbsent(queue, q -> new long[] { send(device, "newSharedEvent"), 0 });
        long value;
        synchronized (queueEvent) {
            value = ++queueEvent[1];
        }
        sendVoid(commandBuffer, "encodeSignalEvent:value:", queueEvent[0], value);
        sendVoid(commandBuffer, "commit");
        long start = System.nanoTime();
        int polls = 0;
        while (send(queueEvent[0], "signaledValue") < value) {
            if (++polls % STATUS_POLL_INTERVAL == 0) {
                if (System.nanoTime() - start > SPIN_BUDGET_NS) {
                    sendVoid(commandBuffer, "waitUntilCompleted");
                    break;
                }
                if (send(commandBuffer, "status") >= MTL_COMMAND_BUFFER_STATUS_COMPLETED) {
                    break;
                }
            }
            Thread.yield();
        }
        if (send(commandBuffer, "error") != 0) {
            throw new TornadoRuntimeException("[ERROR] MLX Metal kernel failed");
        }
    }

    // ---------------------------------------------------------------- Objective-C messaging

    private static RuntimeException rethrow(Throwable t) {
        if (t instanceof RuntimeException e) {
            return e;
        }
        if (t instanceof Error e) {
            throw e;
        }
        return new TornadoRuntimeException(t instanceof Exception e ? e : new Exception(t));
    }

    private static MethodHandle downcall(FunctionDescriptor descriptor, String name) {
        return LIBOBJC == null ? null : FFMSupport.downcall(LIBOBJC, descriptor, name);
    }

    private static long sel(String name) {
        return SELECTORS.computeIfAbsent(name, key -> {
            try (Arena arena = Arena.ofConfined()) {
                return (long) SEL_REGISTER_NAME.invokeExact(FFMSupport.allocateCString(arena, key));
            } catch (Throwable t) {
                throw rethrow(t);
            }
        });
    }

    private static long objcClass(String name) {
        try (Arena arena = Arena.ofConfined()) {
            return (long) OBJC_GET_CLASS.invokeExact(FFMSupport.allocateCString(arena, name));
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    private static long nsString(String value) {
        try (Arena arena = Arena.ofConfined()) {
            return send(objcClass("NSString"), "stringWithUTF8String:", FFMSupport.allocateCString(arena, value).address());
        }
    }

    private static MethodHandle msgSend(FunctionDescriptor descriptor) {
        return SENDS.computeIfAbsent(descriptor, key -> FFMSupport.downcall(LIBOBJC, key, "objc_msgSend"));
    }

    private static long send(long receiver, String selector) {
        try {
            return (long) msgSend(FunctionDescriptor.of(C_LONG, C_LONG, C_LONG)).invokeExact(receiver, sel(selector));
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    private static long send(long receiver, String selector, long argument) {
        try {
            return (long) msgSend(FunctionDescriptor.of(C_LONG, C_LONG, C_LONG, C_LONG)).invokeExact(receiver, sel(selector), argument);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    private static long send(long receiver, String selector, long first, long second) {
        try {
            return (long) msgSend(FunctionDescriptor.of(C_LONG, C_LONG, C_LONG, C_LONG, C_LONG)).invokeExact(receiver, sel(selector), first, second);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    private static void sendVoid(long receiver, String selector) {
        send(receiver, selector);
    }

    private static void sendVoid(long receiver, String selector, long argument) {
        send(receiver, selector, argument);
    }

    private static void sendVoid(long receiver, String selector, long first, long second) {
        send(receiver, selector, first, second);
    }

    private static void sendVoid(long receiver, String selector, long first, long second, long third) {
        try {
            long ignored = (long) msgSend(FunctionDescriptor.of(C_LONG, C_LONG, C_LONG, C_LONG, C_LONG, C_LONG)).invokeExact(receiver, sel(selector), first, second, third);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** {@code dispatchThreads:threadsPerThreadgroup:}, whose two {@code MTLSize} arguments go by value. */
    private static void dispatchThreads(Arena arena, long encoder, long threads, long group) {
        MemorySegment grid = arena.allocate(MTL_SIZE);
        grid.set(C_LONG, 0, threads);
        grid.set(C_LONG, 8, 1);
        grid.set(C_LONG, 16, 1);
        MemorySegment tg = arena.allocate(MTL_SIZE);
        tg.set(C_LONG, 0, group);
        tg.set(C_LONG, 8, 1);
        tg.set(C_LONG, 16, 1);
        try {
            msgSend(FunctionDescriptor.ofVoid(C_LONG, C_LONG, MTL_SIZE, MTL_SIZE)).invokeExact(encoder, sel("dispatchThreads:threadsPerThreadgroup:"), grid, tg);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    private static long poolPush() {
        try {
            return (long) POOL_PUSH.invokeExact();
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    private static void poolPop(long pool) {
        try {
            POOL_POP.invokeExact(pool);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }
}
