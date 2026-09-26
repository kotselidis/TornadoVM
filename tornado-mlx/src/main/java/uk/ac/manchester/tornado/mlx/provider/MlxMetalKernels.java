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

    /** MLX's type suffix for a kernel name, or null for a type MLX's kernels do not take. */
    static String typeName(int dtype) {
        return switch (dtype) {
            case MlxNativeLib.MLX_BOOL -> "bool_";
            case MlxNativeLib.MLX_UINT8 -> "uint8";
            case MlxNativeLib.MLX_UINT16 -> "uint16";
            case MlxNativeLib.MLX_UINT32 -> "uint32";
            case MlxNativeLib.MLX_INT8 -> "int8";
            case MlxNativeLib.MLX_INT16 -> "int16";
            case MlxNativeLib.MLX_INT32 -> "int32";
            case MlxNativeLib.MLX_INT64 -> "int64";
            case MlxNativeLib.MLX_FLOAT16 -> "float16";
            case MlxNativeLib.MLX_FLOAT32 -> "float32";
            case MlxNativeLib.MLX_BFLOAT16 -> "bfloat16";
            case MlxNativeLib.MLX_COMPLEX64 -> "complex64";
            default -> null;
        };
    }

    static boolean isFloating(int dtype) {
        return dtype == MlxNativeLib.MLX_FLOAT32 || dtype == MlxNativeLib.MLX_FLOAT16 || dtype == MlxNativeLib.MLX_BFLOAT16;
    }

    static int itemSize(int dtype) {
        return switch (dtype) {
            case MlxNativeLib.MLX_BOOL, MlxNativeLib.MLX_UINT8, MlxNativeLib.MLX_INT8 -> 1;
            case MlxNativeLib.MLX_UINT16, MlxNativeLib.MLX_INT16, MlxNativeLib.MLX_FLOAT16, MlxNativeLib.MLX_BFLOAT16 -> 2;
            case MlxNativeLib.MLX_INT64, MlxNativeLib.MLX_COMPLEX64 -> 8;
            default -> 4;
        };
    }

    /** MLX's work per thread for contiguous element-wise kernels ({@code get_work_per_thread}). */
    private static int workPerThread(int dtype, long size) {
        return size < WORK_PER_THREAD_THRESHOLD ? 1 : Math.max(1, 8 / itemSize(dtype));
    }

    /** A kernel argument: a buffer with the byte offset of its first element. */
    record Ref(long buffer, long offset) {
    }

    /**
     * A sequence of MLX kernels encoded into one command buffer on TornadoVM's queue and waited on
     * once. The compute encoder dispatches serially, so each kernel sees the previous one's writes,
     * as in MLX's own command buffers. Scratch buffers come from a per-device pool and are reused.
     */
    static final class Program implements AutoCloseable {
        private final long queue;
        private final long device;
        private final long pool;
        private final Arena arena = Arena.ofConfined();
        private final long commandBuffer;
        private final long encoder;
        private int scratchUsed;

        Program(long queue) {
            this.queue = queue;
            this.pool = poolPush();
            this.device = send(queue, "device");
            this.commandBuffer = send(queue, "commandBuffer");
            this.encoder = send(commandBuffer, "computeCommandEncoder");
        }

        /** A scratch buffer of at least {@code bytes}, valid until this program is closed. */
        Ref scratch(long bytes) {
            return new Ref(scratchBuffer(device, scratchUsed++, bytes), 0);
        }

        /** {@code out = op(in)}: MLX's contiguous unary kernel. */
        void unary(String op, int inType, int outType, int size, Ref in, Ref out) {
            int wpt = workPerThread(inType, size);
            dispatch((wpt > 1 ? "vn_" : "v_") + op + typeName(inType) + typeName(outType), size, wpt, new Ref[] { in, out }, null);
        }

        /** {@code out = static_cast<outType>(in)}: MLX's contiguous copy kernel. */
        void copy(int inType, int outType, int size, Ref in, Ref out) {
            int wpt = workPerThread(outType, size);
            dispatch((wpt > 1 ? "vn_copy" : "v_copy") + typeName(inType) + typeName(outType), size, wpt, new Ref[] { in, out }, null);
        }

        /** {@code out[:] = value}: MLX's scalar-fill copy kernel. */
        void fill(int type, int size, double value, Ref out) {
            int wpt = workPerThread(type, size);
            dispatch((wpt > 1 ? "sn_copy" : "s_copy") + typeName(type) + typeName(type), size, wpt, new Ref[] { null, out }, scalarBytes(type, value));
        }

        /** {@code out = op(a, b)}, both vectors: MLX's {@code vv} binary kernel. */
        void binary(String op, int inType, int size, Ref a, Ref b, Ref out) {
            int wpt = workPerThread(inType, size);
            dispatch((wpt > 1 ? "vvn_" : "vv_") + op + typeName(inType), size, wpt, new Ref[] { a, b, out }, null);
        }

        /** {@code out = op(a, scalar)}: MLX's {@code vs} binary kernel. */
        void binaryScalarRight(String op, int inType, int size, Ref a, double scalar, Ref out) {
            int wpt = workPerThread(inType, size);
            dispatch((wpt > 1 ? "vsn_" : "vs_") + op + typeName(inType), size, wpt, new Ref[] { a, null, out }, scalarBytes(inType, scalar));
        }

        /** {@code out = op(scalar, b)}: MLX's {@code sv} binary kernel. */
        void binaryScalarLeft(String op, int inType, int size, double scalar, Ref b, Ref out) {
            int wpt = workPerThread(inType, size);
            dispatch((wpt > 1 ? "svn_" : "sv_") + op + typeName(inType), size, wpt, new Ref[] { null, b, out }, scalarBytes(inType, scalar));
        }

        /** {@code out1, out2 = op(a, b)}: MLX's two-output binary kernel (divmod). */
        void binaryTwo(String op, int inType, int size, Ref a, Ref b, Ref out1, Ref out2) {
            int wpt = workPerThread(inType, size);
            dispatch((wpt > 1 ? "vvn_" : "vv_") + op + typeName(inType), size, wpt, new Ref[] { a, b, out1, out2 }, null);
        }

        /** {@code out = condition ? x : y}, with MLX bools as the condition: MLX's {@code v} Select kernel. */
        void select(int type, int size, Ref condition, Ref x, Ref y, Ref out) {
            int wpt = workPerThread(type, size);
            dispatch((wpt > 1 ? "vn_Select" : "v_Select") + typeName(type), size, wpt, new Ref[] { condition, x, y, out }, null);
        }

        /** {@code out = condition ? scalar : y}: MLX's {@code sv} Select kernel. */
        void selectScalar(int type, int size, Ref condition, double scalar, Ref y, Ref out) {
            int wpt = workPerThread(type, size);
            dispatch((wpt > 1 ? "svn_Select" : "sv_Select") + typeName(type), size, wpt, new Ref[] { condition, null, y, out }, scalarBytes(type, scalar));
        }

        /**
         * Encodes one element-wise dispatch: {@code refs} bound at indices 0.., where a null entry is
         * the scalar operand {@code scalar} (bound with {@code setBytes}), and the element count after
         * them.
         */
        private void dispatch(String name, long size, int wpt, Ref[] refs, byte[] scalar) {
            long[] pipeline = pipeline(device, name);
            sendVoid(encoder, "setComputePipelineState:", pipeline[0]);
            for (int i = 0; i < refs.length; i++) {
                if (refs[i] == null) {
                    MemorySegment bytes = arena.allocate(scalar.length);
                    MemorySegment.copy(scalar, 0, bytes, FFMSupport.C_CHAR, 0, scalar.length);
                    sendVoid(encoder, "setBytes:length:atIndex:", bytes.address(), scalar.length, i);
                } else {
                    sendVoid(encoder, "setBuffer:offset:atIndex:", refs[i].buffer(), refs[i].offset(), i);
                }
            }
            MemorySegment sizeArg = arena.allocate(Integer.BYTES);
            sizeArg.set(FFMSupport.C_INT, 0, (int) size);
            sendVoid(encoder, "setBytes:length:atIndex:", sizeArg.address(), Integer.BYTES, refs.length);
            long threads = (size + wpt - 1) / wpt;
            dispatchThreads(arena, encoder, threads, Math.min(threads, pipeline[1]));
        }

        /** Ends encoding, commits, and waits for the GPU to finish. */
        @Override
        public void close() {
            try {
                sendVoid(encoder, "endEncoding");
                commitAndWait(queue, device, commandBuffer);
            } finally {
                arena.close();
                poolPop(pool);
            }
        }
    }

    /** A scalar in MLX's representation of {@code dtype}, as {@code array(value, dtype)} would hold it. */
    static byte[] scalarBytes(int dtype, double value) {
        java.nio.ByteBuffer b = java.nio.ByteBuffer.allocate(8).order(java.nio.ByteOrder.LITTLE_ENDIAN);
        switch (dtype) {
            case MlxNativeLib.MLX_FLOAT32 -> b.putFloat((float) value);
            case MlxNativeLib.MLX_FLOAT16 -> b.putShort(Float.floatToFloat16((float) value));
            case MlxNativeLib.MLX_BFLOAT16 -> b.putShort(floatToBFloat16((float) value));
            case MlxNativeLib.MLX_INT32, MlxNativeLib.MLX_UINT32 -> b.putInt((int) value);
            case MlxNativeLib.MLX_INT64 -> b.putLong((long) value);
            case MlxNativeLib.MLX_INT16, MlxNativeLib.MLX_UINT16 -> b.putShort((short) value);
            case MlxNativeLib.MLX_INT8, MlxNativeLib.MLX_UINT8, MlxNativeLib.MLX_BOOL -> b.put((byte) value);
            default -> throw new TornadoRuntimeException("[ERROR] No scalar form for MLX dtype " + dtype);
        }
        byte[] out = new byte[itemSize(dtype)];
        System.arraycopy(b.array(), 0, out, 0, out.length);
        return out;
    }

    /** Round-to-nearest-even float to bfloat16, as MLX's {@code bfloat16_t} conversion does. */
    private static short floatToBFloat16(float value) {
        int bits = Float.floatToRawIntBits(value);
        if (Float.isNaN(value)) {
            return (short) ((bits >>> 16) | 0x40);
        }
        int rounding = 0x7fff + ((bits >>> 16) & 1);
        return (short) ((bits + rounding) >>> 16);
    }

    /** Scratch buffers by device and slot, grown on demand and kept for reuse. */
    private static final Map<Long, long[][]> SCRATCH = new ConcurrentHashMap<>();
    private static final int SCRATCH_SLOTS = 8;

    private static synchronized long scratchBuffer(long device, int slot, long bytes) {
        if (slot >= SCRATCH_SLOTS) {
            throw new TornadoRuntimeException("[ERROR] Too many MLX scratch buffers in one program");
        }
        long[][] slots = SCRATCH.computeIfAbsent(device, d -> new long[SCRATCH_SLOTS][2]);
        long[] entry = slots[slot];
        if (entry[1] < bytes) {
            if (entry[0] != 0) {
                sendVoid(entry[0], "release");
            }
            long capacity = Math.max(bytes, 4096);
            entry[0] = send(device, "newBufferWithLength:options:", capacity, 0);
            entry[1] = capacity;
        }
        return entry[0];
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
