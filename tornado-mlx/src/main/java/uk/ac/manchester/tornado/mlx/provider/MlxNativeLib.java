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

import static uk.ac.manchester.tornado.runtime.ffm.FFMSupport.C_INT;
import static uk.ac.manchester.tornado.runtime.ffm.FFMSupport.C_LONG;
import static uk.ac.manchester.tornado.runtime.ffm.FFMSupport.C_POINTER;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.util.concurrent.atomic.AtomicLong;

import uk.ac.manchester.tornado.api.exceptions.TornadoRuntimeException;
import uk.ac.manchester.tornado.runtime.ffm.FFMSupport;

/**
 * FFM bindings to mlx-c (the C API of Apple MLX) and the two Objective-C runtime calls the
 * provider needs.
 *
 * <p>
 * mlx-c handles ({@code mlx_array}, {@code mlx_stream}, {@code mlx_string}) are
 * {@code struct { void* ctx; }} passed by value. On the LP64 ABIs MLX supports (AArch64 and
 * x86-64 System V) a struct holding one pointer is passed and returned exactly like a pointer,
 * so each handle is modelled as an address. Restricted FFM operations go through
 * {@link FFMSupport}, which runs them inside {@code tornado.runtime}.
 * </p>
 */
final class MlxNativeLib {

    /** {@code mlx_dtype} values used by the provider (mlx/c/array.h). */
    static final int MLX_INT32 = 7;
    static final int MLX_FLOAT16 = 9;
    static final int MLX_FLOAT32 = 10;

    private static final SymbolLookup LIBMLXC = FFMSupport.loadLibrary("libmlxc.dylib", "/opt/homebrew/opt/mlx-c/lib/libmlxc.dylib", "/usr/local/opt/mlx-c/lib/libmlxc.dylib",
            "libmlxc.so");
    private static final SymbolLookup LIBOBJC = FFMSupport.loadLibrary("/usr/lib/libobjc.A.dylib");

    private static final MethodHandle VERSION;
    private static final MethodHandle STRING_NEW;
    private static final MethodHandle STRING_DATA;
    private static final MethodHandle STRING_FREE;
    private static final MethodHandle GPU_STREAM_NEW;
    private static final MethodHandle STREAM_FREE;
    private static final MethodHandle ARRAY_NEW;
    private static final MethodHandle ARRAY_NEW_DATA_MANAGED;
    private static final MethodHandle ARRAY_FREE;
    private static final MethodHandle ARRAY_EVAL;
    private static final MethodHandle ARRAY_DATA_UINT8;
    private static final MethodHandle ARRAY_NBYTES;
    private static final MethodHandle ADD;
    private static final MethodHandle OBJC_RETAIN;
    private static final MethodHandle OBJC_RELEASE;

    /**
     * Deleter handed to {@code mlx_array_new_data_managed}. MLX calls it when it no longer needs
     * the memory: right away if it had to copy the data, otherwise when the array is freed. The
     * count therefore tells the provider whether a wrap adopted the memory or copied it.
     */
    private static final AtomicLong RELEASES = new AtomicLong();
    private static final MemorySegment COUNTING_DELETER;

    static {
        if (LIBMLXC == null || LIBOBJC == null) {
            VERSION = STRING_NEW = STRING_DATA = STRING_FREE = GPU_STREAM_NEW = STREAM_FREE = null;
            ARRAY_NEW = ARRAY_NEW_DATA_MANAGED = ARRAY_FREE = ARRAY_EVAL = ARRAY_DATA_UINT8 = ARRAY_NBYTES = ADD = null;
            OBJC_RETAIN = OBJC_RELEASE = null;
            COUNTING_DELETER = null;
        } else {
            VERSION = FFMSupport.downcall(LIBMLXC, FunctionDescriptor.of(C_INT, C_POINTER), "mlx_version");
            STRING_NEW = FFMSupport.downcall(LIBMLXC, FunctionDescriptor.of(C_POINTER), "mlx_string_new");
            STRING_DATA = FFMSupport.downcall(LIBMLXC, FunctionDescriptor.of(C_POINTER, C_POINTER), "mlx_string_data");
            STRING_FREE = FFMSupport.downcall(LIBMLXC, FunctionDescriptor.of(C_INT, C_POINTER), "mlx_string_free");
            GPU_STREAM_NEW = FFMSupport.downcall(LIBMLXC, FunctionDescriptor.of(C_POINTER), "mlx_default_gpu_stream_new");
            STREAM_FREE = FFMSupport.downcall(LIBMLXC, FunctionDescriptor.of(C_INT, C_POINTER), "mlx_stream_free");
            ARRAY_NEW = FFMSupport.downcall(LIBMLXC, FunctionDescriptor.of(C_POINTER), "mlx_array_new");
            ARRAY_NEW_DATA_MANAGED = FFMSupport.downcall(LIBMLXC, FunctionDescriptor.of(C_POINTER, C_POINTER, C_POINTER, C_INT, C_INT, C_POINTER), "mlx_array_new_data_managed");
            ARRAY_FREE = FFMSupport.downcall(LIBMLXC, FunctionDescriptor.of(C_INT, C_POINTER), "mlx_array_free");
            ARRAY_EVAL = FFMSupport.downcall(LIBMLXC, FunctionDescriptor.of(C_INT, C_POINTER), "mlx_array_eval");
            ARRAY_DATA_UINT8 = FFMSupport.downcall(LIBMLXC, FunctionDescriptor.of(C_POINTER, C_POINTER), "mlx_array_data_uint8");
            ARRAY_NBYTES = FFMSupport.downcall(LIBMLXC, FunctionDescriptor.of(C_LONG, C_POINTER), "mlx_array_nbytes");
            ADD = FFMSupport.downcall(LIBMLXC, FunctionDescriptor.of(C_INT, C_POINTER, C_POINTER, C_POINTER, C_POINTER), "mlx_add");
            OBJC_RETAIN = FFMSupport.downcall(LIBOBJC, FunctionDescriptor.of(C_LONG, C_LONG), "objc_retain");
            OBJC_RELEASE = FFMSupport.downcall(LIBOBJC, FunctionDescriptor.ofVoid(C_LONG), "objc_release");
            try {
                MethodHandle target = MethodHandles.lookup().findStatic(MlxNativeLib.class, "onRelease", MethodType.methodType(void.class, MemorySegment.class));
                COUNTING_DELETER = FFMSupport.upcallStub(target, FunctionDescriptor.ofVoid(C_POINTER), Arena.global());
            } catch (ReflectiveOperationException e) {
                throw new ExceptionInInitializerError(e);
            }
        }
    }

    private MlxNativeLib() {
    }

    private static void onRelease(MemorySegment data) {
        RELEASES.incrementAndGet();
    }

    /** Throws if libmlxc (or the Objective-C runtime) could not be loaded. */
    static void load() {
        if (ADD == null) {
            throw new TornadoRuntimeException("[ERROR] Unable to load mlx-c. Install it (e.g. `brew install mlx-c`) and make sure libmlxc is on the library path.");
        }
    }

    private static RuntimeException rethrow(Throwable t) {
        if (t instanceof RuntimeException r) {
            return r;
        }
        if (t instanceof Error e) {
            throw e;
        }
        return new TornadoRuntimeException((Exception) t);
    }

    private static void check(int status, String call) {
        if (status != 0) {
            throw new TornadoRuntimeException("[ERROR] " + call + " failed with status " + status);
        }
    }

    static String version() {
        load();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment slot = arena.allocate(C_POINTER);
            slot.set(C_POINTER, 0, (MemorySegment) STRING_NEW.invokeExact());
            check((int) VERSION.invokeExact(slot), "mlx_version");
            MemorySegment str = slot.get(C_POINTER, 0);
            String version = FFMSupport.readCString((MemorySegment) STRING_DATA.invokeExact(str));
            int ignored = (int) STRING_FREE.invokeExact(str);
            return version;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    static MemorySegment gpuStream() {
        try {
            return (MemorySegment) GPU_STREAM_NEW.invokeExact();
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    static void freeStream(MemorySegment stream) {
        try {
            int ignored = (int) STREAM_FREE.invokeExact(stream);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Number of times MLX has released memory it was handed; see {@link #COUNTING_DELETER}. */
    static long releases() {
        return RELEASES.get();
    }

    /** Wraps {@code shape} elements at {@code address} as an MLX array without asking MLX to copy them. */
    static MemorySegment wrap(long address, int[] shape, int dtype) {
        try (Arena arena = Arena.ofConfined()) {
            // allocate + copy rather than allocateArray/allocateFrom, whose names differ between
            // the JDK 21 preview FFM API and JDK 22+.
            MemorySegment shapeSegment = FFMSupport.allocateArray(arena, C_INT, shape.length);
            MemorySegment.copy(shape, 0, shapeSegment, C_INT, 0, shape.length);
            return (MemorySegment) ARRAY_NEW_DATA_MANAGED.invokeExact(MemorySegment.ofAddress(address), shapeSegment, shape.length, dtype, COUNTING_DELETER);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    static MemorySegment add(MemorySegment a, MemorySegment b, MemorySegment stream) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment result = arena.allocate(C_POINTER);
            result.set(C_POINTER, 0, (MemorySegment) ARRAY_NEW.invokeExact());
            check((int) ADD.invokeExact(result, a, b, stream), "mlx_add");
            return result.get(C_POINTER, 0);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    static void eval(MemorySegment array) {
        try {
            check((int) ARRAY_EVAL.invokeExact(array), "mlx_array_eval");
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Address of an evaluated array's first element, whatever its dtype. */
    static long dataAddress(MemorySegment array) {
        try {
            return ((MemorySegment) ARRAY_DATA_UINT8.invokeExact(array)).address();
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    static long nbytes(MemorySegment array) {
        try {
            return (long) ARRAY_NBYTES.invokeExact(array);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    static void free(MemorySegment array) {
        try {
            int ignored = (int) ARRAY_FREE.invokeExact(array);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    static void retain(long object) {
        try {
            long ignored = (long) OBJC_RETAIN.invokeExact(object);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    static void release(long object) {
        try {
            OBJC_RELEASE.invokeExact(object);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }
}
