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
import static uk.ac.manchester.tornado.runtime.ffm.FFMSupport.C_POINTER;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;
import java.util.function.ToIntFunction;
import java.util.stream.IntStream;

import uk.ac.manchester.tornado.api.exceptions.TornadoRuntimeException;
import uk.ac.manchester.tornado.api.types.arrays.BFloat16Array;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.DoubleArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.Int8Array;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.api.types.arrays.LongArray;
import uk.ac.manchester.tornado.api.types.arrays.ShortArray;
import uk.ac.manchester.tornado.api.types.arrays.TornadoNativeArray;
import uk.ac.manchester.tornado.runtime.ffm.FFMSupport;
import uk.ac.manchester.tornado.runtime.library.spi.LibraryInvocation;

/**
 * One MLX library-task call: reads the task's arguments, wraps TornadoVM buffers as MLX arrays,
 * runs mlx-c operations on the call's stream, and writes results back into output buffers. MLX
 * arrays created during the call are freed when it closes; wrappers of TornadoVM buffers are owned
 * by the execution plan's context and reused across calls.
 */
final class MlxCall implements AutoCloseable {

    /** Results at least this large are copied out by several threads. */
    private static final long PARALLEL_COPY_BYTES = 8L << 20;
    private static final int COPY_THREADS = Math.min(8, Runtime.getRuntime().availableProcessors());

    private final MlxLibraryProvider.MlxContext context;
    private final LibraryInvocation invocation;
    private final MemorySegment stream;
    private final Arena arena = Arena.ofConfined();
    private final List<MemorySegment> temporaries = new ArrayList<>();
    private final List<MemorySegment> vectors = new ArrayList<>();

    MlxCall(MlxLibraryProvider.MlxContext context, LibraryInvocation invocation, MemorySegment stream) {
        this.context = context;
        this.invocation = invocation;
        this.stream = stream;
    }

    /** The stream every operation of this call runs on. */
    MemorySegment stream() {
        return stream;
    }

    int intArg(int index) {
        return (Integer) invocation.getArg(index);
    }

    float floatArg(int index) {
        return ((Number) invocation.getArg(index)).floatValue();
    }

    boolean boolArg(int index) {
        return (Boolean) invocation.getArg(index);
    }

    /** Number of elements of array argument {@code index}. */
    int length(int index) {
        return array(index).getSize();
    }

    private TornadoNativeArray array(int index) {
        if (!(invocation.getArg(index) instanceof TornadoNativeArray array)) {
            throw new TornadoRuntimeException("[ERROR] MLX argument " + index + " is not a TornadoVM native array");
        }
        return array;
    }

    /** MLX dtype of array argument {@code index}. */
    int dtype(int index) {
        return dtypeOf(invocation.getArg(index));
    }

    static int dtypeOf(Object array) {
        return switch (array) {
            case FloatArray ignored -> MlxNativeLib.MLX_FLOAT32;
            case HalfFloatArray ignored -> MlxNativeLib.MLX_FLOAT16;
            case BFloat16Array ignored -> MlxNativeLib.MLX_BFLOAT16;
            case DoubleArray ignored -> MlxNativeLib.MLX_FLOAT64;
            case IntArray ignored -> MlxNativeLib.MLX_INT32;
            case LongArray ignored -> MlxNativeLib.MLX_INT64;
            case ShortArray ignored -> MlxNativeLib.MLX_INT16;
            case Int8Array ignored -> MlxNativeLib.MLX_INT8;
            case ByteArray ignored -> MlxNativeLib.MLX_UINT8;
            default -> throw new TornadoRuntimeException("[ERROR] MLX does not support arguments of type " + array.getClass().getSimpleName());
        };
    }

    /**
     * Array argument {@code index} as an MLX array of the given shape, in its own dtype. The shape
     * must cover the whole array.
     */
    MemorySegment input(int index, int... shape) {
        return inputAs(index, dtype(index), shape);
    }

    /** Array argument {@code index} reinterpreted as {@code dtype} (e.g. packed weights as uint32). */
    MemorySegment inputAs(int index, int dtype, int... shape) {
        long elements = 1;
        for (int d : shape) {
            elements *= d;
        }
        TornadoNativeArray array = array(index);
        long bytes = (long) array.getSize() * array.getElementSize();
        if (elements * elementBytes(dtype) != bytes) {
            throw new TornadoRuntimeException("[ERROR] MLX argument " + index + " holds " + bytes + " bytes, but shape " + java.util.Arrays.toString(shape) + " needs "
                    + elements * elementBytes(dtype));
        }
        return context.wrapper(invocation.getDevicePointer(index), invocation.getNativeBuffer(index), shape, dtype);
    }

    static int elementBytes(int dtype) {
        return switch (dtype) {
            case MlxNativeLib.MLX_BOOL, MlxNativeLib.MLX_UINT8, MlxNativeLib.MLX_INT8 -> 1;
            case MlxNativeLib.MLX_UINT16, MlxNativeLib.MLX_INT16, MlxNativeLib.MLX_FLOAT16, MlxNativeLib.MLX_BFLOAT16 -> 2;
            case MlxNativeLib.MLX_UINT32, MlxNativeLib.MLX_INT32, MlxNativeLib.MLX_FLOAT32 -> 4;
            default -> 8;
        };
    }

    /**
     * Runs an mlx-c operation that writes its result through an {@code mlx_array*}. The result is
     * freed when the call closes.
     */
    MemorySegment op(String name, ToIntFunction<MemorySegment> operation) {
        MemorySegment slot = arena.allocate(C_POINTER);
        slot.set(C_POINTER, 0, MlxC.mlx_array_new());
        int status = operation.applyAsInt(slot);
        MemorySegment result = slot.get(C_POINTER, 0);
        temporaries.add(result);
        MlxNativeLib.check(status, name);
        return result;
    }

    /** An mlx-c operation with two {@code mlx_array*} results. */
    interface PairOperation {
        int apply(MemorySegment res0, MemorySegment res1);
    }

    /** Runs an operation that returns two arrays through separate pointers; both are freed when the call closes. */
    MemorySegment[] pair(String name, PairOperation operation) {
        MemorySegment slot0 = arena.allocate(C_POINTER);
        MemorySegment slot1 = arena.allocate(C_POINTER);
        slot0.set(C_POINTER, 0, MlxC.mlx_array_new());
        slot1.set(C_POINTER, 0, MlxC.mlx_array_new());
        int status = operation.apply(slot0, slot1);
        MemorySegment r0 = slot0.get(C_POINTER, 0);
        MemorySegment r1 = slot1.get(C_POINTER, 0);
        temporaries.add(r0);
        temporaries.add(r1);
        MlxNativeLib.check(status, name);
        return new MemorySegment[] { r0, r1 };
    }

    /** An MLX float32 scalar, freed when the call closes. */
    MemorySegment scalar(float value) {
        MemorySegment s = MlxC.mlx_array_new_float32(value);
        temporaries.add(s);
        return s;
    }

    /** An {@code mlx_vector_array} of the given arrays, freed when the call closes. */
    MemorySegment vector(MemorySegment... arrays) {
        MemorySegment vec = MlxC.mlx_vector_array_new();
        vectors.add(vec);
        for (MemorySegment a : arrays) {
            MlxNativeLib.check(MlxC.mlx_vector_array_append_value(vec, a), "mlx_vector_array_append_value");
        }
        return vec;
    }

    /** An MLX int32 scalar, freed when the call closes. */
    MemorySegment scalar(int value) {
        MemorySegment s = MlxC.mlx_array_new_int(value);
        temporaries.add(s);
        return s;
    }

    /** Scratch memory that lives until the call closes. */
    Arena arena() {
        return arena;
    }

    /** An {@code mlx_optional_int} holding {@code value}. */
    MemorySegment optionalInt(int value) {
        MemorySegment opt = arena.allocate(MlxC.OPT_INT);
        opt.set(C_INT, 0, value);
        opt.set(ValueLayout.JAVA_BOOLEAN, 4, true);
        return opt;
    }

    /** An {@code mlx_optional_float} holding {@code value}. */
    MemorySegment optionalFloat(float value) {
        MemorySegment opt = arena.allocate(MlxC.OPT_FLOAT);
        opt.set(ValueLayout.JAVA_FLOAT, 0, value);
        opt.set(ValueLayout.JAVA_BOOLEAN, 4, true);
        return opt;
    }

    /** An empty {@code mlx_optional_dtype}. */
    MemorySegment noDtype() {
        return arena.allocate(MlxC.OPT_DTYPE);
    }

    /** A NUL-terminated C string. */
    MemorySegment cString(String value) {
        return FFMSupport.allocateCString(arena, value);
    }

    /** A C {@code int[]}. */
    MemorySegment ints(int... values) {
        MemorySegment segment = FFMSupport.allocateArray(arena, C_INT, Math.max(values.length, 1));
        MemorySegment.copy(values, 0, segment, C_INT, 0, values.length);
        return segment;
    }

    /** The null {@code mlx_array}, for optional array arguments. */
    static MemorySegment none() {
        return MemorySegment.NULL;
    }

    /** {@code a} viewed with a new shape (no copy); freed when the call closes. */
    MemorySegment reshape(MemorySegment a, int... shape) {
        return op("mlx_reshape", res -> MlxC.mlx_reshape(res, a, ints(shape), shape.length, stream));
    }

    /**
     * Runs an mlx-c operation that returns several arrays through an {@code mlx_vector_array*} and
     * returns them; they are freed when the call closes.
     */
    MemorySegment[] vectorOp(String name, int count, ToIntFunction<MemorySegment> operation) {
        MemorySegment slot = arena.allocate(C_POINTER);
        slot.set(C_POINTER, 0, MlxC.mlx_vector_array_new());
        int status = operation.applyAsInt(slot);
        MemorySegment vector = slot.get(C_POINTER, 0);
        try {
            MlxNativeLib.check(status, name);
            if (MlxC.mlx_vector_array_size(vector) != count) {
                throw new TornadoRuntimeException("[ERROR] " + name + " returned " + MlxC.mlx_vector_array_size(vector) + " arrays, expected " + count);
            }
            MemorySegment[] results = new MemorySegment[count];
            for (int i = 0; i < count; i++) {
                MemorySegment item = arena.allocate(C_POINTER);
                item.set(C_POINTER, 0, MlxC.mlx_array_new());
                int itemStatus = MlxC.mlx_vector_array_get(item, vector, i);
                results[i] = item.get(C_POINTER, 0);
                temporaries.add(results[i]);
                MlxNativeLib.check(itemStatus, "mlx_vector_array_get");
            }
            return results;
        } finally {
            MlxC.mlx_vector_array_free(vector);
        }
    }

    /**
     * Evaluates {@code result} and copies it into array argument {@code index}, converting to the
     * output's dtype first if MLX produced another one.
     */
    void store(MemorySegment result, int index) {
        int outDtype = dtype(index);
        MemorySegment converted = result;
        if (MlxC.mlx_array_dtype(result) != outDtype) {
            converted = op("mlx_astype", slot -> MlxC.mlx_astype(slot, result, outDtype, stream));
        }
        MemorySegment value = contiguous(converted);
        MlxNativeLib.eval(value);
        long bytes = MlxC.mlx_array_nbytes(value);
        TornadoNativeArray out = array(index);
        long expected = (long) out.getSize() * out.getElementSize();
        if (bytes != expected) {
            throw new TornadoRuntimeException("[ERROR] MLX result is " + bytes + " bytes, output argument " + index + " holds " + expected);
        }
        copyOut(MlxNativeLib.dataAddress(value), invocation.getDevicePointer(index), bytes);
    }

    /**
     * {@code a} laid out row-major and dense. Some operations return strided views (top-k slices
     * a partition, for example), whose bytes cannot be copied out as they are; for an array that
     * is already dense this is a no-op.
     */
    private MemorySegment contiguous(MemorySegment a) {
        return op("mlx_contiguous", slot -> MlxC.mlx_contiguous(slot, a, false, stream));
    }

    /**
     * Evaluates {@code result} and copies its bytes unchanged into array argument {@code index},
     * for data whose bit pattern matters more than its dtype (packed quantized weights).
     */
    void storeRaw(MemorySegment packed, int index) {
        MemorySegment result = contiguous(packed);
        MlxNativeLib.eval(result);
        long bytes = MlxC.mlx_array_nbytes(result);
        TornadoNativeArray out = array(index);
        long expected = (long) out.getSize() * out.getElementSize();
        if (bytes != expected) {
            throw new TornadoRuntimeException("[ERROR] MLX result is " + bytes + " bytes, output argument " + index + " holds " + expected);
        }
        copyOut(MlxNativeLib.dataAddress(result), invocation.getDevicePointer(index), bytes);
    }

    /**
     * Copies an evaluated MLX result into a TornadoVM buffer. MLX cannot write an operation's result
     * into memory it does not own: its kernels always allocate their output, and the only way to
     * reach an existing buffer, a slice update that reuses ("donates") a fresh wrapper of it, is
     * slower than this copy, because Metal maps a newly wrapped buffer for the GPU on first use.
     * Large results are copied by several threads, which roughly halves the time on Apple silicon
     * (64 MB: 1.6 ms on one thread, 1.0 ms on eight).
     */
    private static void copyOut(long source, long destination, long bytes) {
        MemorySegment src = FFMSupport.asSegment(source, bytes);
        MemorySegment dst = FFMSupport.asSegment(destination, bytes);
        if (bytes < PARALLEL_COPY_BYTES) {
            dst.copyFrom(src);
            return;
        }
        long chunk = (bytes + COPY_THREADS - 1) / COPY_THREADS;
        IntStream.range(0, COPY_THREADS).parallel().forEach(k -> {
            long offset = k * chunk;
            long length = Math.min(chunk, bytes - offset);
            if (length > 0) {
                MemorySegment.copy(src, offset, dst, offset, length);
            }
        });
    }

    @Override
    public void close() {
        for (MemorySegment t : temporaries) {
            MlxNativeLib.free(t);
        }
        temporaries.clear();
        vectors.forEach(MlxC::mlx_vector_array_free);
        vectors.clear();
        arena.close();
    }
}
