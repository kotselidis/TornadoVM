package uk.ac.manchester.tornado.mlx.spike;

import static uk.ac.manchester.tornado.runtime.ffm.FFMSupport.C_FLOAT;
import static uk.ac.manchester.tornado.runtime.ffm.FFMSupport.C_INT;
import static uk.ac.manchester.tornado.runtime.ffm.FFMSupport.C_LONG;
import static uk.ac.manchester.tornado.runtime.ffm.FFMSupport.C_POINTER;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.StructLayout;
import java.lang.foreign.ValueLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.util.concurrent.atomic.AtomicInteger;

import uk.ac.manchester.tornado.runtime.ffm.FFMSupport;

/**
 * Hand-written FFM bindings to the handful of mlx-c functions the M0 spike needs.
 *
 * <p>
 * mlx-c handles (mlx_array, mlx_stream, mlx_string) are {@code struct { void* ctx; }} passed by
 * value. On AArch64 a one-pointer struct is passed and returned in a general register exactly
 * like a pointer, so the spike models each handle as an address. M2's generated bindings should
 * use proper struct layouts.
 */
final class MlxC {

    static final int MLX_UINT32 = 3;
    static final int MLX_FLOAT16 = 9;
    static final int MLX_FLOAT32 = 10;

    private static final SymbolLookup LIB = load();

    /** {@code mlx_optional_int} / {@code mlx_optional_float}: {value; bool has_value}, 8 bytes, by value. */
    static final StructLayout OPT_INT = MemoryLayout.structLayout(C_INT, ValueLayout.JAVA_BOOLEAN, MemoryLayout.paddingLayout(3));
    static final StructLayout OPT_FLOAT = MemoryLayout.structLayout(C_FLOAT, ValueLayout.JAVA_BOOLEAN, MemoryLayout.paddingLayout(3));

    private static final MethodHandle VERSION = h("mlx_version", C_INT, C_POINTER);
    private static final MethodHandle STRING_NEW = h("mlx_string_new", C_POINTER);
    private static final MethodHandle STRING_DATA = h("mlx_string_data", C_POINTER, C_POINTER);
    private static final MethodHandle STRING_FREE = h("mlx_string_free", C_INT, C_POINTER);
    private static final MethodHandle GPU_STREAM_NEW = h("mlx_default_gpu_stream_new", C_POINTER);
    private static final MethodHandle SYNCHRONIZE = h("mlx_synchronize", C_INT, C_POINTER);
    private static final MethodHandle ARRAY_NEW = h("mlx_array_new", C_POINTER);
    private static final MethodHandle ARRAY_NEW_F32 = h("mlx_array_new_float32", C_POINTER, C_FLOAT);
    private static final MethodHandle ARRAY_NEW_DATA_MANAGED = h("mlx_array_new_data_managed", C_POINTER, C_POINTER, C_POINTER, C_INT, C_INT, C_POINTER);
    private static final MethodHandle ARRAY_NEW_DATA = h("mlx_array_new_data", C_POINTER, C_POINTER, C_POINTER, C_INT, C_INT);
    private static final MethodHandle ARRAY_FREE = h("mlx_array_free", C_INT, C_POINTER);
    private static final MethodHandle ARRAY_EVAL = h("mlx_array_eval", C_INT, C_POINTER);
    private static final MethodHandle ARRAY_DATA_F32 = h("mlx_array_data_float32", C_POINTER, C_POINTER);
    private static final MethodHandle ARRAY_NBYTES = h("mlx_array_nbytes", C_LONG, C_POINTER);
    private static final MethodHandle AS_STRIDED = h("mlx_as_strided", C_INT, C_POINTER, C_POINTER, C_POINTER, C_LONG, C_POINTER, C_LONG, C_LONG, C_POINTER);
    private static final MethodHandle MULTIPLY = h("mlx_multiply", C_INT, C_POINTER, C_POINTER, C_POINTER, C_POINTER);
    private static final MethodHandle ADD = h("mlx_add", C_INT, C_POINTER, C_POINTER, C_POINTER, C_POINTER);
    private static final MethodHandle RMS_NORM = h("mlx_fast_rms_norm", C_INT, C_POINTER, C_POINTER, C_POINTER, C_FLOAT, C_POINTER);
    private static final MethodHandle ARRAY_DATA_U8 = h("mlx_array_data_uint8", C_POINTER, C_POINTER);
    private static final MethodHandle QUANTIZE = h("mlx_quantize", C_INT, C_POINTER, C_POINTER, OPT_INT, OPT_INT, C_POINTER, C_POINTER, C_POINTER);
    private static final MethodHandle QUANTIZED_MATMUL = h("mlx_quantized_matmul", C_INT, C_POINTER, C_POINTER, C_POINTER, C_POINTER, C_POINTER, ValueLayout.JAVA_BOOLEAN, OPT_INT, OPT_INT,
            C_POINTER, C_POINTER);
    private static final MethodHandle ROPE = h("mlx_fast_rope", C_INT, C_POINTER, C_POINTER, C_INT, ValueLayout.JAVA_BOOLEAN, OPT_FLOAT, C_FLOAT, C_INT, C_POINTER, C_POINTER);
    private static final MethodHandle VECTOR_ARRAY_NEW = h("mlx_vector_array_new", C_POINTER);
    private static final MethodHandle VECTOR_ARRAY_GET = h("mlx_vector_array_get", C_INT, C_POINTER, C_POINTER, C_LONG);
    private static final MethodHandle VECTOR_ARRAY_FREE = h("mlx_vector_array_free", C_INT, C_POINTER);
    private static final MethodHandle MATMUL = h("mlx_matmul", C_INT, C_POINTER, C_POINTER, C_POINTER, C_POINTER);

    /** Counts deleter calls: a call during wrapping means MLX copied instead of adopting the memory. */
    static final AtomicInteger DTOR_CALLS = new AtomicInteger();
    static final MemorySegment COUNTING_DTOR;

    static {
        try {
            MethodHandle target = MethodHandles.lookup().findStatic(MlxC.class, "onRelease", MethodType.methodType(void.class, MemorySegment.class));
            COUNTING_DTOR = FFMSupport.upcallStub(target, FunctionDescriptor.ofVoid(C_POINTER), Arena.global());
        } catch (ReflectiveOperationException e) {
            throw new ExceptionInInitializerError(e);
        }
    }

    private MlxC() {
    }

    private static void onRelease(MemorySegment data) {
        DTOR_CALLS.incrementAndGet();
    }

    private static SymbolLookup load() {
        SymbolLookup lookup = FFMSupport.loadLibrary("libmlxc.dylib", "/opt/homebrew/opt/mlx-c/lib/libmlxc.dylib");
        if (lookup == null) {
            throw new IllegalStateException("libmlxc.dylib not found; install mlx-c (brew install mlx-c)");
        }
        return lookup;
    }

    private static MethodHandle h(String name, java.lang.foreign.MemoryLayout ret, java.lang.foreign.MemoryLayout... args) {
        MethodHandle handle = FFMSupport.downcall(LIB, FunctionDescriptor.of(ret, args), name);
        if (handle == null) {
            throw new IllegalStateException("missing mlx-c symbol " + name);
        }
        return handle;
    }

    private static int check(int status, String what) {
        if (status != 0) {
            throw new IllegalStateException(what + " failed with status " + status);
        }
        return status;
    }

    static String version() throws Throwable {
        MemorySegment str = (MemorySegment) STRING_NEW.invokeExact();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment out = arena.allocate(C_POINTER);
            out.set(C_POINTER, 0, str);
            check((int) VERSION.invokeExact(out), "mlx_version");
            str = out.get(C_POINTER, 0);
            return FFMSupport.readCString((MemorySegment) STRING_DATA.invokeExact(str));
        } finally {
            int ignored = (int) STRING_FREE.invokeExact(str);
        }
    }

    static MemorySegment gpuStream() throws Throwable {
        return (MemorySegment) GPU_STREAM_NEW.invokeExact();
    }

    static void synchronize(MemorySegment stream) throws Throwable {
        check((int) SYNCHRONIZE.invokeExact(stream), "mlx_synchronize");
    }

    static MemorySegment scalar(float v) throws Throwable {
        return (MemorySegment) ARRAY_NEW_F32.invokeExact(v);
    }

    /** Wraps {@code numElements} elements at {@code address} without asking MLX to copy them. */
    static MemorySegment wrap(long address, int[] shape, int dtype, Arena arena) throws Throwable {
        MemorySegment shapeSeg = arena.allocateArray(C_INT, shape);
        return (MemorySegment) ARRAY_NEW_DATA_MANAGED.invokeExact(MemorySegment.ofAddress(address), shapeSeg, shape.length, dtype, COUNTING_DTOR);
    }

    /** New MLX-owned array holding a copy of {@code values}. */
    static MemorySegment fromFloats(float[] values, Arena arena) throws Throwable {
        MemorySegment data = arena.allocateArray(C_FLOAT, values);
        MemorySegment shape = arena.allocateArray(C_INT, new int[] { values.length });
        return (MemorySegment) ARRAY_NEW_DATA.invokeExact(data, shape, 1, MLX_FLOAT32);
    }

    static MemorySegment asStrided(MemorySegment a, int[] shape, long[] strides, long offset, MemorySegment stream, Arena arena) throws Throwable {
        MemorySegment res = arena.allocate(C_POINTER);
        res.set(C_POINTER, 0, (MemorySegment) ARRAY_NEW.invokeExact());
        check((int) AS_STRIDED.invokeExact(res, a, arena.allocateArray(C_INT, shape), (long) shape.length, arena.allocateArray(C_LONG, strides), (long) strides.length, offset, stream),
                "mlx_as_strided");
        return res.get(C_POINTER, 0);
    }

    static MemorySegment multiply(MemorySegment a, MemorySegment b, MemorySegment stream, Arena arena) throws Throwable {
        return binary(MULTIPLY, "mlx_multiply", a, b, stream, arena);
    }

    static MemorySegment add(MemorySegment a, MemorySegment b, MemorySegment stream, Arena arena) throws Throwable {
        return binary(ADD, "mlx_add", a, b, stream, arena);
    }

    static MemorySegment matmul(MemorySegment a, MemorySegment b, MemorySegment stream, Arena arena) throws Throwable {
        return binary(MATMUL, "mlx_matmul", a, b, stream, arena);
    }

    private static MemorySegment binary(MethodHandle op, String name, MemorySegment a, MemorySegment b, MemorySegment stream, Arena arena) throws Throwable {
        MemorySegment res = arena.allocate(C_POINTER);
        res.set(C_POINTER, 0, (MemorySegment) ARRAY_NEW.invokeExact());
        check((int) op.invokeExact(res, a, b, stream), name);
        return res.get(C_POINTER, 0);
    }

    static MemorySegment rmsNorm(MemorySegment x, MemorySegment w, float eps, MemorySegment stream, Arena arena) throws Throwable {
        MemorySegment res = arena.allocate(C_POINTER);
        res.set(C_POINTER, 0, (MemorySegment) ARRAY_NEW.invokeExact());
        check((int) RMS_NORM.invokeExact(res, x, w, eps, stream), "mlx_fast_rms_norm");
        return res.get(C_POINTER, 0);
    }

    static MemorySegment optInt(int v, Arena arena) {
        MemorySegment seg = arena.allocate(OPT_INT);
        seg.set(C_INT, 0, v);
        seg.set(ValueLayout.JAVA_BOOLEAN, 4, true);
        return seg;
    }

    static MemorySegment optFloat(float v, Arena arena) {
        MemorySegment seg = arena.allocate(OPT_FLOAT);
        seg.set(C_FLOAT, 0, v);
        seg.set(ValueLayout.JAVA_BOOLEAN, 4, true);
        return seg;
    }

    /** New MLX-owned array copied from {@code data}. */
    static MemorySegment fromSegment(MemorySegment data, int[] shape, int dtype, Arena arena) throws Throwable {
        return (MemorySegment) ARRAY_NEW_DATA.invokeExact(data, arena.allocateArray(C_INT, shape), shape.length, dtype);
    }

    /** Affine quantization: returns {packed weights (uint32), scales, biases}. */
    static MemorySegment[] quantize(MemorySegment w, int groupSize, int bits, MemorySegment stream, Arena arena) throws Throwable {
        MemorySegment vec = arena.allocate(C_POINTER);
        vec.set(C_POINTER, 0, (MemorySegment) VECTOR_ARRAY_NEW.invokeExact());
        check((int) QUANTIZE.invokeExact(vec, w, optInt(groupSize, arena), optInt(bits, arena), FFMSupport.allocateCString(arena, "affine"), MemorySegment.NULL, stream), "mlx_quantize");
        MemorySegment v = vec.get(C_POINTER, 0);
        MemorySegment[] out = new MemorySegment[3];
        for (int i = 0; i < 3; i++) {
            MemorySegment slot = arena.allocate(C_POINTER);
            slot.set(C_POINTER, 0, (MemorySegment) ARRAY_NEW.invokeExact());
            check((int) VECTOR_ARRAY_GET.invokeExact(slot, v, (long) i), "mlx_vector_array_get");
            out[i] = slot.get(C_POINTER, 0);
        }
        int ignored = (int) VECTOR_ARRAY_FREE.invokeExact(v);
        return out;
    }

    static MemorySegment quantizedMatmul(MemorySegment x, MemorySegment w, MemorySegment scales, MemorySegment biases, int groupSize, int bits, MemorySegment stream, Arena arena)
            throws Throwable {
        MemorySegment res = arena.allocate(C_POINTER);
        res.set(C_POINTER, 0, (MemorySegment) ARRAY_NEW.invokeExact());
        check((int) QUANTIZED_MATMUL.invokeExact(res, x, w, scales, biases, true, optInt(groupSize, arena), optInt(bits, arena), FFMSupport.allocateCString(arena, "affine"), stream),
                "mlx_quantized_matmul");
        return res.get(C_POINTER, 0);
    }

    /** Non-traditional (half-split) RoPE over the last axis of x = [B, H, L, D]. */
    static MemorySegment rope(MemorySegment x, int dims, float base, int offset, MemorySegment stream, Arena arena) throws Throwable {
        MemorySegment res = arena.allocate(C_POINTER);
        res.set(C_POINTER, 0, (MemorySegment) ARRAY_NEW.invokeExact());
        check((int) ROPE.invokeExact(res, x, dims, false, optFloat(base, arena), 1.0f, offset, MemorySegment.NULL, stream), "mlx_fast_rope");
        return res.get(C_POINTER, 0);
    }

    /** Address of an evaluated array's first element, whatever its dtype. */
    static long rawAddress(MemorySegment a) throws Throwable {
        return ((MemorySegment) ARRAY_DATA_U8.invokeExact(a)).address();
    }

    static void eval(MemorySegment a) throws Throwable {
        check((int) ARRAY_EVAL.invokeExact(a), "mlx_array_eval");
    }

    static long dataAddress(MemorySegment a) throws Throwable {
        return ((MemorySegment) ARRAY_DATA_F32.invokeExact(a)).address();
    }

    static long nbytes(MemorySegment a) throws Throwable {
        return (long) ARRAY_NBYTES.invokeExact(a);
    }

    static void free(MemorySegment a) throws Throwable {
        int ignored = (int) ARRAY_FREE.invokeExact(a);
    }
}
