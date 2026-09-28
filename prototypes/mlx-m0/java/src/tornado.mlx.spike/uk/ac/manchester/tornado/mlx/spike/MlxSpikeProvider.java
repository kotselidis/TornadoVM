package uk.ac.manchester.tornado.mlx.spike;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

import uk.ac.manchester.tornado.api.types.arrays.TornadoNativeArray;
import uk.ac.manchester.tornado.drivers.metal.ffm.MetalAPI;
import uk.ac.manchester.tornado.drivers.metal.runtime.MetalTornadoDevice;
import uk.ac.manchester.tornado.runtime.common.TornadoXPUDevice;
import uk.ac.manchester.tornado.runtime.library.spi.LibraryContext;
import uk.ac.manchester.tornado.runtime.library.spi.LibraryInvocation;
import uk.ac.manchester.tornado.runtime.library.spi.TornadoLibraryProvider;

/**
 * Throwaway provider for the M0 spike. Wraps TornadoVM's Metal buffers as MLX arrays, runs one
 * MLX op, and copies the result into the TornadoVM output buffer.
 *
 * <p>
 * Until M1 fixes the interpreter, a reference argument arrives as
 * {@code toBuffer() + ARRAY_HEADER}, which on Metal is the MTLBuffer handle plus the header.
 * The spike subtracts the header to recover the handle and asks Metal for its contents. It
 * assumes {@code getBufferOffset() == 0}, which holds for arrays that are not sub-regions.
 */
public final class MlxSpikeProvider implements TornadoLibraryProvider {

    public static final String NAME = "apple/mlx-spike";

    /** Per-dispatch measurements, read by the driver. */
    public static final class Stats {
        public long wrapNanos;
        public long computeNanos;
        public long copyOutNanos;
        public double gpuMicros;
        public int contextsDestroyed;
        public boolean inputZeroCopy = true;
        public int calls;

        public void reset() {
            wrapNanos = computeNanos = copyOutNanos = 0;
            gpuMicros = 0;
            inputZeroCopy = true;
            calls = 0;
        }
    }

    public static final Stats STATS = new Stats();

    /**
     * How inputs are wrapped: "strided" wraps the MTLBuffer from its base and views past the
     * header with as_strided (the plan's approach); "direct" wraps the first data element
     * directly, which relies on Metal accepting a non-page-aligned no-copy pointer.
     */
    public static final String WRAP_MODE = System.getProperty("mlx.spike.wrap", "strided");

    /** When true, the MLX array wrapping a TornadoVM buffer is kept and reused across executions. */
    public static final boolean CACHE_WRAPS = Boolean.getBoolean("mlx.spike.cache");

    private record WrapKey(long address, java.util.List<Integer> shape, int dtype) {
    }

    private static final java.util.Map<WrapKey, MemorySegment> WRAPS = new java.util.HashMap<>();

    private static final class Context implements LibraryContext {
        final MemorySegment stream;

        Context(MemorySegment stream) {
            this.stream = stream;
        }
    }

    @Override
    public String libraryName() {
        return NAME;
    }

    @Override
    public boolean canHandle(TornadoXPUDevice device) {
        return device instanceof MetalTornadoDevice;
    }

    @Override
    public LibraryContext createContext(TornadoXPUDevice device, long executionPlanId) {
        try {
            return new Context(MlxC.gpuStream());
        } catch (Throwable t) {
            throw new IllegalStateException(t);
        }
    }

    @Override
    public void destroyContext(LibraryContext context) {
        // TornadoVM pools and reuses MTLBuffers, so a wrapper must not outlive the buffers it was
        // made for: a cached wrapper over a freed buffer reads stale memory. The spike drops the
        // whole cache when a plan's context goes away; the real provider must key invalidation
        // on buffer release.
        synchronized (WRAPS) {
            for (MemorySegment wrapped : WRAPS.values()) {
                try {
                    MlxC.free(wrapped);
                } catch (Throwable t) {
                    throw new IllegalStateException(t);
                }
            }
            WRAPS.clear();
        }
        STATS.contextsDestroyed++;
    }

    /** CPU address of the first element of reference argument {@code i}. */
    private static long dataAddress(LibraryInvocation inv, int i) {
        long mtlBuffer = inv.getDevicePointer(i) - TornadoNativeArray.ARRAY_HEADER;
        return MetalAPI.bufferContents(mtlBuffer) + TornadoNativeArray.ARRAY_HEADER;
    }

    @Override
    public void dispatch(String functionName, LibraryInvocation inv) {
        Context ctx = (Context) inv.getContext();
        try (Arena arena = Arena.ofConfined()) {
            switch (functionName) {
                // affine(x, y, n, a, b): y = a * x + b
                case "affine" -> {
                    int n = (Integer) inv.getArg(2);
                    float a = (Float) inv.getArg(3);
                    float b = (Float) inv.getArg(4);
                    long t0 = System.nanoTime();
                    MemorySegment x = wrapInput(inv, 0, new int[] { n }, ctx.stream, arena);
                    long t1 = System.nanoTime();
                    MemorySegment sa = MlxC.scalar(a), sb = MlxC.scalar(b);
                    MemorySegment ax = MlxC.multiply(x, sa, ctx.stream, arena);
                    MemorySegment y = MlxC.add(ax, sb, ctx.stream, arena);
                    MlxC.eval(y);
                    long t2 = System.nanoTime();
                    copyOut(y, dataAddress(inv, 1));
                    long t3 = System.nanoTime();
                    for (MemorySegment m : new MemorySegment[] { y, ax, sb, sa, x }) {
                        if (!isCached(m)) {
                            MlxC.free(m);
                        }
                    }
                    record(t0, t1, t2, t3);
                }
                // matmul(a, b, c, m, k, n): c[m,n] = a[m,k] @ b[k,n]
                case "matmul" -> {
                    int m = (Integer) inv.getArg(3), k = (Integer) inv.getArg(4), n = (Integer) inv.getArg(5);
                    long t0 = System.nanoTime();
                    MemorySegment a = wrapInput(inv, 0, new int[] { m, k }, ctx.stream, arena);
                    MemorySegment b = wrapInput(inv, 1, new int[] { k, n }, ctx.stream, arena);
                    long t1 = System.nanoTime();
                    MemorySegment c = MlxC.matmul(a, b, ctx.stream, arena);
                    MlxC.eval(c);
                    long t2 = System.nanoTime();
                    copyOut(c, dataAddress(inv, 2));
                    long t3 = System.nanoTime();
                    MlxC.free(c);
                    if (!isCached(b)) {
                        MlxC.free(b);
                    }
                    if (!isCached(a)) {
                        MlxC.free(a);
                    }
                    record(t0, t1, t2, t3);
                }
                // rms_mlxc(x, w, out, rows, dim, eps): MLX fast.rms_norm through mlx-c, result copied out
                case "rms_mlxc" -> {
                    int rows = (Integer) inv.getArg(3), dim = (Integer) inv.getArg(4);
                    float eps = (Float) inv.getArg(5);
                    long t0 = System.nanoTime();
                    MemorySegment x = wrapInput(inv, 0, new int[] { rows, dim }, ctx.stream, arena);
                    MemorySegment w = wrapInput(inv, 1, new int[] { dim }, ctx.stream, arena);
                    long t1 = System.nanoTime();
                    MemorySegment y = MlxC.rmsNorm(x, w, eps, ctx.stream, arena);
                    MlxC.eval(y);
                    long t2 = System.nanoTime();
                    copyOut(y, dataAddress(inv, 2));
                    long t3 = System.nanoTime();
                    MlxC.free(y);
                    if (!isCached(w)) {
                        MlxC.free(w);
                    }
                    if (!isCached(x)) {
                        MlxC.free(x);
                    }
                    record(t0, t1, t2, t3);
                }
                // rms_kernel(x, w, out, rows, dim, eps): MLX's metallib kernel on TornadoVM's queue, in place
                case "rms_kernel" -> {
                    int rows = (Integer) inv.getArg(3), dim = (Integer) inv.getArg(4);
                    float eps = (Float) inv.getArg(5);
                    long h = TornadoNativeArray.ARRAY_HEADER;
                    long t0 = System.nanoTime();
                    long queue = MlxKernels.tornadoQueue((MetalTornadoDevice) inv.getDevice(), inv.getExecutionPlanId());
                    long t1 = System.nanoTime();
                    MlxKernels.rmsNorm(queue, inv.getDevicePointer(0) - h, inv.getDevicePointer(1) - h, inv.getDevicePointer(2) - h, h, rows, dim, eps);
                    long t2 = System.nanoTime();
                    record(t0, t1, t2, t2);
                    STATS.gpuMicros += MlxKernels.lastGpuMicros;
                }
                // gemv_f16(w, x, y, rows, cols): y[rows] = w[rows, cols] @ x[cols], float16
                case "gemv_f16" -> {
                    int rows = (Integer) inv.getArg(3), cols = (Integer) inv.getArg(4);
                    long t0 = System.nanoTime();
                    MemorySegment w = wrapInput(inv, 0, new int[] { rows, cols }, MlxC.MLX_FLOAT16, 2, ctx.stream, arena);
                    MemorySegment x = wrapInput(inv, 1, new int[] { cols, 1 }, MlxC.MLX_FLOAT16, 2, ctx.stream, arena);
                    long t1 = System.nanoTime();
                    MemorySegment y = MlxC.matmul(w, x, ctx.stream, arena);
                    MlxC.eval(y);
                    long t2 = System.nanoTime();
                    copyOut(y, dataAddress(inv, 2));
                    long t3 = System.nanoTime();
                    MlxC.free(y);
                    freeUnlessCached(x, w);
                    record(t0, t1, t2, t3);
                }
                // qmv4(wq, scales, biases, x, y, rows, cols, groupSize): 4-bit affine quantized y = W x
                case "qmv4" -> {
                    int rows = (Integer) inv.getArg(5), cols = (Integer) inv.getArg(6), gs = (Integer) inv.getArg(7);
                    long t0 = System.nanoTime();
                    MemorySegment wq = wrapInput(inv, 0, new int[] { rows, cols / 8 }, MlxC.MLX_UINT32, 4, ctx.stream, arena);
                    MemorySegment sc = wrapInput(inv, 1, new int[] { rows, cols / gs }, MlxC.MLX_FLOAT16, 2, ctx.stream, arena);
                    MemorySegment bi = wrapInput(inv, 2, new int[] { rows, cols / gs }, MlxC.MLX_FLOAT16, 2, ctx.stream, arena);
                    MemorySegment x = wrapInput(inv, 3, new int[] { 1, cols }, MlxC.MLX_FLOAT16, 2, ctx.stream, arena);
                    long t1 = System.nanoTime();
                    MemorySegment y = MlxC.quantizedMatmul(x, wq, sc, bi, gs, 4, ctx.stream, arena);
                    MlxC.eval(y);
                    long t2 = System.nanoTime();
                    copyOut(y, dataAddress(inv, 4));
                    long t3 = System.nanoTime();
                    MlxC.free(y);
                    freeUnlessCached(x, bi, sc, wq);
                    record(t0, t1, t2, t3);
                }
                // rope(x, y, heads, headDim, offset, base): half-split RoPE at one position, float16
                case "rope" -> {
                    int heads = (Integer) inv.getArg(2), dim = (Integer) inv.getArg(3), offset = (Integer) inv.getArg(4);
                    float base = (Float) inv.getArg(5);
                    long t0 = System.nanoTime();
                    MemorySegment x = wrapInput(inv, 0, new int[] { 1, heads, 1, dim }, MlxC.MLX_FLOAT16, 2, ctx.stream, arena);
                    long t1 = System.nanoTime();
                    MemorySegment y = MlxC.rope(x, dim, base, offset, ctx.stream, arena);
                    MlxC.eval(y);
                    long t2 = System.nanoTime();
                    copyOut(y, dataAddress(inv, 1));
                    long t3 = System.nanoTime();
                    MlxC.free(y);
                    freeUnlessCached(x);
                    record(t0, t1, t2, t3);
                }
                default -> throw new IllegalArgumentException("unknown function " + functionName);
            }
        } catch (RuntimeException e) {
            throw e;
        } catch (Throwable t) {
            throw new IllegalStateException(t);
        }
    }

    /**
     * Wraps argument {@code i} as a float32 MLX array without copying: the whole MTLBuffer from
     * its base (header included), then a strided view that starts past the header.
     */
    private static MemorySegment wrapInput(LibraryInvocation inv, int i, int[] shape, MemorySegment stream, Arena arena) throws Throwable {
        return wrapInput(inv, i, shape, MlxC.MLX_FLOAT32, Float.BYTES, stream, arena);
    }

    private static MemorySegment wrapInput(LibraryInvocation inv, int i, int[] shape, int dtype, int elemBytes, MemorySegment stream, Arena arena) throws Throwable {
        if (CACHE_WRAPS) {
            WrapKey key = new WrapKey(dataAddress(inv, i), java.util.Arrays.stream(shape).boxed().toList(), dtype);
            MemorySegment cached = WRAPS.get(key);
            if (cached == null) {
                cached = wrapFresh(inv, i, shape, dtype, elemBytes, stream, arena);
                WRAPS.put(key, cached);
            }
            return cached;
        }
        return wrapFresh(inv, i, shape, dtype, elemBytes, stream, arena);
    }

    private static void freeUnlessCached(MemorySegment... arrays) throws Throwable {
        for (MemorySegment a : arrays) {
            if (!isCached(a)) {
                MlxC.free(a);
            }
        }
    }

    private static boolean isCached(MemorySegment array) {
        return CACHE_WRAPS && WRAPS.containsValue(array);
    }

    private static MemorySegment wrapFresh(LibraryInvocation inv, int i, int[] shape, int dtype, int elemBytes, MemorySegment stream, Arena arena) throws Throwable {
        if (WRAP_MODE.equals("direct")) {
            long data = dataAddress(inv, i);
            int before = MlxC.DTOR_CALLS.get();
            MemorySegment arr = MlxC.wrap(data, shape, dtype, arena);
            MlxC.eval(arr);
            if (MlxC.DTOR_CALLS.get() != before || MlxC.rawAddress(arr) != data) {
                STATS.inputZeroCopy = false;
            }
            return arr;
        }
        long base = MetalAPI.bufferContents(inv.getDevicePointer(i) - TornadoNativeArray.ARRAY_HEADER);
        int elements = 1;
        for (int d : shape) {
            elements *= d;
        }
        int headerElems = (int) (TornadoNativeArray.ARRAY_HEADER / elemBytes);
        int before = MlxC.DTOR_CALLS.get();
        MemorySegment whole = MlxC.wrap(base, new int[] { headerElems + elements }, dtype, arena);
        if (MlxC.DTOR_CALLS.get() != before) {
            STATS.inputZeroCopy = false;
        }
        long[] strides = new long[shape.length];
        long s = 1;
        for (int d = shape.length - 1; d >= 0; d--) {
            strides[d] = s;
            s *= shape[d];
        }
        MemorySegment view = MlxC.asStrided(whole, shape, strides, headerElems, stream, arena);
        MlxC.eval(whole);
        if (MlxC.rawAddress(whole) != base) {
            STATS.inputZeroCopy = false;
        }
        MlxC.free(whole);
        return view;
    }

    /** Copy-out threads: 1 = plain memcpy; more splits the copy into chunks across cores. */
    public static final int COPY_THREADS = Integer.getInteger("mlx.spike.copyThreads", 1);

    private static final long PARALLEL_COPY_MIN_BYTES = 1L << 20;

    private static void copyOut(MemorySegment result, long dst) throws Throwable {
        long bytes = MlxC.nbytes(result);
        MemorySegment src = MemorySegment.ofAddress(MlxC.rawAddress(result)).reinterpret(bytes);
        MemorySegment out = MemorySegment.ofAddress(dst).reinterpret(bytes);
        if (COPY_THREADS <= 1 || bytes < PARALLEL_COPY_MIN_BYTES) {
            out.copyFrom(src);
            return;
        }
        long chunk = (bytes + COPY_THREADS - 1) / COPY_THREADS;
        java.util.stream.IntStream.range(0, COPY_THREADS).parallel().forEach(t -> {
            long off = t * chunk;
            long len = Math.min(chunk, bytes - off);
            if (len > 0) {
                out.asSlice(off, len).copyFrom(src.asSlice(off, len));
            }
        });
    }

    private static void record(long t0, long t1, long t2, long t3) {
        STATS.wrapNanos += t1 - t0;
        STATS.computeNanos += t2 - t1;
        STATS.copyOutNanos += t3 - t2;
        STATS.calls++;
    }
}
