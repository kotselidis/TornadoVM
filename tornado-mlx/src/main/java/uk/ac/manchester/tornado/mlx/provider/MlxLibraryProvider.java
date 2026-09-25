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

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
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
            entry("square", c -> unary(c, "mlx_square", MlxC::mlx_square)), //
            // Linear algebra.
            entry("matmul", MlxLibraryProvider::matmul), //
            entry("matmul_transposed", MlxLibraryProvider::matmulTransposed), //
            entry("addmm", MlxLibraryProvider::addmm), //
            // Affine group quantization.
            entry("quantized_matmul", MlxLibraryProvider::quantizedMatmul), //
            entry("gather_qmm", MlxLibraryProvider::gatherQmm), //
            entry("quantize", MlxLibraryProvider::quantize), //
            entry("dequantize", MlxLibraryProvider::dequantize), //
            // mlx.fast
            entry("fast_rms_norm", MlxLibraryProvider::rmsNorm), //
            entry("fast_layer_norm", MlxLibraryProvider::layerNorm), //
            entry("fast_rope", MlxLibraryProvider::rope), //
            entry("fast_rope_dynamic", MlxLibraryProvider::ropeDynamic), //
            entry("fast_scaled_dot_product_attention", MlxLibraryProvider::sdpa), //
            // Softmax, argmax, top-k.
            entry("softmax", MlxLibraryProvider::softmax), //
            entry("softmax_axis", MlxLibraryProvider::softmaxRows), //
            entry("softmax_axes", MlxLibraryProvider::softmaxLastTwoAxes), //
            entry("argmax", MlxLibraryProvider::argmax), //
            entry("argmax_axis", MlxLibraryProvider::argmaxRows), //
            entry("topk", MlxLibraryProvider::topk), //
            entry("topk_axis", MlxLibraryProvider::topkRows));

    /** Affine group quantization mode (scales and biases per group), as used by MLX-LM. */
    private static final String AFFINE = "affine";

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

    /**
     * Bytes MLX currently has allocated for live arrays (excluding its buffer cache). Used by the
     * leak tests.
     */
    public static long activeMemoryBytes() {
        MlxNativeLib.load();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment bytes = arena.allocate(ValueLayout.JAVA_LONG);
            MlxNativeLib.check(MlxC.mlx_get_active_memory(bytes), "mlx_get_active_memory");
            return bytes.get(ValueLayout.JAVA_LONG, 0);
        }
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

    // matmul(a, b, c, m, k, n): c[m, n] = a[m, k] @ b[k, n]
    private static void matmul(MlxCall c) {
        int m = c.intArg(3);
        int k = c.intArg(4);
        int n = c.intArg(5);
        MemorySegment a = c.input(0, m, k);
        MemorySegment b = c.input(1, k, n);
        c.store(c.op("mlx_matmul", res -> MlxC.mlx_matmul(res, a, b, c.stream())), 2);
    }

    // matmul_transposed(a, w, c, m, k, n): c[m, n] = a[m, k] @ w[n, k]^T (weights stored row per output)
    private static void matmulTransposed(MlxCall c) {
        int m = c.intArg(3);
        int k = c.intArg(4);
        int n = c.intArg(5);
        MemorySegment a = c.input(0, m, k);
        MemorySegment w = c.input(1, n, k);
        MemorySegment wt = c.op("mlx_transpose", res -> MlxC.mlx_transpose(res, w, c.stream()));
        c.store(c.op("mlx_matmul", res -> MlxC.mlx_matmul(res, a, wt, c.stream())), 2);
    }

    // addmm(cIn, a, b, out, m, k, n, alpha, beta): out = alpha * a @ b + beta * cIn
    private static void addmm(MlxCall c) {
        int m = c.intArg(4);
        int k = c.intArg(5);
        int n = c.intArg(6);
        float alpha = c.floatArg(7);
        float beta = c.floatArg(8);
        MemorySegment cIn = c.input(0, m, n);
        MemorySegment a = c.input(1, m, k);
        MemorySegment b = c.input(2, k, n);
        c.store(c.op("mlx_addmm", res -> MlxC.mlx_addmm(res, cIn, a, b, alpha, beta, c.stream())), 3);
    }

    // quantized_matmul(x, wq, scales, biases, y, m, k, n, groupSize, bits): y[m, n] = x[m, k] @ dequant(w[n, k])^T
    private static void quantizedMatmul(MlxCall c) {
        int m = c.intArg(5);
        int k = c.intArg(6);
        int n = c.intArg(7);
        int groupSize = c.intArg(8);
        int bits = c.intArg(9);
        MemorySegment x = c.input(0, m, k);
        MemorySegment wq = c.inputAs(1, MlxNativeLib.MLX_UINT32, n, k * bits / 32);
        MemorySegment scales = c.input(2, n, k / groupSize);
        MemorySegment biases = c.input(3, n, k / groupSize);
        c.store(c.op("mlx_quantized_matmul", res -> MlxC.mlx_quantized_matmul(res, x, wq, scales, biases, true, c.optionalInt(groupSize), c.optionalInt(bits), c.cString(AFFINE),
                c.stream())), 4);
    }

    // gather_qmm(x, wq, scales, biases, lhsIndices, rhsIndices, y, batches, experts, m, k, n, groupSize, bits):
    // y[b] = x[lhsIndices[b]] @ dequant(w[rhsIndices[b]])^T, with x [batches, m, k] and w [experts, n, k]
    private static void gatherQmm(MlxCall c) {
        int batches = c.intArg(7);
        int experts = c.intArg(8);
        int m = c.intArg(9);
        int k = c.intArg(10);
        int n = c.intArg(11);
        int groupSize = c.intArg(12);
        int bits = c.intArg(13);
        MemorySegment x = c.input(0, batches, m, k);
        MemorySegment wq = c.inputAs(1, MlxNativeLib.MLX_UINT32, experts, n, k * bits / 32);
        MemorySegment scales = c.input(2, experts, n, k / groupSize);
        MemorySegment biases = c.input(3, experts, n, k / groupSize);
        MemorySegment lhs = c.inputAs(4, MlxNativeLib.MLX_UINT32, batches);
        MemorySegment rhs = c.inputAs(5, MlxNativeLib.MLX_UINT32, batches);
        c.store(c.op("mlx_gather_qmm", res -> MlxC.mlx_gather_qmm(res, x, wq, scales, biases, lhs, rhs, true, c.optionalInt(groupSize), c.optionalInt(bits), c.cString(AFFINE), false,
                c.stream())), 6);
    }

    // quantize(w, wq, scales, biases, rows, cols, groupSize, bits): w[rows, cols] -> packed wq, scales, biases
    private static void quantize(MlxCall c) {
        int rows = c.intArg(4);
        int cols = c.intArg(5);
        int groupSize = c.intArg(6);
        int bits = c.intArg(7);
        MemorySegment w = c.input(0, rows, cols);
        MemorySegment[] q = c.vectorOp("mlx_quantize", 3, vec -> MlxC.mlx_quantize(vec, w, c.optionalInt(groupSize), c.optionalInt(bits), c.cString(AFFINE), MlxCall.none(), c.stream()));
        c.storeRaw(q[0], 1);
        c.store(q[1], 2);
        c.store(q[2], 3);
    }

    // dequantize(wq, scales, biases, w, rows, cols, groupSize, bits): packed wq, scales, biases -> w[rows, cols]
    private static void dequantize(MlxCall c) {
        int rows = c.intArg(4);
        int cols = c.intArg(5);
        int groupSize = c.intArg(6);
        int bits = c.intArg(7);
        MemorySegment wq = c.inputAs(0, MlxNativeLib.MLX_UINT32, rows, cols * bits / 32);
        MemorySegment scales = c.input(1, rows, cols / groupSize);
        MemorySegment biases = c.input(2, rows, cols / groupSize);
        c.store(c.op("mlx_dequantize", res -> MlxC.mlx_dequantize(res, wq, scales, biases, c.optionalInt(groupSize), c.optionalInt(bits), c.cString(AFFINE), MlxCall.none(), c.noDtype(),
                c.stream())), 3);
    }

    // fast_rms_norm(x, weight, out, rows, dim, eps)
    private static void rmsNorm(MlxCall c) {
        int rows = c.intArg(3);
        int dim = c.intArg(4);
        float eps = c.floatArg(5);
        MemorySegment x = c.input(0, rows, dim);
        MemorySegment w = c.input(1, dim);
        c.store(c.op("mlx_fast_rms_norm", res -> MlxC.mlx_fast_rms_norm(res, x, w, eps, c.stream())), 2);
    }

    // fast_layer_norm(x, weight, bias, out, rows, dim, eps)
    private static void layerNorm(MlxCall c) {
        int rows = c.intArg(4);
        int dim = c.intArg(5);
        float eps = c.floatArg(6);
        MemorySegment x = c.input(0, rows, dim);
        MemorySegment w = c.input(1, dim);
        MemorySegment b = c.input(2, dim);
        c.store(c.op("mlx_fast_layer_norm", res -> MlxC.mlx_fast_layer_norm(res, x, w, b, eps, c.stream())), 3);
    }

    // fast_rope(x, out, batch, heads, seqLen, headDim, dims, traditional, base, scale, offset): x [batch, heads, seqLen, headDim]
    private static void rope(MlxCall c) {
        int batch = c.intArg(2);
        int heads = c.intArg(3);
        int seqLen = c.intArg(4);
        int headDim = c.intArg(5);
        int dims = c.intArg(6);
        boolean traditional = c.boolArg(7);
        float base = c.floatArg(8);
        float scale = c.floatArg(9);
        int offset = c.intArg(10);
        MemorySegment x = c.input(0, batch, heads, seqLen, headDim);
        c.store(c.op("mlx_fast_rope", res -> MlxC.mlx_fast_rope(res, x, dims, traditional, c.optionalFloat(base), scale, offset, MlxCall.none(), c.stream())), 1);
    }

    // fast_rope_dynamic(x, offset, out, batch, heads, seqLen, headDim, dims, traditional, base, scale): offset is a one-element IntArray
    private static void ropeDynamic(MlxCall c) {
        int batch = c.intArg(3);
        int heads = c.intArg(4);
        int seqLen = c.intArg(5);
        int headDim = c.intArg(6);
        int dims = c.intArg(7);
        boolean traditional = c.boolArg(8);
        float base = c.floatArg(9);
        float scale = c.floatArg(10);
        MemorySegment x = c.input(0, batch, heads, seqLen, headDim);
        MemorySegment offset = c.input(1);
        c.store(c.op("mlx_fast_rope_dynamic", res -> MlxC.mlx_fast_rope_dynamic(res, x, dims, traditional, c.optionalFloat(base), scale, offset, MlxCall.none(), c.stream())), 2);
    }

    // fast_scaled_dot_product_attention(q, k, v, out, batch, qHeads, kvHeads, qLen, kvLen, headDim, scale, causal)
    private static void sdpa(MlxCall c) {
        int batch = c.intArg(4);
        int qHeads = c.intArg(5);
        int kvHeads = c.intArg(6);
        int qLen = c.intArg(7);
        int kvLen = c.intArg(8);
        int headDim = c.intArg(9);
        float scale = c.floatArg(10);
        boolean causal = c.boolArg(11);
        MemorySegment q = c.input(0, batch, qHeads, qLen, headDim);
        MemorySegment k = c.input(1, batch, kvHeads, kvLen, headDim);
        MemorySegment v = c.input(2, batch, kvHeads, kvLen, headDim);
        c.store(c.op("mlx_fast_scaled_dot_product_attention", res -> MlxC.mlx_fast_scaled_dot_product_attention(res, q, k, v, scale, c.cString(causal ? "causal" : ""), MlxCall.none(),
                MlxCall.none(), c.stream())), 3);
    }

    // softmax(x, out): over the whole array. Low-precision inputs are accumulated in float32.
    private static void softmax(MlxCall c) {
        MemorySegment x = c.input(0, c.length(0));
        c.store(c.op("mlx_softmax", res -> MlxC.mlx_softmax(res, x, true, c.stream())), 1);
    }

    // softmax_axis(x, out, rows, cols): per row
    private static void softmaxRows(MlxCall c) {
        MemorySegment x = c.input(0, c.intArg(2), c.intArg(3));
        c.store(c.op("mlx_softmax_axis", res -> MlxC.mlx_softmax_axis(res, x, -1, true, c.stream())), 1);
    }

    // softmax_axes(x, out, d0, d1, d2): over the last two axes of a [d0, d1, d2] tensor
    private static void softmaxLastTwoAxes(MlxCall c) {
        MemorySegment x = c.input(0, c.intArg(2), c.intArg(3), c.intArg(4));
        c.store(c.op("mlx_softmax_axes", res -> MlxC.mlx_softmax_axes(res, x, c.ints(1, 2), 2, true, c.stream())), 1);
    }

    // argmax(x, out): index of the largest element of the whole array, into a one-element IntArray
    private static void argmax(MlxCall c) {
        MemorySegment x = c.input(0, c.length(0));
        c.store(c.op("mlx_argmax", res -> MlxC.mlx_argmax(res, x, false, c.stream())), 1);
    }

    // argmax_axis(x, out, rows, cols): per row, into an IntArray of length rows
    private static void argmaxRows(MlxCall c) {
        MemorySegment x = c.input(0, c.intArg(2), c.intArg(3));
        c.store(c.op("mlx_argmax_axis", res -> MlxC.mlx_argmax_axis(res, x, -1, false, c.stream())), 1);
    }

    // topk(x, out, k): the k largest elements of the whole array, in no particular order
    private static void topk(MlxCall c) {
        int k = c.intArg(2);
        MemorySegment x = c.input(0, c.length(0));
        c.store(c.op("mlx_topk", res -> MlxC.mlx_topk(res, x, k, c.stream())), 1);
    }

    // topk_axis(x, out, rows, cols, k): the k largest elements of each row, in no particular order
    private static void topkRows(MlxCall c) {
        int rows = c.intArg(2);
        int cols = c.intArg(3);
        int k = c.intArg(4);
        MemorySegment x = c.input(0, rows, cols);
        c.store(c.op("mlx_topk_axis", res -> MlxC.mlx_topk_axis(res, x, k, -1, c.stream())), 1);
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
