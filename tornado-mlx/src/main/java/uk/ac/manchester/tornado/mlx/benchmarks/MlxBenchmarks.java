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
package uk.ac.manchester.tornado.mlx.benchmarks;

import static uk.ac.manchester.tornado.runtime.ffm.FFMSupport.C_FLOAT;
import static uk.ac.manchester.tornado.runtime.ffm.FFMSupport.C_INT;
import static uk.ac.manchester.tornado.runtime.ffm.FFMSupport.C_POINTER;

import java.io.IOException;
import java.io.PrintStream;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.function.ToIntFunction;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.api.types.arrays.TornadoNativeArray;
import uk.ac.manchester.tornado.mlx.Mlx;
import uk.ac.manchester.tornado.mlx.jit.JitFast;
import uk.ac.manchester.tornado.mlx.provider.MlxC;
import uk.ac.manchester.tornado.mlx.provider.MlxLibraryProvider;
import uk.ac.manchester.tornado.runtime.ffm.FFMSupport;

/**
 * M4 benchmarks: Tier 1 MLX operations at LLM shapes, three ways.
 * <ol>
 * <li><b>MLX alone</b>: mlx-c on MLX-owned arrays, no TornadoVM (the upper bound);</li>
 * <li><b>apple/mlx</b>: the MLX library task on TornadoVM buffers;</li>
 * <li><b>JIT</b>: a TornadoVM JIT kernel (jitLLM's tuned kernel where one exists).</li>
 * </ol>
 * For each TornadoVM variant it reports the per-execute() time of a one-task graph and the
 * marginal cost of one more task in a chain of {@value #CHAIN} identical tasks, which is what a
 * task costs inside a larger graph such as an LLM layer. Bandwidth and throughput use the marginal
 * time.
 *
 * <pre>
 * tornado -m tornado.mlx/uk.ac.manchester.tornado.mlx.benchmarks.MlxBenchmarks [--quick] [--out report.md]
 * </pre>
 */
public final class MlxBenchmarks {

    /** Apple M1 Pro (16-core GPU): unified-memory bandwidth and FP32 throughput. */
    static final double PEAK_GBS = 200.0;
    static final double PEAK_GFLOPS = 5200.0;

    static final int CHAIN = 8;
    static final int WARMUP = 10;
    static final int ITERATIONS = 100;
    static final long TIME_BUDGET_NS = 3_000_000_000L;

    private MlxBenchmarks() {
    }

    // ---------------------------------------------------------------- results

    record Timing(double medianUs, double p10Us, double p90Us) {
        static Timing of(long[] samplesNs) {
            long[] s = samplesNs.clone();
            Arrays.sort(s);
            return new Timing(s[s.length / 2] / 1e3, s[s.length / 10] / 1e3, s[(s.length * 9) / 10] / 1e3);
        }
    }

    record Row(String op, String shape, String dtype, String variant, Timing execute, double marginalUs, double bytes, double flops, String note) {
        double gbs() {
            return marginalUs > 0 ? bytes / (marginalUs * 1e3) : Double.NaN;
        }

        double gflops() {
            return marginalUs > 0 ? flops / (marginalUs * 1e3) : Double.NaN;
        }
    }

    private static final List<Row> ROWS = new ArrayList<>();

    /** Adds one instance of the benchmarked operation (one or more tasks) to a graph. */
    interface TaskAdder {
        void add(TaskGraph graph, GridScheduler grid, String graphName, String taskName);
    }

    /** Runs the operation once on MLX-owned arrays and returns the (unevaluated) result. */
    interface MlxOperation {
        MemorySegment run(MemorySegment stream, Arena arena);
    }

    // ---------------------------------------------------------------- measurement

    private static Timing timeExecute(TaskGraph graph, GridScheduler grid) {
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            if (grid != null) {
                plan.withGridScheduler(grid);
            }
            for (int i = 0; i < WARMUP; i++) {
                plan.execute();
            }
            List<Long> samples = new ArrayList<>();
            long start = System.nanoTime();
            while (samples.size() < ITERATIONS && (samples.size() < 20 || System.nanoTime() - start < TIME_BUDGET_NS)) {
                long t0 = System.nanoTime();
                plan.execute();
                samples.add(System.nanoTime() - t0);
            }
            return Timing.of(samples.stream().mapToLong(Long::longValue).toArray());
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private static Timing graphOf(String name, int tasks, Object[] inputs, Object[] outputs, TaskAdder adder, boolean usesGrid) {
        TaskGraph graph = new TaskGraph(name).transferToDevice(DataTransferMode.FIRST_EXECUTION, inputs);
        GridScheduler grid = new GridScheduler();
        for (int i = 0; i < tasks; i++) {
            adder.add(graph, grid, name, "t" + i);
        }
        graph.transferToHost(DataTransferMode.UNDER_DEMAND, outputs);
        return timeExecute(graph, usesGrid ? grid : null);
    }

    private static int graphCounter;

    /** One-task execute() timing and the marginal per-task cost from a chain. */
    private static void measure(String op, String shape, String dtype, String variant, Object[] inputs, Object[] outputs, TaskAdder adder, boolean usesGrid, double bytes, double flops,
            String note) {
        String base = variant.replaceAll("[^A-Za-z0-9]", "") + (graphCounter++);
        Timing one = graphOf(base + "a", 1, inputs, outputs, adder, usesGrid);
        Timing chain = graphOf(base + "b", CHAIN, inputs, outputs, adder, usesGrid);
        double marginal = (chain.medianUs() - one.medianUs()) / (CHAIN - 1);
        Row row = new Row(op, shape, dtype, variant, one, marginal, bytes, flops, note);
        ROWS.add(row);
        System.out.printf(Locale.ROOT, "  %-22s %-18s %-8s %-10s execute %9.1f us (p10 %9.1f, p90 %9.1f)  marginal %9.1f us  %7.1f GB/s  %8.1f GFLOP/s%n", op, shape, dtype, variant,
                one.medianUs(), one.p10Us(), one.p90Us(), marginal, row.gbs(), row.gflops());
    }

    private static void measureMlxAlone(String op, String shape, String dtype, MlxOperation operation, double bytes, double flops) {
        MemorySegment stream = MlxC.mlx_default_gpu_stream_new();
        try (Arena arena = Arena.ofConfined()) {
            for (int i = 0; i < WARMUP; i++) {
                evalAndFree(operation.run(stream, arena));
            }
            List<Long> samples = new ArrayList<>();
            long start = System.nanoTime();
            while (samples.size() < ITERATIONS && (samples.size() < 20 || System.nanoTime() - start < TIME_BUDGET_NS)) {
                long t0 = System.nanoTime();
                evalAndFree(operation.run(stream, arena));
                samples.add(System.nanoTime() - t0);
            }
            Timing t = Timing.of(samples.stream().mapToLong(Long::longValue).toArray());
            // MLX alone has no graph around it: its per-call time is also its marginal cost.
            Row row = new Row(op, shape, dtype, "MLX alone", t, t.medianUs(), bytes, flops, "");
            ROWS.add(row);
            System.out.printf(Locale.ROOT, "  %-22s %-18s %-8s %-10s call    %9.1f us (p10 %9.1f, p90 %9.1f)                         %7.1f GB/s  %8.1f GFLOP/s%n", op, shape, dtype, "MLX alone",
                    t.medianUs(), t.p10Us(), t.p90Us(), row.gbs(), row.gflops());
        } finally {
            MlxC.mlx_stream_free(stream);
        }
    }

    private static void evalAndFree(MemorySegment array) {
        check(MlxC.mlx_array_eval(array), "mlx_array_eval");
        MlxC.mlx_array_free(array);
    }

    // ---------------------------------------------------------------- MLX-owned arrays for "MLX alone"

    private static final int MLX_FLOAT16 = 9;
    private static final int MLX_FLOAT32 = 10;
    private static final int MLX_UINT32 = 3;

    private static void check(int status, String call) {
        if (status != 0) {
            throw new IllegalStateException(call + " failed with status " + status);
        }
    }

    private static MemorySegment ints(Arena arena, int... values) {
        MemorySegment s = FFMSupport.allocateArray(arena, C_INT, Math.max(1, values.length));
        MemorySegment.copy(values, 0, s, C_INT, 0, values.length);
        return s;
    }

    static MemorySegment op(Arena arena, ToIntFunction<MemorySegment> operation) {
        MemorySegment slot = arena.allocate(C_POINTER);
        slot.set(C_POINTER, 0, MlxC.mlx_array_new());
        check(operation.applyAsInt(slot), "mlx op");
        return slot.get(C_POINTER, 0);
    }

    /** An evaluated MLX array holding a copy of a TornadoVM array's contents. */
    static MemorySegment mlxCopy(TornadoNativeArray source, int dtype, int... shape) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment data = source.getSegment();
            MemorySegment arr = MlxC.mlx_array_new_data(data, ints(arena, shape), shape.length, dtype);
            check(MlxC.mlx_array_eval(arr), "mlx_array_eval");
            return arr;
        }
    }

    static MemorySegment optionalInt(Arena arena, int value) {
        MemorySegment o = arena.allocate(MlxC.OPT_INT);
        o.set(C_INT, 0, value);
        o.set(ValueLayout.JAVA_BOOLEAN, 4, true);
        return o;
    }

    static MemorySegment optionalFloat(Arena arena, float value) {
        MemorySegment o = arena.allocate(MlxC.OPT_FLOAT);
        o.set(C_FLOAT, 0, value);
        o.set(ValueLayout.JAVA_BOOLEAN, 4, true);
        return o;
    }

    // ---------------------------------------------------------------- data

    static HalfFloatArray randomHalf(int n, float scale, long seed) {
        Random r = new Random(seed);
        HalfFloatArray a = new HalfFloatArray(n);
        for (int i = 0; i < n; i++) {
            a.set(i, new HalfFloat((r.nextFloat() * 2 - 1) * scale));
        }
        return a;
    }

    static FloatArray randomFloat(int n, float scale, long seed) {
        Random r = new Random(seed);
        FloatArray a = new FloatArray(n);
        for (int i = 0; i < n; i++) {
            a.set(i, (r.nextFloat() * 2 - 1) * scale);
        }
        return a;
    }

    /** GGUF Q4_0 (18 bytes per 32 weights) or Q8_0 (34 bytes) blocks with random quants and a small fp16 scale. */
    static ByteArray randomGguf(int rows, int cols, int blockBytes, long seed) {
        int blocks = rows * cols / 32;
        ByteArray a = new ByteArray(blocks * blockBytes);
        Random r = new Random(seed);
        short scale = new HalfFloat(0.01f).getHalfFloatValue();
        for (int b = 0; b < blocks; b++) {
            int base = b * blockBytes;
            a.set(base, (byte) (scale & 0xFF));
            a.set(base + 1, (byte) ((scale >> 8) & 0xFF));
            for (int i = 2; i < blockBytes; i++) {
                a.set(base + i, (byte) r.nextInt(256));
            }
        }
        return a;
    }

    // ---------------------------------------------------------------- benchmarks

    private static void gemvF16(int n, int k) {
        String shape = n + "x" + k;
        HalfFloatArray w = randomHalf(n * k, 0.02f, 1);
        HalfFloatArray x = randomHalf(k, 1f, 2);
        HalfFloatArray yMlx = new HalfFloatArray(n);
        FloatArray yJit = new FloatArray(n);
        double bytes = 2.0 * n * k + 2.0 * k + 2.0 * n;
        double flops = 2.0 * n * k;
        MemorySegment mw = mlxCopy(w, MLX_FLOAT16, n, k);
        MemorySegment mx = mlxCopy(x, MLX_FLOAT16, 1, k);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment wt = op(arena, res -> MlxC.mlx_transpose(res, mw, MlxC.mlx_default_gpu_stream_new()));
            measureMlxAlone("GEMV fp16", shape, "f16", (s, a) -> op(a, res -> MlxC.mlx_matmul(res, mx, wt, s)), bytes, flops);
            MlxC.mlx_array_free(wt);
        }
        MlxC.mlx_array_free(mw);
        MlxC.mlx_array_free(mx);
        measure("GEMV fp16", shape, "f16", "apple/mlx", new Object[] { w, x }, new Object[] { yMlx }, (g, grid, gn, t) -> g.libraryTask(t, Mlx::matmulTransposed, x, w, yMlx, 1, k, n), false,
                bytes, flops, "");
        measure("GEMV fp16", shape, "f16", "JIT", new Object[] { w, x }, new Object[] { yJit }, (g, grid, gn, t) -> {
            WorkerGrid1D wg = new WorkerGrid1D(n * 32);
            wg.setLocalWork(32, 1, 1);
            grid.addWorkerGrid(gn + "." + t, wg);
            g.task(t, JitKernels::gemvF16Simd32, new KernelContext(), x, yJit, w, k, n);
        }, true, bytes, flops, "jitLLM matrixVectorGenericSimd32 (fp32 output)");
    }

    /** Quantized GEMV: MLX affine (group 32) against jitLLM's GGUF Q4_0 / Q8_0 kernels. */
    private static void gemvQuantized(int n, int k, int bits) {
        String shape = n + "x" + k;
        String dtype = "q" + bits;
        final int groupSize = 32;
        HalfFloatArray w = randomHalf(n * k, 0.02f, 3);
        HalfFloatArray x = randomHalf(k, 1f, 4);
        IntArray wq = new IntArray(n * k * bits / 32);
        HalfFloatArray scales = new HalfFloatArray(n * k / groupSize);
        HalfFloatArray biases = new HalfFloatArray(n * k / groupSize);
        HalfFloatArray yMlx = new HalfFloatArray(n);
        // Quantize once with MLX itself.
        TaskGraph quantize = new TaskGraph("quantize" + (graphCounter++)).transferToDevice(DataTransferMode.FIRST_EXECUTION, w)
                .libraryTask("q", Mlx::quantize, w, wq, scales, biases, n, k, groupSize, bits).transferToHost(DataTransferMode.EVERY_EXECUTION, wq, scales, biases);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(quantize.snapshot())) {
            plan.execute();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        double mlxBytes = n * (double) k * bits / 8 + 2.0 * 2 * n * k / groupSize + 2.0 * k + 2.0 * n;
        double flops = 2.0 * n * k;
        MemorySegment mq = mlxCopy(wq, MLX_UINT32, n, k * bits / 32);
        MemorySegment ms = mlxCopy(scales, MLX_FLOAT16, n, k / groupSize);
        MemorySegment mb = mlxCopy(biases, MLX_FLOAT16, n, k / groupSize);
        MemorySegment mx = mlxCopy(x, MLX_FLOAT16, 1, k);
        measureMlxAlone("GEMV " + dtype, shape, dtype, (s, a) -> op(a, res -> MlxC.mlx_quantized_matmul(res, mx, mq, ms, mb, true, optionalInt(a, groupSize), optionalInt(a, bits),
                FFMSupport.allocateCString(a, "affine"), s)), mlxBytes, flops);
        for (MemorySegment m : new MemorySegment[] { mq, ms, mb, mx }) {
            MlxC.mlx_array_free(m);
        }
        measure("GEMV " + dtype, shape, dtype, "apple/mlx", new Object[] { x, wq, scales, biases }, new Object[] { yMlx }, (g, grid, gn, t) -> g.libraryTask(t, Mlx::quantizedMatmul, x, wq,
                scales, biases, yMlx, 1, k, n, groupSize, bits), false, mlxBytes, flops, "MLX affine, group 32, fp16 scale+bias");

        int blockBytes = bits == 4 ? 18 : 34;
        ByteArray gguf = randomGguf(n, k, blockBytes, 5);
        FloatArray xf = randomFloat(k, 1f, 6);
        FloatArray yJit = new FloatArray(n);
        double jitBytes = n * (double) k / 32 * blockBytes + 4.0 * k + 4.0 * n;
        if (bits == 4) {
            measure("GEMV " + dtype, shape, dtype, "JIT", new Object[] { gguf, xf }, new Object[] { yJit }, (g, grid, gn, t) -> {
                WorkerGrid1D wg = new WorkerGrid1D(n * 32);
                wg.setLocalWork(32, 1, 1);
                grid.addWorkerGrid(gn + "." + t, wg);
                g.task(t, JitKernels::gemvQ4Simd32, new KernelContext(), xf, yJit, gguf, k, n);
            }, true, jitBytes, flops, "jitLLM matrixVectorGenericQ4_0Simd32, GGUF Q4_0");
        } else {
            final int local = 128;
            measure("GEMV " + dtype, shape, dtype, "JIT", new Object[] { gguf, xf }, new Object[] { yJit }, (g, grid, gn, t) -> {
                WorkerGrid1D wg = new WorkerGrid1D(n * local);
                wg.setLocalWork(local, 1, 1);
                grid.addWorkerGrid(gn + "." + t, wg);
                g.task(t, JitKernels::gemvQ8, new KernelContext(), xf, yJit, gguf, k, n, local);
            }, true, jitBytes, flops, "jitLLM matrixVectorGenericQ8Byte, GGUF Q8_0");
        }
    }

    private static void gemmF16(int m, int k, int n) {
        String shape = m + "x" + k + "x" + n;
        HalfFloatArray a = randomHalf(m * k, 1f, 7);
        HalfFloatArray w = randomHalf(n * k, 0.02f, 8);
        HalfFloatArray cMlx = new HalfFloatArray(m * n);
        FloatArray cJit = new FloatArray(m * n);
        double bytes = 2.0 * (m * (double) k + n * (double) k + m * (double) n);
        double flops = 2.0 * m * k * n;
        MemorySegment ma = mlxCopy(a, MLX_FLOAT16, m, k);
        MemorySegment mw = mlxCopy(w, MLX_FLOAT16, n, k);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment wt = op(arena, res -> MlxC.mlx_transpose(res, mw, MlxC.mlx_default_gpu_stream_new()));
            measureMlxAlone("GEMM fp16", shape, "f16", (s, ar) -> op(ar, res -> MlxC.mlx_matmul(res, ma, wt, s)), bytes, flops);
            MlxC.mlx_array_free(wt);
        }
        MlxC.mlx_array_free(ma);
        MlxC.mlx_array_free(mw);
        measure("GEMM fp16", shape, "f16", "apple/mlx", new Object[] { a, w }, new Object[] { cMlx }, (g, grid, gn, t) -> g.libraryTask(t, Mlx::matmulTransposed, a, w, cMlx, m, k, n), false,
                bytes, flops, "");
        measure("GEMM fp16", shape, "f16", "JIT", new Object[] { a, w }, new Object[] { cJit }, (g, grid, gn, t) -> {
            WorkerGrid2D wg = new WorkerGrid2D(n, m);
            wg.setLocalWork(JitKernels.TILE, JitKernels.TILE, 1);
            grid.addWorkerGrid(gn + "." + t, wg);
            g.task(t, JitKernels::gemmTiled, new KernelContext(), a, w, cJit, m, k, n);
        }, true, bytes, flops, "16x16 tiled KernelContext GEMM (not jitLLM-tuned)");
    }

    private static void rmsNorm(int rows, int dim) {
        String shape = rows + "x" + dim;
        FloatArray x = randomFloat(rows * dim, 1f, 9);
        FloatArray w = randomFloat(dim, 1f, 10);
        FloatArray out = new FloatArray(rows * dim);
        double bytes = 4.0 * (2.0 * rows * dim + dim);
        double flops = 4.0 * rows * dim;
        MemorySegment mx = mlxCopy(x, MLX_FLOAT32, rows, dim);
        MemorySegment mw = mlxCopy(w, MLX_FLOAT32, dim);
        measureMlxAlone("RMS norm", shape, "f32", (s, a) -> op(a, res -> MlxC.mlx_fast_rms_norm(res, mx, mw, 1e-5f, s)), bytes, flops);
        MlxC.mlx_array_free(mx);
        MlxC.mlx_array_free(mw);
        measure("RMS norm", shape, "f32", "apple/mlx", new Object[] { x, w }, new Object[] { out }, (g, grid, gn, t) -> g.libraryTask(t, Mlx::rmsNorm, x, w, out, rows, dim, 1e-5f), false,
                bytes, flops, "");
        if (rows == 1) {
            FloatArray temp = new FloatArray(1);
            final int local = 1024;
            measure("RMS norm", shape, "f32", "JIT", new Object[] { x, w }, new Object[] { out }, (g, grid, gn, t) -> {
                WorkerGrid1D reduce = new WorkerGrid1D(local);
                reduce.setLocalWork(local, 1, 1);
                grid.addWorkerGrid(gn + "." + t + "r", reduce);
                WorkerGrid1D apply = new WorkerGrid1D(dim);
                apply.setLocalWork(256, 1, 1);
                grid.addWorkerGrid(gn + "." + t + "a", apply);
                g.task(t + "r", JitKernels::rmsReduce, new KernelContext(), temp, x, dim, 1e-5f, local);
                g.task(t + "a", JitKernels::rmsApply, new KernelContext(), out, x, w, temp);
            }, true, bytes, flops, "jitLLM reductionOneBlockWithLayerSingleGroup + reductionOneBlock2WithLayer (2 kernels)");
        }
    }

    private static void rope(int heads, int headDim, int seqLen) {
        String shape = heads + "x" + seqLen + "x" + headDim;
        final int position = 100;
        final float base = 500000f;
        FloatArray x = randomFloat(heads * seqLen * headDim, 1f, 11);
        FloatArray out = new FloatArray(x.getSize());
        double bytes = 8.0 * x.getSize();
        double flops = 6.0 * x.getSize();
        MemorySegment mx = mlxCopy(x, MLX_FLOAT32, 1, heads, seqLen, headDim);
        measureMlxAlone("RoPE", shape, "f32", (s, a) -> op(a, res -> MlxC.mlx_fast_rope(res, mx, headDim, false, optionalFloat(a, base), 1f, position, MemorySegment.NULL, s)), bytes, flops);
        MlxC.mlx_array_free(mx);
        measure("RoPE", shape, "f32", "apple/mlx", new Object[] { x }, new Object[] { out }, (g, grid, gn, t) -> g.libraryTask(t, Mlx::rope, x, out, 1, heads, seqLen, headDim, headDim, false,
                base, 1f, position), false, bytes, flops, "");
        final int rows = heads * seqLen;
        measure("RoPE", shape, "f32", "JIT", new Object[] { x }, new Object[] { out }, (g, grid, gn, t) -> {
            int pairs = rows * headDim / 2;
            WorkerGrid1D wg = new WorkerGrid1D((pairs + 255) / 256 * 256);
            wg.setLocalWork(256, 1, 1);
            grid.addWorkerGrid(gn + "." + t, wg);
            g.task(t, JitFast::rope, new KernelContext(), x, out, rows, seqLen, headDim, headDim, 0, base, 1f, position);
        }, true, bytes, flops, "KernelContext kernel, one thread per pair (mlx.jit.JitFast#rope)");
    }

    private static void attentionDecode(int ctx) {
        final int qHeads = 32;
        final int kvHeads = 8;
        final int headDim = 128;
        String shape = "ctx " + ctx;
        final float scale = (float) (1.0 / Math.sqrt(headDim));
        FloatArray q = randomFloat(qHeads * headDim, 1f, 12);
        FloatArray k = randomFloat(kvHeads * ctx * headDim, 1f, 13);
        FloatArray v = randomFloat(kvHeads * ctx * headDim, 1f, 14);
        FloatArray out = new FloatArray(qHeads * headDim);
        double bytes = 4.0 * (2.0 * kvHeads * ctx * headDim + 2.0 * qHeads * headDim);
        double flops = 4.0 * qHeads * ctx * headDim;
        MemorySegment mq = mlxCopy(q, MLX_FLOAT32, 1, qHeads, 1, headDim);
        MemorySegment mk = mlxCopy(k, MLX_FLOAT32, 1, kvHeads, ctx, headDim);
        MemorySegment mv = mlxCopy(v, MLX_FLOAT32, 1, kvHeads, ctx, headDim);
        measureMlxAlone("SDPA decode", shape, "f32", (s, a) -> op(a, res -> MlxC.mlx_fast_scaled_dot_product_attention(res, mq, mk, mv, scale, FFMSupport.allocateCString(a, ""),
                MemorySegment.NULL, MemorySegment.NULL, s)), bytes, flops);
        for (MemorySegment m : new MemorySegment[] { mq, mk, mv }) {
            MlxC.mlx_array_free(m);
        }
        measure("SDPA decode", shape, "f32", "apple/mlx", new Object[] { q, k, v }, new Object[] { out }, (g, grid, gn, t) -> g.libraryTask(t, Mlx::scaledDotProductAttention, q, k, v, out, 1,
                qHeads, kvHeads, 1, ctx, headDim, scale, false), false, bytes, flops, "32 query / 8 KV heads, head dim 128");
        measure("SDPA decode", shape, "f32", "JIT", new Object[] { q, k, v }, new Object[] { out }, (g, grid, gn, t) -> {
            WorkerGrid1D wg = new WorkerGrid1D(qHeads * headDim);
            wg.setLocalWork(headDim, 1, 1);
            grid.addWorkerGrid(gn + "." + t, wg);
            g.task(t, JitKernels::attentionDecode, new KernelContext(), q, k, v, out, qHeads, kvHeads, ctx, headDim, scale);
        }, true, bytes, flops, "online-softmax KernelContext kernel (jitLLM's is paged-KV)");
    }

    private static void logits(int vocab) {
        String shape = "vocab " + vocab;
        FloatArray x = randomFloat(vocab, 10f, 15);
        FloatArray probs = new FloatArray(vocab);
        IntArray index = new IntArray(1);
        final int k = 50;
        FloatArray top = new FloatArray(k);
        final int local = 1024;
        MemorySegment mx = mlxCopy(x, MLX_FLOAT32, vocab);

        measureMlxAlone("softmax", shape, "f32", (s, a) -> op(a, res -> MlxC.mlx_softmax(res, mx, true, s)), 8.0 * vocab, 5.0 * vocab);
        measure("softmax", shape, "f32", "apple/mlx", new Object[] { x }, new Object[] { probs }, (g, grid, gn, t) -> g.libraryTask(t, Mlx::softmax, x, probs), false, 8.0 * vocab, 5.0 * vocab, "");
        measure("softmax", shape, "f32", "JIT", new Object[] { x }, new Object[] { probs }, (g, grid, gn, t) -> {
            WorkerGrid1D wg = new WorkerGrid1D(local);
            wg.setLocalWork(local, 1, 1);
            grid.addWorkerGrid(gn + "." + t, wg);
            g.task(t, JitKernels::softmax, new KernelContext(), x, probs, vocab, local);
        }, true, 8.0 * vocab, 5.0 * vocab, "one-workgroup KernelContext kernel");

        measureMlxAlone("argmax", shape, "f32", (s, a) -> op(a, res -> MlxC.mlx_argmax(res, mx, false, s)), 4.0 * vocab, vocab);
        measure("argmax", shape, "f32", "apple/mlx", new Object[] { x }, new Object[] { index }, (g, grid, gn, t) -> g.libraryTask(t, Mlx::argmax, x, index), false, 4.0 * vocab, vocab, "");
        measure("argmax", shape, "f32", "JIT", new Object[] { x }, new Object[] { index }, (g, grid, gn, t) -> {
            WorkerGrid1D wg = new WorkerGrid1D(local);
            wg.setLocalWork(local, 1, 1);
            grid.addWorkerGrid(gn + "." + t, wg);
            g.task(t, JitKernels::argmax, new KernelContext(), x, index, vocab, local);
        }, true, 4.0 * vocab, vocab, "jitLLM argmaxLogits");

        measureMlxAlone("top-k (k=50)", shape, "f32", (s, a) -> op(a, res -> MlxC.mlx_topk(res, mx, k, s)), 4.0 * vocab, vocab);
        measure("top-k (k=50)", shape, "f32", "apple/mlx", new Object[] { x }, new Object[] { top }, (g, grid, gn, t) -> g.libraryTask(t, Mlx::topk, x, top, k), false, 4.0 * vocab, vocab,
                "no JIT top-k to compare against");
        MlxC.mlx_array_free(mx);
    }

    // ---------------------------------------------------------------- report

    private static String fmt(double v) {
        return Double.isNaN(v) ? "n/a" : String.format(Locale.ROOT, "%.1f", v);
    }

    private static void writeReport(PrintStream out) {
        out.println("| op | shape | dtype | variant | execute() median us (p10-p90) | marginal us | GB/s (% of 200) | GFLOP/s | note |");
        out.println("|---|---|---|---|---|---|---|---|---|");
        for (Row r : ROWS) {
            String execute = r.variant().equals("MLX alone") ? fmt(r.execute().medianUs()) + " (call)"
                    : fmt(r.execute().medianUs()) + " (" + fmt(r.execute().p10Us()) + "-" + fmt(r.execute().p90Us()) + ")";
            out.printf(Locale.ROOT, "| %s | %s | %s | %s | %s | %s | %s (%s%%) | %s | %s |%n", r.op(), r.shape(), r.dtype(), r.variant(), execute, fmt(r.marginalUs()), fmt(r.gbs()),
                    fmt(100 * r.gbs() / PEAK_GBS), fmt(r.gflops()), r.note());
        }
        out.println();
        out.println("Ops where the JIT's marginal cost is more than 1.5x apple/mlx's (JIT codegen to-do list, worst first):");
        out.println();
        out.println("| op | shape | JIT marginal us | apple/mlx marginal us | JIT / MLX |");
        out.println("|---|---|---|---|---|");
        List<Row[]> pairs = new ArrayList<>();
        for (Row jit : ROWS) {
            if (!jit.variant().equals("JIT")) {
                continue;
            }
            for (Row mlx : ROWS) {
                if (mlx.variant().equals("apple/mlx") && mlx.op().equals(jit.op()) && mlx.shape().equals(jit.shape())) {
                    pairs.add(new Row[] { jit, mlx });
                }
            }
        }
        pairs.sort((p1, p2) -> Double.compare(p2[0].marginalUs() / p2[1].marginalUs(), p1[0].marginalUs() / p1[1].marginalUs()));
        for (Row[] p : pairs) {
            double ratio = p[0].marginalUs() / p[1].marginalUs();
            if (ratio > 1.5) {
                out.printf(Locale.ROOT, "| %s | %s | %s | %s | %.2fx |%n", p[0].op(), p[0].shape(), fmt(p[0].marginalUs()), fmt(p[1].marginalUs()), ratio);
            }
        }
    }

    public static void main(String[] args) throws IOException {
        boolean quick = Arrays.asList(args).contains("--quick");
        int outIndex = Arrays.asList(args).indexOf("--out");
        if (!MlxLibraryProvider.isAvailable()) {
            System.err.println("mlx-c is not available; install it first (brew install mlx-c)");
            System.exit(1);
        }
        System.out.println("MLX " + MlxLibraryProvider.mlxVersion() + ", mlx-c bindings " + MlxC.MLX_C_VERSION + (quick ? " (quick)" : ""));

        int[] hidden = quick ? new int[] { 2048 } : new int[] { 2048, 3072, 4096 };
        for (int h : hidden) {
            gemvF16(h, h);
        }
        gemvF16(quick ? 32000 : 128256, 2048);
        if (!quick) {
            gemvF16(151936, 2048);
        }
        for (int h : hidden) {
            gemvQuantized(h, h, 4);
            gemvQuantized(h, h, 8);
        }
        gemvQuantized(quick ? 32000 : 128256, 2048, 4);
        for (int h : quick ? new int[] { 2048 } : new int[] { 2048, 4096 }) {
            gemmF16(quick ? 128 : 512, h, h);
        }
        for (int h : hidden) {
            rmsNorm(1, h);
        }
        rmsNorm(quick ? 128 : 512, 4096);
        rope(32, 128, 1);
        rope(32, 128, quick ? 128 : 512);
        for (int ctx : quick ? new int[] { 512 } : new int[] { 512, 2048, 8192 }) {
            attentionDecode(ctx);
        }
        logits(quick ? 32000 : 151936);

        writeReport(System.out);
        if (outIndex >= 0 && outIndex + 1 < args.length) {
            try (PrintStream ps = new PrintStream(Files.newOutputStream(Path.of(args[outIndex + 1])))) {
                writeReport(ps);
            }
            System.out.println("Wrote " + args[outIndex + 1]);
        }
        System.exit(0);
    }
}
