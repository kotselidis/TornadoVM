package uk.ac.manchester.tornado.mlx.spike;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.common.Access;
import uk.ac.manchester.tornado.api.common.LibraryTaskDescriptor;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * M0 step 5: decode-shape LLM ops, three ways (MLX alone, apple/mlx on TornadoVM buffers,
 * TornadoVM JIT). Each op is checked against a Java reference, then timed per execute() and as the
 * marginal cost of one more task in a chain inside one TaskGraph.
 */
public final class DecodeOps {

    private DecodeOps() {
    }

    static final int ROWS = 4096, COLS = 4096, GROUP = 32;
    static final int HEADS = 32, HEAD_DIM = 128, ROPE_OFFSET = 123;
    static final float ROPE_BASE = 500000.0f;
    static final int LOCAL = 128;
    static final int ITERS = 200, CHAIN = 16, CHAIN_ITERS = 100;

    // ---------------------------------------------------------------- library bindings

    private static Access[] readOnlyExcept(int n, int out) {
        Access[] a = new Access[n];
        Arrays.fill(a, Access.READ_ONLY);
        a[out] = Access.WRITE_ONLY;
        return a;
    }

    public static LibraryTaskDescriptor gemvLib(HalfFloatArray w, HalfFloatArray x, HalfFloatArray y, int rows, int cols) {
        return new LibraryTaskDescriptor().withLibrary(MlxSpikeProvider.NAME).withFunction("gemv_f16") //
                .withParameters(new Object[] { w, x, y, rows, cols }).withAccess(readOnlyExcept(5, 2));
    }

    public static LibraryTaskDescriptor qmvLib(IntArray wq, HalfFloatArray scales, HalfFloatArray biases, HalfFloatArray x, HalfFloatArray y, int rows, int cols, int gs) {
        return new LibraryTaskDescriptor().withLibrary(MlxSpikeProvider.NAME).withFunction("qmv4") //
                .withParameters(new Object[] { wq, scales, biases, x, y, rows, cols, gs }).withAccess(readOnlyExcept(8, 4));
    }

    public static LibraryTaskDescriptor ropeLib(HalfFloatArray x, HalfFloatArray y, int heads, int dim, int offset, float base) {
        return new LibraryTaskDescriptor().withLibrary(MlxSpikeProvider.NAME).withFunction("rope") //
                .withParameters(new Object[] { x, y, heads, dim, offset, base }).withAccess(readOnlyExcept(6, 1));
    }

    // ---------------------------------------------------------------- JIT kernels

    /** One threadgroup per output row; float accumulation, tree reduction in threadgroup memory. */
    public static void gemvJit(KernelContext ctx, HalfFloatArray w, HalfFloatArray x, HalfFloatArray y, int cols) {
        int row = ctx.groupIdx;
        int lid = ctx.localIdx;
        int lsz = ctx.localGroupSizeX;
        float[] partial = ctx.allocateFloatLocalArray(LOCAL);
        int base = row * cols;
        float acc = 0.0f;
        for (int j = lid; j < cols; j += lsz) {
            acc += w.get(base + j).getFloat32() * x.get(j).getFloat32();
        }
        partial[lid] = acc;
        for (int stride = lsz / 2; stride > 0; stride >>= 1) {
            ctx.localBarrier();
            if (lid < stride) {
                partial[lid] += partial[lid + stride];
            }
        }
        if (lid == 0) {
            y.set(row, new HalfFloat(partial[0]));
        }
    }

    /** MLX 4-bit affine layout: 8 values per uint32, low nibble first; w = scale * q + bias per group. */
    public static void qmvJit(KernelContext ctx, IntArray wq, HalfFloatArray scales, HalfFloatArray biases, HalfFloatArray x, HalfFloatArray y, int cols, int gs) {
        int row = ctx.groupIdx;
        int lid = ctx.localIdx;
        int lsz = ctx.localGroupSizeX;
        float[] partial = ctx.allocateFloatLocalArray(LOCAL);
        int packedPerRow = cols / 8;
        int groupsPerRow = cols / gs;
        float acc = 0.0f;
        for (int p = lid; p < packedPerRow; p += lsz) {
            int word = wq.get(row * packedPerRow + p);
            int col0 = p * 8;
            int g = row * groupsPerRow + col0 / gs;
            float s = scales.get(g).getFloat32();
            float b = biases.get(g).getFloat32();
            for (int k = 0; k < 8; k++) {
                // Signed >> on purpose: the Metal backend lowers >>> to MSL's arithmetic shift, and
                // Graal drops the mask for k = 7, so >>> gives wrong results today.
                int q = (word >> (4 * k)) & 0xF;
                acc += (s * q + b) * x.get(col0 + k).getFloat32();
            }
        }
        partial[lid] = acc;
        for (int stride = lsz / 2; stride > 0; stride >>= 1) {
            ctx.localBarrier();
            if (lid < stride) {
                partial[lid] += partial[lid + stride];
            }
        }
        if (lid == 0) {
            y.set(row, new HalfFloat(partial[0]));
        }
    }

    /** Half-split RoPE at one position: pairs (i, i + dim/2) rotated by offset * base^(-2i/dim). */
    public static void ropeJit(HalfFloatArray x, HalfFloatArray y, int heads, int dim, int offset, float base) {
        int half = dim / 2;
        for (@Parallel int idx = 0; idx < heads * half; idx++) {
            int h = idx / half;
            int i = idx % half;
            float theta = offset * TornadoMath.pow(base, -2.0f * i / dim);
            float c = TornadoMath.cos(theta);
            float s = TornadoMath.sin(theta);
            int p = h * dim + i;
            float x1 = x.get(p).getFloat32();
            float x2 = x.get(p + half).getFloat32();
            y.set(p, new HalfFloat(x1 * c - x2 * s));
            y.set(p + half, new HalfFloat(x1 * s + x2 * c));
        }
    }

    // ---------------------------------------------------------------- harness

    /** Adds one task reading {@code in} and writing {@code out}. */
    private interface TaskAdder {
        void add(TaskGraph tg, GridScheduler grid, String graphName, String taskName, HalfFloatArray in, HalfFloatArray out);
    }

    private static int failures;
    private static final List<String> TABLE = new ArrayList<>();

    private static double median(long[] t) {
        Arrays.sort(t);
        return t[t.length / 2] / 1e3;
    }

    /**
     * Builds a graph of {@code chain} tasks ping-ponging between a and b; returns the median
     * execute() time in us. With {@code check != null}, runs once and returns the output instead.
     */
    private static double run(Object[] constants, HalfFloatArray a, HalfFloatArray b, TaskAdder adder, boolean jit, int chain, int iters, HalfFloatArray[] resultOut) {
        String gname = "g" + chain;
        TaskGraph tg = new TaskGraph(gname);
        Object[] toDevice = Arrays.copyOf(constants, constants.length + 2);
        toDevice[constants.length] = a;
        toDevice[constants.length + 1] = b;
        tg.transferToDevice(DataTransferMode.FIRST_EXECUTION, toDevice);
        GridScheduler grid = new GridScheduler();
        for (int i = 0; i < chain; i++) {
            adder.add(tg, grid, gname, "t" + i, i % 2 == 0 ? a : b, i % 2 == 0 ? b : a);
        }
        tg.transferToHost(DataTransferMode.UNDER_DEMAND, a, b);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(tg.snapshot())) {
            if (jit && !grid.keySet().isEmpty()) {
                plan.withGridScheduler(grid);
            }
            if (resultOut != null) {
                plan.execute().transferToHost(b);
                resultOut[0] = b;
                return 0;
            }
            long[] t = new long[iters];
            for (int i = -20; i < iters; i++) {
                if (i == 0) {
                    MlxSpikeProvider.STATS.reset();
                }
                long t0 = System.nanoTime();
                plan.execute();
                if (i >= 0) {
                    t[i] = System.nanoTime() - t0;
                }
            }
            return median(t);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private static void check(String label, HalfFloatArray out, float[] ref, float tol) {
        int bad = 0;
        double maxErr = 0;
        for (int i = 0; i < ref.length; i++) {
            float err = Math.abs(out.get(i).getFloat32() - ref[i]);
            maxErr = Math.max(maxErr, err);
            if (err > tol * Math.max(1f, Math.abs(ref[i]))) {
                if (bad++ < 2) {
                    System.out.printf("    %s out[%d] = %f, expected %f%n", label, i, out.get(i).getFloat32(), ref[i]);
                }
            }
        }
        System.out.printf("  %s  %-34s max abs error %.2e%n", bad == 0 ? "PASS" : "FAIL", label, maxErr);
        if (bad != 0) {
            failures++;
        }
    }

    private interface MlxAlone {
        MemorySegment step(MemorySegment stream, Arena call) throws Throwable;
    }

    private static double mlxAlone(MlxAlone op) throws Throwable {
        MemorySegment stream = MlxC.gpuStream();
        long[] t = new long[ITERS];
        for (int i = -20; i < ITERS; i++) {
            long t0 = System.nanoTime();
            try (Arena call = Arena.ofConfined()) {
                MemorySegment y = op.step(stream, call);
                MlxC.eval(y);
                MlxC.free(y);
            }
            if (i >= 0) {
                t[i] = System.nanoTime() - t0;
            }
        }
        return median(t);
    }

    private static void measure(String op, String shape, double boundUs, Object[] constants, HalfFloatArray a, HalfFloatArray b, TaskAdder mlx, TaskAdder jit, float[] ref, float tol,
            double aloneUs) {
        HalfFloatArray[] out = new HalfFloatArray[1];
        run(constants, a, b, mlx, false, 1, 0, out);
        check(op + " apple/mlx", out[0], ref, tol);
        run(constants, a, b, jit, true, 1, 0, out);
        check(op + " TornadoVM JIT", out[0], ref, tol);

        double mlxExec = run(constants, a, b, mlx, false, 1, ITERS, null);
        MlxSpikeProvider.Stats s = MlxSpikeProvider.STATS;
        String split = String.format("wrap %.1f / MLX %.1f / copy %.1f", s.wrapNanos / 1e3 / s.calls, s.computeNanos / 1e3 / s.calls, s.copyOutNanos / 1e3 / s.calls);
        double jitExec = run(constants, a, b, jit, true, 1, ITERS, null);
        double mlxMarg = (run(constants, a, b, mlx, false, CHAIN, CHAIN_ITERS, null) - run(constants, a, b, mlx, false, 1, CHAIN_ITERS, null)) / (CHAIN - 1);
        double jitMarg = (run(constants, a, b, jit, true, CHAIN, CHAIN_ITERS, null) - run(constants, a, b, jit, true, 1, CHAIN_ITERS, null)) / (CHAIN - 1);
        TABLE.add(String.format("| %s | %s | %.0f | %.0f | %.0f (%s) | %.0f | %.0f | %.0f |", op, shape, boundUs, aloneUs, mlxExec, split, jitExec, mlxMarg, jitMarg));
    }

    private static HalfFloatArray randomHalf(int n, float scale, Random r) {
        HalfFloatArray a = new HalfFloatArray(n);
        for (int i = 0; i < n; i++) {
            a.set(i, new HalfFloat((r.nextFloat() * 2 - 1) * scale));
        }
        return a;
    }

    /** M1 Pro unified-memory bandwidth used for the bytes-moved lower bound. */
    static final double PEAK_GBS = 200.0;

    private static double boundUs(double bytes) {
        return bytes / (PEAK_GBS * 1e3);
    }

    // ---------------------------------------------------------------- ops

    private static void gemv() throws Throwable {
        Random r = new Random(1);
        HalfFloatArray w = randomHalf(ROWS * COLS, 0.05f, r);
        HalfFloatArray x = randomHalf(COLS, 1.0f, r), y = new HalfFloatArray(ROWS);
        float[] ref = new float[ROWS];
        for (int i = 0; i < ROWS; i++) {
            float acc = 0;
            for (int j = 0; j < COLS; j++) {
                acc += w.get(i * COLS + j).getFloat32() * x.get(j).getFloat32();
            }
            ref[i] = acc;
        }
        double alone;
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment mw = MlxC.fromSegment(w.getSegment(), new int[] { ROWS, COLS }, MlxC.MLX_FLOAT16, arena);
            MemorySegment mx = MlxC.fromSegment(x.getSegment(), new int[] { COLS, 1 }, MlxC.MLX_FLOAT16, arena);
            MlxC.eval(mw);
            MlxC.eval(mx);
            alone = mlxAlone((st, call) -> MlxC.matmul(mw, mx, st, call));
        }
        TaskAdder mlx = (tg, grid, g, t, in, out) -> tg.libraryTask(t, DecodeOps::gemvLib, w, in, out, ROWS, COLS);
        TaskAdder jit = (tg, grid, g, t, in, out) -> {
            WorkerGrid1D wg = new WorkerGrid1D(ROWS * LOCAL);
            wg.setLocalWork(LOCAL, 1, 1);
            grid.addWorkerGrid(g + "." + t, wg);
            tg.task(t, DecodeOps::gemvJit, new KernelContext(), w, in, out, COLS);
        };
        measure("GEMV fp16", "4096x4096", boundUs(2.0 * ROWS * COLS), new Object[] { w }, x, y, mlx, jit, ref, 2e-2f, alone);
    }

    private static void qmv() throws Throwable {
        Random r = new Random(2);
        HalfFloatArray wFull = randomHalf(ROWS * COLS, 0.05f, r);
        HalfFloatArray x = randomHalf(COLS, 1.0f, r), y = new HalfFloatArray(ROWS);
        IntArray wq = new IntArray(ROWS * COLS / 8);
        HalfFloatArray scales = new HalfFloatArray(ROWS * COLS / GROUP), biases = new HalfFloatArray(ROWS * COLS / GROUP);

        // Quantize with MLX itself, then copy the packed weights, scales and biases into TornadoVM arrays.
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment st = MlxC.gpuStream();
            MemorySegment mw = MlxC.fromSegment(wFull.getSegment(), new int[] { ROWS, COLS }, MlxC.MLX_FLOAT16, arena);
            MemorySegment[] q = MlxC.quantize(mw, GROUP, 4, st, arena);
            MemorySegment[] dst = { wq.getSegment(), scales.getSegment(), biases.getSegment() };
            for (int i = 0; i < 3; i++) {
                MlxC.eval(q[i]);
                long bytes = MlxC.nbytes(q[i]);
                if (bytes != dst[i].byteSize()) {
                    throw new IllegalStateException("quantize output " + i + " is " + bytes + " bytes, expected " + dst[i].byteSize());
                }
                dst[i].copyFrom(MemorySegment.ofAddress(MlxC.rawAddress(q[i])).reinterpret(bytes));
            }
        }

        // Reference from the packed data, assuming MLX's layout (low nibble first).
        int ppr = COLS / 8, gpr = COLS / GROUP;
        float[] ref = new float[ROWS];
        for (int i = 0; i < ROWS; i++) {
            float acc = 0;
            for (int p = 0; p < ppr; p++) {
                int word = wq.get(i * ppr + p);
                int g = i * gpr + (p * 8) / GROUP;
                float s = scales.get(g).getFloat32(), b = biases.get(g).getFloat32();
                for (int k = 0; k < 8; k++) {
                    acc += (s * ((word >>> (4 * k)) & 0xF) + b) * x.get(p * 8 + k).getFloat32();
                }
            }
            ref[i] = acc;
        }

        double alone;
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment mq = MlxC.fromSegment(wq.getSegment(), new int[] { ROWS, ppr }, MlxC.MLX_UINT32, arena);
            MemorySegment ms = MlxC.fromSegment(scales.getSegment(), new int[] { ROWS, gpr }, MlxC.MLX_FLOAT16, arena);
            MemorySegment mb = MlxC.fromSegment(biases.getSegment(), new int[] { ROWS, gpr }, MlxC.MLX_FLOAT16, arena);
            MemorySegment mx = MlxC.fromSegment(x.getSegment(), new int[] { 1, COLS }, MlxC.MLX_FLOAT16, arena);
            for (MemorySegment m : new MemorySegment[] { mq, ms, mb, mx }) {
                MlxC.eval(m);
            }
            alone = mlxAlone((st, call) -> MlxC.quantizedMatmul(mx, mq, ms, mb, GROUP, 4, st, call));
        }
        TaskAdder mlx = (tg, grid, g, t, in, out) -> tg.libraryTask(t, DecodeOps::qmvLib, wq, scales, biases, in, out, ROWS, COLS, GROUP);
        TaskAdder jit = (tg, grid, g, t, in, out) -> {
            WorkerGrid1D wg = new WorkerGrid1D(ROWS * LOCAL);
            wg.setLocalWork(LOCAL, 1, 1);
            grid.addWorkerGrid(g + "." + t, wg);
            tg.task(t, DecodeOps::qmvJit, new KernelContext(), wq, scales, biases, in, out, COLS, GROUP);
        };
        double bytes = ROWS * COLS / 2.0 + 2.0 * 2 * ROWS * gpr;
        measure("Q4 qmv (affine, g32)", "4096x4096", boundUs(bytes), new Object[] { wq, scales, biases }, x, y, mlx, jit, ref, 2e-2f, alone);
    }

    private static void rope() throws Throwable {
        Random r = new Random(3);
        int n = HEADS * HEAD_DIM, half = HEAD_DIM / 2;
        HalfFloatArray x = randomHalf(n, 1.0f, r), y = new HalfFloatArray(n);
        float[] ref = new float[n];
        for (int h = 0; h < HEADS; h++) {
            for (int i = 0; i < half; i++) {
                double theta = ROPE_OFFSET * Math.pow(ROPE_BASE, -2.0 * i / HEAD_DIM);
                double c = Math.cos(theta), s = Math.sin(theta);
                int p = h * HEAD_DIM + i;
                float x1 = x.get(p).getFloat32(), x2 = x.get(p + half).getFloat32();
                ref[p] = (float) (x1 * c - x2 * s);
                ref[p + half] = (float) (x1 * s + x2 * c);
            }
        }
        double alone;
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment mx = MlxC.fromSegment(x.getSegment(), new int[] { 1, HEADS, 1, HEAD_DIM }, MlxC.MLX_FLOAT16, arena);
            MlxC.eval(mx);
            alone = mlxAlone((st, call) -> MlxC.rope(mx, HEAD_DIM, ROPE_BASE, ROPE_OFFSET, st, call));
        }
        TaskAdder mlx = (tg, grid, g, t, in, out) -> tg.libraryTask(t, DecodeOps::ropeLib, in, out, HEADS, HEAD_DIM, ROPE_OFFSET, ROPE_BASE);
        TaskAdder jit = (tg, grid, g, t, in, out) -> tg.task(t, DecodeOps::ropeJit, in, out, HEADS, HEAD_DIM, ROPE_OFFSET, ROPE_BASE);
        measure("rope (half-split)", "32 heads x 128", boundUs(4.0 * n), new Object[0], x, y, mlx, jit, ref, 2e-2f, alone);
    }

    public static void main(String[] args) throws Throwable {
        System.out.printf("Decode ops on %s, MLX %s%n%n", System.getProperty("os.arch"), MlxC.version());
        gemv();
        qmv();
        rope();
        System.out.println("\n| op | shape | bytes-moved bound (us, 200 GB/s) | MLX alone | apple/mlx per execute() (provider split) | JIT per execute() | apple/mlx marginal | JIT marginal |");
        System.out.println("|---|---|---|---|---|---|---|---|");
        TABLE.forEach(System.out::println);
        System.out.println(failures == 0 ? "\nDECODE OPS PASSED" : "\nDECODE OPS FAILED (" + failures + ")");
        System.exit(failures == 0 ? 0 : 1);
    }
}
