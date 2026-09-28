package uk.ac.manchester.tornado.mlx.spike;

import java.util.Arrays;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.common.Access;
import uk.ac.manchester.tornado.api.common.LibraryTaskDescriptor;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.TornadoNativeArray;

/**
 * M0 spike driver. Checks, on real TornadoVM Metal buffers:
 * <ol>
 * <li>JIT -> MLX -> JIT in one TaskGraph gives correct results with MLX reading TornadoVM memory
 * directly;</li>
 * <li>an MLX matmul matches a Java reference;</li>
 * <li>how the library call's time splits into wrapping, MLX compute and copy-out.</li>
 * </ol>
 */
public final class SpikeMain {

    private SpikeMain() {
    }

    // ---------------------------------------------------------------- library bindings

    public static LibraryTaskDescriptor affine(FloatArray x, FloatArray y, int n, float a, float b) {
        return new LibraryTaskDescriptor() //
                .withLibrary(MlxSpikeProvider.NAME) //
                .withFunction("affine") //
                .withParameters(new Object[] { x, y, n, a, b }) //
                .withAccess(new Access[] { Access.READ_ONLY, Access.WRITE_ONLY, Access.READ_ONLY, Access.READ_ONLY, Access.READ_ONLY });
    }

    public static LibraryTaskDescriptor matmul(FloatArray a, FloatArray b, FloatArray c, int m, int k, int n) {
        return new LibraryTaskDescriptor() //
                .withLibrary(MlxSpikeProvider.NAME) //
                .withFunction("matmul") //
                .withParameters(new Object[] { a, b, c, m, k, n }) //
                .withAccess(new Access[] { Access.READ_ONLY, Access.READ_ONLY, Access.WRITE_ONLY, Access.READ_ONLY, Access.READ_ONLY, Access.READ_ONLY });
    }

    private static final Access[] RMS_ACCESS = { Access.READ_ONLY, Access.READ_ONLY, Access.WRITE_ONLY, Access.READ_ONLY, Access.READ_ONLY, Access.READ_ONLY };

    public static LibraryTaskDescriptor rmsMlxc(FloatArray x, FloatArray w, FloatArray out, int rows, int dim, float eps) {
        return new LibraryTaskDescriptor().withLibrary(MlxSpikeProvider.NAME).withFunction("rms_mlxc") //
                .withParameters(new Object[] { x, w, out, rows, dim, eps }).withAccess(RMS_ACCESS);
    }

    public static LibraryTaskDescriptor rmsKernel(FloatArray x, FloatArray w, FloatArray out, int rows, int dim, float eps) {
        return new LibraryTaskDescriptor().withLibrary(MlxSpikeProvider.NAME).withFunction("rms_kernel") //
                .withParameters(new Object[] { x, w, out, rows, dim, eps }).withAccess(RMS_ACCESS);
    }

    // ---------------------------------------------------------------- JIT kernels

    static final int RMS_LOCAL = 256;

    /** One threadgroup per row: strided sum of squares, tree reduction in threadgroup memory. */
    public static void rmsNormJit(KernelContext ctx, FloatArray x, FloatArray w, FloatArray out, int dim, float eps) {
        int row = ctx.groupIdx;
        int lid = ctx.localIdx;
        int lsz = ctx.localGroupSizeX;
        float[] partial = ctx.allocateFloatLocalArray(RMS_LOCAL);
        int base = row * dim;
        float acc = 0.0f;
        for (int i = lid; i < dim; i += lsz) {
            float v = x.get(base + i);
            acc += v * v;
        }
        partial[lid] = acc;
        for (int stride = lsz / 2; stride > 0; stride >>= 1) {
            ctx.localBarrier();
            if (lid < stride) {
                partial[lid] += partial[lid + stride];
            }
        }
        ctx.localBarrier();
        float inv = 1.0f / TornadoMath.sqrt(partial[0] / dim + eps);
        for (int i = lid; i < dim; i += lsz) {
            out.set(base + i, w.get(i) * (x.get(base + i) * inv));
        }
    }

    public static void iota(FloatArray x) {
        for (@Parallel int i = 0; i < x.getSize(); i++) {
            x.set(i, i * 0.5f);
        }
    }

    public static void affineJit(FloatArray x, FloatArray y, float a, float b) {
        for (@Parallel int i = 0; i < x.getSize(); i++) {
            y.set(i, a * x.get(i) + b);
        }
    }

    public static void plusOne(FloatArray y, FloatArray z) {
        for (@Parallel int i = 0; i < y.getSize(); i++) {
            z.set(i, y.get(i) + 1.0f);
        }
    }

    // ---------------------------------------------------------------- checks

    private static int failures;

    private static void check(boolean ok, String what) {
        System.out.println((ok ? "  PASS  " : "  FAIL  ") + what);
        if (!ok) {
            failures++;
        }
    }

    private static void jitMlxJit(int n) {
        FloatArray x = new FloatArray(n), y = new FloatArray(n), z = new FloatArray(n);
        float a = 3.0f, b = -2.0f;

        TaskGraph tg = new TaskGraph("jitMlxJit") //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, x, y, z) //
                .task("iota", SpikeMain::iota, x) //
                .libraryTask("mlx", SpikeMain::affine, x, y, n, a, b) //
                .task("plusOne", SpikeMain::plusOne, y, z) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, z);

        MlxSpikeProvider.STATS.reset();
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(tg.snapshot())) {
            plan.execute();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        int bad = 0;
        for (int i = 0; i < n; i++) {
            float expected = a * (i * 0.5f) + b + 1.0f;
            if (Math.abs(z.get(i) - expected) > 1e-4f * Math.max(1f, Math.abs(expected))) {
                if (bad++ < 3) {
                    System.out.printf("    z[%d] = %f, expected %f%n", i, z.get(i), expected);
                }
            }
        }
        System.out.printf("JIT -> MLX -> JIT, n = %d%n", n);
        check(bad == 0, "results correct (" + bad + " wrong)");
        check(MlxSpikeProvider.STATS.calls == 1, "MLX provider called once");
        check(MlxSpikeProvider.STATS.inputZeroCopy, "MLX adopted TornadoVM's buffer (no copy on wrap, same pointer)");
    }

    private static void matmulCheck(int m, int k, int n) {
        FloatArray a = new FloatArray(m * k), b = new FloatArray(k * n), c = new FloatArray(m * n);
        for (int i = 0; i < m * k; i++) {
            a.set(i, (i % 7) - 3);
        }
        for (int i = 0; i < k * n; i++) {
            b.set(i, (i % 5) - 2);
        }

        TaskGraph tg = new TaskGraph("mm") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, a, b) //
                .libraryTask("mlxMatmul", SpikeMain::matmul, a, b, c, m, k, n) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, c);

        MlxSpikeProvider.STATS.reset();
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(tg.snapshot())) {
            plan.execute();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        int bad = 0;
        for (int i = 0; i < m && bad < 4; i++) {
            for (int j = 0; j < n; j++) {
                float ref = 0;
                for (int p = 0; p < k; p++) {
                    ref += a.get(i * k + p) * b.get(p * n + j);
                }
                if (Math.abs(c.get(i * n + j) - ref) > 1e-3f * Math.max(1f, Math.abs(ref))) {
                    if (bad++ < 3) {
                        System.out.printf("    c[%d,%d] = %f, expected %f%n", i, j, c.get(i * n + j), ref);
                    }
                }
            }
        }
        System.out.printf("MLX matmul %dx%d @ %dx%d%n", m, k, k, n);
        check(bad == 0, "matches Java reference");
        check(MlxSpikeProvider.STATS.inputZeroCopy, "both inputs adopted without a copy");
    }

    /** Splits the library call's time for an n-element affine op over {@code iters} executions. */
    private static void boundaryCost(int n, int iters) {
        FloatArray x = new FloatArray(n), y = new FloatArray(n);
        x.init(1.0f);
        TaskGraph tg = new TaskGraph("cost") //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, x, y) //
                .libraryTask("mlx", SpikeMain::affine, x, y, n, 2.0f, 1.0f) //
                .transferToHost(DataTransferMode.UNDER_DEMAND, y);
        ImmutableTaskGraph itg = tg.snapshot();
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(itg)) {
            for (int i = 0; i < 20; i++) {
                plan.execute();
            }
            MlxSpikeProvider.STATS.reset();
            long[] total = new long[iters];
            for (int i = 0; i < iters; i++) {
                long t = System.nanoTime();
                plan.execute();
                total[i] = System.nanoTime() - t;
            }
            Arrays.sort(total);
            MlxSpikeProvider.Stats s = MlxSpikeProvider.STATS;
            System.out.printf("  n=%-9d execute() median %8.1f us | per call: wrap %7.1f us  MLX compute+eval %8.1f us  copy-out %8.1f us (%.2f GB/s)%n", n, total[iters / 2] / 1e3,
                    s.wrapNanos / 1e3 / s.calls, s.computeNanos / 1e3 / s.calls, s.copyOutNanos / 1e3 / s.calls, (double) n * Float.BYTES * s.calls / s.copyOutNanos);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private static double median(long[] t) {
        Arrays.sort(t);
        return t[t.length / 2] / 1e3;
    }

    /** Baseline 1: MLX alone. MLX-owned arrays, no TornadoVM; the upper bound for an MLX op. */
    private static double mlxAlone(int n, int iters) throws Throwable {
        try (java.lang.foreign.Arena arena = java.lang.foreign.Arena.ofConfined()) {
            float[] ones = new float[n];
            Arrays.fill(ones, 1.0f);
            var stream = MlxC.gpuStream();
            var x = MlxC.fromFloats(ones, arena);
            var a = MlxC.scalar(2.0f);
            var b = MlxC.scalar(1.0f);
            MlxC.eval(x);
            long[] t = new long[iters];
            for (int i = -20; i < iters; i++) {
                long t0 = System.nanoTime();
                try (java.lang.foreign.Arena call = java.lang.foreign.Arena.ofConfined()) {
                    var ax = MlxC.multiply(x, a, stream, call);
                    var y = MlxC.add(ax, b, stream, call);
                    MlxC.eval(y);
                    MlxC.free(y);
                    MlxC.free(ax);
                }
                if (i >= 0) {
                    t[i] = System.nanoTime() - t0;
                }
            }
            return median(t);
        }
    }

    /** Baseline 2: the same op as a TornadoVM JIT kernel on Metal. */
    private static double jitAlone(int n, int iters) {
        FloatArray x = new FloatArray(n), y = new FloatArray(n);
        x.init(1.0f);
        TaskGraph tg = new TaskGraph("jit") //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, x, y) //
                .task("affine", SpikeMain::affineJit, x, y, 2.0f, 1.0f) //
                .transferToHost(DataTransferMode.UNDER_DEMAND, y);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(tg.snapshot())) {
            long[] t = new long[iters];
            for (int i = -20; i < iters; i++) {
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

    /** MLX alone: fast.rms_norm on MLX-owned arrays, no TornadoVM. */
    private static double mlxAloneRms(int rows, int dim, int iters) throws Throwable {
        try (java.lang.foreign.Arena arena = java.lang.foreign.Arena.ofConfined()) {
            float[] xs = new float[rows * dim], ws = new float[dim];
            Arrays.fill(xs, 0.5f);
            Arrays.fill(ws, 1.5f);
            var stream = MlxC.gpuStream();
            var flat = MlxC.fromFloats(xs, arena);
            var x = MlxC.asStrided(flat, new int[] { rows, dim }, new long[] { dim, 1 }, 0, stream, arena);
            var w = MlxC.fromFloats(ws, arena);
            MlxC.eval(x);
            MlxC.eval(w);
            long[] t = new long[iters];
            for (int i = -20; i < iters; i++) {
                long t0 = System.nanoTime();
                try (java.lang.foreign.Arena call = java.lang.foreign.Arena.ofConfined()) {
                    var y = MlxC.rmsNorm(x, w, 1e-5f, stream, call);
                    MlxC.eval(y);
                    MlxC.free(y);
                }
                if (i >= 0) {
                    t[i] = System.nanoTime() - t0;
                }
            }
            return median(t);
        }
    }

    private interface RmsBinding {
        LibraryTaskDescriptor bind(FloatArray x, FloatArray w, FloatArray out, int rows, int dim, float eps);
    }

    /** Runs one rms_norm variant: checks it against the Java reference and returns median execute() in us. */
    private static double rmsVariant(String label, int rows, int dim, FloatArray x, FloatArray w, float[] ref, RmsBinding lib, int iters) {
        FloatArray out = new FloatArray(rows * dim);
        float eps = 1e-5f;
        TaskGraph tg = new TaskGraph("rms").transferToDevice(DataTransferMode.FIRST_EXECUTION, x, w, out);
        GridScheduler grid = null;
        if (lib == null) {
            WorkerGrid1D wg = new WorkerGrid1D(rows * RMS_LOCAL);
            wg.setLocalWork(RMS_LOCAL, 1, 1);
            grid = new GridScheduler("rms.t", wg);
            tg.task("t", SpikeMain::rmsNormJit, new KernelContext(), x, w, out, dim, eps);
        } else {
            tg.libraryTask("t", lib::bind, x, w, out, rows, dim, eps);
        }
        tg.transferToHost(DataTransferMode.UNDER_DEMAND, out);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(tg.snapshot())) {
            if (grid != null) {
                plan.withGridScheduler(grid);
            }
            plan.execute().transferToHost(out);
            int bad = 0;
            for (int i = 0; i < rows * dim; i++) {
                if (Math.abs(out.get(i) - ref[i]) > 1e-4f * Math.max(1f, Math.abs(ref[i]))) {
                    if (bad++ < 2) {
                        System.out.printf("    %s out[%d] = %f, expected %f%n", label, i, out.get(i), ref[i]);
                    }
                }
            }
            check(bad == 0, String.format("rms_norm %-22s rows=%-4d dim=%-5d correct", label, rows, dim));
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
            if (lib != null && MlxSpikeProvider.STATS.calls > 0) {
                MlxSpikeProvider.Stats st = MlxSpikeProvider.STATS;
                System.out.printf("        provider split per call: pre %6.1f us  compute %6.1f us  copy-out %6.1f us%n", st.wrapNanos / 1e3 / st.calls, st.computeNanos / 1e3 / st.calls,
                        st.copyOutNanos / 1e3 / st.calls);
            }
            return median(t);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Marginal cost of one rms_norm inside a TaskGraph: a chain of {@code chain} rms_norm tasks
     * ping-ponging between two buffers, timed per execute() and divided by the chain length after
     * subtracting a one-task graph. Closer to how jitLLM runs a layer than per-execute() timings.
     */
    private static double rmsMarginal(int rows, int dim, RmsBinding lib, int chain, int iters) {
        return (rmsChain(rows, dim, lib, chain, iters) - rmsChain(rows, dim, lib, 1, iters)) / (chain - 1);
    }

    private static double rmsChain(int rows, int dim, RmsBinding lib, int chain, int iters) {
        FloatArray a = new FloatArray(rows * dim), b = new FloatArray(rows * dim), w = new FloatArray(dim);
        a.init(0.5f);
        w.init(1.0f);
        TaskGraph tg = new TaskGraph("chain").transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b, w);
        GridScheduler grid = new GridScheduler();
        for (int i = 0; i < chain; i++) {
            FloatArray in = (i % 2 == 0) ? a : b, out = (i % 2 == 0) ? b : a;
            if (lib == null) {
                WorkerGrid1D wg = new WorkerGrid1D(rows * RMS_LOCAL);
                wg.setLocalWork(RMS_LOCAL, 1, 1);
                grid.addWorkerGrid("chain.t" + i, wg);
                tg.task("t" + i, SpikeMain::rmsNormJit, new KernelContext(), in, w, out, dim, 1e-5f);
            } else {
                tg.libraryTask("t" + i, lib::bind, in, w, out, rows, dim, 1e-5f);
            }
        }
        tg.transferToHost(DataTransferMode.UNDER_DEMAND, a);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(tg.snapshot())) {
            if (lib == null) {
                plan.withGridScheduler(grid);
            }
            long[] t = new long[iters];
            for (int i = -20; i < iters; i++) {
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

    /**
     * Command-buffer round trip vs batching, outside TaskGraph: 16 rms_norm dispatches as 16
     * command buffers (commit + wait each) vs one command buffer with 16 dispatches.
     */
    private static void batchingProbe(int rows, int dim) throws Throwable {
        long device = uk.ac.manchester.tornado.drivers.metal.ffm.MetalAPI.createSystemDefaultDevice();
        long queue = uk.ac.manchester.tornado.drivers.metal.ffm.MetalAPI.newCommandQueue(device);
        long bytes = TornadoNativeArray.ARRAY_HEADER + (long) rows * dim * Float.BYTES;
        long a = uk.ac.manchester.tornado.drivers.metal.ffm.MetalAPI.newBufferWithLength(device, bytes, 0);
        long b = uk.ac.manchester.tornado.drivers.metal.ffm.MetalAPI.newBufferWithLength(device, bytes, 0);
        long w = uk.ac.manchester.tornado.drivers.metal.ffm.MetalAPI.newBufferWithLength(device, bytes, 0);
        long h = TornadoNativeArray.ARRAY_HEADER;
        int n = 16, iters = 200;
        long[] sep = new long[iters], one = new long[iters];
        for (int it = -20; it < iters; it++) {
            long t0 = System.nanoTime();
            for (int i = 0; i < n; i++) {
                MlxKernels.rmsNorm(queue, i % 2 == 0 ? a : b, w, i % 2 == 0 ? b : a, h, rows, dim, 1e-5f);
            }
            long t1 = System.nanoTime();
            MlxKernels.rmsNormBatch(queue, new long[] { a, b }, w, new long[] { b, a }, h, rows, dim, 1e-5f, n);
            long t2 = System.nanoTime();
            if (it >= 0) {
                sep[it] = t1 - t0;
                one[it] = t2 - t1;
            }
        }
        System.out.printf("  rows=%-4d dim=%-5d 16 command buffers: %7.1f us per op | 1 command buffer, 16 dispatches: %6.1f us per op (%.1fx)%n", rows, dim, median(sep) / n, median(one) / n,
                median(sep) / median(one));
    }

    private static void rmsCompare(int rows, int dim, int iters) throws Throwable {
        FloatArray x = new FloatArray(rows * dim), w = new FloatArray(dim);
        java.util.Random r = new java.util.Random(42);
        for (int i = 0; i < rows * dim; i++) {
            x.set(i, r.nextFloat() * 2 - 1);
        }
        for (int i = 0; i < dim; i++) {
            w.set(i, r.nextFloat() + 0.5f);
        }
        float[] ref = new float[rows * dim];
        for (int row = 0; row < rows; row++) {
            double ss = 0;
            for (int i = 0; i < dim; i++) {
                ss += x.get(row * dim + i) * x.get(row * dim + i);
            }
            float inv = (float) (1.0 / Math.sqrt(ss / dim + 1e-5));
            for (int i = 0; i < dim; i++) {
                ref[row * dim + i] = w.get(i) * (x.get(row * dim + i) * inv);
            }
        }
        double alone = mlxAloneRms(rows, dim, iters);
        double mlxc = rmsVariant("apple/mlx (mlx-c)", rows, dim, x, w, ref, SpikeMain::rmsMlxc, iters);
        MlxSpikeProvider.Stats s = MlxSpikeProvider.STATS;
        double mlxcCopy = s.copyOutNanos / 1e3 / s.calls;
        double kernel = rmsVariant("mlx-kernels (metallib)", rows, dim, x, w, ref, SpikeMain::rmsKernel, iters);
        double kernelGpu = s.gpuMicros / s.calls;
        double jit = rmsVariant("TornadoVM JIT", rows, dim, x, w, ref, null, iters);
        RMS_ROWS.add(String.format("  %-5d %-6d %10.1f %12.1f (copy %5.1f) %12.1f (GPU %6.1f) %12.1f", rows, dim, alone, mlxc, mlxcCopy, kernel, kernelGpu, jit));
    }

    /**
     * M0 step 4a: MLX's rms_single_row MSL, adapted to TornadoVM's prebuilt ABI, run with
     * prebuiltTask. Checks it against the Java reference and times it per execute().
     */
    private static void prebuiltRms(int rows, int dim, int iters) {
        if (TornadoNativeArray.ARRAY_HEADER != 16) {
            System.out.println("  SKIP  prebuilt rms_norm: kernel assumes a 16-byte header");
            return;
        }
        FloatArray x = new FloatArray(rows * dim), w = new FloatArray(dim), out = new FloatArray(rows * dim);
        FloatArray params = FloatArray.fromElements(1e-5f, dim);
        java.util.Random r = new java.util.Random(7);
        for (int i = 0; i < rows * dim; i++) {
            x.set(i, r.nextFloat() * 2 - 1);
        }
        for (int i = 0; i < dim; i++) {
            w.set(i, r.nextFloat() + 0.5f);
        }
        uk.ac.manchester.tornado.api.AccessorParameters acc = new uk.ac.manchester.tornado.api.AccessorParameters(4);
        acc.set(0, x, Access.READ_ONLY);
        acc.set(1, w, Access.READ_ONLY);
        acc.set(2, out, Access.WRITE_ONLY);
        acc.set(3, params, Access.READ_ONLY);
        String file = System.getProperty("mlx.spike.kernels", "../../kernels") + "/mlx_rms_f32.metal";
        TaskGraph tg = new TaskGraph("pb") //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, x, w, params) //
                .prebuiltTask("t", "mlx_rms_f32", file, acc) //
                .transferToHost(DataTransferMode.UNDER_DEMAND, out);
        long tgSize = 32L * (((dim + 3) / 4 + 31) / 32);
        WorkerGrid1D wg = new WorkerGrid1D((int) (rows * tgSize));
        wg.setLocalWork(tgSize, 1, 1);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(tg.snapshot())) {
            plan.withGridScheduler(new GridScheduler("pb.t", wg));
            plan.execute().transferToHost(out);
            int bad = 0;
            for (int row = 0; row < rows; row++) {
                double ss = 0;
                for (int i = 0; i < dim; i++) {
                    ss += x.get(row * dim + i) * x.get(row * dim + i);
                }
                float inv = (float) (1.0 / Math.sqrt(ss / dim + 1e-5));
                for (int i = 0; i < dim; i++) {
                    float ref = w.get(i) * (x.get(row * dim + i) * inv);
                    if (Math.abs(out.get(row * dim + i) - ref) > 1e-4f * Math.max(1f, Math.abs(ref)) && bad++ < 2) {
                        System.out.printf("    prebuilt out[%d] = %f, expected %f%n", row * dim + i, out.get(row * dim + i), ref);
                    }
                }
            }
            check(bad == 0, String.format("rms_norm %-22s rows=%-4d dim=%-5d correct", "prebuiltTask (MLX MSL)", rows, dim));
            long[] t = new long[iters];
            for (int i = -20; i < iters; i++) {
                long t0 = System.nanoTime();
                plan.execute();
                if (i >= 0) {
                    t[i] = System.nanoTime() - t0;
                }
            }
            PREBUILT_ROWS.add(String.format("  %-5d %-6d %12.1f", rows, dim, median(t)));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private static final java.util.List<String> PREBUILT_ROWS = new java.util.ArrayList<>();

    private static final java.util.List<String> RMS_ROWS = new java.util.ArrayList<>();

    public static void main(String[] args) throws Throwable {
        System.out.printf("mlx-c spike: MLX %s, ARRAY_HEADER = %d bytes%n%n", MlxC.version(), TornadoNativeArray.ARRAY_HEADER);

        jitMlxJit(1);
        jitMlxJit(1000);
        jitMlxJit(1 << 20);
        matmulCheck(64, 96, 80);
        matmulCheck(257, 129, 65);

        System.out.println("\nBoundary cost (affine y = 2x + 1, float32, 200 executions after warm-up)");
        for (int n : new int[] { 1024, 1 << 16, 1 << 20, 1 << 24 }) {
            boundaryCost(n, 200);
        }

        System.out.println("\nSame op, three ways (median us per execution, 200 iterations after warm-up)");
        System.out.printf("  %-10s %14s %22s %16s%n", "n", "MLX alone", "MLX on TornadoVM bufs", "TornadoVM JIT");
        for (int n : new int[] { 1024, 1 << 16, 1 << 20, 1 << 24 }) {
            double mlx = mlxAlone(n, 200);
            double jit = jitAlone(n, 200);
            FloatArray x = new FloatArray(n), y = new FloatArray(n);
            x.init(1.0f);
            TaskGraph tg = new TaskGraph("hy") //
                    .transferToDevice(DataTransferMode.FIRST_EXECUTION, x, y) //
                    .libraryTask("mlx", SpikeMain::affine, x, y, n, 2.0f, 1.0f) //
                    .transferToHost(DataTransferMode.UNDER_DEMAND, y);
            double hybrid;
            try (TornadoExecutionPlan plan = new TornadoExecutionPlan(tg.snapshot())) {
                long[] t = new long[200];
                for (int i = -20; i < 200; i++) {
                    long t0 = System.nanoTime();
                    plan.execute();
                    if (i >= 0) {
                        t[i] = System.nanoTime() - t0;
                    }
                }
                hybrid = median(t);
            }
            System.out.printf("  %-10d %14.1f %22.1f %16.1f%n", n, mlx, hybrid, jit);
        }

        System.out.println("\nrms_norm float32");
        int[][] rmsShapes = { { 1, 2048 }, { 1, 4096 }, { 1, 8192 }, { 128, 4096 }, { 512, 4096 } };
        for (int[] sh : rmsShapes) {
            rmsCompare(sh[0], sh[1], 200);
        }
        for (int[] sh : rmsShapes) {
            if (sh[1] <= 4096) {
                prebuiltRms(sh[0], sh[1], 200);
            }
        }
        System.out.println("\nrms_norm float32, median us per execute()");
        System.out.printf("  %-5s %-6s %10s %26s %26s %12s%n", "rows", "dim", "MLX alone", "apple/mlx (mlx-c)", "mlx-kernels (metallib)", "TornadoVM JIT");
        RMS_ROWS.forEach(System.out::println);

        System.out.println("\nrms_norm float32 via prebuiltTask (MLX MSL on TornadoVM ABI), median us per execute()");
        PREBUILT_ROWS.forEach(System.out::println);

        System.out.println("\nrms_norm float32, marginal us per task inside one TaskGraph (chain of 16 vs 1)");
        System.out.printf("  %-5s %-6s %18s %24s %14s%n", "rows", "dim", "apple/mlx (mlx-c)", "mlx-kernels (metallib)", "TornadoVM JIT");
        for (int[] sh : new int[][] { { 1, 4096 }, { 512, 4096 } }) {
            System.out.printf("  %-5d %-6d %18.1f %24.1f %14.1f%n", sh[0], sh[1], rmsMarginal(sh[0], sh[1], SpikeMain::rmsMlxc, 16, 100),
                    rmsMarginal(sh[0], sh[1], SpikeMain::rmsKernel, 16, 100), rmsMarginal(sh[0], sh[1], null, 16, 100));
        }

        System.out.println("\nCommand-buffer round trip vs batching (MLX rms_norm kernel, raw Metal, no TaskGraph)");
        for (int[] sh : new int[][] { { 1, 4096 }, { 128, 4096 }, { 512, 4096 } }) {
            batchingProbe(sh[0], sh[1]);
        }

        System.out.println(failures == 0 ? "\nSPIKE PASSED" : "\nSPIKE FAILED (" + failures + " checks)");
        System.exit(failures == 0 ? 0 : 1);
    }
}
