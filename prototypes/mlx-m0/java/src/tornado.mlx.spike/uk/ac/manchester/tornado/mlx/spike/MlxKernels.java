package uk.ac.manchester.tornado.mlx.spike;

import static uk.ac.manchester.tornado.runtime.ffm.FFMSupport.C_FLOAT;
import static uk.ac.manchester.tornado.runtime.ffm.FFMSupport.C_INT;
import static uk.ac.manchester.tornado.runtime.ffm.FFMSupport.C_LONG;
import static uk.ac.manchester.tornado.runtime.ffm.FFMSupport.C_POINTER;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

import uk.ac.manchester.tornado.drivers.metal.ffm.MetalAPI;
import uk.ac.manchester.tornado.drivers.metal.ffm.ObjCRuntime;
import uk.ac.manchester.tornado.drivers.metal.runtime.MetalTornadoDevice;
import uk.ac.manchester.tornado.runtime.ffm.FFMSupport;

/**
 * Runs kernels from MLX's own {@code mlx.metallib} on TornadoVM's Metal command queue, with
 * TornadoVM's buffers bound directly (the header skipped through the {@code setBuffer} offset),
 * so both inputs and output stay in place. This is the mechanism an {@code apple/mlx-kernels}
 * provider would use.
 */
final class MlxKernels {

    static final Path METALLIB = Path.of(System.getProperty("mlx.metallib", "/opt/homebrew/opt/mlx/lib/mlx.metallib"));

    private static final MethodHandle DISPATCH_DATA_CREATE = FFMSupport.downcall(FFMSupport.loadLibrary("/usr/lib/libSystem.B.dylib"),
            FunctionDescriptor.of(C_LONG, C_POINTER, C_LONG, C_POINTER, C_POINTER), "dispatch_data_create");

    /** MLX's constants for rms_norm (kernels/defines.h). */
    private static final int RMS_N_READS = 4;
    private static final int RMS_LOOPED_LIMIT = 4096;
    private static final int SIMD_SIZE = 32;

    private static long library;
    private static final Map<String, Long> PIPELINES = new HashMap<>();

    /** GPU time of the last dispatch, from the command buffer's GPUStart/EndTime. */
    static double lastGpuMicros;

    private MlxKernels() {
    }

    /**
     * TornadoVM's command queue for this execution plan. {@code getCommandQueue} is private
     * today (the spike needs {@code --add-opens}); M1 exposes it through
     * {@code TornadoNativeStreamSupport.getNativeStream}.
     */
    static long tornadoQueue(MetalTornadoDevice device, long executionPlanId) {
        try {
            Object deviceContext = device.getDeviceContext();
            Method getQueue = deviceContext.getClass().getDeclaredMethod("getCommandQueue", long.class);
            getQueue.setAccessible(true);
            Object queue = getQueue.invoke(deviceContext, executionPlanId);
            return (long) queue.getClass().getMethod("getCommandQueuePtr").invoke(queue);
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException("cannot reach TornadoVM's Metal command queue (run with --add-opens)", e);
        }
    }

    private static synchronized long pipeline(long queue, String name) throws Throwable {
        Long cached = PIPELINES.get(name);
        if (cached != null) {
            return cached;
        }
        long device = MetalAPI.queueDevice(queue);
        try (Arena arena = Arena.ofConfined()) {
            if (library == 0) {
                byte[] bytes = Files.readAllBytes(METALLIB);
                MemorySegment seg = arena.allocate(bytes.length);
                MemorySegment.copy(bytes, 0, seg, FFMSupport.C_CHAR, 0, bytes.length);
                long data = (long) DISPATCH_DATA_CREATE.invokeExact(seg, (long) bytes.length, MemorySegment.NULL, MemorySegment.NULL);
                MemorySegment err = FFMSupport.allocatePointer(arena);
                library = MetalAPI.newLibraryWithData(device, data, err);
                ObjCRuntime.release(data);
                if (library == 0) {
                    throw new IllegalStateException("cannot load " + METALLIB);
                }
            }
            long fn = MetalAPI.newFunctionWithName(library, ObjCRuntime.newNSString(name));
            if (fn == 0) {
                throw new IllegalStateException("kernel " + name + " not in " + METALLIB);
            }
            MemorySegment err = FFMSupport.allocatePointer(arena);
            long pso = MetalAPI.newComputePipelineStateWithFunction(device, fn, err);
            if (pso == 0) {
                throw new IllegalStateException("pipeline for " + name + " failed");
            }
            PIPELINES.put(name, pso);
            return pso;
        }
    }

    /**
     * MLX's rms_norm on float32: {@code out[r, :] = w * x[r, :] / sqrt(mean(x[r, :]^2) + eps)}.
     * Arguments are TornadoVM MTLBuffer handles and the byte offset of the first element in each.
     * Mirrors {@code RMSNorm::eval_gpu} in MLX (normalization.cpp).
     */
    static void rmsNorm(long queue, long x, long w, long out, long offset, int rows, int axisSize, float eps) throws Throwable {
        rmsNormBatch(queue, new long[] { x }, w, new long[] { out }, offset, rows, axisSize, eps, 1);
    }

    /**
     * Encodes {@code count} rms_norm dispatches (x[i] -> out[i]) into ONE command buffer and waits
     * once. Used to measure how much of a single op's cost is the per-command-buffer round trip.
     */
    static void rmsNormBatch(long queue, long[] xs, long w, long[] outs, long offset, int rows, int axisSize, float eps, int count) throws Throwable {
        boolean looped = axisSize > RMS_LOOPED_LIMIT;
        long pso = pipeline(queue, looped ? "rms_loopedfloat32" : "rmsfloat32");
        long threadgroupSize;
        if (looped) {
            threadgroupSize = MetalAPI.pipelineMaxTotalThreadsPerThreadgroup(pso);
        } else {
            long threadsNeeded = (axisSize + RMS_N_READS - 1) / RMS_N_READS;
            long simdsNeeded = (threadsNeeded + SIMD_SIZE - 1) / SIMD_SIZE;
            threadgroupSize = SIMD_SIZE * simdsNeeded;
        }
        try (Arena arena = Arena.ofConfined(); ObjCRuntime.AutoreleasePool pool = new ObjCRuntime.AutoreleasePool()) {
            long cb = MetalAPI.commandBuffer(queue);
            long enc = MetalAPI.computeCommandEncoder(cb);
            MetalAPI.setComputePipelineState(enc, pso);
            for (int i = 0; i < count; i++) {
            MetalAPI.setBuffer(enc, xs[i % xs.length], offset, 0);
            MetalAPI.setBuffer(enc, w, offset, 1);
            MetalAPI.setBuffer(enc, outs[i % outs.length], offset, 2);
            MetalAPI.setBytes(enc, arena.allocateArray(C_FLOAT, new float[] { eps }), 4, 3);
            MetalAPI.setBytes(enc, arena.allocateArray(C_INT, new int[] { axisSize }), 4, 4);
            MetalAPI.setBytes(enc, arena.allocateArray(C_INT, new int[] { 1 }), 4, 5); // w_stride
            MemorySegment grid = arena.allocate(ObjCRuntime.MTL_SIZE);
            grid.setAtIndex(C_LONG, 0, rows * threadgroupSize);
            grid.setAtIndex(C_LONG, 1, 1);
            grid.setAtIndex(C_LONG, 2, 1);
            MemorySegment group = arena.allocate(ObjCRuntime.MTL_SIZE);
            group.setAtIndex(C_LONG, 0, threadgroupSize);
            group.setAtIndex(C_LONG, 1, 1);
            group.setAtIndex(C_LONG, 2, 1);
            MetalAPI.dispatchThreads(enc, grid, group);
            }
            MetalAPI.endEncoding(enc);
            MetalAPI.commit(cb);
            MetalAPI.waitUntilCompleted(cb);
            lastGpuMicros = (MetalAPI.gpuEndTime(cb) - MetalAPI.gpuStartTime(cb)) * 1e6;
        }
    }
}
