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
package uk.ac.manchester.tornado.mlx;

/**
 * Per-call options for an MLX library task, attached with
 * {@code LibraryTaskDescriptor.withTuning(...)}:
 *
 * <pre>
 * Mlx.add(a, b, c).withTuning(MlxOptions.cpu())
 * </pre>
 *
 * <p>
 * By default MLX operations run on the GPU stream. Operations MLX only implements on the CPU (some
 * linear-algebra decompositions) always use the CPU stream.
 * </p>
 * <p>
 * On the GPU, operations that have one run MLX's own Metal kernel in place on the TornadoVM buffers.
 * {@code MlxOptions.gpu().inPlaceKernels(false)} sends the call through MLX's C API instead, which
 * allocates an MLX result and copies it back.
 * </p>
 */
public final class MlxOptions {

    /** Where MLX runs the operation. */
    public enum Device {
        GPU, CPU
    }

    private final Device device;
    private final boolean inPlaceKernels;

    private MlxOptions(Device device, boolean inPlaceKernels) {
        this.device = device;
        this.inPlaceKernels = inPlaceKernels;
    }

    public static MlxOptions gpu() {
        return new MlxOptions(Device.GPU, true);
    }

    public static MlxOptions cpu() {
        return new MlxOptions(Device.CPU, true);
    }

    /** These options, with in-place MLX kernels allowed or not. */
    public MlxOptions inPlaceKernels(boolean enabled) {
        return new MlxOptions(device, enabled);
    }

    public Device getDevice() {
        return device;
    }

    public boolean isInPlaceKernels() {
        return inPlaceKernels;
    }
}
