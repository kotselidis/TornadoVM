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
package uk.ac.manchester.tornado.coreml;

/**
 * The processing units Core ML may run a model on, as {@code MLComputeUnits}. Core ML splits the
 * model between the allowed units itself; the Neural Engine is reachable only through Core ML.
 */
public enum ComputeUnits {

    /** The CPU only. */
    CPU_ONLY(0),

    /** The CPU and the GPU. */
    CPU_AND_GPU(1),

    /** Any unit: CPU, GPU and Neural Engine (Core ML's default). */
    ALL(2),

    /** The CPU and the Neural Engine, never the GPU. */
    CPU_AND_NEURAL_ENGINE(3);

    private final long value;

    ComputeUnits(long value) {
        this.value = value;
    }

    /** The {@code MLComputeUnits} value. */
    public long value() {
        return value;
    }
}
