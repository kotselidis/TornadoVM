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
package uk.ac.manchester.tornado.unittests.coreml;

import static org.junit.jupiter.api.Assertions.fail;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Random;

import org.junit.jupiter.api.BeforeEach;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.enums.TornadoVMBackendType;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.coreml.CoreML;
import uk.ac.manchester.tornado.unittests.common.TornadoTestBase;
import uk.ac.manchester.tornado.unittests.common.TornadoVMMetalNotSupported;

/**
 * Common set-up for the Core ML library-task tests: the Metal/Core ML guard, the test model and
 * tolerance checks.
 *
 * <p>
 * The test model, {@code AddMul.mlpackage}, has two float32 inputs {@code x} and {@code y} of shape
 * [16] and two float32 outputs, {@code prod = x * y} and {@code sum = x + y}. It is written by
 * {@code tornado-coreml/scripts/make_test_model.py}.
 * </p>
 */
public abstract class CoreMLTestBase extends TornadoTestBase {

    /** The number of elements of every feature of the test model. */
    protected static final int N = 16;

    private static final String[] MODEL_FILES = { "Manifest.json", "Data/com.apple.CoreML/model.mlmodel" };

    private static Path modelPath;

    /**
     * Core ML tasks need the Metal backend and Core ML (macOS 13 or newer). Unavailable configurations
     * throw the typed *NotSupported exceptions that TornadoTestRunner reports as [UNSUPPORTED].
     */
    @BeforeEach
    public void coreMLMustBeAvailable() {
        TornadoVMBackendType backend = getTornadoRuntime().getDefaultDevice().getTornadoVMBackend();
        if (backend != TornadoVMBackendType.METAL) {
            assertNotBackend(backend, "Core ML library tasks require the Metal backend (default device is " + backend + ")");
        }
        if (!CoreML.isAvailable()) {
            throw new TornadoVMMetalNotSupported("Core ML (macOS 13 or newer) is not available on this host");
        }
    }

    /** The test model, unpacked from the test resources into a temporary directory once. */
    protected static synchronized String addMulModel() {
        if (modelPath == null) {
            try {
                Path dir = Files.createTempDirectory("tornado-coreml").resolve("AddMul.mlpackage");
                for (String file : MODEL_FILES) {
                    Path target = dir.resolve(file);
                    Files.createDirectories(target.getParent());
                    try (InputStream in = CoreMLTestBase.class.getResourceAsStream("AddMul.mlpackage/" + file)) {
                        if (in == null) {
                            throw new IOException("missing test resource AddMul.mlpackage/" + file);
                        }
                        Files.copy(in, target, StandardCopyOption.REPLACE_EXISTING);
                    }
                }
                modelPath = dir;
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }
        return modelPath.toString();
    }

    /** Uniform values in [lo, hi). */
    protected static FloatArray values(int n, float lo, float hi, long seed) {
        Random random = new Random(seed);
        FloatArray v = new FloatArray(n);
        for (int i = 0; i < n; i++) {
            v.set(i, lo + (hi - lo) * random.nextFloat());
        }
        return v;
    }

    /**
     * Within float16 rounding of the expected value: Core ML may run the model on the GPU or the
     * Neural Engine in half precision.
     */
    protected static void assertClose(String what, int index, double expected, double actual) {
        if (Math.abs(expected - actual) > 2e-3 + 2e-3 * Math.abs(expected)) {
            fail(what + " element " + index + ": expected " + expected + " but was " + actual);
        }
    }

    /** {@code prod = x * y} and {@code sum = x + y}, element by element. */
    protected static void assertAddMul(FloatArray x, FloatArray y, FloatArray prod, FloatArray sum) {
        for (int i = 0; i < x.getSize(); i++) {
            assertClose("prod", i, (double) x.get(i) * y.get(i), prod.get(i));
            assertClose("sum", i, (double) x.get(i) + y.get(i), sum.get(i));
        }
    }

    /** {@code x[i] = i / 8 - 1}, {@code y[i] = 1 - i / 16}. */
    public static void ramps(KernelContext context, FloatArray x, FloatArray y) {
        int id = context.globalIdx;
        if (id < x.getSize()) {
            x.set(id, id / 8.0f - 1.0f);
            y.set(id, 1.0f - id / 16.0f);
        }
    }

    /** {@code out[i] = prod[i] + sum[i]}. */
    public static void combine(KernelContext context, FloatArray prod, FloatArray sum, FloatArray out) {
        int id = context.globalIdx;
        if (id < out.getSize()) {
            out.set(id, prod.get(id) + sum.get(id));
        }
    }
}
