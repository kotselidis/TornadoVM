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

import java.util.Arrays;

import uk.ac.manchester.tornado.api.common.Access;
import uk.ac.manchester.tornado.api.common.LibraryTaskDescriptor;
import uk.ac.manchester.tornado.api.types.arrays.BFloat16Array;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Factory methods for Apple MLX library tasks on the Metal backend. Each method builds a
 * {@link LibraryTaskDescriptor} consumed by {@code TaskGraph#libraryTask(String, ...)}:
 *
 * <pre>
 * taskGraph.libraryTask("add", Mlx::add, a, b, c);
 * </pre>
 *
 * <p>
 * MLX reads TornadoVM's buffers in place (no copy on the way in). MLX operations always allocate
 * their own result, so the result is copied into the output array. Every factory is annotated
 * with the mlx-c operation it binds ({@link MlxOp}); the build checks each one has a test.
 * </p>
 */
public final class Mlx {

    public static final String LIBRARY_NAME = "apple/mlx";

    private Mlx() {
    }

    /** All arguments are READ_ONLY except the output at {@code outputIndex}, which is WRITE_ONLY. */
    private static Access[] readOnlyExcept(int numArgs, int outputIndex) {
        Access[] accesses = new Access[numArgs];
        Arrays.fill(accesses, Access.READ_ONLY);
        accesses[outputIndex] = Access.WRITE_ONLY;
        return accesses;
    }

    private static LibraryTaskDescriptor task(String function, int outputIndex, Object... parameters) {
        return new LibraryTaskDescriptor() //
                .withLibrary(LIBRARY_NAME) //
                .withFunction(function) //
                .withParameters(parameters) //
                .withAccess(readOnlyExcept(parameters.length, outputIndex));
    }

    // ---------------------------------------------------------------- element-wise binary: (a, b, c), same length

    /** Element-wise {@code c = a + b}. */
    @MlxOp("mlx_add")
    public static LibraryTaskDescriptor add(FloatArray a, FloatArray b, FloatArray c) {
        return task("add", 2, a, b, c);
    }

    /** Element-wise {@code c = a + b}. */
    @MlxOp("mlx_add")
    public static LibraryTaskDescriptor add(HalfFloatArray a, HalfFloatArray b, HalfFloatArray c) {
        return task("add", 2, a, b, c);
    }

    /** Element-wise {@code c = a + b}. */
    @MlxOp("mlx_add")
    public static LibraryTaskDescriptor add(BFloat16Array a, BFloat16Array b, BFloat16Array c) {
        return task("add", 2, a, b, c);
    }

    /** Element-wise {@code c = a + b}. */
    @MlxOp("mlx_add")
    public static LibraryTaskDescriptor add(IntArray a, IntArray b, IntArray c) {
        return task("add", 2, a, b, c);
    }

    /** Element-wise {@code c = a - b}. */
    @MlxOp("mlx_subtract")
    public static LibraryTaskDescriptor subtract(FloatArray a, FloatArray b, FloatArray c) {
        return task("subtract", 2, a, b, c);
    }

    /** Element-wise {@code c = a - b}. */
    @MlxOp("mlx_subtract")
    public static LibraryTaskDescriptor subtract(HalfFloatArray a, HalfFloatArray b, HalfFloatArray c) {
        return task("subtract", 2, a, b, c);
    }

    /** Element-wise {@code c = a - b}. */
    @MlxOp("mlx_subtract")
    public static LibraryTaskDescriptor subtract(BFloat16Array a, BFloat16Array b, BFloat16Array c) {
        return task("subtract", 2, a, b, c);
    }

    /** Element-wise {@code c = a - b}. */
    @MlxOp("mlx_subtract")
    public static LibraryTaskDescriptor subtract(IntArray a, IntArray b, IntArray c) {
        return task("subtract", 2, a, b, c);
    }

    /** Element-wise {@code c = a * b}. */
    @MlxOp("mlx_multiply")
    public static LibraryTaskDescriptor multiply(FloatArray a, FloatArray b, FloatArray c) {
        return task("multiply", 2, a, b, c);
    }

    /** Element-wise {@code c = a * b}. */
    @MlxOp("mlx_multiply")
    public static LibraryTaskDescriptor multiply(HalfFloatArray a, HalfFloatArray b, HalfFloatArray c) {
        return task("multiply", 2, a, b, c);
    }

    /** Element-wise {@code c = a * b}. */
    @MlxOp("mlx_multiply")
    public static LibraryTaskDescriptor multiply(BFloat16Array a, BFloat16Array b, BFloat16Array c) {
        return task("multiply", 2, a, b, c);
    }

    /** Element-wise {@code c = a * b}. */
    @MlxOp("mlx_multiply")
    public static LibraryTaskDescriptor multiply(IntArray a, IntArray b, IntArray c) {
        return task("multiply", 2, a, b, c);
    }

    /** Element-wise {@code c = a / b}. */
    @MlxOp("mlx_divide")
    public static LibraryTaskDescriptor divide(FloatArray a, FloatArray b, FloatArray c) {
        return task("divide", 2, a, b, c);
    }

    /** Element-wise {@code c = a / b}. */
    @MlxOp("mlx_divide")
    public static LibraryTaskDescriptor divide(HalfFloatArray a, HalfFloatArray b, HalfFloatArray c) {
        return task("divide", 2, a, b, c);
    }

    /** Element-wise {@code c = a / b}. */
    @MlxOp("mlx_divide")
    public static LibraryTaskDescriptor divide(BFloat16Array a, BFloat16Array b, BFloat16Array c) {
        return task("divide", 2, a, b, c);
    }

    /** Element-wise {@code c = a / b}. */
    @MlxOp("mlx_divide")
    public static LibraryTaskDescriptor divide(IntArray a, IntArray b, IntArray c) {
        return task("divide", 2, a, b, c);
    }

    /** Element-wise {@code c = max(a, b)}. */
    @MlxOp("mlx_maximum")
    public static LibraryTaskDescriptor maximum(FloatArray a, FloatArray b, FloatArray c) {
        return task("maximum", 2, a, b, c);
    }

    /** Element-wise {@code c = max(a, b)}. */
    @MlxOp("mlx_maximum")
    public static LibraryTaskDescriptor maximum(HalfFloatArray a, HalfFloatArray b, HalfFloatArray c) {
        return task("maximum", 2, a, b, c);
    }

    /** Element-wise {@code c = max(a, b)}. */
    @MlxOp("mlx_maximum")
    public static LibraryTaskDescriptor maximum(BFloat16Array a, BFloat16Array b, BFloat16Array c) {
        return task("maximum", 2, a, b, c);
    }

    /** Element-wise {@code c = max(a, b)}. */
    @MlxOp("mlx_maximum")
    public static LibraryTaskDescriptor maximum(IntArray a, IntArray b, IntArray c) {
        return task("maximum", 2, a, b, c);
    }

    /** Element-wise {@code c = min(a, b)}. */
    @MlxOp("mlx_minimum")
    public static LibraryTaskDescriptor minimum(FloatArray a, FloatArray b, FloatArray c) {
        return task("minimum", 2, a, b, c);
    }

    /** Element-wise {@code c = min(a, b)}. */
    @MlxOp("mlx_minimum")
    public static LibraryTaskDescriptor minimum(HalfFloatArray a, HalfFloatArray b, HalfFloatArray c) {
        return task("minimum", 2, a, b, c);
    }

    /** Element-wise {@code c = min(a, b)}. */
    @MlxOp("mlx_minimum")
    public static LibraryTaskDescriptor minimum(BFloat16Array a, BFloat16Array b, BFloat16Array c) {
        return task("minimum", 2, a, b, c);
    }

    /** Element-wise {@code c = min(a, b)}. */
    @MlxOp("mlx_minimum")
    public static LibraryTaskDescriptor minimum(IntArray a, IntArray b, IntArray c) {
        return task("minimum", 2, a, b, c);
    }

    // ---------------------------------------------------------------- element-wise unary: (a, out), same length

    /** Element-wise {@code out = -a}. */
    @MlxOp("mlx_negative")
    public static LibraryTaskDescriptor negative(FloatArray a, FloatArray out) {
        return task("negative", 1, a, out);
    }

    /** Element-wise {@code out = -a}. */
    @MlxOp("mlx_negative")
    public static LibraryTaskDescriptor negative(HalfFloatArray a, HalfFloatArray out) {
        return task("negative", 1, a, out);
    }

    /** Element-wise {@code out = -a}. */
    @MlxOp("mlx_negative")
    public static LibraryTaskDescriptor negative(BFloat16Array a, BFloat16Array out) {
        return task("negative", 1, a, out);
    }

    /** Element-wise {@code out = -a}. */
    @MlxOp("mlx_negative")
    public static LibraryTaskDescriptor negative(IntArray a, IntArray out) {
        return task("negative", 1, a, out);
    }

    /** Element-wise {@code out = exp(a)}. */
    @MlxOp("mlx_exp")
    public static LibraryTaskDescriptor exp(FloatArray a, FloatArray out) {
        return task("exp", 1, a, out);
    }

    /** Element-wise {@code out = exp(a)}. */
    @MlxOp("mlx_exp")
    public static LibraryTaskDescriptor exp(HalfFloatArray a, HalfFloatArray out) {
        return task("exp", 1, a, out);
    }

    /** Element-wise {@code out = exp(a)}. */
    @MlxOp("mlx_exp")
    public static LibraryTaskDescriptor exp(BFloat16Array a, BFloat16Array out) {
        return task("exp", 1, a, out);
    }

    /** Element-wise {@code out = tanh(a)}. */
    @MlxOp("mlx_tanh")
    public static LibraryTaskDescriptor tanh(FloatArray a, FloatArray out) {
        return task("tanh", 1, a, out);
    }

    /** Element-wise {@code out = tanh(a)}. */
    @MlxOp("mlx_tanh")
    public static LibraryTaskDescriptor tanh(HalfFloatArray a, HalfFloatArray out) {
        return task("tanh", 1, a, out);
    }

    /** Element-wise {@code out = tanh(a)}. */
    @MlxOp("mlx_tanh")
    public static LibraryTaskDescriptor tanh(BFloat16Array a, BFloat16Array out) {
        return task("tanh", 1, a, out);
    }

    /** Element-wise {@code out = erf(a)}. */
    @MlxOp("mlx_erf")
    public static LibraryTaskDescriptor erf(FloatArray a, FloatArray out) {
        return task("erf", 1, a, out);
    }

    /** Element-wise {@code out = erf(a)}. */
    @MlxOp("mlx_erf")
    public static LibraryTaskDescriptor erf(HalfFloatArray a, HalfFloatArray out) {
        return task("erf", 1, a, out);
    }

    /** Element-wise {@code out = erf(a)}. */
    @MlxOp("mlx_erf")
    public static LibraryTaskDescriptor erf(BFloat16Array a, BFloat16Array out) {
        return task("erf", 1, a, out);
    }

    /** Element-wise {@code out = 1 / (1 + exp(-a))}. */
    @MlxOp("mlx_sigmoid")
    public static LibraryTaskDescriptor sigmoid(FloatArray a, FloatArray out) {
        return task("sigmoid", 1, a, out);
    }

    /** Element-wise {@code out = 1 / (1 + exp(-a))}. */
    @MlxOp("mlx_sigmoid")
    public static LibraryTaskDescriptor sigmoid(HalfFloatArray a, HalfFloatArray out) {
        return task("sigmoid", 1, a, out);
    }

    /** Element-wise {@code out = 1 / (1 + exp(-a))}. */
    @MlxOp("mlx_sigmoid")
    public static LibraryTaskDescriptor sigmoid(BFloat16Array a, BFloat16Array out) {
        return task("sigmoid", 1, a, out);
    }

    /** Element-wise {@code out = sqrt(a)}. */
    @MlxOp("mlx_sqrt")
    public static LibraryTaskDescriptor sqrt(FloatArray a, FloatArray out) {
        return task("sqrt", 1, a, out);
    }

    /** Element-wise {@code out = sqrt(a)}. */
    @MlxOp("mlx_sqrt")
    public static LibraryTaskDescriptor sqrt(HalfFloatArray a, HalfFloatArray out) {
        return task("sqrt", 1, a, out);
    }

    /** Element-wise {@code out = sqrt(a)}. */
    @MlxOp("mlx_sqrt")
    public static LibraryTaskDescriptor sqrt(BFloat16Array a, BFloat16Array out) {
        return task("sqrt", 1, a, out);
    }

    /** Element-wise {@code out = 1 / sqrt(a)}. */
    @MlxOp("mlx_rsqrt")
    public static LibraryTaskDescriptor rsqrt(FloatArray a, FloatArray out) {
        return task("rsqrt", 1, a, out);
    }

    /** Element-wise {@code out = 1 / sqrt(a)}. */
    @MlxOp("mlx_rsqrt")
    public static LibraryTaskDescriptor rsqrt(HalfFloatArray a, HalfFloatArray out) {
        return task("rsqrt", 1, a, out);
    }

    /** Element-wise {@code out = 1 / sqrt(a)}. */
    @MlxOp("mlx_rsqrt")
    public static LibraryTaskDescriptor rsqrt(BFloat16Array a, BFloat16Array out) {
        return task("rsqrt", 1, a, out);
    }

    /** Element-wise {@code out = a * a}. */
    @MlxOp("mlx_square")
    public static LibraryTaskDescriptor square(FloatArray a, FloatArray out) {
        return task("square", 1, a, out);
    }

    /** Element-wise {@code out = a * a}. */
    @MlxOp("mlx_square")
    public static LibraryTaskDescriptor square(HalfFloatArray a, HalfFloatArray out) {
        return task("square", 1, a, out);
    }

    /** Element-wise {@code out = a * a}. */
    @MlxOp("mlx_square")
    public static LibraryTaskDescriptor square(BFloat16Array a, BFloat16Array out) {
        return task("square", 1, a, out);
    }

    /** Element-wise {@code out = a * a}. */
    @MlxOp("mlx_square")
    public static LibraryTaskDescriptor square(IntArray a, IntArray out) {
        return task("square", 1, a, out);
    }

}
