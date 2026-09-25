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

import uk.ac.manchester.tornado.api.common.Access;
import uk.ac.manchester.tornado.api.common.LibraryTaskDescriptor;
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
 * MLX reads TornadoVM's buffers in place (no copy on the way in). MLX operations always
 * allocate their own result, so the result is copied into the output array.
 * </p>
 */
public final class Mlx {

    public static final String LIBRARY_NAME = "apple/mlx";

    private Mlx() {
    }

    private static final Access[] BINARY_ACCESS = { Access.READ_ONLY, Access.READ_ONLY, Access.WRITE_ONLY };

    private static LibraryTaskDescriptor binary(String function, Object a, Object b, Object c) {
        return new LibraryTaskDescriptor() //
                .withLibrary(LIBRARY_NAME) //
                .withFunction(function) //
                .withParameters(new Object[] { a, b, c }) //
                .withAccess(BINARY_ACCESS.clone());
    }

    /** Element-wise {@code c = a + b}; all three arrays have the same length. */
    @MlxOp("mlx_add")
    public static LibraryTaskDescriptor add(FloatArray a, FloatArray b, FloatArray c) {
        return binary("add", a, b, c);
    }

    /** Element-wise {@code c = a + b} in float16. */
    @MlxOp("mlx_add")
    public static LibraryTaskDescriptor add(HalfFloatArray a, HalfFloatArray b, HalfFloatArray c) {
        return binary("add", a, b, c);
    }

    /** Element-wise {@code c = a + b} in int32. */
    @MlxOp("mlx_add")
    public static LibraryTaskDescriptor add(IntArray a, IntArray b, IntArray c) {
        return binary("add", a, b, c);
    }
}
