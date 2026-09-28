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

import uk.ac.manchester.tornado.api.common.LibraryTaskDescriptor;
import uk.ac.manchester.tornado.api.types.arrays.BFloat16Array;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * MLX indexing (Tier 2) as TornadoVM library tasks: take, gather, scatter, slicing and slice
 * updates, masked scatter and the gathered batched matmul. Indices are int32. Operations that
 * modify an array write the modified copy to {@code out} and leave {@code x} unchanged.
 */
public final class MlxIndex {

    private MlxIndex() {
    }

    /** {@code out[i] = x[indices[i]]} (x as a flat array). */
    @MlxOp("mlx_take")
    public static LibraryTaskDescriptor take(FloatArray x, IntArray indices, FloatArray out) {
        return Mlx.task("take", 2, x, indices, out);
    }

    /** {@code out[i] = x[indices[i]]} (x as a flat array). */
    @MlxOp("mlx_take")
    public static LibraryTaskDescriptor take(HalfFloatArray x, IntArray indices, HalfFloatArray out) {
        return Mlx.task("take", 2, x, indices, out);
    }

    /** {@code out[i] = x[indices[i]]} (x as a flat array). */
    @MlxOp("mlx_take")
    public static LibraryTaskDescriptor take(BFloat16Array x, IntArray indices, BFloat16Array out) {
        return Mlx.task("take", 2, x, indices, out);
    }

    /** {@code out[i] = x[indices[i]]} (x as a flat array). */
    @MlxOp("mlx_take")
    public static LibraryTaskDescriptor take(IntArray x, IntArray indices, IntArray out) {
        return Mlx.task("take", 2, x, indices, out);
    }

    /**
     * {@code out[o, i, j] = x[o, indices[i], j]} for {@code x} viewed as {@code [outer, len, inner]};
     * {@code out} is {@code [outer, indices.length, inner]} (an embedding lookup when {@code outer = 1}).
     */
    @MlxOp("mlx_take_axis")
    public static LibraryTaskDescriptor takeAxis(FloatArray x, IntArray indices, FloatArray out, int outer, int len, int inner) {
        return Mlx.task("take_axis", 2, x, indices, out, outer, len, inner);
    }

    /**
     * {@code out[o, i, j] = x[o, indices[i], j]} for {@code x} viewed as {@code [outer, len, inner]};
     * {@code out} is {@code [outer, indices.length, inner]} (an embedding lookup when {@code outer = 1}).
     */
    @MlxOp("mlx_take_axis")
    public static LibraryTaskDescriptor takeAxis(HalfFloatArray x, IntArray indices, HalfFloatArray out, int outer, int len, int inner) {
        return Mlx.task("take_axis", 2, x, indices, out, outer, len, inner);
    }

    /**
     * {@code out[o, i, j] = x[o, indices[i], j]} for {@code x} viewed as {@code [outer, len, inner]};
     * {@code out} is {@code [outer, indices.length, inner]} (an embedding lookup when {@code outer = 1}).
     */
    @MlxOp("mlx_take_axis")
    public static LibraryTaskDescriptor takeAxis(BFloat16Array x, IntArray indices, BFloat16Array out, int outer, int len, int inner) {
        return Mlx.task("take_axis", 2, x, indices, out, outer, len, inner);
    }

    /**
     * {@code out[o, i, j] = x[o, indices[i], j]} for {@code x} viewed as {@code [outer, len, inner]};
     * {@code out} is {@code [outer, indices.length, inner]} (an embedding lookup when {@code outer = 1}).
     */
    @MlxOp("mlx_take_axis")
    public static LibraryTaskDescriptor takeAxis(IntArray x, IntArray indices, IntArray out, int outer, int len, int inner) {
        return Mlx.task("take_axis", 2, x, indices, out, outer, len, inner);
    }

    /**
     * {@code out[o, i, j] = x[o, indices[o, i, j], j]} for {@code x} viewed as {@code [outer, len, inner]}
     * and {@code indices}, {@code out} as {@code [outer, m, inner]}.
     */
    @MlxOp("mlx_take_along_axis")
    public static LibraryTaskDescriptor takeAlongAxis(FloatArray x, IntArray indices, FloatArray out, int outer, int len, int m, int inner) {
        return Mlx.task("take_along_axis", 2, x, indices, out, outer, len, m, inner);
    }

    /**
     * {@code out[o, i, j] = x[o, indices[o, i, j], j]} for {@code x} viewed as {@code [outer, len, inner]}
     * and {@code indices}, {@code out} as {@code [outer, m, inner]}.
     */
    @MlxOp("mlx_take_along_axis")
    public static LibraryTaskDescriptor takeAlongAxis(HalfFloatArray x, IntArray indices, HalfFloatArray out, int outer, int len, int m, int inner) {
        return Mlx.task("take_along_axis", 2, x, indices, out, outer, len, m, inner);
    }

    /**
     * {@code out[o, i, j] = x[o, indices[o, i, j], j]} for {@code x} viewed as {@code [outer, len, inner]}
     * and {@code indices}, {@code out} as {@code [outer, m, inner]}.
     */
    @MlxOp("mlx_take_along_axis")
    public static LibraryTaskDescriptor takeAlongAxis(BFloat16Array x, IntArray indices, BFloat16Array out, int outer, int len, int m, int inner) {
        return Mlx.task("take_along_axis", 2, x, indices, out, outer, len, m, inner);
    }

    /**
     * {@code out[o, i, j] = x[o, indices[o, i, j], j]} for {@code x} viewed as {@code [outer, len, inner]}
     * and {@code indices}, {@code out} as {@code [outer, m, inner]}.
     */
    @MlxOp("mlx_take_along_axis")
    public static LibraryTaskDescriptor takeAlongAxis(IntArray x, IntArray indices, IntArray out, int outer, int len, int m, int inner) {
        return Mlx.task("take_along_axis", 2, x, indices, out, outer, len, m, inner);
    }

    /**
     * {@code out} = {@code x} with {@code out[o, indices[o, i, j], j] = values[o, i, j]} ({@code indices}
     * and {@code values} viewed as {@code [outer, m, inner]}).
     */
    @MlxOp("mlx_put_along_axis")
    public static LibraryTaskDescriptor putAlongAxis(FloatArray x, IntArray indices, FloatArray values, FloatArray out, int outer, int len, int m, int inner) {
        return Mlx.task("put_along_axis", 3, x, indices, values, out, outer, len, m, inner);
    }

    /**
     * {@code out} = {@code x} with {@code out[o, indices[o, i, j], j] = values[o, i, j]} ({@code indices}
     * and {@code values} viewed as {@code [outer, m, inner]}).
     */
    @MlxOp("mlx_put_along_axis")
    public static LibraryTaskDescriptor putAlongAxis(HalfFloatArray x, IntArray indices, HalfFloatArray values, HalfFloatArray out, int outer, int len, int m, int inner) {
        return Mlx.task("put_along_axis", 3, x, indices, values, out, outer, len, m, inner);
    }

    /**
     * {@code out} = {@code x} with {@code out[o, indices[o, i, j], j] = values[o, i, j]} ({@code indices}
     * and {@code values} viewed as {@code [outer, m, inner]}).
     */
    @MlxOp("mlx_put_along_axis")
    public static LibraryTaskDescriptor putAlongAxis(BFloat16Array x, IntArray indices, BFloat16Array values, BFloat16Array out, int outer, int len, int m, int inner) {
        return Mlx.task("put_along_axis", 3, x, indices, values, out, outer, len, m, inner);
    }

    /**
     * {@code out} = {@code x} with {@code out[o, indices[o, i, j], j] = values[o, i, j]} ({@code indices}
     * and {@code values} viewed as {@code [outer, m, inner]}).
     */
    @MlxOp("mlx_put_along_axis")
    public static LibraryTaskDescriptor putAlongAxis(IntArray x, IntArray indices, IntArray values, IntArray out, int outer, int len, int m, int inner) {
        return Mlx.task("put_along_axis", 3, x, indices, values, out, outer, len, m, inner);
    }

    /** {@code out} = {@code x} with {@code values[o, i, j]} added at {@code out[o, indices[o, i, j], j]} (repeated indices accumulate). */
    @MlxOp("mlx_scatter_add_axis")
    public static LibraryTaskDescriptor scatterAddAxis(FloatArray x, IntArray indices, FloatArray values, FloatArray out, int outer, int len, int m, int inner) {
        return Mlx.task("scatter_add_axis", 3, x, indices, values, out, outer, len, m, inner);
    }

    /** {@code out} = {@code x} with {@code values[o, i, j]} added at {@code out[o, indices[o, i, j], j]} (repeated indices accumulate). */
    @MlxOp("mlx_scatter_add_axis")
    public static LibraryTaskDescriptor scatterAddAxis(HalfFloatArray x, IntArray indices, HalfFloatArray values, HalfFloatArray out, int outer, int len, int m, int inner) {
        return Mlx.task("scatter_add_axis", 3, x, indices, values, out, outer, len, m, inner);
    }

    /** {@code out} = {@code x} with {@code values[o, i, j]} added at {@code out[o, indices[o, i, j], j]} (repeated indices accumulate). */
    @MlxOp("mlx_scatter_add_axis")
    public static LibraryTaskDescriptor scatterAddAxis(BFloat16Array x, IntArray indices, BFloat16Array values, BFloat16Array out, int outer, int len, int m, int inner) {
        return Mlx.task("scatter_add_axis", 3, x, indices, values, out, outer, len, m, inner);
    }

    /** {@code out} = {@code x} with {@code values[o, i, j]} added at {@code out[o, indices[o, i, j], j]} (repeated indices accumulate). */
    @MlxOp("mlx_scatter_add_axis")
    public static LibraryTaskDescriptor scatterAddAxis(IntArray x, IntArray indices, IntArray values, IntArray out, int outer, int len, int m, int inner) {
        return Mlx.task("scatter_add_axis", 3, x, indices, values, out, outer, len, m, inner);
    }

    /** {@code out[i] = x[rows[i], cols[i]]} for {@code x} viewed as {@code [rowCount, colCount]}. */
    @MlxOp("mlx_gather")
    public static LibraryTaskDescriptor gather(FloatArray x, IntArray rows, IntArray cols, FloatArray out, int rowCount, int colCount) {
        return Mlx.task("gather", 3, x, rows, cols, out, rowCount, colCount);
    }

    /** {@code out[i] = x[rows[i], cols[i]]} for {@code x} viewed as {@code [rowCount, colCount]}. */
    @MlxOp("mlx_gather")
    public static LibraryTaskDescriptor gather(HalfFloatArray x, IntArray rows, IntArray cols, HalfFloatArray out, int rowCount, int colCount) {
        return Mlx.task("gather", 3, x, rows, cols, out, rowCount, colCount);
    }

    /** {@code out[i] = x[rows[i], cols[i]]} for {@code x} viewed as {@code [rowCount, colCount]}. */
    @MlxOp("mlx_gather")
    public static LibraryTaskDescriptor gather(BFloat16Array x, IntArray rows, IntArray cols, BFloat16Array out, int rowCount, int colCount) {
        return Mlx.task("gather", 3, x, rows, cols, out, rowCount, colCount);
    }

    /** {@code out[i] = x[rows[i], cols[i]]} for {@code x} viewed as {@code [rowCount, colCount]}. */
    @MlxOp("mlx_gather")
    public static LibraryTaskDescriptor gather(IntArray x, IntArray rows, IntArray cols, IntArray out, int rowCount, int colCount) {
        return Mlx.task("gather", 3, x, rows, cols, out, rowCount, colCount);
    }

    /**
     * {@code out[i, r, c] = x[indices[i] + r, c]}: windows of {@code sliceRows} rows of {@code x} viewed
     * as {@code [rows, cols]}, starting at each index.
     */
    @MlxOp("mlx_gather_single")
    public static LibraryTaskDescriptor gatherRows(FloatArray x, IntArray indices, FloatArray out, int rows, int cols, int sliceRows) {
        return Mlx.task("gather_single", 2, x, indices, out, rows, cols, sliceRows);
    }

    /**
     * {@code out[i, r, c] = x[indices[i] + r, c]}: windows of {@code sliceRows} rows of {@code x} viewed
     * as {@code [rows, cols]}, starting at each index.
     */
    @MlxOp("mlx_gather_single")
    public static LibraryTaskDescriptor gatherRows(HalfFloatArray x, IntArray indices, HalfFloatArray out, int rows, int cols, int sliceRows) {
        return Mlx.task("gather_single", 2, x, indices, out, rows, cols, sliceRows);
    }

    /**
     * {@code out[i, r, c] = x[indices[i] + r, c]}: windows of {@code sliceRows} rows of {@code x} viewed
     * as {@code [rows, cols]}, starting at each index.
     */
    @MlxOp("mlx_gather_single")
    public static LibraryTaskDescriptor gatherRows(BFloat16Array x, IntArray indices, BFloat16Array out, int rows, int cols, int sliceRows) {
        return Mlx.task("gather_single", 2, x, indices, out, rows, cols, sliceRows);
    }

    /**
     * {@code out[i, r, c] = x[indices[i] + r, c]}: windows of {@code sliceRows} rows of {@code x} viewed
     * as {@code [rows, cols]}, starting at each index.
     */
    @MlxOp("mlx_gather_single")
    public static LibraryTaskDescriptor gatherRows(IntArray x, IntArray indices, IntArray out, int rows, int cols, int sliceRows) {
        return Mlx.task("gather_single", 2, x, indices, out, rows, cols, sliceRows);
    }

    /** {@code out} = {@code x} (viewed as {@code [rowCount, colCount]}) with each {@code out[rows[i], cols[i]]} set to {@code updates[i]}. */
    @MlxOp("mlx_scatter")
    public static LibraryTaskDescriptor scatter(FloatArray x, IntArray rows, IntArray cols, FloatArray updates, FloatArray out, int rowCount, int colCount) {
        return Mlx.task("scatter", 4, x, rows, cols, updates, out, rowCount, colCount);
    }

    /** {@code out} = {@code x} (viewed as {@code [rowCount, colCount]}) with each {@code out[rows[i], cols[i]]} set to {@code updates[i]}. */
    @MlxOp("mlx_scatter")
    public static LibraryTaskDescriptor scatter(HalfFloatArray x, IntArray rows, IntArray cols, HalfFloatArray updates, HalfFloatArray out, int rowCount, int colCount) {
        return Mlx.task("scatter", 4, x, rows, cols, updates, out, rowCount, colCount);
    }

    /** {@code out} = {@code x} (viewed as {@code [rowCount, colCount]}) with each {@code out[rows[i], cols[i]]} set to {@code updates[i]}. */
    @MlxOp("mlx_scatter")
    public static LibraryTaskDescriptor scatter(BFloat16Array x, IntArray rows, IntArray cols, BFloat16Array updates, BFloat16Array out, int rowCount, int colCount) {
        return Mlx.task("scatter", 4, x, rows, cols, updates, out, rowCount, colCount);
    }

    /** {@code out} = {@code x} (viewed as {@code [rowCount, colCount]}) with each {@code out[rows[i], cols[i]]} set to {@code updates[i]}. */
    @MlxOp("mlx_scatter")
    public static LibraryTaskDescriptor scatter(IntArray x, IntArray rows, IntArray cols, IntArray updates, IntArray out, int rowCount, int colCount) {
        return Mlx.task("scatter", 4, x, rows, cols, updates, out, rowCount, colCount);
    }

    /** {@code out} = {@code x} (viewed as {@code [rows, cols]}) with each row {@code out[indices[i], :]} set to {@code updates[i, :]}. */
    @MlxOp("mlx_scatter_single")
    public static LibraryTaskDescriptor scatterRows(FloatArray x, IntArray indices, FloatArray updates, FloatArray out, int rows, int cols) {
        return Mlx.task("scatter_single", 3, x, indices, updates, out, rows, cols);
    }

    /** {@code out} = {@code x} (viewed as {@code [rows, cols]}) with each row {@code out[indices[i], :]} set to {@code updates[i, :]}. */
    @MlxOp("mlx_scatter_single")
    public static LibraryTaskDescriptor scatterRows(HalfFloatArray x, IntArray indices, HalfFloatArray updates, HalfFloatArray out, int rows, int cols) {
        return Mlx.task("scatter_single", 3, x, indices, updates, out, rows, cols);
    }

    /** {@code out} = {@code x} (viewed as {@code [rows, cols]}) with each row {@code out[indices[i], :]} set to {@code updates[i, :]}. */
    @MlxOp("mlx_scatter_single")
    public static LibraryTaskDescriptor scatterRows(BFloat16Array x, IntArray indices, BFloat16Array updates, BFloat16Array out, int rows, int cols) {
        return Mlx.task("scatter_single", 3, x, indices, updates, out, rows, cols);
    }

    /** {@code out} = {@code x} (viewed as {@code [rows, cols]}) with each row {@code out[indices[i], :]} set to {@code updates[i, :]}. */
    @MlxOp("mlx_scatter_single")
    public static LibraryTaskDescriptor scatterRows(IntArray x, IntArray indices, IntArray updates, IntArray out, int rows, int cols) {
        return Mlx.task("scatter_single", 3, x, indices, updates, out, rows, cols);
    }

    /** {@code out} = {@code x} (viewed as {@code [rowCount, colCount]}) with each {@code out[rows[i], cols[i]]} incremented by {@code updates[i]}. */
    @MlxOp("mlx_scatter_add")
    public static LibraryTaskDescriptor scatterAdd(FloatArray x, IntArray rows, IntArray cols, FloatArray updates, FloatArray out, int rowCount, int colCount) {
        return Mlx.task("scatter_add", 4, x, rows, cols, updates, out, rowCount, colCount);
    }

    /** {@code out} = {@code x} (viewed as {@code [rowCount, colCount]}) with each {@code out[rows[i], cols[i]]} incremented by {@code updates[i]}. */
    @MlxOp("mlx_scatter_add")
    public static LibraryTaskDescriptor scatterAdd(HalfFloatArray x, IntArray rows, IntArray cols, HalfFloatArray updates, HalfFloatArray out, int rowCount, int colCount) {
        return Mlx.task("scatter_add", 4, x, rows, cols, updates, out, rowCount, colCount);
    }

    /** {@code out} = {@code x} (viewed as {@code [rowCount, colCount]}) with each {@code out[rows[i], cols[i]]} incremented by {@code updates[i]}. */
    @MlxOp("mlx_scatter_add")
    public static LibraryTaskDescriptor scatterAdd(BFloat16Array x, IntArray rows, IntArray cols, BFloat16Array updates, BFloat16Array out, int rowCount, int colCount) {
        return Mlx.task("scatter_add", 4, x, rows, cols, updates, out, rowCount, colCount);
    }

    /** {@code out} = {@code x} (viewed as {@code [rowCount, colCount]}) with each {@code out[rows[i], cols[i]]} incremented by {@code updates[i]}. */
    @MlxOp("mlx_scatter_add")
    public static LibraryTaskDescriptor scatterAdd(IntArray x, IntArray rows, IntArray cols, IntArray updates, IntArray out, int rowCount, int colCount) {
        return Mlx.task("scatter_add", 4, x, rows, cols, updates, out, rowCount, colCount);
    }

    /** {@code out} = {@code x} (viewed as {@code [rows, cols]}) with each row {@code out[indices[i], :]} incremented by {@code updates[i, :]}. */
    @MlxOp("mlx_scatter_add_single")
    public static LibraryTaskDescriptor scatterAddRows(FloatArray x, IntArray indices, FloatArray updates, FloatArray out, int rows, int cols) {
        return Mlx.task("scatter_add_single", 3, x, indices, updates, out, rows, cols);
    }

    /** {@code out} = {@code x} (viewed as {@code [rows, cols]}) with each row {@code out[indices[i], :]} incremented by {@code updates[i, :]}. */
    @MlxOp("mlx_scatter_add_single")
    public static LibraryTaskDescriptor scatterAddRows(HalfFloatArray x, IntArray indices, HalfFloatArray updates, HalfFloatArray out, int rows, int cols) {
        return Mlx.task("scatter_add_single", 3, x, indices, updates, out, rows, cols);
    }

    /** {@code out} = {@code x} (viewed as {@code [rows, cols]}) with each row {@code out[indices[i], :]} incremented by {@code updates[i, :]}. */
    @MlxOp("mlx_scatter_add_single")
    public static LibraryTaskDescriptor scatterAddRows(BFloat16Array x, IntArray indices, BFloat16Array updates, BFloat16Array out, int rows, int cols) {
        return Mlx.task("scatter_add_single", 3, x, indices, updates, out, rows, cols);
    }

    /** {@code out} = {@code x} (viewed as {@code [rows, cols]}) with each row {@code out[indices[i], :]} incremented by {@code updates[i, :]}. */
    @MlxOp("mlx_scatter_add_single")
    public static LibraryTaskDescriptor scatterAddRows(IntArray x, IntArray indices, IntArray updates, IntArray out, int rows, int cols) {
        return Mlx.task("scatter_add_single", 3, x, indices, updates, out, rows, cols);
    }

    /**
     * {@code out} = {@code x} (viewed as {@code [rowCount, colCount]}) with each {@code out[rows[i],
     * cols[i]]} raised to at least {@code updates[i]}.
     */
    @MlxOp("mlx_scatter_max")
    public static LibraryTaskDescriptor scatterMax(FloatArray x, IntArray rows, IntArray cols, FloatArray updates, FloatArray out, int rowCount, int colCount) {
        return Mlx.task("scatter_max", 4, x, rows, cols, updates, out, rowCount, colCount);
    }

    /**
     * {@code out} = {@code x} (viewed as {@code [rowCount, colCount]}) with each {@code out[rows[i],
     * cols[i]]} raised to at least {@code updates[i]}.
     */
    @MlxOp("mlx_scatter_max")
    public static LibraryTaskDescriptor scatterMax(HalfFloatArray x, IntArray rows, IntArray cols, HalfFloatArray updates, HalfFloatArray out, int rowCount, int colCount) {
        return Mlx.task("scatter_max", 4, x, rows, cols, updates, out, rowCount, colCount);
    }

    /**
     * {@code out} = {@code x} (viewed as {@code [rowCount, colCount]}) with each {@code out[rows[i],
     * cols[i]]} raised to at least {@code updates[i]}.
     */
    @MlxOp("mlx_scatter_max")
    public static LibraryTaskDescriptor scatterMax(BFloat16Array x, IntArray rows, IntArray cols, BFloat16Array updates, BFloat16Array out, int rowCount, int colCount) {
        return Mlx.task("scatter_max", 4, x, rows, cols, updates, out, rowCount, colCount);
    }

    /**
     * {@code out} = {@code x} (viewed as {@code [rowCount, colCount]}) with each {@code out[rows[i],
     * cols[i]]} raised to at least {@code updates[i]}.
     */
    @MlxOp("mlx_scatter_max")
    public static LibraryTaskDescriptor scatterMax(IntArray x, IntArray rows, IntArray cols, IntArray updates, IntArray out, int rowCount, int colCount) {
        return Mlx.task("scatter_max", 4, x, rows, cols, updates, out, rowCount, colCount);
    }

    /** {@code out} = {@code x} (viewed as {@code [rows, cols]}) with each row {@code out[indices[i], :]} raised to at least {@code updates[i, :]}. */
    @MlxOp("mlx_scatter_max_single")
    public static LibraryTaskDescriptor scatterMaxRows(FloatArray x, IntArray indices, FloatArray updates, FloatArray out, int rows, int cols) {
        return Mlx.task("scatter_max_single", 3, x, indices, updates, out, rows, cols);
    }

    /** {@code out} = {@code x} (viewed as {@code [rows, cols]}) with each row {@code out[indices[i], :]} raised to at least {@code updates[i, :]}. */
    @MlxOp("mlx_scatter_max_single")
    public static LibraryTaskDescriptor scatterMaxRows(HalfFloatArray x, IntArray indices, HalfFloatArray updates, HalfFloatArray out, int rows, int cols) {
        return Mlx.task("scatter_max_single", 3, x, indices, updates, out, rows, cols);
    }

    /** {@code out} = {@code x} (viewed as {@code [rows, cols]}) with each row {@code out[indices[i], :]} raised to at least {@code updates[i, :]}. */
    @MlxOp("mlx_scatter_max_single")
    public static LibraryTaskDescriptor scatterMaxRows(BFloat16Array x, IntArray indices, BFloat16Array updates, BFloat16Array out, int rows, int cols) {
        return Mlx.task("scatter_max_single", 3, x, indices, updates, out, rows, cols);
    }

    /** {@code out} = {@code x} (viewed as {@code [rows, cols]}) with each row {@code out[indices[i], :]} raised to at least {@code updates[i, :]}. */
    @MlxOp("mlx_scatter_max_single")
    public static LibraryTaskDescriptor scatterMaxRows(IntArray x, IntArray indices, IntArray updates, IntArray out, int rows, int cols) {
        return Mlx.task("scatter_max_single", 3, x, indices, updates, out, rows, cols);
    }

    /**
     * {@code out} = {@code x} (viewed as {@code [rowCount, colCount]}) with each {@code out[rows[i],
     * cols[i]]} lowered to at most {@code updates[i]}.
     */
    @MlxOp("mlx_scatter_min")
    public static LibraryTaskDescriptor scatterMin(FloatArray x, IntArray rows, IntArray cols, FloatArray updates, FloatArray out, int rowCount, int colCount) {
        return Mlx.task("scatter_min", 4, x, rows, cols, updates, out, rowCount, colCount);
    }

    /**
     * {@code out} = {@code x} (viewed as {@code [rowCount, colCount]}) with each {@code out[rows[i],
     * cols[i]]} lowered to at most {@code updates[i]}.
     */
    @MlxOp("mlx_scatter_min")
    public static LibraryTaskDescriptor scatterMin(HalfFloatArray x, IntArray rows, IntArray cols, HalfFloatArray updates, HalfFloatArray out, int rowCount, int colCount) {
        return Mlx.task("scatter_min", 4, x, rows, cols, updates, out, rowCount, colCount);
    }

    /**
     * {@code out} = {@code x} (viewed as {@code [rowCount, colCount]}) with each {@code out[rows[i],
     * cols[i]]} lowered to at most {@code updates[i]}.
     */
    @MlxOp("mlx_scatter_min")
    public static LibraryTaskDescriptor scatterMin(BFloat16Array x, IntArray rows, IntArray cols, BFloat16Array updates, BFloat16Array out, int rowCount, int colCount) {
        return Mlx.task("scatter_min", 4, x, rows, cols, updates, out, rowCount, colCount);
    }

    /**
     * {@code out} = {@code x} (viewed as {@code [rowCount, colCount]}) with each {@code out[rows[i],
     * cols[i]]} lowered to at most {@code updates[i]}.
     */
    @MlxOp("mlx_scatter_min")
    public static LibraryTaskDescriptor scatterMin(IntArray x, IntArray rows, IntArray cols, IntArray updates, IntArray out, int rowCount, int colCount) {
        return Mlx.task("scatter_min", 4, x, rows, cols, updates, out, rowCount, colCount);
    }

    /** {@code out} = {@code x} (viewed as {@code [rows, cols]}) with each row {@code out[indices[i], :]} lowered to at most {@code updates[i, :]}. */
    @MlxOp("mlx_scatter_min_single")
    public static LibraryTaskDescriptor scatterMinRows(FloatArray x, IntArray indices, FloatArray updates, FloatArray out, int rows, int cols) {
        return Mlx.task("scatter_min_single", 3, x, indices, updates, out, rows, cols);
    }

    /** {@code out} = {@code x} (viewed as {@code [rows, cols]}) with each row {@code out[indices[i], :]} lowered to at most {@code updates[i, :]}. */
    @MlxOp("mlx_scatter_min_single")
    public static LibraryTaskDescriptor scatterMinRows(HalfFloatArray x, IntArray indices, HalfFloatArray updates, HalfFloatArray out, int rows, int cols) {
        return Mlx.task("scatter_min_single", 3, x, indices, updates, out, rows, cols);
    }

    /** {@code out} = {@code x} (viewed as {@code [rows, cols]}) with each row {@code out[indices[i], :]} lowered to at most {@code updates[i, :]}. */
    @MlxOp("mlx_scatter_min_single")
    public static LibraryTaskDescriptor scatterMinRows(BFloat16Array x, IntArray indices, BFloat16Array updates, BFloat16Array out, int rows, int cols) {
        return Mlx.task("scatter_min_single", 3, x, indices, updates, out, rows, cols);
    }

    /** {@code out} = {@code x} (viewed as {@code [rows, cols]}) with each row {@code out[indices[i], :]} lowered to at most {@code updates[i, :]}. */
    @MlxOp("mlx_scatter_min_single")
    public static LibraryTaskDescriptor scatterMinRows(IntArray x, IntArray indices, IntArray updates, IntArray out, int rows, int cols) {
        return Mlx.task("scatter_min_single", 3, x, indices, updates, out, rows, cols);
    }

    /** {@code out} = {@code x} (viewed as {@code [rowCount, colCount]}) with each {@code out[rows[i], cols[i]]} multiplied by {@code updates[i]}. */
    @MlxOp("mlx_scatter_prod")
    public static LibraryTaskDescriptor scatterProd(FloatArray x, IntArray rows, IntArray cols, FloatArray updates, FloatArray out, int rowCount, int colCount) {
        return Mlx.task("scatter_prod", 4, x, rows, cols, updates, out, rowCount, colCount);
    }

    /** {@code out} = {@code x} (viewed as {@code [rowCount, colCount]}) with each {@code out[rows[i], cols[i]]} multiplied by {@code updates[i]}. */
    @MlxOp("mlx_scatter_prod")
    public static LibraryTaskDescriptor scatterProd(HalfFloatArray x, IntArray rows, IntArray cols, HalfFloatArray updates, HalfFloatArray out, int rowCount, int colCount) {
        return Mlx.task("scatter_prod", 4, x, rows, cols, updates, out, rowCount, colCount);
    }

    /** {@code out} = {@code x} (viewed as {@code [rowCount, colCount]}) with each {@code out[rows[i], cols[i]]} multiplied by {@code updates[i]}. */
    @MlxOp("mlx_scatter_prod")
    public static LibraryTaskDescriptor scatterProd(BFloat16Array x, IntArray rows, IntArray cols, BFloat16Array updates, BFloat16Array out, int rowCount, int colCount) {
        return Mlx.task("scatter_prod", 4, x, rows, cols, updates, out, rowCount, colCount);
    }

    /** {@code out} = {@code x} (viewed as {@code [rowCount, colCount]}) with each {@code out[rows[i], cols[i]]} multiplied by {@code updates[i]}. */
    @MlxOp("mlx_scatter_prod")
    public static LibraryTaskDescriptor scatterProd(IntArray x, IntArray rows, IntArray cols, IntArray updates, IntArray out, int rowCount, int colCount) {
        return Mlx.task("scatter_prod", 4, x, rows, cols, updates, out, rowCount, colCount);
    }

    /** {@code out} = {@code x} (viewed as {@code [rows, cols]}) with each row {@code out[indices[i], :]} multiplied by {@code updates[i, :]}. */
    @MlxOp("mlx_scatter_prod_single")
    public static LibraryTaskDescriptor scatterProdRows(FloatArray x, IntArray indices, FloatArray updates, FloatArray out, int rows, int cols) {
        return Mlx.task("scatter_prod_single", 3, x, indices, updates, out, rows, cols);
    }

    /** {@code out} = {@code x} (viewed as {@code [rows, cols]}) with each row {@code out[indices[i], :]} multiplied by {@code updates[i, :]}. */
    @MlxOp("mlx_scatter_prod_single")
    public static LibraryTaskDescriptor scatterProdRows(HalfFloatArray x, IntArray indices, HalfFloatArray updates, HalfFloatArray out, int rows, int cols) {
        return Mlx.task("scatter_prod_single", 3, x, indices, updates, out, rows, cols);
    }

    /** {@code out} = {@code x} (viewed as {@code [rows, cols]}) with each row {@code out[indices[i], :]} multiplied by {@code updates[i, :]}. */
    @MlxOp("mlx_scatter_prod_single")
    public static LibraryTaskDescriptor scatterProdRows(BFloat16Array x, IntArray indices, BFloat16Array updates, BFloat16Array out, int rows, int cols) {
        return Mlx.task("scatter_prod_single", 3, x, indices, updates, out, rows, cols);
    }

    /** {@code out} = {@code x} (viewed as {@code [rows, cols]}) with each row {@code out[indices[i], :]} multiplied by {@code updates[i, :]}. */
    @MlxOp("mlx_scatter_prod_single")
    public static LibraryTaskDescriptor scatterProdRows(IntArray x, IntArray indices, IntArray updates, IntArray out, int rows, int cols) {
        return Mlx.task("scatter_prod_single", 3, x, indices, updates, out, rows, cols);
    }

    /** {@code out = x[r0:r1:rowStep, c0:c1:colStep]} for {@code x} viewed as {@code [rows, cols]}. */
    @MlxOp("mlx_slice")
    public static LibraryTaskDescriptor slice(FloatArray x, FloatArray out, int rows, int cols, int r0, int r1, int rowStep, int c0, int c1, int colStep) {
        return Mlx.task("slice", 1, x, out, rows, cols, r0, r1, rowStep, c0, c1, colStep);
    }

    /** {@code out = x[r0:r1:rowStep, c0:c1:colStep]} for {@code x} viewed as {@code [rows, cols]}. */
    @MlxOp("mlx_slice")
    public static LibraryTaskDescriptor slice(HalfFloatArray x, HalfFloatArray out, int rows, int cols, int r0, int r1, int rowStep, int c0, int c1, int colStep) {
        return Mlx.task("slice", 1, x, out, rows, cols, r0, r1, rowStep, c0, c1, colStep);
    }

    /** {@code out = x[r0:r1:rowStep, c0:c1:colStep]} for {@code x} viewed as {@code [rows, cols]}. */
    @MlxOp("mlx_slice")
    public static LibraryTaskDescriptor slice(BFloat16Array x, BFloat16Array out, int rows, int cols, int r0, int r1, int rowStep, int c0, int c1, int colStep) {
        return Mlx.task("slice", 1, x, out, rows, cols, r0, r1, rowStep, c0, c1, colStep);
    }

    /** {@code out = x[r0:r1:rowStep, c0:c1:colStep]} for {@code x} viewed as {@code [rows, cols]}. */
    @MlxOp("mlx_slice")
    public static LibraryTaskDescriptor slice(IntArray x, IntArray out, int rows, int cols, int r0, int r1, int rowStep, int c0, int c1, int colStep) {
        return Mlx.task("slice", 1, x, out, rows, cols, r0, r1, rowStep, c0, c1, colStep);
    }

    /**
     * {@code out = x[start[0] : start[0] + sliceRows, :]} for {@code x} viewed as {@code [rows, cols]},
     * with the offset read on the device (a KV-cache read at the current position).
     */
    @MlxOp("mlx_slice_dynamic")
    public static LibraryTaskDescriptor sliceRowsAt(FloatArray x, IntArray start, FloatArray out, int rows, int cols, int sliceRows) {
        return Mlx.task("slice_dynamic", 2, x, start, out, rows, cols, sliceRows);
    }

    /**
     * {@code out = x[start[0] : start[0] + sliceRows, :]} for {@code x} viewed as {@code [rows, cols]},
     * with the offset read on the device (a KV-cache read at the current position).
     */
    @MlxOp("mlx_slice_dynamic")
    public static LibraryTaskDescriptor sliceRowsAt(HalfFloatArray x, IntArray start, HalfFloatArray out, int rows, int cols, int sliceRows) {
        return Mlx.task("slice_dynamic", 2, x, start, out, rows, cols, sliceRows);
    }

    /**
     * {@code out = x[start[0] : start[0] + sliceRows, :]} for {@code x} viewed as {@code [rows, cols]},
     * with the offset read on the device (a KV-cache read at the current position).
     */
    @MlxOp("mlx_slice_dynamic")
    public static LibraryTaskDescriptor sliceRowsAt(BFloat16Array x, IntArray start, BFloat16Array out, int rows, int cols, int sliceRows) {
        return Mlx.task("slice_dynamic", 2, x, start, out, rows, cols, sliceRows);
    }

    /**
     * {@code out = x[start[0] : start[0] + sliceRows, :]} for {@code x} viewed as {@code [rows, cols]},
     * with the offset read on the device (a KV-cache read at the current position).
     */
    @MlxOp("mlx_slice_dynamic")
    public static LibraryTaskDescriptor sliceRowsAt(IntArray x, IntArray start, IntArray out, int rows, int cols, int sliceRows) {
        return Mlx.task("slice_dynamic", 2, x, start, out, rows, cols, sliceRows);
    }

    /**
     * {@code out} = {@code x} (viewed as {@code [rows, cols]}) with the block {@code out[r0 : r0 +
     * update.rows, c0 : c0 + update.cols]} replaced by {@code update} (viewed as {@code [updateRows,
     * updateCols]}).
     */
    @MlxOp("mlx_slice_update")
    public static LibraryTaskDescriptor sliceUpdate(FloatArray x, FloatArray update, FloatArray out, int rows, int cols, int r0, int c0, int updateRows, int updateCols) {
        return Mlx.task("slice_update", 2, x, update, out, rows, cols, r0, c0, updateRows, updateCols);
    }

    /**
     * {@code out} = {@code x} (viewed as {@code [rows, cols]}) with the block {@code out[r0 : r0 +
     * update.rows, c0 : c0 + update.cols]} replaced by {@code update} (viewed as {@code [updateRows,
     * updateCols]}).
     */
    @MlxOp("mlx_slice_update")
    public static LibraryTaskDescriptor sliceUpdate(HalfFloatArray x, HalfFloatArray update, HalfFloatArray out, int rows, int cols, int r0, int c0, int updateRows, int updateCols) {
        return Mlx.task("slice_update", 2, x, update, out, rows, cols, r0, c0, updateRows, updateCols);
    }

    /**
     * {@code out} = {@code x} (viewed as {@code [rows, cols]}) with the block {@code out[r0 : r0 +
     * update.rows, c0 : c0 + update.cols]} replaced by {@code update} (viewed as {@code [updateRows,
     * updateCols]}).
     */
    @MlxOp("mlx_slice_update")
    public static LibraryTaskDescriptor sliceUpdate(BFloat16Array x, BFloat16Array update, BFloat16Array out, int rows, int cols, int r0, int c0, int updateRows, int updateCols) {
        return Mlx.task("slice_update", 2, x, update, out, rows, cols, r0, c0, updateRows, updateCols);
    }

    /**
     * {@code out} = {@code x} (viewed as {@code [rows, cols]}) with the block {@code out[r0 : r0 +
     * update.rows, c0 : c0 + update.cols]} replaced by {@code update} (viewed as {@code [updateRows,
     * updateCols]}).
     */
    @MlxOp("mlx_slice_update")
    public static LibraryTaskDescriptor sliceUpdate(IntArray x, IntArray update, IntArray out, int rows, int cols, int r0, int c0, int updateRows, int updateCols) {
        return Mlx.task("slice_update", 2, x, update, out, rows, cols, r0, c0, updateRows, updateCols);
    }

    /**
     * {@code out} = {@code x} (viewed as {@code [rows, cols]}) with the block {@code out[r0 : r0 +
     * update.rows, c0 : c0 + update.cols]} incremented by {@code update} (viewed as {@code [updateRows,
     * updateCols]}).
     */
    @MlxOp("mlx_slice_update_add")
    public static LibraryTaskDescriptor sliceUpdateAdd(FloatArray x, FloatArray update, FloatArray out, int rows, int cols, int r0, int c0, int updateRows, int updateCols) {
        return Mlx.task("slice_update_add", 2, x, update, out, rows, cols, r0, c0, updateRows, updateCols);
    }

    /**
     * {@code out} = {@code x} (viewed as {@code [rows, cols]}) with the block {@code out[r0 : r0 +
     * update.rows, c0 : c0 + update.cols]} incremented by {@code update} (viewed as {@code [updateRows,
     * updateCols]}).
     */
    @MlxOp("mlx_slice_update_add")
    public static LibraryTaskDescriptor sliceUpdateAdd(HalfFloatArray x, HalfFloatArray update, HalfFloatArray out, int rows, int cols, int r0, int c0, int updateRows, int updateCols) {
        return Mlx.task("slice_update_add", 2, x, update, out, rows, cols, r0, c0, updateRows, updateCols);
    }

    /**
     * {@code out} = {@code x} (viewed as {@code [rows, cols]}) with the block {@code out[r0 : r0 +
     * update.rows, c0 : c0 + update.cols]} incremented by {@code update} (viewed as {@code [updateRows,
     * updateCols]}).
     */
    @MlxOp("mlx_slice_update_add")
    public static LibraryTaskDescriptor sliceUpdateAdd(BFloat16Array x, BFloat16Array update, BFloat16Array out, int rows, int cols, int r0, int c0, int updateRows, int updateCols) {
        return Mlx.task("slice_update_add", 2, x, update, out, rows, cols, r0, c0, updateRows, updateCols);
    }

    /**
     * {@code out} = {@code x} (viewed as {@code [rows, cols]}) with the block {@code out[r0 : r0 +
     * update.rows, c0 : c0 + update.cols]} incremented by {@code update} (viewed as {@code [updateRows,
     * updateCols]}).
     */
    @MlxOp("mlx_slice_update_add")
    public static LibraryTaskDescriptor sliceUpdateAdd(IntArray x, IntArray update, IntArray out, int rows, int cols, int r0, int c0, int updateRows, int updateCols) {
        return Mlx.task("slice_update_add", 2, x, update, out, rows, cols, r0, c0, updateRows, updateCols);
    }

    /**
     * {@code out} = {@code x} (viewed as {@code [rows, cols]}) with the block {@code out[r0 : r0 +
     * update.rows, c0 : c0 + update.cols]} raised to at least {@code update} (viewed as {@code
     * [updateRows, updateCols]}).
     */
    @MlxOp("mlx_slice_update_max")
    public static LibraryTaskDescriptor sliceUpdateMax(FloatArray x, FloatArray update, FloatArray out, int rows, int cols, int r0, int c0, int updateRows, int updateCols) {
        return Mlx.task("slice_update_max", 2, x, update, out, rows, cols, r0, c0, updateRows, updateCols);
    }

    /**
     * {@code out} = {@code x} (viewed as {@code [rows, cols]}) with the block {@code out[r0 : r0 +
     * update.rows, c0 : c0 + update.cols]} raised to at least {@code update} (viewed as {@code
     * [updateRows, updateCols]}).
     */
    @MlxOp("mlx_slice_update_max")
    public static LibraryTaskDescriptor sliceUpdateMax(HalfFloatArray x, HalfFloatArray update, HalfFloatArray out, int rows, int cols, int r0, int c0, int updateRows, int updateCols) {
        return Mlx.task("slice_update_max", 2, x, update, out, rows, cols, r0, c0, updateRows, updateCols);
    }

    /**
     * {@code out} = {@code x} (viewed as {@code [rows, cols]}) with the block {@code out[r0 : r0 +
     * update.rows, c0 : c0 + update.cols]} raised to at least {@code update} (viewed as {@code
     * [updateRows, updateCols]}).
     */
    @MlxOp("mlx_slice_update_max")
    public static LibraryTaskDescriptor sliceUpdateMax(BFloat16Array x, BFloat16Array update, BFloat16Array out, int rows, int cols, int r0, int c0, int updateRows, int updateCols) {
        return Mlx.task("slice_update_max", 2, x, update, out, rows, cols, r0, c0, updateRows, updateCols);
    }

    /**
     * {@code out} = {@code x} (viewed as {@code [rows, cols]}) with the block {@code out[r0 : r0 +
     * update.rows, c0 : c0 + update.cols]} raised to at least {@code update} (viewed as {@code
     * [updateRows, updateCols]}).
     */
    @MlxOp("mlx_slice_update_max")
    public static LibraryTaskDescriptor sliceUpdateMax(IntArray x, IntArray update, IntArray out, int rows, int cols, int r0, int c0, int updateRows, int updateCols) {
        return Mlx.task("slice_update_max", 2, x, update, out, rows, cols, r0, c0, updateRows, updateCols);
    }

    /**
     * {@code out} = {@code x} (viewed as {@code [rows, cols]}) with the block {@code out[r0 : r0 +
     * update.rows, c0 : c0 + update.cols]} lowered to at most {@code update} (viewed as {@code
     * [updateRows, updateCols]}).
     */
    @MlxOp("mlx_slice_update_min")
    public static LibraryTaskDescriptor sliceUpdateMin(FloatArray x, FloatArray update, FloatArray out, int rows, int cols, int r0, int c0, int updateRows, int updateCols) {
        return Mlx.task("slice_update_min", 2, x, update, out, rows, cols, r0, c0, updateRows, updateCols);
    }

    /**
     * {@code out} = {@code x} (viewed as {@code [rows, cols]}) with the block {@code out[r0 : r0 +
     * update.rows, c0 : c0 + update.cols]} lowered to at most {@code update} (viewed as {@code
     * [updateRows, updateCols]}).
     */
    @MlxOp("mlx_slice_update_min")
    public static LibraryTaskDescriptor sliceUpdateMin(HalfFloatArray x, HalfFloatArray update, HalfFloatArray out, int rows, int cols, int r0, int c0, int updateRows, int updateCols) {
        return Mlx.task("slice_update_min", 2, x, update, out, rows, cols, r0, c0, updateRows, updateCols);
    }

    /**
     * {@code out} = {@code x} (viewed as {@code [rows, cols]}) with the block {@code out[r0 : r0 +
     * update.rows, c0 : c0 + update.cols]} lowered to at most {@code update} (viewed as {@code
     * [updateRows, updateCols]}).
     */
    @MlxOp("mlx_slice_update_min")
    public static LibraryTaskDescriptor sliceUpdateMin(BFloat16Array x, BFloat16Array update, BFloat16Array out, int rows, int cols, int r0, int c0, int updateRows, int updateCols) {
        return Mlx.task("slice_update_min", 2, x, update, out, rows, cols, r0, c0, updateRows, updateCols);
    }

    /**
     * {@code out} = {@code x} (viewed as {@code [rows, cols]}) with the block {@code out[r0 : r0 +
     * update.rows, c0 : c0 + update.cols]} lowered to at most {@code update} (viewed as {@code
     * [updateRows, updateCols]}).
     */
    @MlxOp("mlx_slice_update_min")
    public static LibraryTaskDescriptor sliceUpdateMin(IntArray x, IntArray update, IntArray out, int rows, int cols, int r0, int c0, int updateRows, int updateCols) {
        return Mlx.task("slice_update_min", 2, x, update, out, rows, cols, r0, c0, updateRows, updateCols);
    }

    /**
     * {@code out} = {@code x} (viewed as {@code [rows, cols]}) with the block {@code out[r0 : r0 +
     * update.rows, c0 : c0 + update.cols]} multiplied by {@code update} (viewed as {@code [updateRows,
     * updateCols]}).
     */
    @MlxOp("mlx_slice_update_prod")
    public static LibraryTaskDescriptor sliceUpdateProd(FloatArray x, FloatArray update, FloatArray out, int rows, int cols, int r0, int c0, int updateRows, int updateCols) {
        return Mlx.task("slice_update_prod", 2, x, update, out, rows, cols, r0, c0, updateRows, updateCols);
    }

    /**
     * {@code out} = {@code x} (viewed as {@code [rows, cols]}) with the block {@code out[r0 : r0 +
     * update.rows, c0 : c0 + update.cols]} multiplied by {@code update} (viewed as {@code [updateRows,
     * updateCols]}).
     */
    @MlxOp("mlx_slice_update_prod")
    public static LibraryTaskDescriptor sliceUpdateProd(HalfFloatArray x, HalfFloatArray update, HalfFloatArray out, int rows, int cols, int r0, int c0, int updateRows, int updateCols) {
        return Mlx.task("slice_update_prod", 2, x, update, out, rows, cols, r0, c0, updateRows, updateCols);
    }

    /**
     * {@code out} = {@code x} (viewed as {@code [rows, cols]}) with the block {@code out[r0 : r0 +
     * update.rows, c0 : c0 + update.cols]} multiplied by {@code update} (viewed as {@code [updateRows,
     * updateCols]}).
     */
    @MlxOp("mlx_slice_update_prod")
    public static LibraryTaskDescriptor sliceUpdateProd(BFloat16Array x, BFloat16Array update, BFloat16Array out, int rows, int cols, int r0, int c0, int updateRows, int updateCols) {
        return Mlx.task("slice_update_prod", 2, x, update, out, rows, cols, r0, c0, updateRows, updateCols);
    }

    /**
     * {@code out} = {@code x} (viewed as {@code [rows, cols]}) with the block {@code out[r0 : r0 +
     * update.rows, c0 : c0 + update.cols]} multiplied by {@code update} (viewed as {@code [updateRows,
     * updateCols]}).
     */
    @MlxOp("mlx_slice_update_prod")
    public static LibraryTaskDescriptor sliceUpdateProd(IntArray x, IntArray update, IntArray out, int rows, int cols, int r0, int c0, int updateRows, int updateCols) {
        return Mlx.task("slice_update_prod", 2, x, update, out, rows, cols, r0, c0, updateRows, updateCols);
    }

    /**
     * {@code out} = {@code x} (viewed as {@code [rows, cols]}) with rows {@code start[0] ..} replaced by
     * {@code update} ({@code [updateRows, cols]}), the offset read on the device (a KV-cache write at the
     * current position).
     */
    @MlxOp("mlx_slice_update_dynamic")
    public static LibraryTaskDescriptor sliceUpdateRowsAt(FloatArray x, FloatArray update, IntArray start, FloatArray out, int rows, int cols, int updateRows) {
        return Mlx.task("slice_update_dynamic", 3, x, update, start, out, rows, cols, updateRows);
    }

    /**
     * {@code out} = {@code x} (viewed as {@code [rows, cols]}) with rows {@code start[0] ..} replaced by
     * {@code update} ({@code [updateRows, cols]}), the offset read on the device (a KV-cache write at the
     * current position).
     */
    @MlxOp("mlx_slice_update_dynamic")
    public static LibraryTaskDescriptor sliceUpdateRowsAt(HalfFloatArray x, HalfFloatArray update, IntArray start, HalfFloatArray out, int rows, int cols, int updateRows) {
        return Mlx.task("slice_update_dynamic", 3, x, update, start, out, rows, cols, updateRows);
    }

    /**
     * {@code out} = {@code x} (viewed as {@code [rows, cols]}) with rows {@code start[0] ..} replaced by
     * {@code update} ({@code [updateRows, cols]}), the offset read on the device (a KV-cache write at the
     * current position).
     */
    @MlxOp("mlx_slice_update_dynamic")
    public static LibraryTaskDescriptor sliceUpdateRowsAt(BFloat16Array x, BFloat16Array update, IntArray start, BFloat16Array out, int rows, int cols, int updateRows) {
        return Mlx.task("slice_update_dynamic", 3, x, update, start, out, rows, cols, updateRows);
    }

    /**
     * {@code out} = {@code x} (viewed as {@code [rows, cols]}) with rows {@code start[0] ..} replaced by
     * {@code update} ({@code [updateRows, cols]}), the offset read on the device (a KV-cache write at the
     * current position).
     */
    @MlxOp("mlx_slice_update_dynamic")
    public static LibraryTaskDescriptor sliceUpdateRowsAt(IntArray x, IntArray update, IntArray start, IntArray out, int rows, int cols, int updateRows) {
        return Mlx.task("slice_update_dynamic", 3, x, update, start, out, rows, cols, updateRows);
    }

    /** {@code out} = {@code x} with the positions where {@code mask} is non-zero filled, in order, from {@code src}. */
    @MlxOp("mlx_masked_scatter")
    public static LibraryTaskDescriptor maskedScatter(FloatArray x, ByteArray mask, FloatArray src, FloatArray out) {
        return Mlx.task("masked_scatter", 3, x, mask, src, out);
    }

    /** {@code out} = {@code x} with the positions where {@code mask} is non-zero filled, in order, from {@code src}. */
    @MlxOp("mlx_masked_scatter")
    public static LibraryTaskDescriptor maskedScatter(HalfFloatArray x, ByteArray mask, HalfFloatArray src, HalfFloatArray out) {
        return Mlx.task("masked_scatter", 3, x, mask, src, out);
    }

    /** {@code out} = {@code x} with the positions where {@code mask} is non-zero filled, in order, from {@code src}. */
    @MlxOp("mlx_masked_scatter")
    public static LibraryTaskDescriptor maskedScatter(BFloat16Array x, ByteArray mask, BFloat16Array src, BFloat16Array out) {
        return Mlx.task("masked_scatter", 3, x, mask, src, out);
    }

    /** {@code out} = {@code x} with the positions where {@code mask} is non-zero filled, in order, from {@code src}. */
    @MlxOp("mlx_masked_scatter")
    public static LibraryTaskDescriptor maskedScatter(IntArray x, ByteArray mask, IntArray src, IntArray out) {
        return Mlx.task("masked_scatter", 3, x, mask, src, out);
    }

    /**
     * Gathered batched matmul: {@code out[i] = a[lhs[i]] @ b[rhs[i]]}, with {@code a[batchesA, m, k]},
     * {@code b[batchesB, k, n]} and {@code out[count, m, n]} (the mixture-of-experts building block).
     */
    @MlxOp("mlx_gather_mm")
    public static LibraryTaskDescriptor gatherMm(FloatArray a, FloatArray b, IntArray lhs, IntArray rhs, FloatArray out, int batchesA, int batchesB, int m, int k, int n) {
        return Mlx.task("gather_mm", 4, a, b, lhs, rhs, out, batchesA, batchesB, m, k, n);
    }

    /**
     * Gathered batched matmul: {@code out[i] = a[lhs[i]] @ b[rhs[i]]}, with {@code a[batchesA, m, k]},
     * {@code b[batchesB, k, n]} and {@code out[count, m, n]} (the mixture-of-experts building block).
     */
    @MlxOp("mlx_gather_mm")
    public static LibraryTaskDescriptor gatherMm(HalfFloatArray a, HalfFloatArray b, IntArray lhs, IntArray rhs, HalfFloatArray out, int batchesA, int batchesB, int m, int k, int n) {
        return Mlx.task("gather_mm", 4, a, b, lhs, rhs, out, batchesA, batchesB, m, k, n);
    }

    /**
     * Gathered batched matmul: {@code out[i] = a[lhs[i]] @ b[rhs[i]]}, with {@code a[batchesA, m, k]},
     * {@code b[batchesB, k, n]} and {@code out[count, m, n]} (the mixture-of-experts building block).
     */
    @MlxOp("mlx_gather_mm")
    public static LibraryTaskDescriptor gatherMm(BFloat16Array a, BFloat16Array b, IntArray lhs, IntArray rhs, BFloat16Array out, int batchesA, int batchesB, int m, int k, int n) {
        return Mlx.task("gather_mm", 4, a, b, lhs, rhs, out, batchesA, batchesB, m, k, n);
    }
}
