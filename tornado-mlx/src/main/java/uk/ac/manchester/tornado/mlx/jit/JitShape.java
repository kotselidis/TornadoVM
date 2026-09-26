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
package uk.ac.manchester.tornado.mlx.jit;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * JIT counterparts of the MLX shape and layout operations ({@link uk.ac.manchester.tornado.mlx.MlxShape}),
 * float32, one thread per output element. Operations that only relabel the shape of a contiguous
 * array (reshape, flatten, squeeze, expand_dims, ...) leave the data in place, so their counterpart
 * is a copy.
 */
public final class JitShape {

    private JitShape() {
    }

    @JitBaseline({ "mlx_reshape", "mlx_flatten", "mlx_unflatten", "mlx_squeeze", "mlx_squeeze_axis", "mlx_squeeze_axes", "mlx_expand_dims", "mlx_expand_dims_axes", "mlx_atleast_1d",
            "mlx_atleast_2d", "mlx_atleast_3d", "mlx_contiguous", "mlx_copy" })
    public static void copy(KernelContext ctx, FloatArray x, FloatArray out, int n) {
        int i = ctx.globalIdx;
        if (i < n) {
            out.set(i, x.get(i));
        }
    }

    /** The bits of each float32 as an int32. */
    @JitBaseline("mlx_view")
    public static void viewAsInt(KernelContext ctx, FloatArray x, IntArray out, int n) {
        int i = ctx.globalIdx;
        if (i < n) {
            out.set(i, Float.floatToRawIntBits(x.get(i)));
        }
    }

    /** float32 to float16. */
    @JitBaseline("mlx_astype")
    public static void toHalf(KernelContext ctx, FloatArray x, HalfFloatArray out, int n) {
        int i = ctx.globalIdx;
        if (i < n) {
            out.set(i, new HalfFloat(x.get(i)));
        }
    }

    /**
     * out = x [d0, d1, d2] with its axes permuted: output axis k is input axis p_k. Serves
     * transpose_axes, swapaxes and moveaxis.
     */
    @JitBaseline({ "mlx_transpose_axes", "mlx_swapaxes", "mlx_moveaxis" })
    public static void permute3(KernelContext ctx, FloatArray x, FloatArray out, int d0, int d1, int d2, int p0, int p1, int p2) {
        int total = d0 * d1 * d2;
        int t = ctx.globalIdx;
        if (t < total) {
            int e0 = p0 == 0 ? d0 : p0 == 1 ? d1 : d2;
            int e1 = p1 == 0 ? d0 : p1 == 1 ? d1 : d2;
            int e2 = p2 == 0 ? d0 : p2 == 1 ? d1 : d2;
            int o0 = t / (e1 * e2);
            int o1 = (t / e2) % e1;
            int o2 = t % e2;
            // Input coordinate along axis a is the output coordinate k with p_k == a.
            int i0 = p0 == 0 ? o0 : p1 == 0 ? o1 : o2;
            int i1 = p0 == 1 ? o0 : p1 == 1 ? o1 : o2;
            int i2 = p0 == 2 ? o0 : p1 == 2 ? o1 : o2;
            out.set(t, x.get((i0 * d1 + i1) * d2 + i2));
        }
    }

    /** out[r, c] = x[c]: x [cols] broadcast to [rows, cols]. */
    @JitBaseline("mlx_broadcast_to")
    public static void broadcastRows(KernelContext ctx, FloatArray x, FloatArray out, int rows, int cols) {
        int t = ctx.globalIdx;
        if (t < rows * cols) {
            out.set(t, x.get(t % cols));
        }
    }

    /** a [cols] and b [rows] (a column) broadcast together to [rows, cols]. */
    @JitBaseline("mlx_broadcast_arrays")
    public static void broadcastPair(KernelContext ctx, FloatArray a, FloatArray b, FloatArray outA, FloatArray outB, int rows, int cols) {
        int t = ctx.globalIdx;
        if (t < rows * cols) {
            outA.set(t, a.get(t % cols));
            outB.set(t, b.get(t / cols));
        }
    }

    /** out[r, c] = x[offset + r * rowStride + c * colStride]. */
    @JitBaseline("mlx_as_strided")
    public static void asStrided(KernelContext ctx, FloatArray x, FloatArray out, int rows, int cols, int rowStride, int colStride, int offset) {
        int t = ctx.globalIdx;
        if (t < rows * cols) {
            out.set(t, x.get(offset + (t / cols) * rowStride + (t % cols) * colStride));
        }
    }

    @JitBaseline("mlx_number_of_elements")
    public static void writeInt(KernelContext ctx, IntArray out, int value) {
        if (ctx.globalIdx == 0) {
            out.set(0, value);
        }
    }

    /** out [rows, colsA + colsB] = [a | b] for a [rows, colsA] and b [rows, colsB]; rows = 1 joins flat arrays. */
    @JitBaseline({ "mlx_concatenate", "mlx_concatenate_axis", "mlx_stack" })
    public static void concat(KernelContext ctx, FloatArray a, FloatArray b, FloatArray out, int rows, int colsA, int colsB) {
        int cols = colsA + colsB;
        int t = ctx.globalIdx;
        if (t < rows * cols) {
            int r = t / cols;
            int c = t % cols;
            out.set(t, c < colsA ? a.get(r * colsA + c) : b.get(r * colsB + c - colsA));
        }
    }

    /** out [n, 2] = a and b stacked along a new last axis. */
    @JitBaseline("mlx_stack_axis")
    public static void interleave(KernelContext ctx, FloatArray a, FloatArray b, FloatArray out, int n) {
        int i = ctx.globalIdx;
        if (i < n) {
            out.set(2 * i, a.get(i));
            out.set(2 * i + 1, b.get(i));
        }
    }

    /** x [rows, cols] split at column {@code index} into first [rows, index] and second [rows, cols - index]. */
    @JitBaseline({ "mlx_split", "mlx_split_sections" })
    public static void split(KernelContext ctx, FloatArray x, FloatArray first, FloatArray second, int rows, int cols, int index) {
        int t = ctx.globalIdx;
        if (t < rows * cols) {
            int r = t / cols;
            int c = t % cols;
            if (c < index) {
                first.set(r * index + c, x.get(t));
            } else {
                second.set(r * (cols - index) + c - index, x.get(t));
            }
        }
    }

    /** Each row of x [rows, cols] repeated {@code times} times; rows = n, cols = 1 repeats each element. */
    @JitBaseline({ "mlx_repeat", "mlx_repeat_axis" })
    public static void repeatRows(KernelContext ctx, FloatArray x, FloatArray out, int rows, int cols, int times) {
        int t = ctx.globalIdx;
        if (t < rows * times * cols) {
            int r = t / (times * cols);
            out.set(t, x.get(r * cols + t % cols));
        }
    }

    /** x [rows, cols] tiled repsRows x repsCols times. */
    @JitBaseline("mlx_tile")
    public static void tile(KernelContext ctx, FloatArray x, FloatArray out, int rows, int cols, int repsRows, int repsCols) {
        int outCols = cols * repsCols;
        int t = ctx.globalIdx;
        if (t < rows * repsRows * outCols) {
            int r = (t / outCols) % rows;
            int c = (t % outCols) % cols;
            out.set(t, x.get(r * cols + c));
        }
    }

    /** x [rows, cols] rolled by shiftRows along axis 0 and shiftCols along axis 1; rows = 1 rolls a flat array. */
    @JitBaseline({ "mlx_roll", "mlx_roll_axis", "mlx_roll_axes" })
    public static void roll(KernelContext ctx, FloatArray x, FloatArray out, int rows, int cols, int shiftRows, int shiftCols) {
        int t = ctx.globalIdx;
        if (t < rows * cols) {
            int r = t / cols;
            int c = t % cols;
            int sr = ((r - shiftRows) % rows + rows) % rows;
            int sc = ((c - shiftCols) % cols + cols) % cols;
            out.set(t, x.get(sr * cols + sc));
        }
    }

    /** x [rows, cols] padded with {@code value}: top rows above, left columns before, into out [outRows, outCols]. */
    @JitBaseline({ "mlx_pad", "mlx_pad_symmetric" })
    public static void pad(KernelContext ctx, FloatArray x, FloatArray out, int rows, int cols, int top, int left, int outRows, int outCols, float value) {
        int t = ctx.globalIdx;
        if (t < outRows * outCols) {
            int r = t / outCols - top;
            int c = t % outCols - left;
            out.set(t, r >= 0 && r < rows && c >= 0 && c < cols ? x.get(r * cols + c) : value);
        }
    }
}
