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
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * MLX linear algebra (Tier 2) as TornadoVM library tasks. Matrices are batched,
 * {@code [batch, n, n]}. MLX runs the decompositions, inverses and solves on its CPU stream
 * (LAPACK, float32 only), which the provider selects automatically; cross products and norms run on
 * the GPU.
 */
public final class MlxLinalg {

    private MlxLinalg() {
    }

    /** {@code out[i] = a[i] x b[i]} for {@code count} 3-vectors stored as {@code [count, 3]}. */
    @MlxOp("mlx_linalg_cross")
    public static LibraryTaskDescriptor cross(FloatArray a, FloatArray b, FloatArray out, int count) {
        return Mlx.task("linalg_cross", 2, a, b, out, count);
    }

    /**
     * {@code out[r]} = the {@code ord}-norm of row {@code r} of {@code x[rows, cols]}:
     * {@code (sum |x|^ord)^(1/ord)}; infinity and -infinity give the largest and smallest
     * magnitude, 0 the number of non-zeros.
     */
    @MlxOp("mlx_linalg_norm")
    public static LibraryTaskDescriptor norm(FloatArray x, FloatArray out, int rows, int cols, float ord) {
        return Mlx.task("linalg_norm", 1, x, out, rows, cols, ord);
    }

    /** {@code out[r]} = the Euclidean norm of row {@code r} of {@code x[rows, cols]}. */
    @MlxOp("mlx_linalg_norm_l2")
    public static LibraryTaskDescriptor l2Norm(FloatArray x, FloatArray out, int rows, int cols) {
        return Mlx.task("linalg_norm_l2", 1, x, out, rows, cols);
    }

    /** {@code out[b]} = the Frobenius norm of matrix {@code b} of {@code x[batch, rows, cols]}. */
    @MlxOp("mlx_linalg_norm_matrix")
    public static LibraryTaskDescriptor frobeniusNorm(FloatArray x, FloatArray out, int batch, int rows, int cols) {
        return Mlx.task("linalg_norm_matrix", 1, x, out, batch, rows, cols);
    }

    /** {@code out[i] = a[i] x b[i]} for {@code count} 3-vectors stored as {@code [count, 3]}. */
    @MlxOp("mlx_linalg_cross")
    public static LibraryTaskDescriptor cross(HalfFloatArray a, HalfFloatArray b, HalfFloatArray out, int count) {
        return Mlx.task("linalg_cross", 2, a, b, out, count);
    }

    /**
     * {@code out[r]} = the {@code ord}-norm of row {@code r} of {@code x[rows, cols]}:
     * {@code (sum |x|^ord)^(1/ord)}; infinity and -infinity give the largest and smallest
     * magnitude, 0 the number of non-zeros.
     */
    @MlxOp("mlx_linalg_norm")
    public static LibraryTaskDescriptor norm(HalfFloatArray x, HalfFloatArray out, int rows, int cols, float ord) {
        return Mlx.task("linalg_norm", 1, x, out, rows, cols, ord);
    }

    /** {@code out[r]} = the Euclidean norm of row {@code r} of {@code x[rows, cols]}. */
    @MlxOp("mlx_linalg_norm_l2")
    public static LibraryTaskDescriptor l2Norm(HalfFloatArray x, HalfFloatArray out, int rows, int cols) {
        return Mlx.task("linalg_norm_l2", 1, x, out, rows, cols);
    }

    /** {@code out[b]} = the Frobenius norm of matrix {@code b} of {@code x[batch, rows, cols]}. */
    @MlxOp("mlx_linalg_norm_matrix")
    public static LibraryTaskDescriptor frobeniusNorm(HalfFloatArray x, HalfFloatArray out, int batch, int rows, int cols) {
        return Mlx.task("linalg_norm_matrix", 1, x, out, batch, rows, cols);
    }

    /** {@code out[i] = a[i] x b[i]} for {@code count} 3-vectors stored as {@code [count, 3]}. */
    @MlxOp("mlx_linalg_cross")
    public static LibraryTaskDescriptor cross(BFloat16Array a, BFloat16Array b, BFloat16Array out, int count) {
        return Mlx.task("linalg_cross", 2, a, b, out, count);
    }

    /**
     * {@code out[r]} = the {@code ord}-norm of row {@code r} of {@code x[rows, cols]}:
     * {@code (sum |x|^ord)^(1/ord)}; infinity and -infinity give the largest and smallest
     * magnitude, 0 the number of non-zeros.
     */
    @MlxOp("mlx_linalg_norm")
    public static LibraryTaskDescriptor norm(BFloat16Array x, BFloat16Array out, int rows, int cols, float ord) {
        return Mlx.task("linalg_norm", 1, x, out, rows, cols, ord);
    }

    /** {@code out[r]} = the Euclidean norm of row {@code r} of {@code x[rows, cols]}. */
    @MlxOp("mlx_linalg_norm_l2")
    public static LibraryTaskDescriptor l2Norm(BFloat16Array x, BFloat16Array out, int rows, int cols) {
        return Mlx.task("linalg_norm_l2", 1, x, out, rows, cols);
    }

    /** {@code out[b]} = the Frobenius norm of matrix {@code b} of {@code x[batch, rows, cols]}. */
    @MlxOp("mlx_linalg_norm_matrix")
    public static LibraryTaskDescriptor frobeniusNorm(BFloat16Array x, BFloat16Array out, int batch, int rows, int cols) {
        return Mlx.task("linalg_norm_matrix", 1, x, out, batch, rows, cols);
    }

    /** Cholesky factor of each symmetric positive-definite {@code a[b]} ({@code [batch, n, n]}): lower, or upper. */
    @MlxOp("mlx_linalg_cholesky")
    public static LibraryTaskDescriptor cholesky(FloatArray a, FloatArray out, int batch, int n, boolean upper) {
        return Mlx.task("linalg_cholesky", 1, a, out, batch, n, upper);
    }

    /** {@code (L L^T)^-1} (or {@code (U^T U)^-1}) from each Cholesky factor {@code a[b]}. */
    @MlxOp("mlx_linalg_cholesky_inv")
    public static LibraryTaskDescriptor choleskyInv(FloatArray a, FloatArray out, int batch, int n, boolean upper) {
        return Mlx.task("linalg_cholesky_inv", 1, a, out, batch, n, upper);
    }

    /** Inverse of each triangular {@code a[b]} (lower, or upper). */
    @MlxOp("mlx_linalg_tri_inv")
    public static LibraryTaskDescriptor triInv(FloatArray a, FloatArray out, int batch, int n, boolean upper) {
        return Mlx.task("linalg_tri_inv", 1, a, out, batch, n, upper);
    }

    /** Inverse of each {@code a[b]} ({@code [batch, n, n]}). */
    @MlxOp("mlx_linalg_inv")
    public static LibraryTaskDescriptor inv(FloatArray a, FloatArray out, int batch, int n) {
        return Mlx.task("linalg_inv", 1, a, out, batch, n);
    }

    /** {@code x[b] = a[b]^-1 rhs[b]} for {@code a[batch, n, n]} and {@code rhs}, {@code x} {@code [batch, n, nrhs]}. */
    @MlxOp("mlx_linalg_solve")
    public static LibraryTaskDescriptor solve(FloatArray a, FloatArray rhs, FloatArray x, int batch, int n, int nrhs) {
        return Mlx.task("linalg_solve", 2, a, rhs, x, batch, n, nrhs);
    }

    /** {@link #solve} for triangular {@code a} (lower, or upper). */
    @MlxOp("mlx_linalg_solve_triangular")
    public static LibraryTaskDescriptor solveTriangular(FloatArray a, FloatArray rhs, FloatArray x, int batch, int n, int nrhs, boolean upper) {
        return Mlx.task("linalg_solve_triangular", 2, a, rhs, x, batch, n, nrhs, upper);
    }

    /**
     * LU factorisation with partial pivoting of each {@code a[b]} ({@code [batch, n, n]}):
     * {@code l} unit lower triangular, {@code u} upper triangular, and {@code perm[b]} the row
     * permutation, with {@code a[b][i, :] = (l u)[perm[i], :]}.
     */
    @MlxOp("mlx_linalg_lu")
    public static LibraryTaskDescriptor lu(FloatArray a, IntArray perm, FloatArray l, FloatArray u, int batch, int n) {
        return Mlx.task("linalg_lu", new int[] { 1, 2, 3 }, a, perm, l, u, batch, n);
    }

    /**
     * Packed LU factorisation of each {@code a[b]}: {@code lu} holds L (unit lower, below the
     * diagonal) and U, and {@code pivots[b][k]} is the row swapped with row {@code k} at step
     * {@code k} (LAPACK getrf, 0-based).
     */
    @MlxOp("mlx_linalg_lu_factor")
    public static LibraryTaskDescriptor luFactor(FloatArray a, FloatArray lu, IntArray pivots, int batch, int n) {
        return Mlx.task("linalg_lu_factor", new int[] { 1, 2 }, a, lu, pivots, batch, n);
    }

    /** QR factorisation of each square {@code a[b]}: {@code q} orthogonal, {@code r} upper triangular. */
    @MlxOp("mlx_linalg_qr")
    public static LibraryTaskDescriptor qr(FloatArray a, FloatArray q, FloatArray r, int batch, int n) {
        return Mlx.task("linalg_qr", new int[] { 1, 2 }, a, q, r, batch, n);
    }

    /**
     * Eigenvalues (ascending) and eigenvectors (the columns of {@code vectors}) of each symmetric
     * {@code a[b]} ({@code [batch, n, n]}), read from its lower (or upper) triangle.
     */
    @MlxOp("mlx_linalg_eigh")
    public static LibraryTaskDescriptor eigh(FloatArray a, FloatArray values, FloatArray vectors, int batch, int n, boolean upper) {
        return Mlx.task("linalg_eigh", new int[] { 1, 2 }, a, values, vectors, batch, n, upper);
    }

    /** Eigenvalues (ascending) of each symmetric {@code a[b]}, read from its lower (or upper) triangle. */
    @MlxOp("mlx_linalg_eigvalsh")
    public static LibraryTaskDescriptor eigvalsh(FloatArray a, FloatArray values, int batch, int n, boolean upper) {
        return Mlx.task("linalg_eigvalsh", 1, a, values, batch, n, upper);
    }

    /** Singular value decomposition {@code a[b] = u diag(s) vt} of each square {@code a[b]}; {@code s} descending. */
    @MlxOp("mlx_linalg_svd")
    public static LibraryTaskDescriptor svd(FloatArray a, FloatArray u, FloatArray s, FloatArray vt, int batch, int n) {
        return Mlx.task("linalg_svd", new int[] { 1, 2, 3 }, a, u, s, vt, batch, n);
    }

    /** Singular values (descending) of each square {@code a[b]}. */
    @MlxOp("mlx_linalg_svd")
    public static LibraryTaskDescriptor singularValues(FloatArray a, FloatArray s, int batch, int n) {
        return Mlx.task("linalg_svd_values", 1, a, s, batch, n);
    }

    /** Moore-Penrose pseudo-inverse of each square {@code a[b]}. */
    @MlxOp("mlx_linalg_pinv")
    public static LibraryTaskDescriptor pinv(FloatArray a, FloatArray out, int batch, int n) {
        return Mlx.task("linalg_pinv", 1, a, out, batch, n);
    }
}
