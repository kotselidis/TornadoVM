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
package uk.ac.manchester.tornado.mlx.provider;

import static java.util.Map.entry;
import static uk.ac.manchester.tornado.mlx.provider.MlxMetalKernels.contiguousStrides;
import static uk.ac.manchester.tornado.mlx.provider.MlxMetalKernels.scalarBytes;
import static uk.ac.manchester.tornado.mlx.provider.MlxNativeLib.MLX_BOOL;
import static uk.ac.manchester.tornado.mlx.provider.MlxNativeLib.MLX_COMPLEX64;
import static uk.ac.manchester.tornado.mlx.provider.MlxNativeLib.MLX_FLOAT32;
import static uk.ac.manchester.tornado.mlx.provider.MlxNativeLib.MLX_INT32;
import static uk.ac.manchester.tornado.mlx.provider.MlxNativeLib.MLX_UINT8;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

import uk.ac.manchester.tornado.api.types.arrays.TornadoNativeArray;
import uk.ac.manchester.tornado.mlx.provider.MlxMetalKernels.Program;
import uk.ac.manchester.tornado.mlx.provider.MlxMetalKernels.Ref;
import uk.ac.manchester.tornado.runtime.library.spi.LibraryInvocation;
import uk.ac.manchester.tornado.runtime.library.spi.TornadoNativeStreamSupport;

/**
 * The MLX operations that run as MLX's own Metal kernels in place on TornadoVM's buffers
 * ({@link MlxMetalKernels}), with how each maps onto MLX's kernels. Composite operations follow the
 * primitives MLX's {@code ops.cpp} builds them from, encoded into one command buffer; where an
 * intermediate needs its own storage it goes to a scratch buffer. A route declines (returns null)
 * when the arguments are outside what it reproduces exactly, and the operation then goes through the
 * C API as before.
 */
final class MlxKernelRoutes {

    /**
     * Whether routed operations run MLX's kernels in place rather than through the C API, which would
     * allocate a result and copy it back. {@code -Dtornado.mlx.kernels=False} turns this off.
     */
    private static final boolean ENABLED = Boolean.parseBoolean(System.getProperty("tornado.mlx.kernels", "True")) && MlxMetalKernels.isAvailable();

    private static final AtomicLong DISPATCHES = new AtomicLong();

    private static final double INF = Double.POSITIVE_INFINITY;

    /** Plans the kernels for one call, or returns null to leave it to the C API. */
    interface Route {
        Encoder plan(View v);
    }

    /** Encodes the planned kernels into a program. */
    interface Encoder {
        void encode(Program p);
    }

    /** The arguments of one call, as MLX kernels see them. */
    static final class View {
        private final LibraryInvocation invocation;

        View(LibraryInvocation invocation) {
            this.invocation = invocation;
        }

        int args() {
            return invocation.getNumArgs();
        }

        boolean isArray(int i) {
            return i < args() && invocation.getArg(i) instanceof TornadoNativeArray && invocation.getNativeBuffer(i) != 0;
        }

        int dtype(int i) {
            return MlxCall.dtypeOf(invocation.getArg(i));
        }

        int size(int i) {
            return ((TornadoNativeArray) invocation.getArg(i)).getSize();
        }

        Ref ref(int i) {
            return new Ref(invocation.getNativeBuffer(i), invocation.getNativeOffset(i));
        }

        float floatArg(int i) {
            return ((Number) invocation.getArg(i)).floatValue();
        }

        int intArg(int i) {
            return ((Number) invocation.getArg(i)).intValue();
        }

        boolean boolArg(int i) {
            Object arg = invocation.getArg(i);
            return arg instanceof Boolean b ? b : ((Number) arg).intValue() != 0;
        }

        /** Arrays {@code indices} all exist, share one type from {@code allowed} and one size; returns that type, or -1. */
        int sameType(TypeSet allowed, int... indices) {
            if (indices.length == 0 || !isArray(indices[0])) {
                return -1;
            }
            int type = dtype(indices[0]);
            int n = size(indices[0]);
            for (int i : indices) {
                if (!isArray(i) || dtype(i) != type || size(i) != n) {
                    return -1;
                }
            }
            return allowed.contains(type) && n > 0 ? type : -1;
        }

        /** Array {@code i} exists, has {@code type} and {@code n} elements. */
        boolean is(int i, int type, int n) {
            return isArray(i) && dtype(i) == type && size(i) == n;
        }
    }

    interface TypeSet {
        boolean contains(int dtype);
    }

    private static final TypeSet FLOATS = MlxMetalKernels::isFloating;
    private static final TypeSet FLOAT32 = t -> t == MLX_FLOAT32;
    private static final TypeSet INTEGERS = t -> t != MLX_BOOL && t != MLX_COMPLEX64 && !MlxMetalKernels.isFloating(t) && MlxMetalKernels.typeName(t) != null;
    private static final TypeSet NUMBERS = t -> FLOATS.contains(t) || INTEGERS.contains(t);
    private static final TypeSet ANY = t -> MlxMetalKernels.typeName(t) != null;

    private static final Map<String, Route> ROUTES = new HashMap<>();

    /** Unary math: provider name to MLX's Metal op name (floating types). */
    private static final Map<String, String> UNARY = Map.ofEntries(entry("negative", "Negative"), entry("exp", "Exp"), entry("tanh", "Tanh"), entry("erf", "Erf"), //
            entry("sigmoid", "Sigmoid"), entry("sqrt", "Sqrt"), entry("rsqrt", "Rsqrt"), entry("square", "Square"), entry("abs", "Abs"), entry("arccos", "ArcCos"), //
            entry("arccosh", "ArcCosh"), entry("arcsin", "ArcSin"), entry("arcsinh", "ArcSinh"), entry("arctan", "ArcTan"), entry("arctanh", "ArcTanh"), entry("ceil", "Ceil"), //
            entry("cos", "Cos"), entry("cosh", "Cosh"), entry("erfinv", "ErfInv"), entry("expm1", "Expm1"), entry("floor", "Floor"), entry("log", "Log"), entry("log10", "Log10"), //
            entry("log1p", "Log1p"), entry("log2", "Log2"), entry("sign", "Sign"), entry("sin", "Sin"), entry("sinh", "Sinh"), entry("tan", "Tan"));

    /** Binary arithmetic (floating types). */
    private static final Map<String, String> BINARY = Map.ofEntries(entry("add", "Add"), entry("subtract", "Subtract"), entry("multiply", "Multiply"), //
            entry("divide", "Divide"), entry("maximum", "Maximum"), entry("minimum", "Minimum"), entry("arctan2", "ArcTan2"), entry("logaddexp", "LogAddExp"), //
            entry("power", "Power"), entry("remainder", "Remainder"));

    /** Comparisons, writing MLX bools (one byte, 0 or 1) into a byte array. */
    private static final Map<String, String> COMPARISON = Map.ofEntries(entry("equal", "Equal"), entry("not_equal", "NotEqual"), entry("greater", "Greater"), //
            entry("greater_equal", "GreaterEqual"), entry("less", "Less"), entry("less_equal", "LessEqual"));

    /** Bitwise operations (integer types). */
    private static final Map<String, String> BITWISE = Map.ofEntries(entry("bitwise_and", "BitwiseAnd"), entry("bitwise_or", "BitwiseOr"), entry("bitwise_xor", "BitwiseXor"), //
            entry("left_shift", "LeftShift"), entry("right_shift", "RightShift"));

    static {
        UNARY.forEach((name, op) -> ROUTES.put(name, v -> {
            int t = v.sameType(FLOATS, 0, 1);
            return t < 0 || v.args() != 2 ? null : p -> p.unary(op, t, t, v.size(0), v.ref(0), v.ref(1));
        }));
        BINARY.forEach((name, op) -> ROUTES.put(name, v -> {
            int t = v.sameType(FLOATS, 0, 1, 2);
            return t < 0 || v.args() != 3 ? null : p -> p.binary(op, t, v.size(0), v.ref(0), v.ref(1), v.ref(2));
        }));
        COMPARISON.forEach((name, op) -> ROUTES.put(name, v -> {
            int t = v.sameType(NUMBERS, 0, 1);
            return t < 0 || v.args() != 3 || !v.is(2, MLX_UINT8, v.size(0)) ? null : p -> p.binary(op, t, v.size(0), v.ref(0), v.ref(1), v.ref(2));
        }));
        BITWISE.forEach((name, op) -> ROUTES.put(name, v -> {
            int t = v.sameType(INTEGERS, 0, 1, 2);
            return t < 0 || v.args() != 3 ? null : p -> p.binary(op, t, v.size(0), v.ref(0), v.ref(1), v.ref(2));
        }));
        ROUTES.put("bitwise_invert", v -> {
            int t = v.sameType(INTEGERS, 0, 1);
            return t < 0 ? null : p -> p.unary("BitwiseInvert", t, t, v.size(0), v.ref(0), v.ref(1));
        });

        // degrees/radians: multiply(a, array(180 / pi, dtype)); reciprocal: divide(array(1, dtype), a).
        ROUTES.put("degrees", v -> scaled(v, 180.0 / Math.PI));
        ROUTES.put("radians", v -> scaled(v, Math.PI / 180.0));
        ROUTES.put("reciprocal", v -> {
            int t = v.sameType(FLOATS, 0, 1);
            return t < 0 ? null : p -> p.binaryScalarLeft("Divide", t, v.size(0), 1.0, v.ref(0), v.ref(1));
        });
        // floor_divide: floor(divide(a, b)) for floating types; integer Divide otherwise.
        ROUTES.put("floor_divide", v -> {
            int t = v.sameType(NUMBERS, 0, 1, 2);
            if (t < 0) {
                return null;
            }
            int n = v.size(0);
            return MlxMetalKernels.isFloating(t) ? p -> {
                p.binary("Divide", t, n, v.ref(0), v.ref(1), v.ref(2));
                p.unary("Floor", t, t, n, v.ref(2), v.ref(2));
            } : p -> p.binary("Divide", t, n, v.ref(0), v.ref(1), v.ref(2));
        });
        // round(a, decimals): Round, or multiply by 10^d, Round, multiply by 1 / 10^d.
        ROUTES.put("round", v -> {
            int t = v.sameType(FLOATS, 0, 1);
            if (t < 0 || v.args() != 3) {
                return null;
            }
            int n = v.size(0);
            int decimals = v.intArg(2);
            if (decimals == 0) {
                return p -> p.unary("Round", t, t, n, v.ref(0), v.ref(1));
            }
            float scale = (float) Math.pow(10, decimals);
            return p -> {
                p.binaryScalarRight("Multiply", t, n, v.ref(0), scale, v.ref(1));
                p.unary("Round", t, t, n, v.ref(1), v.ref(1));
                p.binaryScalarRight("Multiply", t, n, v.ref(1), 1 / scale, v.ref(1));
            };
        });
        ROUTES.put("divmod", v -> {
            int t = v.sameType(NUMBERS, 0, 1, 2, 3);
            return t < 0 ? null : p -> p.binaryTwo("DivMod", t, v.size(0), v.ref(0), v.ref(1), v.ref(2), v.ref(3));
        });
        // clip: minimum(maximum(a, lo), hi).
        ROUTES.put("clip", v -> {
            int t = v.sameType(NUMBERS, 0, 1);
            if (t < 0 || v.args() != 4) {
                return null;
            }
            int n = v.size(0);
            double lo = t == MLX_INT32 ? v.intArg(2) : v.floatArg(2);
            double hi = t == MLX_INT32 ? v.intArg(3) : v.floatArg(3);
            return p -> {
                p.binaryScalarRight("Maximum", t, n, v.ref(0), lo, v.ref(1));
                p.binaryScalarRight("Minimum", t, n, v.ref(1), hi, v.ref(1));
            };
        });
        // where: Select over astype(condition, bool).
        ROUTES.put("where", v -> {
            int t = v.sameType(NUMBERS, 1, 2, 3);
            if (t < 0 || !v.isArray(0) || v.size(0) != v.size(1)) {
                return null;
            }
            int n = v.size(1);
            int conditionType = v.dtype(0);
            return p -> {
                Ref condition = p.scratch(n);
                p.copy(conditionType, MLX_BOOL, n, v.ref(0), condition);
                p.select(t, n, condition, v.ref(1), v.ref(2), v.ref(3));
            };
        });

        // Predicates on floating types, writing bools.
        ROUTES.put("isnan", v -> predicate(v, (p, t, n) -> p.binary("NotEqual", t, n, v.ref(0), v.ref(0), v.ref(1))));
        ROUTES.put("isposinf", v -> predicate(v, (p, t, n) -> p.binaryScalarRight("Equal", t, n, v.ref(0), INF, v.ref(1))));
        ROUTES.put("isneginf", v -> predicate(v, (p, t, n) -> p.binaryScalarRight("Equal", t, n, v.ref(0), -INF, v.ref(1))));
        // isinf: |a| == inf (equal to isposinf or isneginf). isfinite: |a| < inf (false for NaN and infinities).
        ROUTES.put("isinf", v -> predicate(v, (p, t, n) -> {
            Ref magnitude = p.scratch((long) n * MlxMetalKernels.itemSize(t));
            p.unary("Abs", t, t, n, v.ref(0), magnitude);
            p.binaryScalarRight("Equal", t, n, magnitude, INF, v.ref(1));
        }));
        ROUTES.put("isfinite", v -> predicate(v, (p, t, n) -> {
            Ref magnitude = p.scratch((long) n * MlxMetalKernels.itemSize(t));
            p.unary("Abs", t, t, n, v.ref(0), magnitude);
            p.binaryScalarRight("Less", t, n, magnitude, INF, v.ref(1));
        }));

        // Logical operations on astype(x, bool).
        ROUTES.put("logical_and", v -> logical(v, "LogicalAnd"));
        ROUTES.put("logical_or", v -> logical(v, "LogicalOr"));
        ROUTES.put("logical_not", v -> {
            if (!v.isArray(0) || !v.is(1, MLX_UINT8, v.size(0)) || MlxMetalKernels.typeName(v.dtype(0)) == null) {
                return null;
            }
            int n = v.size(0);
            int t = v.dtype(0);
            return p -> {
                Ref b = p.scratch(n);
                p.copy(t, MLX_BOOL, n, v.ref(0), b);
                p.unary("LogicalNot", MLX_BOOL, MLX_BOOL, n, b, v.ref(1));
            };
        });

        // nan_to_num: three Selects of scalars over isnan, isposinf and isneginf masks.
        ROUTES.put("nan_to_num", v -> {
            int t = v.sameType(FLOATS, 0, 1);
            if (t < 0 || v.args() != 5) {
                return null;
            }
            int n = v.size(0);
            double nan = v.floatArg(2);
            double posinf = v.floatArg(3);
            double neginf = v.floatArg(4);
            return p -> {
                Ref mask = p.scratch(n);
                p.binary("NotEqual", t, n, v.ref(0), v.ref(0), mask);
                p.selectScalar(t, n, mask, nan, v.ref(0), v.ref(1));
                p.binaryScalarRight("Equal", t, n, v.ref(0), INF, mask);
                p.selectScalar(t, n, mask, posinf, v.ref(1), v.ref(1));
                p.binaryScalarRight("Equal", t, n, v.ref(0), -INF, mask);
                p.selectScalar(t, n, mask, neginf, v.ref(1), v.ref(1));
            };
        });

        ROUTES.put("isclose", MlxKernelRoutes::isclose);

        // Complex parts, with complex64 data held as interleaved float pairs.
        ROUTES.put("real", v -> complexPart(v, "Real", false));
        ROUTES.put("imag", v -> complexPart(v, "Imag", false));
        ROUTES.put("conjugate", v -> complexPart(v, "Conjugate", true));

        registerShapeRoutes();
        registerConstructionRoutes();
    }

    private MlxKernelRoutes() {
    }

    // ---------------------------------------------------------------- shape and layout

    /** Arrays {@code in} and {@code out} exist and share a type MLX's kernels take; returns it or -1. */
    private static int pairType(View v, int in, int out) {
        if (!v.isArray(in) || !v.isArray(out) || v.dtype(in) != v.dtype(out) || MlxMetalKernels.typeName(v.dtype(in)) == null) {
            return -1;
        }
        return v.dtype(in);
    }

    /** A same-size, same-type copy: the result of any reshape-like view made contiguous. */
    private static Encoder contiguousCopy(View v) {
        int t = pairType(v, 0, 1);
        if (t < 0 || v.size(0) != v.size(1)) {
            return null;
        }
        return p -> p.copy(t, t, v.size(0), v.ref(0), v.ref(1));
    }

    /** {@code out} (contiguous) = {@code in} read as {@code shape} with {@code strides} from element {@code offset}. */
    private static Encoder view(View v, int in, int out, int[] shape, long[] strides, long offset) {
        int t = pairType(v, in, out);
        long n = 1;
        for (int d : shape) {
            n *= d;
        }
        if (t < 0 || n != v.size(out)) {
            return null;
        }
        return p -> p.generalCopy(t, t, shape, strides, v.ref(in).plus(offset, t), null, v.ref(out), null);
    }

    /** The permutation of a 3D contiguous array {@code dims}: the output shape and the input strides it reads with. */
    private static Encoder permute3(View v, int[] dims, int[] perm) {
        long[] strides = contiguousStrides(dims);
        int[] shape = new int[3];
        long[] permuted = new long[3];
        for (int i = 0; i < 3; i++) {
            if (perm[i] < 0 || perm[i] > 2) {
                return null;
            }
            shape[i] = dims[perm[i]];
            permuted[i] = strides[perm[i]];
        }
        return view(v, 0, 1, shape, permuted, 0);
    }

    private static int[] dims3(View v, int first) {
        return new int[] { v.intArg(first), v.intArg(first + 1), v.intArg(first + 2) };
    }

    /** Normalises an axis of a 3D array (negative axes count from the end). */
    private static int axis3(int axis) {
        return axis < 0 ? axis + 3 : axis;
    }

    /** Copies block {@code rows x cols} of {@code src} (row stride {@code srcCols}) to {@code dst} (row stride {@code dstCols}). */
    private static void block(Program p, int t, Ref src, int srcCols, long srcOffset, Ref dst, int dstCols, long dstOffset, int rows, int cols) {
        if (rows <= 0 || cols <= 0) {
            return;
        }
        p.generalCopy(t, t, new int[] { rows, cols }, new long[] { srcCols, 1 }, src.plus(srcOffset, t), new long[] { dstCols, 1 }, dst.plus(dstOffset, t), null);
    }

    /** Rolls a {@code rows x cols} array by {@code s0} rows and {@code s1} columns (non-negative, reduced), as four block copies. */
    private static Encoder roll2(View v, int rows, int cols, int shiftRows, int shiftCols) {
        int t = pairType(v, 0, 1);
        if (t < 0 || rows <= 0 || cols <= 0 || v.size(0) != rows * cols || v.size(1) != rows * cols) {
            return null;
        }
        int s0 = Math.floorMod(shiftRows, rows);
        int s1 = Math.floorMod(shiftCols, cols);
        return p -> {
            Ref x = v.ref(0);
            Ref out = v.ref(1);
            int[][] rowParts = { { 0, s0, rows - s0 }, { rows - s0, 0, s0 } };
            int[][] colParts = { { 0, s1, cols - s1 }, { cols - s1, 0, s1 } };
            for (int[] r : rowParts) {
                for (int[] c : colParts) {
                    block(p, t, x, cols, (long) r[0] * cols + c[0], out, cols, (long) r[1] * cols + c[1], r[2], c[2]);
                }
            }
        };
    }

    private static void registerShapeRoutes() {
        for (String name : new String[] { "reshape", "flatten", "unflatten", "squeeze", "squeeze_axis", "squeeze_axes", "expand_dims", "expand_dims_axes", "atleast_1d",
                "atleast_2d", "atleast_3d", "contiguous", "copy" }) {
            ROUTES.put(name, MlxKernelRoutes::contiguousCopy);
        }
        ROUTES.put("astype", v -> {
            if (!v.isArray(0) || !v.isArray(1) || v.size(0) != v.size(1) || !ANY.contains(v.dtype(0)) || !ANY.contains(v.dtype(1))) {
                return null;
            }
            int from = v.dtype(0);
            int to = v.dtype(1);
            return p -> p.copy(from, to, v.size(0), v.ref(0), v.ref(1));
        });
        // view: the same bytes under another type.
        ROUTES.put("view", v -> {
            if (!v.isArray(0) || !v.isArray(1) || !ANY.contains(v.dtype(0)) || !ANY.contains(v.dtype(1))) {
                return null;
            }
            long bytes = (long) v.size(0) * MlxMetalKernels.itemSize(v.dtype(0));
            if (bytes != (long) v.size(1) * MlxMetalKernels.itemSize(v.dtype(1)) || bytes > Integer.MAX_VALUE) {
                return null;
            }
            return p -> p.copy(MLX_UINT8, MLX_UINT8, (int) bytes, v.ref(0), v.ref(1));
        });
        // number_of_elements(x[d0, d1, d2], axes (1, 2)) as int32.
        ROUTES.put("number_of_elements", v -> {
            if (!v.is(1, MLX_INT32, 1) || v.args() < 5) {
                return null;
            }
            long count = (long) v.intArg(3) * v.intArg(4);
            return p -> p.fill(MLX_INT32, 1, count, v.ref(1));
        });
        ROUTES.put("transpose_axes", v -> v.args() < 8 ? null : permute3(v, dims3(v, 2), new int[] { axis3(v.intArg(5)), axis3(v.intArg(6)), axis3(v.intArg(7)) }));
        ROUTES.put("swapaxes", v -> {
            if (v.args() < 7) {
                return null;
            }
            int[] perm = { 0, 1, 2 };
            int a = axis3(v.intArg(5));
            int b = axis3(v.intArg(6));
            if (a < 0 || a > 2 || b < 0 || b > 2) {
                return null;
            }
            perm[a] = b;
            perm[b] = a;
            return permute3(v, dims3(v, 2), perm);
        });
        ROUTES.put("moveaxis", v -> {
            if (v.args() < 7) {
                return null;
            }
            int source = axis3(v.intArg(5));
            int destination = axis3(v.intArg(6));
            if (source < 0 || source > 2 || destination < 0 || destination > 2) {
                return null;
            }
            java.util.List<Integer> order = new java.util.ArrayList<>(java.util.List.of(0, 1, 2));
            order.remove(Integer.valueOf(source));
            order.add(destination, source);
            return permute3(v, dims3(v, 2), new int[] { order.get(0), order.get(1), order.get(2) });
        });
        // broadcast_to(x[L], (s0, s1)): L == s1 or L == 1.
        ROUTES.put("broadcast_to", v -> {
            if (!v.isArray(0) || v.args() < 4) {
                return null;
            }
            int n = v.size(0);
            int s0 = v.intArg(2);
            int s1 = v.intArg(3);
            if (n != s1 && n != 1) {
                return null;
            }
            return view(v, 0, 1, new int[] { s0, s1 }, new long[] { 0, n == 1 ? 0 : 1 }, 0);
        });
        // broadcast_arrays(a[1, cols], b[rows, 1]) -> two [rows, cols] arrays.
        ROUTES.put("broadcast_arrays", v -> {
            if (v.args() < 6) {
                return null;
            }
            int rows = v.intArg(4);
            int cols = v.intArg(5);
            if (!v.isArray(0) || !v.isArray(1) || v.size(0) != cols || v.size(1) != rows) {
                return null;
            }
            Encoder first = view(v, 0, 2, new int[] { rows, cols }, new long[] { 0, 1 }, 0);
            Encoder second = view(v, 1, 3, new int[] { rows, cols }, new long[] { 1, 0 }, 0);
            return first == null || second == null ? null : p -> {
                first.encode(p);
                second.encode(p);
            };
        });
        ROUTES.put("as_strided", v -> {
            if (v.args() < 7 || !v.isArray(0)) {
                return null;
            }
            int[] shape = { v.intArg(2), v.intArg(3) };
            long[] strides = { v.intArg(4), v.intArg(5) };
            long offset = v.intArg(6);
            long last = offset + (shape[0] - 1L) * strides[0] + (shape[1] - 1L) * strides[1];
            if (offset < 0 || last >= v.size(0) || strides[0] < 0 || strides[1] < 0) {
                return null;
            }
            return view(v, 0, 1, shape, strides, offset);
        });
        ROUTES.put("concatenate", v -> {
            int t = pairType(v, 0, 2);
            if (t < 0 || !v.isArray(1) || v.dtype(1) != t || v.size(0) + v.size(1) != v.size(2)) {
                return null;
            }
            return p -> {
                p.copy(t, t, v.size(0), v.ref(0), v.ref(2));
                p.copy(t, t, v.size(1), v.ref(1), v.ref(2).plus(v.size(0), t));
            };
        });
        // concatenate_axis(a[r, ca], b[r, cb]) along axis 1.
        ROUTES.put("concatenate_axis", v -> {
            int t = pairType(v, 0, 2);
            if (t < 0 || v.args() < 6 || !v.isArray(1) || v.dtype(1) != t) {
                return null;
            }
            int r = v.intArg(3);
            int ca = v.intArg(4);
            int cb = v.intArg(5);
            if (v.size(0) != r * ca || v.size(1) != r * cb || v.size(2) != r * (ca + cb)) {
                return null;
            }
            return p -> {
                block(p, t, v.ref(0), ca, 0, v.ref(2), ca + cb, 0, r, ca);
                block(p, t, v.ref(1), cb, 0, v.ref(2), ca + cb, ca, r, cb);
            };
        });
        ROUTES.put("stack", v -> {
            int t = pairType(v, 0, 2);
            if (t < 0 || !v.isArray(1) || v.dtype(1) != t || v.size(0) != v.size(1) || v.size(2) != 2 * v.size(0)) {
                return null;
            }
            return p -> {
                p.copy(t, t, v.size(0), v.ref(0), v.ref(2));
                p.copy(t, t, v.size(1), v.ref(1), v.ref(2).plus(v.size(0), t));
            };
        });
        // stack_axis(a[n], b[n], axis 1) -> [n, 2].
        ROUTES.put("stack_axis", v -> {
            int t = pairType(v, 0, 2);
            if (t < 0 || !v.isArray(1) || v.dtype(1) != t || v.size(0) != v.size(1) || v.size(2) != 2 * v.size(0)) {
                return null;
            }
            int n = v.size(0);
            return p -> {
                p.generalCopy(t, t, new int[] { n }, new long[] { 1 }, v.ref(0), new long[] { 2 }, v.ref(2), null);
                p.generalCopy(t, t, new int[] { n }, new long[] { 1 }, v.ref(1), new long[] { 2 }, v.ref(2).plus(1, t), null);
            };
        });
        // split(x[r, c]) into two halves along axis 1; split_sections at column k.
        ROUTES.put("split", v -> v.args() < 5 ? null : splitColumns(v, v.intArg(3), v.intArg(4), v.intArg(4) / 2));
        ROUTES.put("split_sections", v -> v.args() < 6 ? null : splitColumns(v, v.intArg(3), v.intArg(4), v.intArg(5)));
        // repeat(x[n], reps): x read as [n, reps] with strides [1, 0].
        ROUTES.put("repeat", v -> !v.isArray(0) || v.args() < 3 ? null : view(v, 0, 1, new int[] { v.size(0), v.intArg(2) }, new long[] { 1, 0 }, 0));
        // repeat_axis(x[d0, d1], reps, axis 0): x read as [d0, reps, d1] with strides [d1, 0, 1].
        ROUTES.put("repeat_axis", v -> {
            if (v.args() < 5) {
                return null;
            }
            int d0 = v.intArg(2);
            int d1 = v.intArg(3);
            return view(v, 0, 1, new int[] { d0, v.intArg(4), d1 }, new long[] { d1, 0, 1 }, 0);
        });
        // tile(x[d0, d1], (r0, r1)): x read as [r0, d0, r1, d1] with strides [0, d1, 0, 1].
        ROUTES.put("tile", v -> {
            if (v.args() < 6) {
                return null;
            }
            int d0 = v.intArg(2);
            int d1 = v.intArg(3);
            return view(v, 0, 1, new int[] { v.intArg(4), d0, v.intArg(5), d1 }, new long[] { 0, d1, 0, 1 }, 0);
        });
        ROUTES.put("roll", v -> !v.isArray(0) || v.args() < 3 ? null : roll2(v, 1, v.size(0), 0, v.intArg(2)));
        ROUTES.put("roll_axis", v -> v.args() < 5 ? null : roll2(v, v.intArg(2), v.intArg(3), 0, v.intArg(4)));
        ROUTES.put("roll_axes", v -> v.args() < 6 ? null : roll2(v, v.intArg(2), v.intArg(3), v.intArg(4), v.intArg(5)));
        // pad(x[r, c], low (b0, b1), high (a0, a1), value): fill, then copy x into the interior.
        ROUTES.put("pad", v -> v.args() < 9 ? null : pad(v, v.intArg(2), v.intArg(3), v.intArg(4), v.intArg(5), v.intArg(6), v.intArg(7), v.floatArg(8)));
        ROUTES.put("pad_symmetric", v -> v.args() < 6 ? null : pad(v, v.intArg(2), v.intArg(3), v.intArg(4), v.intArg(4), v.intArg(4), v.intArg(4), v.floatArg(5)));
    }

    private static Encoder splitColumns(View v, int rows, int cols, int k) {
        int t = pairType(v, 0, 1);
        if (t < 0 || !v.isArray(2) || v.dtype(2) != t || k < 0 || k > cols || v.size(0) != rows * cols || v.size(1) != rows * k || v.size(2) != rows * (cols - k)) {
            return null;
        }
        return p -> {
            block(p, t, v.ref(0), cols, 0, v.ref(1), k, 0, rows, k);
            block(p, t, v.ref(0), cols, k, v.ref(2), cols - k, 0, rows, cols - k);
        };
    }

    private static Encoder pad(View v, int rows, int cols, int before0, int after0, int before1, int after1, double value) {
        int t = pairType(v, 0, 1);
        int outCols = cols + before1 + after1;
        int outRows = rows + before0 + after0;
        if (t < 0 || before0 < 0 || after0 < 0 || before1 < 0 || after1 < 0 || v.size(0) != rows * cols || v.size(1) != outRows * outCols) {
            return null;
        }
        return p -> {
            p.fill(t, outRows * outCols, value, v.ref(1));
            block(p, t, v.ref(0), cols, 0, v.ref(1), outCols, (long) before0 * outCols + before1, rows, cols);
        };
    }

    // ---------------------------------------------------------------- construction

    /** The single output array {@code out} (argument {@code index}) if its type is routable; -1 otherwise. */
    private static int outType(View v, int index, TypeSet allowed) {
        return v.isArray(index) && allowed.contains(v.dtype(index)) && v.size(index) > 0 ? v.dtype(index) : -1;
    }

    /** Fills a {@code rows x cols} matrix with zeros and its {@code k}-th diagonal from {@code src} (stride {@code srcStride}), or ones. */
    private static void diagonalFill(Program p, int t, Ref out, int rows, int cols, int k, Ref src, long srcStride) {
        p.fill(t, rows * cols, 0, out);
        int length = k >= 0 ? Math.min(rows, cols - k) : Math.min(rows + k, cols);
        if (length <= 0) {
            return;
        }
        long start = k >= 0 ? k : (long) -k * cols;
        Ref dst = out.plus(start, t);
        if (src == null) {
            p.generalCopy(t, t, new int[] { length }, new long[] { 0 }, null, new long[] { cols + 1 }, dst, scalarBytes(t, 1));
        } else {
            p.generalCopy(t, t, new int[] { length }, new long[] { srcStride }, src, new long[] { cols + 1 }, dst, null);
        }
    }

    /** MLX's tri(n, m, k) as a bool mask: arange(n)[:, None] >= arange(-k, m - k)[None, :]. */
    private static void triMask(Program p, int n, int m, int k, Ref mask) {
        Ref rows = p.scratch(4L * n);
        Ref cols = p.scratch(4L * m);
        p.arange(MLX_INT32, n, 0, 1, rows);
        p.arange(MLX_INT32, m, -k, 1, cols);
        p.generalBinary("GreaterEqual", MLX_INT32, new int[] { n, m }, new long[] { 1, 0 }, rows, new long[] { 0, 1 }, cols, mask);
    }

    /** A float32 window over arange(0, M): the kernel steps MLX's {@code ops.cpp} uses for it. */
    interface Window {
        void encode(Program p, int m, Ref n, Ref out);
    }

    private static Encoder window(View v, Window steps) {
        if (outType(v, 0, FLOAT32) < 0) {
            return null;
        }
        int m = v.size(0);
        return p -> {
            if (m == 1) {
                p.fill(MLX_FLOAT32, 1, 1, v.ref(0));
                return;
            }
            p.arange(MLX_FLOAT32, m, 0, 1, v.ref(0));
            steps.encode(p, m, v.ref(0), v.ref(0));
        };
    }

    private static void registerConstructionRoutes() {
        ROUTES.put("full", v -> {
            int t = outType(v, 0, NUMBERS);
            return t < 0 || v.args() < 2 ? null : p -> p.fill(t, v.size(0), v.floatArg(1), v.ref(0));
        });
        ROUTES.put("full_like", v -> {
            int t = outType(v, 1, NUMBERS);
            return t < 0 || v.args() < 3 || !v.isArray(0) || v.size(0) != v.size(1) ? null : p -> p.fill(t, v.size(1), v.floatArg(2), v.ref(1));
        });
        ROUTES.put("zeros", v -> fillRoute(v, 0, 0));
        ROUTES.put("ones", v -> fillRoute(v, 0, 1));
        ROUTES.put("zeros_like", v -> !v.isArray(0) || !v.isArray(1) || v.size(0) != v.size(1) ? null : fillRoute(v, 1, 0));
        ROUTES.put("ones_like", v -> !v.isArray(0) || !v.isArray(1) || v.size(0) != v.size(1) ? null : fillRoute(v, 1, 1));
        // arange(start, stop, step): MLX's arange kernel when the element count matches.
        ROUTES.put("arange", v -> {
            int t = outType(v, 0, NUMBERS);
            if (t < 0 || v.args() < 4) {
                return null;
            }
            double start = v.floatArg(1);
            double stop = v.floatArg(2);
            double step = v.floatArg(3);
            if (step == 0 || Double.isInfinite(step) || Math.max((int) Math.ceil((stop - start) / step), 0) != v.size(0)) {
                return null;
            }
            return p -> p.arange(t, v.size(0), start, step, v.ref(0));
        });
        // linspace(start, stop, num): t = arange(0, num) / (num - 1); (1 - t) * start + t * stop in float32.
        ROUTES.put("linspace", v -> {
            if (outType(v, 0, FLOATS) < 0 || v.args() < 3) {
                return null;
            }
            int t = v.dtype(0);
            int num = v.size(0);
            double start = v.floatArg(1);
            double stop = v.floatArg(2);
            return p -> {
                if (num == 1) {
                    p.fill(t, 1, (float) start, v.ref(0));
                    return;
                }
                Ref steps = p.scratch(4L * num);
                Ref complement = p.scratch(4L * num);
                p.arange(MLX_FLOAT32, num, 0, 1, steps);
                p.binaryScalarRight("Divide", MLX_FLOAT32, num, steps, num - 1, steps);
                p.binaryScalarLeft("Subtract", MLX_FLOAT32, num, 1, steps, complement);
                p.binaryScalarRight("Multiply", MLX_FLOAT32, num, complement, start, complement);
                p.binaryScalarRight("Multiply", MLX_FLOAT32, num, steps, stop, steps);
                if (t == MLX_FLOAT32) {
                    p.binary("Add", MLX_FLOAT32, num, complement, steps, v.ref(0));
                } else {
                    p.binary("Add", MLX_FLOAT32, num, complement, steps, complement);
                    p.copy(MLX_FLOAT32, t, num, complement, v.ref(0));
                }
            };
        });
        ROUTES.put("eye", v -> {
            int t = outType(v, 0, NUMBERS);
            if (t < 0 || v.args() < 4 || v.size(0) != v.intArg(1) * v.intArg(2)) {
                return null;
            }
            return p -> diagonalFill(p, t, v.ref(0), v.intArg(1), v.intArg(2), v.intArg(3), null, 0);
        });
        ROUTES.put("identity", v -> {
            int t = outType(v, 0, NUMBERS);
            if (t < 0 || v.args() < 2 || v.size(0) != v.intArg(1) * v.intArg(1)) {
                return null;
            }
            return p -> diagonalFill(p, t, v.ref(0), v.intArg(1), v.intArg(1), 0, null, 0);
        });
        ROUTES.put("tri", v -> {
            int t = outType(v, 0, NUMBERS);
            if (t < 0 || v.args() < 4 || v.size(0) != v.intArg(1) * v.intArg(2)) {
                return null;
            }
            int n = v.intArg(1);
            int m = v.intArg(2);
            int k = v.intArg(3);
            return p -> {
                Ref mask = p.scratch((long) n * m);
                triMask(p, n, m, k, mask);
                p.copy(MLX_BOOL, t, n * m, mask, v.ref(0));
            };
        });
        // tril: where(tri(r, c, k), x, 0); triu: where(tri(r, c, k - 1), 0, x).
        ROUTES.put("tril", v -> triangle(v, true));
        ROUTES.put("triu", v -> triangle(v, false));
        // diag(x[n], k): an (n + |k|)^2 zero matrix with x on its k-th diagonal.
        ROUTES.put("diag", v -> {
            int t = pairType(v, 0, 1);
            if (t < 0 || v.args() < 3) {
                return null;
            }
            int k = v.intArg(2);
            int n = v.size(0) + Math.abs(k);
            return v.size(1) != n * n ? null : p -> diagonalFill(p, t, v.ref(1), n, n, k, v.ref(0), 1);
        });
        // diagonal(x[r, c], k): x read along stride c + 1.
        ROUTES.put("diagonal", v -> {
            if (v.args() < 5) {
                return null;
            }
            int r = v.intArg(2);
            int c = v.intArg(3);
            int k = v.intArg(4);
            int length = Math.max(0, k >= 0 ? Math.min(r, c - k) : Math.min(r + k, c));
            long start = k >= 0 ? k : (long) -k * c;
            return length == 0 || !v.isArray(0) || v.size(0) != r * c ? null : view(v, 0, 1, new int[] { length }, new long[] { c + 1L }, start);
        });
        // meshgrid(x[nx], y[ny]): "xy" gives [ny, nx] grids, "ij" gives [nx, ny].
        ROUTES.put("meshgrid", v -> {
            if (!v.isArray(0) || !v.isArray(1) || v.args() < 5) {
                return null;
            }
            int nx = v.size(0);
            int ny = v.size(1);
            boolean ij = v.boolArg(4);
            int[] shape = ij ? new int[] { nx, ny } : new int[] { ny, nx };
            Encoder first = view(v, 0, 2, shape, ij ? new long[] { 1, 0 } : new long[] { 0, 1 }, 0);
            Encoder second = view(v, 1, 3, shape, ij ? new long[] { 0, 1 } : new long[] { 1, 0 }, 0);
            return first == null || second == null ? null : p -> {
                first.encode(p);
                second.encode(p);
            };
        });
        // Windows over n = arange(0, M) in float32, step by step as MLX's ops.cpp builds them.
        ROUTES.put("hanning", v -> window(v, (p, m, n, out) -> {
            p.binaryScalarLeft("Multiply", MLX_FLOAT32, m, (float) (Math.PI / (m - 1)), n, out);
            p.unary("Sin", MLX_FLOAT32, MLX_FLOAT32, m, out, out);
            p.unary("Square", MLX_FLOAT32, MLX_FLOAT32, m, out, out);
        }));
        ROUTES.put("hamming", v -> window(v, (p, m, n, out) -> {
            p.binaryScalarLeft("Multiply", MLX_FLOAT32, m, (float) ((2.0 * Math.PI) / (m - 1)), n, out);
            p.unary("Cos", MLX_FLOAT32, MLX_FLOAT32, m, out, out);
            p.binaryScalarLeft("Multiply", MLX_FLOAT32, m, 0.46f, out, out);
            p.binaryScalarLeft("Subtract", MLX_FLOAT32, m, 0.54f, out, out);
        }));
        ROUTES.put("blackman", v -> window(v, (p, m, n, out) -> {
            Ref term2 = p.scratch(4L * m);
            p.binaryScalarLeft("Multiply", MLX_FLOAT32, m, (float) ((2.0 * Math.PI) / (m - 1)), n, out);
            p.unary("Cos", MLX_FLOAT32, MLX_FLOAT32, m, out, out);
            p.unary("Square", MLX_FLOAT32, MLX_FLOAT32, m, out, term2);
            p.binaryScalarLeft("Multiply", MLX_FLOAT32, m, 0.16f, term2, term2);
            p.binaryScalarLeft("Multiply", MLX_FLOAT32, m, 0.5f, out, out);
            p.binaryScalarLeft("Subtract", MLX_FLOAT32, m, 0.34f, out, out);
            p.binary("Add", MLX_FLOAT32, m, out, term2, out);
        }));
        ROUTES.put("bartlett", v -> window(v, (p, m, n, out) -> {
            p.binaryScalarLeft("Multiply", MLX_FLOAT32, m, 2.0f / (m - 1), n, out);
            p.binaryScalarRight("Subtract", MLX_FLOAT32, m, out, 1.0f, out);
            p.unary("Abs", MLX_FLOAT32, MLX_FLOAT32, m, out, out);
            p.binaryScalarLeft("Subtract", MLX_FLOAT32, m, 1.0f, out, out);
        }));
    }

    private static Encoder fillRoute(View v, int index, double value) {
        int t = outType(v, index, NUMBERS);
        return t < 0 ? null : p -> p.fill(t, v.size(index), value, v.ref(index));
    }

    private static Encoder triangle(View v, boolean lower) {
        int t = pairType(v, 0, 1);
        if (t < 0 || v.args() < 5 || !NUMBERS.contains(t)) {
            return null;
        }
        int r = v.intArg(2);
        int c = v.intArg(3);
        int k = v.intArg(4);
        if (v.size(0) != r * c || v.size(1) != r * c) {
            return null;
        }
        return p -> {
            Ref mask = p.scratch((long) r * c);
            triMask(p, r, c, lower ? k : k - 1, mask);
            if (lower) {
                p.selectScalarRight(t, r * c, mask, v.ref(0), 0, v.ref(1));
            } else {
                p.selectScalar(t, r * c, mask, 0, v.ref(0), v.ref(1));
            }
        };
    }

    private static Encoder scaled(View v, double factor) {
        int t = v.sameType(FLOATS, 0, 1);
        return t < 0 ? null : p -> p.binaryScalarRight("Multiply", t, v.size(0), v.ref(0), factor, v.ref(1));
    }

    interface PredicateBody {
        void encode(Program p, int type, int n);
    }

    /** A predicate (a floating array, a byte array of the same length). */
    private static Encoder predicate(View v, PredicateBody body) {
        if (!v.isArray(0) || !MlxMetalKernels.isFloating(v.dtype(0)) || !v.is(1, MLX_UINT8, v.size(0))) {
            return null;
        }
        int t = v.dtype(0);
        int n = v.size(0);
        return p -> body.encode(p, t, n);
    }

    private static Encoder logical(View v, String op) {
        if (!v.isArray(0) || !v.isArray(1) || v.size(1) != v.size(0) || !v.is(2, MLX_UINT8, v.size(0))) {
            return null;
        }
        int n = v.size(0);
        int ta = v.dtype(0);
        int tb = v.dtype(1);
        if (!ANY.contains(ta) || !ANY.contains(tb)) {
            return null;
        }
        return p -> {
            Ref a = p.scratch(n);
            Ref b = p.scratch(n);
            p.copy(ta, MLX_BOOL, n, v.ref(0), a);
            p.copy(tb, MLX_BOOL, n, v.ref(1), b);
            p.binary(op, MLX_BOOL, n, a, b, v.ref(2));
        };
    }

    private static Encoder complexPart(View v, String op, boolean complexOut) {
        if (!v.isArray(0) || !v.isArray(1) || v.dtype(0) != MLX_FLOAT32 || v.dtype(1) != MLX_FLOAT32 || v.size(0) % 2 != 0) {
            return null;
        }
        int n = v.size(0) / 2;
        if (v.size(1) != (complexOut ? 2 * n : n)) {
            return null;
        }
        return p -> p.unary(op, MLX_COMPLEX64, complexOut ? MLX_COMPLEX64 : MLX_FLOAT32, n, v.ref(0), v.ref(1));
    }

    /**
     * isclose(a, b, rtol, atol, equal_nan), float32: |a - b| <= atol + rtol * |b|, false where either
     * value is infinite unless both are infinite with the same sign, and optionally true where both
     * are NaN. Other types promote through float32 in MLX and stay on the C API.
     */
    private static Encoder isclose(View v) {
        int t = v.sameType(FLOAT32, 0, 1);
        if (t < 0 || v.args() != 6 || !v.is(2, MLX_UINT8, v.size(0))) {
            return null;
        }
        int n = v.size(0);
        double rtol = v.floatArg(3);
        double atol = v.floatArg(4);
        boolean equalNan = v.boolArg(5);
        Ref a = v.ref(0);
        Ref b = v.ref(1);
        Ref out = v.ref(2);
        return p -> {
            Ref rhs = p.scratch(4L * n);
            Ref lhs = p.scratch(4L * n);
            Ref m1 = p.scratch(n);
            Ref m2 = p.scratch(n);
            Ref m3 = p.scratch(n);
            p.unary("Abs", t, t, n, b, rhs);
            p.binaryScalarLeft("Multiply", t, n, rtol, rhs, rhs);
            p.binaryScalarLeft("Add", t, n, atol, rhs, rhs);
            p.binary("Subtract", t, n, a, b, lhs);
            p.unary("Abs", t, t, n, lhs, lhs);
            p.binary("LessEqual", t, n, lhs, rhs, out);
            // any_inf = |a| == inf or |b| == inf; out = out and not any_inf.
            p.unary("Abs", t, t, n, a, lhs);
            p.binaryScalarRight("Equal", t, n, lhs, INF, m1);
            p.unary("Abs", t, t, n, b, lhs);
            p.binaryScalarRight("Equal", t, n, lhs, INF, m2);
            p.binary("LogicalOr", MLX_BOOL, n, m1, m2, m1);
            p.unary("LogicalNot", MLX_BOOL, MLX_BOOL, n, m1, m1);
            p.binary("LogicalAnd", MLX_BOOL, n, out, m1, out);
            // both_inf with the same sign = a == b and |a| == inf.
            p.binary("Equal", t, n, a, b, m2);
            p.unary("Abs", t, t, n, a, lhs);
            p.binaryScalarRight("Equal", t, n, lhs, INF, m3);
            p.binary("LogicalAnd", MLX_BOOL, n, m2, m3, m2);
            p.binary("LogicalOr", MLX_BOOL, n, out, m2, out);
            if (equalNan) {
                p.binary("NotEqual", t, n, a, a, m2);
                p.binary("NotEqual", t, n, b, b, m3);
                p.binary("LogicalAnd", MLX_BOOL, n, m2, m3, m2);
                p.binary("LogicalOr", MLX_BOOL, n, out, m2, out);
            }
        };
    }

    /** How many calls ran as in-place MLX kernels. */
    static long dispatches() {
        return DISPATCHES.get();
    }

    /** Whether {@code functionName} has an in-place route (for reports). */
    static boolean hasRoute(String functionName) {
        return ROUTES.containsKey(functionName);
    }

    /** Runs {@code functionName} in place if it has a route that takes these arguments; false otherwise. */
    static boolean dispatch(String functionName, LibraryInvocation invocation) {
        Route route = ROUTES.get(functionName);
        if (!ENABLED || route == null || invocation.isCapturing() || !(invocation.getDevice() instanceof TornadoNativeStreamSupport streams)) {
            return false;
        }
        Encoder encoder = route.plan(new View(invocation));
        if (encoder == null) {
            return false;
        }
        long queue = streams.getNativeStream(invocation.getExecutionPlanId());
        if (queue == 0) {
            return false;
        }
        try (Program program = new Program(queue)) {
            encoder.encode(program);
        }
        DISPATCHES.incrementAndGet();
        return true;
    }
}
