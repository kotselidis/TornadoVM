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

        /** The GPU architecture name of this call's device, or null if it cannot be read. */
        String architecture() {
            if (!(invocation.getDevice() instanceof TornadoNativeStreamSupport streams)) {
                return null;
            }
            long queue = streams.getNativeStream(invocation.getExecutionPlanId());
            return queue == 0 ? null : MlxMetalKernels.architecture(queue);
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
        registerRowRoutes();
        registerReductionRoutes();
        registerCompositeRoutes();
        registerMatmulRoutes();
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

    // ---------------------------------------------------------------- row kernels (mlx.fast, softmax)

    private static final int SIMD = 32;

    /** MLX's threadgroup size for its one-threadgroup-per-row kernels: enough simdgroups for {@code axis / nReads}, or all threads when looped. */
    private static long rowGroup(int axis, int nReads, int loopedLimit, long maxThreads) {
        if (axis > loopedLimit) {
            return maxThreads;
        }
        long needed = (axis + nReads - 1) / nReads;
        return SIMD * ((needed + SIMD - 1) / SIMD);
    }

    /** softmax over the last axis of {@code rows x axis}, with {@code precise} accumulation for half types (as the provider asks). */
    private static Encoder softmax(View v, int rows, int axis) {
        int t = v.sameType(FLOATS, 0, 1);
        if (t < 0 || rows <= 0 || axis <= 0 || (long) rows * axis != v.size(0)) {
            return null;
        }
        String name = (axis > 4096 ? "looped_" : "block_") + "softmax_" + (t != MLX_FLOAT32 ? "precise_" : "") + MlxMetalKernels.typeName(t);
        return p -> {
            Program.Launch k = p.launch(name).buffer(0, v.ref(0)).buffer(1, v.ref(1)).i32(2, axis);
            long group = rowGroup(axis, 4, 4096, k.maxThreads);
            k.threads(rows * group, group, 1, 1, 1, 1);
        };
    }

    /** RoPE over x[B, H, T, D] (contiguous) with a static offset (as bytes) or an offset array. */
    private static Encoder rope(View v, int xIndex, int outIndex, int offsetIndex, int first) {
        int t = pairType(v, xIndex, outIndex);
        if (t < 0 || !FLOATS.contains(t) || v.args() < first + 8) {
            return null;
        }
        int b = v.intArg(first);
        int h = v.intArg(first + 1);
        int seq = v.intArg(first + 2);
        int d = v.intArg(first + 3);
        int dims = v.intArg(first + 4);
        boolean traditional = v.boolArg(first + 5);
        float base = (float) (Math.log(v.floatArg(first + 6)) / Math.log(2.0));
        float scale = v.floatArg(first + 7);
        boolean staticOffset = offsetIndex < 0;
        if ((long) b * h * seq * d != v.size(xIndex) || v.size(outIndex) != v.size(xIndex) || dims <= 0 || dims > d || dims % 2 != 0
                || (!staticOffset && !v.is(offsetIndex, MLX_INT32, 1))) {
            return null;
        }
        int offset = staticOffset ? v.intArg(first + 8) : 0;
        long mat = (long) seq * d;
        boolean single = seq == 1;
        String name = "rope_" + (single ? "single_" : "") + MlxMetalKernels.typeName(t);
        return p -> {
            Ref in = v.ref(xIndex);
            if (dims < d) {
                // MLX copies x to the output and rotates the first dims of each row in place.
                p.copy(t, t, v.size(xIndex), in, v.ref(outIndex));
                in = v.ref(outIndex);
            }
            Program.Launch k = p.launch(name, true, traditional, false).buffer(0, in).buffer(1, v.ref(outIndex));
            if (staticOffset) {
                k.i32(2, offset);
            } else {
                k.buffer(2, v.ref(offsetIndex));
            }
            k.f32(3, scale).f32(10, base);
            if (single) {
                k.i64(4, mat).blocks(dims / 2, (long) b * h, 1);
            } else {
                k.i64(4, mat, d, 1).i64(5, mat, d, 1).i64(6, 0).i32(7, h).blocks(dims / 2, seq, (long) b * ((h + 3) / 4));
            }
        };
    }

    private static void registerRowRoutes() {
        ROUTES.put("softmax", v -> v.isArray(0) ? softmax(v, 1, v.size(0)) : null);
        ROUTES.put("softmax_axis", v -> v.args() < 4 ? null : softmax(v, v.intArg(2), v.intArg(3)));
        // rms_norm(x[rows, dim], w[dim], eps).
        ROUTES.put("fast_rms_norm", v -> {
            int t = v.sameType(FLOATS, 0, 2);
            if (t < 0 || v.args() < 6 || !v.isArray(1) || v.dtype(1) != t) {
                return null;
            }
            int rows = v.intArg(3);
            int dim = v.intArg(4);
            float eps = v.floatArg(5);
            if ((long) rows * dim != v.size(0) || v.size(1) != dim) {
                return null;
            }
            String name = "rms" + (dim > 4096 ? "_looped" : "") + MlxMetalKernels.typeName(t);
            return p -> {
                Program.Launch k = p.launch(name).buffer(0, v.ref(0)).buffer(1, v.ref(1)).buffer(2, v.ref(2)).f32(3, eps).i32(4, dim).i32(5, 1);
                long group = rowGroup(dim, 4, 4096, k.maxThreads);
                k.threads(rows * group, group, 1, 1, 1, 1);
            };
        });
        // layer_norm(x[rows, dim], w[dim], b[dim], eps).
        ROUTES.put("fast_layer_norm", v -> {
            int t = v.sameType(FLOATS, 0, 3);
            if (t < 0 || v.args() < 7 || !v.is(1, t, v.intArg(5)) || !v.is(2, t, v.intArg(5))) {
                return null;
            }
            int rows = v.intArg(4);
            int dim = v.intArg(5);
            float eps = v.floatArg(6);
            if ((long) rows * dim != v.size(0)) {
                return null;
            }
            boolean looped = dim > 6656;
            String name = "layer_norm" + (looped ? "_looped" : "") + MlxMetalKernels.typeName(t);
            return p -> {
                Program.Launch k = p.launch(name).buffer(0, v.ref(0)).buffer(1, v.ref(1)).buffer(2, v.ref(2)).buffer(3, v.ref(3)).f32(4, eps).i32(5, dim).i32(6, 1).i32(7, 1);
                long group = rowGroup(dim, looped ? 4 : 8, 6656, k.maxThreads);
                k.threads(rows * group, group, 1, 1, 1, 1);
            };
        });
        // rope(x, out, B, H, T, D, dims, traditional, base, scale, offset); rope_dynamic(x, offset[1], out, B, H, T, D, dims, traditional, base, scale).
        ROUTES.put("fast_rope", v -> rope(v, 0, 1, -1, 2));
        ROUTES.put("fast_rope_dynamic", v -> rope(v, 0, 2, 1, 3));
    }

    // ---------------------------------------------------------------- reductions

    private static final int REDUCE_N_READS = 4;
    private static final int REDUCE_N_WRITES = 4;

    /** MLX's remap_reduce_types: the type the kernel reads and the type it accumulates into. */
    private static int[] reduceTypes(int in, String op) {
        boolean integer = INTEGERS.contains(in);
        if (op.equals("sum") || op.equals("prod")) {
            if (in == MLX_BOOL) {
                return new int[] { MlxNativeLib.MLX_INT8, MLX_INT32 };
            }
            if (integer) {
                return switch (in) {
                    case MLX_UINT8, MlxNativeLib.MLX_UINT16, MlxNativeLib.MLX_UINT32 -> new int[] { in, MlxNativeLib.MLX_UINT32 };
                    case MlxNativeLib.MLX_INT64 -> new int[] { in, in };
                    default -> new int[] { in, MLX_INT32 };
                };
            }
            return new int[] { in, in };
        }
        if (op.equals("and") || op.equals("or")) {
            if (FLOATS.contains(in)) {
                return new int[] { in, MLX_BOOL };
            }
            return switch (MlxMetalKernels.itemSize(in)) {
                case 1 -> new int[] { MLX_BOOL, MLX_BOOL };
                case 2 -> new int[] { MlxNativeLib.MLX_INT16, MLX_BOOL };
                case 4 -> new int[] { MLX_INT32, MLX_BOOL };
                default -> new int[] { MlxNativeLib.MLX_INT64, MLX_BOOL };
            };
        }
        return new int[] { in, in };
    }

    private static long roundUp32(long n) {
        return (n + 31) / 32 * 32;
    }

    /** MLX's all_reduce over {@code n} contiguous elements into {@code out[0]}. */
    private static void allReduce(Program p, String op, int in, long n, Ref x, Ref out) {
        int[] types = reduceTypes(in, op);
        String name = "all_reduce_" + op + MlxMetalKernels.typeName(types[0]);
        if (n <= REDUCE_N_READS * 1024L) {
            long group = roundUp32((n + REDUCE_N_READS - 1) / REDUCE_N_READS);
            p.launch(name).buffer(0, x).buffer(1, out).i64(2, n).i64(3, n).threads(group, group, 1, 1, 1, 1);
            return;
        }
        boolean huge = n * MlxMetalKernels.itemSize(in) > (1L << 26);
        int rows = huge ? 1024 * REDUCE_N_READS : 32 * REDUCE_N_READS;
        long secondGroup = huge ? 1024 : 32;
        Ref partial = p.scratch((long) rows * MlxMetalKernels.itemSize(types[1]));
        long rowSize = (n + rows - 1) / rows;
        long group = roundUp32(Math.min((rowSize + REDUCE_N_READS - 1) / REDUCE_N_READS, 1024));
        p.launch(name).buffer(0, x).buffer(1, partial).i64(2, n).i64(3, rowSize).threads(group, group, rows, 1, 1, 1);
        p.launch("all_reduce_" + op + MlxMetalKernels.typeName(types[1])).buffer(0, partial).buffer(1, out).i64(2, rows).i64(3, rows).threads(secondGroup, secondGroup, 1, 1, 1, 1);
    }

    /** MLX's threadgroup_size_from_row_size. */
    private static long rowReduceGroup(long rowSize) {
        if (rowSize <= 512) {
            return 32;
        }
        if (rowSize <= 1024) {
            return 128;
        }
        return Math.min(1024, roundUp32((rowSize + REDUCE_N_READS - 1) / REDUCE_N_READS));
    }

    /** MLX's row reduce over {@code rows} contiguous rows of {@code rowSize}, with no other reduced axes, into {@code out[rows]}. */
    private static void rowReduce(Program p, String op, int in, long rows, long rowSize, Ref x, Ref out) {
        int[] types = reduceTypes(in, op);
        String type = MlxMetalKernels.typeName(types[0]);
        if (rowSize <= 64) {
            // row_reduce_small with no non-row reductions: one thread per output.
            Program.Launch k = p.launch("row_reduce_small_1_reduce_" + op + type).buffer(0, x).buffer(1, out);
            rowArgs(k, rows, rowSize);
            k.threads(rows, Math.min(rows, 1024), 1, 1, 1, 1);
        } else if (rows >= 32) {
            long group = rowReduceGroup(rowSize);
            if (MlxMetalKernels.itemSize(types[0]) == 8) {
                group = Math.min(group, 512);
            }
            long width = (rows + REDUCE_N_WRITES - 1) / REDUCE_N_WRITES;
            p.launch("row_reduce_simple_" + op + type).buffer(0, x).buffer(1, out).i64(2, rowSize).i64(3, rows).threads(group, group, width, 1, 1, 1);
        } else {
            long group = rowReduceGroup(rowSize);
            Program.Launch k = p.launch("row_reduce_looped_1_reduce_" + op + type).buffer(0, x).buffer(1, out);
            rowArgs(k, rows, rowSize);
            k.threads(group, group, rows, 1, 1, 1);
        }
    }

    /** RowReduceArgs.encode for contiguous rows: one non-reduced dimension, no extra reduced axes. */
    private static void rowArgs(Program.Launch k, long rows, long rowSize) {
        k.i64(2, rowSize).i64(3, 1).i32(4, (int) rows).i64(5, rowSize).i32(6, 1).i32(7, 0).i64(8, 0).i32(9, 0);
    }

    /** The reduced extent of a reduce call: whole array (null), or rows x len when the reduced axes are trailing and contiguous. */
    private record Extent(long rows, long len, boolean whole) {
    }

    /** reduce(x, out): whole; reduce_axis(x, out, outer, len, inner); reduce_axes(x, out, outer, l1, l2, inner). Only inner == 1 routes. */
    private static Extent extent(View v, String form) {
        if (!v.isArray(0)) {
            return null;
        }
        long n = v.size(0);
        switch (form) {
            case "whole":
                return new Extent(1, n, true);
            case "axis":
                if (v.args() < 5 || v.intArg(4) != 1 || (long) v.intArg(2) * v.intArg(3) != n) {
                    return null;
                }
                return new Extent(v.intArg(2), v.intArg(3), false);
            default:
                if (v.args() < 6 || v.intArg(5) != 1 || (long) v.intArg(2) * v.intArg(3) * v.intArg(4) != n) {
                    return null;
                }
                return new Extent(v.intArg(2), (long) v.intArg(3) * v.intArg(4), false);
        }
    }

    /** The reduction itself: all_reduce for a whole array, row reduce otherwise. */
    private static void reduceInto(Program p, String op, int in, Extent e, Ref x, Ref out) {
        if (e.whole()) {
            allReduce(p, op, in, e.len(), x, out);
        } else {
            rowReduce(p, op, in, e.rows(), e.len(), x, out);
        }
    }

    /** sum, prod, max, min, all (and), any (or): output type as MLX gives it. */
    private static Encoder plainReduce(View v, String op, String form) {
        Extent e = extent(v, form);
        if (e == null || !v.isArray(1) || v.size(1) != e.rows() || !ANY.contains(v.dtype(0)) || v.dtype(0) == MLX_COMPLEX64) {
            return null;
        }
        int in = v.dtype(0);
        boolean logical = op.equals("and") || op.equals("or");
        String kernelOp = !logical && in == MLX_BOOL ? (op.equals("min") ? "and" : op.equals("max") ? "or" : op) : op;
        int accumulated = reduceTypes(in, kernelOp)[1];
        int outType = v.dtype(1);
        boolean boolOut = accumulated == MLX_BOOL;
        if (boolOut ? outType != MLX_UINT8 : outType != accumulated) {
            return null;
        }
        return p -> {
            if (boolOut) {
                // MLX's reduce kernels need at least 4 bytes of output; reduce into scratch and copy the bools.
                Ref bools = p.scratch(Math.max(4, e.rows()));
                reduceInto(p, kernelOp, in, e, v.ref(0), bools);
                p.copy(MLX_BOOL, MLX_BOOL, (int) e.rows(), bools, v.ref(1));
            } else {
                reduceInto(p, kernelOp, in, e, v.ref(0), v.ref(1));
            }
        };
    }

    /** static_cast of a double into MLX dtype {@code t}, as NumberOfElements stores it, read back as a double. */
    private static double castTo(int t, double value) {
        return switch (t) {
            case MLX_FLOAT32 -> (float) value;
            case MlxNativeLib.MLX_FLOAT16 -> Float.float16ToFloat(Float.floatToFloat16((float) value));
            default -> Float.intBitsToFloat((Float.floatToRawIntBits((float) value) + 0x7fff + ((Float.floatToRawIntBits((float) value) >>> 16) & 1)) & 0xffff0000);
        };
    }

    /** mean: sum, then multiply by number_of_elements(inverted) as a scalar of the output type (floating inputs). */
    private static Encoder mean(View v, String form) {
        Extent e = extent(v, form);
        int t = v.isArray(0) ? v.dtype(0) : -1;
        if (e == null || !FLOATS.contains(t) || !v.is(1, t, (int) e.rows())) {
            return null;
        }
        double inverse = 1.0 / e.len();
        return p -> {
            reduceInto(p, "sum", t, e, v.ref(0), v.ref(1));
            p.binaryScalarRight("Multiply", t, (int) e.rows(), v.ref(1), inverse, v.ref(1));
        };
    }

    /** var(ddof) and std: mean with keepdims, broadcast subtract, square, sum, and the normaliser MLX uses. */
    private static Encoder variance(View v, String form, int ddofIndex, boolean std) {
        Extent e = extent(v, form);
        int t = v.isArray(0) ? v.dtype(0) : -1;
        if (e == null || !FLOATS.contains(t) || !v.is(1, t, (int) e.rows()) || v.args() <= ddofIndex) {
            return null;
        }
        int ddof = v.intArg(ddofIndex);
        long rows = e.rows();
        long len = e.len();
        int n = v.size(0);
        return p -> {
            int item = MlxMetalKernels.itemSize(t);
            Ref mu = p.scratch(Math.max(4, rows * item));
            Ref d = p.scratch((long) n * item);
            reduceInto(p, "sum", t, e, v.ref(0), mu);
            p.binaryScalarRight("Multiply", t, (int) rows, mu, 1.0 / len, mu);
            if (e.whole()) {
                p.binaryScalarRight("Subtract", t, n, v.ref(0), mu, d);
            } else {
                p.generalBinary("Subtract", t, new int[] { (int) rows, (int) len }, new long[] { len, 1 }, v.ref(0), new long[] { 1, 0 }, mu, d);
            }
            p.unary("Square", t, t, n, d, d);
            reduceInto(p, "sum", t, e, d, v.ref(1));
            if (ddof == 0) {
                p.binaryScalarRight("Multiply", t, (int) rows, v.ref(1), 1.0 / len, v.ref(1));
            } else {
                // maximum(number_of_elements - ddof, 0), each step rounded to the output type.
                double normaliser = Math.max(castTo(t, castTo(t, len) - castTo(t, ddof)), 0.0);
                p.binaryScalarRight("Divide", t, (int) rows, v.ref(1), normaliser, v.ref(1));
            }
            if (std) {
                p.unary("Sqrt", t, t, (int) rows, v.ref(1), v.ref(1));
            }
        };
    }

    /**
     * logsumexp of a whole array: MLX's fused LogSumExp kernel. (The axis forms reduce an inner axis
     * of [outer, len, inner], for which MLX builds a composite instead, so they stay on the C API.)
     */
    private static Encoder logsumexp(View v, String form) {
        Extent e = extent(v, form);
        int t = v.isArray(0) ? v.dtype(0) : -1;
        if (e == null || !FLOATS.contains(t) || !v.is(1, t, (int) e.rows())) {
            return null;
        }
        int axis = (int) e.len();
        String name = (axis > 4096 ? "looped_" : "block_") + "logsumexp_" + MlxMetalKernels.typeName(t);
        return p -> {
            Program.Launch k = p.launch(name).buffer(0, v.ref(0)).buffer(1, v.ref(1)).i32(2, axis);
            long group = rowGroup(axis, 4, 4096, k.maxThreads);
            k.threads(e.rows() * group, group, 1, 1, 1, 1);
        };
    }

    /** argmin/argmax over an axis of x viewed as [outer, len, inner] (uint32 indices into an int array). */
    private static Encoder argReduce(View v, String op, long outer, long len, long inner) {
        if (!v.isArray(0) || !v.is(1, MLX_INT32, (int) (outer * inner)) || outer * len * inner != v.size(0) || !NUMBERS.contains(v.dtype(0))) {
            return null;
        }
        String name = op + "_" + MlxMetalKernels.typeName(v.dtype(0));
        return p -> {
            Program.Launch k = p.launch(name).buffer(0, v.ref(0)).buffer(1, v.ref(1));
            long group = roundUp32(Math.min((len + 3) / 4, k.maxThreads));
            boolean scalar = outer * inner == 1;
            if (scalar) {
                k.i32(2, 0).i64(3, 0).i64(4, 0).i64(5, 0);
            } else {
                k.bytes(2, MlxMetalKernels.intBytes(new int[] { (int) outer, (int) inner })).i64(3, len * inner, 1).i64(4, inner, 1).i64(5, 2);
            }
            k.i64(6, inner).i64(7, len).threads(group, group, outer * inner, 1, 1, 1);
        };
    }

    private static void registerReductionRoutes() {
        String[][] plain = { { "sum", "sum" }, { "prod", "prod" }, { "max", "max" }, { "min", "min" }, { "all", "and" }, { "any", "or" } };
        for (String[] r : plain) {
            ROUTES.put(r[0], v -> plainReduce(v, r[1], "whole"));
            ROUTES.put(r[0] + "_axis", v -> plainReduce(v, r[1], "axis"));
            ROUTES.put(r[0] + "_axes", v -> plainReduce(v, r[1], "axes"));
        }
        ROUTES.put("mean", v -> mean(v, "whole"));
        ROUTES.put("mean_axis", v -> mean(v, "axis"));
        ROUTES.put("mean_axes", v -> mean(v, "axes"));
        ROUTES.put("var", v -> variance(v, "whole", 2, false));
        ROUTES.put("var_axis", v -> variance(v, "axis", 5, false));
        ROUTES.put("var_axes", v -> variance(v, "axes", 6, false));
        ROUTES.put("std", v -> variance(v, "whole", 2, true));
        ROUTES.put("std_axis", v -> variance(v, "axis", 5, true));
        ROUTES.put("std_axes", v -> variance(v, "axes", 6, true));
        ROUTES.put("logsumexp", v -> logsumexp(v, "whole"));
        ROUTES.put("argmin", v -> v.isArray(0) ? argReduce(v, "argmin", 1, v.size(0), 1) : null);
        ROUTES.put("argmin_axis", v -> v.args() < 5 ? null : argReduce(v, "argmin", v.intArg(2), v.intArg(3), v.intArg(4)));
        ROUTES.put("argmax", v -> v.isArray(0) ? argReduce(v, "argmax", 1, v.size(0), 1) : null);
        ROUTES.put("argmax_axis", v -> v.args() < 4 ? null : argReduce(v, "argmax", v.intArg(2), v.intArg(3), 1));
    }

    // ---------------------------------------------------------------- slices and reduction composites

    /** slice(x[rows, cols], [r0:r1:rs, c0:c1:cs]) with positive steps. */
    private static Encoder slice(View v) {
        if (v.args() < 10 || !v.isArray(0)) {
            return null;
        }
        int rows = v.intArg(2);
        int cols = v.intArg(3);
        int r0 = v.intArg(4);
        int r1 = v.intArg(5);
        int rs = v.intArg(6);
        int c0 = v.intArg(7);
        int c1 = v.intArg(8);
        int cs = v.intArg(9);
        if (rs <= 0 || cs <= 0 || r0 < 0 || c0 < 0 || r1 > rows || c1 > cols || r0 >= r1 || c0 >= c1 || v.size(0) != rows * cols) {
            return null;
        }
        int outRows = (r1 - r0 + rs - 1) / rs;
        int outCols = (c1 - c0 + cs - 1) / cs;
        return view(v, 0, 1, new int[] { outRows, outCols }, new long[] { (long) rs * cols, cs }, (long) r0 * cols + c0);
    }

    /** slice_update(x[rows, cols], update[ur, uc] at (r0, c0)): copy x, then write (or add / multiply) the update into its region. */
    private static Encoder sliceUpdate(View v, String op) {
        int t = pairType(v, 0, 2);
        if (t < 0 || v.args() < 9 || !v.isArray(1) || v.dtype(1) != t) {
            return null;
        }
        int rows = v.intArg(3);
        int cols = v.intArg(4);
        int r0 = v.intArg(5);
        int c0 = v.intArg(6);
        int ur = v.intArg(7);
        int uc = v.intArg(8);
        if (r0 < 0 || c0 < 0 || r0 + ur > rows || c0 + uc > cols || ur <= 0 || uc <= 0 || v.size(0) != rows * cols || v.size(2) != rows * cols || v.size(1) != ur * uc) {
            return null;
        }
        long offset = (long) r0 * cols + c0;
        return p -> {
            p.copy(t, t, rows * cols, v.ref(0), v.ref(2));
            if (op == null) {
                block(p, t, v.ref(1), uc, 0, v.ref(2), cols, offset, ur, uc);
            } else {
                // out[region] = op(x[region], update), computed contiguously and written back.
                Ref combined = p.scratch((long) ur * uc * MlxMetalKernels.itemSize(t));
                p.generalBinary(op, t, new int[] { ur, uc }, new long[] { cols, 1 }, v.ref(0).plus(offset, t), new long[] { uc, 1 }, v.ref(1), combined);
                block(p, t, combined, uc, 0, v.ref(2), cols, offset, ur, uc);
            }
        };
    }

    /** softmax over the last two axes of x[d0, d1, d2] (precise): MLX's composite with max, exp, sum and divide over float32. */
    private static Encoder softmaxLastTwo(View v) {
        int t = v.sameType(FLOATS, 0, 1);
        if (t < 0 || v.args() < 5) {
            return null;
        }
        int d0 = v.intArg(2);
        int d1 = v.intArg(3);
        int d2 = v.intArg(4);
        if ((long) d0 * d1 * d2 != v.size(0)) {
            return null;
        }
        if (d1 == 1) {
            return softmax(v, d0, d2);
        }
        int len = d1 * d2;
        int n = d0 * len;
        int f = MLX_FLOAT32;
        return p -> {
            Ref in = v.ref(0);
            if (t != f) {
                Ref cast = p.scratch(4L * n);
                p.copy(t, f, n, in, cast);
                in = cast;
            }
            Ref max = p.scratch(4L * Math.max(1, d0));
            Ref ex = p.scratch(4L * n);
            Ref sum = p.scratch(4L * Math.max(1, d0));
            rowReduce(p, "max", f, d0, len, in, max);
            p.generalBinary("Subtract", f, new int[] { d0, len }, new long[] { len, 1 }, in, new long[] { 1, 0 }, max, ex);
            p.unary("Exp", f, f, n, ex, ex);
            rowReduce(p, "sum", f, d0, len, ex, sum);
            Ref target = t == f ? v.ref(1) : ex;
            p.generalBinary("Divide", f, new int[] { d0, len }, new long[] { len, 1 }, ex, new long[] { 1, 0 }, sum, target);
            if (t != f) {
                p.copy(f, t, n, ex, v.ref(1));
            }
        };
    }

    /** logsumexp over an inner axis (or two): max, exp(x - max), sum, log, add max, and max where it is infinite. */
    private static Encoder logsumexpComposite(View v, String form) {
        Extent e = extent(v, form);
        int t = v.isArray(0) ? v.dtype(0) : -1;
        if (e == null || !FLOATS.contains(t) || !v.is(1, t, (int) e.rows())) {
            return null;
        }
        long rows = e.rows();
        long len = e.len();
        int n = v.size(0);
        return p -> {
            int item = MlxMetalKernels.itemSize(t);
            Ref max = p.scratch(Math.max(4, rows * item));
            Ref ex = p.scratch((long) n * item);
            Ref mask = p.scratch(Math.max(4, rows));
            Ref magnitude = p.scratch(Math.max(4, rows * item));
            rowReduce(p, "max", t, rows, len, v.ref(0), max);
            p.generalBinary("Subtract", t, new int[] { (int) rows, (int) len }, new long[] { len, 1 }, v.ref(0), new long[] { 1, 0 }, max, ex);
            p.unary("Exp", t, t, n, ex, ex);
            rowReduce(p, "sum", t, rows, len, ex, v.ref(1));
            p.unary("Log", t, t, (int) rows, v.ref(1), v.ref(1));
            p.binary("Add", t, (int) rows, v.ref(1), max, v.ref(1));
            p.unary("Abs", t, t, (int) rows, max, magnitude);
            p.binaryScalarRight("Equal", t, (int) rows, magnitude, INF, mask);
            p.select(t, (int) rows, mask, max, v.ref(1), v.ref(1));
        };
    }

    private static void registerCompositeRoutes() {
        ROUTES.put("slice", MlxKernelRoutes::slice);
        ROUTES.put("slice_update", v -> sliceUpdate(v, null));
        ROUTES.put("slice_update_add", v -> sliceUpdate(v, "Add"));
        ROUTES.put("slice_update_prod", v -> sliceUpdate(v, "Multiply"));
        // trace(x[r, c], k): sum(astype(diagonal(x, k), dtype)) as an all-reduce of the contiguous diagonal.
        ROUTES.put("trace", v -> {
            if (v.args() < 5 || !v.isArray(0) || !v.isArray(1) || v.size(1) != 1 || !NUMBERS.contains(v.dtype(0)) || !NUMBERS.contains(v.dtype(1))) {
                return null;
            }
            int r = v.intArg(2);
            int c = v.intArg(3);
            int k = v.intArg(4);
            int length = Math.max(0, k >= 0 ? Math.min(r, c - k) : Math.min(r + k, c));
            int in = v.dtype(0);
            int out = v.dtype(1);
            if (length == 0 || v.size(0) != r * c || reduceTypes(out, "sum")[1] != out) {
                return null;
            }
            long start = k >= 0 ? k : (long) -k * c;
            return p -> {
                Ref diagonal = p.scratch((long) length * MlxMetalKernels.itemSize(out));
                p.generalCopy(in, out, new int[] { length }, new long[] { c + 1L }, v.ref(0).plus(start, in), null, diagonal, null);
                allReduce(p, "sum", out, length, diagonal, v.ref(1));
            };
        });
        // allclose: all(isclose(...)) (float32, as isclose).
        ROUTES.put("allclose", v -> {
            if (v.args() < 6 || v.sameType(FLOAT32, 0, 1) < 0 || !v.is(2, MLX_UINT8, 1)) {
                return null;
            }
            int n = v.size(0);
            double rtol = v.floatArg(3);
            double atol = v.floatArg(4);
            boolean equalNan = v.boolArg(5);
            return p -> {
                Ref flags = p.scratch(n);
                Ref result = p.scratch(4);
                iscloseKernels(p, n, v.ref(0), v.ref(1), flags, rtol, atol, equalNan);
                allReduce(p, "and", MLX_BOOL, n, flags, result);
                p.copy(MLX_BOOL, MLX_BOOL, 1, result, v.ref(2));
            };
        });
        // array_equal: all(equal(a, b)), with NaNEqual when equal_nan is set on floating types.
        ROUTES.put("array_equal", v -> {
            int t = v.sameType(NUMBERS, 0, 1);
            if (t < 0 || v.args() < 4 || !v.is(2, MLX_UINT8, 1)) {
                return null;
            }
            int n = v.size(0);
            String op = v.boolArg(3) && FLOATS.contains(t) ? "NaNEqual" : "Equal";
            return p -> {
                Ref flags = p.scratch(n);
                Ref result = p.scratch(4);
                p.binary(op, t, n, v.ref(0), v.ref(1), flags);
                allReduce(p, "and", MLX_BOOL, n, flags, result);
                p.copy(MLX_BOOL, MLX_BOOL, 1, result, v.ref(2));
            };
        });
        ROUTES.put("softmax_axes", MlxKernelRoutes::softmaxLastTwo);
        ROUTES.put("logsumexp_axis", v -> logsumexpComposite(v, "axis"));
        ROUTES.put("logsumexp_axes", v -> logsumexpComposite(v, "axes"));
    }

    // ---------------------------------------------------------------- matmul (GEMV and steel GEMM)

    /** GEMMParams: M, N, K, lda, ldb, ldd, tiles_n, tiles_m, three int64 batch strides, swizzle_log, k iterations, batch_ndim. */
    private static byte[] gemmParams(int m, int n, int k, int lda, int ldb, int ldd, int tilesN, int tilesM, long strideA, long strideB, long strideD, int kIterations) {
        java.nio.ByteBuffer b = java.nio.ByteBuffer.allocate(72).order(java.nio.ByteOrder.LITTLE_ENDIAN);
        b.putInt(m).putInt(n).putInt(k).putInt(lda).putInt(ldb).putInt(ldd).putInt(tilesN).putInt(tilesM);
        b.putLong(strideA).putLong(strideB).putLong(strideD);
        b.putInt(0).putInt(kIterations).putInt(1);
        return b.array();
    }

    private static int nextPowerOfTwo(int n) {
        int p = 1;
        while (p < n) {
            p <<= 1;
        }
        return p;
    }

    /**
     * a[m, k] @ b for contiguous a, with b either [k, n] contiguous ({@code bTransposed} false, ldb = n)
     * or the transpose of w[n, k] ({@code bTransposed} true, ldb = k): MLX's GEMV, split-K or regular
     * steel GEMM exactly as Matmul::eval_gpu picks them on GPUs before generation 15.
     */
    private static Encoder matmul(View v, int ai, int bi, int oi, int m, int k, int n, boolean bTransposed, String architecture) {
        int t = v.isArray(ai) ? v.dtype(ai) : -1;
        boolean types = FLOATS.contains(t) && v.isArray(oi) && v.dtype(oi) == t && v.isArray(bi) && v.dtype(bi) == t;
        if (!types || v.size(ai) != m * k || v.size(bi) != k * n || v.size(oi) != m * n || m <= 0 || k <= 0 || n <= 0) {
            return null;
        }
        if (MlxMetalKernels.architectureGeneration(architecture) >= 15 || architecture.isEmpty() || (m == 1 && n == 1)) {
            return null;
        }
        char devc = architecture.charAt(architecture.length() - 1);
        String type = MlxMetalKernels.typeName(t);
        int lda = k;
        int ldb = bTransposed ? k : n;
        Ref a = v.ref(ai);
        Ref b = v.ref(bi);
        Ref out = v.ref(oi);
        if (Math.min(m, n) == 1) {
            // gemv_axbpy: the matrix is b when n != 1.
            boolean bMatrix = n != 1;
            boolean transposeMat = bMatrix ? !bTransposed : false;
            int inLen = k;
            int outLen = bMatrix ? n : m;
            int matLd = bMatrix ? ldb : lda;
            int tm = 4;
            int tn = 4;
            int sm = 1;
            int sn = 32;
            int bm = 1;
            int bn = 1;
            int perGroup;
            String name;
            if (transposeMat) {
                if (inLen >= 8192 && outLen >= 2048) {
                    sm = 4;
                    sn = 8;
                } else {
                    sm = 8;
                    sn = 4;
                }
                bn = outLen >= 2048 ? 16 : outLen >= 512 ? 4 : 2;
                tn = outLen < tn ? 1 : tn;
                perGroup = bn * sn * tn;
                name = "gemv_t_" + type;
            } else {
                bm = outLen >= 4096 ? 8 : 4;
                sn = 32;
                if (k <= 64) {
                    bm = 1;
                    sm = 8;
                    sn = 4;
                } else if (k >= 16 * outLen) {
                    bm = 1;
                    bn = 8;
                }
                tm = outLen < tm ? 1 : tm;
                perGroup = bm * sm * tm;
                name = "gemv_" + type;
            }
            String kernel = name + "_bm" + bm + "_bn" + bn + "_sm" + sm + "_sn" + sn + "_tm" + tm + "_tn" + tn + "_nc0_axpby0";
            Ref mat = bMatrix ? b : a;
            Ref vec = bMatrix ? a : b;
            int groups = (outLen + perGroup - 1) / perGroup;
            int gy = bn;
            int gz = bm;
            return p -> p.launch(kernel).buffer(0, mat).buffer(1, vec).buffer(3, out).i32(4, inLen).i32(5, outLen).i32(6, matLd).i32(9, 1)
                    .bytes(10, MlxMetalKernels.intBytes(new int[] { 1 })).i64(11, 0).i64(12, 0).threadgroups(groups, 1, 1, 32, gy, gz);
        }
        int tmTiles = (m + 15) / 16;
        int tnTiles = (n + 15) / 16;
        int tk = k / 16;
        int threshold = (devc == 's' || devc == 'd') ? 2048 : 1024;
        char ta = 'n';
        char tb = bTransposed ? 't' : 'n';
        if (tmTiles * tnTiles <= threshold && tk >= 8 && k >= Math.max(m, n)) {
            // steel_gemm_splitk: float32 partial products, then an accumulate pass.
            int bm = m < 40 ? 16 : 32;
            int bn = n < 40 ? 16 : 32;
            int bk = 16;
            int partitions = Math.min(Math.max(2, nextPowerOfTwo(tk / (((m + 31) / 32) * ((n + 31) / 32)))), 32);
            int kIterations = (k / bk) / partitions;
            boolean mnAligned = m % bm == 0 && n % bn == 0;
            boolean kAligned = k % bk == 0;
            String kernel = "steel_gemm_splitk_" + ta + tb + "_" + type + "_float32_bm" + bm + "_bn" + bn + "_bk" + bk + "_wm2_wn2_MN_" + (mnAligned ? "t" : "n") + "aligned_K_"
                    + (kAligned ? "t" : "n") + "aligned";
            int tilesN = (n + bn - 1) / bn;
            int tilesM = (m + bm - 1) / bm;
            java.nio.ByteBuffer params = java.nio.ByteBuffer.allocate(52).order(java.nio.ByteOrder.LITTLE_ENDIAN);
            params.putInt(m).putInt(n).putInt(k).putInt(lda).putInt(ldb).putInt(n).putInt(tilesN).putInt(tilesM).putInt(partitions).putInt(m * n).putInt(kIterations * bk).putInt(0)
                    .putInt(kIterations);
            byte[] paramBytes = params.array();
            return p -> {
                Ref split = p.scratch(4L * partitions * m * n);
                p.launch(kernel).buffer(0, a).buffer(1, b).buffer(2, split).bytes(3, paramBytes).threadgroups(tilesN, tilesM, partitions, 32, 2, 2);
                p.launch("steel_gemm_splitk_accum_" + type + "_float32").buffer(0, split).buffer(1, out).i32(2, partitions).i32(3, m * n).i32(4, n).blocks(n, m, 1);
            };
        }
        // steel_matmul_regular with the device's tile parameters.
        int bm = 64;
        int bn = 64;
        int bk = 16;
        int wm = 2;
        int wn = 2;
        if (devc == 'g' || devc == 'p') {
            if (bTransposed) {
                bm = 64;
                bn = 32;
                bk = 32;
                wm = 2;
                wn = 2;
            } else if (t != MLX_FLOAT32) {
                bm = 64;
                bn = 64;
                bk = 16;
                wm = 1;
                wn = 2;
            }
        } else if (devc == 'd') {
            boolean nt = bTransposed;
            if ((long) m * n >= 1L << 20) {
                if (t != MLX_FLOAT32) {
                    if (2 * Math.max(m, n) > k) {
                        bm = 64;
                        bn = 64;
                        bk = 16;
                        wm = 1;
                        wn = 2;
                    } else if (nt) {
                        bm = 64;
                        bn = 32;
                        bk = 32;
                        wm = 2;
                        wn = 2;
                    } else {
                        bm = 32;
                        bn = 64;
                        bk = 16;
                        wm = 1;
                        wn = 2;
                    }
                }
            } else if (t != MLX_FLOAT32) {
                if (nt) {
                    bm = 64;
                    bn = 32;
                    bk = 32;
                    wm = 2;
                    wn = 2;
                } else {
                    bm = 64;
                    bn = 64;
                    bk = 16;
                    wm = 1;
                    wn = 2;
                }
            } else if (nt) {
                bm = 32;
                bn = 64;
                bk = 16;
                wm = 1;
                wn = 2;
            } else {
                bm = 64;
                bn = 32;
                bk = 32;
                wm = 2;
                wn = 2;
            }
        }
        String kernel = "steel_gemm_fused_" + ta + tb + "_" + type + "_" + type + "_bm" + bm + "_bn" + bn + "_bk" + bk + "_wm" + wm + "_wn" + wn;
        // Function constants 10 has_batch, 100 use_out_source, 110 do_axpby, 200/201/202 align_M/N/K.
        boolean alignM = m % bm == 0;
        boolean alignN = n % bn == 0;
        boolean alignK = k % bk == 0;
        int tilesN = (n + bn - 1) / bn;
        int tilesM = (m + bm - 1) / bm;
        byte[] params = gemmParams(m, n, k, lda, ldb, n, tilesN, tilesM, 0, 0, (long) m * n, k / bk);
        int gy = wn;
        int gz = wm;
        return p -> p.launchIndexed(kernel, new int[] { 10, 100, 110, 200, 201, 202 }, new boolean[] { false, false, false, alignM, alignN, alignK }).buffer(0, a).buffer(1, b)
                .buffer(3, out).bytes(4, params).threadgroups(tilesN, tilesM, 1, 32, gy, gz);
    }

    private static void registerMatmulRoutes() {
        // matmul(a, b, out, m, k, n); matmul_transposed(a, w[n, k], out, m, k, n) = a @ w^T.
        ROUTES.put("matmul", v -> v.args() < 6 ? null : matmulRoute(v, false));
        ROUTES.put("matmul_transposed", v -> v.args() < 6 ? null : matmulRoute(v, true));
    }

    private static Encoder matmulRoute(View v, boolean transposed) {
        String architecture = v.architecture();
        return architecture == null ? null : matmul(v, 0, 1, 2, v.intArg(3), v.intArg(4), v.intArg(5), transposed, architecture);
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
        return p -> iscloseKernels(p, n, v.ref(0), v.ref(1), v.ref(2), rtol, atol, equalNan);
    }

    /** The kernels of isclose (float32) writing MLX bools to {@code out}. */
    private static void iscloseKernels(Program p, int n, Ref a, Ref b, Ref out, double rtol, double atol, boolean equalNan) {
        int t = MLX_FLOAT32;
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
