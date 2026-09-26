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
    }

    private MlxKernelRoutes() {
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
