/*
 * This file is part of Tornado: A heterogeneous programming framework:
 * https://github.com/beehive-lab/tornadovm
 *
 * Copyright (c) 2026, APT Group, Department of Computer Science,
 * School of Engineering, The University of Manchester. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 */
package uk.ac.manchester.tornado.drivers.cuda.graal.backend;

import java.util.regex.Pattern;

/**
 * CUDA C preamble prepended to compiled kernels that need it.
 *
 * <p>The code generator emits native CUDA C / NVRTC types and intrinsics
 * directly (native unsigned types, inline relational operators, {@code fmin}/
 * {@code fmax}-based clamp, inline radians/sign, real {@code atomic*}
 * intrinsics, and componentwise vector expressions). The half-precision header
 * is only injected when the emitted kernel actually references fp16 constructs;
 * the generated source is scanned for them in
 * {@code CUDACompilationResultBuilder#finish} through {@link #needsFp16Header}.
 * That scan matches whole identifiers rather than substrings, which is what a
 * shared fp16 tile needs: its declaration and the pointer casts around it are the
 * kernel's only fp16 constructs, and while they used to be spelt with the header's
 * unprefixed {@code half} alias, a {@code contains("__half")} test did not see them
 * and NVRTC rejected the kernel with {@code identifier "half" is undefined}. The
 * backend now spells every emitted fp16 type {@code __half} / {@code __half2}, and
 * the scan still accepts the unprefixed aliases so the two cannot drift apart
 * again. Keeping the
 * include conditional limits the blast radius on toolkits whose on-disk
 * {@code cuda_fp16.hpp} does not compile under NVRTC (it references
 * {@code NV_IF_ELSE_TARGET} from {@code <nv/target>}, which is excluded when
 * {@code __CUDACC_RTC__} is defined): only kernels that genuinely need fp16
 * depend on the header resolving. How the include is resolved at compile time
 * (NVRTC built-ins first, toolkit include paths as a fallback) is handled in
 * the JNI layer ({@code CUDAProgram.cpp#compile_with_nvrtc}).
 *
 * <p>Note: DP4A is emitted as inline PTX ({@code dp4a.s32.s32}) directly at the
 * call site (see {@code CUDALIRStmt.Dp4aStmt}), so it needs no preamble helper.
 */
public final class CUDAPreamble {

    private CUDAPreamble() {
    }

    // @formatter:off
    public static final String PREAMBLE =
        "#include <cuda_fp16.h>\n";

    /**
     * FP8 header, injected (after the fp16 include - cuda_fp8.h builds on
     * cuda_fp16.h's __half_raw) only when the kernel references cuda_fp8.h
     * constructs, with the same source-scan gating as PREAMBLE.
     */
    public static final String FP8_PREAMBLE =
        "#include <cuda_fp8.h>\n";

    /**
     * Header and namespace aliases for a CUDA Tile kernel. Requires CUDA 13.3 or newer and a
     * compilation with --enable-tile; the literals namespace supplies the _ic suffix used for
     * the compile-time constants in tile shapes.
     */
    /**
     * Header for bfloat16. A tile whose element type is __nv_bfloat16 needs this, and the type
     * appears only in generated tile code, so it is gated on the same source scan as the others.
     */
    public static final String BF16_PREAMBLE =
        "#include <cuda_bf16.h>\n";

    /** The name a grid barrier is emitted as; see {@link #GRID_SYNC_PREAMBLE}. */
    public static final String GRID_SYNC_FUNCTION = "__tornado_grid_sync";

    /**
     * Device-side grid barrier for {@code KernelContext.gridBarrier()}, injected only when the
     * kernel calls it.
     *
     * <p>The arrival scheme of CUDA's cooperative-groups grid sync: one word per module, and one
     * atomic per block per barrier. Thread 0 of each block publishes the block's writes and adds 1
     * to the word: a release atomic from sm_70, cumulative over the writes the block's
     * {@code __syncthreads} ordered before it, and {@code __threadfence} before sm_70. Block 0 adds
     * {@code 0x80000000 - (blocks - 1)}. The adds of one barrier sum to exactly {@code 0x80000000},
     * so the last arrival flips bit 31, and every block waits for bit 31 to differ from the value
     * it saw on arrival. Nothing is reset, so the word is reusable after every barrier and by
     * launches of any grid size. The two {@code __syncthreads} hold the rest of the block on
     * either side.
     *
     * <p>It is correct only when every block of the grid is resident at once, which is why a
     * kernel that contains it is launched with {@code cuLaunchCooperativeKernel}. Two grids of
     * the same kernel running concurrently share the word and must not overlap.
     */
    public static final String GRID_SYNC_PREAMBLE =
        "__device__ unsigned int __tornado_grid_sync_state;\n"
        + "static __device__ __forceinline__ void " + GRID_SYNC_FUNCTION + "() {\n"
        + "  __syncthreads();\n"
        + "  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {\n"
        + "    unsigned int blocks = gridDim.x * gridDim.y * gridDim.z;\n"
        + "    bool first = blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0;\n"
        + "    unsigned int add = first ? 0x80000000u - (blocks - 1u) : 1u;\n"
        + "#if __CUDA_ARCH__ >= 700\n"
        + "    unsigned int old;\n"
        + "    unsigned int now;\n"
        + "    asm volatile(\"atom.add.release.gpu.u32 %0, [%1], %2;\" : \"=r\"(old) : \"l\"(&__tornado_grid_sync_state), \"r\"(add) : \"memory\");\n"
        + "    do {\n"
        + "      asm volatile(\"ld.acquire.gpu.u32 %0, [%1];\" : \"=r\"(now) : \"l\"(&__tornado_grid_sync_state) : \"memory\");\n"
        + "    } while (((old ^ now) & 0x80000000u) == 0u);\n"
        + "#else\n"
        + "    volatile unsigned int *arrived = &__tornado_grid_sync_state;\n"
        + "    __threadfence();\n"
        + "    unsigned int old = atomicAdd(&__tornado_grid_sync_state, add);\n"
        + "    while (((old ^ *arrived) & 0x80000000u) == 0u) {\n"
        + "    }\n"
        + "    __threadfence();\n"
        + "#endif\n"
        + "  }\n"
        + "  __syncthreads();\n"
        + "}\n";

    public static final String TILE_PREAMBLE =
        "#include \"cuda_tile.h\"\n"
        + "namespace ct = cuda::tiles;\n"
        + "using namespace ct::literals;\n";
    // @formatter:on

    /**
     * Matches any identifier declared by {@code cuda_fp16.h}.
     *
     * <p>The alternatives are bounded by explicit "not an identifier character" look-around
     * rather than {@code \b}, because {@code _} is a word character: a plain {@code \bhalf\b}
     * would also fire on the generated variable name {@code half_4}. The alternatives are:
     *
     * <ul>
     * <li>{@code half} / {@code half2} - the unprefixed aliases the header typedefs in C++.
     * The backend itself emits {@code __half} everywhere, but hand-written CUDA C inside a LIR
     * statement can reach for the shorter alias, and it needs the same header.</li>
     * <li>{@code __h...} - every conversion, arithmetic and comparison intrinsic in the header
     * shares that prefix ({@code __half}, {@code __half2float}, {@code __hadd2}, {@code __hfma2},
     * {@code __h2div}, {@code __high2float}). No other identifier the backend emits starts with
     * {@code __h}; keep it that way, or this scan starts pulling the header into kernels that do
     * not need it.</li>
     * <li>{@code __float2half...} and {@code __ushort_as_half} - the two families that do not
     * start with {@code __h}.</li>
     * </ul>
     */
    private static final Pattern FP16_CONSTRUCT = Pattern.compile(
        "(?<![A-Za-z0-9_])(?:half2?|__h[A-Za-z0-9_]*|__float2half[A-Za-z0-9_]*|__ushort_as_half)(?![A-Za-z0-9_])");

    /** Matches the {@code cuda_fp8.h} types and the conversions the backend emits around them. */
    private static final Pattern FP8_CONSTRUCT = Pattern.compile(
        "(?<![A-Za-z0-9_])(?:__nv_fp8[A-Za-z0-9_]*|__nv_cvt[A-Za-z0-9_]*fp8[A-Za-z0-9_]*|cvt_float_to_fp8)(?![A-Za-z0-9_])");

    /** Matches the {@code cuda_bf16.h} types. */
    private static final Pattern BF16_CONSTRUCT = Pattern.compile(
        "(?<![A-Za-z0-9_])__nv_bfloat16[A-Za-z0-9_]*(?![A-Za-z0-9_])");

    /**
     * Returns whether {@code source} calls the grid barrier and does not already define it.
     */
    public static boolean needsGridSync(String source) {
        return !source.contains("__tornado_grid_sync_state") && source.contains(GRID_SYNC_FUNCTION + "(");
    }

    /**
     * Returns whether a kernel compiled from {@code source} synchronises its whole grid and so must
     * be launched cooperatively.
     */
    public static boolean needsCooperativeLaunch(String source) {
        return source.contains(GRID_SYNC_FUNCTION + "(");
    }

    /**
     * Returns whether {@code source} references an fp16 construct and does not already include
     * {@link #PREAMBLE}.
     */
    public static boolean needsFp16Header(String source) {
        return !source.contains("cuda_fp16.h") && FP16_CONSTRUCT.matcher(source).find();
    }

    /**
     * Returns whether {@code source} references an fp8 construct and does not already include
     * {@link #FP8_PREAMBLE}.
     */
    public static boolean needsFp8Header(String source) {
        return !source.contains("cuda_fp8.h") && FP8_CONSTRUCT.matcher(source).find();
    }

    /**
     * Returns whether {@code source} references a bf16 construct and does not already include
     * {@link #BF16_PREAMBLE}.
     */
    public static boolean needsBf16Header(String source) {
        return !source.contains("cuda_bf16.h") && BF16_CONSTRUCT.matcher(source).find();
    }
}
