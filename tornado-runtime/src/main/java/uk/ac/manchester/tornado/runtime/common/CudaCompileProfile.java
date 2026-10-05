/*
 * This file is part of Tornado: A heterogeneous programming framework:
 * https://github.com/beehive-lab/tornadovm
 *
 * Copyright (c) 2026, APT Group, Department of Computer Science,
 * The University of Manchester. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.
 *
 * Linking this library statically or dynamically with other modules is
 * making a combined work based on this library. Thus, the terms and
 * conditions of the GNU General Public License cover the whole
 * combination.
 *
 * As a special exception, the copyright holders of this library give you
 * permission to link this library with independent modules to produce an
 * executable, regardless of the license terms of these independent
 * modules, and to copy and distribute the resulting executable under
 * terms of your choice, provided that you also meet, for each linked
 * independent module, the terms and conditions of the license of that
 * module. An independent module is a module which is not derived from
 * or based on this library. If you modify this library, you may extend
 * this exception to your version of the library, but you are not
 * obligated to do so. If you do not wish to do so, delete this
 * exception statement from your version.
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
package uk.ac.manchester.tornado.runtime.common;

/**
 * Named bundle of NVRTC options, selected with {@code -Dtornado.cuda.compile.profile=<name>}.
 *
 * <p>
 * The CUDA backend already compiles through NVRTC and already threads per-backend compiler flags all the
 * way to {@code CUDACodeCache}, but reaching a useful combination meant knowing which NVRTC spellings to
 * put in {@code tornado.cuda.compiler.flags}. These profiles name the combinations that are actually
 * wanted; {@link #DEFAULT} changes nothing, and flags set explicitly through
 * {@code tornado.cuda.compiler.flags} or {@code withCompilerFlags(CUDA, ...)} are kept and take
 * precedence, since they are appended after the profile.
 * </p>
 */
public enum CudaCompileProfile {

    /** Whatever NVRTC does by default: no extra options. */
    DEFAULT(""),

    /** Trade floating-point strictness for speed. */
    FAST("--use_fast_math --extra-device-vectorization"),

    /**
     * Keep line-number information in the cubin so Nsight Systems and Nsight Compute can attribute
     * samples to the generated CUDA C. Without it, kernel-level profiling has to be inferred.
     */
    DEBUG("-lineinfo"),

    /**
     * Reproducible arithmetic: no fast-math and no FMA contraction, so a kernel's result does not depend
     * on how the compiler fused its multiplies and adds.
     */
    REPRO("--fmad=false");

    private final String flags;

    CudaCompileProfile(String flags) {
        this.flags = flags;
    }

    public String getFlags() {
        return flags;
    }

    public static CudaCompileProfile parse(String value) {
        if (value == null || value.isBlank()) {
            return DEFAULT;
        }
        return switch (value.trim().toLowerCase()) {
            case "default", "none" -> DEFAULT;
            case "fast" -> FAST;
            case "debug", "lineinfo" -> DEBUG;
            case "repro", "deterministic" -> REPRO;
            default -> {
                System.err.println("[TornadoVM] unknown tornado.cuda.compile.profile value '" + value + "', using 'default'. " //
                        + "Valid values: default, fast, debug, repro.");
                yield DEFAULT;
            }
        };
    }
}
