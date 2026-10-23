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

import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Moves the {@code __shared__} arrays of a kernel into dynamic shared memory when they do not fit
 * in the static limit.
 *
 * <p>CUDA caps statically declared shared memory at 48 KB per block. Larger allocations, up to
 * the device's opt-in limit (99 KB on sm_86, 227 KB on sm_90), are only reachable through dynamic
 * shared memory: an {@code extern __shared__} buffer whose size is passed at launch, after the
 * function has been granted it with {@code cuFuncSetAttribute}. When the local arrays of a kernel
 * add up to more than 48 KB, each declaration {@code __shared__ T name[N]} becomes a typed pointer
 * into one dynamic buffer, so the kernel body is unchanged. Kernels at or below the limit keep
 * their static declarations.</p>
 *
 * <p>The number of bytes to request is recorded in the source as a marker comment naming the
 * kernel, and read back when the module is installed. Keeping it in the source means the on-disk
 * module cache, which skips code generation and NVRTC, still launches the kernel correctly.</p>
 */
public final class CUDADynamicSharedMemory {

    /** Static {@code __shared__} memory a block may use without opting in. */
    public static final int STATIC_SHARED_MEMORY_LIMIT = 48 * 1024;

    /** Name of the dynamic shared memory buffer the local arrays are carved from. */
    public static final String BUFFER_NAME = "tornado_dynamic_smem";

    /**
     * Alignment of each array inside the buffer: enough for 16-byte vector and {@code cp.async}
     * accesses, and keeps every array on a 128-byte boundary so swizzled layouts see the same bank
     * mapping as a static declaration.
     */
    private static final int ARRAY_ALIGNMENT = 128;

    private static final String MARKER = "// tornado-dynamic-shared-memory ";

    private static final Pattern DECLARATION = Pattern.compile("__shared__ (signed char|int|float|double|long|short|__half2|__half) ([A-Za-z_]\\w*)\\[(\\d+)\\]");

    private CUDADynamicSharedMemory() {
    }

    private static int sizeOf(String type) {
        return switch (type) {
            case "signed char" -> 1;
            case "short", "__half" -> 2;
            case "int", "float", "__half2" -> 4;
            default -> 8; // long, double
        };
    }

    /**
     * Rewrites the {@code __shared__} array declarations of the kernel {@code kernelName} into
     * pointers to one dynamic buffer when they need more than {@link #STATIC_SHARED_MEMORY_LIMIT}
     * bytes. The source is returned unchanged otherwise.
     */
    public static String rewrite(String source, String kernelName) {
        Matcher matcher = DECLARATION.matcher(source);
        long staticBytes = 0;
        while (matcher.find()) {
            staticBytes += Long.parseLong(matcher.group(3)) * sizeOf(matcher.group(1));
        }
        if (staticBytes <= STATIC_SHARED_MEMORY_LIMIT) {
            return source;
        }

        StringBuilder body = new StringBuilder();
        long offset = 0;
        matcher.reset();
        while (matcher.find()) {
            String type = matcher.group(1);
            long bytes = Long.parseLong(matcher.group(3)) * sizeOf(type);
            offset = (offset + ARRAY_ALIGNMENT - 1) / ARRAY_ALIGNMENT * ARRAY_ALIGNMENT;
            String pointer = String.format("%s* %s = (%s*) (%s + %d)", type, matcher.group(2), type, BUFFER_NAME, offset);
            matcher.appendReplacement(body, Matcher.quoteReplacement(pointer));
            offset += bytes;
        }
        matcher.appendTail(body);
        if (offset > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Kernel " + kernelName + " allocates " + offset + " bytes of shared memory");
        }

        // The buffer is declared at file scope, so that every array declared in a nested scope of
        // the kernel still sees it. Repeating the declaration for another kernel of the module is
        // legal, as the type is the same.
        return "extern __shared__ __align__(" + ARRAY_ALIGNMENT + ") unsigned char " + BUFFER_NAME + "[];\n" //
                + MARKER + kernelName + " " + offset + "\n" //
                + body;
    }

    /**
     * Bytes of dynamic shared memory the kernel {@code entryPoint} needs at launch, or zero when
     * its shared arrays are static.
     */
    public static int bytesFor(String source, String entryPoint) {
        Matcher matcher = Pattern.compile(Pattern.quote(MARKER + entryPoint + " ") + "(\\d+)").matcher(source);
        return matcher.find() ? Integer.parseInt(matcher.group(1)) : 0;
    }
}
