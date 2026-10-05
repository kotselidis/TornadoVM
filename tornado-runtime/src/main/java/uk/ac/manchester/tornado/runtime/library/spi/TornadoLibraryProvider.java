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
package uk.ac.manchester.tornado.runtime.library.spi;

import uk.ac.manchester.tornado.runtime.common.TornadoXPUDevice;

/**
 * Service-provider interface for external native library bindings (e.g.,
 * NVIDIA cuBLAS, cuDNN, cuFFT). Library binding modules implement this
 * interface and register it via {@link java.util.ServiceLoader}
 * ({@code provides} in their module descriptor), so new libraries plug in
 * without changes to the TornadoVM core runtime.
 *
 * <p>
 * A provider is responsible for loading its native binding, creating an opaque
 * per-(device, execution-plan) {@link LibraryContext} (e.g., a cuBLAS handle
 * bound to the device's native stream), and dispatching function calls with
 * arguments already resolved to device pointers by the interpreter.
 * </p>
 */
public interface TornadoLibraryProvider {

    /**
     * Unique library identifier matched against
     * {@link uk.ac.manchester.tornado.api.common.LibraryTaskDescriptor#getLibraryName()},
     * e.g. {@code "nvidia/cublas"}.
     */
    String libraryName();

    /**
     * Returns true if this provider can execute on the given device (e.g.,
     * requires a CUDA backend device).
     */
    boolean canHandle(TornadoXPUDevice device);

    /**
     * Creates the native execution context for the given device and execution
     * plan (e.g., creates a cuBLAS handle and binds it to the device's stream).
     */
    LibraryContext createContext(TornadoXPUDevice device, long executionPlanId);

    /**
     * Optional hook invoked before every launch region, after the context
     * exists - in particular BEFORE CUDA graph capture starts on the first
     * execution. Providers whose per-shape plans allocate device memory (e.g.,
     * cuFFT work areas, cuDNN convolution workspaces) create them here so the
     * later {@link #dispatch} is capture-safe. Called repeatedly:
     * implementations must be idempotent (typically a plan-cache lookup).
     */
    default void prepare(uk.ac.manchester.tornado.api.common.LibraryTaskDescriptor descriptor, LibraryContext context) {
    }

    /**
     * Executes the given library function. Arguments are provided resolved to
     * device pointers via the {@link LibraryInvocation}.
     */
    void dispatch(String functionName, LibraryInvocation invocation);

    /**
     * Releases native resources held by a context previously returned by
     * {@link #createContext}.
     */
    void destroyContext(LibraryContext context);
}
