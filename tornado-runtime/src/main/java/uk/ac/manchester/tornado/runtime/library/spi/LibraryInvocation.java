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
 * A single library-task call with its arguments already resolved by the
 * TornadoVM interpreter: reference arguments (off-heap arrays/tensors) carry
 * the raw device pointer of their TornadoVM-managed device buffer (past the
 * array header), primitive arguments carry their boxed value.
 */
public final class LibraryInvocation {

    private final Object[] javaArgs;
    private final long[] devicePointers;
    private final boolean[] isReference;
    private final TornadoXPUDevice device;
    private final long executionPlanId;
    private final LibraryContext context;
    private final Object tuning;
    private final boolean capturing;

    public LibraryInvocation(Object[] javaArgs, long[] devicePointers, boolean[] isReference, TornadoXPUDevice device, long executionPlanId, LibraryContext context, Object tuning, boolean capturing) {
        this.javaArgs = javaArgs;
        this.devicePointers = devicePointers;
        this.isReference = isReference;
        this.device = device;
        this.executionPlanId = executionPlanId;
        this.context = context;
        this.tuning = tuning;
        this.capturing = capturing;
    }

    public int getNumArgs() {
        return javaArgs.length;
    }

    /**
     * The original Java argument at the given position (boxed primitive, or the
     * host-side array/tensor object for reference arguments).
     */
    public Object getArg(int index) {
        return javaArgs[index];
    }

    /**
     * The raw device pointer for a reference argument at the given position,
     * pointing at the first data element (past the TornadoVM array header).
     */
    public long getDevicePointer(int index) {
        return devicePointers[index];
    }

    public boolean isReference(int index) {
        return isReference[index];
    }

    public TornadoXPUDevice getDevice() {
        return device;
    }

    public long getExecutionPlanId() {
        return executionPlanId;
    }

    public LibraryContext getContext() {
        return context;
    }

    /**
     * Library-specific tuning options attached via
     * {@link uk.ac.manchester.tornado.api.common.LibraryTaskDescriptor#withTuning(Object)},
     * or null. Opaque to the runtime; interpreted by the provider.
     */
    public Object getTuning() {
        return tuning;
    }

    /**
     * Whether this call is being recorded into a CUDA graph rather than executed
     * immediately. Device allocations, host synchronisation and stream queries are
     * not capture-safe, so a provider that would need one must reject the call
     * instead of performing it. Sizing work of that kind belongs in
     * {@link TornadoLibraryProvider#prepare}, which runs before the capture starts.
     */
    public boolean isCapturing() {
        return capturing;
    }
}
