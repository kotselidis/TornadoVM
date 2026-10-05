/*
 * This file is part of Tornado: A heterogeneous programming framework:
 * https://github.com/beehive-lab/tornadovm
 *
 * Copyright (c) 2025, APT Group, Department of Computer Science,
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
package uk.ac.manchester.tornado.drivers.cuda.natives;

import uk.ac.manchester.tornado.drivers.cuda.ffm.CUDADriverAPI;
import uk.ac.manchester.tornado.drivers.cuda.ffm.CUDAHandles;

public class NativeCommandQueue {

    /**
     * CUDA device pointers are flat addresses, so mapping a source region onto a destination simply
     * yields the source device pointer; no host mapping is needed.
     */
    public static long mapOnDeviceMemoryRegion(long destDevicePtr, long srcDevicePtr) {
        return srcDevicePtr;
    }

    /**
     * Device-to-device copy of the payload, skipping the TornadoVM segment header. Returns the
     * destination pointer on success, or the failing {@code CUresult}.
     */
    public static long mapOnDeviceMemoryNDRegion(long commandQueuePtr, long destDevicePtr, long srcDevicePtr, long offset, int sizeDataType, long headerSize, long sizeSource, long sizeDest) {
        CUDAHandles.Queue queue = CUDAHandles.resolve(commandQueuePtr, CUDAHandles.Queue.class);
        long headerBytes = headerSize * sizeDataType;
        long payloadBytes = Math.max(sizeDest - headerBytes, 0);
        if (queue != null) {
            CUDADriverAPI.cuCtxSetCurrent(queue.context());
        }
        int result = CUDADriverAPI.cuMemcpyDtoD(destDevicePtr + headerBytes, srcDevicePtr + (offset * sizeDataType) + headerBytes, payloadBytes);
        if (result != CUDADriverAPI.CUDA_SUCCESS) {
            return result;
        }
        return destDevicePtr;
    }
}
