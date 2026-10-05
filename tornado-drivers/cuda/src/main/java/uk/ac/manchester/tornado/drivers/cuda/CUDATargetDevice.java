/*
 * This file is part of Tornado: A heterogeneous programming framework:
 * https://github.com/beehive-lab/tornadovm
 *
 * Copyright (c) 2020, APT Group, Department of Computer Science,
 * School of Engineering, The University of Manchester. All rights reserved.
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
package uk.ac.manchester.tornado.drivers.cuda;

import java.nio.ByteOrder;

import uk.ac.manchester.tornado.api.TornadoTargetDevice;
import uk.ac.manchester.tornado.drivers.cuda.enums.CUDADeviceType;

public interface CUDATargetDevice extends TornadoTargetDevice {

    long getDevicePointer();

    String getVersion();

    int getIndex();

    int getWordSize();

    ByteOrder getByteOrder();

    boolean isDeviceDoubleFPSupported();

    String getDeviceExtensions();

    CUDADeviceType getDeviceType();

    String getDeviceVendor();

    String getDriverVersion();

    boolean isDeviceAvailable();

    String getDeviceOpenCLCVersion();

    boolean isLittleEndian();

    CUDADeviceContextInterface getDeviceContext();

    void setDeviceContext(CUDADeviceContextInterface deviceContext);

    int deviceVersion();

    /**
     * Number of asynchronous DMA copy engines (CUDA {@code CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT}).
     * {@code 0} = copies cannot overlap compute; {@code 1} = one direction can overlap compute;
     * {@code >= 2} = H2D and D2H can both overlap compute and each other. This is the hardware ceiling
     * on the number of useful role-based transfer streams. Virtual devices report {@code 0}.
     */
    default int getAsyncEngineCount() {
        return 0;
    }

    /**
     * Whether the device can execute multiple kernels from the same context concurrently
     * (CUDA {@code CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS}). Virtual devices report {@code false}.
     */
    default boolean supportsConcurrentKernels() {
        return false;
    }
}
