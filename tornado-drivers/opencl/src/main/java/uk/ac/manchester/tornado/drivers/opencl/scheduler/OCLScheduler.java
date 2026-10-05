/*
 * This file is part of Tornado: A heterogeneous programming framework:
 * https://github.com/beehive-lab/tornadovm
 *
 * Copyright (c) 2013-2020, APT Group, Department of Computer Science,
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
package uk.ac.manchester.tornado.drivers.opencl.scheduler;

import uk.ac.manchester.tornado.drivers.opencl.OCLDeviceContext;
import uk.ac.manchester.tornado.drivers.opencl.OCLTargetDevice;
import uk.ac.manchester.tornado.drivers.opencl.enums.OCLDeviceType;
import uk.ac.manchester.tornado.runtime.common.TornadoLogger;
import uk.ac.manchester.tornado.runtime.common.TornadoOptions;

public class OCLScheduler {

    /**
     * Supported Vendors for specific Thread-Schedulers
     */
    public enum SUPPORTED_VENDORS {
        AMD("Advanced Micro Devices"), //
        CODEPLAY("Codeplay"), //
        INTEL("Intel"), //
        NVIDIA("NVIDIA");

        private String name;

        SUPPORTED_VENDORS(String name) {
            this.name = name;
        }

        String getName() {
            return this.name;
        }
    }

    private static final int NVIDIA_MAJOR_VERSION_GENERIC_SCHEDULER = 550;
    private static final int NVIDIA_MINOR_VERSION_GENERIC_SCHEDULER = 67;

    private static boolean isDriverVersionCompatible(OCLTargetDevice device) {
        int majorVersion = Integer.parseInt(device.getDriverVersion().split("\\.")[0]);
        int minorVersion = Integer.parseInt(device.getDriverVersion().split("\\.")[1]);

        if (majorVersion == NVIDIA_MAJOR_VERSION_GENERIC_SCHEDULER && minorVersion >= NVIDIA_MINOR_VERSION_GENERIC_SCHEDULER) {
            return true;
        } else {
            return majorVersion > NVIDIA_MAJOR_VERSION_GENERIC_SCHEDULER;
        }
    }

    private static OCLKernelScheduler getInstanceGPUScheduler(final OCLDeviceContext context) {
        OCLTargetDevice device = context.getDevice();
        if (device.getDeviceVendor().contains(SUPPORTED_VENDORS.AMD.getName())) {
            return new OCLAMDScheduler(context);
        } else if (device.getDeviceVendor().contains(SUPPORTED_VENDORS.NVIDIA.getName())) {
            return isDriverVersionCompatible(device) ? new OCLNVIDIAGPUScheduler(context) : new OCLGenericGPUScheduler(context);
        } else {
            return new OCLGenericGPUScheduler(context);
        }
    }

    public static OCLKernelScheduler instanceScheduler(OCLDeviceType type, final OCLDeviceContext context) {
        switch (type) {
            case CL_DEVICE_TYPE_GPU -> {
                return getInstanceGPUScheduler(context);
            }
            case CL_DEVICE_TYPE_ACCELERATOR -> {
                if (context.getDevice().getDeviceVendor().contains(SUPPORTED_VENDORS.CODEPLAY.getName())) {
                    return getInstanceGPUScheduler(context);
                } else {
                    return new OCLCPUScheduler(context);
                }
            }
            case CL_DEVICE_TYPE_CPU -> {
                return TornadoOptions.USE_BLOCK_SCHEDULER ? new OCLCPUScheduler(context) : getInstanceGPUScheduler(context);
            }
            default -> new TornadoLogger().fatal("No scheduler available for device: %s", context);
        }
        return null;
    }

    public static OCLKernelScheduler create(final OCLDeviceContext context) {
        if (context.getDevice().getDeviceType() != null) {
            OCLDeviceType type = context.getDevice().getDeviceType();
            return instanceScheduler(type, context);
        }
        return null;
    }
}