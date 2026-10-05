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
package uk.ac.manchester.tornado.drivers.metal;

import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.List;

import uk.ac.manchester.tornado.api.exceptions.TornadoBailoutRuntimeException;
import uk.ac.manchester.tornado.drivers.metal.enums.MetalDeviceType;
import uk.ac.manchester.tornado.drivers.metal.enums.MetalPlatformInfo;
import uk.ac.manchester.tornado.drivers.metal.exceptions.MetalException;
import uk.ac.manchester.tornado.drivers.metal.ffm.MetalObjects;

public class MetalPlatform implements TornadoPlatformInterface {

    private final int index;
    private final long metalPlatformPtr;
    private final List<MetalTargetDevice> devices;

    private enum Vendor {
        CODEPLAY("Codeplay"), //
        INTEL("Intel"), //
        AMD("AMD"), //
        NVIDIA("Nvidia"), //
        MESA("Mesa/X.org");

        final String vendorName;

        Vendor(String vendorName) {
            this.vendorName = vendorName;
        }

        String getVendorName() {
            return vendorName;
        }
    }

    public MetalPlatform(int index, long platformPointers) {
        this.index = index;
        this.metalPlatformPtr = platformPointers;
        this.devices = new ArrayList<>();

        final int deviceCount;

        if (isVendor(Vendor.MESA)) {
            deviceCount = metalGetDeviceCount(platformPointers, MetalDeviceType.METAL_DEVICE_TYPE_GPU.getValue());
        } else {
            deviceCount = metalGetDeviceCount(platformPointers, MetalDeviceType.METAL_DEVICE_TYPE_ALL.getValue());
        }

        final long[] ids = new long[deviceCount];
        if (isVendor(Vendor.MESA)) {
            metalGetDeviceIDs(platformPointers, MetalDeviceType.METAL_DEVICE_TYPE_GPU.getValue(), ids);
        } else {
            metalGetDeviceIDs(platformPointers, MetalDeviceType.METAL_DEVICE_TYPE_ALL.getValue(), ids);
        }
        for (int i = 0; i < ids.length; i++) {
            devices.add(new MetalDevice(i, ids[i]));
        }

    }

    private boolean isVendor(Vendor vendor) {
        return this.getVendor().toLowerCase().startsWith(vendor.getVendorName().toLowerCase());
    }

    String metalGetPlatformInfo(long id, int info) {
        return MetalObjects.platformInfo(info);
    }

    int metalGetDeviceCount(long id, long type) {
        return MetalObjects.deviceCount(type);
    }

    int metalGetDeviceIDs(long id, long type, long[] devices) {
        return MetalObjects.deviceIDs(type, devices);
    }

    long metalCreateContext(long platform, long[] devices) throws MetalException {
        return MetalObjects.createContext(devices);
    }

    public List<MetalTargetDevice> getDevices() {
        return devices;
    }

    public MetalContext createContext() {
        MetalContext contextObject;
        final LongBuffer deviceIds = LongBuffer.allocate(devices.size());
        devices.stream().mapToLong(MetalTargetDevice::getDevicePointer).forEach(deviceIds::put);
        try {
            long contextPtr = metalCreateContext(metalPlatformPtr, deviceIds.array());
            contextObject = new MetalContext(this, contextPtr, devices);
        } catch (MetalException e) {
            throw new TornadoBailoutRuntimeException(e.getMessage());
        }
        return contextObject;
    }

    public void cleanup() {
    }

    public String getProfile() {
        return metalGetPlatformInfo(metalPlatformPtr, MetalPlatformInfo.METAL_PLATFORM_PROFILE.getValue());
    }

    @Override
    public String getVersion() {
        return metalGetPlatformInfo(metalPlatformPtr, MetalPlatformInfo.METAL_PLATFORM_VERSION.getValue());
    }

    public String getName() {
        return metalGetPlatformInfo(metalPlatformPtr, MetalPlatformInfo.METAL_PLATFORM_NAME.getValue());
    }

    public String getVendor() {
        return metalGetPlatformInfo(metalPlatformPtr, MetalPlatformInfo.METAL_PLATFORM_VENDOR.getValue());
    }

    public String getExtensions() {
        return metalGetPlatformInfo(metalPlatformPtr, MetalPlatformInfo.METAL_PLATFORM_EXTENSIONS.getValue());
    }

    @Override
    public String toString() {
        String sb = String.format("name=%s, num. devices=%d, ", getName(), devices.size()) + String.format("version=%s", getVersion());
        return sb.trim();
    }

    public int getIndex() {
        return index;
    }

}
