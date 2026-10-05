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
package uk.ac.manchester.tornado.drivers.opencl.enums;

public enum OCLDeviceType {

    // @formatter:off
    Unknown (-1),
	CL_DEVICE_TYPE_DEFAULT(1 << 0),
	CL_DEVICE_TYPE_CPU(1 << 1),
	CL_DEVICE_TYPE_GPU(1 << 2),
	CL_DEVICE_TYPE_ACCELERATOR(1 << 3),
	CL_DEVICE_TYPE_CUSTOM(1 << 4),
	CL_DEVICE_TYPE_ALL(0xFFFFFFFFL);
    // @formatter:on

    private final long value;

    OCLDeviceType(final long v) {
        value = v;
    }

    public long getValue() {
        return value;
    }

    public static OCLDeviceType toDeviceType(final long v) {
        return switch ((int) v) {
            case 1 -> OCLDeviceType.CL_DEVICE_TYPE_DEFAULT;
            case 1 << 1 -> OCLDeviceType.CL_DEVICE_TYPE_CPU;
            case 1 << 2 -> OCLDeviceType.CL_DEVICE_TYPE_GPU;
            case 1 << 3 -> OCLDeviceType.CL_DEVICE_TYPE_ACCELERATOR;
            case 1 << 4 -> OCLDeviceType.CL_DEVICE_TYPE_CUSTOM;
            case 0xFFFFFFFF -> OCLDeviceType.CL_DEVICE_TYPE_ALL;
            default -> OCLDeviceType.Unknown;
        };
    }
}
