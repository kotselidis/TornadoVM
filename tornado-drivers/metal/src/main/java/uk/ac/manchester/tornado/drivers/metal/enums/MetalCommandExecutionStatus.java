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
package uk.ac.manchester.tornado.drivers.metal.enums;

import uk.ac.manchester.tornado.api.enums.TornadoExecutionStatus;

public enum MetalCommandExecutionStatus {

    // @formatter:off
	 METAL_UNKNOWN (0x4),
	 METAL_COMPLETE(0x0),
	 METAL_RUNNING (0x1),
	 METAL_SUBMITTED (0x2),
	 METAL_QUEUED (0x3),
	 METAL_ERROR (-1);
    // @formatter:on

    private final int value;

    MetalCommandExecutionStatus(final int v) {
        value = v;
    }

    public int getValue() {
        return value;
    }

    public static MetalCommandExecutionStatus createMetalCommandExecutionStatus(final int v) {
        MetalCommandExecutionStatus result;
        switch (v) {
            case 0:
                result = MetalCommandExecutionStatus.METAL_COMPLETE;
                break;
            case 1:
                result = MetalCommandExecutionStatus.METAL_RUNNING;
                break;
            case 2:
                result = MetalCommandExecutionStatus.METAL_SUBMITTED;
                break;
            case 3:
                result = MetalCommandExecutionStatus.METAL_QUEUED;
                break;
            default:
                result = MetalCommandExecutionStatus.METAL_ERROR;
        }
        return result;
    }

    public TornadoExecutionStatus toTornadoExecutionStatus() {
        TornadoExecutionStatus result = TornadoExecutionStatus.UNKNOWN;
        switch (this) {
            case METAL_COMPLETE:
                result = TornadoExecutionStatus.COMPLETE;
                break;
            case METAL_QUEUED:
                result = TornadoExecutionStatus.QUEUED;
                break;
            case METAL_RUNNING:
                result = TornadoExecutionStatus.RUNNING;
                break;
            case METAL_SUBMITTED:
                result = TornadoExecutionStatus.SUBMITTED;
                break;
            default:
                result = TornadoExecutionStatus.ERROR;
                break;
        }
        return result;
    }
}
