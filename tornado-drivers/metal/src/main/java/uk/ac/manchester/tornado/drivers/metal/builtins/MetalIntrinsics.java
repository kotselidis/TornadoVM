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
package uk.ac.manchester.tornado.drivers.metal.builtins;

import tornado.graal.compiler.api.replacements.Fold;

public class MetalIntrinsics {

    public static native int get_global_id(int value);

    public static native int get_local_id(int value);

    public static native int get_global_size(int value);

    public static native int get_local_size(int value);

    public static native int get_group_id(int value);

    public static native int get_group_size(int value);

    /**
     * <p>
     * <code>
     * barrier(CLK_LOCAL_MEM_FENCE);
     * </code>
     * </p>
     */
    public static native void localBarrier();

    /**
     * <p>
     * <code>
     * barrier(CLK_GLOBAL_MEM_FENCE);
     * </code>
     * </p>
     */
    public static native void globalBarrier();

    public static native void printf();

    public static native void printEmpty();

    public static native void createLocalMemory(int[] array, int size);

    public static int fmax(float a, float b) {
        return 0;
    }

    @Fold
    public static int fmax() {
        return 0;
    }

}
