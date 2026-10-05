/*
 * Copyright (c) 2018, 2020-2022, 2024, 2025, APT Group, Department of Computer Science,
 * The University of Manchester. All rights reserved.
 * Copyright (c) 2009, 2017, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation. Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
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

package uk.ac.manchester.tornado.drivers.cuda.graal;

import jdk.vm.ci.meta.JavaType;
import jdk.vm.ci.meta.ResolvedJavaMethod;
import jdk.vm.ci.meta.Signature;

/**
 * Recognises tile kernels and holds the small amount of shared vocabulary the CUDA Tile path
 * needs across the backend.
 *
 * <p>
 * A task is a tile task when its kernel method takes a
 * {@code uk.ac.manchester.tornado.api.tile.TileContext} parameter, in the same way that a
 * {@code KernelContext} parameter selects the kernel-parallel API. The check is by type name
 * rather than by class literal so that recognising a tile kernel never depends on the API
 * class being loadable in the compiler.
 * </p>
 */
public final class CUDATileKernels {

    /**
     * Fully qualified name of the API type whose presence marks a tile kernel.
     */
    public static final String TILE_CONTEXT = "uk.ac.manchester.tornado.api.tile.TileContext";

    private CUDATileKernels() {
    }

    /**
     * @param method
     *     candidate kernel method
     * @return true when this method is a tile kernel entry point
     */
    public static boolean isTileKernel(ResolvedJavaMethod method) {
        if (method == null) {
            return false;
        }
        Signature signature = method.getSignature();
        for (int i = 0; i < signature.getParameterCount(false); i++) {
            JavaType parameterType = signature.getParameterType(i, null);
            if (isTileContext(parameterType)) {
                return true;
            }
        }
        return false;
    }

    /**
     * @param type
     *     parameter type to test
     * @return true when the type is the tile context, which is host side only and is therefore
     *     dropped from the generated kernel signature
     */
    public static boolean isTileContext(JavaType type) {
        return type != null && TILE_CONTEXT.equals(type.toJavaName());
    }
}
