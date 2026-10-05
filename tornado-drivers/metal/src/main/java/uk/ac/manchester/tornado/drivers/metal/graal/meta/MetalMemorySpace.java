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
package uk.ac.manchester.tornado.drivers.metal.graal.meta;

import tornado.graal.compiler.core.common.LIRKind;

import jdk.vm.ci.meta.Value;
import uk.ac.manchester.tornado.api.exceptions.TornadoInternalError;
import uk.ac.manchester.tornado.drivers.metal.graal.MetalArchitecture;
import uk.ac.manchester.tornado.drivers.metal.graal.asm.MetalAssemblerConstants;

public class MetalMemorySpace extends Value {

    public static final MetalMemorySpace GLOBAL = new MetalMemorySpace(MetalAssemblerConstants.GLOBAL_MEM_MODIFIER);
    public static final MetalMemorySpace SHARED = new MetalMemorySpace(MetalAssemblerConstants.SHARED_MEM_MODIFIER);
    public static final MetalMemorySpace LOCAL = new MetalMemorySpace(MetalAssemblerConstants.LOCAL_MEM_MODIFIER);
    public static final MetalMemorySpace PRIVATE = new MetalMemorySpace(MetalAssemblerConstants.PRIVATE_MEM_MODIFIER);
    public static final MetalMemorySpace CONSTANT = new MetalMemorySpace(MetalAssemblerConstants.CONSTANT_MEM_MODIFIER);

    private final String name;

    protected MetalMemorySpace(String name) {
        super(LIRKind.Illegal);
        this.name = name;
    }

    public MetalArchitecture.MetalMemoryBase getBase() {
        if (this == GLOBAL) {
            return MetalArchitecture.globalSpace;
        } else if (this == LOCAL) {
            return MetalArchitecture.localSpace;
        } else if (this == CONSTANT) {
            return MetalArchitecture.constantSpace;
        } else if (this == PRIVATE) {
            return MetalArchitecture.privateSpace;
        } else {
            TornadoInternalError.shouldNotReachHere();
            return null;
        }
    }

    public String name() {
        return name;
    }
}
