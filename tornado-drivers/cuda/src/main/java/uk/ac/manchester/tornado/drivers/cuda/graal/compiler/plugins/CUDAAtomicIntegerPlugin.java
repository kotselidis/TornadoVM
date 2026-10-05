/*
 * Copyright (c) 2020, APT Group, Department of Computer Science,
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
package uk.ac.manchester.tornado.drivers.cuda.graal.compiler.plugins;

import tornado.graal.compiler.nodes.graphbuilderconf.GraphBuilderContext;
import tornado.graal.compiler.nodes.graphbuilderconf.NodePlugin;

import jdk.vm.ci.meta.JavaKind;
import jdk.vm.ci.meta.ResolvedJavaType;
import uk.ac.manchester.tornado.api.internal.annotations.Vector;
import uk.ac.manchester.tornado.drivers.cuda.graal.lir.CUDAKind;
import uk.ac.manchester.tornado.drivers.cuda.graal.nodes.TornadoAtomicIntegerNode;

public class CUDAAtomicIntegerPlugin implements NodePlugin {

    @Override
    public boolean handleNewInstance(GraphBuilderContext b, ResolvedJavaType type) {
        if (type.getAnnotation(Vector.class) != null) {
            return false;
        }
        return createAtomicIntegerInstance(b, type);
    }

    private boolean createAtomicIntegerInstance(GraphBuilderContext b, ResolvedJavaType type) {
        CUDAKind kind = resolveCUDAKind(type);
        if (kind != CUDAKind.ILLEGAL) {
            if (kind == CUDAKind.INTEGER_ATOMIC_JAVA) {
                b.push(JavaKind.Object, b.append(new TornadoAtomicIntegerNode(kind)));
                return true;
            }
        }
        return false;
    }

    private CUDAKind resolveCUDAKind(ResolvedJavaType type) {
        // JDK-neutral: fromResolvedJavaType returns ILLEGAL for any type it cannot map, so no
        // HotSpot-specific instanceof guard is needed (the reflection path uses ReflectionResolvedJavaType).
        return CUDAKind.fromResolvedJavaType(type);
    }
}
