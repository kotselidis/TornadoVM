/*
 * Copyright (c) 2020, APT Group, Department of Computer Science,
 * School of Engineering, The University of Manchester. All rights reserved.
 * Copyright (c) 2018, 2020, APT Group, Department of Computer Science,
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

import tornado.graal.compiler.nodes.ValueNode;
import tornado.graal.compiler.nodes.extended.GuardingNode;
import tornado.graal.compiler.nodes.graphbuilderconf.GraphBuilderContext;
import tornado.graal.compiler.nodes.graphbuilderconf.NodePlugin;
import tornado.graal.compiler.nodes.java.StoreIndexedNode;

import jdk.vm.ci.meta.JavaKind;
import jdk.vm.ci.meta.ResolvedJavaType;
import uk.ac.manchester.tornado.api.internal.annotations.Vector;
import uk.ac.manchester.tornado.drivers.cuda.graal.lir.CUDAKind;
import uk.ac.manchester.tornado.drivers.cuda.graal.nodes.LocalArrayNode;
import uk.ac.manchester.tornado.drivers.cuda.graal.nodes.vector.LoadIndexedVectorNode;
import uk.ac.manchester.tornado.drivers.cuda.graal.nodes.vector.VectorValueNode;

public class CUDAVectorNodePlugin implements NodePlugin {

    @Override
    public boolean handleNewInstance(GraphBuilderContext b, ResolvedJavaType type) {
        if (type.getAnnotation(Vector.class) != null) {
            return createVectorInstance(b, type);
        }
        return false;
    }

    /**
     * Indexed loads on a packed half2 local array ({@link LocalArrayNode} of kind
     * {@link CUDAKind#HALF2}) produce a packed 32-bit {@code __half2} element, so they
     * are parsed as vector loads instead of object array accesses.
     */
    @Override
    public boolean handleLoadIndexed(GraphBuilderContext b, ValueNode array, ValueNode index, GuardingNode boundsCheck, JavaKind elementKind) {
        if (array instanceof LocalArrayNode localArrayNode && localArrayNode.getCUDAKind() == CUDAKind.HALF2) {
            LoadIndexedVectorNode indexedLoad = new LoadIndexedVectorNode(CUDAKind.HALF2, array, index, JavaKind.Short);
            b.push(JavaKind.Object, b.append(indexedLoad));
            return true;
        }
        return false;
    }

    /**
     * Indexed stores on a packed half2 local array write the vector value as a single
     * 32-bit {@code __half2} element.
     */
    @Override
    public boolean handleStoreIndexed(GraphBuilderContext b, ValueNode array, ValueNode index, GuardingNode boundsCheck, GuardingNode storeCheck, JavaKind elementKind, ValueNode value) {
        if (array instanceof LocalArrayNode localArrayNode && localArrayNode.getCUDAKind() == CUDAKind.HALF2) {
            StoreIndexedNode indexedStore = new StoreIndexedNode(array, index, null, null, JavaKind.Short, value);
            b.add(b.append(indexedStore));
            return true;
        }
        return false;
    }

    private boolean createVectorInstance(GraphBuilderContext b, ResolvedJavaType type) {
        CUDAKind vectorKind = resolveCUDAKind(type);
        if (vectorKind != CUDAKind.ILLEGAL) {
            if (vectorKind.isVector()) {
                b.push(JavaKind.Object, b.append(new VectorValueNode(vectorKind)));
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
