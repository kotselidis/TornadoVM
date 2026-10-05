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
package uk.ac.manchester.tornado.drivers.cuda.graal.nodes.vector;

import tornado.graal.compiler.graph.Node;
import tornado.graal.compiler.graph.NodeClass;
import tornado.graal.compiler.nodeinfo.NodeInfo;
import tornado.graal.compiler.nodes.NodeView;
import tornado.graal.compiler.nodes.ValueNode;
import tornado.graal.compiler.nodes.java.AccessIndexedNode;
import tornado.graal.compiler.nodes.spi.CanonicalizerTool;

import jdk.vm.ci.meta.JavaKind;
import uk.ac.manchester.tornado.drivers.cuda.graal.CUDAStamp;
import uk.ac.manchester.tornado.drivers.cuda.graal.CUDAStampFactory;
import uk.ac.manchester.tornado.drivers.cuda.graal.lir.CUDAKind;

/**
 * The {@code VectorLoadNode} represents a vector-read from a set of contiguous
 * elements of an array.
 */
@NodeInfo(nameTemplate = "VectorLoad")
public class VectorLoadNode extends AccessIndexedNode {

    public static final NodeClass<VectorLoadNode> TYPE = NodeClass.create(VectorLoadNode.class);

    private final CUDAKind kind;

    /**
     * Creates a new LoadIndexedNode.
     *
     * @param kind
     *     the element type
     * @param array
     *     the instruction producing the array
     * @param index
     *     the instruction producing the index
     */
    public VectorLoadNode(CUDAKind kind, ValueNode array, ValueNode index) {
        super(TYPE, CUDAStampFactory.getStampFor(kind), array, index, null, JavaKind.Illegal);
        this.kind = kind;
    }

    public Node canonical(CanonicalizerTool tool) {
        return this;
    }

    public int length() {
        return kind.getVectorLength();
    }

    public CUDAKind elementType() {
        return kind.getElementKind();
    }

    public CUDAKind vectorKind() {
        return kind;
    }

    @Override
    public JavaKind elementKind() {
        return ((CUDAStamp) stamp(NodeView.DEFAULT)).getCUDAKind().getElementKind().asJavaKind();
    }
}
