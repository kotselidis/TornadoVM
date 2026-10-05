/*
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
package uk.ac.manchester.tornado.drivers.opencl.graal.nodes.vector;

import static uk.ac.manchester.tornado.api.exceptions.TornadoInternalError.shouldNotReachHere;

import tornado.graal.compiler.graph.NodeClass;
import tornado.graal.compiler.nodeinfo.InputType;
import tornado.graal.compiler.nodeinfo.NodeInfo;
import tornado.graal.compiler.nodes.ConstantNode;
import tornado.graal.compiler.nodes.FixedWithNextNode;
import tornado.graal.compiler.nodes.ValueNode;

import uk.ac.manchester.tornado.drivers.opencl.graal.OCLStampFactory;
import uk.ac.manchester.tornado.drivers.opencl.graal.lir.OCLKind;

/**
 * The {@code StoreIndexedNode} represents a write to an array element.
 */
@Deprecated()
@NodeInfo(nameTemplate = "Load .s{p#lane}")
public final class VectorLoadElementProxyNode extends FixedWithNextNode {

    public static final NodeClass<VectorLoadElementProxyNode> TYPE = NodeClass.create(VectorLoadElementProxyNode.class);

    @OptionalInput(InputType.Association)
    ValueNode origin;
    @OptionalInput(InputType.Association)
    ValueNode laneOrigin;

    private final OCLKind kind;

    protected VectorLoadElementProxyNode(NodeClass<? extends VectorLoadElementProxyNode> c, OCLKind kind, ValueNode origin, ValueNode lane) {
        super(c, OCLStampFactory.getStampFor(kind));
        this.kind = kind;
        this.origin = origin;
        this.laneOrigin = lane;
    }

    public VectorLoadElementNode tryResolve() {
        VectorLoadElementNode loadNode = null;
        if (canResolve()) {
            /*
             * If we can resolve this node properly, this operation should be
             * applied to the vector node and this node should be discarded.
             */
            VectorValueNode vector = null;
            if (origin instanceof VectorValueNode) {
                vector = (VectorValueNode) origin;
            } else {
                shouldNotReachHere();
            }

            loadNode = new VectorLoadElementNode(kind, vector, laneOrigin);
            clearInputs();
        }

        return loadNode;
    }

    @Override
    public boolean inferStamp() {
        return true;
    }

    public OCLKind getOCLKind() {
        return kind;
    }

    public boolean canResolve() {
        return (isVectorValueNode() && laneOrigin != null && laneOrigin instanceof ConstantNode);
    }

    private boolean isVectorValueNode() {
        return (origin != null) && (origin instanceof VectorValueNode);
    }
}
