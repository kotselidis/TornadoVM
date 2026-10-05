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
package uk.ac.manchester.tornado.drivers.metal.graal.nodes.vector;

import tornado.graal.compiler.graph.Node;
import tornado.graal.compiler.graph.NodeClass;
import tornado.graal.compiler.nodeinfo.InputType;
import tornado.graal.compiler.nodeinfo.NodeInfo;
import tornado.graal.compiler.nodes.ConstantNode;
import tornado.graal.compiler.nodes.FixedWithNextNode;
import tornado.graal.compiler.nodes.ValueNode;
import tornado.graal.compiler.nodes.spi.Canonicalizable;
import tornado.graal.compiler.nodes.spi.CanonicalizerTool;

import uk.ac.manchester.tornado.drivers.metal.graal.MetalStampFactory;
import uk.ac.manchester.tornado.drivers.metal.graal.lir.MetalKind;

/**
 * The {@code StoreIndexedNode} represents a write to an array element.
 */
@NodeInfo(nameTemplate = "Store .s{p#lane}")
public final class VectorStoreElementProxyNode extends FixedWithNextNode implements Canonicalizable {

    public static final NodeClass<VectorStoreElementProxyNode> TYPE = NodeClass.create(VectorStoreElementProxyNode.class);

    @Input
    ValueNode value;

    @OptionalInput(InputType.Association)
    ValueNode origin;
    @OptionalInput(InputType.Association)
    ValueNode laneOrigin;

    public ValueNode value() {
        return value;
    }

    private VectorStoreElementProxyNode(NodeClass<? extends VectorStoreElementProxyNode> c, MetalKind kind, ValueNode origin, ValueNode lane) {
        super(c, MetalStampFactory.getStampFor(kind));
        this.origin = origin;
        this.laneOrigin = lane;

    }

    public boolean tryResolve() {
        if (canResolve()) {
            /*
             * If we can resolve this node properly, this operation should be applied to the
             * vector node and this node should be discarded.
             */
            final VectorValueNode vector = (VectorValueNode) origin;
            vector.setElement(((ConstantNode) laneOrigin).asJavaConstant().asInt(), value);
            clearInputs();
            return true;
        } else {
            return false;
        }

    }

    public VectorStoreElementProxyNode(MetalKind kind, ValueNode origin, ValueNode lane, ValueNode value) {
        this(TYPE, kind, origin, lane);
        this.value = value;
    }

    @Override
    public boolean inferStamp() {
        return true;
    }

    public boolean canResolve() {
        return ((origin != null && laneOrigin != null) && origin instanceof VectorValueNode && laneOrigin instanceof ConstantNode && ((VectorValueNode) origin).getMetalKind()
                .getVectorLength() > laneOrigin.asJavaConstant().asInt());
    }

    public ValueNode getOrigin() {
        return origin;
    }

    public void setOrigin(ValueNode value) {
        origin = value;
    }

    public int getLane() {
        // System.out.printf("vector store proxy: this=%s,
        // origin=%s\n",this,laneOrigin);
        return ((ConstantNode) laneOrigin).asJavaConstant().asInt();
    }

    @Override
    public Node canonical(CanonicalizerTool ct) {
        if (tryResolve()) {
            return null;
        } else {
            return this;
        }
    }

}
