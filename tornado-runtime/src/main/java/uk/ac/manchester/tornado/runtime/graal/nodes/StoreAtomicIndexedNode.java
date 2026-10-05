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
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Authors: Juan Fumero
 *
 */
package uk.ac.manchester.tornado.runtime.graal.nodes;

import tornado.graal.compiler.core.common.type.StampFactory;
import tornado.graal.compiler.graph.NodeClass;
import tornado.graal.compiler.nodeinfo.NodeInfo;
import tornado.graal.compiler.nodes.FrameState;
import tornado.graal.compiler.nodes.StateSplit;
import tornado.graal.compiler.nodes.ValueNode;
import tornado.graal.compiler.nodes.extended.GuardingNode;
import tornado.graal.compiler.nodes.java.AccessIndexedNode;
import tornado.graal.compiler.nodes.spi.Lowerable;
import tornado.graal.compiler.nodes.spi.Virtualizable;
import tornado.graal.compiler.nodes.spi.VirtualizerTool;

import jdk.vm.ci.meta.JavaKind;

@NodeInfo(nameTemplate = "AtomicIndexedStore")
public final class StoreAtomicIndexedNode extends AccessIndexedNode implements StateSplit, Lowerable, Virtualizable {

    public static final NodeClass<StoreAtomicIndexedNode> TYPE = NodeClass.create(StoreAtomicIndexedNode.class);

    //@formatter:off
    @Input ValueNode value;
    @Input ValueNode accumulator;
    @Input ValueNode inputArray;
    @Input StoreAtomicIndexedNodeExtension storeAtomicExtraNode;
    //@formatter:on

    @Override
    public FrameState stateAfter() {
        return storeAtomicExtraNode.getStateAfter();
    }

    @Override
    public void setStateAfter(FrameState x) {
        assert x == null || x.isAlive() : "frame state must be in a graph";
        updateUsages(storeAtomicExtraNode.getStateAfter(), x);
        storeAtomicExtraNode.setStateAfter(x);
    }

    @Override
    public boolean hasSideEffect() {
        return true;
    }

    public ValueNode value() {
        return value;
    }

    public StoreAtomicIndexedNode(ValueNode outputArray, ValueNode index, JavaKind elementKind, GuardingNode boundsCheck, ValueNode value, ValueNode accumulator, ValueNode inputArray,
            StoreAtomicIndexedNodeExtension extension) {
        super(TYPE, StampFactory.forVoid(), outputArray, index, boundsCheck, elementKind);
        this.value = value;
        this.accumulator = accumulator;
        this.inputArray = inputArray;
        this.storeAtomicExtraNode = extension;
    }

    @Override
    public void virtualize(VirtualizerTool tool) {
        throw new RuntimeException("StoreAtomic Virtual Node not supported yet");
    }

    public FrameState getState() {
        return storeAtomicExtraNode.getStateAfter();
    }

    public ValueNode getAccumulator() {
        return accumulator;
    }

    public ValueNode getStartNode() {
        return storeAtomicExtraNode.getStartNode();
    }

    public ValueNode getInputArray() {
        return inputArray;
    }

    public void setOptionalOperation(ValueNode node) {
        storeAtomicExtraNode.setExtraOperation(node);
    }

    public ValueNode getExtraOperation() {
        return storeAtomicExtraNode.getExtraOperation();
    }

    public StoreAtomicIndexedNodeExtension getStoreAtomicExtraNode() {
        return storeAtomicExtraNode;
    }
}
