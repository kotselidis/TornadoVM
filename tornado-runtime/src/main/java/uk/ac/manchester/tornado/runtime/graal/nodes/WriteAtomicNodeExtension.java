/*
 * Copyright (c) 2023, APT Group, Department of Computer Science,
 * The University of Manchester.
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
 */
package uk.ac.manchester.tornado.runtime.graal.nodes;

import tornado.graal.compiler.core.common.type.StampFactory;
import tornado.graal.compiler.graph.NodeClass;
import tornado.graal.compiler.nodeinfo.NodeInfo;
import tornado.graal.compiler.nodes.FrameState;
import tornado.graal.compiler.nodes.ValueNode;
import tornado.graal.compiler.nodes.calc.FloatingNode;

import static tornado.graal.compiler.nodeinfo.InputType.State;

@NodeInfo(nameTemplate = "AtomicWriteExtension")
public class WriteAtomicNodeExtension extends FloatingNode {

    public static final NodeClass<WriteAtomicNodeExtension> TYPE = NodeClass.create(WriteAtomicNodeExtension.class);

    @OptionalInput(State)
    FrameState stateAfter;
    @OptionalInput
    ValueNode extraOperation;
    @OptionalInput
    ValueNode startNode;

    public WriteAtomicNodeExtension(ValueNode startNode) {
        super(TYPE, StampFactory.forVoid());
        this.startNode = startNode;
    }

    public FrameState getStateAfter() {
        return stateAfter;
    }

    public void setStateAfter(FrameState stateAfter) {
        this.stateAfter = stateAfter;
    }

    public ValueNode getExtraOperation() {
        return extraOperation;
    }

    public ValueNode getStartNode() {
        return startNode;
    }

    public void setExtraOperation(ValueNode extraOperation) {
        this.extraOperation = extraOperation;
    }

}
