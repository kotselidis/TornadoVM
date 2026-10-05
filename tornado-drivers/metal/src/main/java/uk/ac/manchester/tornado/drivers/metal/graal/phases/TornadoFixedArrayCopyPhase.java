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
package uk.ac.manchester.tornado.drivers.metal.graal.phases;

import jdk.vm.ci.meta.ResolvedJavaType;
import tornado.graal.compiler.graph.Node;
import tornado.graal.compiler.nodes.GraphState;
import tornado.graal.compiler.nodes.StructuredGraph;
import tornado.graal.compiler.nodes.ValuePhiNode;
import tornado.graal.compiler.nodes.memory.address.OffsetAddressNode;
import tornado.graal.compiler.phases.Phase;

import uk.ac.manchester.tornado.api.exceptions.TornadoCompilationException;
import uk.ac.manchester.tornado.drivers.metal.graal.MetalArchitecture;
import uk.ac.manchester.tornado.drivers.metal.graal.nodes.FixedArrayCopyNode;
import uk.ac.manchester.tornado.drivers.metal.graal.nodes.FixedArrayNode;

import java.util.Optional;

/**
 * This phase examines if a copy takes place between two arrays in private memory based on
 * an if condition and, if so, inserts a {@link FixedArrayCopyNode} to generate an update in the references.
 */
public class TornadoFixedArrayCopyPhase extends Phase {

    @Override
    public Optional<NotApplicable> notApplicableTo(GraphState graphState) {
        return ALWAYS_APPLICABLE;
    }

    protected void run(StructuredGraph graph) {
        for (ValuePhiNode phiNode : graph.getNodes().filter(ValuePhiNode.class)) {
            if (isFixedArrayCopied(phiNode)) {
                FixedArrayNode fixedArrayNode = phiNode.values().filter(FixedArrayNode.class).first();
                ResolvedJavaType resolvedJavaType = fixedArrayNode.getElementType();
                MetalArchitecture.MetalMemoryBase oclMemoryBase = fixedArrayNode.getMemoryRegister();
                OffsetAddressNode offsetAddressNode = phiNode.usages().filter(OffsetAddressNode.class).first();
                FixedArrayCopyNode fixedArrayCopyNode = new FixedArrayCopyNode(phiNode, resolvedJavaType, oclMemoryBase);
                graph.addWithoutUnique(fixedArrayCopyNode);
                offsetAddressNode.replaceFirstInput(phiNode, fixedArrayCopyNode);
                // finally, since we know that the data accessed is a fixed array, fix the offset
                ValuePhiNode privateIndex = getPrivateArrayIndex(offsetAddressNode.getOffset());
                if (privateIndex == null) {
                    throw new TornadoCompilationException("Index of FixedArrayNode is null.");
                }
                offsetAddressNode.setOffset(privateIndex);
            }
        }
    }

    private static boolean isFixedArrayCopied(ValuePhiNode phiNode) {
        return phiNode.usages().filter(OffsetAddressNode.class).isNotEmpty() && phiNode.values().filter(FixedArrayNode.class).isNotEmpty();
    }

    private static ValuePhiNode getPrivateArrayIndex(Node node) {
        // identify the index
        for (Node input : node.inputs()) {
            if (input instanceof ValuePhiNode phiNode) {
                return phiNode;
            } else {
                return getPrivateArrayIndex(input);
            }
        }
        return null;
    }

}
