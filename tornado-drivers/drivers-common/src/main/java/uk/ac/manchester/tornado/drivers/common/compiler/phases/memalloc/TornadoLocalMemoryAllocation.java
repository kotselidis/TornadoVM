/*
 * Copyright (c) 2018-2024 APT Group, Department of Computer Science,
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
 * Authors: Michalis Papadimitriou
 *
 */
package uk.ac.manchester.tornado.drivers.common.compiler.phases.memalloc;

import java.util.Optional;

import tornado.graal.compiler.graph.Node;
import tornado.graal.compiler.graph.iterators.NodeIterable;
import tornado.graal.compiler.nodes.ConstantNode;
import tornado.graal.compiler.nodes.GraphState;
import tornado.graal.compiler.nodes.StructuredGraph;
import tornado.graal.compiler.phases.BasePhase;

import uk.ac.manchester.tornado.runtime.graal.nodes.interfaces.MarkLocalArray;
import uk.ac.manchester.tornado.runtime.graal.phases.TornadoHighTierContext;

/**
 * 
 * It filters the nodes for the length of the FixedArray node for the local
 * memory and calculates an optimal local memory size based on driver info.
 * Then, it replaces the length with the optimal size.
 *
 */
public class TornadoLocalMemoryAllocation extends BasePhase<TornadoHighTierContext> {
    @Override
    public Optional<NotApplicable> notApplicableTo(GraphState graphState) {
        return ALWAYS_APPLICABLE;
    }

    @Override
    protected void run(StructuredGraph graph, TornadoHighTierContext context) {

        if (!context.hasMeta()) {
            return;
        }

        if ((context.getMeta().getDomain() != null)) {
            if ((context.getMeta().getDomain().getDepth() != 0)) {
                NodeIterable<Node> sumNodes = graph.getNodes();

                for (Node n : sumNodes) {
                    if (n instanceof MarkLocalArray) {
                        ConstantNode newLengthNode = ConstantNode.forInt(calculateLocalMemAllocSize(context), graph);
                        if (newLengthNode != n.inputs().first()) {
                            n.inputs().first().replaceAndDelete(newLengthNode);
                        }
                    }
                }
            }
        }
    }

    private int calculateLocalMemAllocSize(TornadoHighTierContext context) {
        int maxBlockSize = (int) context.getDeviceMapping().getPhysicalDevice().getDeviceMaxWorkItemSizes()[0];

        if (context.getDeviceMapping().getPhysicalDevice().getDeviceMaxWorkItemSizes()[0] == context.getMeta().getDomain().get(0).cardinality()) {
            maxBlockSize /= 4;
        }

        int value = Math.min(Math.max(maxBlockSize, context.getMeta().getOpenCLGpuBlock2DX()), context.getMeta().getDomain().get(0).cardinality());

        if (value == 0) {
            return 0;
        } else {
            while (context.getMeta().getDomain().get(0).cardinality() % value != 0) {
                value--;
            }
            return value;
        }
    }
}
