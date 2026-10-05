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
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 */
package uk.ac.manchester.tornado.drivers.opencl.graal.phases;

import java.util.Optional;

import tornado.graal.compiler.graph.iterators.NodeIterable;
import tornado.graal.compiler.nodes.ConstantNode;
import tornado.graal.compiler.nodes.FixedNode;
import tornado.graal.compiler.nodes.GraphState;
import tornado.graal.compiler.nodes.ParameterNode;
import tornado.graal.compiler.nodes.StartNode;
import tornado.graal.compiler.nodes.StructuredGraph;
import tornado.graal.compiler.phases.Phase;

import uk.ac.manchester.tornado.drivers.opencl.graal.lir.OCLKind;
import uk.ac.manchester.tornado.drivers.opencl.graal.nodes.IncAtomicNode;
import uk.ac.manchester.tornado.drivers.opencl.graal.nodes.NodeAtomic;
import uk.ac.manchester.tornado.drivers.opencl.graal.nodes.TornadoAtomicIntegerNode;

/**
 * This phase scans the IR and checks whether there is a
 * {@link TornadoAtomicIntegerNode} associated for each {@link IncAtomicNode}.
 *
 * <p>
 * If it is not the case, this would mean that the atomic is passed as a
 * parameter to the lambda expression. Therefore, this phase introduces the
 * {@link TornadoAtomicIntegerNode} at the beginning of the Control-Flow-Graph.
 * </p>
 */
public class TornadoAtomicsParametersPhase extends Phase {
    @Override
    public Optional<NotApplicable> notApplicableTo(GraphState graphState) {
        return ALWAYS_APPLICABLE;
    }

    @Override
    protected void run(StructuredGraph graph) {
        NodeIterable<NodeAtomic> filter = graph.getNodes().filter(NodeAtomic.class);

        if (!filter.isEmpty()) {
            for (NodeAtomic atomic : filter) {
                if (atomic.getAtomicNode() instanceof ParameterNode) {

                    ParameterNode atomicArgument = (ParameterNode) atomic.getAtomicNode();
                    int indexNode = atomicArgument.index();

                    TornadoAtomicIntegerNode newNode = new TornadoAtomicIntegerNode(OCLKind.INTEGER_ATOMIC_JAVA);
                    graph.addOrUnique(newNode);
                    newNode.assignIndexFromParameter(indexNode);

                    final ConstantNode index = graph.addOrUnique(ConstantNode.forInt(0));
                    newNode.setInitialValue(index);

                    // Add the new control flow node
                    StartNode startNode = graph.start();
                    FixedNode first = (FixedNode) startNode.successors().first();
                    startNode.setNext(newNode);
                    newNode.setNext(first);

                    // Replace usages for this new node
                    ParameterNode parameter = (ParameterNode) atomic.getAtomicNode();
                    newNode.replaceAtMatchingUsages(atomic, node -> !node.equals(atomic));
                    parameter.replaceAtMatchingUsages(newNode, node -> node.equals(atomic));

                    assert graph.verify();
                }
            }
        }

    }

}
