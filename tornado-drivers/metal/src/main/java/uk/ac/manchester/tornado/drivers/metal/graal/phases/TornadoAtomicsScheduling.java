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

import static uk.ac.manchester.tornado.runtime.TornadoCoreRuntime.getDebugContext;

import java.util.Optional;

import tornado.graal.compiler.debug.DebugContext;
import tornado.graal.compiler.graph.Node;
import tornado.graal.compiler.graph.iterators.NodeIterable;
import tornado.graal.compiler.nodes.FixedNode;
import tornado.graal.compiler.nodes.FixedWithNextNode;
import tornado.graal.compiler.nodes.GraphState;
import tornado.graal.compiler.nodes.StartNode;
import tornado.graal.compiler.nodes.StructuredGraph;
import tornado.graal.compiler.nodes.memory.WriteNode;
import tornado.graal.compiler.phases.Phase;

import uk.ac.manchester.tornado.drivers.metal.graal.nodes.DecAtomicNode;
import uk.ac.manchester.tornado.drivers.metal.graal.nodes.IncAtomicNode;
import uk.ac.manchester.tornado.drivers.metal.graal.nodes.NodeAtomic;
import uk.ac.manchester.tornado.drivers.metal.graal.nodes.TornadoAtomicIntegerNode;

/**
 * Compiler phase that relocates the TornadoAtomicIntegerNode(s) from the first
 * basic block (B0) to the basic block that requests the atomic (usually within
 * the same basic block that READ/WRITES the atomic value). This phase is needed
 * since the integration with the Graal 22.3.1 JIT compiler.
 *
 * <p>
 * This phase is expected to be invoked from the LIR of the
 * compilation/optimization pipeline.
 * </p>
 *
 * @since v0.15.1
 */
public class TornadoAtomicsScheduling extends Phase {
    @Override
    public Optional<NotApplicable> notApplicableTo(GraphState graphState) {
        return ALWAYS_APPLICABLE;
    }

    private Node getAtomicUsage(NodeAtomic nodeAtomic) {
        Node atomicUsage = nodeAtomic.usages().first();
        while (!(atomicUsage instanceof WriteNode)) {
            atomicUsage = atomicUsage.usages().first();
        }
        return atomicUsage;
    }

    private void fixStartWithNext(StructuredGraph graph, TornadoAtomicIntegerNode atomic) {
        StartNode startNode = graph.start();
        Node first = atomic.successors().first();
        while (first instanceof TornadoAtomicIntegerNode) {
            first = first.successors().first();
        }

        if (first != null && !(first instanceof StartNode)) {
            first.replaceAtPredecessor(startNode);
            startNode.setNext((FixedNode) first);
        }
    }

    private void moveAtomicNodeToWriteBasicBloc(Node atomicUsage, TornadoAtomicIntegerNode atomic) {
        WriteNode writeNode = (WriteNode) atomicUsage;

        FixedWithNextNode pre = (FixedWithNextNode) writeNode.predecessor();
        if (atomic.predecessor() != null) {
            atomic.replaceAtPredecessor(null);
        }
        writeNode.replaceAtPredecessor(atomic);
        pre.setNext(atomic);
        atomic.setNext(writeNode);
    }

    @Override
    protected void run(StructuredGraph graph) {
        NodeIterable<TornadoAtomicIntegerNode> filter = graph.getNodes().filter(TornadoAtomicIntegerNode.class);

        if (!filter.isEmpty()) {
            for (TornadoAtomicIntegerNode atomic : filter) {
                NodeIterable<Node> usages = atomic.usages();
                int irCount = 1;
                for (Node usage : usages) {
                    if (usage instanceof IncAtomicNode || usage instanceof DecAtomicNode) {
                        Node atomicUsage = getAtomicUsage((NodeAtomic) usage);

                        // Fix the link between the START with the next node that follows all atomic
                        // nodes.
                        fixStartWithNext(graph, atomic);

                        // Dump IR for Debugging
                        getDebugContext().dump(DebugContext.BASIC_LEVEL, graph, "Atomics #" + irCount++);

                        // Move the TornadoAtomicIntegerNode to the basic block that performs the
                        // writes.
                        // This will also move all data-flow nodes (FloatingNodes) associated with it.
                        moveAtomicNodeToWriteBasicBloc(atomicUsage, atomic);
                    }
                }
            }
        }
    }
}
