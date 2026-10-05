/*
 * Copyright (c) 2024, APT Group, Department of Computer Science,
 * School of Engineering, The University of Manchester. All rights reserved.
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
package uk.ac.manchester.tornado.runtime.graal.phases;

import java.util.ArrayList;
import java.util.Optional;

import tornado.graal.compiler.graph.Node;
import tornado.graal.compiler.nodes.FixedGuardNode;
import tornado.graal.compiler.nodes.GraphState;
import tornado.graal.compiler.nodes.PiNode;
import tornado.graal.compiler.nodes.StructuredGraph;
import tornado.graal.compiler.nodes.ValueNode;
import tornado.graal.compiler.nodes.calc.IsNullNode;
import tornado.graal.compiler.phases.BasePhase;

import uk.ac.manchester.tornado.runtime.graal.nodes.HalfFloatPlaceholder;

public class TornadoHalfFloatFixedGuardElimination extends BasePhase<TornadoSketchTierContext> {

    private static void deleteFixed(Node node) {
        if (!node.isDeleted()) {
            Node predecessor = node.predecessor();
            Node successor = node.successors().first();

            node.replaceFirstSuccessor(successor, null);
            node.replaceAtPredecessor(successor);
            predecessor.replaceFirstSuccessor(node, successor);

            for (Node us : node.usages()) {
                node.removeUsage(us);
            }
            node.clearInputs();
            node.safeDelete();
        }
    }

    @Override
    public Optional<NotApplicable> notApplicableTo(GraphState graphState) {
        return ALWAYS_APPLICABLE;
    }

    protected void run(StructuredGraph graph, TornadoSketchTierContext context) {
        ArrayList<ValueNode> nodesToBeDeleted = new ArrayList<ValueNode>();
        for (HalfFloatPlaceholder placeholderNode : graph.getNodes().filter(HalfFloatPlaceholder.class)) {
            if (placeholderNode.getInput() instanceof PiNode placeholderInput) {
                ValueNode halfFloatValue = placeholderInput.object();
                FixedGuardNode placeholderGuard = (FixedGuardNode) placeholderInput.getGuard();
                if (placeholderGuard.inputs().filter(IsNullNode.class).isNotEmpty()) {
                    IsNullNode isNullNode = placeholderGuard.inputs().filter(IsNullNode.class).first();
                    nodesToBeDeleted.add(isNullNode);
                }
                deleteFixed(placeholderGuard);
                placeholderNode.setInput(halfFloatValue);
                nodesToBeDeleted.add(placeholderInput);
            }
        }

        for (ValueNode node : nodesToBeDeleted) {
            node.safeDelete();
        }
    }
}
