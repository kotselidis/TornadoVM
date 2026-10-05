/*
 * Copyright (c) 2023, 2024 APT Group, Department of Computer Science,
 * The University of Manchester. All rights reserved.
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
package uk.ac.manchester.tornado.drivers.common.compiler.phases.memalloc;

import java.util.Optional;

import tornado.graal.compiler.graph.Node;
import tornado.graal.compiler.nodes.GraphState;
import tornado.graal.compiler.nodes.PiNode;
import tornado.graal.compiler.nodes.StructuredGraph;
import tornado.graal.compiler.phases.BasePhase;

import uk.ac.manchester.tornado.runtime.graal.nodes.NewArrayNonVirtualizableNode;
import uk.ac.manchester.tornado.runtime.graal.nodes.PanamaPrivateMemoryNode;
import uk.ac.manchester.tornado.runtime.graal.phases.TornadoHighTierContext;

public class TornadoPrivateArrayPiRemoval extends BasePhase<TornadoHighTierContext> {

    public static void removeFixed(Node n) {
        if (!n.isDeleted()) {
            Node pred = n.predecessor();
            Node suc = n.successors().first();

            n.replaceFirstSuccessor(suc, null);
            n.replaceAtPredecessor(suc);
            pred.replaceFirstSuccessor(n, suc);

            for (Node us : n.usages()) {
                n.removeUsage(us);
            }
            n.clearInputs();

            n.safeDelete();
        }
    }

    @Override
    public Optional<NotApplicable> notApplicableTo(GraphState graphState) {
        return ALWAYS_APPLICABLE;
    }

    @Override
    protected void run(StructuredGraph graph, TornadoHighTierContext context) {
        for (NewArrayNonVirtualizableNode fixedArray : graph.getNodes().filter(NewArrayNonVirtualizableNode.class)) {
            if (fixedArray.successors().filter(PanamaPrivateMemoryNode.class).isNotEmpty()) {
                for (PiNode p : fixedArray.usages().filter(PiNode.class)) {
                    p.replaceAtUsages(fixedArray);
                    p.safeDelete();
                }
                PanamaPrivateMemoryNode panamaPrivateMemoryNode = fixedArray.successors().filter(PanamaPrivateMemoryNode.class).first();
                removeFixed(panamaPrivateMemoryNode);
            }
        }
    }

}
