/*
 * This file is part of Tornado: A heterogeneous programming framework:
 * https://github.com/beehive-lab/tornadovm
 *
 * Copyright (c) 2013-2020, 2024 APT Group, Department of Computer Science,
 * The University of Manchester. All rights reserved.
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
package uk.ac.manchester.tornado.drivers.common.compiler.phases.loops;

import tornado.graal.compiler.nodes.EndNode;
import tornado.graal.compiler.nodes.LoopBeginNode;
import tornado.graal.compiler.nodes.LoopEndNode;
import tornado.graal.compiler.nodes.MergeNode;
import tornado.graal.compiler.nodes.PhiNode;
import tornado.graal.compiler.nodes.StructuredGraph;

/*
 * * @author James Clarkson
 */
public class LoopCanonicalizer {

    public static void canonicalizeLoop(StructuredGraph graph, LoopBeginNode loopBegin) {
        int numBackedges = loopBegin.loopEnds().count();
        final LoopEndNode[] oldLoopEndNodes = new LoopEndNode[numBackedges];
        final EndNode[] replacementEndNodes = new EndNode[numBackedges];
        final PhiNode[] oldPhiNodes = new PhiNode[loopBegin.phis().count()];
        final PhiNode[] newPhiNodes = new PhiNode[loopBegin.phis().count()];

        int index = 0;
        for (final PhiNode phi : loopBegin.phis()) {
            oldPhiNodes[index] = phi;
            index++;
        }

        index = 0;
        // assumes loopEnds.distinct()
        for (final LoopEndNode oldLoopEnd : loopBegin.loopEnds()) {
            oldLoopEndNodes[index] = oldLoopEnd;
            index++;
        }

        final MergeNode mergeNode = graph.addWithoutUnique(new MergeNode());
        final LoopEndNode newLoopEnd = graph.addWithoutUnique(new LoopEndNode(loopBegin));
        mergeNode.setNext(newLoopEnd);

        for (index = 0; index < numBackedges; index++) {
            replacementEndNodes[index] = graph.addWithoutUnique(new EndNode());
            mergeNode.addForwardEnd(replacementEndNodes[index]);
        }

        index = 0;
        for (final PhiNode oldPhi : oldPhiNodes) {
            PhiNode newPhi = (PhiNode) oldPhi.copyWithInputs(true);
            newPhi.clearInputs();
            newPhi.setMerge(mergeNode);

            for (int i = 0; i < numBackedges; i++) {
                newPhi.initializeValueAt(i, oldPhi.valueAt(oldLoopEndNodes[i]));
            }

            newPhiNodes[index] = newPhi;
            index++;
        }

        for (index = 0; index < numBackedges; index++) {
            loopBegin.removeEnd(oldLoopEndNodes[index]);
            oldLoopEndNodes[index].replaceAndDelete(replacementEndNodes[index]);
        }

        index = 0;
        for (final PhiNode oldPhi : oldPhiNodes) {
            oldPhi.initializeValueAt(1, newPhiNodes[index]);
            index++;
        }

    }
}
