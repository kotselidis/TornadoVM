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

import java.util.Optional;

import tornado.graal.compiler.nodes.GraphState;
import tornado.graal.compiler.nodes.StructuredGraph;
import tornado.graal.compiler.nodes.ValueNode;
import tornado.graal.compiler.nodes.calc.AddNode;
import tornado.graal.compiler.nodes.calc.MulNode;
import tornado.graal.compiler.phases.Phase;

import jdk.vm.ci.meta.JavaKind;
import uk.ac.manchester.tornado.drivers.metal.graal.nodes.MetalFMANode;

public class MetalFMAPhase extends Phase {

    /**
     * Instrinsics in Metal:
     * https://www.khronos.org/registry/Metal/sdk/1.0/docs/man/xhtml/fma.html
     * 
     * 
     * From the Metal documentation:
     * 
     * The generic type name gentype is used to indicate that the function can take
     * float, float2, float4, float8, or float16 as the type for the arguments. For
     * any specific use of these function, the actual type has to be the same for
     * all arguments and the return type.
     *
     * If extended with cl_khr_fp64, generic type name gentype may indicate double
     * and double{2|4|8|16} as arguments and return values. If extended with
     * cl_khr_fp16, generic type name gentype may indicate half and half{2|4|8|16}
     * as arguments and return values.
     * 
     * @param x
     *            valueNode
     * @return returns true if ValueNode is either Float of Double
     */
    private boolean isValidType(ValueNode x) {
        return (x.getStackKind() == JavaKind.Float || x.getStackKind() == JavaKind.Double);
    }

    @Override
    public Optional<NotApplicable> notApplicableTo(GraphState graphState) {
        return ALWAYS_APPLICABLE;
    }

    @Override
    protected void run(StructuredGraph graph) {

        graph.getNodes().filter(AddNode.class).forEach(addNode -> {
            MulNode mulNode = null;
            if (addNode.getX() instanceof MulNode) {
                mulNode = (MulNode) addNode.getX();
            } else if (addNode.getY() instanceof MulNode) {
                mulNode = (MulNode) addNode.getY();
            }
            if (mulNode != null) {
                ValueNode x = mulNode.getX();
                ValueNode y = mulNode.getY();
                if (isValidType(x) && isValidType(y)) {
                    MulNode finalMulNode = mulNode;
                    ValueNode z = (ValueNode) addNode.inputs().filter(node -> !node.equals(finalMulNode)).first();
                    MetalFMANode oclFMA = new MetalFMANode(x, y, z);
                    graph.addOrUnique(oclFMA);
                    mulNode.removeUsage(addNode);
                    if (mulNode.hasNoUsages()) {
                        mulNode.safeDelete();
                    }
                    addNode.replaceAtUsages(oclFMA);
                    addNode.safeDelete();
                }
            }
        });

    }
}
