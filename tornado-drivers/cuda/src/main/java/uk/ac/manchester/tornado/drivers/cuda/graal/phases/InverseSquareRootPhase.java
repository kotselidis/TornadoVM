/*
 * Copyright (c) 2021, APT Group, Department of Computer Science,
 * The University of Manchester. All rights reserved.
 * Copyright (c) 2009-2021, Oracle and/or its affiliates. All rights reserved.
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
package uk.ac.manchester.tornado.drivers.cuda.graal.phases;

import java.util.Optional;

import tornado.graal.compiler.nodes.ConstantNode;
import tornado.graal.compiler.nodes.GraphState;
import tornado.graal.compiler.nodes.StructuredGraph;
import tornado.graal.compiler.nodes.ValueNode;
import tornado.graal.compiler.nodes.calc.FloatDivNode;
import tornado.graal.compiler.nodes.calc.SqrtNode;
import tornado.graal.compiler.phases.Phase;

import uk.ac.manchester.tornado.drivers.cuda.graal.nodes.CUDAFPUnaryIntrinsicNode;
import uk.ac.manchester.tornado.drivers.cuda.graal.nodes.RSqrtNode;

public class InverseSquareRootPhase extends Phase {
    private static final String ONE = "1.0";
    private static final String SQRT = "SQRT";

    @Override
    public Optional<NotApplicable> notApplicableTo(GraphState graphState) {
        return ALWAYS_APPLICABLE;
    }

    @Override
    protected void run(StructuredGraph graph) {
        graph.getNodes().filter(FloatDivNode.class).forEach(floatDivisionNode -> {

            // The combination is 1/sqrt(x)
            if (floatDivisionNode.getX() instanceof ConstantNode constant) {
                if ((constant.getValue().toValueString().equals(ONE))) {
                    if ((floatDivisionNode.getY() instanceof SqrtNode intrinsicNode)) {
                        ValueNode n = intrinsicNode.getValue();
                        RSqrtNode rsqrtNode = new RSqrtNode(n);
                        graph.addOrUnique(rsqrtNode);
                        intrinsicNode.removeUsage(floatDivisionNode);
                        if (intrinsicNode.hasNoUsages()) {
                            intrinsicNode.safeDelete();
                        }
                        floatDivisionNode.replaceAtUsages(rsqrtNode);
                        floatDivisionNode.safeDelete();
                    } else if ((floatDivisionNode.getY() instanceof CUDAFPUnaryIntrinsicNode CUDAFPUnaryIntrinsicNode)) {
                        if (CUDAFPUnaryIntrinsicNode.getOperation().equals(SQRT)) {
                            ValueNode n = CUDAFPUnaryIntrinsicNode.getValue();
                            RSqrtNode rsqrtNode = new RSqrtNode(n);
                            graph.addOrUnique(rsqrtNode);
                            CUDAFPUnaryIntrinsicNode.removeUsage(floatDivisionNode);
                            if (CUDAFPUnaryIntrinsicNode.hasNoUsages()) {
                                CUDAFPUnaryIntrinsicNode.safeDelete();
                            }
                            floatDivisionNode.replaceAtUsages(rsqrtNode);
                            floatDivisionNode.safeDelete();
                        }
                    }
                }
            }
        });
    }
}
