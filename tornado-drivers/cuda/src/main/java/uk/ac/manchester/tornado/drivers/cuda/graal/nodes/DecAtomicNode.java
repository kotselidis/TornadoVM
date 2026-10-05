/*
 * Copyright (c) 2020, 2025, APT Group, Department of Computer Science,
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
 */
package uk.ac.manchester.tornado.drivers.cuda.graal.nodes;

import tornado.graal.compiler.core.common.type.StampFactory;
import tornado.graal.compiler.graph.NodeClass;
import tornado.graal.compiler.lir.Variable;
import tornado.graal.compiler.lir.gen.LIRGeneratorTool;
import tornado.graal.compiler.nodeinfo.NodeInfo;
import tornado.graal.compiler.nodes.ValueNode;
import tornado.graal.compiler.nodes.spi.LIRLowerable;
import tornado.graal.compiler.nodes.spi.NodeLIRBuilderTool;

import jdk.vm.ci.meta.JavaKind;
import uk.ac.manchester.tornado.drivers.cuda.graal.asm.CUDAAssembler;
import uk.ac.manchester.tornado.drivers.cuda.graal.lir.CUDALIRStmt;
import uk.ac.manchester.tornado.drivers.cuda.graal.lir.CUDAUnary;

@NodeInfo(shortName = "DECREMENT_ATOMIC")
public class DecAtomicNode extends NodeAtomic implements LIRLowerable {

    public static final NodeClass<DecAtomicNode> TYPE = NodeClass.create(DecAtomicNode.class);

    private static boolean ATOMIC_2_0 = false;

    @Input
    ValueNode atomicNode;
    CUDAUnary.AtomicOperator atomicOperator;

    public DecAtomicNode(ValueNode atomicValue, CUDAUnary.AtomicOperator atomicOperator) {
        super(TYPE, StampFactory.forKind(JavaKind.Int));
        this.atomicNode = atomicValue;
        this.atomicOperator = atomicOperator;
    }

    private void generateExpressionForOpenCL2_0(NodeLIRBuilderTool generator) {
        LIRGeneratorTool tool = generator.getLIRGeneratorTool();
        Variable result = tool.newVariable(tool.getLIRKind(stamp));

        CUDAUnary.IntrinsicAtomicFetch intrinsicAtomicFetch = new CUDAUnary.IntrinsicAtomicFetch( //
                CUDAAssembler.CUDAUnaryIntrinsic.ATOMIC_FETCH_SUB_EXPLICIT, //
                tool.getLIRKind(stamp), //
                generator.operand(atomicNode));

        CUDALIRStmt.AssignStmt assignStmt = new CUDALIRStmt.AssignStmt(result, intrinsicAtomicFetch);
        tool.append(assignStmt);
        generator.setResult(this, result);
    }

    private void generateExpressionForOpenCL1_0(NodeLIRBuilderTool generator) {
        LIRGeneratorTool tool = generator.getLIRGeneratorTool();
        Variable result = tool.newVariable(tool.getLIRKind(stamp));

        if (atomicNode instanceof TornadoAtomicIntegerNode) {
            TornadoAtomicIntegerNode atomicIntegerNode = (TornadoAtomicIntegerNode) atomicNode;

            int indexFromGlobal = atomicIntegerNode.getIndexFromGlobalMemory();

            CUDAUnary.IntrinsicAtomicOperator intrinsicAtomicAdd = new CUDAUnary.IntrinsicAtomicOperator( //
                    CUDAAssembler.CUDAUnaryIntrinsic.ATOMIC_DEC, //
                    tool.getLIRKind(stamp), //
                    generator.operand(atomicNode), //
                    indexFromGlobal, //
                    atomicOperator);

            CUDALIRStmt.AssignStmt assignStmt = new CUDALIRStmt.AssignStmt(result, intrinsicAtomicAdd);
            tool.append(assignStmt);
            generator.setResult(this, result);
        }
    }

    @Override
    public void generate(NodeLIRBuilderTool generator) {
        if (ATOMIC_2_0) {
            generateExpressionForOpenCL2_0(generator);
        } else {
            generateExpressionForOpenCL1_0(generator);
        }
    }

    @Override
    public ValueNode getAtomicNode() {
        return atomicNode;
    }
}
