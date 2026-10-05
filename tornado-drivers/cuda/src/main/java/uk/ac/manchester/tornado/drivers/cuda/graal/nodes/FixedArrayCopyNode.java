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
package uk.ac.manchester.tornado.drivers.cuda.graal.nodes;

import jdk.vm.ci.meta.JavaKind;
import jdk.vm.ci.meta.ResolvedJavaType;
import jdk.vm.ci.meta.Value;
import tornado.graal.compiler.core.common.LIRKind;
import tornado.graal.compiler.core.common.type.StampFactory;
import tornado.graal.compiler.graph.NodeClass;
import tornado.graal.compiler.lir.Variable;
import tornado.graal.compiler.nodeinfo.NodeInfo;
import tornado.graal.compiler.nodes.ValuePhiNode;
import tornado.graal.compiler.nodes.calc.FloatingNode;
import tornado.graal.compiler.nodes.spi.LIRLowerable;
import tornado.graal.compiler.nodes.spi.NodeLIRBuilderTool;
import uk.ac.manchester.tornado.drivers.cuda.graal.CUDAArchitecture;
import uk.ac.manchester.tornado.drivers.cuda.graal.asm.CUDAAssembler;
import uk.ac.manchester.tornado.drivers.cuda.graal.lir.CUDABinary;
import uk.ac.manchester.tornado.drivers.cuda.graal.lir.CUDAKind;
import uk.ac.manchester.tornado.drivers.cuda.graal.lir.CUDALIRStmt;

/**
 * This node generates a pointer copy between two arrays in private memory.
 */
@NodeInfo
public class FixedArrayCopyNode extends FloatingNode implements LIRLowerable {

    public static final NodeClass<FixedArrayCopyNode> TYPE = NodeClass.create(FixedArrayCopyNode.class);

    @Input
    protected ValuePhiNode conditionalPhiNode;

    protected ResolvedJavaType elementType;
    protected CUDAAssembler.CUDABinaryTemplate pointerCopyTemplate;
    protected CUDAArchitecture.CUDAMemoryBase memoryRegister;

    public FixedArrayCopyNode(ValuePhiNode conditionalPhiNode, ResolvedJavaType elementType, CUDAArchitecture.CUDAMemoryBase memoryRegister) {
        super(TYPE, StampFactory.forKind(JavaKind.Object));
        this.conditionalPhiNode = conditionalPhiNode;
        this.elementType = elementType;
        this.memoryRegister = memoryRegister;
        this.pointerCopyTemplate = CUDAKind.resolvePrivatePointerCopyTemplate(elementType);
    }

    public CUDAArchitecture.CUDAMemoryBase getMemoryRegister() {
        return memoryRegister;
    }

    @Override
    public void generate(NodeLIRBuilderTool gen) {
        LIRKind lirKind = LIRKind.value(gen.getLIRGeneratorTool().target().arch.getWordKind());
        final Variable ptr = gen.getLIRGeneratorTool().newVariable(lirKind);
        Value fixedArrayValue = gen.operand(conditionalPhiNode);
        final CUDABinary.Expr declarationPtr = new CUDABinary.Expr(pointerCopyTemplate, lirKind, ptr, fixedArrayValue);
        final CUDALIRStmt.ExprStmt ptrExpr = new CUDALIRStmt.ExprStmt(declarationPtr);
        gen.getLIRGeneratorTool().append(ptrExpr);
        gen.setResult(this, ptr);
    }
}
