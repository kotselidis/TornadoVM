/*
 * Copyright (c) 2026, APT Group, Department of Computer Science,
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
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
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
import jdk.vm.ci.meta.Value;
import uk.ac.manchester.tornado.drivers.cuda.graal.lir.CUDALIRStmt;

/**
 * Computes {@code __dp4a(a, b, c)} over two already-packed 32-bit operands, each holding
 * four signed 8-bit integers. Maps directly to the CUDA {@code __dp4a} device builtin
 * (compute capability >= 6.1), the CUDA-C counterpart of the PTX {@code dp4a.s32.s32}.
 */
@NodeInfo
public class DP4APackedNode extends ValueNode implements LIRLowerable {

    public static final NodeClass<DP4APackedNode> TYPE = NodeClass.create(DP4APackedNode.class);

    @Input
    private ValueNode a;

    @Input
    private ValueNode b;

    @Input
    private ValueNode c;

    public DP4APackedNode(ValueNode a, ValueNode b, ValueNode c) {
        super(TYPE, StampFactory.forKind(JavaKind.Int));
        this.a = a;
        this.b = b;
        this.c = c;
    }

    public void generate(NodeLIRBuilderTool generator) {
        LIRGeneratorTool tool = generator.getLIRGeneratorTool();
        Variable result = tool.newVariable(tool.getLIRKind(stamp));
        Value a_value = generator.operand(a);
        Value b_value = generator.operand(b);
        Value accumulator_c = generator.operand(c);

        tool.append(new CUDALIRStmt.Dp4aPackedStmt(result, a_value, b_value, accumulator_c));
        generator.setResult(this, result);
    }

}
