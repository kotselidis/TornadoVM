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
import tornado.graal.compiler.lir.gen.LIRGeneratorTool;
import tornado.graal.compiler.nodeinfo.NodeInfo;
import tornado.graal.compiler.nodes.FixedWithNextNode;
import tornado.graal.compiler.nodes.ValueNode;
import tornado.graal.compiler.nodes.spi.LIRLowerable;
import tornado.graal.compiler.nodes.spi.NodeLIRBuilderTool;
import uk.ac.manchester.tornado.drivers.cuda.graal.lir.CUDALIRStmt;

@NodeInfo
public class CUDAMMAStoreBSwizzledNode extends FixedWithNextNode implements LIRLowerable {

    public static final NodeClass<CUDAMMAStoreBSwizzledNode> TYPE = NodeClass.create(CUDAMMAStoreBSwizzledNode.class);

    @Input private ValueNode tile;
    @Input private ValueNode row;
    @Input private ValueNode column;
    @Input private ValueNode stride;
    @Input private ValueNode value;
    @Input private ValueNode byteOffset;

    public CUDAMMAStoreBSwizzledNode(ValueNode tile, ValueNode row, ValueNode column,
                                     ValueNode stride, ValueNode value, ValueNode byteOffset) {
        super(TYPE, StampFactory.forVoid());
        this.tile = tile; this.row = row; this.column = column;
        this.stride = stride; this.value = value; this.byteOffset = byteOffset;
    }

    @Override
    public void generate(NodeLIRBuilderTool gen) {
        LIRGeneratorTool tool = gen.getLIRGeneratorTool();
        tool.append(new CUDALIRStmt.SwizzledStoreFP16Stride32Stmt(
                gen.operand(tile), gen.operand(row), gen.operand(column),
                gen.operand(stride), gen.operand(value), gen.operand(byteOffset)));
    }
}
