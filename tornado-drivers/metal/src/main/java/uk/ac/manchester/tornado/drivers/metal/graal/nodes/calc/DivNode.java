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

package uk.ac.manchester.tornado.drivers.metal.graal.nodes.calc;

import tornado.graal.compiler.core.common.type.ArithmeticOpTable;
import tornado.graal.compiler.core.common.type.Stamp;
import tornado.graal.compiler.graph.NodeClass;
import tornado.graal.compiler.lir.LIRFrameState;
import tornado.graal.compiler.lir.gen.ArithmeticLIRGeneratorTool;
import tornado.graal.compiler.nodeinfo.NodeCycles;
import tornado.graal.compiler.nodeinfo.NodeInfo;
import tornado.graal.compiler.nodes.ConstantNode;
import tornado.graal.compiler.nodes.NodeView;
import tornado.graal.compiler.nodes.ValueNode;
import tornado.graal.compiler.nodes.calc.BinaryArithmeticNode;
import tornado.graal.compiler.nodes.spi.NodeLIRBuilderTool;


/*
    This implementation is copied from the Graal compiler 0:22. We need to do this because on later versions of the compiler,
    the DivNode as a child of FloatingNode does not exist any longer.
 */

@NodeInfo(
        shortName = "div_node",
        cycles = NodeCycles.CYCLES_32
)
public class DivNode extends BinaryArithmeticNode<ArithmeticOpTable.BinaryOp.Div> {
    public static final NodeClass<DivNode> TYPE = NodeClass.create(DivNode.class);

    private DivNode(ValueNode x, ValueNode y) {
        super(TYPE, getArithmeticOpTable(x).getDiv(), x, y);
    }

    public static ValueNode create(ValueNode x, ValueNode y) {
        ArithmeticOpTable.BinaryOp<ArithmeticOpTable.BinaryOp.Div> op = ArithmeticOpTable.forStamp(x.stamp(NodeView.DEFAULT)).getDiv();
        Stamp stamp = op.foldStamp(x.stamp(NodeView.DEFAULT), y.stamp(NodeView.DEFAULT));
        ConstantNode tryConstantFold = tryConstantFold(op, x, y, stamp, NodeView.DEFAULT);
        return tryConstantFold != null ? tryConstantFold : new DivNode(x, y);
    }

    @Override
    protected ArithmeticOpTable.BinaryOp<ArithmeticOpTable.BinaryOp.Div> getOp(ArithmeticOpTable table) {
        return table.getDiv();
    }

    @Override
    public void generate(NodeLIRBuilderTool builder) {
        generate(builder, builder.getLIRGeneratorTool().getArithmetic());
    }

    public void generate(NodeLIRBuilderTool nodeValueMap, ArithmeticLIRGeneratorTool gen) {
        nodeValueMap.setResult(this, gen.emitDiv(nodeValueMap.operand(this.getX()), nodeValueMap.operand(this.getY()), (LIRFrameState)null));
    }
}
