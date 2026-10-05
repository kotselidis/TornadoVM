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
package uk.ac.manchester.tornado.drivers.metal.graal.nodes;

import jdk.vm.ci.meta.JavaKind;
import jdk.vm.ci.meta.Value;
import tornado.graal.compiler.core.common.LIRKind;
import tornado.graal.compiler.core.common.type.StampFactory;
import tornado.graal.compiler.graph.NodeClass;
import tornado.graal.compiler.lir.Variable;
import tornado.graal.compiler.lir.gen.LIRGeneratorTool;
import tornado.graal.compiler.nodeinfo.NodeInfo;
import tornado.graal.compiler.nodes.ValueNode;
import tornado.graal.compiler.nodes.spi.LIRLowerable;
import tornado.graal.compiler.nodes.spi.NodeLIRBuilderTool;
import uk.ac.manchester.tornado.drivers.metal.graal.lir.MetalKind;
import uk.ac.manchester.tornado.drivers.metal.graal.lir.MetalLIRStmt;

/**
 * Reinterprets the bit pattern of an f16 value as an unsigned 32-bit integer (zero-extended),
 * which is what {@code HalfFloat.getHalfFloatValue()} returns.
 *
 * <p>Metal Shading Language emitted (via MetalLIRStmt.HalfBitsToIntStmt):
 * <pre>u32_result = (uint) as_type&lt;ushort&gt;(half_value);</pre>
 *
 * <p>{@code as_type} is a reinterpretation, not a conversion: it keeps the 16 bits as they are,
 * where a cast would renumber them.
 */
@NodeInfo
public class MetalConvertHalfBitsToIntNode extends ValueNode implements LIRLowerable {

    public static final NodeClass<MetalConvertHalfBitsToIntNode> TYPE = NodeClass.create(MetalConvertHalfBitsToIntNode.class);

    @Input
    private ValueNode halfValueNode;

    public MetalConvertHalfBitsToIntNode(ValueNode halfValueNode) {
        super(TYPE, StampFactory.forKind(JavaKind.Int));
        this.halfValueNode = halfValueNode;
    }

    @Override
    public void generate(NodeLIRBuilderTool generator) {
        LIRGeneratorTool tool = generator.getLIRGeneratorTool();
        Variable result = tool.newVariable(LIRKind.value(MetalKind.UINT));
        Value halfValue = generator.operand(halfValueNode);
        tool.append(new MetalLIRStmt.HalfBitsToIntStmt(result, halfValue));
        generator.setResult(this, result);
    }
}
