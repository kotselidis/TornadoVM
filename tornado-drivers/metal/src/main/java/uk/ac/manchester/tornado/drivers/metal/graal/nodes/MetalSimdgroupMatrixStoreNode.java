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

import tornado.graal.compiler.core.common.type.StampFactory;
import tornado.graal.compiler.graph.NodeClass;
import tornado.graal.compiler.lir.gen.LIRGeneratorTool;
import tornado.graal.compiler.nodeinfo.NodeInfo;
import tornado.graal.compiler.nodes.FixedWithNextNode;
import tornado.graal.compiler.nodes.ValueNode;
import tornado.graal.compiler.nodes.spi.LIRLowerable;
import tornado.graal.compiler.nodes.spi.NodeLIRBuilderTool;

import uk.ac.manchester.tornado.api.common.Access;
import uk.ac.manchester.tornado.drivers.metal.graal.lir.MetalLIRStmt;
import uk.ac.manchester.tornado.runtime.graal.nodes.interfaces.MarkArrayParameterAccess;

/**
 * Stores an 8x8 fragment to device memory ({@code simdgroup_store}), for
 * {@link uk.ac.manchester.tornado.api.KernelContext#simdgroupMatrixStore}.
 *
 * <p>A pure side-effect that must run convergently across the SIMD group, so it is
 * fixed in the control flow. Implements {@link MarkArrayParameterAccess} so the
 * dataflow analysis marks the destination array as written — otherwise the
 * device-to-host copy is skipped and results come back as zeros.
 */
@NodeInfo
public class MetalSimdgroupMatrixStoreNode extends FixedWithNextNode implements LIRLowerable, MarkArrayParameterAccess {

    public static final NodeClass<MetalSimdgroupMatrixStoreNode> TYPE = NodeClass.create(MetalSimdgroupMatrixStoreNode.class);

    @Input
    private ValueNode matrix;
    @Input
    private ValueNode array;
    @Input
    private ValueNode base;
    @Input
    private ValueNode stride;

    public MetalSimdgroupMatrixStoreNode(ValueNode matrix, ValueNode array, ValueNode base, ValueNode stride) {
        super(TYPE, StampFactory.forVoid());
        this.matrix = matrix;
        this.array = array;
        this.base = base;
        this.stride = stride;
    }

    @Override
    public void generate(NodeLIRBuilderTool gen) {
        LIRGeneratorTool tool = gen.getLIRGeneratorTool();
        tool.append(new MetalLIRStmt.SimdgroupMatrixStoreStmt(gen.operand(matrix), gen.operand(array), gen.operand(base), gen.operand(stride)));
    }

    @Override
    public Access getArrayParameterAccess(ValueNode parameter) {
        return (parameter == MarkArrayParameterAccess.unwrapPi(array)) ? Access.WRITE_ONLY : Access.NONE;
    }
}
