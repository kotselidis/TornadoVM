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
package uk.ac.manchester.tornado.drivers.cuda.graal.nodes;

import tornado.graal.compiler.core.common.type.StampFactory;
import tornado.graal.compiler.graph.NodeClass;
import tornado.graal.compiler.lir.Variable;
import tornado.graal.compiler.lir.gen.LIRGeneratorTool;
import tornado.graal.compiler.nodeinfo.NodeInfo;
import tornado.graal.compiler.nodes.FixedWithNextNode;
import tornado.graal.compiler.nodes.ValueNode;
import tornado.graal.compiler.nodes.spi.LIRLowerable;
import tornado.graal.compiler.nodes.spi.NodeLIRBuilderTool;

import jdk.vm.ci.meta.JavaKind;
import jdk.vm.ci.meta.Value;
import uk.ac.manchester.tornado.drivers.cuda.graal.lir.CUDALIRStmt;

/**
 * Graal IR node for {@code KernelContext.simdShuffleDown(float, int)}.
 *
 * <p>Lowered to a single CUDA warp-shuffle intrinsic
 * {@code __shfl_down_sync(0xffffffff, val, delta)}: the calling lane receives
 * the value held by the lane {@code delta} positions ahead in the warp.
 *
 * <p>Extends {@link FixedWithNextNode} because warp-shuffle operations are
 * convergent — all lanes must execute them together, so the node must not be
 * floated or reordered.
 */
@NodeInfo(shortName = "CUDAShuffleDown")
public class CUDAShuffleDownNode extends FixedWithNextNode implements LIRLowerable {

    public static final NodeClass<CUDAShuffleDownNode> TYPE = NodeClass.create(CUDAShuffleDownNode.class);

    @Input
    private ValueNode value;

    @Input
    private ValueNode delta;

    public CUDAShuffleDownNode(ValueNode value, ValueNode delta) {
        super(TYPE, StampFactory.forKind(JavaKind.Float));
        this.value = value;
        this.delta = delta;
    }

    @Override
    public void generate(NodeLIRBuilderTool gen) {
        LIRGeneratorTool tool = gen.getLIRGeneratorTool();
        Value source = gen.operand(value);
        Value operand = gen.operand(delta);
        Variable result = tool.newVariable(tool.getLIRKind(stamp));
        tool.append(new CUDALIRStmt.ShuffleSyncStmt(CUDALIRStmt.ShuffleSyncStmt.Mode.DOWN, result, source, operand));
        gen.setResult(this, result);
    }
}
