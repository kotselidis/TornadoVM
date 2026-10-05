/*
 * Copyright (c) 2018, 2020, APT Group, Department of Computer Science,
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
package uk.ac.manchester.tornado.drivers.cuda.graal.compiler;

import java.util.List;

import tornado.graal.compiler.core.common.cfg.BasicBlock;
import tornado.graal.compiler.core.common.cfg.BlockMap;
import tornado.graal.compiler.graph.Node;
import tornado.graal.compiler.lir.gen.LIRGenerationResult;
import tornado.graal.compiler.lir.gen.LIRGenerator;
import tornado.graal.compiler.lir.gen.LIRGeneratorTool;
import tornado.graal.compiler.lir.phases.LIRPhase;
import tornado.graal.compiler.lir.ssa.SSAUtil;
import tornado.graal.compiler.nodes.StructuredGraph;
import tornado.graal.compiler.nodes.StructuredGraph.ScheduleResult;
import tornado.graal.compiler.nodes.cfg.HIRBlock;
import tornado.graal.compiler.nodes.spi.NodeLIRBuilderTool;

import jdk.vm.ci.code.TargetDescription;

public class CUDALIRGenerationPhase extends LIRPhase<CUDALIRGenerationPhase.LIRGenerationContext> {

    private static void emitBlock(final CUDANodeLIRBuilder nodeLirGen, final LIRGenerationResult lirGenRes, final HIRBlock b, final StructuredGraph graph, final BlockMap<List<Node>> blockMap,
            boolean isKernel) {
        if (lirGenRes.getLIR().getLIRforBlock(b) == null) {
            for (int i = 0; i < b.getPredecessorCount(); i++) {
                if (!b.isLoopHeader() || !b.getPredecessorAt(i).isLoopEnd()) {
                    emitBlock(nodeLirGen, lirGenRes, b.getPredecessorAt(i), graph, blockMap, isKernel);
                }
            }
            nodeLirGen.doBlock(b, graph, blockMap, isKernel);
        }
    }

    @Override
    protected final void run(final TargetDescription target, final LIRGenerationResult lirGenRes, final CUDALIRGenerationPhase.LIRGenerationContext context) {

        final NodeLIRBuilderTool nodeLirBuilder = context.nodeLirBuilder;
        final StructuredGraph graph = context.graph;
        final ScheduleResult schedule = context.schedule;
        final BlockMap<List<Node>> blockMap = schedule.getBlockToNodesMap();
        BasicBlock<?>[] blocks = lirGenRes.getLIR().getControlFlowGraph().getBlocks();

        for (BasicBlock<?> b : blocks) {
            emitBlock((CUDANodeLIRBuilder) nodeLirBuilder, lirGenRes, (HIRBlock) b, graph, blockMap, context.isKernel);
        }
        ((LIRGenerator) context.lirGen).beforeRegisterAllocation();
    }

    public static final class LIRGenerationContext {

        private final StructuredGraph graph;
        private final LIRGeneratorTool lirGen;
        private final NodeLIRBuilderTool nodeLirBuilder;
        private final ScheduleResult schedule;
        private final boolean isKernel;

        public LIRGenerationContext(final LIRGeneratorTool lirGen, final NodeLIRBuilderTool nodeLirBuilder, final StructuredGraph graph, final ScheduleResult schedule, final boolean isKernel) {
            this.nodeLirBuilder = nodeLirBuilder;
            this.lirGen = lirGen;
            this.graph = graph;
            this.schedule = schedule;
            this.isKernel = isKernel;
        }
    }

}
