/*
 * Copyright (c) 2020, APT Group, Department of Computer Science,
 * School of Engineering, The University of Manchester. All rights reserved.
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
package uk.ac.manchester.tornado.drivers.cuda.graal.phases;

import static uk.ac.manchester.tornado.runtime.TornadoCoreRuntime.getDebugContext;

import java.util.Optional;

import tornado.graal.compiler.debug.DebugContext;
import tornado.graal.compiler.nodes.ConstantNode;
import tornado.graal.compiler.nodes.GraphState;
import tornado.graal.compiler.nodes.StructuredGraph;
import tornado.graal.compiler.nodes.ValueNode;
import tornado.graal.compiler.nodes.calc.AddNode;
import tornado.graal.compiler.nodes.calc.MulNode;
import tornado.graal.compiler.nodes.calc.SubNode;
import tornado.graal.compiler.phases.BasePhase;

import jdk.vm.ci.meta.JavaKind;
import uk.ac.manchester.tornado.drivers.cuda.graal.nodes.GlobalThreadIdNode;
import uk.ac.manchester.tornado.drivers.cuda.graal.nodes.GlobalThreadSizeNode;
import uk.ac.manchester.tornado.drivers.cuda.graal.nodes.CUDAIntBinaryIntrinsicNode;
import uk.ac.manchester.tornado.drivers.cuda.graal.nodes.calc.DivNode;
import uk.ac.manchester.tornado.runtime.common.TornadoSchedulingStrategy;
import uk.ac.manchester.tornado.runtime.common.TornadoXPUDevice;
import uk.ac.manchester.tornado.runtime.graal.nodes.AbstractParallelNode;
import uk.ac.manchester.tornado.runtime.graal.nodes.ParallelOffsetNode;
import uk.ac.manchester.tornado.runtime.graal.nodes.ParallelRangeNode;
import uk.ac.manchester.tornado.runtime.graal.nodes.ParallelStrideNode;
import uk.ac.manchester.tornado.runtime.graal.phases.TornadoHighTierContext;

public class TornadoParallelScheduler extends BasePhase<TornadoHighTierContext> {

    @Override
    public Optional<NotApplicable> notApplicableTo(GraphState graphState) {
        return ALWAYS_APPLICABLE;
    }

    private void replaceOffsetNode(TornadoSchedulingStrategy schedule, StructuredGraph graph, ParallelOffsetNode offset, ParallelRangeNode range, ValueNode blockSize) {
        switch (schedule) {
            case PER_CPU_BLOCK -> replaceOffsetPerBlock(graph, offset, blockSize);
            case PER_ACCELERATOR_ITERATION -> replaceOffsetPerIteration(graph, offset, range);
        }
    }

    private void replaceOffsetPerIteration(StructuredGraph graph, ParallelOffsetNode offset, ParallelRangeNode range) {
        final ConstantNode index = graph.addOrUnique(ConstantNode.forInt(offset.index()));
        final GlobalThreadIdNode threadId = graph.addOrUnique(new GlobalThreadIdNode(index));
        final AddNode addNode = graph.addOrUnique(new AddNode(threadId, offset.value()));
        final MulNode mulNode = graph.addOrUnique(new MulNode(addNode, range.stride().value()));
        offset.replaceAtUsages(mulNode);
        offset.safeDelete();
    }

    private void replaceOffsetPerBlock(StructuredGraph graph, ParallelOffsetNode offset, ValueNode blockSize) {
        final GlobalThreadIdNode threadId = graph.addOrUnique(new GlobalThreadIdNode(ConstantNode.forInt(offset.index(), graph)));
        final MulNode newOffset = graph.addOrUnique(new MulNode(threadId, blockSize));
        offset.replaceAtUsages(newOffset);
        offset.safeDelete();
    }

    private void replaceStrideNode(TornadoSchedulingStrategy schedule, StructuredGraph graph, ParallelStrideNode stride) {
        switch (schedule) {
            case PER_CPU_BLOCK -> replaceStridePerBlock(stride);
            case PER_ACCELERATOR_ITERATION -> replaceStridePerIteration(graph, stride);
        }
    }

    private void replaceStridePerIteration(StructuredGraph graph, ParallelStrideNode stride) {
        final ConstantNode index = graph.addOrUnique(ConstantNode.forInt(stride.index()));
        final GlobalThreadSizeNode threadCount = graph.addOrUnique(new GlobalThreadSizeNode(index));
        stride.replaceAtUsages(threadCount);
        stride.safeDelete();
    }

    private void replaceStridePerBlock(ParallelStrideNode stride) {
        stride.replaceAtUsages(stride.value());
        stride.safeDelete();
    }

    private ValueNode replaceRangeNode(TornadoSchedulingStrategy schedule, StructuredGraph graph, ParallelRangeNode range) {
        switch (schedule) {
            case PER_CPU_BLOCK -> {
                return replaceRangePerBlock(graph, range);
            }
            case PER_ACCELERATOR_ITERATION -> {
                replaceRangePerIteration(range);
                return null;
            }
        }
        return null;
    }

    private void replaceRangePerIteration(ParallelRangeNode range) {
        range.replaceAtUsages(range.value());
    }

    // CPU-Scheduling with Stride
    private ValueNode buildBlockSize(StructuredGraph graph, ParallelRangeNode range) {
        final ValueNode rangeByStride = graph.addOrUnique(DivNode.create(range.value(), range.stride().value()));
        final SubNode trueRange = graph.addOrUnique(new SubNode(rangeByStride, range.offset().value()));
        final ConstantNode index = ConstantNode.forInt(range.index(), graph);
        final GlobalThreadSizeNode threadCount = graph.addOrUnique(new GlobalThreadSizeNode(index));
        final SubNode threadCountM1 = graph.addOrUnique(new SubNode(threadCount, ConstantNode.forInt(1, graph)));
        final AddNode adjustedTrueRange = graph.addOrUnique(new AddNode(trueRange, threadCountM1));
        final ValueNode div = graph.addOrUnique(DivNode.create(adjustedTrueRange, threadCount));
        return graph.addOrUnique(new MulNode(div, range.stride().value()));
    }

    // CPU-Scheduling with Stride
    private ValueNode replaceRangePerBlock(StructuredGraph graph, ParallelRangeNode range) {
        ValueNode blockSize = buildBlockSize(graph, range);
        final GlobalThreadIdNode threadId = graph.addOrUnique(new GlobalThreadIdNode(ConstantNode.forInt(range.index(), graph)));
        final MulNode newOffset = graph.addOrUnique(new MulNode(threadId, blockSize));
        final AddNode newRange = graph.addOrUnique(new AddNode(newOffset, blockSize));
        final MulNode stride = graph.addOrUnique(new MulNode(newRange, range.stride().value()));
        final ValueNode adjustedRange = graph.addOrUnique(CUDAIntBinaryIntrinsicNode.create(stride, range.value(), CUDAIntBinaryIntrinsicNode.Operation.MIN, JavaKind.Int));
        range.replaceAtUsages(adjustedRange);
        range.safeDelete();
        return blockSize;
    }

    @Override
    protected void run(StructuredGraph graph, TornadoHighTierContext context) {
        if (context.getMeta() == null) {
            return;
        }
        TornadoXPUDevice device = context.getDeviceMapping();
        final TornadoSchedulingStrategy strategy = device.getPreferredSchedule();
        long[] maxWorkItemSizes = device.getPhysicalDevice().getDeviceMaxWorkItemSizes();

        graph.getNodes().filter(ParallelRangeNode.class).forEach(node -> {
            if (maxWorkItemSizes[node.index()] > 1) {
                ParallelOffsetNode offset = node.offset();
                ParallelStrideNode stride = node.stride();
                ValueNode blockSize = replaceRangeNode(strategy, graph, node);
                replaceOffsetNode(strategy, graph, offset, node, blockSize);
                replaceStrideNode(strategy, graph, stride);

            } else {
                serialiseLoop(node);
            }
            getDebugContext().dump(DebugContext.BASIC_LEVEL, graph, "after scheduling loop index=" + node.index());
        });
        graph.clearLastSchedule();
    }

    private void serialiseLoop(ParallelRangeNode range) {
        ParallelOffsetNode offset = range.offset();
        ParallelStrideNode stride = range.stride();
        range.replaceAtUsages(range.value());
        killNode(range);
        offset.replaceAtUsages(offset.value());
        stride.replaceAtUsages(stride.value());
        killNode(offset);
        killNode(stride);
    }

    private void killNode(AbstractParallelNode node) {
        if (node.inputs().isNotEmpty()) {
            node.clearInputs();
        }
        if (!node.isDeleted()) {
            node.safeDelete();
        }
    }
}
