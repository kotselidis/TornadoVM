/*
 * Copyright (c) 2023, 2024 APT Group, Department of Computer Science,
 * The University of Manchester. All rights reserved.
 * Copyright (c) 2009, 2017, 2025, Oracle and/or its affiliates. All rights reserved.
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
package uk.ac.manchester.tornado.drivers.common.compiler.phases.loops;

import java.util.Optional;

import tornado.graal.compiler.loop.phases.LoopTransformations;
import tornado.graal.compiler.nodes.GraphState;
import tornado.graal.compiler.nodes.StructuredGraph;
import tornado.graal.compiler.nodes.loop.LoopFragmentInside;
import tornado.graal.compiler.nodes.loop.LoopsData;
import tornado.graal.compiler.phases.BasePhase;
import tornado.graal.compiler.phases.common.CanonicalizerPhase;
import tornado.graal.compiler.phases.common.DeadCodeEliminationPhase;
import tornado.graal.compiler.phases.tiers.MidTierContext;

import uk.ac.manchester.tornado.runtime.TornadoCoreRuntime;
import uk.ac.manchester.tornado.runtime.common.TornadoOptions;
import uk.ac.manchester.tornado.runtime.graal.nodes.TornadoLoopsData;
import uk.ac.manchester.tornado.runtime.graal.phases.TornadoMidTierContext;

/**
 * Applies partial unroll on counted loops of more than 128 elements. By default,
 * the unroll factor is set to 2 except if the user explicitly passes a
 * different value power of two.
 *
 * @see tornado.graal.compiler.loop.phases.LoopTransformations
 */

public class TornadoPartialLoopUnrollPhase extends BasePhase<MidTierContext> {

    private static final int LOOP_UNROLL_FACTOR_DEFAULT = 2;
    private static final int LOOP_BOUND_UPPER_LIMIT = 16384;

    private static final int GRAPH_NODES_UPPER_LIMIT = 40000;

    private enum OptimizationStatus {
        SUCCESS, //
        ERROR;
    }

    private static OptimizationStatus partialUnroll(StructuredGraph graph, MidTierContext context) {
        LoopsData dataCounted;
        CanonicalizerPhase canonicalizer = CanonicalizerPhase.create();
        canonicalizer.apply(graph, context);
        try {
            dataCounted = new TornadoLoopsData(graph);
        } catch (NullPointerException nullPointerException) {
            return OptimizationStatus.ERROR;
        }
        dataCounted.detectCountedLoops();
        try {
            dataCounted.countedLoops().forEach(loop -> {
                if (LoopTransformations.isUnrollableLoop(loop)) {
                    int loopBound = loop.counted().getLimit().asJavaConstant().asInt();
                    if (isPowerOfTwo(loopBound) && (loopBound < LOOP_BOUND_UPPER_LIMIT)) {
                        LoopFragmentInside loopBody = loop.inside().duplicate();
                        loopBody.insertWithinAfter(loop, null);
                    }
                }
            });

            new DeadCodeEliminationPhase().apply(graph);
        } catch (NullPointerException runtimeException) {
            return OptimizationStatus.ERROR;
        }
        return OptimizationStatus.SUCCESS;
    }

    private static int getUnrollFactor() {
        return (isPowerOfTwo(TornadoOptions.UNROLL_FACTOR) && TornadoOptions.UNROLL_FACTOR <= 32) ? TornadoOptions.UNROLL_FACTOR : LOOP_UNROLL_FACTOR_DEFAULT;
    }

    private static int getUpperGraphLimit(int initialGraphNodeCount) {
        return (initialGraphNodeCount + (GRAPH_NODES_UPPER_LIMIT));
    }

    private static boolean isPowerOfTwo(int number) {
        return number > 0 && ((number & (number - 1)) == 0);
    }

    @Override
    public Optional<NotApplicable> notApplicableTo(GraphState graphState) {
        return ALWAYS_APPLICABLE;
    }

    private StructuredGraph checkStatus(StructuredGraph graph, StructuredGraph snapshot, OptimizationStatus status) {
        return status != OptimizationStatus.SUCCESS ? snapshot : graph;
    }

    @Override
    protected void run(StructuredGraph graph, MidTierContext context) {

        TornadoMidTierContext tornadoMidTierContext = (TornadoMidTierContext) context;
        if (!tornadoMidTierContext.getMeta().applyPartialLoopUnroll()) {
            return;
        }

        if (!graph.hasLoops()) {
            return;
        }

        int initialNodeCount = graph.getNodeCount();
        int unrollFactor = getUnrollFactor();

        StructuredGraph snapshot = (StructuredGraph) graph.copy(TornadoCoreRuntime.getDebugContext());
        for (int i = 0; Math.pow(2, i) < unrollFactor; i++) {
            if (graph.getNodeCount() < getUpperGraphLimit(initialNodeCount)) {
                OptimizationStatus status = partialUnroll(graph, context);
                graph = checkStatus(graph, snapshot, status);
                if (status != OptimizationStatus.SUCCESS) {
                    return;
                }
            }
        }
    }
}
