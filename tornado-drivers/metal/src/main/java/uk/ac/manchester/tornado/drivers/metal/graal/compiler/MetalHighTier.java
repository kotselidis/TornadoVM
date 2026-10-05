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
package uk.ac.manchester.tornado.drivers.metal.graal.compiler;

import static tornado.graal.compiler.core.common.GraalOptions.ConditionalElimination;
import static tornado.graal.compiler.core.common.GraalOptions.OptConvertDeoptsToGuards;
import static tornado.graal.compiler.core.common.GraalOptions.PartialEscapeAnalysis;
import static tornado.graal.compiler.core.phases.HighTier.Options.Inline;
import static tornado.graal.compiler.phases.common.DeadCodeEliminationPhase.Optionality.Optional;

import tornado.graal.compiler.loop.phases.ConvertDeoptimizeToGuardPhase;
import tornado.graal.compiler.loop.phases.LoopFullUnrollPhase;
import tornado.graal.compiler.nodes.loop.DefaultLoopPolicies;
import tornado.graal.compiler.nodes.loop.LoopPolicies;
import tornado.graal.compiler.options.OptionValues;
import tornado.graal.compiler.phases.common.CanonicalizerPhase;
import tornado.graal.compiler.phases.common.DeadCodeEliminationPhase;
import tornado.graal.compiler.phases.common.HighTierLoweringPhase;
import tornado.graal.compiler.phases.common.IterativeConditionalEliminationPhase;
import tornado.graal.compiler.phases.common.inlining.InliningPhase;
import tornado.graal.compiler.phases.schedule.SchedulePhase;
import tornado.graal.compiler.virtual.phases.ea.PartialEscapePhase;

import jdk.vm.ci.meta.MetaAccessProvider;
import uk.ac.manchester.tornado.api.TornadoDeviceContext;
import uk.ac.manchester.tornado.drivers.common.compiler.phases.analysis.TornadoShapeAnalysis;
import uk.ac.manchester.tornado.drivers.common.compiler.phases.guards.ExceptionSuppression;
import uk.ac.manchester.tornado.drivers.common.compiler.phases.guards.TornadoValueTypeCleanup;
import uk.ac.manchester.tornado.drivers.common.compiler.phases.memalloc.TornadoFieldAccessFixup;
import uk.ac.manchester.tornado.drivers.common.compiler.phases.memalloc.TornadoLocalMemoryAllocation;
import uk.ac.manchester.tornado.drivers.common.compiler.phases.memalloc.TornadoNewArrayDevirtualizationReplacement;
import uk.ac.manchester.tornado.drivers.common.compiler.phases.memalloc.TornadoPrivateArrayPiRemoval;
import uk.ac.manchester.tornado.drivers.metal.graal.phases.TornadoBatchGlobalIndexOffset;
import uk.ac.manchester.tornado.drivers.metal.graal.phases.TornadoHalfFloatReplacement;
import uk.ac.manchester.tornado.drivers.metal.graal.phases.TornadoMetalIntrinsicsReplacements;
import uk.ac.manchester.tornado.drivers.metal.graal.phases.TornadoParallelScheduler;
import uk.ac.manchester.tornado.drivers.metal.graal.phases.TornadoTaskSpecialisation;
import uk.ac.manchester.tornado.runtime.common.TornadoOptions;
import uk.ac.manchester.tornado.runtime.graal.compiler.TornadoHighTier;
import uk.ac.manchester.tornado.runtime.graal.phases.TornadoInliningPolicy;
import uk.ac.manchester.tornado.runtime.graal.phases.sketcher.TornadoFullInliningPolicy;
import uk.ac.manchester.tornado.runtime.graal.phases.sketcher.TornadoPartialInliningPolicy;

public class MetalHighTier extends TornadoHighTier {

    public MetalHighTier(OptionValues options, TornadoDeviceContext deviceContext, CanonicalizerPhase.CustomSimplification customCanonicalizer, MetaAccessProvider metaAccessProvider) {
        super(customCanonicalizer);

        CanonicalizerPhase canonicalizer = createCanonicalizerPhase(options, customCanonicalizer);
        appendPhase(canonicalizer);

        if (Inline.getValue(options)) {
            TornadoInliningPolicy inliningPolicy = (TornadoOptions.FULL_INLINING) ? new TornadoFullInliningPolicy() : new TornadoPartialInliningPolicy();
            appendPhase(new InliningPhase(inliningPolicy, canonicalizer));
            appendPhase(new DeadCodeEliminationPhase(Optional));
            if (ConditionalElimination.getValue(options)) {
                appendPhase(canonicalizer);
                appendPhase(new IterativeConditionalEliminationPhase(canonicalizer, false));
            }
        }
        appendPhase(new TornadoTaskSpecialisation(canonicalizer));
        appendPhase(new TornadoBatchGlobalIndexOffset());
        appendPhase(new TornadoFieldAccessFixup());
        appendPhase(canonicalizer);
        appendPhase(new DeadCodeEliminationPhase(Optional));

        appendPhase(canonicalizer);

        appendPhase(new TornadoNewArrayDevirtualizationReplacement());

        appendPhase(new TornadoHalfFloatReplacement());

        if (PartialEscapeAnalysis.getValue(options)) {
            appendPhase(new PartialEscapePhase(true, canonicalizer, options));
        }

        appendPhase(new TornadoPrivateArrayPiRemoval());

        appendPhase(new TornadoValueTypeCleanup());

        if (OptConvertDeoptsToGuards.getValue(options)) {
            appendPhase(new ConvertDeoptimizeToGuardPhase(canonicalizer));
        }

        appendPhase(new TornadoShapeAnalysis());
        appendPhase(canonicalizer);
        appendPhase(new TornadoParallelScheduler());

        appendPhase(new SchedulePhase(SchedulePhase.SchedulingStrategy.EARLIEST));

        LoopPolicies loopPolicies = new DefaultLoopPolicies();
        appendPhase(new LoopFullUnrollPhase(canonicalizer, loopPolicies));

        appendPhase(canonicalizer);
        appendPhase(new DeadCodeEliminationPhase(Optional));

        appendPhase(new SchedulePhase(SchedulePhase.SchedulingStrategy.EARLIEST));

        appendPhase(new HighTierLoweringPhase(canonicalizer));

        // After the first Lowering, TornadoVM replaces reductions with snippets
        // that contains method calls to barriers.

        appendPhase(new TornadoMetalIntrinsicsReplacements(metaAccessProvider));

        appendPhase(new TornadoLocalMemoryAllocation());

        appendPhase(new ExceptionSuppression());
    }

    private CanonicalizerPhase createCanonicalizerPhase(OptionValues options, CanonicalizerPhase.CustomSimplification customCanonicalizer) {
        CanonicalizerPhase canonicalizerPhase = CanonicalizerPhase.create();
        return canonicalizerPhase.copyWithCustomSimplification(customCanonicalizer);

    }
}
