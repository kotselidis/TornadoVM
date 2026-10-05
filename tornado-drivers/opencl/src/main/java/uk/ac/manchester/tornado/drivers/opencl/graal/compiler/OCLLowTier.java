/*
 * Copyright (c) 2020, 2023, 2024, APT Group, Department of Computer Science,
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
package uk.ac.manchester.tornado.drivers.opencl.graal.compiler;

import static tornado.graal.compiler.core.common.GraalOptions.ConditionalElimination;

import tornado.graal.compiler.options.OptionValues;
import tornado.graal.compiler.phases.common.AddressLoweringByNodePhase;
import tornado.graal.compiler.phases.common.AddressLoweringByNodePhase.AddressLowering;
import tornado.graal.compiler.phases.common.CanonicalizerPhase;
import tornado.graal.compiler.phases.common.DeadCodeEliminationPhase;
import tornado.graal.compiler.phases.common.FixReadsPhase;
import tornado.graal.compiler.phases.common.IterativeConditionalEliminationPhase;
import tornado.graal.compiler.phases.common.LowTierLoweringPhase;
import tornado.graal.compiler.phases.common.UseTrappingNullChecksPhase;
import tornado.graal.compiler.phases.schedule.SchedulePhase;

import uk.ac.manchester.tornado.api.TornadoDeviceContext;
import uk.ac.manchester.tornado.drivers.common.compiler.phases.analysis.TornadoFeatureExtraction;
import uk.ac.manchester.tornado.drivers.common.compiler.phases.loops.TornadoLoopCanonicalization;
import uk.ac.manchester.tornado.drivers.common.compiler.phases.utils.DumpLowTierGraph;
import uk.ac.manchester.tornado.drivers.opencl.graal.phases.InfinityReplacementPhase;
import uk.ac.manchester.tornado.drivers.opencl.graal.phases.InverseSquareRootPhase;
import uk.ac.manchester.tornado.drivers.opencl.graal.phases.OCLFMAPhase;
import uk.ac.manchester.tornado.drivers.opencl.graal.phases.OCLFP16SupportPhase;
import uk.ac.manchester.tornado.drivers.opencl.graal.phases.OCLFP64SupportPhase;
import uk.ac.manchester.tornado.drivers.opencl.graal.phases.OCLFieldCoopsAccessPhase;
import uk.ac.manchester.tornado.drivers.opencl.graal.phases.TornadoAtomicsParametersPhase;
import uk.ac.manchester.tornado.drivers.opencl.graal.phases.TornadoAtomicsScheduling;
import uk.ac.manchester.tornado.drivers.opencl.graal.phases.TornadoFixedArrayCopyPhase;
import uk.ac.manchester.tornado.drivers.opencl.graal.phases.TornadoHalfFloatVectorOffset;
import uk.ac.manchester.tornado.runtime.common.TornadoOptions;
import uk.ac.manchester.tornado.runtime.graal.compiler.TornadoLowTier;

public class OCLLowTier extends TornadoLowTier {

    TornadoDeviceContext tornadoDeviceContext;

    public OCLLowTier(OptionValues options, TornadoDeviceContext tornadoDeviceContext, AddressLowering addressLowering) {
        this.tornadoDeviceContext = tornadoDeviceContext;
        CanonicalizerPhase canonicalizer = getCannonicalizer(options);

        appendPhase(new OCLFP64SupportPhase(tornadoDeviceContext));

        appendPhase(new OCLFP16SupportPhase(tornadoDeviceContext));

        appendPhase(new LowTierLoweringPhase(canonicalizer));

        if (ConditionalElimination.getValue(options)) {
            appendPhase(new IterativeConditionalEliminationPhase(canonicalizer, true));
        }

        // TODO Investigate why FixReads break kfusion on Nvidia GPUs
        if (TornadoOptions.ENABLE_FIX_READS) {
            appendPhase(new FixReadsPhase(true, new SchedulePhase(SchedulePhase.SchedulingStrategy.LATEST_OUT_OF_LOOPS)));
        }
        appendPhase(new UseTrappingNullChecksPhase());

        appendPhase(new TornadoFixedArrayCopyPhase());

        appendPhase(new AddressLoweringByNodePhase(addressLowering));

        appendPhase(new DeadCodeEliminationPhase(DeadCodeEliminationPhase.Optionality.Required));

        appendPhase(new TornadoHalfFloatVectorOffset());

        appendPhase(new TornadoLoopCanonicalization());

        if (TornadoOptions.ENABLE_FMA) {
            appendPhase(new OCLFMAPhase());
        }

        if (TornadoOptions.MATH_OPTIMIZATIONS) {
            appendPhase(new InverseSquareRootPhase());
        }

        appendPhase(new InfinityReplacementPhase());

        appendPhase(new TornadoAtomicsParametersPhase());

        appendPhase(new TornadoAtomicsScheduling());

        appendPhase(new OCLFieldCoopsAccessPhase());

        appendPhase(new SchedulePhase(SchedulePhase.SchedulingStrategy.LATEST_OUT_OF_LOOPS));

        if (TornadoOptions.FEATURE_EXTRACTION) {
            appendPhase(new TornadoFeatureExtraction(tornadoDeviceContext));
        }

        if (TornadoOptions.DUMP_LOW_TIER_WITH_IGV) {
            appendPhase(new DumpLowTierGraph());
        }

    }

    private CanonicalizerPhase getCannonicalizer(OptionValues options) {
        return CanonicalizerPhase.create();
    }
}
