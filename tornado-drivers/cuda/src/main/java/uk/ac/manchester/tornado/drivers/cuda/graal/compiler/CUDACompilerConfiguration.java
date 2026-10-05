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
package uk.ac.manchester.tornado.drivers.cuda.graal.compiler;

import tornado.graal.compiler.lir.phases.LIRPhaseSuite;
import tornado.graal.compiler.lir.phases.PostAllocationOptimizationPhase.PostAllocationOptimizationContext;
import tornado.graal.compiler.lir.phases.PostAllocationOptimizationStage;
import tornado.graal.compiler.lir.phases.PreAllocationOptimizationPhase.PreAllocationOptimizationContext;
import tornado.graal.compiler.lir.phases.PreAllocationOptimizationStage;
import tornado.graal.compiler.options.OptionValues;
import tornado.graal.compiler.phases.common.AddressLoweringByNodePhase.AddressLowering;
import tornado.graal.compiler.phases.common.CanonicalizerPhase;

import jdk.vm.ci.meta.MetaAccessProvider;
import uk.ac.manchester.tornado.api.TornadoDeviceContext;
import uk.ac.manchester.tornado.runtime.graal.compiler.TornadoCompilerConfiguration;
import uk.ac.manchester.tornado.runtime.graal.compiler.TornadoHighTier;
import uk.ac.manchester.tornado.runtime.graal.compiler.TornadoLowTier;
import uk.ac.manchester.tornado.runtime.graal.compiler.TornadoMidTier;
import uk.ac.manchester.tornado.runtime.graal.compiler.TornadoSketchTier;
import uk.ac.manchester.tornado.runtime.graal.phases.TornadoAllocationStage;

public class CUDACompilerConfiguration implements TornadoCompilerConfiguration {

    @Override
    public TornadoAllocationStage createAllocationStage(OptionValues options) {
        return new TornadoAllocationStage();
    }

    @Override
    public TornadoSketchTier createSketchTier(OptionValues options, CanonicalizerPhase.CustomSimplification canonicalizer) {
        return new TornadoSketchTier(options, canonicalizer);
    }

    @Override
    public TornadoHighTier createHighTier(OptionValues options, TornadoDeviceContext deviceContext, CanonicalizerPhase.CustomSimplification canonicalizer, MetaAccessProvider metaAccessProvider) {
        return new CUDAHighTier(options, deviceContext, canonicalizer, metaAccessProvider);
    }

    @Override
    public TornadoLowTier createLowTier(OptionValues options, TornadoDeviceContext deviceContext, AddressLowering addressLowering) {
        return new CUDALowTier(options, deviceContext, addressLowering);
    }

    @Override
    public TornadoMidTier createMidTier(OptionValues options) {
        return new CUDAMidTier(options);
    }

    @Override
    public LIRPhaseSuite<PostAllocationOptimizationContext> createPostAllocationOptimizationStage(OptionValues options) {
        return new PostAllocationOptimizationStage(options);
    }

    @Override
    public LIRPhaseSuite<PreAllocationOptimizationContext> createPreAllocationOptimizationStage(OptionValues options) {
        return new PreAllocationOptimizationStage(options);
    }

}
