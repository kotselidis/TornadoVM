/*
 * Copyright (c) 2018-2022, 2024, APT Group, Department of Computer Science,
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
package uk.ac.manchester.tornado.runtime.graal.backend;

import tornado.graal.compiler.core.common.CompilationIdentifier;
import tornado.graal.compiler.core.common.alloc.RegisterAllocationConfig;
import tornado.graal.compiler.core.target.Backend;
import tornado.graal.compiler.lir.LIR;
import tornado.graal.compiler.lir.asm.CompilationResultBuilder;
import tornado.graal.compiler.lir.framemap.FrameMapBuilder;
import tornado.graal.compiler.lir.gen.LIRGenerationResult;
import tornado.graal.compiler.lir.gen.LIRGeneratorTool;
import tornado.graal.compiler.nodes.StructuredGraph;
import tornado.graal.compiler.nodes.spi.NodeLIRBuilderTool;
import tornado.graal.compiler.phases.util.Providers;

import jdk.vm.ci.code.RegisterConfig;
import jdk.vm.ci.meta.ResolvedJavaMethod;
import uk.ac.manchester.tornado.api.TornadoDeviceContext;
import uk.ac.manchester.tornado.api.profiler.TornadoProfiler;
import uk.ac.manchester.tornado.runtime.graal.compiler.TornadoSuitesProvider;

public abstract class XPUBackend<P extends Providers> extends Backend {

    protected XPUBackend(Providers providers) {
        super(providers);
    }

    @Override
    public Providers getProviders() {
        return super.getProviders();
    }

    @Override
    public RegisterAllocationConfig newRegisterAllocationConfig(RegisterConfig registerConfig, String[] allocationRestrictedTo) {
        return new RegisterAllocationConfig(registerConfig, allocationRestrictedTo);
    }

    public abstract String decodeDeopt(long value);

    public abstract TornadoSuitesProvider getTornadoSuites();

    public abstract TornadoDeviceContext getDeviceContext();

    public abstract boolean isInitialised();

    public abstract void init();

    public abstract int getMethodIndex();

    public abstract void allocateTornadoVMBuffersOnDevice();

    public abstract FrameMapBuilder newFrameMapBuilder(RegisterConfig registerConfig);

    public abstract LIRGenerationResult newLIRGenerationResult(CompilationIdentifier identifier, LIR lir, FrameMapBuilder frameMapBuilder, RegisterAllocationConfig registerAllocationConfig);

    public abstract NodeLIRBuilderTool newNodeLIRBuilder(StructuredGraph graph, LIRGeneratorTool lirGen);

    public abstract LIRGeneratorTool newLIRGenerator(LIRGenerationResult lirGenRes);

    public abstract void emitCode(CompilationResultBuilder crb, LIR lir, ResolvedJavaMethod method, TornadoProfiler profiler);

}
