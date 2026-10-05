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
package uk.ac.manchester.tornado.runtime.graal.phases;

import tornado.graal.compiler.phases.OptimisticOptimizations;
import tornado.graal.compiler.phases.PhaseSuite;
import tornado.graal.compiler.phases.tiers.HighTierContext;
import tornado.graal.compiler.phases.util.Providers;

import jdk.vm.ci.meta.ResolvedJavaMethod;
import uk.ac.manchester.tornado.api.common.Access;
import uk.ac.manchester.tornado.api.common.TornadoDevice;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;
import uk.ac.manchester.tornado.runtime.graal.phases.sketcher.TornadoDataflowAnalysis;


public class TornadoSketchTierContext extends HighTierContext {

    private final ResolvedJavaMethod method;

    /**
     * Contains the argument accesses of the {@link #method}. The array gets populated in the {@link TornadoDataflowAnalysis} phase.
     * It includes accesses of arguments passed to non-inlined callees of the {@link #method}.
     */
    private final Access[] argumentAccess;
    private boolean batchWriteThreadIndex;

    private TornadoDevice device;

    public TornadoSketchTierContext(Providers providers, PhaseSuite<HighTierContext> graphBuilderSuite, OptimisticOptimizations optimisticOpts, ResolvedJavaMethod method, int backendIndex,
            int deviceIndex) {
        super(providers, graphBuilderSuite, optimisticOpts);
        this.method = method;
        int parameterCount = method.getParameters().length;
        this.argumentAccess = new Access[method.isStatic() ? parameterCount : parameterCount + 1];
        device = TornadoRuntimeProvider.getTornadoRuntime().getBackend(backendIndex).getDevice(deviceIndex);
    }

    public TornadoDevice getDevice() {
        return device;
    }

    public ResolvedJavaMethod getMethod() {
        return method;
    }

    public Access[] getAccesses() {
        return argumentAccess;
    }

    public void setBatchWriteThreadIndex() {
        this.batchWriteThreadIndex = true;
    }

    public boolean getBatchWriteThreadIndex() {
        return this.batchWriteThreadIndex;
    }
}
