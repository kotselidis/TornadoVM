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
package uk.ac.manchester.tornado.drivers.cuda.graal;

import static uk.ac.manchester.tornado.api.exceptions.TornadoInternalError.unimplemented;

import jdk.vm.ci.code.CodeCacheProvider;
import jdk.vm.ci.code.CompiledCode;
import jdk.vm.ci.code.InstalledCode;
import jdk.vm.ci.code.RegisterConfig;
import jdk.vm.ci.code.TargetDescription;
import jdk.vm.ci.meta.ResolvedJavaMethod;
import jdk.vm.ci.meta.SpeculationLog;
import uk.ac.manchester.tornado.drivers.cuda.CUDATargetDescription;
import uk.ac.manchester.tornado.runtime.common.TornadoLogger;

public class CUDACodeProvider implements CodeCacheProvider {

    private final TargetDescription target;

    public CUDACodeProvider(TargetDescription target) {
        this.target = target;
    }

    @Override
    public SpeculationLog createSpeculationLog() {
        return null;
    }

    @Override
    public long getMaxCallTargetOffset(long l) {
        unimplemented("Max call target offset not implemented yet.");
        return -1;
    }

    @Override
    public int getMinimumOutgoingSize() {
        return 0;
    }

    @Override
    public RegisterConfig getRegisterConfig() {
        return new CUDARegisterConfig();
    }

    @Override
    public CUDATargetDescription getTarget() {
        return (CUDATargetDescription) target;
    }

    @Override
    public InstalledCode installCode(ResolvedJavaMethod rjm, CompiledCode cc, InstalledCode ic, SpeculationLog sl, boolean bln) {
        unimplemented("waiting for CompiledCode to be implemented first");
        return null;
    }

    @Override
    public void invalidateInstalledCode(InstalledCode ic) {
        ic.invalidate();
    }

    @Override
    public boolean shouldDebugNonSafepoints() {
        new TornadoLogger().warn("Debug non safe points not implemented yet.");
        return false;
    }

}
