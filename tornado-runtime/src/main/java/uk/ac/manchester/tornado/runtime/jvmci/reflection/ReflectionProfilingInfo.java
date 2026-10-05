/*
 * Copyright (c) 2026, APT Group, Department of Computer Science,
 * The University of Manchester. All rights reserved.
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
package uk.ac.manchester.tornado.runtime.jvmci.reflection;

import jdk.vm.ci.meta.DeoptimizationReason;
import jdk.vm.ci.meta.JavaMethodProfile;
import jdk.vm.ci.meta.JavaTypeProfile;
import jdk.vm.ci.meta.ProfilingInfo;
import jdk.vm.ci.meta.TriState;

/**
 * An "unprofiled" {@link ProfilingInfo} - reports no runtime profile data. This
 * is the JDK-neutral counterpart to HotSpot's profiling info: TornadoVM does
 * not use profile-guided optimisation for kernel compilation (branch/type
 * profiles are absent), so the compiler simply proceeds without speculation.
 */
final class ReflectionProfilingInfo implements ProfilingInfo {

    private final int codeSize;

    ReflectionProfilingInfo(int codeSize) {
        this.codeSize = codeSize;
    }

    @Override
    public int getCodeSize() {
        return codeSize;
    }

    @Override
    public JavaTypeProfile getTypeProfile(int bci) {
        return null;
    }

    @Override
    public JavaMethodProfile getMethodProfile(int bci) {
        return null;
    }

    @Override
    public double getBranchTakenProbability(int bci) {
        return -1D;
    }

    @Override
    public double[] getSwitchProbabilities(int bci) {
        return null;
    }

    @Override
    public TriState getExceptionSeen(int bci) {
        return TriState.UNKNOWN;
    }

    @Override
    public TriState getNullSeen(int bci) {
        return TriState.UNKNOWN;
    }

    @Override
    public int getExecutionCount(int bci) {
        return -1;
    }

    @Override
    public int getDeoptimizationCount(DeoptimizationReason reason) {
        return 0;
    }

    @Override
    public boolean setCompilerIRSize(Class<?> irType, int size) {
        return false;
    }

    @Override
    public int getCompilerIRSize(Class<?> irType) {
        return -1;
    }

    @Override
    public boolean isMature() {
        return false;
    }

    @Override
    public void setMature() {
        // no-op
    }
}
