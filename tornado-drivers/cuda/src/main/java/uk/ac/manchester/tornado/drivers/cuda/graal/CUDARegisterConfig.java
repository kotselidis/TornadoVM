/*
 * Copyright (c) 2018, 2020, 2026, APT Group, Department of Computer Science,
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
package uk.ac.manchester.tornado.drivers.cuda.graal;

import static uk.ac.manchester.tornado.api.exceptions.TornadoInternalError.unimplemented;

import jdk.vm.ci.code.CallingConvention;
import jdk.vm.ci.code.CallingConvention.Type;
import jdk.vm.ci.code.Register;
import jdk.vm.ci.code.RegisterArray;
import jdk.vm.ci.code.RegisterAttributes;
import jdk.vm.ci.code.RegisterConfig;
import jdk.vm.ci.code.ValueKindFactory;
import jdk.vm.ci.meta.JavaKind;
import jdk.vm.ci.meta.JavaType;
import jdk.vm.ci.meta.PlatformKind;

/**
 * Register configuration for kernels. Kernel code has no machine registers: values live in variables of
 * the generated source, and the device compiler allocates registers. Graal still asks for a frame
 * register and for register sets, so this returns a single placeholder register and empty sets, and
 * reports the queries the backend never makes as unimplemented.
 */
public class CUDARegisterConfig implements RegisterConfig {

    private static final Register FRAME = new Register(0, 0, "dummy", CUDAArchitecture.CUDA_ABI);
    private static final RegisterArray NONE = new RegisterArray(new Register[0]);

    private static <T> T notUsedByKernels(String query) {
        unimplemented(query + " is not used when compiling kernels.");
        return null;
    }

    @Override
    public Register getFrameRegister() {
        return FRAME;
    }

    @Override
    public RegisterArray getAllocatableRegisters() {
        return NONE;
    }

    @Override
    public RegisterArray getCallerSaveRegisters() {
        return NONE;
    }

    @Override
    public RegisterArray getCalleeSaveRegisters() {
        return NONE;
    }

    @Override
    public RegisterArray getCallingConventionRegisters(Type type, JavaKind kind) {
        return NONE;
    }

    @Override
    public CallingConvention getCallingConvention(Type type, JavaType returnType, JavaType[] parameterTypes, ValueKindFactory<?> valueKindFactory) {
        return notUsedByKernels("The calling convention");
    }

    @Override
    public Register getReturnRegister(JavaKind kind) {
        return notUsedByKernels("The return register");
    }

    @Override
    public RegisterArray filterAllocatableRegisters(PlatformKind kind, RegisterArray registers) {
        return notUsedByKernels("Filtering allocatable registers");
    }

    @Override
    public RegisterAttributes[] getAttributesMap() {
        return notUsedByKernels("The register attributes map");
    }

    @Override
    public boolean areAllAllocatableRegistersCallerSaved() {
        unimplemented("Caller-saved allocatable registers are not used when compiling kernels.");
        return false;
    }
}
