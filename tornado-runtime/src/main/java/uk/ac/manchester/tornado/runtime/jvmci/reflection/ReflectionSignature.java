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

import java.lang.reflect.Constructor;
import java.lang.reflect.Executable;
import java.lang.reflect.Method;

import jdk.vm.ci.meta.JavaType;
import jdk.vm.ci.meta.ResolvedJavaType;
import jdk.vm.ci.meta.Signature;

/**
 * Reflection-backed {@link Signature}: parameter and return types derived from
 * a {@link java.lang.reflect.Executable} rather than from a parsed method
 * descriptor obtained through HotSpot JVMCI.
 */
final class ReflectionSignature implements Signature {

    private final ReflectionUniverse universe;
    private final Class<?>[] parameterTypes;
    private final Class<?> returnType;

    ReflectionSignature(ReflectionUniverse universe, Executable executable) {
        this.universe = universe;
        this.parameterTypes = executable.getParameterTypes();
        this.returnType = (executable instanceof Method m) ? m.getReturnType() : void.class;
        assert executable instanceof Method || executable instanceof Constructor;
    }

    @Override
    public int getParameterCount(boolean withReceiver) {
        return parameterTypes.length + (withReceiver ? 1 : 0);
    }

    @Override
    public JavaType getParameterType(int index, ResolvedJavaType accessingClass) {
        return universe.lookupType(parameterTypes[index]);
    }

    @Override
    public JavaType getReturnType(ResolvedJavaType accessingClass) {
        return universe.lookupType(returnType);
    }
}
