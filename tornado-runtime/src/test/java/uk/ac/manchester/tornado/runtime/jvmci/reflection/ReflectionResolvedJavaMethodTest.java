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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

import java.lang.reflect.Constructor;
import java.lang.reflect.Method;

import org.junit.Test;

import jdk.vm.ci.meta.ResolvedJavaMethod;
import jdk.vm.ci.meta.Signature;

/**
 * Pure-JVM (no-GPU) tests for {@link ReflectionResolvedJavaMethod}: name/signature/modifiers, the
 * canonical {@code toString} format the NodePlugin matcher relies on, and universe canonicalization.
 */
public class ReflectionResolvedJavaMethodTest {

    private final ReflectionUniverse universe = new ReflectionUniverse();

    private ResolvedJavaMethod compute() throws NoSuchMethodException {
        Method m = ReflectionTestSample.class.getDeclaredMethod("compute", int.class, int[].class);
        return universe.lookupMethod(m);
    }

    @Test
    public void nameAndModifiers() throws Exception {
        Method reflected = ReflectionTestSample.class.getDeclaredMethod("compute", int.class, int[].class);
        ResolvedJavaMethod method = universe.lookupMethod(reflected);
        assertEquals("compute", method.getName());
        assertEquals(reflected.getModifiers(), method.getModifiers());
        assertFalse(method.isConstructor());
    }

    @Test
    public void signatureShape() throws Exception {
        Signature signature = compute().getSignature();
        assertEquals(2, signature.getParameterCount(false));
        assertEquals(3, signature.getParameterCount(true));
        assertEquals("I", signature.getParameterType(0, null).getName());
        assertEquals("[I", signature.getParameterType(1, null).getName());
        assertEquals("J", signature.getReturnType(null).getName());
    }

    @Test
    public void signatureIsMemoized() throws Exception {
        ResolvedJavaMethod method = compute();
        assertSame("getSignature must be memoized", method.getSignature(), method.getSignature());
    }

    @Test
    public void toStringIsCanonicalPluginMatchingForm() throws Exception {
        String s = compute().toString();
        // The vector/half-float NodePlugins match on this exact shape: holder.name(SimpleParamTypes).
        assertTrue(s, s.startsWith("ReflectionMethod<"));
        assertTrue(s, s.contains(ReflectionTestSample.class.getName() + ".compute("));
        assertTrue(s, s.contains("int"));
        assertTrue(s, s.contains("int[]"));
        assertTrue(s, s.endsWith(")>"));
    }

    @Test
    public void constructorNaming() throws Exception {
        Constructor<?> ctor = ReflectionTestSample.class.getDeclaredConstructor();
        ResolvedJavaMethod method = universe.lookupMethod(ctor);
        assertEquals("<init>", method.getName());
        assertTrue(method.isConstructor());
    }

    @Test
    public void canonicalizationAndEquality() throws Exception {
        ResolvedJavaMethod first = compute();
        ResolvedJavaMethod second = compute();
        // Same Executable must resolve to the SAME instance (InvocationPlugins key by ResolvedJavaMethod).
        assertSame(first, second);
        assertEquals(first, second);
        assertEquals(first.hashCode(), second.hashCode());

        Method other = ReflectionTestSample.class.getDeclaredMethod("compute", int.class, int[].class);
        assertSame(first, universe.lookupMethod(other));

        Constructor<?> ctor = ReflectionTestSample.class.getDeclaredConstructor();
        assertNotSame(first, universe.lookupMethod(ctor));
        assertFalse(first.equals(universe.lookupMethod(ctor)));
    }
}
