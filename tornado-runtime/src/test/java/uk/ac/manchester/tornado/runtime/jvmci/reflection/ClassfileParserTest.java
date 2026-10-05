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
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import java.io.InputStream;

import org.junit.Test;

/**
 * Pure-JVM (no-GPU) tests for {@link ClassfileParser}: recover the {@code Code} attribute and field
 * metadata from a real compiled classfile read via {@code getResourceAsStream}.
 */
public class ClassfileParserTest {

    private static byte[] sampleClassfile() throws Exception {
        String resource = ReflectionTestSample.class.getName().replace('.', '/') + ".class";
        try (InputStream in = ReflectionTestSample.class.getClassLoader().getResourceAsStream(resource)) {
            assertNotNull("sample classfile must be on the test classpath", in);
            return in.readAllBytes();
        }
    }

    @Test
    public void recoversCodeAttribute() throws Exception {
        ClassfileParser.MethodCode code = ClassfileParser.parse(sampleClassfile(), "compute", "(I[I)J");
        assertNotNull(code);
        assertTrue("compute must have bytecode", code.code().length > 0);
        assertTrue("compute must reserve locals", code.maxLocals() > 0);
        assertTrue("compute must reserve stack", code.maxStack() > 0);
    }

    @Test
    public void findsFieldInfo() throws Exception {
        byte[] bytes = sampleClassfile();
        ClassfileParser.FieldInfo alpha = ClassfileParser.findFieldInfo(bytes, "alpha");
        assertNotNull(alpha);
        assertEquals("I", alpha.descriptor());

        ClassfileParser.FieldInfo beta = ClassfileParser.findFieldInfo(bytes, "beta");
        assertNotNull(beta);
        assertEquals("J", beta.descriptor());
    }

    @Test
    public void missingFieldReturnsNull() throws Exception {
        assertNull(ClassfileParser.findFieldInfo(sampleClassfile(), "doesNotExist"));
    }
}
