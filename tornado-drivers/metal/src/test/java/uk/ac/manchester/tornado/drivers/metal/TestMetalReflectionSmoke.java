/*
 * This file is part of Tornado: A heterogeneous programming framework:
 * https://github.com/beehive-lab/tornadovm
 *
 * Copyright (c) 2026, APT Group, Department of Computer Science,
 * School of Engineering, The University of Manchester. All rights reserved.
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
package uk.ac.manchester.tornado.drivers.metal;

import static org.junit.Assume.assumeTrue;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import uk.ac.manchester.tornado.drivers.metal.graal.MetalInstalledCode;

public class TestMetalReflectionSmoke {

    private static boolean isMac() {
        String os = System.getProperty("os.name", "").toLowerCase();
        return os.contains("mac");
    }

    private static boolean hasXcrun() {
        try {
            Process p = new ProcessBuilder("xcrun", "--version").start();
            int rc = p.waitFor();
            return rc == 0;
        } catch (Exception e) {
            return false;
        }
    }

    @Test
    public void smokeReflection() throws Exception {
        assumeTrue("macOS required for Metal tests", isMac());
        assumeTrue("xcrun required for Metal toolchain", hasXcrun());

            // trivial Metal kernel: takes 2 buffers and writes result to out[0]
            String msl = "using namespace metal; kernel void add(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* out [[buffer(2)]]) { uint id = static_cast<uint>(get_thread_position_in_grid().x); out[id] = a[id] + b[id]; }";

        // Obtain platform/context/device context using public APIs
        MetalPlatform platform = (MetalPlatform) Metal.getPlatform(0);
        MetalContext context = platform.createContext();
        context.createCommandQueue(0);
        MetalDeviceContext ctx = context.createDeviceContext(0);

        MetalCodeCache cache = new MetalCodeCache(ctx);
        MetalInstalledCode installed = cache.compileAndLink("add", msl);
        assertTrue("installed code should not be null", installed != null);
        MetalKernel k = installed.getKernel();
        assertTrue("kernel must be present", k != null);

        int n = k.getArgCount();
        // Expect at least 3 buffer arguments (a,b,out)
        assertTrue("expected at least 3 args, got " + n, n >= 3);
        MetalKernel.KernelArgInfo info0 = k.getArgInfoObject(0);
        assertTrue("arg0 info should be non-null", info0 != null);
        // Name may be empty depending on reflection granularity; just ensure parsing succeeded
        assertTrue("index must match", info0.index == 0);
    }
}
