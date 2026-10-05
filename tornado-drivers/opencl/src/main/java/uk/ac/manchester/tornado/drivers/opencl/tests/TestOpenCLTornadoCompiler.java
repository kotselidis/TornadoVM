/*
 * This file is part of Tornado: A heterogeneous programming framework:
 * https://github.com/beehive-lab/tornadovm
 *
 * Copyright (c) 2013-2020, APT Group, Department of Computer Science,
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
 *
 */
package uk.ac.manchester.tornado.drivers.opencl.tests;

import uk.ac.manchester.tornado.drivers.opencl.OCLBackendImpl;
import uk.ac.manchester.tornado.drivers.opencl.OCLCodeCache;
import uk.ac.manchester.tornado.drivers.opencl.OCLContext;
import uk.ac.manchester.tornado.drivers.opencl.OCLDeviceContext;
import uk.ac.manchester.tornado.drivers.opencl.OCLPlatform;
import uk.ac.manchester.tornado.drivers.opencl.OpenCL;
import uk.ac.manchester.tornado.drivers.opencl.graal.OCLInstalledCode;
import uk.ac.manchester.tornado.drivers.opencl.graal.backend.OCLBackend;
import uk.ac.manchester.tornado.drivers.opencl.graal.compiler.OCLCompilationResult;
import uk.ac.manchester.tornado.runtime.TornadoCoreRuntime;
import uk.ac.manchester.tornado.runtime.tasks.meta.ScheduleContext;
import uk.ac.manchester.tornado.runtime.tasks.meta.TaskDataContext;

public class TestOpenCLTornadoCompiler {

    // @formatter:off
    private static final String OPENCL_KERNEL = "__kernel void saxpy(__global float *a," + 
            "    __global float *b, __global float *c) { " +
            "      int idx = get_global_id(0); " + 
            "      c[idx]  =  a[idx] + b[idx]; " + 
            "}";
    // @formatter:on

    public static void main(String[] args) {

        OCLPlatform platform = (OCLPlatform) OpenCL.getPlatform(0);
        // Create context for the platform
        OCLContext oclContext = platform.createContext();

        // Create command queue
        oclContext.createCommandQueue(0);

        // 1. Compile the code:
        OCLDeviceContext deviceContext = oclContext.createDeviceContext(0);
        OCLCodeCache codeCache = new OCLCodeCache(deviceContext);

        TornadoCoreRuntime tornadoRuntime = TornadoCoreRuntime.getTornadoRuntime();
        OCLBackend backend = tornadoRuntime.getBackend(OCLBackendImpl.class).getDefaultBackend();
        ScheduleContext scheduleMeta = new ScheduleContext("oclbackend");
        TaskDataContext meta = new TaskDataContext(scheduleMeta, "saxpy");
        new OCLCompilationResult("internal", "saxpy", meta, backend);

        byte[] source = OPENCL_KERNEL.getBytes();
        OCLInstalledCode code = codeCache.installSource(meta, "saxpy", "saxpy", source);

        String generatedSourceCode = code.getGeneratedSourceCode();
        if (meta.isPrintKernelEnabled()) {
            System.out.println("Compiled code: " + generatedSourceCode);
        }
    }
}
