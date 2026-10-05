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
package uk.ac.manchester.tornado.drivers.cuda.graal.compiler;

import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.List;
import java.util.Set;

import tornado.graal.compiler.code.CompilationResult;

import jdk.vm.ci.meta.ResolvedJavaMethod;
import uk.ac.manchester.tornado.drivers.cuda.graal.backend.CUDABackend;
import uk.ac.manchester.tornado.runtime.tasks.meta.TaskDataContext;

public class CUDACompilationResult extends CompilationResult {

    private Set<ResolvedJavaMethod> nonInlinedMethods;

    private Set<ResolvedJavaMethod> deviceLaunchedKernels = Set.of();
    private TaskDataContext meta;
    private CUDABackend backend;
    private String id;

    public CUDACompilationResult(String id, String name, TaskDataContext meta, CUDABackend backend) {
        super(name);
        this.id = id;
        this.meta = meta;
        this.backend = backend;
    }

    public Set<ResolvedJavaMethod> getNonInlinedMethods() {
        return nonInlinedMethods;
    }

    public void setNonInlinedMethods(Set<ResolvedJavaMethod> value) {
        nonInlinedMethods = value;
    }

    private boolean deviceLaunched;

    /** A launch from device code: the kernel launched and the stream mode it was launched into. */
    public record DeviceLaunch(ResolvedJavaMethod kernel, String mode) {
    }

    private List<DeviceLaunch> deviceLaunches = List.of();
    private List<String> deviceLaunchTree = List.of();

    /** The launches this code makes from the device, in program order. */
    public List<DeviceLaunch> getDeviceLaunches() {
        return deviceLaunches;
    }

    public void setDeviceLaunches(List<DeviceLaunch> value) {
        deviceLaunches = value;
    }

    /** The whole tree of device launches below the task's kernel, for --printBytecodes. */
    public List<String> getDeviceLaunchTree() {
        return deviceLaunchTree;
    }

    public void setDeviceLaunchTree(List<String> value) {
        deviceLaunchTree = value;
    }

    /** Whether this is a kernel launched from device code rather than by the host. */
    public boolean isDeviceLaunched() {
        return deviceLaunched;
    }

    public void setDeviceLaunched(boolean value) {
        deviceLaunched = value;
    }

    /** Kernels this code launches from the device (CUDA Dynamic Parallelism). */
    public Set<ResolvedJavaMethod> getDeviceLaunchedKernels() {
        return deviceLaunchedKernels;
    }

    public void setDeviceLaunchedKernels(Set<ResolvedJavaMethod> value) {
        deviceLaunchedKernels = value;
    }

    public void addCompiledMethodCode(byte[] code) {
        final byte[] oldCode = getTargetCode();
        final int size = oldCode.length + code.length + 1;

        final byte[] newCode = new byte[size];
        Arrays.fill(newCode, (byte) 0);

        final ByteBuffer buffer = ByteBuffer.wrap(newCode);
        buffer.put(code);
        buffer.put((byte) '\n');
        buffer.put(oldCode);
        setTargetCode(newCode, size);
    }

    public TaskDataContext getMeta() {
        return meta;
    }

    // FIXME <REFACTOR> This method can be removed
    public CUDABackend getBackend() {
        return backend;
    }

    public String getId() {
        return id;
    }
}
