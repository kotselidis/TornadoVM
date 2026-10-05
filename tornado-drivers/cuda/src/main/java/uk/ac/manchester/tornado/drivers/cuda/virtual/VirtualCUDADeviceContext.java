/*
 * This file is part of Tornado: A heterogeneous programming framework:
 * https://github.com/beehive-lab/tornadovm
 *
 * Copyright (c) 2020, APT Group, Department of Computer Science,
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
package uk.ac.manchester.tornado.drivers.cuda.virtual;

import java.util.Set;

import uk.ac.manchester.tornado.api.common.Event;
import uk.ac.manchester.tornado.api.common.SchedulableTask;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;
import uk.ac.manchester.tornado.drivers.common.TornadoBufferProvider;
import uk.ac.manchester.tornado.drivers.cuda.CUDABackendImpl;
import uk.ac.manchester.tornado.drivers.cuda.CUDACodeCache;
import uk.ac.manchester.tornado.drivers.cuda.CUDADeviceContextInterface;
import uk.ac.manchester.tornado.drivers.cuda.CUDAProgram;
import uk.ac.manchester.tornado.drivers.cuda.CUDATargetDevice;
import uk.ac.manchester.tornado.drivers.cuda.enums.CUDADeviceType;
import uk.ac.manchester.tornado.drivers.cuda.graal.CUDAInstalledCode;
import uk.ac.manchester.tornado.drivers.cuda.graal.compiler.CUDACompilationResult;
import uk.ac.manchester.tornado.drivers.cuda.mm.CUDAMemoryManager;
import uk.ac.manchester.tornado.runtime.EmptyEvent;
import uk.ac.manchester.tornado.runtime.tasks.meta.TaskDataContext;

public class VirtualCUDADeviceContext implements CUDADeviceContextInterface {

    private final CUDATargetDevice device;
    private final VirtualCUDAContext context;
    private final CUDACodeCache codeCache;
    private boolean wasReset;

    protected VirtualCUDADeviceContext(CUDATargetDevice device, VirtualCUDAContext context) {
        this.device = device;
        this.context = context;
        this.codeCache = new CUDACodeCache(this);
        device.setDeviceContext(this);
    }

    public CUDATargetDevice getDevice() {
        return device;
    }

    @Override
    public String getDeviceName() {
        return String.format(device.getDeviceName());
    }

    @Override
    public int getDriverIndex() {
        return TornadoRuntimeProvider.getTornadoRuntime().getBackendIndex(CUDABackendImpl.class);
    }

    @Override
    public Set<Long> getRegisteredPlanIds() {
        return Set.of();
    }

    @Override
    public String toString() {
        return getClass().getName();
    }

    public VirtualCUDAContext getPlatformContext() {
        return context;
    }

    @Override
    public long getDeviceId() {
        return 0;
    }

    @Override
    public CUDAProgram createProgramWithSource(byte[] source, long[] lengths) {
        return null;
    }

    @Override
    public CUDAProgram createProgramWithBinary(byte[] binary, long[] lengths) {
        return null;
    }

    @Override
    public CUDAProgram createProgramWithIL(byte[] binary, long[] lengths) {
        return null;
    }

    @Override
    public CUDAMemoryManager getMemoryManager() {
        return null;
    }

    @Override
    public TornadoBufferProvider getBufferProvider() {
        return null;
    }

    @Override
    public void sync(long executionPlanId) {
    }

    @Override
    public int enqueueBarrier(long executionPlanId) {
        return 0;
    }

    @Override
    public int enqueueBarrier(long executionPlanId, int[] events) {
        return 0;
    }

    @Override
    public int enqueueMarker(long executionPlanId) {
        return 0;
    }

    @Override
    public int enqueueMarker(long executionPlanId, int[] events) {
        return 0;
    }

    @Override
    public Event resolveEvent(long executionPlanId, int event) {
        return new EmptyEvent();
    }

    @Override
    public void flushEvents(long executionPlanId) {
    }

    @Override
    public void reset(long executionPlanId) {
        wasReset = true;
    }

    @Override
    public VirtualCUDATornadoDevice toDevice() {
        return new VirtualCUDATornadoDevice(context.getPlatformIndex(), device.getIndex());
    }

    public String getId() {
        return String.format("opencl-%d-%d", context.getPlatformIndex(), device.getIndex());
    }

    @Override
    public void dumpEvents() {
    }

    @Override
    public void flush(long executionPlanId) {
    }

    @Override
    public boolean wasReset() {
        return wasReset;
    }

    @Override
    public void setResetToFalse() {
        wasReset = false;
    }

    @Override
    public boolean isFP64Supported() {
        return device.isDeviceDoubleFPSupported();
    }

    @Override
    public int getDeviceIndex() {
        return device.getIndex();
    }

    @Override
    public int getDevicePlatform() {
        return context.getPlatformIndex();
    }

    @Override
    public boolean isKernelAvailable(long executionPlanId) {
        return true;
    }

    @Override
    public CUDAInstalledCode installCode(long executionPlanId, CUDACompilationResult result) {
        return null;
    }

    @Override
    public CUDAInstalledCode installCode(long executionPlanId, TaskDataContext meta, String id, String entryPoint, byte[] code) {
        return null;
    }

    @Override
    public boolean isCached(long executionPlanId, String id, String entryPoint) {
        return false;
    }

    @Override
    public CUDAInstalledCode getInstalledCode(long executionPlanId, String id, String entryPoint) {
        return null;
    }

    @Override
    public CUDACodeCache getCodeCache(long executionPlanId) {
        return codeCache;
    }

    @Override
    public boolean isCached(long executionPlanId, String methodName, SchedulableTask task) {
        return false;
    }

}
