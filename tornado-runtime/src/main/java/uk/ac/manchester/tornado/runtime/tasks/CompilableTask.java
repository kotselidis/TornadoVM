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
 */
package uk.ac.manchester.tornado.runtime.tasks;

import java.lang.reflect.Method;
import java.util.Objects;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.common.Access;
import uk.ac.manchester.tornado.api.common.SchedulableTask;
import uk.ac.manchester.tornado.api.common.TornadoDevice;
import uk.ac.manchester.tornado.api.profiler.TornadoProfiler;
import uk.ac.manchester.tornado.runtime.common.TornadoXPUDevice;
import uk.ac.manchester.tornado.runtime.tasks.meta.ScheduleContext;
import uk.ac.manchester.tornado.runtime.tasks.meta.TaskDataContext;

public class CompilableTask implements SchedulableTask {

    protected final Object[] args;
    protected final Method method;
    private final Object[] resolvedArgs;
    protected TaskDataContext meta;
    protected boolean shouldCompile;
    private long batchNumThreads;
    private int batchNumber;
    private long batchSize;

    private TornadoProfiler profiler;
    private boolean forceCompiler;

    public CompilableTask(ScheduleContext meta, String id, Method method, Object... args) {
        this.method = method;
        this.args = args;
        this.shouldCompile = true;
        this.resolvedArgs = args;
        this.meta = TaskDataContext.create(meta, id, method);
    }

    @Override
    public String toString() {
        final StringBuilder buffer = new StringBuilder();
        buffer.append("task: ").append(meta.getId()).append(" ").append(method.getName()).append("()\n");
        Access[] argumentsAccess = meta.getArgumentsAccess();
        for (int i = 0; i < args.length; i++) {
            buffer.append(String.format("arg  : [%s] %s -> %s%n", argumentsAccess[i], args[i], resolvedArgs[i]));
        }
        buffer.append("meta : ").append(meta.toString());
        return buffer.toString();
    }

    @Override
    public Object[] getArguments() {
        return resolvedArgs;
    }

    @Override
    public Access[] getArgumentsAccess() {
        return meta.getArgumentsAccess();
    }

    @Override
    public TornadoXPUDevice getDevice() {
        return meta.getXPUDevice();
    }

    @Override
    public String getFullName() {
        return "task " + meta.getId() + " - " + method.getName();
    }

    @Override
    public String getNormalizedName() {
        return meta.getId() + "." + method.getName();
    }

    @Override
    public String getTaskName() {
        return method.getName();
    }

    @Override
    public void setDevice(final TornadoDevice device) {
        meta.setDevice(device);
    }

    @Override
    public TaskDataContext meta() {
        return meta;
    }

    public Method getMethod() {
        return method;
    }

    @Override
    public String getId() {
        return meta.getId();
    }

    @Override
    public boolean equals(Object obj) {
        if (obj instanceof CompilableTask) {
            CompilableTask other = (CompilableTask) obj;
            return getId().equals(other.getId());
        }
        return false;
    }

    @Override
    public int hashCode() {
        int hash = 7;
        hash = 71 * hash + Objects.hashCode(getId());
        hash = 71 * hash + Objects.hashCode(this.method);
        return hash;
    }

    @Override
    public long getBatchThreads() {
        return batchNumThreads;
    }

    @Override
    public void setBatchThreads(long batchThreads) {
        this.batchNumThreads = batchThreads;
    }

    @Override
    public void setBatchNumber(int batchNumber) {
        this.batchNumber = batchNumber;
    }

    @Override
    public int getBatchNumber() {
        return this.batchNumber;
    }

    @Override
    public void setBatchSize(long batchSize) {
        this.batchSize = batchSize;
    }

    @Override
    public long getBatchSize() {
        return this.batchSize;
    }

    @Override
    public void attachProfiler(TornadoProfiler tornadoProfiler) {
        this.profiler = tornadoProfiler;
    }

    @Override
    public TornadoProfiler getProfiler() {
        return this.profiler;
    }

    @Override
    public void forceCompilation() {
        forceCompiler = true;
    }

    @Override
    public boolean shouldCompile() {
        return forceCompiler;
    }

    @Override
    public void enableDefaultThreadScheduler(boolean useDefaultScheduler) {
        meta.enableDefaultThreadScheduler(useDefaultScheduler);
    }

    @Override
    public void setUseGridScheduler(boolean use) {
        meta.setUseGridScheduler(use);
    }

    @Override
    public void setGridScheduler(GridScheduler gridScheduler) {
        meta.setGridScheduler(gridScheduler);
    }

    @Override
    public boolean isGridSchedulerEnabled() {
        return meta.isGridSchedulerEnabled();
    }

}
