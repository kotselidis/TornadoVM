/*
 * This file is part of Tornado: A heterogeneous programming framework:
 * https://github.com/beehive-lab/tornadovm
 *
 * Copyright (c) 2020, APT Group, Department of Computer Science,
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
package uk.ac.manchester.tornado.runtime.profiler;

import uk.ac.manchester.tornado.api.profiler.ProfilerType;
import uk.ac.manchester.tornado.api.profiler.TornadoProfiler;

public class EmptyProfiler implements TornadoProfiler {

    public EmptyProfiler() {

    }

    @Override
    public synchronized void addValueToMetric(ProfilerType type, String taskName, long value) {
    }

    @Override
    public synchronized void start(ProfilerType type) {
    }

    @Override
    public synchronized void start(ProfilerType type, String taskName) {
    }

    @Override
    public synchronized void registerDeviceName(String taskName, String deviceInfo) {

    }

    @Override
    public synchronized void registerBackend(String taskName, String backend) {

    }

    @Override
    public synchronized void registerDeviceID(String taskName, String deviceID) {
    }

    @Override
    public synchronized void registerMethodHandle(ProfilerType type, String taskName, String methodName) {
    }

    @Override
    public synchronized void stop(ProfilerType type) {
    }

    @Override
    public synchronized void stop(ProfilerType type, String taskName) {
    }

    @Override
    public synchronized long getTimer(ProfilerType type) {
        System.out.println("Enable the profiler with: -Dtornado.profiler=True");
        return 0;
    }

    @Override
    public long getSize(ProfilerType type) {
        return 0;
    }

    @Override
    public synchronized long getTaskTimer(ProfilerType type, String taskName) {
        return 0;
    }

    @Override
    public synchronized void setTimer(ProfilerType type, long time) {

    }

    @Override
    public synchronized void dump() {
    }

    @Override
    public synchronized String createJson(StringBuilder json, String sectionName) {
        return null;
    }

    @Override
    public synchronized void dumpJson(StringBuilder stringBuffer, String id) {
    }

    @Override
    public synchronized void clean() {
    }

    @Override
    public synchronized void setTaskTimer(ProfilerType type, String taskId, long timer) {
    }

    @Override
    public void setTaskPowerUsage(ProfilerType type, String taskId, long power) {
    }

    @Override
    public void setSystemPowerConsumption(ProfilerType systemPowerConsumptionType, String taskId, long powerConsumption) {
    }

    @Override
    public void setSystemVoltage(ProfilerType systemPowerVoltageType, String taskId, long voltage) {
    }

    @Override
    public synchronized void sum(ProfilerType type, long sum) {

    }

}
