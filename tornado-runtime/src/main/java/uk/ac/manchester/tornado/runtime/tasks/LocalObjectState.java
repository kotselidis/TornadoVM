/*
 * This file is part of Tornado: A heterogeneous programming framework:
 * https://github.com/beehive-lab/tornadovm
 *
 * Copyright (c) 2013-2020, 2024, APT Group, Department of Computer Science,
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

import uk.ac.manchester.tornado.api.common.Event;
import uk.ac.manchester.tornado.api.common.TornadoDevice;
import uk.ac.manchester.tornado.runtime.common.XPUDeviceBufferState;

/**
 * Data structure to keep the state for each parameter used by the TornadoVM runtime.
 * The Local Object states identifies if a parameter is used for stream-in, stream-out,
 * or it needs to be forced to stream-in (for example due to a device migration, or a
 * task-graph rewrite for reductions).
 */
public class LocalObjectState {

    /**
     * Identifies a variable (or parameter) is used for
     * stream-in (host -> device).
     */
    private boolean streamIn;

    /**
     * Identifies a variable (or parameter) must be copy-in again
     * from the host to the device.
     */
    private boolean forceStreamIn;

    /**
     * Identifies a variable (or parameter) is used for
     * stream-out (device -> host).
     */
    private boolean streamOut;

    private boolean underDemand;

    private boolean onDevice;

    /**
     * For each variable, we need to keep track of all devices in which there is a shadow
     * copy. This is achieved by using the {@link DataObjectState} object.
     */
    private DataObjectState dataObjectState;

    private Object object;

    public LocalObjectState(Object object) {
        this.object = object;
        dataObjectState = new DataObjectState();
        streamIn = false;
        streamOut = false;
        underDemand = false;
        onDevice = false;
    }

    public Object getObject() {
        return object;
    }

    public boolean isStreamIn() {
        return streamIn;
    }

    public boolean isUnderDemand() {
        return underDemand;
    }

    public void enableUnderDemand() {
        this.underDemand = true;
    }

    public void setOnDevice(boolean isOnDevice) {
        this.onDevice = isOnDevice;
    }


    public boolean isOnDevice() {
        return onDevice;
    }

    public void setStreamIn(boolean streamIn) {
        this.streamIn = streamIn;
    }

    public void setForceStreamIn(boolean streamIn) {
        this.forceStreamIn = streamIn;
    }

    public boolean isForcedStreamIn() {
        return this.forceStreamIn;
    }

    public boolean isStreamOut() {
        return streamOut;
    }

    public void setStreamOut(boolean streamOut) {
        this.streamOut = streamOut;
    }

    public DataObjectState getDataObjectState() {
        return dataObjectState;
    }

    public Event sync(long executionPlanId, Object object, TornadoDevice device) {
        XPUDeviceBufferState deviceState = dataObjectState.getDeviceBufferState(device);
        if (deviceState.isLockedBuffer()) {
            int eventId = device.streamOutBlocking(executionPlanId, object, 0, deviceState, null);
            return device.resolveEvent(executionPlanId, eventId);
        }
        return null;
    }

    @Override
    public LocalObjectState clone() {
        LocalObjectState newLocalObjectState = new LocalObjectState(this.object);
        newLocalObjectState.streamIn = this.streamIn;
        newLocalObjectState.streamOut = this.streamOut;
        newLocalObjectState.forceStreamIn = this.forceStreamIn;
        newLocalObjectState.underDemand = this.underDemand;
        newLocalObjectState.dataObjectState = dataObjectState.clone();
        newLocalObjectState.onDevice = this.onDevice;
        return newLocalObjectState;
    }

    @Override
    public String toString() {
        return (streamIn ? "SIN" : "--") + (streamOut ? "SOUT" : "--") + " " + dataObjectState;
    }


}
