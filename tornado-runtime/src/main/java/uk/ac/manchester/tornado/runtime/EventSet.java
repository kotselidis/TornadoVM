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
 * , Juan Fumero
 *
 */
package uk.ac.manchester.tornado.runtime;

import java.util.BitSet;

import uk.ac.manchester.tornado.api.common.Event;
import uk.ac.manchester.tornado.api.common.TornadoEvents;
import uk.ac.manchester.tornado.runtime.common.TornadoXPUDevice;

public class EventSet implements TornadoEvents {

    private final TornadoXPUDevice device;
    private final BitSet profiles;
    private int index;
    private Event event;

    private long executionPlanId;

    public EventSet(TornadoXPUDevice device, BitSet profiles, long executionPlanId) {
        this.device = device;
        this.profiles = profiles;
        this.executionPlanId = executionPlanId;
        index = profiles.nextSetBit(0);
        event = device.resolveEvent(executionPlanId, index);
    }

    public int cardinality() {
        return profiles.cardinality();
    }

    public void reset() {
        index = profiles.nextSetBit(0);
    }

    public boolean hasNext() {
        return index != -1;
    }

    public Event next() {
        if (index == -1) {
            return null;
        }
        event = device.resolveEvent(executionPlanId, index);
        index = profiles.nextSetBit(index);
        return event;
    }

    public TornadoXPUDevice getDevice() {
        return device;
    }

    public BitSet getProfiles() {
        return profiles;
    }
}
