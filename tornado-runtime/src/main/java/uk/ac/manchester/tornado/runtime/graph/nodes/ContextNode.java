/*
 * This file is part of Tornado: A heterogeneous programming framework:
 * https://github.com/beehive-lab/tornadovm
 *
 * Copyright (c) 2013-2020, 2023 APT Group, Department of Computer Science,
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
 */
package uk.ac.manchester.tornado.runtime.graph.nodes;

import uk.ac.manchester.tornado.runtime.common.TornadoXPUDevice;

/**
 * It represents a context node used in a
 * {@link uk.ac.manchester.tornado.runtime.graph.TornadoVMGraphCompiler}.
 */
public class ContextNode extends AbstractNode {

    private int deviceIndex;
    private TornadoXPUDevice device;

    /**
     * It constructs a ContextNode with the given device index and
     * {@link TornadoXPUDevice}.
     *
     * @param index
     *     The index of the device.
     * @param device
     *     The {@link TornadoXPUDevice} associated with this context
     *     node.
     */
    public ContextNode(int index, TornadoXPUDevice device) {
        this.deviceIndex = index;
        this.device = device;
    }

    @Override
    public int compareTo(AbstractNode o) {
        if (!(o instanceof ContextNode)) {
            return -1;
        }

        return Integer.compare(deviceIndex, ((ContextNode) o).deviceIndex);
    }

    /**
     * It gets the device index associated with this context node.
     *
     * @return The device index.
     */
    public int getDeviceIndex() {
        return deviceIndex;
    }

    /**
     * It sets the device index associated with this context node.
     *
     * @param deviceIndex
     *     The device index to set.
     */
    public void setDeviceIndex(int deviceIndex) {
        this.deviceIndex = deviceIndex;
    }

    /**
     * It gets the {@link TornadoXPUDevice} associated with this context
     * node.
     *
     * @return {@link TornadoXPUDevice}
     */
    public TornadoXPUDevice getDevice() {
        return this.device;
    }

    /**
     * It sets the {@link TornadoXPUDevice} associated with this context
     * node.
     *
     * @param device
     *     The {@link TornadoXPUDevice} to set.
     */
    public void setDevice(TornadoXPUDevice device) {
        this.device = device;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();

        sb.append(String.format("[%d]: context device=%d, [ ", id, deviceIndex));
        for (AbstractNode use : uses) {
            sb.append("").append(use.getId()).append(" ");
        }
        sb.append("]");
        return sb.toString();
    }
}
