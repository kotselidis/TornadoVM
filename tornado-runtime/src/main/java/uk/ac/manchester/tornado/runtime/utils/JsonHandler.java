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
 */
package uk.ac.manchester.tornado.runtime.utils;

import java.util.Map;

import uk.ac.manchester.tornado.api.TornadoDeviceContext;
import uk.ac.manchester.tornado.api.enums.TornadoVMBackendType;
import uk.ac.manchester.tornado.api.profiler.ProfilerType;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;
import uk.ac.manchester.tornado.runtime.common.RuntimeUtilities;
import uk.ac.manchester.tornado.runtime.common.TornadoOptions;

public class JsonHandler {

    private static final String SEPARATOR = "\":  \"";
    private static final String END_LINE = "\",\n";
    private static final String DEVICE_ID = "Device ID";
    private static final String DEVICE = "Device";
    private static final String IP = "IP";
    private StringBuilder indent;

    private void increaseIndent() {
        indent.append("    ");
    }

    private void decreaseIndent() {
        indent.delete(0, 4);
    }

    public String createJSon(Map<String, Integer> entry, String name, TornadoDeviceContext device) {
        indent = new StringBuilder();
        StringBuilder json = new StringBuilder("");
        json.append("{\n");
        increaseIndent();
        json.append(indent.toString() + "\"" + name + "\": { \n");
        increaseIndent();
        if (TornadoOptions.LOG_IP) {
            json.append(indent.toString() + "\"" + IP + "\"" + ": " + "\"" + RuntimeUtilities.getTornadoInstanceIP() + END_LINE);
        }
        TornadoVMBackendType backendType = TornadoRuntimeProvider.getTornadoRuntime().getBackend(device.getDriverIndex()).getBackendType();
        json.append(indent.toString() + "\"" + ProfilerType.BACKEND + "\" : \"" + backendType + END_LINE);
        json.append(indent.toString() + "\"" + DEVICE_ID + SEPARATOR + device.getDriverIndex() + ":" + device.getDevicePlatform() + END_LINE);
        json.append(indent.toString() + "\"" + DEVICE + SEPARATOR + device.getDeviceName() + END_LINE);
        for (String s : entry.keySet()) {
            json.append(indent.toString() + "\"" + s + SEPARATOR + entry.get(s) + END_LINE);
        }
        json.delete(json.length() - 2, json.length() - 1); // remove last comma
        decreaseIndent();
        json.append(indent.toString() + "}\n");
        decreaseIndent();
        json.append(indent.toString() + "}\n");
        return json.toString();
    }
}
