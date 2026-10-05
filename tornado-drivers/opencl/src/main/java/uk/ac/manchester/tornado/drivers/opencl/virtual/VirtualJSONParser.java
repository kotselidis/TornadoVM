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
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 */

package uk.ac.manchester.tornado.drivers.opencl.virtual;

import uk.ac.manchester.tornado.api.exceptions.TornadoInternalError;
import uk.ac.manchester.tornado.drivers.opencl.enums.OCLDeviceType;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static uk.ac.manchester.tornado.runtime.common.TornadoOptions.VIRTUAL_DEVICE_FILE;

public class VirtualJSONParser {

    private static final Pattern pattern =  Pattern.compile(" |\",|\"|\t|\\r|]|\\[");

    private enum JsonKey {
        deviceName,
        doubleFPSupport,
        maxWorkItemSizes,
        deviceAddressBits,
        deviceType,
        deviceExtensions,
        availableProcessors;
    }

    public static VirtualDeviceDescriptor getDeviceDescriptor() {
        String json = readVirtualDeviceJson().replace("\r", "");
        HashMap<JsonKey, String> jsonEntries = new HashMap<>();
        for (String line : json.split("\n")) {
            Matcher matcher = pattern.matcher(line);
            String[] keyValue = matcher.replaceAll("").split(":");
            String key = keyValue[0];
            String value = keyValue.length > 1 ? keyValue[1] : null;
            if (value != null) {
                value = value.charAt(value.length() - 1) == ',' ? value.substring(0, value.length() - 1) : value;
                jsonEntries.put(JsonKey.valueOf(key), value);
            }
        }

        String deviceName = (String) getEntryForKey(JsonKey.deviceName, jsonEntries);
        boolean doubleFPSupport = (boolean) getEntryForKey(JsonKey.doubleFPSupport, jsonEntries);
        long[] maxWorkItemSizes = (long[]) getEntryForKey(JsonKey.maxWorkItemSizes, jsonEntries);
        int deviceAddressBits = (int) getEntryForKey(JsonKey.deviceAddressBits, jsonEntries);
        OCLDeviceType deviceType = (OCLDeviceType) getEntryForKey(JsonKey.deviceType, jsonEntries);
        String deviceExtensions = (String) getEntryForKey(JsonKey.deviceExtensions, jsonEntries);
        int availableProcessors = (int) getEntryForKey(JsonKey.availableProcessors, jsonEntries);

        return new VirtualDeviceDescriptor(deviceName, doubleFPSupport, maxWorkItemSizes, deviceAddressBits, deviceType, deviceExtensions, availableProcessors);
    }

    private static Object getEntryForKey(JsonKey jsonKey, Map<JsonKey, String> jsonEntries) {
        switch (jsonKey) {
            case deviceName:
            case deviceExtensions:
                return jsonEntries.get(jsonKey);
            case doubleFPSupport:
                return Boolean.parseBoolean(jsonEntries.get(jsonKey));
            case maxWorkItemSizes:
                long[] values = new long[3];
                String[] numbers = jsonEntries.get(jsonKey).split(",");
                values[0] = Long.parseLong(numbers[0]);
                values[1] = Long.parseLong(numbers[1]);
                values[2] = Long.parseLong(numbers[2]);
                return values;
            case deviceAddressBits:
            case availableProcessors:
                return Integer.parseInt(jsonEntries.get(jsonKey));
            case deviceType:
                return OCLDeviceType.valueOf(jsonEntries.get(jsonKey));
        }
        throw new RuntimeException("Virtual device JSON parser failed ! Unknown json key: " + jsonKey.name());
    }

    private static String readVirtualDeviceJson() {
        Path path = Paths.get(VIRTUAL_DEVICE_FILE);
        TornadoInternalError.guarantee(path.toFile().exists(), "Virtual device file does not exist: %s", VIRTUAL_DEVICE_FILE);

        try {
            byte[] bytes = Files.readAllBytes(path);
            return new String(bytes);
        } catch (IOException e) {
            throw new RuntimeException(String.format("Failed to read from %s", VIRTUAL_DEVICE_FILE), e);
        }
    }
}
