/*
 * This file is part of Tornado: A heterogeneous programming framework:
 * https://github.com/beehive-lab/tornadovm
 *
 * Copyright (c) 2013-2022, 2024, APT Group, Department of Computer Science,
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
package uk.ac.manchester.tornado.runtime.common;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.HashSet;
import java.util.Properties;
import java.util.Set;

import uk.ac.manchester.tornado.api.TornadoSetting;

public final class Tornado implements TornadoSetting {

    private static final String TORNADOVM_HOME_VARIABLE = "TORNADOVM_HOME";

    private static final Properties settings = System.getProperties();

    static {
        tryLoadSettings();
    }

    private static void setProperty(String key, String value) {
        settings.setProperty(key, value);
    }

    public static String getProperty(String key) {
        return settings.getProperty(key);
    }

    public static String getProperty(String key, String defaultValue) {
        return settings.getProperty(key, defaultValue);
    }

    private static void loadSettings(String filename) {
        final File localSettings = new File(filename);
        Properties loadProperties = new Properties();
        if (localSettings.exists()) {
            try (FileInputStream fileInputStream = new FileInputStream(localSettings)) {
                loadProperties.load(fileInputStream);
            } catch (IOException e) {
                new TornadoLogger().warn("Unable to load settings from %s", localSettings.getAbsolutePath());
            }
        }
        Set<String> localKeys = loadProperties.stringPropertyNames();
        Set<String> systemKeys = settings.stringPropertyNames();
        Set<String> diff = new HashSet<>(localKeys);
        diff.removeAll(systemKeys);
        diff.forEach(key -> settings.setProperty(key, loadProperties.getProperty(key)));
    }

    private static void tryLoadSettings() {
        final String tornadoRoot = System.getenv(TORNADOVM_HOME_VARIABLE);
        loadSettings(tornadoRoot + "/etc/tornado.properties");
    }

    @Override
    public void setTornadoProperty(String key, String value) {
        Tornado.setProperty(key, value);
    }

    @Override
    public String getTornadoProperty(String key) {
        return Tornado.getProperty(key);
    }

    @Override
    public String getTornadoProperty(String key, String defaultValue) {
        return Tornado.getProperty(key, defaultValue);
    }

    @Override
    public void loadTornadoProperty(String filename) {
        loadSettings(filename);
    }

}
