/*
 * This file is part of Tornado: A heterogeneous programming framework:
 * https://github.com/beehive-lab/tornadovm
 *
 * Copyright (c) 2013-2020, 2023, APT Group, Department of Computer Science,
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

import java.util.logging.Level;
import java.util.logging.Logger;

public class TornadoLogger {

    private final Logger logger;
    private final boolean isLogOptionEnabled = TornadoOptions.DEBUG || TornadoOptions.FULL_DEBUG;

    public TornadoLogger(Class<?> clazz) {
        if (clazz == null) {
            logger = Logger.getAnonymousLogger();
        } else {
            logger = Logger.getLogger(clazz.getName());
        }
    }

    public TornadoLogger() {
        this(null);
    }

    public void debug(final String msg) {
        if (isLogOptionEnabled) {
            logger.setLevel(Level.INFO);
            logger.info(msg);
        }
    }

    public void debug(final String pattern, final Object... args) {
        if (isLogOptionEnabled) {
            debug(String.format(pattern, args));
        }
    }

    public void error(final String msg) {
        if (isLogOptionEnabled) {
            logger.setLevel(Level.SEVERE);
            logger.severe(msg);
        }
    }

    public void error(final String pattern, final Object... args) {
        if (isLogOptionEnabled) {
            error(String.format(pattern, args));
        }
    }

    public void fatal(final String msg) {
        if (isLogOptionEnabled) {
            logger.setLevel(Level.SEVERE);
            logger.severe(msg);
        }
    }

    public void fatal(final String pattern, final Object... args) {
        if (isLogOptionEnabled) {
            fatal(String.format(pattern, args));
        }
    }

    public void info(final String msg) {
        if (isLogOptionEnabled) {
            logger.setLevel(Level.INFO);
            logger.info(msg);
        }
    }

    public void info(final String pattern, final Object... args) {
        if (isLogOptionEnabled) {
            info(String.format(pattern, args));
        }
    }

    public void trace(final String msg) {
        if (isLogOptionEnabled) {
            logger.setLevel(Level.INFO);
            logger.info(msg);
        }
    }

    public void trace(final String pattern, final Object... args) {
        if (isLogOptionEnabled) {
            trace(String.format(pattern, args));
        }
    }

    public void warn(final String msg) {
        if (isLogOptionEnabled) {
            logger.setLevel(Level.WARNING);
            logger.warning(msg);
        }
    }

    public void warn(final String msg, final Object... args) {
        if (isLogOptionEnabled) {
            trace(String.format(msg, args));
        }
    }
}
