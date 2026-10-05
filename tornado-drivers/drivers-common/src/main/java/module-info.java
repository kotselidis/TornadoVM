/*
 * Copyright (c) 2020, APT Group, Department of Computer Science,
 * The University of Manchester.
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
open module tornado.drivers.common {
    requires transitive jdk.internal.vm.ci;
    requires transitive tornado.graal;
    requires transitive tornado.runtime;
    
    exports uk.ac.manchester.tornado.drivers.providers;
    exports uk.ac.manchester.tornado.drivers.common;
    exports uk.ac.manchester.tornado.drivers.common.architecture;
    exports uk.ac.manchester.tornado.drivers.common.mm;
    exports uk.ac.manchester.tornado.drivers.common.code;
    exports uk.ac.manchester.tornado.drivers.common.logging;
    exports uk.ac.manchester.tornado.drivers.common.compiler.cfg;
    exports uk.ac.manchester.tornado.drivers.common.compiler.phases.guards;
    exports uk.ac.manchester.tornado.drivers.common.compiler.phases.memalloc;
    exports uk.ac.manchester.tornado.drivers.common.compiler.phases.loops;
    exports uk.ac.manchester.tornado.drivers.common.compiler.phases.utils;
    exports uk.ac.manchester.tornado.drivers.common.compiler.phases.analysis;
    exports uk.ac.manchester.tornado.drivers.common.utils;
    exports uk.ac.manchester.tornado.drivers.common.power;
    exports uk.ac.manchester.tornado.drivers.common.ffm;
}
