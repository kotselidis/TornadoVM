/*
 * Copyright (c) 2025, APT Group, Department of Computer Science,
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
import uk.ac.manchester.tornado.runtime.TornadoBackendProvider;

module tornado.drivers.metal {
    requires transitive jdk.internal.vm.ci;
    requires transitive tornado.graal;
    requires transitive org.graalvm.collections;
    requires transitive org.graalvm.word;
    requires transitive tornado.api;
    requires transitive tornado.runtime;
    requires tornado.drivers.common;

    requires java.management;
    requires jdk.management;

    exports uk.ac.manchester.tornado.drivers.metal;
    exports uk.ac.manchester.tornado.drivers.metal.builtins;
    exports uk.ac.manchester.tornado.drivers.metal.enums;
    exports uk.ac.manchester.tornado.drivers.metal.ffm;
    exports uk.ac.manchester.tornado.drivers.metal.exceptions;
    exports uk.ac.manchester.tornado.drivers.metal.graal;
    exports uk.ac.manchester.tornado.drivers.metal.graal.asm;
    exports uk.ac.manchester.tornado.drivers.metal.graal.backend;
    exports uk.ac.manchester.tornado.drivers.metal.graal.compiler;
    exports uk.ac.manchester.tornado.drivers.metal.graal.compiler.plugins;
    exports uk.ac.manchester.tornado.drivers.metal.graal.lir;
    exports uk.ac.manchester.tornado.drivers.metal.graal.meta;
    exports uk.ac.manchester.tornado.drivers.metal.graal.nodes;
    exports uk.ac.manchester.tornado.drivers.metal.graal.nodes.logic;
    exports uk.ac.manchester.tornado.drivers.metal.graal.nodes.vector;
    exports uk.ac.manchester.tornado.drivers.metal.graal.phases;
    exports uk.ac.manchester.tornado.drivers.metal.graal.snippets;
    exports uk.ac.manchester.tornado.drivers.metal.mm;
    exports uk.ac.manchester.tornado.drivers.metal.runtime;
    exports uk.ac.manchester.tornado.drivers.metal.tests;
    exports uk.ac.manchester.tornado.drivers.metal.power;
    exports uk.ac.manchester.tornado.drivers.metal.scheduler;
    exports uk.ac.manchester.tornado.drivers.metal.natives;

    provides TornadoBackendProvider with
            uk.ac.manchester.tornado.drivers.metal.MetalTornadoDriverProvider;
}
