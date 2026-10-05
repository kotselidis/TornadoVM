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
package uk.ac.manchester.tornado.runtime.common;

import java.util.HashSet;

public class MetalTokens {
    public static HashSet<String> metalTokens = new HashSet<>();
    static {
        metalTokens.add("kernel");

        // Address spaces
        metalTokens.add("device");
        metalTokens.add("threadgroup");
        metalTokens.add("constant");
        metalTokens.add("thread");

        // Types
        metalTokens.add("half");
        metalTokens.add("float");
        metalTokens.add("int");
        metalTokens.add("uint");
        metalTokens.add("bool");

        // Vector math builtins
        metalTokens.add("dot");
        metalTokens.add("cross");
        metalTokens.add("distance");
        metalTokens.add("normalize");
        metalTokens.add("length");

        metalTokens.add("buffer");
        metalTokens.add("texture");
        metalTokens.add("sampler");
        metalTokens.add("thread_position_in_grid");
        metalTokens.add("thread_index_in_threadgroup");
        metalTokens.add("threads_per_threadgroup");
        metalTokens.add("threadgroup_position_in_grid");

        // Atomics
        metalTokens.add("atomic_int");
        metalTokens.add("atomic_uint");
        metalTokens.add("atomic_fetch_add_explicit");
        metalTokens.add("atomic_fetch_sub_explicit");
        metalTokens.add("atomic_store_explicit");
        metalTokens.add("atomic_load_explicit");
        metalTokens.add("atomic_exchange_explicit");
        metalTokens.add("atomic_compare_exchange_weak_explicit");
        metalTokens.add("atomic_compare_exchange_strong_explicit");

        // C++/MSL keywords
        metalTokens.add("this");
        metalTokens.add("class");
        metalTokens.add("struct");
        metalTokens.add("enum");
        metalTokens.add("template");
        metalTokens.add("constexpr");
        metalTokens.add("typedef");
        metalTokens.add("using");
        metalTokens.add("namespace");
    }
}
