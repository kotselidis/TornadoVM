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
package uk.ac.manchester.tornado.runtime.profiler;

public enum ProfilerCodeFeatures {

    // @formatter:off
    GLOBAL_LOADS ("Global Memory Loads"),
    GLOBAL_STORES ("Global Memory Stores"),
    LOCAL_LOADS ("Local Memory Loads"),
    LOCAL_STORES ("Local Memory Stores"),
    CONSTANT_LOADS ("Constant Memory Loads"),
    CONSTANT_STORES ("Constant Memory Stores"),
    PRIVATE_LOADS ("Private Memory Loads"),
    PRIVATE_STORES ("Private Memory Stores"),
    LOOPS ("Total Loops"),
    PARALLEL_LOOPS ("Parallel Loops"),
    IFS ("If Statements"),
    SWITCH ("Switch Statements"),
    CASE ("Switch Cases"),
    CAST ("Cast Operations"),
    VECTORS ("Vector Operations"),
    INTEGER_OPS ("Total Integer Operations"),
    FLOAT_OPS ("Total Float Operations"),
    FP32 ("Single Precision Float Operations"),
    DOUBLES ("Double Precision Float Operations"),
    BINARY ("Binary Operations"),
    BOOLEAN ("Boolean Operations"),
    F_MATH ("Float Math Functions"),
    I_MATH ("Integer Math Functions"),
    I_CMP ("Integer Comparison"),
    F_CMP ("Float Comparison");
    // @formatter:on

    private String feature;

    ProfilerCodeFeatures(String featureType) {
        this.feature = featureType;
    }

    @Override
    public String toString() {
        return feature;
    }
}
