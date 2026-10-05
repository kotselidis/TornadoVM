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
package uk.ac.manchester.tornado.runtime.sketcher;

import tornado.graal.compiler.phases.PhaseSuite;
import tornado.graal.compiler.phases.tiers.HighTierContext;
import tornado.graal.compiler.phases.util.Providers;

import java.util.Set;

import jdk.vm.ci.meta.ResolvedJavaMethod;
import uk.ac.manchester.tornado.runtime.graal.compiler.TornadoSketchTier;

public class SketchRequest {

    final int driverIndex;
    final int deviceIndex;
    final ResolvedJavaMethod resolvedMethod;
    final Providers providers;
    final PhaseSuite<HighTierContext> graphBuilderSuite;
    final TornadoSketchTier sketchTier;
    /**
     * Kernels whose sketches are being built and that (transitively) launch this method from the
     * device. Their sketches cannot be waited on from here: that would wait on itself.
     */
    final Set<ResolvedJavaMethod> launchAncestors;

    public SketchRequest(ResolvedJavaMethod resolvedMethod, Providers providers, PhaseSuite<HighTierContext> graphBuilderSuite, TornadoSketchTier sketchTier, int driverIndex, int deviceIndex) {
        this(resolvedMethod, providers, graphBuilderSuite, sketchTier, driverIndex, deviceIndex, Set.of());
    }

    SketchRequest(ResolvedJavaMethod resolvedMethod, Providers providers, PhaseSuite<HighTierContext> graphBuilderSuite, TornadoSketchTier sketchTier, int driverIndex, int deviceIndex,
            Set<ResolvedJavaMethod> launchAncestors) {
        this.launchAncestors = launchAncestors;
        this.resolvedMethod = resolvedMethod;
        this.providers = providers;
        this.graphBuilderSuite = graphBuilderSuite;
        this.sketchTier = sketchTier;
        this.driverIndex = driverIndex;
        this.deviceIndex = deviceIndex;
    }

    public void run() {
        TornadoSketcher.buildSketch(this);
    }
}
