/*
 * Copyright (c) 2018, 2020, 2024 APT Group, Department of Computer Science,
 * The University of Manchester. All rights reserved.
 * Copyright (c) 2009, 2017, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation. Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
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
package uk.ac.manchester.tornado.drivers.common.compiler.phases.analysis;

import static uk.ac.manchester.tornado.runtime.graal.compiler.TornadoCodeGenerator.debug;

import java.util.HashMap;
import java.util.Map;

import tornado.graal.compiler.graph.Node;
import tornado.graal.compiler.graph.iterators.NodePredicate;
import tornado.graal.compiler.nodes.StructuredGraph;
import tornado.graal.compiler.nodes.ValueNode;
import tornado.graal.compiler.nodes.java.LoadFieldNode;
import tornado.graal.compiler.nodes.java.NewInstanceNode;
import tornado.graal.compiler.nodes.java.StoreFieldNode;
import tornado.graal.compiler.phases.BasePhase;

import jdk.vm.ci.meta.ResolvedJavaField;
import uk.ac.manchester.tornado.api.internal.annotations.Vector;
import uk.ac.manchester.tornado.runtime.graal.phases.TornadoHighTierContext;

public class TornadoValueTypeReplacement extends BasePhase<TornadoHighTierContext> {

    private static final NodePredicate valueTypeFilter = new NodePredicate() {
        @Override
        public boolean apply(Node node) {
            return ((NewInstanceNode) node).instanceClass().getAnnotation(Vector.class) != null;
        }
    };

    private void simplify(NewInstanceNode newInstance) {
        debug("simplify: node=%s", newInstance.toString());

        /*
         * make dict
         */
        final Map<ResolvedJavaField, ValueNode> fieldToValue = new HashMap<>();
        newInstance.usages().filter(StoreFieldNode.class).forEach((store) -> {
            fieldToValue.put(store.field(), store.value());
            store.clearInputs();
            store.graph().removeFixed(store);
        });

        if (fieldToValue.isEmpty()) {
            return;
        }

        // print dict
        for (ResolvedJavaField field : fieldToValue.keySet()) {
            debug("simplify: field=%s -> value=%s", field.getName(), fieldToValue.get(field).toString());
        }

        /*
         * contract fields
         */
        newInstance.graph().getNodes().filter(LoadFieldNode.class).forEach((load) -> {
            if (load.object() == newInstance) {
                debug("simplify: load field=%s", load.field().getName());
                debug("simplify: load field=%s -> value=%s", load.field().getName(), fieldToValue.get(load.field()).toString());
                load.replaceAtUsages(fieldToValue.get(load.field()));
                load.graph().removeFixed(load);

            }
        });
    }

    public void execute(StructuredGraph graph, TornadoHighTierContext context) {
        run(graph, context);
    }

    @Override
    protected void run(StructuredGraph graph, TornadoHighTierContext context) {
        graph.getNodes().filter(NewInstanceNode.class).filter(valueTypeFilter).forEach(this::simplify);
        graph.maybeCompress();
    }
}
