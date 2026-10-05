/*
 * Copyright (c) 2018, 2020-2022, 2024, 2025, APT Group, Department of Computer Science,
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

package uk.ac.manchester.tornado.drivers.cuda.graal.nodes;

import jdk.vm.ci.meta.JavaKind;
import tornado.graal.compiler.core.common.type.StampFactory;
import tornado.graal.compiler.graph.Node.OptionalInput;
import tornado.graal.compiler.graph.NodeClass;
import tornado.graal.compiler.graph.NodeInputList;
import tornado.graal.compiler.nodeinfo.NodeInfo;
import tornado.graal.compiler.nodes.ValueNode;
import tornado.graal.compiler.nodes.calc.FloatingNode;
import uk.ac.manchester.tornado.api.tile.DType;

/**
 * A tensor view: the buffer, its element type and its extents, carried from
 * {@code TileContext.view(...)} to the {@code partition(...)} that consumes it.
 *
 * <p>
 * This node is deliberately not lowerable. It is a parse time descriptor that
 * {@link CUDATilePartitionViewNode} reads and then discards, because a view only reaches the
 * generated code as part of the partition view that a load or store addresses through. If one
 * ever survives to LIR that is a bug in the plugin pairing, and it fails loudly rather than
 * emitting something meaningless.
 * </p>
 */
@NodeInfo
public class CUDATileViewNode extends FloatingNode {

    public static final NodeClass<CUDATileViewNode> TYPE = NodeClass.create(CUDATileViewNode.class);

    @Input
    protected ValueNode buffer;
    @Input
    protected NodeInputList<ValueNode> extents;

    @OptionalInput
    protected ValueNode rowStride;
    @OptionalInput
    protected ValueNode elementOffset;

    private final DType dtype;

    public CUDATileViewNode(ValueNode buffer, ValueNode[] extents, ValueNode rowStride, ValueNode elementOffset, DType dtype) {
        this(buffer, extents, dtype);
        this.rowStride = rowStride;
        this.elementOffset = elementOffset;
    }

    public ValueNode getRowStride() {
        return rowStride;
    }

    public ValueNode getElementOffset() {
        return elementOffset;
    }

    public CUDATileViewNode(ValueNode buffer, ValueNode[] extents, DType dtype) {
        super(TYPE, StampFactory.forKind(JavaKind.Object));
        this.buffer = buffer;
        this.extents = new NodeInputList<>(this, extents);
        this.dtype = dtype;
    }

    public ValueNode getBuffer() {
        return buffer;
    }

    public ValueNode[] getExtents() {
        return extents.toArray(new ValueNode[0]);
    }

    public int getRank() {
        return extents.size();
    }

    public DType getDType() {
        return dtype;
    }
}
