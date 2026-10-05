/*
 * Copyright (c) 2025, APT Group, Department of Computer Science,
 * School of Engineering, The University of Manchester. All rights reserved.
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
package uk.ac.manchester.tornado.drivers.opencl.graal.nodes;

import jdk.vm.ci.meta.Value;
import tornado.graal.compiler.core.common.LIRKind;
import tornado.graal.compiler.core.common.type.Stamp;
import tornado.graal.compiler.graph.NodeClass;
import tornado.graal.compiler.lir.Variable;
import tornado.graal.compiler.nodeinfo.NodeInfo;
import tornado.graal.compiler.nodes.FixedWithNextNode;
import tornado.graal.compiler.nodes.ValueNode;
import tornado.graal.compiler.nodes.memory.address.AddressNode;
import tornado.graal.compiler.nodes.spi.LIRLowerable;
import tornado.graal.compiler.nodes.spi.NodeLIRBuilderTool;
import uk.ac.manchester.tornado.drivers.opencl.graal.compiler.OCLLIRGenerator;
import uk.ac.manchester.tornado.drivers.opencl.graal.lir.OCLKind;
import uk.ac.manchester.tornado.drivers.opencl.graal.lir.OCLLIRStmt;

/**
 * Node for reading and decompressing a 4-byte object reference from an object's field into a full 64-bit address.
 */
@NodeInfo
public class OCLDecompressedReadFieldNode extends FixedWithNextNode implements LIRLowerable {
    public static final NodeClass<OCLDecompressedReadFieldNode> TYPE = NodeClass.create(OCLDecompressedReadFieldNode.class);
    @Input
    private ValueNode object;
    @Input
    private AddressNode address;

    public OCLDecompressedReadFieldNode(ValueNode object, AddressNode address, Stamp stamp) {
        super(TYPE, stamp);
        this.object = object;
        this.address = address;
    }

    public ValueNode getObject() {
        return object;
    }

    @Override
    public void generate(NodeLIRBuilderTool gen) {
        OCLLIRGenerator tool = (OCLLIRGenerator) gen.getLIRGeneratorTool();
        Value addressBase = gen.operand(address.getBase());
        Value index = gen.operand(address.getIndex());
        Variable fieldAddress = tool.getArithmetic().emitAdd(addressBase, index, false);
        Variable compressedPointer = tool.newVariable(LIRKind.value(OCLKind.UINT));
        tool.append(new OCLLIRStmt.CastCompressedStmt(compressedPointer, fieldAddress));
        Variable decompressedPointer = tool.newVariable(LIRKind.value(OCLKind.ULONG));
        tool.append(new OCLLIRStmt.DecompressPointerStmt(decompressedPointer, addressBase, compressedPointer));
        gen.setResult(this, decompressedPointer);
    }
}
