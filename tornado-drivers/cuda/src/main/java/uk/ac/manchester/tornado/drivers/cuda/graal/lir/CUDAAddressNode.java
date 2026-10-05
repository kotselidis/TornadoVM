/*
 * Copyright (c) 2018, 2020-2022 APT Group, Department of Computer Science,
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
package uk.ac.manchester.tornado.drivers.cuda.graal.lir;

import tornado.graal.compiler.graph.NodeClass;
import tornado.graal.compiler.lir.Variable;
import tornado.graal.compiler.nodeinfo.NodeInfo;
import tornado.graal.compiler.nodes.NodeView;
import tornado.graal.compiler.nodes.ParameterNode;
import tornado.graal.compiler.nodes.ValueNode;
import tornado.graal.compiler.nodes.memory.address.AddressNode;
import tornado.graal.compiler.nodes.spi.LIRLowerable;
import tornado.graal.compiler.nodes.spi.NodeLIRBuilderTool;

import jdk.vm.ci.meta.Value;
import uk.ac.manchester.tornado.drivers.cuda.graal.CUDAArchitecture.CUDAMemoryBase;
import uk.ac.manchester.tornado.drivers.cuda.graal.CUDAStamp;
import uk.ac.manchester.tornado.drivers.cuda.graal.asm.CUDAAssemblerConstants;
import uk.ac.manchester.tornado.drivers.cuda.graal.compiler.CUDALIRGenerator;
import uk.ac.manchester.tornado.drivers.cuda.graal.lir.CUDAUnary.MemoryAccess;

@NodeInfo
public class CUDAAddressNode extends AddressNode implements LIRLowerable {

    public static final NodeClass<CUDAAddressNode> TYPE = NodeClass.create(CUDAAddressNode.class);

    @OptionalInput
    private ValueNode base;

    @OptionalInput
    private ValueNode index;

    private CUDAMemoryBase memoryRegister;

    public CUDAAddressNode(ValueNode base, ValueNode index, CUDAMemoryBase memoryRegister) {
        super(TYPE);
        this.base = base;
        this.index = index;
        this.memoryRegister = memoryRegister;
    }

    public CUDAAddressNode(ValueNode base, ValueNode index) {
        super(TYPE);
        this.base = base;
        this.index = index;
    }

    @Override
    public void generate(NodeLIRBuilderTool gen) {
        CUDALIRGenerator tool = (CUDALIRGenerator) gen.getLIRGeneratorTool();

        Value baseValue = base == null ? Value.ILLEGAL : gen.operand(base);
        if (base instanceof ParameterNode && base.stamp(NodeView.DEFAULT) instanceof CUDAStamp) {
            CUDAStamp stamp = (CUDAStamp) base.stamp(NodeView.DEFAULT);
            CUDAKind kind = stamp.getCUDAKind();
            if (kind.isVector()) {
                baseValue = tool.getOclGenTool().getParameterToVariable().get(base);
            }
        }

        Value indexValue = index == null ? Value.ILLEGAL : gen.operand(index);
        if (index == null) {
            gen.setResult(this, new MemoryAccess(memoryRegister, baseValue));
        } else {
            setMemoryAccess(gen, baseValue, indexValue, tool);
        }
    }

    private boolean isLocalMemoryAccess() {
        return this.memoryRegister.getName().equals(CUDAAssemblerConstants.LOCAL_REGION_NAME);
    }

    private boolean isPrivateMemoryAccess() {
        return this.memoryRegister.getName().equals(CUDAAssemblerConstants.PRIVATE_REGION_NAME);
    }

    @Override
    public ValueNode getBase() {
        return base;
    }

    @Override
    public ValueNode getIndex() {
        return index;
    }

    @Override
    public long getMaxConstantDisplacement() {
        return 0;
    }

    private void setMemoryAccess(NodeLIRBuilderTool gen, Value baseValue, Value indexValue, CUDALIRGenerator tool) {
        Variable addressValue;

        if (isLocalMemoryAccess() || isPrivateMemoryAccess()) {
            gen.setResult(this, new MemoryAccess(memoryRegister, baseValue, indexValue));
        } else {
            addressValue = tool.getArithmetic().emitAdd(baseValue, indexValue, false);
            gen.setResult(this, new MemoryAccess(memoryRegister, addressValue));
        }
    }
}
