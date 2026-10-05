/*
 * This file is part of Tornado: A heterogeneous programming framework:
 * https://github.com/beehive-lab/tornadovm
 *
 * Copyright (c) 2026, APT Group, Department of Computer Science,
 * School of Engineering, The University of Manchester. All rights reserved.
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
package uk.ac.manchester.tornado.drivers.metal.graal.nodes;

import tornado.graal.compiler.core.common.LIRKind;
import tornado.graal.compiler.core.common.type.StampFactory;
import tornado.graal.compiler.core.common.type.TypeReference;
import tornado.graal.compiler.graph.NodeClass;
import tornado.graal.compiler.lir.Variable;
import tornado.graal.compiler.nodeinfo.NodeInfo;
import tornado.graal.compiler.nodes.FixedNode;
import tornado.graal.compiler.nodes.ValueNode;
import tornado.graal.compiler.nodes.memory.MemoryKill;
import tornado.graal.compiler.nodes.spi.LIRLowerable;
import tornado.graal.compiler.nodes.spi.NodeLIRBuilderTool;

import jdk.vm.ci.meta.JavaKind;
import jdk.vm.ci.meta.ResolvedJavaType;
import jdk.vm.ci.meta.Value;
import uk.ac.manchester.tornado.drivers.metal.graal.MetalArchitecture;
import uk.ac.manchester.tornado.drivers.metal.graal.asm.MetalAssembler;
import uk.ac.manchester.tornado.drivers.metal.graal.lir.MetalBinary;
import uk.ac.manchester.tornado.drivers.metal.graal.lir.MetalKind;
import uk.ac.manchester.tornado.drivers.metal.graal.lir.MetalLIRStmt;
import uk.ac.manchester.tornado.runtime.graal.nodes.interfaces.MarkLocalArray;

@NodeInfo
public class LocalArrayNode extends FixedNode implements LIRLowerable, MarkLocalArray, MemoryKill {

    public static final NodeClass<LocalArrayNode> TYPE = NodeClass.create(LocalArrayNode.class);

    @Input
    protected ValueNode length;
    protected MetalArchitecture.MetalMemoryBase memoryRegister;
    protected MetalAssembler.MetalBinaryTemplate arrayTemplate;
    private MetalKind kind;

    public LocalArrayNode(MetalArchitecture.MetalMemoryBase memoryRegister, ResolvedJavaType elementType, ValueNode length) {
        super(TYPE, StampFactory.objectNonNull(TypeReference.createTrustedWithoutAssumptions(elementType.getArrayClass())));
        this.memoryRegister = memoryRegister;
        this.length = length;
        this.kind = MetalKind.fromResolvedJavaType(elementType);
        this.arrayTemplate = MetalKind.resolveTemplateType(elementType);
    }

    public LocalArrayNode(MetalArchitecture.MetalMemoryBase memoryRegister, JavaKind elementKind, ValueNode length) {
        super(TYPE, StampFactory.forKind(JavaKind.Object));
        this.memoryRegister = memoryRegister;
        this.length = length;
        this.kind = MetalKind.fromResolvedJavaKind(elementKind);
        this.arrayTemplate = MetalKind.resolveTemplateType(elementKind);
    }

    public LocalArrayNode(MetalArchitecture.MetalMemoryBase memoryRegister, MetalKind overrideKind, MetalAssembler.MetalBinaryTemplate template, ValueNode length) {
        super(TYPE, StampFactory.forKind(JavaKind.Object));
        this.memoryRegister = memoryRegister;
        this.length = length;
        this.kind = overrideKind;
        this.arrayTemplate = template;
    }

    public MetalArchitecture.MetalMemoryBase getMemoryRegister() {
        return memoryRegister;
    }

    public ValueNode getLength() {
        return length;
    }

    public MetalKind getMetalKind() {
        return kind;
    }

    @Override
    public void generate(NodeLIRBuilderTool gen) {
        final Value lengthValue = gen.operand(length);

        LIRKind lirKind = LIRKind.value(kind);
        final Variable variable = gen.getLIRGeneratorTool().newVariable(lirKind);
        final MetalBinary.Expr declaration = new MetalBinary.Expr(arrayTemplate, lirKind, variable, lengthValue);

        final MetalLIRStmt.ExprStmt expr = new MetalLIRStmt.ExprStmt(declaration);
        gen.getLIRGeneratorTool().append(expr);
        gen.setResult(this, variable);
    }
}
