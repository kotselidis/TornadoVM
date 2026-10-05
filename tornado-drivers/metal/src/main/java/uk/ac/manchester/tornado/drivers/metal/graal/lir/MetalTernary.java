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
package uk.ac.manchester.tornado.drivers.metal.graal.lir;

import static uk.ac.manchester.tornado.drivers.metal.graal.asm.MetalAssemblerConstants.CLOSE_PARENTHESIS;
import static uk.ac.manchester.tornado.drivers.metal.graal.asm.MetalAssemblerConstants.OPEN_PARENTHESIS;

import tornado.graal.compiler.core.common.LIRKind;
import tornado.graal.compiler.lir.LIRInstruction.Use;
import tornado.graal.compiler.lir.Opcode;

import jdk.vm.ci.meta.Value;
import uk.ac.manchester.tornado.drivers.metal.graal.asm.MetalAssembler;
import uk.ac.manchester.tornado.drivers.metal.graal.asm.MetalAssembler.MetalTernaryOp;
import uk.ac.manchester.tornado.drivers.metal.graal.asm.MetalAssembler.MetalTernaryTemplate;
import uk.ac.manchester.tornado.drivers.metal.graal.compiler.MetalCompilationResultBuilder;

public class MetalTernary {

    /**
     * Abstract operation which consumes two inputs
     */
    protected static class TernaryConsumer extends MetalLIROp {

        @Opcode
        protected final MetalTernaryOp opcode;

        @Use
        protected Value x;
        @Use
        protected Value y;
        @Use
        protected Value z;

        protected TernaryConsumer(MetalTernaryOp opcode, LIRKind lirKind, Value x, Value y, Value z) {
            super(lirKind);
            this.opcode = opcode;
            this.x = x;
            this.y = y;
            this.z = z;
        }

        @Override
        public void emit(MetalCompilationResultBuilder crb, MetalAssembler asm) {
            opcode.emit(crb, x, y, z);
        }

        @Override
        public String toString() {
            return String.format("%s %s %s %s", opcode.toString(), x, y, z);
        }

    }

    public static class Expr extends TernaryConsumer {

        public Expr(MetalTernaryOp opcode, LIRKind lirKind, Value x, Value y, Value z) {
            super(opcode, lirKind, x, y, z);
        }
    }

    public static class Select extends TernaryConsumer {

        protected MetalLIROp condition;

        public Select(LIRKind lirKind, MetalLIROp condition, Value y, Value z) {
            super(MetalTernaryTemplate.SELECT, lirKind, null, y, z);
            this.condition = condition;
        }

        @Override
        public void emit(MetalCompilationResultBuilder crb, MetalAssembler asm) {
            asm.emit(OPEN_PARENTHESIS);
            condition.emit(crb, asm);
            asm.emit(CLOSE_PARENTHESIS);
            asm.emit(" ? ");
            asm.emitValue(crb, y);
            asm.emit(" : ");
            asm.emitValue(crb, z);
        }
    }

    /**
     * Metal intrinsic call which consumes three inputs
     */
    public static class Intrinsic extends TernaryConsumer {

        public Intrinsic(MetalTernaryOp opcode, LIRKind lirKind, Value x, Value y, Value z) {
            super(opcode, lirKind, x, y, z);
        }

        @Override
        public String toString() {
            return String.format("%s(%s, %s, %s)", opcode.toString(), x, y, z);
        }
    }

}
