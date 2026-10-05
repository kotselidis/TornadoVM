/*
 * Copyright (c) 2026, APT Group, Department of Computer Science,
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
 *
 */
package uk.ac.manchester.tornado.drivers.cuda.graal.nodes;

import jdk.vm.ci.meta.JavaKind;
import tornado.graal.compiler.core.common.LIRKind;
import tornado.graal.compiler.core.common.type.StampFactory;
import tornado.graal.compiler.graph.NodeClass;
import tornado.graal.compiler.lir.Variable;
import tornado.graal.compiler.lir.gen.LIRGeneratorTool;
import tornado.graal.compiler.nodeinfo.NodeInfo;
import tornado.graal.compiler.nodes.FixedWithNextNode;
import tornado.graal.compiler.nodes.ValueNode;
import tornado.graal.compiler.nodes.spi.LIRLowerable;
import tornado.graal.compiler.nodes.spi.NodeLIRBuilderTool;
import uk.ac.manchester.tornado.api.tile.DType;
import uk.ac.manchester.tornado.drivers.cuda.graal.lir.CUDAKind;
import uk.ac.manchester.tornado.drivers.cuda.graal.lir.CUDATileStmt;

/**
 * An accumulator-free tile product, {@code ct::matmul}.
 *
 * <p>
 * Separate from {@link CUDATileMmaNode} because the result type comes from a different place:
 * {@code mma} carries the accumulator's type and shape, whereas here both are derived from the
 * operands - the shape by contracting them and the element type through
 * {@link DType#matmulResultType()}.
 * </p>
 */
@NodeInfo
public class CUDATileMatmulNode extends FixedWithNextNode implements LIRLowerable, CUDATileNode {

    public static final NodeClass<CUDATileMatmulNode> TYPE = NodeClass.create(CUDATileMatmulNode.class);

    @Input
    protected ValueNode tileA;
    @Input
    protected ValueNode tileB;

    private final DType resultType;
    private final int[] resultShape;

    public CUDATileMatmulNode(ValueNode tileA, ValueNode tileB, DType resultType, int[] resultShape) {
        super(TYPE, StampFactory.forKind(JavaKind.Object));
        this.tileA = tileA;
        this.tileB = tileB;
        this.resultType = resultType;
        this.resultShape = resultShape;
    }

    @Override
    public String tileOperationName() {
        return "tile matmul";
    }

    @Override
    public DType tileDType() {
        return resultType;
    }

    @Override
    public int[] tileShape() {
        return resultShape;
    }

    @Override
    public void generate(NodeLIRBuilderTool gen) {
        LIRGeneratorTool tool = gen.getLIRGeneratorTool();
        Variable result = tool.newVariable(LIRKind.value(CUDAKind.TILE));
        tool.append(new CUDATileStmt.TileMatmulStmt(result, gen.operand(tileA), gen.operand(tileB), resultType, resultShape));
        gen.setResult(this, result);
    }
}
