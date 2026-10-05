/*
 * Copyright (c) 2023, APT Group, Department of Computer Science,
 * The University of Manchester. All rights reserved.
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
package uk.ac.manchester.tornado.drivers.cuda.graal.phases;

import java.util.Optional;

import tornado.graal.compiler.core.common.type.Stamp;
import tornado.graal.compiler.nodes.GraphState;
import tornado.graal.compiler.nodes.NodeView;
import tornado.graal.compiler.nodes.StructuredGraph;
import tornado.graal.compiler.nodes.calc.SqrtNode;
import tornado.graal.compiler.nodes.memory.ReadNode;
import tornado.graal.compiler.nodes.memory.WriteNode;
import tornado.graal.compiler.phases.Phase;

import uk.ac.manchester.tornado.api.TornadoDeviceContext;
import uk.ac.manchester.tornado.api.exceptions.TornadoDeviceFP64NotSupported;
import uk.ac.manchester.tornado.drivers.cuda.graal.nodes.CUDAFPBinaryIntrinsicNode;
import uk.ac.manchester.tornado.drivers.cuda.graal.nodes.CUDAFPUnaryIntrinsicNode;

public class CUDAFP64SupportPhase extends Phase {

    private TornadoDeviceContext deviceContext;

    public CUDAFP64SupportPhase(TornadoDeviceContext deviceContext) {
        this.deviceContext = deviceContext;
    }

    @Override
    public Optional<NotApplicable> notApplicableTo(GraphState graphState) {
        return ALWAYS_APPLICABLE;
    }

    /**
     * This method evaluates if a stamp is of type that requires double floating
     * point precision, or not.
     *
     * @param stamp
     * @return returns true if stamp is f64 or double
     */
    private boolean isStampFP64Type(Stamp stamp) {
        return stamp.toString().contains("f64") || stamp.toString().toLowerCase().contains("double");
    }

    /**
     * This method checks if a stamp requires double floating point precision or
     * not. Additionally, the deviceContext is used to check if double precision is
     * supported by the target device. In CUDADriver, a device supports double precision
     * if the cl_khr_fp64 attribute is enabled.
     * 
     * If the input stamp requires double precision and the target device does not
     * support this feature, a {@link TornadoDeviceFP64NotSupported} exception is
     * thrown.
     *
     * @param stamp
     *            a stamp of a node
     */
    private void checkStampForFP64Support(Stamp stamp) {
        boolean isStampFP64Type = isStampFP64Type(stamp);
        if (isStampFP64Type && !deviceContext.isFP64Supported()) {
            throw new TornadoDeviceFP64NotSupported("The current CUDADriver device (" + deviceContext.getDeviceName() + ") does not support FP64");
        }
    }

    @Override
    protected void run(StructuredGraph graph) {

        graph.getNodes().filter(WriteNode.class).forEach(writeNode -> checkStampForFP64Support(writeNode.getAccessStamp(NodeView.DEFAULT)));

        graph.getNodes().filter(ReadNode.class).forEach(readNode -> checkStampForFP64Support(readNode.getAccessStamp(NodeView.DEFAULT)));

        graph.getNodes().filter(CUDAFPUnaryIntrinsicNode.class).forEach(node -> checkStampForFP64Support(node.stamp(NodeView.DEFAULT)));

        graph.getNodes().filter(CUDAFPBinaryIntrinsicNode.class).forEach(node -> checkStampForFP64Support(node.stamp(NodeView.DEFAULT)));

        graph.getNodes().filter(SqrtNode.class).forEach(node -> checkStampForFP64Support(node.stamp(NodeView.DEFAULT)));

    }
}
