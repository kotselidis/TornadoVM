#!/usr/bin/env python3
#
# Copyright (c) 2026, APT Group, Department of Computer Science,
# The University of Manchester.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Writes the float16 ML program the tornado-coreml Neural Engine tests load: Mlp.mlpackage, a two-layer
perceptron y = relu(x @ W1^T) @ W2^T with a float16 input x and output y of shape [ROWS, DIM].

The weights follow a formula, so the tests compute the expected output without reading the model:
    W1[o][i] = 0.0625 * sin(0.37 * o + 0.11 * i)
    W2[o][i] = 0.0625 * cos(0.23 * o - 0.17 * i)

Needs coremltools (pip install coremltools numpy).

Usage: make_ane_model.py <output directory> [--check]
    --check  also prints where Core ML places each operation (macOS 14.4 or newer)
"""

import os
import shutil
import sys

import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

ROWS = 64
DIM = 256


def weights():
    o = np.arange(DIM, dtype=np.float64)[:, None]
    i = np.arange(DIM, dtype=np.float64)[None, :]
    w1 = 0.0625 * np.sin(0.37 * o + 0.11 * i)
    w2 = 0.0625 * np.cos(0.23 * o - 0.17 * i)
    return w1.astype(np.float16), w2.astype(np.float16)


def build():
    w1, w2 = weights()

    @mb.program(input_specs=[mb.TensorSpec(shape=(ROWS, DIM), dtype=types.fp16)], opset_version=ct.target.macOS13)
    def mlp(x):
        h = mb.linear(x=x, weight=w1, name="fc1")
        h = mb.relu(x=h, name="relu")
        return mb.linear(x=h, weight=w2, name="y")

    return ct.convert(mlp,
                      convert_to="mlprogram",
                      minimum_deployment_target=ct.target.macOS13,
                      compute_precision=ct.precision.FLOAT16,
                      inputs=[ct.TensorType(name="x", shape=(ROWS, DIM), dtype=np.float16)],
                      outputs=[ct.TensorType(name="y", dtype=np.float16)])


def check(path):
    from coremltools.models.compute_plan import MLComputePlan
    compiled = ct.utils.compile_model(path)
    for units in (ct.ComputeUnit.CPU_ONLY, ct.ComputeUnit.CPU_AND_GPU, ct.ComputeUnit.CPU_AND_NE, ct.ComputeUnit.ALL):
        plan = MLComputePlan.load_from_path(path=compiled, compute_units=units)
        program = plan.model_structure.program
        print(units.name)
        for operation in program.functions["main"].block.operations:
            usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
            if usage is not None:
                print("   %-8s %-10s %s" % (operation.operator_name, operation.outputs[0].name, type(usage.preferred_compute_device).__name__))


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    path = os.path.join(sys.argv[1], "Mlp.mlpackage")
    if os.path.exists(path):
        shutil.rmtree(path)
    build().save(path)
    print(path)
    if "--check" in sys.argv:
        check(path)


if __name__ == "__main__":
    main()
