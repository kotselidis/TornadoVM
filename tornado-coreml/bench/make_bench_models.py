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
Writes the float16 perceptrons the Core ML benchmark runs: Mlp_<rows>x<dim>x<layers>.mlpackage, with
input x and output y of shape [rows, dim] and <layers> linear layers of [dim, dim] with ReLU between
them. Weights are random (seeded); the benchmark measures time, not values.

Needs coremltools (pip install coremltools numpy).

Usage: make_bench_models.py <output directory> [rows,dim,layers ...]
"""

import os
import shutil
import sys

import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

DEFAULT_SHAPES = ["64,256,2", "256,1024,4", "512,2048,8"]


def build(rows, dim, layers):
    rng = np.random.default_rng(7)
    weights = [(rng.standard_normal((dim, dim)) / np.sqrt(dim)).astype(np.float16) for _ in range(layers)]

    @mb.program(input_specs=[mb.TensorSpec(shape=(rows, dim), dtype=types.fp16)], opset_version=ct.target.macOS13)
    def mlp(x):
        for layer, w in enumerate(weights):
            x = mb.linear(x=x, weight=w, name="fc%d" % layer)
            if layer < layers - 1:
                x = mb.relu(x=x, name="relu%d" % layer)
        return mb.identity(x=x, name="y")

    return ct.convert(mlp,
                      convert_to="mlprogram",
                      minimum_deployment_target=ct.target.macOS13,
                      compute_precision=ct.precision.FLOAT16,
                      inputs=[ct.TensorType(name="x", shape=(rows, dim), dtype=np.float16)],
                      outputs=[ct.TensorType(name="y", dtype=np.float16)])


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    for shape in sys.argv[2:] or DEFAULT_SHAPES:
        rows, dim, layers = (int(v) for v in shape.split(","))
        path = os.path.join(out, "Mlp_%dx%dx%d.mlpackage" % (rows, dim, layers))
        if os.path.exists(path):
            shutil.rmtree(path)
        build(rows, dim, layers).save(path)
        print(path)


if __name__ == "__main__":
    main()
