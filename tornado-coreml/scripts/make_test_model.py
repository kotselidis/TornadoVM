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
Writes the Core ML model the tornado-coreml unit tests load: AddMul.mlpackage, with two float32
inputs x and y of shape [N] and two float32 outputs, prod = x * y and sum = x + y.

The model specification (Model.proto of coremltools) is encoded by hand, so no Python package is
needed. With coremltools installed the same model is:

    from coremltools.models.neural_network import NeuralNetworkBuilder
    from coremltools.models import datatypes
    b = NeuralNetworkBuilder([("x", datatypes.Array(N)), ("y", datatypes.Array(N))],
                             [("prod", datatypes.Array(N)), ("sum", datatypes.Array(N))])
    b.add_elementwise("mul", ["x", "y"], "prod", mode="MULTIPLY")
    b.add_elementwise("add", ["x", "y"], "sum", mode="ADD")

Usage: make_test_model.py <output directory> [N]
"""

import json
import os
import struct
import sys

N = 16
UUID = "6F0D2A57-6C4B-4E0A-9C1E-2B3C4D5E6F70"

# ArrayFeatureType.ArrayDataType.FLOAT32
FLOAT32 = 0x10000 | 32


def varint(value):
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def key(field, wire_type):
    return varint((field << 3) | wire_type)


def int_field(field, value):
    return key(field, 0) + varint(value)


def bytes_field(field, payload):
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    return key(field, 2) + varint(len(payload)) + payload


def float_field(field, value):
    return key(field, 5) + struct.pack("<f", value)


def array_feature(name, shape):
    # ArrayFeatureType: shape = 1 (repeated int64), dataType = 2
    array_type = b"".join(int_field(1, d) for d in shape) + int_field(2, FLOAT32)
    # FeatureType: multiArrayType = 5
    feature_type = bytes_field(5, array_type)
    # FeatureDescription: name = 1, type = 3
    return bytes_field(1, name) + bytes_field(3, feature_type)


def layer(name, inputs, output, kind_field):
    # NeuralNetworkLayer: name = 1, input = 2, output = 3, add = 230, multiply = 231
    body = bytes_field(1, name)
    body += b"".join(bytes_field(2, i) for i in inputs)
    body += bytes_field(3, output)
    # AddLayerParams / MultiplyLayerParams: alpha = 1 (unused with two inputs)
    body += bytes_field(kind_field, float_field(1, 0.0))
    return body


def model(n):
    # ModelDescription: input = 1, output = 10
    description = bytes_field(1, array_feature("x", [n])) + bytes_field(1, array_feature("y", [n]))
    description += bytes_field(10, array_feature("prod", [n])) + bytes_field(10, array_feature("sum", [n]))
    # NeuralNetwork: layers = 1
    network = bytes_field(1, layer("mul", ["x", "y"], "prod", 231)) + bytes_field(1, layer("add", ["x", "y"], "sum", 230))
    # Model: specificationVersion = 1, description = 2, neuralNetwork = 500
    return int_field(1, 1) + bytes_field(2, description) + bytes_field(500, network)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    n = int(sys.argv[2]) if len(sys.argv) > 2 else N
    package = os.path.join(sys.argv[1], "AddMul.mlpackage")
    data = os.path.join(package, "Data", "com.apple.CoreML")
    os.makedirs(data, exist_ok=True)
    with open(os.path.join(data, "model.mlmodel"), "wb") as f:
        f.write(model(n))
    manifest = {
        "fileFormatVersion": "1.0.0",
        "itemInfoEntries": {
            UUID: {
                "author": "com.apple.CoreML",
                "description": "CoreML Model Specification",
                "name": "model.mlmodel",
                "path": "com.apple.CoreML/model.mlmodel",
            }
        },
        "rootModelIdentifier": UUID,
    }
    with open(os.path.join(package, "Manifest.json"), "w") as f:
        json.dump(manifest, f, indent=4)
        f.write("\n")
    print(package)


if __name__ == "__main__":
    main()
