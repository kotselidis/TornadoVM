//
// Copyright (c) 2026, APT Group, Department of Computer Science,
// The University of Manchester.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Native Core ML baseline for the tornado-coreml benchmark: the same model, compute units and
// float16 [rows, dim] input and output, predicted in a loop on preallocated buffers, with the
// output written through an output backing as tornado-coreml does.
//
// Build:  swiftc -O coreml_bench.swift -o coreml_bench
// Usage:  coreml_bench <model.mlpackage|.mlmodelc> <CPU_ONLY|CPU_AND_GPU|CPU_AND_NEURAL_ENGINE|ALL> <warmup> <iterations>
// Prints: native,<model>,<units>,<load ms>,<first ms>,<median us>,<p10 us>,<p90 us>

import CoreML
import Foundation

func fail(_ message: String) -> Never {
    FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
    exit(1)
}

let args = CommandLine.arguments
guard args.count == 5, let warmup = Int(args[3]), let iterations = Int(args[4]) else {
    fail("usage: coreml_bench <model> <units> <warmup> <iterations>")
}
let units: MLComputeUnits
switch args[2] {
case "CPU_ONLY": units = .cpuOnly
case "CPU_AND_GPU": units = .cpuAndGPU
case "CPU_AND_NEURAL_ENGINE": units = .cpuAndNeuralEngine
case "ALL": units = .all
default: fail("unknown compute units " + args[2])
}

func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }

let loadStart = now()
var url = URL(fileURLWithPath: args[1])
if url.pathExtension != "mlmodelc" {
    url = try! MLModel.compileModel(at: url)
}
let configuration = MLModelConfiguration()
configuration.computeUnits = units
let model = try! MLModel(contentsOf: url, configuration: configuration)
let loadMs = Double(now() - loadStart) / 1e6

guard let input = model.modelDescription.inputDescriptionsByName["x"]?.multiArrayConstraint,
      model.modelDescription.outputDescriptionsByName["y"] != nil else {
    fail("model needs a multi-array input x and output y")
}
let shape = input.shape
let x = try! MLMultiArray(shape: shape, dataType: .float16)
let y = try! MLMultiArray(shape: shape, dataType: .float16)
x.withUnsafeMutableBytes { bytes, _ in
    let p = bytes.bindMemory(to: UInt16.self)
    for i in 0..<p.count { p[i] = Float16(Float((i % 17) - 8) / 8).bitPattern }
}
let features = try! MLDictionaryFeatureProvider(dictionary: ["x": MLFeatureValue(multiArray: x)])
let options = MLPredictionOptions()
options.outputBackings = ["y": y]

let firstStart = now()
_ = try! model.prediction(from: features, options: options)
let firstMs = Double(now() - firstStart) / 1e6

for _ in 0..<warmup {
    _ = try! model.prediction(from: features, options: options)
}
var times = [Double]()
times.reserveCapacity(iterations)
for _ in 0..<iterations {
    let start = now()
    _ = try! model.prediction(from: features, options: options)
    times.append(Double(now() - start) / 1e3)
}
times.sort()
func pct(_ p: Double) -> Double { times[min(times.count - 1, Int(p * Double(times.count)))] }
let name = URL(fileURLWithPath: args[1]).deletingPathExtension().lastPathComponent
print(String(format: "native,%@,%@,%.1f,%.2f,%.1f,%.1f,%.1f", name, args[2], loadMs, firstMs, pct(0.5), pct(0.1), pct(0.9)))
