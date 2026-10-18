# TornadoVM Hybrid API — Apple Core ML Library Tasks

This module lets a TornadoVM `TaskGraph` run [Core ML](https://developer.apple.com/documentation/coreml)
models between JIT-compiled tasks on the Metal backend. A Core ML library task is one prediction of a
whole model. Core ML schedules the model on the CPU, the GPU and the Neural Engine itself, within the
compute units you allow. It is the only public way to use the Neural Engine.

```java
CoreMLModel model = CoreML.load("Classifier.mlpackage", CoreMLOptions.computeUnits(ComputeUnits.ALL));

TaskGraph taskGraph = new TaskGraph("g")
    .transferToDevice(DataTransferMode.EVERY_EXECUTION, image)
    .task("pre", MyKernels::normalise, context, image, pixels)        // JIT-compiled kernel
    .libraryTask("classify", CoreML::predict, model, pixels, logits)  // Core ML prediction
    .task("post", MyKernels::softmax, context, logits, probabilities) // JIT-compiled kernel
    .transferToHost(DataTransferMode.EVERY_EXECUTION, probabilities);

try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
    plan.withGridScheduler(gridScheduler).execute();
}
model.close();
```

## Requirements

- macOS 13 or newer on Apple silicon.
- JDK 21

## Build

From the repository root:

```bash
make BACKEND=metal
source setvars.sh
```

This builds the Metal backend and the `tornado-coreml` Java module. The module talks to Core ML through
the Objective-C runtime with `java.lang.foreign`, so there is no native module to build. It is
registered as a `TornadoLibraryProvider` (`apple/coreml`) and found through `ServiceLoader`.

## Models

`CoreML.load(path, options)` takes an `.mlpackage` or `.mlmodel`, which it compiles first, or a
compiled `.mlmodelc`, which it loads as it is. Models usually come from
[coremltools](https://github.com/apple/coremltools), e.g. `ct.convert(traced_model, ...).save("M.mlpackage")`.
Compiling a large model takes a while; compile it once with `xcrun coremlcompiler compile` and load
the `.mlmodelc` to skip that step.

Every input and output must be a multi-array:

| Core ML type | TornadoVM array |
|---|---|
| float32 | `FloatArray` |
| float16 | `HalfFloatArray` |
| float64 | `DoubleArray` |
| int32 | `IntArray` |

Each array holds exactly as many elements as its feature's shape, in row-major order.

## Options

| `CoreMLOptions` | Effect |
|---|---|
| `computeUnits(ComputeUnits)` | `CPU_ONLY`, `CPU_AND_GPU`, `CPU_AND_NEURAL_ENGINE` or `ALL` (default) |
| `inputs(String...)`, `outputs(String...)` | The order of the arrays of a task. Default: inputs by name, then outputs by name |
| `shape(String, long...)` | A fixed shape for a feature whose shape is flexible |
| `function(String)` | A function of a multi-function model (macOS 15 or newer) |

`model.getInputs()` and `model.getOutputs()` list the features in task order, with their shapes and
types.

## Where Core ML runs a model

`model.computePlan()` (macOS 14.4 or newer) asks Core ML where it places each operation of the model
under the model's compute units: the preferred device (`CPU`, `GPU` or `NEURAL_ENGINE`), the devices
that could run it, and for ML programs the operation's estimated share of the model's cost.
`CoreML.availableComputeDevices()` lists the devices of the host.

```java
try (CoreMLModel model = CoreML.load("Model.mlpackage", CoreMLOptions.computeUnits(ComputeUnits.CPU_AND_NEURAL_ENGINE))) {
    CoreMLComputePlan plan = model.computePlan();
    System.out.println(plan.countByDevice());           // e.g. {CPU=3, NEURAL_ENGINE=118}
    plan.getOperations().forEach(System.out::println);  // name, type, preferred, supported, cost
}
```

## How a task runs

- On the Metal backend every TornadoVM buffer is a shared-storage `MTLBuffer`, so the CPU can address
  it. Each input is wrapped in place as an `MLMultiArray`, without copying it.
- Each output is passed to Core ML as an output backing (`MLPredictionOptions.outputBackings`), so
  Core ML writes the result directly into the TornadoVM buffer. When Core ML returns an output
  elsewhere (a different element type, or padded strides), it is copied into the buffer afterwards.
  When Core ML refuses the backings, the task predicts without them and copies every output.
- The Core ML objects for a set of buffers are built on the first execution and reused afterwards.
- Core ML runs on the host and returns when the prediction is done. The kernels queued before the
  task are finished first, also with `-Dtornado.mlx.async=true`. The kernels after it see its outputs.

## Tests

```bash
tornado-test --coreml -V
```

The tests load `AddMul.mlpackage` (two float32 inputs of 16 elements, outputs `prod = x * y` and
`sum = x + y`), written by `scripts/make_test_model.py` without any Python package.
`TestCoreMLNeuralEngine` loads `Mlp.mlpackage`, a float16 two-layer perceptron on [64, 256]
`HalfFloatArray`s that Core ML places entirely on the Neural Engine. `scripts/make_ane_model.py`
writes it with coremltools; `--check` prints where Core ML places each operation.
