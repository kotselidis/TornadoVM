#!/usr/bin/env bash
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
# Core ML benchmark: TornadoVM library tasks against a native Swift loop, on the same float16
# perceptrons and compute units. Run from a shell where TornadoVM's setvars.sh is sourced.
#
# Usage: run.sh <output directory>
# Environment: COREMLTOOLS_PYTHON (python with coremltools), WARMUP, ITERATIONS, UNITS, SHAPES

set -euo pipefail

OUT=${1:?usage: run.sh <output directory>}
HERE=$(cd "$(dirname "$0")" && pwd)
PYTHON=${COREMLTOOLS_PYTHON:-$HOME/.venvs/coremltools/bin/python}
WARMUP=${WARMUP:-50}
ITERATIONS=${ITERATIONS:-500}
UNITS=${UNITS:-CPU_ONLY,CPU_AND_GPU,CPU_AND_NEURAL_ENGINE,ALL}
SHAPES=${SHAPES:-"64,256,2 256,1024,4 512,2048,8"}

mkdir -p "$OUT/models"
command -v tornado >/dev/null || { echo "tornado not on PATH: source setvars.sh first" >&2; exit 1; }

models=()
for shape in $SHAPES; do
    IFS=, read -r rows dim layers <<< "$shape"
    model="$OUT/models/Mlp_${rows}x${dim}x${layers}.mlpackage"
    [ -d "$model" ] || "$PYTHON" -I "$HERE/make_bench_models.py" "$OUT/models" "$shape" 2>/dev/null | grep -v '^$' || true
    models+=("$model")
done

swiftc -O "$HERE/coreml_bench.swift" -o "$OUT/coreml_bench"

echo "mode,model,units,load_ms,first_ms,median_us,p10_us,p90_us" > "$OUT/native.csv"
for model in "${models[@]}"; do
    for units in ${UNITS//,/ }; do
        "$OUT/coreml_bench" "$model" "$units" "$WARMUP" "$ITERATIONS" | tee -a "$OUT/native.csv"
    done
done

tornado --jvm "-Dbench.warmup=$WARMUP -Dbench.iterations=$ITERATIONS -Dbench.units=$UNITS" \
    -m tornado.examples/uk.ac.manchester.tornado.examples.coreml.CoreMLBenchmark "${models[@]}" \
    | grep -E '^(mode|tornado-)' | tee "$OUT/tornado.csv"

python3 -I "$HERE/summarize.py" "$OUT/native.csv" "$OUT/tornado.csv" | tee "$OUT/summary.md"
