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
"""Joins the native and TornadoVM benchmark CSVs into a Markdown table (medians in microseconds)."""

import csv
import re
import sys


def flops(model):
    m = re.match(r"Mlp_(\d+)x(\d+)x(\d+)", model)
    if not m:
        return None
    rows, dim, layers = (int(v) for v in m.groups())
    return 2.0 * rows * dim * dim * layers


def main():
    native = {(r["model"], r["units"]): r for r in csv.DictReader(open(sys.argv[1]))}
    tornado = {}
    for r in csv.DictReader(open(sys.argv[2])):
        tornado[(r["mode"], r["model"], r["units"])] = r
    print("| model | units | placement | native us | tornado task us | overhead us | tornado e2e us | TFLOP/s (task) | in place |")
    print("|---|---|---|---:|---:|---:|---:|---:|---|")
    for (model, units), n in native.items():
        task = tornado.get(("tornado-task", model, units))
        e2e = tornado.get(("tornado-e2e", model, units))
        if not task:
            continue
        nat = float(n["median_us"])
        t = float(task["median_us"])
        f = flops(model)
        tflops = "%.2f" % (f / (t * 1e-6) / 1e12) if f else ""
        print("| %s | %s | %s | %.1f | %.1f | %+.1f | %s | %s | %s |" % (model, units, task["placement"], nat, t, t - nat,
                                                                         e2e["median_us"] if e2e else "", tflops, task["in_place"]))


if __name__ == "__main__":
    main()
