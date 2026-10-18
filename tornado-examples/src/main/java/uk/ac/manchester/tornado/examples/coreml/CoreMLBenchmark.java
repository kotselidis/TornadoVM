/*
 * Copyright (c) 2026, APT Group, Department of Computer Science,
 * The University of Manchester.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */
package uk.ac.manchester.tornado.examples.coreml;

import java.nio.file.Path;
import java.util.Arrays;

import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.coreml.ComputeUnits;
import uk.ac.manchester.tornado.coreml.CoreML;
import uk.ac.manchester.tornado.coreml.CoreMLComputePlan;
import uk.ac.manchester.tornado.coreml.CoreMLModel;
import uk.ac.manchester.tornado.coreml.CoreMLOptions;
import uk.ac.manchester.tornado.coreml.provider.CoreMLLibraryProvider;

/**
 * Times Core ML predictions run as TornadoVM library tasks, for models with one float16 input
 * {@code x} and one float16 output {@code y}, under each compute-unit setting. Two plans are timed:
 * {@code tornado-task} runs the task alone on buffers already on the device, {@code tornado-e2e} also
 * copies {@code x} in and {@code y} out on every execution.
 *
 * <p>
 * Prints one CSV line per plan:
 * {@code mode,model,units,load ms,first ms,median us,p10 us,p90 us,placement,outputs in place}.
 * </p>
 *
 * <p>
 * How to run?
 * </p>
 * <code>
 * tornado --jvm "-Dbench.warmup=50 -Dbench.iterations=500" -m tornado.examples/uk.ac.manchester.tornado.examples.coreml.CoreMLBenchmark Mlp_256x1024x4.mlpackage ...
 * </code>
 */
public class CoreMLBenchmark {

    private static final int WARMUP = Integer.getInteger("bench.warmup", 50);
    private static final int ITERATIONS = Integer.getInteger("bench.iterations", 500);
    private static final String UNITS = System.getProperty("bench.units", "CPU_ONLY,CPU_AND_GPU,CPU_AND_NEURAL_ENGINE,ALL");

    private record Timing(double firstMs, double median, double p10, double p90) {
    }

    private static Timing time(TornadoExecutionPlan plan) {
        long start = System.nanoTime();
        plan.execute();
        double firstMs = (System.nanoTime() - start) / 1e6;
        for (int i = 0; i < WARMUP; i++) {
            plan.execute();
        }
        double[] us = new double[ITERATIONS];
        for (int i = 0; i < ITERATIONS; i++) {
            start = System.nanoTime();
            plan.execute();
            us[i] = (System.nanoTime() - start) / 1e3;
        }
        Arrays.sort(us);
        return new Timing(firstMs, us[ITERATIONS / 2], us[ITERATIONS / 10], us[Math.min(ITERATIONS - 1, ITERATIONS * 9 / 10)]);
    }

    private static String placement(CoreMLModel model) {
        try {
            CoreMLComputePlan plan = model.computePlan();
            StringBuilder sb = new StringBuilder();
            plan.countByDevice().forEach((device, count) -> sb.append(sb.isEmpty() ? "" : " ").append(device).append('=').append(count));
            return sb.toString();
        } catch (RuntimeException e) {
            return "unknown";
        }
    }

    private static void run(String path, ComputeUnits units) throws TornadoExecutionPlanException {
        String name = Path.of(path).getFileName().toString().replaceAll("\\.ml(package|model|modelc)$", "");
        long loadStart = System.nanoTime();
        try (CoreMLModel model = CoreML.load(path, CoreMLOptions.computeUnits(units))) {
            double loadMs = (System.nanoTime() - loadStart) / 1e6;
            int elements = (int) model.getInputs().getFirst().elements();
            HalfFloatArray x = new HalfFloatArray(elements);
            HalfFloatArray y = new HalfFloatArray((int) model.getOutputs().getFirst().elements());
            for (int i = 0; i < elements; i++) {
                x.set(i, new HalfFloat(((i % 17) - 8) / 8.0f));
            }
            String placement = placement(model);

            for (boolean endToEnd : new boolean[] { false, true }) {
                int mode = endToEnd ? DataTransferMode.EVERY_EXECUTION : DataTransferMode.FIRST_EXECUTION;
                TaskGraph taskGraph = new TaskGraph("bench") //
                        .transferToDevice(mode, x) //
                        .libraryTask("predict", CoreML::predict, model, x, y) //
                        .transferToHost(mode, y);
                long inPlace = CoreMLLibraryProvider.inPlaceOutputs();
                long predictions = CoreMLLibraryProvider.predictions();
                Timing t;
                try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
                    t = time(plan);
                }
                long ran = CoreMLLibraryProvider.predictions() - predictions;
                long landed = CoreMLLibraryProvider.inPlaceOutputs() - inPlace;
                System.out.printf("%s,%s,%s,%.1f,%.2f,%.1f,%.1f,%.1f,%s,%d/%d%n", endToEnd ? "tornado-e2e" : "tornado-task", name, units, loadMs, t.firstMs(), t.median(), t.p10(), t.p90(),
                        placement, landed, ran);
            }
        }
    }

    public static void main(String[] args) throws TornadoExecutionPlanException {
        if (args.length == 0) {
            System.err.println("usage: CoreMLBenchmark <model> ...  [-Dbench.units=ALL,...] [-Dbench.warmup=N] [-Dbench.iterations=N]");
            System.exit(1);
        }
        System.out.println("mode,model,units,load_ms,first_ms,median_us,p10_us,p90_us,placement,in_place");
        for (String path : args) {
            for (String units : UNITS.split(",")) {
                run(path, ComputeUnits.valueOf(units.trim()));
            }
        }
    }
}
