/*
 * Copyright (c) 2019, APT Group, Department of Computer Science,
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
 */
open module tornado.unittests {
    requires transitive junit;
    requires transitive tornado.api;
    requires tornado.cublas;
    requires tornado.curand;
    requires tornado.cufft;
    requires tornado.cudnn;
    requires tornado.cusparse;
    requires tornado.cudf;
    requires tornado.cutlass;
    requires tornado.cuvs;
    requires lucene.core;
    requires java.desktop;
    requires jdk.incubator.vector;

    exports uk.ac.manchester.tornado.unittests;
    exports uk.ac.manchester.tornado.unittests.api;
    exports uk.ac.manchester.tornado.unittests.arrays;
    exports uk.ac.manchester.tornado.unittests.atomics;
    exports uk.ac.manchester.tornado.unittests.batches;
    exports uk.ac.manchester.tornado.unittests.bitsets;
    exports uk.ac.manchester.tornado.unittests.branching;
    exports uk.ac.manchester.tornado.unittests.common;
    exports uk.ac.manchester.tornado.unittests.cublas;
    exports uk.ac.manchester.tornado.unittests.curand;
    exports uk.ac.manchester.tornado.unittests.nvtx;
    exports uk.ac.manchester.tornado.unittests.cufft;
    exports uk.ac.manchester.tornado.unittests.cudnn;
    exports uk.ac.manchester.tornado.unittests.cusparse;
    exports uk.ac.manchester.tornado.unittests.cudf;
    exports uk.ac.manchester.tornado.unittests.cutlass;
    exports uk.ac.manchester.tornado.unittests.cuvs;
    exports uk.ac.manchester.tornado.unittests.fields;
    exports uk.ac.manchester.tornado.unittests.flatmap;
    exports uk.ac.manchester.tornado.unittests.fuzz;
    exports uk.ac.manchester.tornado.unittests.functional;
    exports uk.ac.manchester.tornado.unittests.images;
    exports uk.ac.manchester.tornado.unittests.kernelcontext.api;
    exports uk.ac.manchester.tornado.unittests.kernelcontext.matrices;
    exports uk.ac.manchester.tornado.unittests.kernelcontext.reductions;
    exports uk.ac.manchester.tornado.unittests.instances;
    exports uk.ac.manchester.tornado.unittests.lambdas;
    exports uk.ac.manchester.tornado.unittests.logic;
    exports uk.ac.manchester.tornado.unittests.loops;
    exports uk.ac.manchester.tornado.unittests.math;
    exports uk.ac.manchester.tornado.unittests.matrices;
    exports uk.ac.manchester.tornado.unittests.prebuilt;
    exports uk.ac.manchester.tornado.unittests.profiler;
    exports uk.ac.manchester.tornado.unittests.reductions;
    exports uk.ac.manchester.tornado.unittests.slam;
    exports uk.ac.manchester.tornado.unittests.tasks;
    exports uk.ac.manchester.tornado.unittests.temporary.values;
    exports uk.ac.manchester.tornado.unittests.tile;
    exports uk.ac.manchester.tornado.unittests.tools;
    exports uk.ac.manchester.tornado.unittests.vectortypes;
    exports uk.ac.manchester.tornado.unittests.virtualization;
    exports uk.ac.manchester.tornado.unittests.memory.leak;
}
