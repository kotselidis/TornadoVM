/*
 * M0 spike: throwaway apple/mlx-spike provider used to check that mlx-c can adopt
 * real TornadoVM Metal buffers from inside a TaskGraph. Not production code.
 */
open module tornado.mlx.spike {
    requires tornado.api;
    requires tornado.runtime;
    requires tornado.drivers.metal;

    provides uk.ac.manchester.tornado.runtime.library.spi.TornadoLibraryProvider with //
            uk.ac.manchester.tornado.mlx.spike.MlxSpikeProvider;
}
