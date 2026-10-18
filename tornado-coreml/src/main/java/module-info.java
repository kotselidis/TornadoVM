open module tornado.coreml {
    requires transitive tornado.api;
    requires tornado.runtime;

    exports uk.ac.manchester.tornado.coreml;
    exports uk.ac.manchester.tornado.coreml.provider;

    provides uk.ac.manchester.tornado.runtime.library.spi.TornadoLibraryProvider with //
            uk.ac.manchester.tornado.coreml.provider.CoreMLLibraryProvider;
}
